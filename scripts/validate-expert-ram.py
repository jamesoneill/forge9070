#!/usr/bin/env python3
"""CPU/storage integration checks for the owned expert tier; no GPU benchmark.

Run after building: python scripts/validate-expert-ram.py --exe target/release/forge9070
Only the bundled synthetic model is used. Temporary server/catalog files are removed.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/expert-ram-integration-0.6.0.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    exe = args.exe.resolve()
    fixture = root / 'tests/fixtures/tiny-qwen3moe.gguf'
    checks = []
    version = 'unknown'

    def command(name, argv, expected=0):
        p = subprocess.run([str(exe), *map(str, argv)], cwd=root, capture_output=True, text=True, timeout=60)
        record = {'name': name, 'command': [str(exe), *map(str, argv)], 'exit_code': p.returncode,
                  'stdout': p.stdout, 'stderr': p.stderr, 'passed': p.returncode == expected}
        checks.append(record)
        assert record['passed'], f'{name}: {p.stderr}'
        return p

    def check(name, condition, detail=None):
        checks.append({'name': name, 'passed': bool(condition), 'detail': detail})
        assert condition, f'{name}: {detail}'

    try:
        version = command('version', ['--version']).stdout.strip().removeprefix('forge9070 ')
        plan = json.loads(command('memory-preview', ['memory', '--model', fixture, '--json']).stdout)
        payload = plan['weight_memory']['routed_expert_bytes']
        check('preview-counts-actual-experts', payload == 98_304 and plan['weight_memory']['lazy_embedding_bytes'] == 0, plan)
        check('preview-ceiling-within-payload', 0 <= plan['plan']['budget_bytes'] <= payload)
        disabled = json.loads(command('memory-disabled', ['memory', '--model', fixture, '--expert-ram', '0', '--json']).stdout)
        check('disabled-preview-zero', disabled['plan']['budget_bytes'] == 0)
        command('invalid-ceiling-rejected', ['memory', '--model', fixture, '--expert-ram', '64GB'], expected=1)
        help_text = command('tune-memory-controls', ['tune', '--help']).stdout
        check('tune-exposes-policy', all(flag in help_text for flag in ['--expert-ram', '--expert-warmup', '--ram-margin']))

        common = ['generate', '--cpu', '--model', fixture, '--ctx', '256', '--chunk', '32', '--threads', '2',
                  '--prompt', 'hello world', '--max-tokens', '8', '--temperature', '0', '--seed', '7',
                  '--ram-margin', '0', '--cache-ram', '0']
        baseline = command('cpu-mapped-generation', [*common, '--expert-ram', '0'])
        warm = command('cpu-owned-warm-generation', [*common, '--expert-ram', '1', '--expert-warmup', 'true'])
        demand = command('cpu-owned-demand-generation', [*common, '--expert-ram', '1', '--expert-warmup', 'false'])
        check('generated-output-identical', baseline.stdout == warm.stdout == demand.stdout,
              {'mapped': baseline.stdout, 'warm': warm.stdout, 'demand': demand.stdout})
        check('warmup-actually-populates', 'expert RAM warmup:' in warm.stderr and '4 experts' in warm.stderr, warm.stderr)

        with tempfile.TemporaryDirectory(prefix='forge-expert-ram-test-') as tmp_name:
            tmp = Path(tmp_name)
            models = tmp / 'models'
            models.mkdir()
            for name in ['fixture-a.gguf', 'fixture-b.gguf']:
                shutil.copyfile(fixture, models / name)
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            env = os.environ.copy()
            env.pop('STRATA_API_KEY', None)
            env['XDG_CACHE_HOME'] = str(tmp / 'cache')
            argv = [str(exe), 'serve', '--cpu', '--models-dir', str(models), '--port', str(port),
                    '--ctx', '256', '--chunk', '32', '--threads', '2', '--max-tokens', '16',
                    '--expert-ram', '1', '--expert-warmup', 'true', '--ram-margin', '0', '--cache-ram', '0']
            # Only this test's private loopback server; no external HTTP calls.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

            def request(path, body=None):
                data = None if body is None else json.dumps(body).encode()
                req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data,
                                             headers={} if data is None else {'Content-Type': 'application/json'})
                with opener.open(req, timeout=15) as response:
                    return json.load(response)

            with (tmp / 'server.log').open('w+') as log:
                server = subprocess.Popen(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    catalog = None
                    deadline = time.monotonic() + 15
                    while time.monotonic() < deadline:
                        if server.poll() is not None:
                            raise RuntimeError('test server exited before becoming ready')
                        try:
                            catalog = request('/api/models')
                            break
                        except (OSError, urllib.error.URLError):
                            time.sleep(0.05)
                    check('server-catalog-ready', catalog is not None)
                    ids = sorted(model['id'] for model in catalog['models'])
                    check('two-independent-catalog-models', len(ids) == 2, ids)
                    loaded = request('/api/load', {'model': ids[0]})
                    stats = request('/api/stats')
                    first = stats['expert_ram']
                    check('server-first-load-warmed', first['resident_bytes'] == payload and first['warm_admissions'] == 4,
                          {'load': loaded, 'expert_ram': first})
                    response = request('/v1/chat/completions', {'model': ids[0], 'messages': [{'role': 'user', 'content': 'hello'}],
                                                               'max_tokens': 4, 'temperature': 0})
                    check('server-generates-with-owned-cache', bool(response.get('choices')), response)
                    active = request('/api/stats')['expert_ram']
                    check('server-cache-hit-counters-advance', active['hits'] > 0 and active['resident_bytes'] <= active['budget_bytes'], active)
                    loaded = request('/api/load', {'model': ids[1]})
                    second = request('/api/stats')['expert_ram']
                    check('model-swap-gets-fresh-owned-cache', second['warm_admissions'] == 4 and second['hits'] == 0
                          and second['resident_bytes'] == payload, {'load': loaded, 'expert_ram': second})
                    request('/api/unload', {})
                    unloaded = request('/api/stats')
                    check('unload-releases-model-cache-status', unloaded['model'] is None and unloaded['expert_ram'] is None, unloaded)
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=5)
                    log.seek(0)
                    checks.append({'name': 'server-log', 'command': argv, 'output': log.read(), 'passed': True})
    finally:
        output = {'version': version, 'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'scope': 'Synthetic CPU/storage/loopback-server integration only; not Qwen3.8 inference or a GPU benchmark.',
                  'passed': sum(c['passed'] for c in checks), 'checks': checks}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + '\n')
        print(json.dumps({'passed': output['passed'], 'recorded_checks': len(checks), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
