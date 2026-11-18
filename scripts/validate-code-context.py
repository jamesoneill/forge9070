#!/usr/bin/env python3
"""Exercise code presets, local excerpts and complete-request budgets.

Uses a temporary Git worktree, bundled synthetic tokenizer/model and private
CPU loopback server. Does not download models, execute repository code or
measure GPU throughput. Run after building: --exe target/release/forge9070.
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
    parser.add_argument('--output', type=Path, default=Path('artifacts/code-context-integration-0.6.0.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    exe = args.exe.resolve()
    fixture = root / 'tests/fixtures/tiny-qwen3moe.gguf'
    records = []

    def check(name, condition, detail=None):
        records.append({'name': name, 'passed': bool(condition), 'detail': detail})
        assert condition, f'{name}: {detail}'

    def run(name, argv, success=True):
        proc = subprocess.run([str(exe), *map(str, argv)], cwd=root, capture_output=True, text=True, timeout=60)
        records.append({'name': name, 'command': [str(exe), *map(str, argv)], 'exit_code': proc.returncode,
                        'stdout': proc.stdout, 'stderr': proc.stderr,
                        'passed': (proc.returncode == 0) == success})
        assert records[-1]['passed'], f'{name}: {proc.stdout}\n{proc.stderr}'
        return proc

    version = 'unknown'
    try:
        version = run('version', ['--version']).stdout.strip()
        for name, ctx, cap in [('code', 16384, 4096), ('code-long', 32768, 8192)]:
            result = json.loads(run('profile-' + name, ['profile', '--preset', name]).stdout)
            check('profile-settings-' + name,
                  result['context_tokens'] == ctx and result['maximum_output_tokens'] == cap
                  and result['prompt_cache_ram_mib'] == 1024 and result['ram_margin_mib'] == 6144
                  and result['kv_format'] == 'f16' and result['parallel_requests'] == 1,
                  result)

        with tempfile.TemporaryDirectory(prefix='forge-code-context-') as tmp_name:
            tmp = Path(tmp_name)
            repo = tmp / 'repo'
            repo.mkdir()
            subprocess.run(['git', '-c', 'init.defaultBranch=main', 'init', '-q', str(repo)], check=True)
            for rel, text in {
                'src/Orders.cs': 'public sealed class Orders {\n    public void ReturnRefundForCancellation() { RefundOnce(); }\n    private void RefundOnce() {}\n}\n',
                'tests/OrdersTests.cs': 'public class OrdersTests { public void RefundRetryIsIdempotent() {} }\n',
                'src/Unrelated.cs': 'public class UnrelatedWidget { }\n',
                '.env': 'EXCLUDED_FIXTURE_VALUE=must_not_be_in_pack\n',
                'vendor/dependency.cs': 'EXCLUDED_VENDOR_VALUE\n',
            }.items():
                path = repo / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            subprocess.run(['git', '-C', str(repo), 'add', '--all'], check=True)
            (repo / 'src/Untracked.cs').write_text('public class UntrackedRefundEndpoint { }\n')
            base = ['context', '--repo', repo, '--query', 'refund cancellation retries', '--include', 'src/Orders.cs', '--max-bytes', '4096']
            packed = json.loads(run('repository-pack', [*base, '--json']).stdout)
            pack = packed['pack']
            text = pack['context']
            check('pack-relevant-line-labelled-code', 'ReturnRefundForCancellation' in text and 'src/Orders.cs' in text and bool(pack['selected']), pack)
            check('pack-excludes-known-secrets-vendor-untracked', all(s not in text for s in ['EXCLUDED_FIXTURE_VALUE', 'EXCLUDED_VENDOR_VALUE', 'UntrackedRefundEndpoint']), text)
            check('pack-byte-budget-and-unknown-token-count', len(text.encode()) == pack['bytes'] <= 4096 and pack['tokens'] is None, pack)
            again = json.loads(run('repository-pack-repeat', [*base, '--json']).stdout)
            check('pack-is-deterministic', again == packed)
            opt = json.loads(run('repository-untracked-opt-in', [*base, '--include-untracked', '--include', 'src/Untracked.cs', '--json']).stdout)
            check('untracked-included-only-on-request', 'UntrackedRefundEndpoint' in opt['pack']['context'])
            run('token-cap-needs-tokenizer', [*base, '--budget-tokens', '100'], success=False)

            # Larger metadata context for tokenizer-only packing. This remains
            # the same untrained fixture, not evidence of model long-context quality.
            token_model = tmp / 'tokenizer-fixture.gguf'
            data = bytearray(fixture.read_bytes())
            key = b'qwen3moe.context_length'
            at = data.index(key) + len(key)
            assert int.from_bytes(data[at:at + 4], 'little') == 4
            assert int.from_bytes(data[at + 4:at + 8], 'little') == 512
            data[at + 4:at + 8] = (32768).to_bytes(4, 'little')
            token_model.write_bytes(data)
            out = tmp / 'context.txt'
            precise = json.loads(run('exact-token-pack', [*base, '--model', token_model, '--preset', 'code-long', '--budget-tokens', '2048', '--output', out, '--json']).stdout)
            check('pack-exact-count-with-reserves', 0 < precise['pack']['tokens'] <= 2048 and precise['reserved_output_tokens'] == 8192
                  and precise['reserved_other_tokens'] == 2048 and out.read_text() == precise['pack']['context'], precise)
            run('pack-cannot-consume-answer-reserve', [*base, '--model', token_model, '--preset', 'code-long', '--budget-tokens', '30000'], success=False)
            run('model-context-limit-before-compute', ['generate', '--cpu', '--model', fixture, '--ctx', '1024', '--max-tokens', '4', '--prompt', 'hello'], success=False)

            models = tmp / 'models'
            models.mkdir()
            shutil.copyfile(fixture, models / 'fixture.gguf')
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            key = 'local-integration-only-key'
            env = os.environ.copy()
            env['STRATA_API_KEY'] = key
            env['XDG_CACHE_HOME'] = str(tmp / 'cache')
            argv = [str(exe), 'serve', '--cpu', '--preset', 'code-long', '--models-dir', str(models), '--port', str(port),
                    '--ctx', '512', '--chunk', '32', '--threads', '2', '--max-tokens', '64', '--temperature', '0',
                    '--expert-ram', '1', '--ram-margin', '0', '--cache-ram', '0']

            def request(path, body=None, auth=True):
                headers = {'Authorization': 'Bearer ' + key} if auth else {}
                data = None if body is None else json.dumps(body).encode()
                if data is not None:
                    headers['Content-Type'] = 'application/json'
                req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data, headers=headers)
                try:
                    with opener.open(req, timeout=20) as response:
                        raw = response.read().decode()
                        return response.status, response.headers, json.loads(raw)
                except urllib.error.HTTPError as error:
                    raw = error.read().decode()
                    try:
                        result = json.loads(raw)
                    except ValueError:
                        result = {'raw': raw}
                    return error.code, error.headers, result

            with (tmp / 'server.log').open('w+') as log:
                server = subprocess.Popen(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 20
                    catalog = None
                    while time.monotonic() < deadline:
                        if server.poll() is not None:
                            raise RuntimeError('test server exited before startup')
                        try:
                            status, _, catalog = request('/api/models')
                            if status == 200:
                                break
                        except (OSError, urllib.error.URLError):
                            pass
                        time.sleep(.05)
                    check('private-cpu-server-ready', catalog is not None, catalog)
                    model_id = catalog['models'][0]['id']
                    status, _, loaded = request('/api/load', {'model': model_id})
                    check('fixture-loaded', status == 200, loaded)
                    _, _, catalog = request('/api/models')
                    options = catalog['options']
                    check('api-reports-actual-overrides-and-limits', options['ctx'] == 512 and options['max_tokens'] == 64
                          and options['context_max'] == 131072 and options['temperature'] == 0, options)
                    before = request('/api/stats')[2]
                    body = {'messages': [{'role': 'user', 'content': 'hello'}], 'max_tokens': 4,
                            'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}
                    status, _, preview = request('/api/context', body)
                    check('complete-request-preview', status == 200 and preview['fits'] and preview['output_tokens'] == 4
                          and preview['server_max_tokens'] == 64 and preview['model_context_tokens'] == 512, preview)
                    after = request('/api/stats')[2]
                    check('preview-does-not-run-inference', before.get('requests') == after.get('requests')
                          and before.get('expert_ram') == after.get('expert_ram'), {'before': before, 'after': after})
                    status, _, _ = request('/api/context', body, auth=False)
                    check('preview-requires-server-auth', status in (401, 403))
                    status, _, answer = request('/v1/chat/completions', {**body, 'model': preview['model']})
                    check('preview-token-count-matches-generation', status == 200 and answer['usage']['prompt_tokens'] == preview['prompt_tokens']
                          and answer['usage']['completion_tokens'] == 4, answer)

                    too_much = {**body, 'max_tokens': 65}
                    status, _, budget = request('/api/context', too_much)
                    check('preview-explains-output-cap', status == 200 and not budget['fits'] and budget['error']['code'] == 'output_limit_exceeded', budget)
                    for stream in (False, True):
                        status, headers, result = request('/v1/chat/completions', {**too_much, 'stream': stream})
                        check('oversize-output-before-stream-' + str(stream), status == 400 and 'error' in result
                              and 'text/event-stream' not in headers.get('Content-Type', ''), result)

                    long_body = {**body, 'messages': [{'role': 'user', 'content': 'x' * 470}], 'max_tokens': 64}
                    status, _, budget = request('/api/context', long_body)
                    check('preview-catches-prompt-plus-answer-overflow', status == 200 and not budget['fits'], budget)
                    status, _, result = request('/v1/chat/completions', long_body)
                    check('overflow-rejected-without-silent-truncation', status == 400 and 'error' in result, result)

                    status, _, result = request('/api/load', {'model': model_id, 'options': {'ctx': 1024}})
                    check('api-reload-enforces-model-context', status == 400, result)
                    status, _, current = request('/api/models')
                    check('failed-reload-keeps-active-model-and-context', status == 200 and current['options']['ctx'] == 512
                          and any(m['loaded'] and m['id'] == model_id for m in current['models']), current)
                    status, _, answer = request('/v1/chat/completions', body)
                    check('valid-generation-after-budget-rejections', status == 200 and answer['choices'][0]['message']['content'] == 'AAAA', answer)
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=5)
                    log.seek(0)
                    records.append({'name': 'server-log', 'passed': server.returncode is not None, 'command': argv, 'output': log.read()})
    finally:
        result = {'executable_version': version, 'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'scope': 'Synthetic CPU/local Git/loopback API checks; no Qwen3.8 full checkpoint, Windows execution or GPU speed test.',
                  'passed': sum(bool(r['passed']) for r in records), 'records': records}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'passed': result['passed'], 'records': len(records), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
