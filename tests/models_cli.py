#!/usr/bin/env python3
"""Run the native Linux launcher and exercise the model-management CLI against it."""
from __future__ import annotations
import argparse
import http.client
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'target/debug/forge9070')
    args = parser.parse_args()
    binary = str(args.binary.resolve())
    key = 'forge9070-model-management-local-test'
    env = dict(os.environ, STRATA_API_KEY=key)
    checks = []
    with tempfile.TemporaryDirectory(prefix='forge9070 models ') as tmp:
        directory = Path(tmp)
        identifiers = ['repo-a/rev-a/weights.gguf', 'community/uncensored variant/rev-b/weights.gguf']
        for identifier in identifiers:
            destination = directory / identifier
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / 'tests/fixtures/tiny-qwen3moe.gguf', destination)
        shutil.copyfile(ROOT / 'tests/fixtures/tiny-qwen3.gguf', directory / 'dense.gguf')
        with socket.socket() as reserve:
            reserve.bind(('127.0.0.1', 0))
            port = reserve.getsockname()[1]
        url = f'http://127.0.0.1:{port}'
        def cli(*arguments: str):
            result = subprocess.run([binary, 'models', *arguments, '--json'], env=env,
                                    cwd=ROOT, capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stderr
            return json.loads(result.stdout)
        def api(path: str, data: dict | None = None):
            connection = http.client.HTTPConnection('127.0.0.1', port, timeout=15)
            try:
                headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
                connection.request('POST' if data is not None else 'GET', path,
                                   json.dumps(data) if data is not None else None, headers)
                response = connection.getresponse()
                return response.status, json.loads(response.read())
            finally:
                connection.close()
        local = cli('list', '--models-dir', str(directory))
        assert {model['id'] for model in local['models']} == set(identifiers)
        checks.append('CLI local catalog finds nested repositories and preserves spaces')
        assert local['policy'] == 'moe_only' and local['excluded_count'] == 1, local
        checks.append('Dense models are excluded from the compatible MoE catalog')
        with tempfile.TemporaryFile(mode='w+b') as log:
            process = subprocess.Popen(['bash', 'scripts/serve-wsl.sh', '--backend', 'cpu',
                                        '--bin', binary, '--models-dir', str(directory), '--port', str(port),
                                        '--', '--ctx', '128', '--max-tokens', '16', '--ram-margin', '0'],
                                       cwd=ROOT, env=env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 30
                while True:
                    assert process.poll() is None, 'native launcher/server exited'
                    try:
                        if api('/api/models')[0] == 200:
                            break
                    except (OSError, http.client.HTTPException):
                        pass
                    assert time.monotonic() < deadline, 'server startup timed out'
                    time.sleep(0.05)
                status = cli('status', '--server', url)
                assert not any(model['loaded'] for model in status['models'])
                checks.append('Actual native Linux launcher starts an unloaded CPU server')
                for identifier in identifiers:
                    loaded = cli('load', identifier, '--server', url)
                    assert loaded['loaded'] == identifier, loaded
                    status = cli('status', '--server', url)
                    assert [model['id'] for model in status['models'] if model['loaded']] == [identifier]
                    code, chat = api('/v1/chat/completions', {'model': identifier,
                        'messages': [{'role': 'user', 'content': 'hello'}], 'max_tokens': 4,
                        'chat_template_kwargs': {'enable_thinking': False}})
                    assert code == 200, chat
                    assert chat['choices'][0]['message']['content'] == 'AAAA', chat
                checks.append('CLI switches between catalog IDs and actual inference continues')
                rejected = subprocess.run([binary, 'models', 'load', 'dense.gguf', '--server', url],
                    env=env, cwd=ROOT, capture_output=True, text=True, timeout=30)
                assert rejected.returncode != 0 and '400' in rejected.stderr, rejected.stderr
                assert api('/v1/models')[1]['data'][0]['id'] == identifiers[-1]
                checks.append('Direct CLI load cannot bypass MoE policy and preserves the active model')
                code, invalid = api('/api/load', {'model':identifiers[-1], 'options':{'parallel':2}})
                assert code == 400, invalid
                assert api('/v1/models')[1]['data'][0]['id'] == identifiers[-1]
                checks.append('Unsupported CPU parallel settings are rejected before unloading')
                code, stale = api('/v1/chat/completions', {'model': identifiers[0],
                    'messages': [{'role': 'user', 'content': 'hello'}], 'max_tokens': 4})
                assert code == 409, stale
                checks.append('Explicit stale model ID is rejected rather than answered by a different model')
                assert cli('unload', '--server', url)['unloaded'] is True
                assert api('/v1/models')[1]['data'] == []
                checks.append('CLI unload releases the model while leaving management available')
                cli('load', identifiers[0], '--server', url)
                assert api('/v1/models')[1]['data'][0]['id'] == identifiers[0]
                checks.append('CLI reload succeeds after unload')
            except BaseException:
                log.seek(0)
                print(log.read().decode(errors='replace'))
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    print(json.dumps({'status': 'passed', 'checks': checks, 'actual_linux_launcher': True,
                      'wsl_kernel_tested': False, 'windows_interop_tested': False,
                      'gpu_compute_tested': False}, indent=2))

if __name__ == '__main__':
    main()
