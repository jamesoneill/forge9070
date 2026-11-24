#!/usr/bin/env python3
"""Check Nextora's Hub HTTP API using local GGUF fixtures and a curl stand-in.

Runs on Linux/WSL and other POSIX hosts that can run Nextora. No Hub account,
network download or GPU is required. Windows needs a native curl.exe stand-in
and is excluded from this particular test.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests' / 'fixtures'
REVISION = 'a' * 40
NAMES = ['good-Q4_K_M.gguf', 'dense-Q4_K_M.gguf', 'corrupt-Q4_K_M.gguf', 'cancel-Q4_K_M.gguf']


def fixture(name):
    filename = 'tiny-qwen3.gguf' if name.startswith('dense-') else 'tiny-qwen3moe.gguf'
    return (FIXTURES / filename).read_bytes()


def size(name, payload):
    # A valid header with a huge advertised file checks the space preflight,
    # without ever allocating or transferring that amount.
    return 1 << 55 if name.startswith('oversize-') else len(payload)


def fake_curl(args):
    url = urlsplit(args[args.index('--url') + 1])
    assert (url.scheme, url.netloc) == ('https', 'huggingface.co')
    control = Path(os.environ['NEXTORA_FIXTURE_CONTROL'])
    if url.path == '/api/models':
        query = parse_qs(url.query)['search'][0]
        repo = 'fixtures/oversize' if query == 'oversize' else 'fixtures/test'
        print(json.dumps([{'id': repo, 'sha': REVISION, 'gguf': {'architecture': 'qwen3moe'}}]))
        return
    if url.path.startswith('/api/models/'):
        assert url.path.endswith('/revision/' + REVISION)
        names = ['oversize-Q4_K_M.gguf'] if '/oversize/' in url.path else NAMES
        siblings = []
        for name in names:
            payload = fixture(name)
            siblings.append({'rfilename': name, 'size': size(name, payload),
                             'lfs': {'sha256': hashlib.sha256(payload).hexdigest()}})
        print(json.dumps({'sha': REVISION, 'gguf': {'architecture': 'qwen3moe'}, 'siblings': siblings}))
        return

    assert '/resolve/' + REVISION + '/' in url.path
    name = url.path.rsplit('/', 1)[1]
    payload = fixture(name)
    if '--range' in args:
        last = int(args[args.index('--range') + 1].removeprefix('0-'))
        sys.stdout.buffer.write(payload[:last + 1].ljust(last + 1, b'\0'))
        sys.stderr.write(f'206 bytes 0-{last}/{size(name, payload)}')
        return

    with (control / 'transfers').open('a') as transfers:
        print(name, file=transfers)
    if name.startswith('cancel-'):
        # Synchronize with the HTTP cancellation check instead of relying on
        # the transfer staying in flight for a particular number of seconds.
        (control / 'started').touch()
        wait_for(lambda: (control / 'release').exists(), 'test cancellation signal')
    if name.startswith('corrupt-'):
        payload = payload[:-1] + bytes([payload[-1] ^ 1])
    sys.stdout.buffer.write(payload)


def wait_for(condition, description):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = condition()
        if result:
            return result
        time.sleep(.02)
    raise AssertionError(f'Timed out waiting for {description}')


def validate(binary, root):
    models = root / 'Models'
    shim = root / 'curl'
    shim.write_text('#!/bin/sh\nexec ' + shlex.join([sys.executable, str(Path(__file__).resolve()),
                                                   '--mock-curl']) + ' "$@"\n')
    shim.chmod(0o755)
    env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
               NEXTORA_MODELS_DIR=str(models), NEXTORA_FIXTURE_CONTROL=str(root))
    with socket.socket() as reserved:
        reserved.bind(('127.0.0.1', 0))
        port = reserved.getsockname()[1]
    command = [str(binary), 'serve', '--host', '127.0.0.1', '--port', str(port), '--backend', 'cpu']

    def request(path, body=None, expected=200):
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=20)
        try:
            conn.request('GET' if body is None else 'POST', path, None if body is None else json.dumps(body),
                         {'Content-Type': 'application/json'})
            response = conn.getresponse()
            data = json.load(response)
            assert response.status == expected, (path, response.status, data)
            return data
        finally:
            conn.close()

    def ready():
        try:
            return request('/api/config')
        except OSError:
            return None

    def finish():
        def poll():
            job = request('/api/hub/progress')
            return job if job['phase'] in {'failed', 'cancelled', 'ready'} else None
        return wait_for(poll, 'terminal Hub job')

    def start(model):
        selection = {key: model[key] for key in ('repo', 'revision', 'file')}
        request('/api/hub/download', selection, expected=202)

    with tempfile.TemporaryFile() as log:
        service = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        try:
            wait_for(ready, 'local configuration service')
            result = request('/api/hub/search', {'query': 'fixture'})
            choices = {item['file']: item for item in result['models']}
            assert set(choices) == set(NAMES), result
            assert not (root / 'transfers').exists() and not models.exists()

            start(choices['corrupt-Q4_K_M.gguf'])
            failed = finish()
            assert failed['phase'] == 'failed' and 'SHA-256' in failed['error'], failed
            assert not list(models.iterdir())

            start(choices['cancel-Q4_K_M.gguf'])
            wait_for(lambda: (root / 'started').is_file(), 'fixture transfer')
            progress = request('/api/hub/progress')
            assert progress['phase'] == 'downloading' and progress['total_files'] == 1, progress
            request('/api/hub/cancel', {})
            (root / 'release').touch()
            assert finish()['phase'] == 'cancelled'
            assert not list(models.iterdir())

            good = choices['good-Q4_K_M.gguf']
            start(good)
            installed = finish()
            assert installed['phase'] == 'ready' and installed['completed_files'] == 1, installed
            assert installed['received_bytes'] == installed['total_bytes'] == good['bytes']
            path = Path(installed['directory']) / installed['model']
            assert path.read_bytes() == fixture(good['file'])
            assert request('/v1/status')['model'] is None
            ctx = min(128, good['context_length'])
            settings = dict(request('/api/config')['settings'], backend='cpu', ctx=ctx, max_tokens=1)
            loaded = request('/api/config', {'directory': installed['directory'], 'model': installed['model'],
                                             'settings': settings})
            assert loaded['model'] == installed['model']
            start(good)
            assert 'already exists' in finish()['error']
            assert path.read_bytes() == fixture(good['file'])

            oversized = request('/api/hub/search', {'query': 'oversize'})['models']
            assert len(oversized) == 1
            start(oversized[0])
            refused = finish()
            assert refused['phase'] == 'failed' and 'Insufficient space' in refused['error'], refused
            assert 'oversize' not in (root / 'transfers').read_text()
            assert not list(models.glob('.nextora-partial-*'))
        except BaseException:
            log.seek(0)
            sys.stderr.write(log.read().decode(errors='replace'))
            raise
        finally:
            service.terminate()
            service.wait(timeout=10)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--mock-curl']:
        fake_curl(sys.argv[2:])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--binary', required=True, type=Path)
        args = parser.parse_args()
        if os.name != 'posix':
            parser.exit(0, 'Skipped: the isolated transport fixture requires a POSIX shell.\n')
        with tempfile.TemporaryDirectory(prefix='nextora-hub-test-') as tmp:
            validate(args.binary.resolve(), Path(tmp))
        print('Hub checks passed: verified search, explicit download and load, hash failure, '
              'cancellation, space preflight and snapshot protection.')
