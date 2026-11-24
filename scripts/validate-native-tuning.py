#!/usr/bin/env python3
"""Exercise the tuning worker lifecycle on local untrained CPU models."""
import argparse
import http.client
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
KEY = 'nextora-native-tuning-regression-only'


def fixture(destination):
    spec = importlib.util.spec_from_file_location('tuning_api_helpers', ROOT / 'scripts/validate-native-api.py')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    helpers.fixtures.fixture_copy(ROOT / 'tests/fixtures/tiny-qwen3moe.gguf', destination, False)
    helpers.add_template(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    with tempfile.TemporaryDirectory(prefix='nextora-tuning-api-') as temporary:
        root = Path(temporary)
        weights = root / 'model.gguf'
        fixture(weights)
        settings = dict(backend='cpu', device=None, ctx=512, max_tokens=32, parallel=1,
                        max_queued=2, queue_timeout_secs=5, vram_mib=0, expert_ram_mib=16,
                        ram_margin_mib=0, expert_warmup=True, threads=2)
        profile = dict(directory=str(root), model=weights.name, settings=settings)
        saved = root / 'service.json'
        saved.write_text(json.dumps(profile))
        with socket.socket() as reserve:
            reserve.bind(('127.0.0.1', 0))
            port = reserve.getsockname()[1]
        command = [str(binary), 'serve', '--settings-file', str(saved), '--port', str(port), '--api-key', KEY]
        log = tempfile.TemporaryFile()
        process = None

        def request(path, body=None, auth=True, expected=200):
            connection = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
            headers = {'Authorization': 'Bearer ' + KEY} if auth else {}
            if body is not None:
                headers['Content-Type'] = 'application/json'
            try:
                connection.request('GET' if body is None else 'POST', path,
                                   None if body is None else json.dumps(body), headers)
                response = connection.getresponse()
                data = json.load(response)
                assert response.status in (expected if isinstance(expected, tuple) else (expected,)), (path, response.status, data)
                return data
            finally:
                connection.close()

        def wait(predicate, seconds=60):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                result = predicate()
                if result:
                    return result
                time.sleep(.02)
            raise AssertionError('Tuning lifecycle transition timed out')

        def start():
            nonlocal process
            process = subprocess.Popen(command, stdout=log, stderr=log)
            def ready():
                assert process.poll() is None, 'Service exited during startup'
                try:
                    return request('/health')
                except OSError:
                    return None
            wait(ready)

        def stop():
            process.terminate()
            process.wait(timeout=10)

        def finish():
            def terminal():
                job = request('/api/tuning/status')
                return job if job['phase'] in ('complete', 'failed', 'cancelled') else None
            return wait(terminal)

        try:
            start()
            request('/api/tuning/status', auth=False, expected=401)
            idle = request('/api/tuning/status')
            assert idle['phase'] == 'idle', idle
            started = request('/api/tuning/start', {}, expected=(200, 202))
            job = finish()
            assert job['phase'] == 'complete', job
            assert job['worker_pid'] and job['worker_pid'] != process.pid, job
            result = job['result']
            valid = [c for c in result['candidates'] if not c.get('error') and c.get('decode_tps', 0) > 0]
            assert valid and result['winner']['decode_tps'] == max(c['decode_tps'] for c in valid), result
            effective = request('/api/config')
            assert effective['settings']['ctx'] == 512 and effective['model'] == weights.name
            assert effective['settings']['threads'] == result['winner']['threads'], effective
            assert json.loads(saved.read_text())['settings'] == effective['settings']
            cached = request('/api/tuning/start', {}, expected=(200, 202))
            completed = finish()
            assert completed['cached'] and completed['result'] == result, completed
            stop()
            start()
            reused = request('/api/tuning/status')
            assert reused['cached'] and reused['result'] == result, reused
            # A changed context uses a new cache entry and remains idle until requested.
            config = request('/api/config')
            changed = dict(directory=config['models_dir'], model=config['model'], settings=dict(config['settings'], ctx=384))
            request('/api/config', changed)
            assert request('/api/tuning/status')['phase'] == 'idle'
            request('/api/tuning/start', {}, expected=(200, 202))
            request('/api/tuning/cancel', {}, expected=(200, 202))
            cancelled = finish()
            assert cancelled['phase'] == 'cancelled', cancelled
            assert request('/api/config')['settings'] == changed['settings']
            assert request('/health')['status'] == 'ok'
            assert not request('/v1/status')['admission']['model_changing']
            print('Tuning checks passed: separate worker, measured winner, saved settings, cached reuse, restart, context identity and cancellation with restoration.')
        except BaseException:
            log.seek(0)
            sys.stderr.write(log.read().decode(errors='replace'))
            raise
        finally:
            if process and process.poll() is None:
                stop()
            log.close()


if __name__ == '__main__':
    main()
