#!/usr/bin/env python3
"""Check unloaded-model setup, saved profiles and exact restoration on tiny CPU models."""
import argparse
import http.client
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
KEY = 'nextora-native-setup-validation-only'


def fixture(path):
    spec = importlib.util.spec_from_file_location('setup_fixture_helpers', ROOT / 'scripts/validate-native-api.py')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    helpers.fixtures.fixture_copy(ROOT / 'tests/fixtures/tiny-qwen3moe.gguf', path, False)
    helpers.add_template(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    with tempfile.TemporaryDirectory(prefix='nextora-model-setup-') as temporary:
        root = Path(temporary)
        old = root / 'old.gguf'
        new = root / 'new.gguf'
        fixture(old)
        fixture(new)
        settings = dict(backend='cpu', device=None, ctx=512, max_tokens=32, parallel=1,
                        max_queued=2, queue_timeout_secs=5, vram_mib=0, expert_ram_mib=16,
                        ram_margin_mib=0, expert_warmup=True, threads=2)
        saved = root / 'service.json'
        saved.write_text(json.dumps(dict(directory=str(root), model=old.name, settings=settings)))
        with socket.socket() as reserve:
            reserve.bind(('127.0.0.1', 0))
            port = reserve.getsockname()[1]
        log = tempfile.TemporaryFile()
        process = None

        def request(path, body=None, expected=200):
            connection = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
            headers = {'Authorization': 'Bearer ' + KEY}
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

        def wait(predicate, seconds=90):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                result = predicate()
                if result:
                    return result
                time.sleep(.02)
            raise AssertionError('Model setup transition timed out')

        def start():
            nonlocal process
            process = subprocess.Popen([str(binary), 'serve', '--settings-file', str(saved), '--port', str(port),
                                        '--api-key', KEY], stdout=log, stderr=log)
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

        def terminal():
            job = request('/api/tuning/status')
            return job if job['phase'] in ('complete', 'failed', 'cancelled') else None

        target = dict(directory=str(root), model=new.name)
        setup = dict(target=target, mode='prepare', goal=dict(context='maximum_fit', max_context=384), activate=False)
        try:
            start()
            original_bytes = saved.read_bytes()
            original = request('/api/config')
            assert request('/api/tuning/profile', target)['phase'] == 'idle'
            request('/api/tuning/start', setup, expected=(200, 202))
            result = wait(terminal)
            assert result['phase'] == 'complete', result
            assert result['worker_pid'] and result['worker_pid'] != process.pid
            assert result['preparing'] and not result['activated']
            report = result['result']
            assert report['mode'] == 'prepare' and report['memory_fit_context'] == 384
            assert report['fit_trials'] and all(t['settings']['expert_warmup'] is False for t in report['fit_trials'])
            assert report['fit_evidence']['actual_probe_tokens'] <= 4
            assert report['winner']['report']['prompt_tokens'] < report['memory_fit_context']
            assert saved.read_bytes() == original_bytes, 'Preparing changed the active profile file'
            assert request('/api/config')['settings'] == original['settings']
            status = request('/v1/status')
            assert status['model'] == old.name and Path(status['runtime']['shards'][0]).samefile(old), status
            cached = request('/api/tuning/profile', target)
            assert cached['cached'] and cached['result'] == report
            repeated = request('/api/tuning/start', setup, expected=(200, 202))
            assert repeated['phase'] == 'complete' and repeated['cached']
            assert request('/v1/status')['model'] == old.name
            # A forced retry can be cancelled without loading the new file under the old ID.
            request('/api/tuning/start', dict(setup, force=True), expected=(200, 202))
            request('/api/tuning/cancel', {})
            cancelled = wait(terminal)
            assert cancelled['phase'] == 'cancelled', cancelled
            status = request('/v1/status')
            assert status['model'] == old.name and Path(status['runtime']['shards'][0]).samefile(old)
            assert saved.read_bytes() == original_bytes
            # Profile lookup is target-based even while a different model is serving.
            assert request('/api/tuning/profile', target)['cached']
            request('/api/tuning/use', target, expected=(200, 202))
            used = wait(terminal)
            assert used['phase'] == 'complete' and used['activated'], used
            status = request('/v1/status')
            assert status['model'] == new.name
            assert status['load_progress']['stage'] == 'ready'
            assert status['admission']['queued_prompt_token_limit'] == 384 * settings['parallel'] * 4
            assert request('/api/config')['settings']['ctx'] == 384
            # Corrupt and traversal targets fail before pausing the loaded service.
            request('/api/tuning/profile', dict(directory=str(root), model='../outside.gguf'), expected=400)
            assert request('/v1/status')['model'] == new.name
            stop()
            start()
            assert request('/api/tuning/profile', target)['cached'], 'Profile was not reusable after restart'
            # Setup also works with no serving model and restores the unloaded state.
            empty = dict(directory=str(root), model=None, settings=settings)
            request('/api/config', empty)
            assert request('/v1/status')['model'] is None
            request('/api/tuning/start', dict(setup, force=True), expected=(200, 202))
            prepared_empty = wait(terminal)
            assert prepared_empty['phase'] == 'complete', prepared_empty
            assert request('/v1/status')['model'] is None
            assert not request('/v1/status')['admission']['model_changing']
            assert json.loads(saved.read_text())['model'] is None
            print('Setup passed: unloaded target, memory-fit probes, separate benchmark worker, saved profile, exact old-path restoration, cancellation, explicit activation, restart reuse and unloaded restoration.')
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
