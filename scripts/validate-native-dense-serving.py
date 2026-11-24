#!/usr/bin/env python3
"""Serve Qwen3 dense and Qwen3.5 (+MTP) through real setup, loading and chat APIs."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    helpers = module('dense_serving_helpers', ROOT / 'scripts/validate-native-ui.py')
    api = helpers.api_helpers
    families = [('qwen3', 'tiny-qwen3.gguf'), ('qwen35', 'tiny-qwen35.gguf'),
                ('qwen35', 'tiny-qwen35-mtp.gguf')]
    with tempfile.TemporaryDirectory(prefix='nextora-dense-serving-') as temporary:
        root = Path(temporary)
        old = root / 'old-moe.gguf'
        api.fixtures.fixture_copy(ROOT / 'tests/fixtures/tiny-qwen3moe.gguf', old, False)
        api.add_template(old)
        for architecture, filename in families:
            target = root / filename
            api.fixtures.fixture_copy(ROOT / 'tests/fixtures' / filename, target, False)
            api.add_template(target)
        settings = dict(backend='cpu', device=None, ctx=512, max_tokens=32, parallel=1,
                        max_queued=2, queue_timeout_secs=5, vram_mib=0, expert_ram_mib=16,
                        ram_margin_mib=0, expert_warmup=True, threads=2)
        saved = root / 'service.json'
        saved.write_text(json.dumps(dict(directory=str(root), model=old.name, settings=settings)))

        def exercise(origin, request, json_request):
            def call(path, body=None, expected=(200,)):
                code, value = json_request(path, body)
                assert code in expected, (path, code, value)
                return value

            def finish():
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    job = call('/api/tuning/status')
                    if job['phase'] in ('complete', 'failed', 'cancelled'):
                        assert job['phase'] == 'complete', job
                        return job
                    time.sleep(.02)
                raise AssertionError('Dense model setup timed out')

            def chat(model):
                response = call('/v1/chat/completions', api.body(model, max_tokens=4))
                assert response['model'] == model, response
                assert response['choices'] and response['choices'][0]['message']['role'] == 'assistant'
                assert isinstance(response['choices'][0]['message']['content'], str)
                assert response['usage']['prompt_tokens'] > 0 and response['usage']['completion_tokens'] <= 4

            scanned = call('/api/models/scan', {'directory': str(root)})
            entries = {entry['id']: entry for entry in scanned['models']}
            previous = old
            for architecture, filename in families:
                assert filename in entries, scanned
                assert entries[filename]['architecture'] == architecture
                if architecture == 'qwen35':
                    assert entries[filename]['native_backends'] == ['cpu', 'hybrid']
                target = dict(directory=str(root), model=filename)
                unchanged = saved.read_bytes()
                call('/api/tuning/start', dict(target=target, mode='prepare',
                     goal=dict(context='maximum_fit', max_context=384), activate=False), expected=(200, 202))
                prepared = finish()
                assert prepared['preparing'] and not prepared['activated']
                assert prepared['result']['winner'], prepared
                active = call('/v1/status')
                assert active['model'] == previous.name and Path(active['runtime']['shards'][0]).resolve() == previous.resolve()
                assert saved.read_bytes() == unchanged
                call('/api/tuning/use', target, expected=(200, 202))
                loaded = finish()
                assert loaded['activated'], loaded
                active = call('/v1/status')
                assert active['model'] == filename, active
                assert active['runtime']['config']['architecture'] == architecture, active
                assert active['runtime']['config']['expert_count'] == 0, 'Expected an actually served dense model'
                assert Path(active['runtime']['shards'][0]).resolve() == (root / filename).resolve()
                assert any(entry['id'] == filename for entry in call('/v1/models')['data'])
                chat(filename)
                profile = call('/api/tuning/profile', target)
                assert profile['cached'] and profile['result']['winner']
                previous = root / filename
                print(f'Served dense {architecture}: {filename} scan→Prepare→Use→chat→saved profile', flush=True)

        with helpers.service(binary, saved) as service:
            exercise(*service)
        # Saved model startup itself goes through ServedModel::load; this must
        # accept dense checkpoints too, rather than silently falling back unloaded.
        with helpers.service(binary, saved) as (_, request, json_request):
            code, active = json_request('/v1/status')
            assert code == 200 and active['model'] == 'tiny-qwen35-mtp.gguf', active
            for _, filename in families:
                code, profile = json_request('/api/tuning/profile', dict(directory=str(root), model=filename))
                assert code == 200 and profile['cached'], profile
            code, job = json_request('/api/tuning/use', dict(directory=str(root), model='tiny-qwen3.gguf'))
            assert code in (200, 202)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                _, job = json_request('/api/tuning/status')
                if job['phase'] == 'complete':
                    break
                assert job['phase'] != 'failed', job
                time.sleep(.02)
            assert job['phase'] == 'complete' and job['cached'], job
            code, response = json_request('/v1/chat/completions', api.body('tiny-qwen3.gguf', max_tokens=4))
            assert code == 200 and response['model'] == 'tiny-qwen3.gguf', response
        print('Dense serving regression passed: Qwen3 dense, Qwen3.5, Qwen3.5+MTP, real chat, restoration, restart and saved-profile reuse. CPU fixtures only.')


if __name__ == '__main__':
    main()
