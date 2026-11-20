#!/usr/bin/env python3
"""Exercise the native configuration service with real synthetic CPU execution.

The optional Chromium checks use isolated browser storage and loopback-only
networking. No model download, Windows execution or GPU performance is inferred.
Requires a built forge-engine; --browser additionally requires Python Playwright.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
KEY = 'forge-native-ui-local-validation-only'


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


api_helpers = import_module('forge_native_ui_fixtures', ROOT / 'scripts/validate-native-api.py')
api_helpers.KEY = KEY


def missing_shard(path: Path) -> None:
    def text(value: str) -> bytes:
        raw = value.encode()
        return struct.pack('<Q', len(raw)) + raw

    entries = [
        text('split.count') + struct.pack('<II', 4, 2),
        text('split.no') + struct.pack('<II', 4, 0),
        text('split.tensors.count') + struct.pack('<IQ', 10, 1),
        text('general.architecture') + struct.pack('<I', 8) + text('qwen3moe'),
    ]
    raw = b'GGUF' + struct.pack('<IQQ', 3, 0, len(entries)) + b''.join(entries)
    path.write_bytes(raw + bytes(-len(raw) % 32))


@contextlib.contextmanager
def service(binary: Path, settings_file: Path, *, bind_host='127.0.0.1',
            connect_host='127.0.0.1'):
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1', 0))
        port = reserve.getsockname()[1]
    origin = f'http://{connect_host}:{port}'
    command = [str(binary), '--threads', '2', 'serve', '--host', bind_host, '--port', str(port),
               '--api-key', KEY, '--settings-file', str(settings_file)]
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log)

        def request(path, body=None, *, auth=True, content_type='application/json',
                    origin_header=None, host=None, method=None):
            headers = {'Authorization': 'Bearer ' + KEY} if auth else {}
            if body is not None:
                headers['Content-Type'] = content_type
            if origin_header is not None:
                headers['Origin'] = origin_header
            if host is not None:
                headers['Host'] = host
            connection = http.client.HTTPConnection(connect_host, port, timeout=60)
            try:
                connection.request(method or ('POST' if body is not None else 'GET'), path,
                                   None if body is None else json.dumps(body), headers)
                response = connection.getresponse()
                return response.status, {k.lower(): v for k, v in response.getheaders()}, response.read().decode()
            finally:
                connection.close()

        def json_request(path, body=None, **kwargs):
            status, headers, raw = request(path, body, **kwargs)
            return status, json.loads(raw)

        try:
            deadline = time.monotonic() + 45
            while True:
                if process.poll() is not None:
                    raise AssertionError('Native configuration service exited before startup')
                try:
                    if request('/', auth=False)[0] == 200:
                        break
                except (OSError, http.client.HTTPException):
                    pass
                if time.monotonic() >= deadline:
                    raise AssertionError('Native configuration service startup timed out')
                time.sleep(.025)
            yield origin, request, json_request
        except BaseException:
            log.seek(0)
            print(log.read().decode(errors='replace'), file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--browser', action='store_true')
    parser.add_argument('--chromium', type=Path, default=Path('/usr/bin/chromium'))
    parser.add_argument('--screenshots', type=Path, help='Optional desktop/mobile review screenshots directory')
    args = parser.parse_args()
    binary = args.binary.resolve()
    checks: list[dict] = []
    failure = None
    started = time.monotonic()

    def check(name: str, value: bool, detail=None):
        checks.append({'name': name, 'passed': bool(value), 'detail': detail})
        if not value:
            raise AssertionError(f'{name}: {detail}')

    try:
        version = subprocess.run([str(binary), '--version'], capture_output=True, text=True,
                                 timeout=20, check=True).stdout.strip()
        check('built-executable', True, {'version': version, 'sha256': hashlib.sha256(binary.read_bytes()).hexdigest()})
        with tempfile.TemporaryDirectory(prefix='forge-native-ui-') as temporary:
            root = Path(temporary)
            models = root / 'Models with spaces'
            models.mkdir()
            fixture = models / 'fixture.gguf'
            api_helpers.fixtures.fixture_copy(ROOT / 'tests/fixtures/tiny-qwen3moe.gguf', fixture, False)
            api_helpers.add_template(fixture)
            shutil.copyfile(ROOT / 'tests/fixtures/tiny-qwen3.gguf', models / 'dense.gguf')
            (models / 'invalid.gguf').write_bytes(b'invalid GGUF test input')
            missing_shard(models / 'partial-00001-of-00002.gguf')
            (models / 'weights.safetensors').write_bytes(b'not loaded by this test')
            suspicious_name = '<img src=x onerror=window.forgeXss=1>.gguf' if os.name != 'nt' else 'literal-ampersand-&.gguf'
            shutil.copyfile(fixture, models / suspicious_name)
            outside = root / 'outside'
            outside.mkdir()
            shutil.copyfile(fixture, outside / 'outside.gguf')
            symlink_created = False
            try:
                (models / 'linked').symlink_to(outside, target_is_directory=True)
                symlink_created = True
            except OSError:
                pass
            config_file = root / 'native-service.json'
            settings = {
                'backend': 'cpu', 'device': None, 'ctx': 2048, 'max_tokens': 64,
                'parallel': 2, 'max_queued': 5, 'queue_timeout_secs': 9,
                'vram_mib': 0, 'expert_ram_mib': 16, 'ram_margin_mib': 0,
                'expert_warmup': True,
            }
            payload = {'directory': str(models), 'model': 'fixture.gguf', 'settings': settings}

            with service(binary, config_file) as (origin, request, json_request):
                status, headers, page = request('/', auth=False)
                check('unloaded-public-ui-without-gpu-initialization', status == 200 and 'Forge9070' in page
                      and KEY not in page and not config_file.exists(), {'status': status})
                csp = headers.get('content-security-policy', '')
                check('ui-security-headers', headers.get('x-content-type-options') == 'nosniff'
                      and "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
                      and "'unsafe-inline'" not in csp, headers)
                for asset, mime in [('/ui/app.js', 'javascript'), ('/ui/style.css', 'text/css')]:
                    status, headers, raw = request(asset, auth=False)
                    check('public-asset-' + asset.rsplit('/', 1)[-1], status == 200
                          and mime in headers.get('content-type', '') and bool(raw) and KEY not in raw,
                          {'status': status, 'content_type': headers.get('content-type')})
                for path in ['/api/config', '/api/hardware', '/v1/status', '/v1/models', '/health']:
                    check('api-auth-' + path, request(path, auth=False)[0] == 401)
                check('no-model-health-is-not-ready', request('/health')[0] == 503)
                status, config = json_request('/api/config')
                check('configuration-with-no-model-or-secret', status == 200 and config['model'] is None
                      and config['settings']['backend'] == 'auto' and KEY not in json.dumps(config), config)
                status, hardware = json_request('/api/hardware')
                check('hardware-report-remains-available-without-a-gpu', status == 200
                      and isinstance(hardware.get('adapters'), list)
                      and isinstance(hardware.get('vulkan_available'), bool)
                      and hardware.get('cpu_available') is True, hardware)
                check('foreign-host-rejected', request('/api/config', host='foreign.invalid')[0] in [400, 403])
                check('foreign-origin-rejected', request('/api/config', origin_header='https://foreign.invalid')[0] in [400, 403])
                check('matching-origin-accepted', request('/api/config', origin_header=origin)[0] == 200)
                check('scan-requires-json', request('/api/models/scan', {'directory': str(models)}, content_type='text/plain')[0] == 400)
                check('apply-requires-json', request('/api/config', payload, content_type='text/plain')[0] == 400)
                check('scan-does-not-create-missing-directory', request('/api/models/scan', {'directory': str(root / 'absent')})[0] == 400
                      and not (root / 'absent').exists())
                status, scan = json_request('/api/models/scan', {'directory': str(models)})
                found = {model['id']: model for model in scan.get('models', [])}
                rejected = {Path(model['path']).name: model['reason'] for model in scan.get('rejected', [])}
                check('scan-validates-complete-supported-moe-files', status == 200
                      and set(found) == {'fixture.gguf', suspicious_name}
                      and all(model['bytes'] > 0 and model['architecture'] == 'qwen3moe' for model in found.values()), scan)
                check('scan-explains-dense-corrupt-and-missing-shards', all(name in rejected for name in
                      ['dense.gguf', 'invalid.gguf', 'partial-00001-of-00002.gguf'])
                      and bool(rejected['partial-00001-of-00002.gguf']), rejected)
                check('scan-reports-non-gguf-weights', scan.get('safetensors_files') == 1
                      and scan.get('truncated') is False, scan.get('safetensors_files'))
                if symlink_created:
                    check('scan-does-not-follow-directory-symlinks', not any('outside.gguf' in item for item in found), found)
                check('scan-does-not-load-or-save-model', json_request('/api/config')[1]['model'] is None
                      and not config_file.exists())
                status, applied = json_request('/api/config', payload)
                check('apply-loads-model-and-reconfigures-admission', status == 200
                      and applied['model'] == 'fixture.gguf', applied)
                status, stats = json_request('/v1/status')
                check('runtime-uses-requested-settings', status == 200 and stats['admission']['capacity'] == 2
                      and stats['admission']['queue_limit'] == 5 and stats['runtime']['execution']['backend'] == 'cpu', stats)
                saved = config_file.read_bytes()
                check('successful-apply-persists-without-secrets', json.loads(saved)['settings'] == settings
                      and json.loads(saved)['model'] == 'fixture.gguf' and KEY.encode() not in saved, json.loads(saved))
                status, answer = json_request('/v1/chat/completions', api_helpers.body('fixture.gguf'))
                check('applied-model-executes-real-generation', status == 200
                      and answer['choices'][0]['message']['content'] == 'AAAA', answer)
                for invalid in [{'parallel': 0}, {'parallel': 5}, {'ctx': 0}, {'max_tokens': 2048},
                                {'max_queued': 4097}, {'queue_timeout_secs': 0},
                                {'backend': 'pretend'}, {'expert_ram_mib': -1}]:
                    bad = dict(payload, settings=dict(settings, **invalid))
                    check('invalid-settings-rejected-' + next(iter(invalid)), request('/api/config', bad)[0] == 400, invalid)
                check('rejected-settings-preserve-running-profile', config_file.read_bytes() == saved
                      and json_request('/api/config')[1]['settings'] == settings)
                missing = dict(payload, model='not-downloaded.gguf')
                check('missing-model-does-not-replace-runtime', request('/api/config', missing)[0] in [400, 404]
                      and config_file.read_bytes() == saved and json_request('/api/config')[1]['model'] == 'fixture.gguf')
                traversal = dict(payload, model='../outside/outside.gguf')
                check('model-selection-cannot-escape-selected-root', request('/api/config', traversal)[0] in [400, 404])
                status, answer = json_request('/v1/chat/completions', api_helpers.body('fixture.gguf'))
                check('failed-configuration-keeps-generation-working', status == 200
                      and answer['choices'][0]['message']['content'] == 'AAAA')
                backup = config_file.with_suffix('.backup')
                config_file.rename(backup)
                config_file.mkdir()
                try:
                    failed_save = dict(payload, settings=dict(settings, ctx=1024, max_tokens=32))
                    status, data = json_request('/api/config', failed_save)
                    check('settings-write-failure-does-not-commit-runtime', status >= 400
                          and json_request('/api/config')[1]['settings'] == settings
                          and json_request('/api/config')[1]['model'] == 'fixture.gguf', data)
                finally:
                    config_file.rmdir()
                    backup.rename(config_file)
                expanded = models / 'expanded.gguf'
                api_helpers.scheduling.expand_fixture(ROOT / 'tests/fixtures/tiny-qwen3moe.gguf', expanded)
                api_helpers.add_template(expanded)
                long_settings = dict(settings, ctx=32768, max_tokens=16384, parallel=1)
                busy_payload = dict(payload, model='expanded.gguf', settings=long_settings)
                status, data = json_request('/api/config', busy_payload)
                check('prepare-real-busy-configuration-check', status == 200, data)
                busy_saved = config_file.read_bytes()
                port = urllib.parse.urlsplit(origin).port
                job = api_helpers.Job(port, api_helpers.body('expanded.gguf', max_tokens=16384, stream=True))
                try:
                    api_helpers.wait_for(request, lambda value: value['admission']['active'] == 1)
                    status, data = json_request('/api/config', payload)
                    check('active-generation-blocks-configuration-without-saving', status == 409
                          and config_file.read_bytes() == busy_saved
                          and json_request('/api/config')[1]['settings'] == long_settings, data)
                finally:
                    job.cancel()
                    api_helpers.wait_for(request, lambda value: value['admission']['active'] == 0)
                changed_settings = dict(settings, ctx=1024, max_tokens=32, parallel=4, max_queued=7,
                                        queue_timeout_secs=3)
                changed = dict(payload, settings=changed_settings)
                status, data = json_request('/api/config', changed)
                stats = json_request('/v1/status')[1]
                check('idle-apply-updates-context-and-queue-capacity', status == 200
                      and data['settings'] == changed_settings and stats['admission']['capacity'] == 4
                      and stats['admission']['queue_limit'] == 7, data)
                unloaded = dict(changed, model=None)
                status, data = json_request('/api/config', unloaded)
                check('unload-retains-settings-and-frees-model', status == 200 and data['model'] is None
                      and request('/health')[0] == 503 and json_request('/v1/status')[1]['runtime'] is None, data)
                check('generation-rejected-cleanly-without-model',
                      request('/v1/chat/completions', api_helpers.body('fixture.gguf'))[0] == 503)

            with service(binary, config_file) as (origin, request, json_request):
                status, restored = json_request('/api/config')
                # Windows canonicalization expands short names and adds the
                # extended-path prefix. Verify the directory, not its spelling.
                check('saved-settings-restore-after-process-restart', status == 200
                      and restored['model'] is None and restored['settings'] == changed_settings
                      and Path(restored['models_dir']).samefile(models), restored)
                if args.browser:
                    browser_checks(origin, request, json_request, models, suspicious_name, args, check)
                else:
                    checks.append({'name': 'browser-checks', 'passed': None, 'skipped': True,
                                   'detail': 'Not requested; API-only run.'})
                # Leave the profile loaded, then verify automatic startup loading independently.
                status, data = json_request('/api/config', payload)
                check('prepare-profile-for-loaded-restart', status == 200, data)
            with service(binary, config_file) as (_, request, json_request):
                status, restored = json_request('/api/config')
                check('saved-model-loads-after-process-restart', status == 200 and restored['model'] == 'fixture.gguf'
                      and restored['settings'] == settings and request('/health')[0] == 200, restored)
            saved_profile = config_file.read_bytes()
            missing = fixture.with_suffix('.unavailable')
            fixture.rename(missing)
            try:
                with service(binary, config_file) as (_, request, json_request):
                    status, restored = json_request('/api/config')
                    check('missing-saved-checkpoint-keeps-recovery-ui-accessible', status == 200
                          and restored['model'] is None and bool(restored.get('startup_error'))
                          and request('/health')[0] == 503 and config_file.read_bytes() == saved_profile,
                          restored)
            finally:
                missing.rename(fixture)
            moved_folder = root / 'temporarily unavailable models'
            models.rename(moved_folder)
            try:
                with service(binary, config_file) as (_, request, json_request):
                    status, restored = json_request('/api/config')
                    check('missing-saved-model-folder-keeps-recovery-ui-accessible', status == 200
                          and restored['model'] is None and bool(restored.get('startup_error'))
                          and request('/health')[0] == 503 and config_file.read_bytes() == saved_profile,
                          restored)
            finally:
                moved_folder.rename(models)
            auto_file = root / 'auto-profile.json'
            auto_settings = dict(settings, backend='auto')
            auto_payload = {'directory': str(models), 'model': None, 'settings': auto_settings}
            with service(binary, auto_file) as (_, request, json_request):
                status, configured = json_request('/api/config', auto_payload)
                check('automatic-backend-persists-without-loading-model', status == 200
                      and configured['settings'] == auto_settings and configured['model'] is None
                      and json.loads(auto_file.read_bytes())['settings'] == auto_settings
                      and json_request('/v1/status')[1]['runtime'] is None, configured)
            with service(binary, auto_file) as (_, request, json_request):
                status, restored = json_request('/api/config')
                check('automatic-backend-restores-after-restart', status == 200
                      and restored['settings'] == auto_settings and restored['model'] is None, restored)
            # An alternate loopback address exercises the same Host path as a
            # LAN client without requiring a particular network interface.
            with service(binary, root / 'wildcard-profile.json', bind_host='0.0.0.0',
                         connect_host='127.0.0.2') as (origin, request, json_request):
                check('wildcard-listener-accepts-actual-interface-host',
                      request('/api/config', origin_header=origin)[0] == 200)
                check('wildcard-listener-still-requires-authentication',
                      request('/api/config', auth=False)[0] == 401)
                check('wildcard-listener-rejects-unrelated-host',
                      request('/api/config', host='127.0.0.3:9070')[0] == 400)
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        report = {
            'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope': 'Native configuration HTTP API and optional headless Chromium, synthetic CPU model, local connections including an authenticated wildcard-bind regression. No real-model quality, WSL GPU, or hardware performance claims.',
            'seconds': round(time.monotonic() - started, 3),
            'passed': sum(bool(record['passed']) for record in checks),
            'skipped': sum(bool(record.get('skipped')) for record in checks),
            'failure': failure, 'records': checks,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'passed': report['passed'], 'skipped': report['skipped'],
                          'failure': failure, 'output': str(args.output)}))


def browser_checks(origin, request, json_request, models, suspicious_name, args, check):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(args.chromium), headless=True)
        try:
            context = browser.new_context(viewport={'width': 1440, 'height': 1080}, reduced_motion='reduce')
            context.route('**/*', lambda route: route.continue_()
                          if route.request.url.startswith(origin + '/') else route.abort())
            page = context.new_page()
            page.set_default_timeout(15000)
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(origin + '/', wait_until='domcontentloaded')
            page.locator('#auth-panel').wait_for(state='visible')
            check('browser-prompts-for-existing-api-key', page.locator('#api-key').is_visible())
            page.locator('#api-key').fill(KEY)
            page.locator('#unlock').click()
            page.locator('#auth-panel').wait_for(state='hidden')
            page.wait_for_function("() => document.querySelector('#ctx').value === '1024'")
            check('browser-inherits-effective-saved-settings', page.locator('#backend').input_value() == 'cpu'
                  and page.locator('#parallel').input_value() == '4')
            page.locator('#backend').select_option('auto')
            check('browser-exposes-automatic-backend-with-gpu-controls',
                  'Automatically' in page.locator('#backend-help').inner_text()
                  and page.locator('#device').is_enabled() and page.locator('#vram-mib').is_enabled())
            # UI presentation checks use explicitly synthetic reports. They do
            # not imply that these adapters or automatic GPU execution ran.
            hardware = {'cpu_available': True, 'adapters': [
                {'index': 0, 'name': 'Synthetic compatible GPU', 'software': False,
                 'compatible': True, 'rejection_reasons': [], 'compute_queue': 0},
                {'index': 1, 'name': 'Synthetic incompatible GPU', 'software': False,
                 'compatible': False, 'rejection_reasons': ['Requires seven storage buffers'],
                 'compute_queue': 0},
                {'index': 2, 'name': 'Synthetic software adapter', 'software': True,
                 'compatible': False, 'rejection_reasons': ['Software is disabled'], 'compute_queue': 0},
            ]}
            hardware_url = origin + '/api/hardware'
            page.route(hardware_url, lambda route: route.fulfill(json=hardware))
            try:
                page.locator('#refresh-hardware').click()
                page.wait_for_function("() => document.querySelector('#hardware-list').textContent.includes('Synthetic compatible GPU')")
                check('browser-filters-synthetic-hardware-by-all-capabilities',
                      not page.locator('#device option[value="0"]').is_disabled()
                      and page.locator('#device option[value="1"]').is_disabled()
                      and page.locator('#device option[value="2"]').is_disabled()
                      and 'Requires seven storage buffers' in page.locator('#hardware-list').inner_text()
                      and '1 hardware adapter meets' in page.locator('#hardware-status').inner_text())
            finally:
                page.unroute(hardware_url)
            status_url = origin + '/v1/status'
            selection_reason = 'No compatible hardware Vulkan adapter was found'
            synthetic_status = dict(json_request('/v1/status')[1], model='synthetic-auto-cpu', runtime={
                'config': {'architecture': 'qwen3moe'}, 'mapped_weight_bytes': 0,
                'execution': {'backend': 'cpu'}, 'backend_selection': {
                    'requested': 'auto', 'selected': 'cpu', 'automatic': True, 'reason': selection_reason,
                },
            })
            page.route(status_url, lambda route: route.fulfill(json=synthetic_status))
            try:
                page.locator('#refresh-status').click()
                page.wait_for_function("() => document.querySelector('#loaded-model-name').textContent === 'synthetic-auto-cpu'")
                check('browser-explains-synthetic-automatic-cpu-selection',
                      page.locator('#execution-name').inner_text() == 'CPU'
                      and 'Automatic selection' in page.locator('#execution-detail').inner_text()
                      and selection_reason in page.locator('#execution-detail').inner_text()
                      and selection_reason in page.locator('#runtime-stats').inner_text())
            finally:
                page.unroute(status_url)
            page.locator('#refresh-status').click()
            page.wait_for_function("() => document.querySelector('#loaded-model-name').textContent === 'No model loaded'")
            page.locator('#backend').select_option('cpu')
            page.locator('#model-directory').fill(str(models / 'missing download folder'))
            with page.expect_response(lambda response: urllib.parse.urlsplit(response.url).path == '/api/models/scan'
                                      and response.request.method == 'POST') as missing:
                page.locator('#scan-models').click()
            page.wait_for_function("() => !document.querySelector('#scan-models').disabled")
            check('browser-explains-missing-folder-and-allows-retry', missing.value.status == 400
                  and page.locator('#notice').is_visible()
                  and bool(page.locator('#notice').inner_text()))
            page.locator('#model-directory').fill(str(models))
            with page.expect_response(lambda response: urllib.parse.urlsplit(response.url).path == '/api/models/scan'
                                      and response.request.method == 'POST') as scanned:
                page.locator('#scan-models').click()
            check('browser-scans-real-model-folder', scanned.value.status == 200)
            page.locator('input[name="model"][value="fixture.gguf"]').wait_for()
            displayed = page.locator('#model-results').inner_text()
            check('filesystem-names-render-as-text', suspicious_name in displayed
                  and page.evaluate('() => window.forgeXss === undefined')
                  and page.locator('#model-results img').count() == 0, displayed)
            page.locator('input[name="model"][value="fixture.gguf"]').check()
            page.locator('#ctx').fill('2048')
            page.locator('#max-tokens').fill('64')
            page.locator('#parallel').select_option('2')
            with page.expect_response(lambda response: urllib.parse.urlsplit(response.url).path == '/api/config'
                                      and response.request.method == 'POST') as applied:
                page.locator('#apply-settings').click()
            check('browser-loads-real-model-with-chosen-settings', applied.value.status == 200, applied.value.json())
            page.wait_for_function("() => !document.querySelector('#scan-models').disabled")
            effective = json_request('/api/config')[1]
            check('browser-configuration-takes-effect', effective['settings']['ctx'] == 2048
                  and effective['settings']['parallel'] == 2 and effective['model'] == 'fixture.gguf', effective)
            page.locator('#ctx').fill('3072')
            with page.expect_response(lambda response: urllib.parse.urlsplit(response.url).path == '/v1/status', timeout=15000):
                pass
            check('status-poll-preserves-unsaved-form-edits', page.locator('#ctx').input_value() == '3072'
                  and json_request('/api/config')[1]['settings']['ctx'] == 2048)
            page.locator('#ctx').fill('2048')
            with page.expect_download() as exported:
                page.locator('#export-settings').click()
            exported_profile = Path(exported.value.path()).read_text()
            profile = json.loads(exported_profile)
            check('browser-exports-usable-profile-without-api-key', exported.value.suggested_filename == 'forge-service.json'
                  and set(profile) == {'directory', 'model', 'settings'}
                  and profile['model'] == 'fixture.gguf' and profile['settings']['ctx'] == 2048
                  and profile['settings']['parallel'] == 2 and KEY not in exported_profile, profile)
            check('browser-stores-no-api-secret', page.evaluate(
                "key => !JSON.stringify(localStorage).includes(key) && !JSON.stringify(sessionStorage).includes(key)", KEY))
            desktop_overflow = page.evaluate('() => ({width: innerWidth, scroll: document.documentElement.scrollWidth})')
            check('desktop-has-no-horizontal-overflow', desktop_overflow['scroll'] <= desktop_overflow['width'] + 1, desktop_overflow)
            if args.screenshots:
                args.screenshots.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(args.screenshots / 'native-ui-desktop.png'), full_page=True)
            page.set_viewport_size({'width': 390, 'height': 844})
            mobile_overflow = page.evaluate('() => ({width: innerWidth, scroll: document.documentElement.scrollWidth})')
            check('mobile-has-no-horizontal-overflow', mobile_overflow['scroll'] <= mobile_overflow['width'] + 1, mobile_overflow)
            if args.screenshots:
                page.screenshot(path=str(args.screenshots / 'native-ui-mobile.png'), full_page=True)
            with page.expect_response(lambda response: urllib.parse.urlsplit(response.url).path == '/api/config'
                                      and response.request.method == 'POST') as unloaded:
                page.locator('#unload-model').click()
            check('browser-unloads-model', unloaded.value.status == 200
                  and json_request('/api/config')[1]['model'] is None)
            check('no-uncaught-browser-errors', not errors, errors)
            page.reload(wait_until='domcontentloaded')
            page.locator('#auth-panel').wait_for(state='visible')
            check('browser-key-is-memory-only-after-reload', page.locator('#api-key').input_value() == '')
            context.close()
        finally:
            browser.close()


if __name__ == '__main__':
    main()
