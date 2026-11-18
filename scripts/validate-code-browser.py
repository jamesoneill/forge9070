#!/usr/bin/env python3
"""Check the served code-context UI in isolated, headless Chromium.

Requires Python Playwright and a Chromium executable; neither is a product
dependency. Uses the bundled untrained CPU fixture, a private loopback server
and fresh browser storage. No downloads, GPU execution or quality benchmark.
Run after building: --exe target/release/forge9070
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True, type=Path)
    parser.add_argument('--chromium', type=Path, default=Path('/usr/bin/chromium'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/code-browser-integration-0.6.0.json'))
    parser.add_argument('--screenshot', type=Path, help='Optional internal-review settings screenshot; omit from release archives.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    exe = args.exe.resolve()
    records: list[dict] = []
    version = 'unknown'
    failure = None

    def check(name: str, condition: bool, detail=None) -> None:
        records.append({'name': name, 'passed': bool(condition), 'detail': detail})
        if not condition:
            raise AssertionError(f'{name}: {detail}')

    try:
        result = subprocess.run([str(exe), '--version'], cwd=root, capture_output=True, text=True, timeout=20, check=True)
        version = result.stdout.strip()
        check('executable-version', bool(version), version)
        digest = hashlib.sha256(exe.read_bytes()).hexdigest()
        records.append({'name': 'executable-identity', 'passed': True, 'path': str(exe), 'sha256': digest})
        with tempfile.TemporaryDirectory(prefix='forge-code-browser-') as temporary:
            tmp = Path(temporary)
            models = tmp / 'models'
            models.mkdir()
            model = models / 'fixture.gguf'
            shutil.copyfile(root / 'tests/fixtures/tiny-qwen3moe.gguf', model)
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            origin = f'http://127.0.0.1:{port}'
            key = 'forge-browser-local-test-only'
            env = os.environ.copy()
            env.update({'STRATA_API_KEY': key, 'FORGE9070_API_KEY': key,
                        'XDG_CACHE_HOME': str(tmp / 'cache')})
            argv = [str(exe), 'serve', '--cpu', '--model', str(model), '--models-dir', str(models),
                    '--port', str(port), '--ctx', '512', '--chunk', '32', '--threads', '2',
                    '--max-tokens', '64', '--temperature', '1', '--top-p', '0.95', '--top-k', '20',
                    '--think', 'true', '--preserve-thinking', 'true', '--expert-ram', '0',
                    '--expert-warmup', 'false', '--ram-margin', '0', '--cache-ram', '0']
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

            def request(path: str):
                req = urllib.request.Request(origin + path, headers={'Authorization': 'Bearer ' + key})
                with opener.open(req, timeout=20) as response:
                    return response.status, json.load(response)

            with (tmp / 'server.log').open('w+') as log:
                server = subprocess.Popen(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 45
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError('CPU test server exited during startup')
                        try:
                            status, health = request('/health')
                            if status == 200 and health['status'] == 'ok':
                                break
                        except (OSError, urllib.error.URLError):
                            pass
                        if time.monotonic() >= deadline:
                            raise TimeoutError('CPU test server did not become ready')
                        time.sleep(.05)
                    check('private-cpu-server-ready', True, health)
                    status, catalog = request('/api/models')
                    actual_options = catalog['options']
                    check('server-configuration', status == 200 and actual_options['ctx'] == 512
                          and actual_options['max_tokens'] == 64 and not actual_options['gpu']
                          and actual_options['temperature'] == 1 and abs(actual_options['top_p'] - .95) < 1e-6
                          and actual_options['top_k'] == 20, actual_options)

                    with sync_playwright() as playwright:
                        browser = playwright.chromium.launch(executable_path=str(args.chromium), headless=True)
                        try:
                            context = browser.new_context(viewport={'width': 1360, 'height': 1000}, reduced_motion='reduce',
                                                          extra_http_headers={'Authorization': 'Bearer ' + key})
                            # The product page is self contained. Keep this verification entirely on loopback.
                            context.route('**/*', lambda route: route.continue_() if route.request.url.startswith(origin + '/') else route.abort())
                            page = context.new_page()
                            page.set_default_timeout(15000)
                            browser_errors: list[str] = []
                            posts: list[dict] = []
                            page.on('pageerror', lambda error: browser_errors.append(str(error)))

                            def observe(req):
                                path = urllib.parse.urlsplit(req.url).path
                                if req.method == 'POST' and path in ('/api/context', '/v1/chat/completions', '/api/load'):
                                    posts.append({'path': path, 'body': json.loads(req.post_data or '{}')})

                            page.on('request', observe)

                            def ready():
                                page.wait_for_function("() => typeof engineSettings !== 'undefined' && engineSettings?.ctx === 512 && $('model').value && !busy")

                            def controls():
                                return page.evaluate("""() => ({output: Number($('maxtok').value), temperature: Number($('temp').value),
                                    top_p: Number($('topp').value), top_k: Number($('topk').value), think: $('think').checked,
                                    ctx: Number($('ectx').value), ctx_max: Number($('ectx').max), output_max: Number($('maxtok').max)})""")

                            def has_path(response, path):
                                return urllib.parse.urlsplit(response.url).path == path and response.request.method == 'POST'

                            def setting(name: str, value):
                                # Exercise the actual input/store listener for controls such as range sliders.
                                page.locator('#' + name).evaluate("(el, value) => {el.value = String(value); el.dispatchEvent(new Event('input', {bubbles: true}));}", value)

                            page.goto(origin + '/', wait_until='domcontentloaded')
                            ready()
                            initial = controls()
                            check('first-visit-inherits-real-server-defaults', initial == {
                                'output': 64, 'temperature': 1, 'top_p': .95, 'top_k': 20, 'think': True,
                                'ctx': 512, 'ctx_max': 512, 'output_max': 64}, initial)
                            check('initial-defaults-are-not-saved-user-overrides', page.evaluate(
                                "() => ['temp','topp','topk','maxtok','think'].every(id => store.get(id, null) === null)"))
                            if args.screenshot:
                                page.locator('#settings').click()
                                page.locator('[data-tab="chat"]').click()
                                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                                page.screenshot(path=str(args.screenshot), full_page=True)
                                page.locator('#close').click()
                                records.append({'name': 'optional-settings-screenshot', 'passed': True, 'path': str(args.screenshot)})

                            text = 'Explain why cancellation must refund only once.'
                            page.locator('#input').fill(text)
                            before = request('/api/stats')[1]
                            offset = len(posts)
                            with page.expect_response(lambda response: has_path(response, '/api/context')) as preview_response:
                                page.locator('#checkbudget').click()
                            preview = preview_response.value.json()
                            page.wait_for_function("() => !$('checkbudget').disabled")
                            note = page.locator('#contextbudget').inner_text()
                            after = request('/api/stats')[1]
                            check('manual-token-preview-renders-exact-budget', preview_response.value.status == 200 and preview['fits']
                                  and preview['prompt_tokens'] > 0 and preview['output_tokens'] == 64
                                  and f"{preview['prompt_tokens']} prompt" in note and '64 reserved output / 512' in note,
                                  {'preview': preview, 'note': note})
                            check('manual-preview-keeps-composer-and-does-no-inference', page.locator('#input').input_value() == text
                                  and before.get('requests') == after.get('requests')
                                  and all(p['path'] != '/v1/chat/completions' for p in posts[offset:]), posts[offset:])

                            # 470 source bytes plus template and a 64-token reservation exceed this fixture's context.
                            page.locator('#input').fill('x' * 470)
                            before = request('/api/stats')[1]
                            offset = len(posts)
                            with page.expect_response(lambda response: has_path(response, '/api/context')) as overflow_response:
                                page.locator('#send').click()
                            overflow = overflow_response.value.json()
                            page.wait_for_function("() => !busy && cur?.msgs.at(-1)?.error === true")
                            after = request('/api/stats')[1]
                            check('automatic-preview-blocks-oversize-before-generation', overflow_response.value.status == 200
                                  and not overflow['fits'] and overflow['error']['code'] == 'context_length_exceeded'
                                  and not any(p['path'] == '/v1/chat/completions' for p in posts[offset:])
                                  and before.get('requests') == after.get('requests'),
                                  {'preview': overflow, 'posts': posts[offset:], 'ui_error': page.locator('#status').inner_text()})
                            check('rejected-request-leaves-ui-controls-usable', page.locator('#input').is_editable()
                                  and page.locator('#send').is_enabled() and page.locator('#new').is_enabled())

                            # Start a clean chat and select explicit user settings for a deterministic four-token response.
                            page.locator('#new').click()
                            page.locator('#settings').click()
                            page.locator('[data-tab="chat"]').click()
                            page.locator('#maxtok').fill('4')
                            setting('temp', 0)
                            page.locator('#close').click()
                            page.locator('#think').uncheck()
                            page.locator('#input').fill('hello')
                            offset = len(posts)
                            with page.expect_response(lambda response: has_path(response, '/v1/chat/completions')) as answer_response:
                                page.locator('#send').click()
                            page.wait_for_function("() => !busy && cur?.msgs.at(-1)?.finish === 'length'")
                            answer = page.evaluate("() => cur.msgs.at(-1)")
                            sent = posts[offset:]
                            check('generation-works-after-browser-budget-rejection', answer_response.value.status == 200
                                  and answer['content'] == 'AAAA' and not answer.get('error')
                                  and [p['path'] for p in sent] == ['/api/context', '/v1/chat/completions']
                                  and bool(sent[-1]['body'].get('model')), {'answer': answer, 'posts': sent})
                            page.reload(wait_until='domcontentloaded')
                            ready()
                            saved = controls()
                            check('explicit-browser-settings-survive-server-default-refresh', saved['output'] == 4
                                  and saved['temperature'] == 0 and saved['think'] is False
                                  and page.evaluate('() => engineSettings.temperature') == 1, saved)

                            # Use the real reply(true) handler, not a reimplementation of requestMessages.
                            prefix = '```rust\nfn answer() {\n    '
                            page.evaluate("""prefix => {
                                cur = {id: 'browser-continuation-check', title: 'Code continuation', at: Date.now(), system: '', msgs: [
                                    {role: 'user', content: 'Write Rust.'},
                                    {role: 'assistant', content: 'Earlier.', think: 'Earlier test reasoning.'},
                                    {role: 'user', content: 'Continue.'},
                                    {role: 'assistant', content: prefix, think: 'Current test reasoning.', finish: 'length'}]};
                                chats.push(cur); save(); show();
                            }""", prefix)
                            offset = len(posts)
                            with page.expect_response(lambda response: has_path(response, '/v1/chat/completions')) as continuation_response:
                                page.evaluate('() => { void reply(true); }')
                            page.wait_for_function('() => !busy')
                            sent = posts[offset:]
                            check('continuation-http-bodies-retain-reasoning-and-code-whitespace', continuation_response.value.status == 200
                                  and [p['path'] for p in sent] == ['/api/context', '/v1/chat/completions']
                                  and all(p['body']['messages'][1].get('reasoning_content') == 'Earlier test reasoning.'
                                          and p['body']['messages'][-1].get('reasoning_content') == 'Current test reasoning.'
                                          and p['body']['messages'][-1]['content'] == prefix for p in sent), sent)
                            continuation = page.evaluate('() => cur.msgs.at(-1).content')
                            check('continued-output-keeps-original-indentation', continuation == prefix + 'AAAA', continuation)

                            page.locator('#settings').click()
                            page.locator('[data-tab="models"]').click()
                            page.evaluate("() => { $('codepreset').value = 'code-long'; $('codepreset').sync(); }")
                            before_options = page.evaluate('() => engineOpts()')
                            offset = len(posts)
                            page.locator('#applycodepreset').click()
                            actual_block = page.locator('#enote').inner_text()
                            check('actual-cpu-preset-is-blocked-without-load-or-cap-change', 'require a GPU server' in actual_block
                                  and page.evaluate('() => engineOpts()') == before_options
                                  and page.evaluate('() => engineSettings.max_tokens') == 64
                                  and not any(p['path'] == '/api/load' for p in posts[offset:]), actual_block)

                            # UI-only capability fixtures reach guards that the real CPU/512-token model cannot pass.
                            # They do not change the server, run a GPU, or claim these contexts fit hardware.
                            page.evaluate('options => showOpts({...options, gpu: true})', actual_options)
                            page.locator('#applycodepreset').click()
                            training_block = page.locator('#enote').inner_text()
                            check('ui-preset-respects-declared-model-context', 'allows at most 512' in training_block
                                  and page.evaluate('() => engineOpts()') == before_options, {
                                      'scope': 'Actual showOpts/button with UI-only gpu capability; real model training limit remains 512.',
                                      'message': training_block})
                            page.evaluate('options => showOpts({...options, gpu: true, ctx_train: 32768})', actual_options)
                            page.locator('#applycodepreset').click()
                            cap_block = page.locator('#enote').inner_text()
                            live_options = request('/api/models')[1]['options']
                            check('ui-preset-cannot-increase-immutable-server-output-cap', 'server cap is 64' in cap_block
                                  and '--max-tokens 8192' in cap_block
                                  and page.evaluate('() => engineOpts()') == before_options
                                  and live_options['max_tokens'] == 64 and live_options['ctx'] == 512 and not live_options['gpu']
                                  and not any(p['path'] == '/api/load' for p in posts[offset:]), {
                                      'scope': 'Actual showOpts/button with UI-only gpu/training capabilities; immutable live cap is unchanged.',
                                      'message': cap_block, 'live_options': live_options})
                            page.evaluate('options => showOpts(options)', actual_options)
                            check('no-uncaught-browser-errors', not browser_errors, browser_errors)
                            records.append({'name': 'captured-chat-and-preview-requests', 'passed': True, 'requests': posts})
                            context.close()
                        finally:
                            browser.close()
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=5)
                    log.seek(0)
                    records.append({'name': 'server-log', 'passed': server.returncode is not None, 'command': argv, 'output': log.read()})
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        report = {'executable_version': version,
                  'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'scope': 'Headless Chromium, isolated browser storage and synthetic CPU loopback server. UI-only capability fixtures are labelled. No Qwen3.8 full-model, Windows or GPU test.',
                  'passed': sum(bool(r['passed']) for r in records), 'failure': failure, 'records': records}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'passed': report['passed'], 'records': len(records), 'failure': failure, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
