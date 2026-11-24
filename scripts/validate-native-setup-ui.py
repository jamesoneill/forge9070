#!/usr/bin/env python3
"""Exercise selecting an unloaded model, preparing its profile, and explicitly using it."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--chromium', required=True, type=Path)
    parser.add_argument('--screenshot', type=Path)
    args = parser.parse_args()
    helpers = module('setup_browser_service', ROOT / 'scripts/validate-native-ui.py')
    setup = module('setup_browser_fixture', ROOT / 'scripts/validate-native-setup.py')
    with tempfile.TemporaryDirectory(prefix='nextora-setup-browser-') as temporary:
        root = Path(temporary)
        old = root / 'old.gguf'
        target = root / 'target.gguf'
        setup.fixture(old)
        helpers.api_helpers.fixtures.fixture_copy(ROOT / 'tests/fixtures/tiny-qwen35-mtp.gguf', target, False)
        helpers.api_helpers.add_template(target)
        settings = dict(backend='cpu', device=None, ctx=512, max_tokens=32, parallel=1,
                        max_queued=2, queue_timeout_secs=5, vram_mib=0, expert_ram_mib=16,
                        ram_margin_mib=0, expert_warmup=True, threads=2)
        saved = root / 'service.json'
        saved.write_text(json.dumps(dict(directory=str(root), model=old.name, settings=settings)))
        with helpers.service(args.binary.resolve(), saved) as (origin, request, json_request):
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(executable_path=str(args.chromium), headless=True)
                try:
                    context = browser.new_context(viewport={'width': 1440, 'height': 1080}, reduced_motion='reduce')
                    context.route('**/*', lambda route: route.continue_()
                                  if route.request.url.startswith(origin + '/') else route.abort())
                    page = context.new_page()
                    page.set_default_timeout(90000)
                    errors = []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.goto(origin + '/', wait_until='domcontentloaded')
                    page.locator('#api-key').fill(helpers.KEY)
                    page.locator('#unlock').click()
                    expect(page.locator('#auth-panel')).to_be_hidden()
                    expect(page.locator('#loaded-model-name')).to_have_text(old.name)
                    page.locator('#model-directory').fill(str(root))
                    page.locator('#scan-models').click()
                    page.locator('input[name="model"][value="target.gguf"]').check()
                    expect(page.locator('#selected-model-name')).to_have_text(target.name)
                    page.locator('#tuning-context-goal').select_option('maximum_fit')
                    page.locator('#tuning-context-ceiling').fill('384')
                    expect(page.locator('#tuning-prepare')).to_be_enabled()
                    page.locator('#tuning-prepare').click()
                    expect(page.locator('#tuning-use')).to_be_visible()
                    expect(page.locator('#tuning-use')).to_be_enabled()
                    expect(page.locator('#tuning-status')).to_contain_text('Profile ready')
                    expect(page.locator('#loaded-model-name')).to_have_text(old.name)
                    code, current = json_request('/v1/status')
                    assert code == 200 and current['model'] == old.name
                    assert Path(current['runtime']['shards'][0]).samefile(old)
                    expect(page.locator('#tuning-prepare-rerun')).to_be_visible()
                    page.locator('#tuning-use').click()
                    expect(page.locator('#loaded-model-name')).to_have_text(target.name)
                    expect(page.locator('#ctx')).to_have_value('384')
                    code, current = json_request('/v1/status')
                    assert code == 200 and Path(current['runtime']['shards'][0]).samefile(target)
                    assert current['runtime']['config']['architecture'] == 'qwen35'
                    assert current['runtime']['config']['expert_count'] == 0
                    page.locator('#chat-prompt').fill('hello')
                    page.locator('#chat-limit').fill('4')
                    page.locator('#chat-send').click()
                    expect(page.locator('.chat-measurements')).to_contain_text('tok/s decode')
                    page.locator('#insight-refresh').click()
                    expect(page.locator('#insight-trend .insight-trend-bar')).to_have_count(1)
                    code, insight = json_request('/api/insight/live')
                    assert code == 200 and insight['measurements']['completed_requests'] == 1, insight
                    assert not errors, errors
                    if args.screenshot:
                        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                        page.screenshot(path=str(args.screenshot), full_page=True)
                    print('Setup UI passed: selected dense Qwen3.5+MTP target, Prepare action, saved-profile readiness, old-model restoration, explicit Use, applied context and no uncaught browser errors.')
                finally:
                    browser.close()


if __name__ == '__main__':
    main()
