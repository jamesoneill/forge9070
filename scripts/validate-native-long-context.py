#!/usr/bin/env python3
"""Validate long-context settings/UI with untrained local fixtures, without a model download.

This checks controls, persistence and admission; it makes no full-model speed or
quality claim. GPU attention and allocation are tested separately.
"""
import argparse, importlib.util, json, struct, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--chromium', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('long_context_ui_helpers', ROOT/'scripts/validate-native-ui.py')
    ui = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ui; spec.loader.exec_module(ui)
    checks = []
    with tempfile.TemporaryDirectory(prefix='nextora-long-context-ui-') as temporary:
        root = Path(temporary)
        fixtures = [('short.gguf', 'qwen3moe'), ('long.gguf', 'qwen3moe'),
                    ('dense.gguf', 'qwen3'), ('standard.gguf', 'qwen35')]
        for name, architecture in fixtures:
            weights = root / name
            ui.api_helpers.fixtures.fixture_copy(ROOT / f'tests/fixtures/tiny-{architecture}.gguf', weights, False)
            ui.api_helpers.add_template(weights)
            if name != 'short.gguf':
                raw = bytearray(weights.read_bytes())
                key = f'{architecture}.context_length'.encode()
                marker = struct.pack('<Q', len(key)) + key + struct.pack('<I', 4)
                assert raw.count(marker) == 1
                struct.pack_into('<I', raw, raw.index(marker) + len(marker), 1048576)
                weights.write_bytes(raw)
        settings=dict(backend='cpu',device=None,ctx=512,max_tokens=128,parallel=1,max_queued=2,
                      queue_timeout_secs=5,vram_mib=0,expert_ram_mib=16,ram_margin_mib=0,
                      expert_warmup=False,threads=2)
        profile=dict(directory=str(root),model='short.gguf',settings=settings)
        saved=root/'service.json';saved.write_text(json.dumps(profile));original=saved.read_bytes()
        with ui.service(args.binary.resolve(),saved) as (origin,request,json_request):
            status,config=json_request('/api/config');assert status==200 and config['settings'].get('kv_precision','f32')=='f32'
            checks.append('legacy profile defaults to standard context memory')
            for overrides, expected in [({'backend':'cpu','kv_precision':'q4_64'},'Vulkan'),({'backend':'vulkan','kv_precision':'q4_64','parallel':2},'one active request')]:
                status,error=json_request('/api/config',dict(profile,settings=dict(settings,**overrides)))
                assert status==400 and expected.lower() in json.dumps(error).lower(),(status,error)
                assert saved.read_bytes()==original
                assert json_request('/api/config')[1]['model']=='short.gguf'
            checks.append('unsupported compact backend and parallel settings rejected without changing profile')
            from playwright.sync_api import sync_playwright, expect
            with sync_playwright() as p:
                browser=p.chromium.launch(executable_path=str(args.chromium),headless=True,args=['--no-sandbox'])
                page=browser.new_page(viewport={'width':1440,'height':1000})
                page.goto(origin,wait_until='domcontentloaded');page.locator('#api-key').fill(ui.KEY);page.locator('#unlock').click()
                expect(page.locator('#backend')).to_have_value('cpu');expect(page.locator('#scan-models')).to_be_enabled()
                assert page.locator('#context-512k').is_disabled();checks.append('512K unavailable for short-context checkpoint')
                page.locator('#scan-models').click();page.locator('input[name="model"][value="long.gguf"]').wait_for();page.locator('input[name="model"][value="long.gguf"]').check()
                expect(page.locator('#context-512k')).to_be_enabled();page.locator('#context-512k').click()
                assert page.locator('#ctx').input_value()=='524288'
                assert page.locator('#kv-precision').input_value()=='q4_64'
                assert page.locator('#backend').input_value()=='vulkan'
                assert page.locator('#parallel').input_value()=='1'
                assert int(page.locator('#vram-mib').input_value())>0
                assert json_request('/api/config')[1]['model']=='short.gguf' and saved.read_bytes()==original
                checks.append('512K preset prepares compact GPU single-request form without applying or switching model')
                with page.expect_download() as download:page.locator('#export-settings').click()
                exported=json.loads(Path(download.value.path()).read_text())
                assert exported['settings']['kv_precision']=='q4_64' and exported['settings']['ctx']==524288
                checks.append('exported preset retains compact precision and requested context')
                for name, eligible in [('standard.gguf', False), ('dense.gguf', True)]:
                    page.locator(f'input[name="model"][value="{name}"]').check()
                    assert page.locator('#context-512k').is_enabled() == eligible
                    expect(page.locator('#large-context')).to_be_enabled()
                checks.append('compact preset allows long-context dense Qwen3; Qwen3.5 offers its ordinary context setting')
                families = [('short', 'qwen3moe', 40960), ('long', 'qwen3moe', 1048576),
                            ('dense', 'qwen3', 524288), ('standard', 'qwen35', 1048576),
                            ('other', 'qwen4exp', 1048576)]
                models = [dict(repo='test/' + name, revision='abc', file=name + '.gguf', architecture=arch,
                               context_length=length, bytes=100, shards=1) for name, arch, length in families]
                page.route('**/api/hub/search',lambda route:route.fulfill(status=200,content_type='application/json',body=json.dumps(dict(models=models,warnings=[],directory=str(root)))))
                page.locator('#hub-long-context').check();page.locator('#hub-query').fill('test');page.locator('#hub-search').click()
                results = page.locator('.hub-result strong')
                expect(results).to_have_text(['long.gguf', 'dense.gguf'])
                checks.append('512K Hub filter only returns long-context Qwen3 MoE and dense Qwen3')
                page.screenshot(path=str(args.output.with_suffix('.png')),full_page=True)
                browser.close()
    report=dict(passed=len(checks),checks=checks,scope='untrained synthetic API/UI controls; no performance or pretrained-quality claim')
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
