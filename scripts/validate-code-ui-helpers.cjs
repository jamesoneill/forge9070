#!/usr/bin/env node
/* Focused unit checks of the actual UI helper source, without a DOM or browser.
 * Uses only Node builtins. Element/store stubs intentionally do not implement
 * layout, rendering, browser input sanitization, events, storage, or networking.
 */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');

const root = path.resolve(__dirname, '..');
let sourcePath = path.join(root, 'vendor/rusting-strata/src/web/index.html');
let outputPath = path.join(root, 'artifacts/code-ui-helpers-0.7.0.json');
for (let i = 2; i < process.argv.length; i += 2) {
  const flag = process.argv[i], value = process.argv[i + 1];
  assert(value && ['--source', '--output'].includes(flag), 'Usage: node scripts/validate-code-ui-helpers.cjs [--source HTML] [--output JSON]');
  if (flag === '--source') sourcePath = path.resolve(value);
  else outputPath = path.resolve(value);
}

const html = fs.readFileSync(sourcePath, 'utf8');
const scriptStart = html.indexOf('<script>'), scriptEnd = html.lastIndexOf('</script>');
assert(scriptStart >= 0 && scriptEnd > scriptStart, 'Standalone UI script not found');
const script = html.slice(scriptStart + '<script>'.length, scriptEnd);
const sourceHash = crypto.createHash('sha256').update(html).digest('hex');

// Extract complete, named blocks from the shipped page. No helper behavior is
// duplicated in the harness; a boundary change fails explicitly for review.
function block(start, end) {
  const begin = script.indexOf(start);
  assert(begin >= 0 && script.indexOf(start, begin + start.length) === -1, 'Missing or ambiguous source block: ' + start);
  const finish = script.indexOf(end, begin + start.length);
  assert(finish > begin, 'Missing block end: ' + end);
  return script.slice(begin, finish);
}
const blocks = {
  numberLabel: block('const num = ', '\nconst secs = '),
  setControl: block('const SHOW = ', '\nfor (const id of [\'system\''),
  requestMessages: block('function requestMessages(msgs) {', '\nfunction requestBody('),
  engineFields: block('const EO = ', '\ndrop($(\'ekv\'))'),
  engineOpts: block('function engineOpts() {', '\nfunction showOpts('),
  showOpts: block('function showOpts(o) {', '\n$(\'applycodepreset\').onclick'),
  presetHandler: block('$(\'applycodepreset\').onclick = ', '\n$(\'eparallel\').addEventListener'),
  admissionView: block('function admissionView(a) {', '\nconst layers = '),
};
const actualHelpers = new vm.Script(Object.values(blocks).join('\n'), {filename: sourcePath + ':extracted-helpers'});

function harness() {
  const elements = new Map(), values = new Map();
  function element(id) {
    if (!elements.has(id)) {
      const node = {id, type: ['temp', 'topp', 'topk', 'ectx', 'emtp'].includes(id) ? 'range' : 'number',
        dataset: {}, style: {}, checked: false, disabled: false, hidden: false, textContent: '',
        sync() { this.syncCalls = (this.syncCalls || 0) + 1; }};
      for (const property of ['value', 'min', 'max', 'placeholder']) {
        let value = '';
        Object.defineProperty(node, property, {get: () => value, set: v => {value = String(v);}, enumerable: true});
      }
      elements.set(id, node);
    }
    return elements.get(id);
  }
  const store = {get: (key, fallback) => values.has(key) ? values.get(key) : fallback,
    set: (key, value) => {values.set(key, String(value)); return true;}};
  const context = vm.createContext({$: element, store});
  actualHelpers.runInContext(context, {timeout: 1000});
  function evaluate(expression, input) {
    context.__input = input;
    return vm.runInContext(expression, context, {timeout: 1000});
  }
  function plain(expression, input) {return JSON.parse(JSON.stringify(evaluate(expression, input)));}
  function options(options) {evaluate('showOpts(__input)', options);}
  function stage(preset) {element('codepreset').value = preset; evaluate("$('applycodepreset').onclick()");}
  function controls() {
    return {output: Number(element('maxtok').value), temperature: Number(element('temp').value),
      top_p: Number(element('topp').value), top_k: Number(element('topk').value), think: element('think').checked,
      ctx: Number(element('ectx').value), ctx_min: Number(element('ectx').min), ctx_max: Number(element('ectx').max),
      output_max: Number(element('maxtok').max)};
  }
  return {element, store, values, evaluate, plain, options, stage, controls};
}

const base = {ctx: 32768, ctx_train: 262144, context_min: 128, context_max: 131072,
  max_tokens: 8192, default_max_tokens: 8192, temperature: 1, top_p: .95, top_k: 20,
  think: true, preserve_thinking: true, kv: 'f16', cache_ram: 1024, ram_margin: 6144,
  vram_margin: '2048', chunk: 256, mtp: 0, parallel: 1, parallel_mode: 'auto', max_parallel: 4,
  concurrency_reason: 'This model supports one active sequence; additional requests queue.', gpu: true, calibrated: false,
  expert_tail_min_rank: 0};
const records = [];
let failure = null;
function check(name, run) {
  try {records.push({name, passed: true, detail: run()});}
  catch (error) {records.push({name, passed: false, error: String(error)}); throw error;}
}

try {
  check('whole-shipped-ui-script-parses', () => {
    new vm.Script(script, {filename: sourcePath + ':complete-script'});
    return {scope: 'Syntax compilation only; the full script was not executed in a browser.'};
  });

  check('requestMessages-keeps-earlier-reasoning-and-open-code-whitespace', () => {
    const h = harness(), prefix = '```rust\nfn answer() {\n    \t';
    const input = [
      {role: 'user', content: 'Write Rust.'},
      {role: 'assistant', content: 'Earlier.', think: 'Earlier fixture reasoning.'},
      {role: 'user', content: 'Continue.'},
      {role: 'assistant', content: prefix, think: 'Current fixture reasoning.'},
      {role: 'assistant', content: 'Previous request error', error: true},
    ];
    const before = JSON.stringify(input);
    const messages = h.plain('requestMessages(__input)', input);
    assert.equal(messages.length, 4);
    assert.equal(messages[1].reasoning_content, 'Earlier fixture reasoning.');
    assert.equal(messages[3].reasoning_content, 'Current fixture reasoning.');
    assert.equal(messages[3].content, prefix);
    assert.equal(JSON.stringify(input), before, 'Request construction must not mutate stored messages');
    return messages;
  });

  check('showOpts-inherits-server-defaults-without-writing-user-overrides', () => {
    const h = harness();
    h.options({...base});
    assert.deepEqual(h.controls(), {output: 8192, temperature: 1, top_p: .95, top_k: 20, think: true,
      ctx: 32768, ctx_min: 128, ctx_max: 131072, output_max: 8192});
    assert.equal(h.values.size, 0, 'Inherited defaults must remain distinguishable from explicit user changes');
    return h.controls();
  });

  check('showOpts-preserves-stored-sampling-and-pending-context-edits', () => {
    const h = harness();
    for (const [id, value] of Object.entries({temp: .25, topp: .8, topk: 7, maxtok: 12})) {
      h.element(id).value = value; h.store.set(id, value);
    }
    h.element('think').checked = false; h.store.set('think', '0');
    h.element('ectx').value = 16384; h.element('ectx').dataset.dirty = 1;
    h.options({...base});
    assert.deepEqual(h.controls(), {output: 12, temperature: .25, top_p: .8, top_k: 7, think: false,
      ctx: 16384, ctx_min: 128, ctx_max: 131072, output_max: 8192});
    assert.equal(h.evaluate('engineSettings.temperature'), 1, 'UI user override must not rewrite the server setting');
    return h.controls();
  });

  check('showOpts-sets-model-context-and-output-cap-bounds', () => {
    const h = harness();
    h.options({...base, ctx: 4096, ctx_train: 8192, max_tokens: 16384, default_max_tokens: 4095});
    assert.equal(h.controls().ctx_max, 8192);
    assert.equal(h.controls().output_max, 4095, 'Output max must leave at least one context token for the prompt');
    const belowForgeCap = h.controls();
    h.options({...base, ctx: 65536, ctx_train: 262144});
    assert.equal(h.controls().ctx_max, 131072, 'A larger training context must not raise Forge\'s maximum');
    assert.equal(h.controls().output_max, 8192);
    return {belowForgeCap, aboveForgeCap: h.controls(), limitation: 'Checks assigned bounds, not browser range-value sanitization.'};
  });

  check('code-preset-refuses-immutable-output-cap-without-staging-changes', () => {
    const h = harness(), settings = {...base, max_tokens: 64, default_max_tokens: 64};
    h.options(settings);
    const before = h.plain('engineOpts()');
    h.stage('code-long');
    const message = h.element('enote').textContent;
    assert(message.includes('server cap is 64') && message.includes('--max-tokens 8192'), message);
    assert.deepEqual(h.plain('engineOpts()'), before);
    assert.equal(h.controls().output, 64);
    assert.equal(settings.max_tokens, 64);
    assert.equal(h.values.size, 0);
    return {message, staged: h.plain('engineOpts()'), serverCap: settings.max_tokens};
  });

  check('code-preset-refuses-unavailable-gpu-and-short-model-context', () => {
    const h = harness();
    h.options({...base, gpu: false});
    h.stage('code');
    const cpu = h.element('enote').textContent;
    assert(cpu.includes('require a GPU server'), cpu);
    assert.deepEqual(h.plain('engineOpts()'), {});
    h.options({...base, ctx: 512, ctx_train: 512, default_max_tokens: 511});
    h.stage('code-long');
    const context = h.element('enote').textContent;
    assert(context.includes('allows at most 512'), context);
    assert.deepEqual(h.plain('engineOpts()'), {});
    return {cpu, context};
  });

  check('code-presets-stage-declared-options-only-on-explicit-action', () => {
    const outcomes = [];
    for (const [preset, ctx, output] of [['code', 16384, 4096], ['code-long', 32768, 8192]]) {
      const h = harness();
      h.options({...base});
      assert.deepEqual(h.plain('engineOpts()'), {}, 'Reading server options must not stage a preset');
      h.stage(preset);
      const staged = h.plain('engineOpts()');
      assert.deepEqual(staged, {ctx, mtp: 0, kv: 'f16', cache_ram: 1024, chunk: 256, parallel: 1, ram_margin: 6144});
      assert.equal(h.controls().output, output);
      assert.equal(h.store.get('maxtok', ''), String(output));
      assert.equal(h.controls().temperature, 1, 'Staging context options must not overwrite sampling');
      outcomes.push({preset, staged, output});
    }
    return outcomes;
  });

  check('auto-selection-retains-effective-one-slot-model-capacity', () => {
    const h = harness();
    h.options({...base});
    assert.equal(h.element('eparallel').value, 'auto');
    assert.equal(h.element('emaxparallel').value, '4');
    assert.equal(h.element('emaxparallel').disabled, false);
    assert.equal(h.element('emtp').disabled, false, 'Auto may resolve to one slot for MTP');
    assert(h.element('capacitynote').textContent.includes('Current capacity: 1 active request.'));
    assert(h.element('capacitynote').textContent.includes(base.concurrency_reason));
    assert.deepEqual(h.plain('engineOpts()'), {}, 'Reading an auto policy must not stage the effective integer as a fixed override');
    return {mode: h.element('eparallel').value, ceiling: h.element('emaxparallel').value,
      note: h.element('capacitynote').textContent};
  });

  check('load-options-encode-auto-as-string-and-fixed-as-number', () => {
    const h = harness();
    h.options({...base, parallel_mode: '1'});
    h.element('eparallel').value = 'auto'; h.element('eparallel').dataset.dirty = 1;
    assert.deepEqual(h.plain('engineOpts()'), {parallel: 'auto'});
    h.element('emaxparallel').value = '2'; h.element('emaxparallel').dataset.dirty = 1;
    assert.deepEqual(h.plain('engineOpts()'), {parallel: 'auto', max_parallel: 2});
    h.element('eparallel').value = '3';
    assert.deepEqual(h.plain('engineOpts()'), {parallel: 3, max_parallel: 2});
    return {auto: 'auto', fixed: 3, max_parallel: 2};
  });

  check('pending-capacity-options-survive-refresh-and-backend-controls-remain-honest', () => {
    const h = harness();
    h.options({...base});
    h.element('eparallel').value = '2'; h.element('eparallel').dataset.dirty = 1;
    h.element('emaxparallel').value = '3'; h.element('emaxparallel').dataset.dirty = 1;
    h.options({...base, parallel: 1, max_parallel: 4});
    assert.equal(h.element('eparallel').value, '2');
    assert.equal(h.element('emaxparallel').value, '3');
    assert.equal(h.element('emaxparallel').disabled, true, 'The ceiling affects auto only');
    assert.equal(h.element('emtp').disabled, true);
    assert(h.element('capacitynote').textContent.includes('Current capacity: 1 active request.'), 'A staged choice is not loaded capacity');
    const pending = h.plain('engineOpts()');
    h.options({...base, gpu: false});
    assert.equal(h.element('eparallel').disabled, true);
    assert.equal(h.element('emaxparallel').disabled, true);
    assert.equal(h.element('emtp').disabled, true);
    assert.deepEqual(h.plain('engineOpts()'), pending);
    return {pending, disabledForCpu: true};
  });

  check('request-counters-separate-admitted-reserved-and-queued-work', () => {
    const h = harness();
    const view = h.plain('admissionView(__input)', {capacity: 4, active: 2, reserved: 1, queued: 7,
      queue_capacity: 32, queue_timeout_seconds: 120, rejected_full_total: 3, timed_out_total: 2, cancelled_total: 5});
    const rows = Object.fromEntries(view.rows);
    assert.equal(view.status, '2 admitted · 7 queued');
    assert.equal(rows['Generation capacity'], '4 slots');
    assert.equal(rows['Admitted requests'], '2');
    assert.equal(rows['Reserved slots'], '1');
    assert.equal(rows['Queued requests'], '7 / 32');
    assert.equal(rows['Queue wait limit'], '120 seconds');
    assert.equal(rows['Queue-full rejections'], '3');
    assert.equal(rows['Queue timeouts'], '2');
    assert.equal(rows['Cancelled before starting'], '5');
    assert(view.note.includes('preparation, prompt processing and replies'));
    return view;
  });

  check('request-counters-handle-offline-and-immediate-admission-only-mode', () => {
    const h = harness();
    const unavailable = h.plain('admissionView(__input)', null);
    assert.equal(unavailable.status, 'Unavailable');
    assert.deepEqual(unavailable.rows, []);
    const zeroQueue = h.plain('admissionView(__input)', {capacity: 1, active: 1, reserved: 0, queued: 0,
      queue_capacity: 0, queue_timeout_seconds: 120, rejected_full_total: 0, timed_out_total: 0, cancelled_total: 0});
    assert.equal(Object.fromEntries(zeroQueue.rows)['Queued requests'], '0 / 0');
    assert(zeroQueue.note.includes('Waiting is disabled; requests need a free slot.'));
    const malformed = h.plain('admissionView(__input)', {capacity: '4', active: -1, queued: 0});
    assert.equal(malformed.status, '– admitted · 0 queued');
    assert.equal(Object.fromEntries(malformed.rows)['Generation capacity'], '– slots');
    return {unavailable, zeroQueue, malformed};
  });
} catch (error) {
  failure = String(error.stack || error);
  process.exitCode = 1;
} finally {
  const report = {checked_at_utc: new Date().toISOString(), node_version: process.version,
    source: sourcePath, source_sha256: sourceHash, extracted_blocks: Object.keys(blocks),
    scope: 'Actual UI JavaScript helper unit validation using Node VM and minimal element/store stubs. No DOM, browser rendering, layout, events, HTTP, GPU, or model-quality validation.',
    passed: records.filter(r => r.passed).length, failure, records};
  fs.mkdirSync(path.dirname(outputPath), {recursive: true});
  fs.writeFileSync(outputPath, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({passed: report.passed, records: records.length, failure, output: outputPath}));
}
