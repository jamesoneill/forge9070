'use strict';

function supportsCompact512k(architecture, contextLength) {
  return (architecture === 'qwen3' || architecture === 'qwen3moe') && contextLength >= 524288;
}

function createNextoraLibrary(state, api, loadModel, prepareModel) {
  const el = id => document.getElementById(id);
  let searching = false, starting = false, polling = false, timer = null, job = null;
  const active = () => ['queued', 'downloading', 'validating'].includes(job?.phase);
  const size = n => n >= 1024 ** 3 ? `${(n / 1024 ** 3).toFixed(2)} GiB` : `${(n / 1024 ** 2).toFixed(1)} MiB`;
  const message = text => { el('hub-message').textContent = text; };
  function controls() {
    el('hub-search').disabled = !state.connected || searching;
    for (const button of el('hub-results').querySelectorAll('button')) button.disabled = !state.connected || starting || active();
    el('hub-prepare').disabled = el('hub-configure').disabled = el('hub-load').disabled = !state.connected || state.applying || state.scanning || state.tuningActive || starting;
    el('hub-cancel').disabled = !state.connected || starting;
    if (state.connected && timer === null) {
      timer = setInterval(() => { if (!document.hidden && state.connected) void poll(); }, 2000);
      void poll();
    }
  }
  function bar(id, done, total) {
    const progress = el(id);
    if (Number.isFinite(done) && total > 0) { progress.max = total; progress.value = Math.min(total, done); }
    else progress.removeAttribute('value');
  }
  function render(value) {
    job = value;
    el('hub-job').hidden = !job || job.phase === 'idle';
    if (el('hub-job').hidden) { controls(); return; }
    el('hub-file').textContent = job.current_file || job.file || '';
    const progress = job.phase === 'downloading' && job.total_bytes ? ` · ${size(job.received_bytes || 0)} of ${size(job.total_bytes)}` : '';
    const files = job.total_files > 0 ? ` · ${job.completed_files}/${job.total_files} files` : '';
    const phase = {queued:'Queued',downloading:'Downloading',validating:'Checking the local files',ready:'Download checked · ready for setup',failed:'Download failed',cancelled:'Download cancelled'}[job.phase] || job.phase;
    el('hub-detail').textContent = `${phase}${progress}${files} · ${job.directory || ''}${job.error ? ' · ' + job.error : ''}`;
    el('hub-cancel').hidden = !active(); el('hub-prepare').hidden = el('hub-configure').hidden = el('hub-load').hidden = job.phase !== 'ready' || !job.model;
    el('hub-bar').hidden = ['failed','cancelled'].includes(job.phase);
    if (job.phase === 'ready') bar('hub-bar',1,1); else if (job.phase === 'downloading') bar('hub-bar',job.received_bytes,job.total_bytes); else bar('hub-bar');
    controls();
  }
  let version = 0;
  async function poll() {
    if (polling || starting) return;
    const expected = version; polling = true;
    try { const next = await api('/api/hub/progress', {timeout:8000}); if (version === expected) render(next); }
    catch (error) { if (active()) el('hub-detail').textContent = 'Waiting for download status · ' + error.message; }
    finally { polling = false; }
  }
  async function command(path, body) {
    if (starting) return;
    starting = true; version++; controls();
    try { render(await api(path, {body})); }
    catch (error) { message(error.message); }
    finally { starting = false; controls(); }
  }
  el('hub-cancel').addEventListener('click', () => command('/api/hub/cancel', {}));
  el('hub-prepare').addEventListener('click', () => { if (job?.phase === 'ready') void prepareModel(job); });
  el('hub-configure').addEventListener('click', () => { if (job?.phase === 'ready') void loadModel(job, false); });
  el('hub-load').addEventListener('click', () => { if (job?.phase === 'ready') void loadModel(job, true); });
  el('hub-search-form').addEventListener('submit', async event => {
    event.preventDefault(); if (searching || !state.connected) return;
    const query = el('hub-query').value.trim(); if (!query) return;
    searching = true; controls(); el('hub-results').replaceChildren(); message('Searching repositories and checking GGUF headers…');
    try {
      const result = await api('/api/hub/search', {body:{query}});
      const models = el('hub-long-context').checked ? result.models.filter(model => supportsCompact512k(model.architecture, model.context_length)) : result.models;
      for (const model of models) {
        const row = document.createElement('div'), details = document.createElement('div'), name = document.createElement('strong'), meta = document.createElement('small'), download = document.createElement('button');
        row.className = 'hub-result'; name.textContent = model.file;
        meta.textContent = [model.repo, model.architecture, (Array.isArray(model.native_backends) ? model.native_backends : model.architecture === 'qwen35' ? ['cpu','hybrid'] : []).map(name => ({cpu:'CPU',vulkan:'GPU',hybrid:'hybrid'})[name] || name).join(' / '), model.context_length ? `${model.context_length.toLocaleString()} context` : '', model.bytes ? size(model.bytes) : '', `${model.shards} file(s)`].filter(Boolean).join(' · ');
        download.type = 'button'; download.className = 'button button-secondary'; download.textContent = 'Download';
        download.addEventListener('click', () => command('/api/hub/download', {repo:model.repo, revision:model.revision, file:model.file}));
        details.append(name,meta); row.append(details,download); el('hub-results').append(row);
      }
      message(models.length ? `${models.length} verified GGUF choices · destination ${result.directory}` : 'No matching verified GGUF choices found. For 512K, search for a compatible long-context checkpoint.');
      const notes = Array.isArray(result.warnings) ? result.warnings : [];
      if (notes.length) {
        const details = document.createElement('details'), summary = document.createElement('summary');
        summary.textContent = 'Skipped files and verification details'; details.append(summary);
        for (const note of notes) { const text = document.createElement('p'); text.textContent = note; details.append(text); }
        el('hub-results').append(details);
      }
    } catch (error) { message(error.message); }
    finally { searching = false; controls(); }
  });
  function loadProgress(p) {
    el('load-progress-panel').hidden = !p || p.stage === 'idle'; if (el('load-progress-panel').hidden) return;
    const pct = p.total > 0 && Number.isFinite(p.completed) ? Math.min(100, 100 * p.completed / p.total) : null;
    const count = pct === null ? '' : p.unit === 'bytes' ? `${size(p.completed)} / ${size(p.total)}` : `${p.completed.toLocaleString()} / ${p.total.toLocaleString()} ${p.unit || ''}`;
    el('load-caption').textContent = p.message || p.stage;
    el('load-detail').textContent = [count, p.experts_ready == null ? '' : `${p.experts_ready.toLocaleString()} cached`, p.bytes_loaded == null ? '' : `${size(p.bytes_loaded)} read`, pct === null ? '' : `${pct.toFixed(0)}%`, p.elapsed_seconds == null ? '' : `${Number(p.elapsed_seconds).toFixed(1)} s`].filter(Boolean).join(' · ');
    el('load-bar').hidden = p.stage === 'failed';
    if (p.stage === 'ready') bar('load-bar',1,1); else bar('load-bar',p.completed,p.total);
  }
  return { controls, loadProgress };
}
