'use strict';

function createNextoraTuning(state, api, onComplete) {
  const el = id => document.getElementById(id);
  const activePhases = new Set(['starting', 'fitting', 'running', 'saving', 'restoring', 'applying']);
  let job = null;
  let pending = false;
  let polling = false;
  let timer = null;
  let observedModel = '';
  let requestVersion = 0;
  let updating = false;
  let selectedTarget = null;
  let observedSelection = '';
  const targetKey = target => target ? `${target.directory}\n${target.model}` : '';
  function target() {
    if (selectedTarget) return selectedTarget;
    if (state.selection?.id && state.selection?.root) return { directory:state.selection.root, model:state.selection.id };
    return state.status?.model && state.config?.models_dir ? { directory:state.config.models_dir, model:state.status.model } : null;
  }
  async function selectTarget(next) {
    if (!next || active()) return;
    selectedTarget = { directory:next.directory, model:next.model };
    const expected = targetKey(selectedTarget);
    requestVersion++;
    render();
    try {
      const record = await api('/api/tuning/profile', { body:selectedTarget, timeout:15000 });
      if (!active() && targetKey(selectedTarget) === expected) { job=record; render(); }
    } catch (error) { el('tuning-status').textContent=error.message; }
  }

  function active() {
    return pending || activePhases.has(job?.phase);
  }

  function currentModel() {
    return state.status?.model || (active() ? job?.model : null);
  }

  function context() {
    return state.status?.runtime?.config?.context_length || state.config?.settings?.ctx;
  }

  function rate(value) {
    return Number.isFinite(value) && value > 0 ? `${value.toFixed(2)} tok/s` : '—';
  }

  function controls() {
    state.tuningActive = active();
    const admission = state.status?.admission || {};
    const chatting = !el('chat-stop').hidden || !el('speed-stop').hidden;
    const unavailable = !state.connected || !state.status?.model || state.applying || state.scanning || state.dirty ||
      admission.active > 0 || admission.queued > 0 || admission.model_changing || chatting || active();
    const saved = !job?.preparing && job?.phase === 'complete' && !!job.result?.winner;
    const busy = !state.connected || state.applying || state.scanning || admission.active > 0 || admission.queued > 0 || admission.model_changing || chatting || active();
    el('tuning-prepare').disabled = busy || !target();
    el('tuning-use').hidden = !job?.preparing || job?.phase !== 'complete' || !job?.result?.winner;
    el('tuning-use').disabled = busy;
    el('tuning-prepare-rerun').hidden = !job?.preparing || job?.phase !== 'complete' || !job?.result?.winner;
    el('tuning-prepare-rerun').disabled = busy;
    el('tuning-context-goal').disabled = active();
    el('tuning-context-ceiling').disabled = active() || el('tuning-context-goal').value === 'fixed';
    el('tuning-start').textContent = saved ? 'Apply saved settings' : 'Run tuning';
    el('tuning-start').disabled = unavailable;
    el('tuning-rerun').hidden = !saved;
    el('tuning-rerun').disabled = unavailable;
    el('tuning-cancel').hidden = !activePhases.has(job?.phase);
    el('tuning-cancel').disabled = !state.connected || pending || ['saving', 'restoring', 'applying'].includes(job?.phase);
  }

  function renderCandidates(result) {
    const rows = el('tuning-candidates');
    rows.replaceChildren();
    el('tuning-results').hidden = !result?.candidates?.length;
    if (!result?.candidates?.length) return;
    for (const candidate of result.candidates) {
      const row = document.createElement('tr');
      const setting = document.createElement('th');
      setting.scope = 'row';
      setting.textContent = [candidate.backend, candidate.settings?.direct_experts === true ? (candidate.settings?.fast_math ? 'Direct · Q8 activation kernels' : 'Direct · experimental setup') : candidate.settings?.direct_experts === false ? 'packed expert copies' : '', `${candidate.threads} CPU workers`, candidate.backend === 'vulkan' ? candidate.settings?.kv_placement === 'ram' ? 'context in RAM' : 'context in graphics memory' : ''].filter(Boolean).join(' · ');
      const generation = document.createElement('td');
      generation.textContent = rate(candidate.decode_tps);
      const prompt = document.createElement('td');
      prompt.textContent = rate(candidate.prefill_tps);
      const outcome = document.createElement('td');
      const winner = result.winner;
      const chosen = winner && !candidate.error && candidate.backend === winner.backend &&
        candidate.threads === winner.threads && JSON.stringify(candidate.settings) === JSON.stringify(winner.settings);
      outcome.textContent = candidate.error || (chosen ? 'Best measured' : 'Measured');
      if (chosen) row.dataset.winner = 'true';
      row.append(setting, generation, prompt, outcome);
      rows.append(row);
    }
  }

  function render() {
    const model = currentModel();
    const name = (job?.preparing ? job.model : null) || target()?.model || job?.model || model;
    el('tuning-model').textContent = name
      ? `${name.split(/[\\/]/).pop()} · ${(job?.context || context() || 0).toLocaleString()} context`
      : 'Select a downloaded model to prepare it for this machine.';
    const progress = job?.progress;
    const phase = job?.phase || 'idle';
    let message = 'Run tuning when the service is idle. Saved results are reused for this model, machine, and context.';
    if (phase === 'starting') message = 'Releasing the model so the separate tuning process can use the machine.';
    else if (phase === 'fitting') message = progress?.message || 'Checking supported contexts and memory use.';
    else if (phase === 'saving') message = 'Saving the reusable profile for this model.';
    else if (phase === 'running') message = progress?.message || 'Comparing runtime settings in a separate process.';
    else if (phase === 'restoring') message = 'Restoring the previous model and settings.';
    else if (phase === 'applying') message = 'Loading the model with the fastest measured settings.';
    else if (phase === 'complete') message = job.preparing ? (job.activated ? 'Model loaded with its saved profile.' : 'Profile ready. Use it to load this model; the previous service has been restored.') : job.cached ? 'Saved tuning result for this model and context.' : 'Tuning complete. The fastest measured settings are saved and applied.';
    else if (phase === 'cancelled') message = 'Tuning cancelled. The previous model and settings have been restored.';
    else if (phase === 'failed') message = 'Tuning could not finish.';
    if (phase === 'idle') message = target() ? 'Prepare this model once to check memory fit, benchmark speed, and save a profile. Manual loading remains available.' : 'Select or download a supported model to begin setup.';
    if (job?.error) message += ` ${job.error}`;
    el('tuning-status').textContent = message;
    el('tuning-status').dataset.tone = phase === 'failed' ? 'error' : 'status';
    const bar = el('tuning-progress');
    bar.hidden = !activePhases.has(phase);
    const count = progress?.completed, total = progress?.total;
    if (Number.isFinite(count) && total > 0) {
      bar.max = total;
      bar.value = Math.min(count, total);
    } else bar.removeAttribute('value');
    const detail = [];
    if (progress?.candidate && progress.candidate_total) detail.push(`Configuration ${progress.candidate} of ${progress.candidate_total}`);
    if (Number.isFinite(count) && total > 0) detail.push(`${count.toLocaleString()} / ${total.toLocaleString()} ${progress.unit || ''}`.trim());
    el('tuning-detail').textContent = activePhases.has(phase) ? detail.join(' · ') : '';
    const winner = job?.result?.winner;
    el('tuning-winner').hidden = !winner;
    el('tuning-winner').textContent = winner
      ? `${rate(winner.decode_tps)} generation · ${winner.backend} · ${winner.threads} CPU workers · ${(job.result.context || job.context || 0).toLocaleString()} context`
      : '';
    el('tuning-scope').textContent = job?.result?.scope || '';
    if (job?.result?.memory_fit_context) el('tuning-winner').textContent += ` · memory-fit ceiling ${job.result.memory_fit_context.toLocaleString()} tokens`;
    renderCandidates(job?.result);
    controls();
  }

  async function accept(next) {
    const wasActive = activePhases.has(job?.phase);
    job = next;
    if (next.preparing && next.directory && next.model) selectedTarget={directory:next.directory,model:next.model};
    render();
    if (wasActive && !activePhases.has(next.phase)) await onComplete(!next.preparing || !!next.activated);
  }

  async function poll() {
    if (polling || pending || !state.connected) return;
    polling = true;
    const version = requestVersion;
    try {
      const next = await api('/api/tuning/status', { timeout: 10000 });
      const selected = target();
      const otherTarget = selected && !activePhases.has(next.phase) &&
        (selected.model !== next.model || (next.directory && targetKey(selected) !== targetKey({directory:next.directory,model:next.model})));
      if (version === requestVersion && !otherTarget) await accept(next);
    } catch (error) {
      if (active()) el('tuning-status').textContent = `Waiting for tuning status. ${error.message}`;
    } finally { polling = false; }
  }

  async function command(path, body) {
    if (pending || !state.connected) return;
    pending = true;
    requestVersion++;
    controls();
    try {
      const next = await api(path, { body, timeout: 15000 });
      const ranInstantly = !activePhases.has(next.phase) && path === '/api/tuning/start';
      await accept(next);
      if (ranInstantly) await onComplete(!next.preparing || !!next.activated);
    } catch (error) {
      el('tuning-status').textContent = error.message;
    } finally {
      pending = false;
      controls();
    }
  }

  async function prepare(next, force = false) {
    if (next) selectedTarget={directory:next.directory,model:next.model};
    const chosen=target(); if (!chosen || active()) return;
    const goal={context:el('tuning-context-goal').value};
    const ceiling=el('tuning-context-ceiling').value.trim();
    if (ceiling) goal.max_context=Number(ceiling);
    el('tuning').scrollIntoView({behavior:'smooth'});
    await command('/api/tuning/start',{target:chosen,mode:'prepare',goal,activate:false,force});
  }
  el('tuning-context-goal').addEventListener('change',controls);
  el('tuning-prepare').addEventListener('click', () => void prepare());
  el('tuning-prepare-rerun').addEventListener('click', () => void prepare(null,true));
  el('tuning-use').addEventListener('click', () => { if (target()) void command('/api/tuning/use',target()); });
  el('tuning-start').addEventListener('click', () => {
    if (!el('tuning-start').disabled) void command('/api/tuning/start', { force: false });
  });
  el('tuning-rerun').addEventListener('click', () => {
    if (!el('tuning-rerun').disabled) void command('/api/tuning/start', { force: true });
  });
  el('tuning-cancel').addEventListener('click', () => {
    if (!el('tuning-cancel').disabled) void command('/api/tuning/cancel', {});
  });

  function update() {
    if (updating) return;
    updating = true;
    try {
      controls();
      const selection = state.selection?.id && state.selection?.root
        ? {directory:state.selection.root,model:state.selection.id} : null;
      const selectionKey=targetKey(selection);
      if (selectionKey && selectionKey!==observedSelection && !active()) {
        observedSelection=selectionKey;
        void selectTarget(selection);
      }
      const key = `${currentModel() || ''}\n${context() || ''}`;
      if (key !== observedModel && !active()) {
        observedModel = key;
        requestVersion++;
        if (!job?.preparing) job = null;
        render();
        void poll();
      }
      if (state.connected && timer === null) {
        timer = setInterval(() => { if (!document.hidden) void poll(); }, 1500);
        void poll();
      }
    } finally { updating = false; }
  }

  return { update, busy: active, prepare, select:selectTarget };
}
