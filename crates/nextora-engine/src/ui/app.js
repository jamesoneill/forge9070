'use strict';

(() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    config: null, status: null, scan: null, selection: null,
    apiKey: '', connected: false, dirty: false,
    scanning: false, applying: false, refreshing: false, hardwareLoading: false,
    polling: false, insightLoading: false, lastError: '', hardware: [], pendingDevice: null,
  };
  const fields = {
    backend: 'backend', device: 'device', ctx: 'ctx', max_tokens: 'max-tokens',
    parallel: 'parallel', max_queued: 'max-queued', queue_timeout_secs: 'queue-timeout',
    vram_mib: 'vram-mib', expert_ram_mib: 'expert-ram-mib',
    ram_margin_mib: 'ram-margin-mib', expert_warmup: 'expert-warmup',
  };
  const defaults = {
    backend: 'auto', device: null, ctx: 8192, max_tokens: 2048, parallel: 1,
    max_queued: 32, queue_timeout_secs: 120, vram_mib: 0,
    expert_ram_mib: null, ram_margin_mib: 4096, expert_warmup: true,
  };

  class ApiError extends Error {
    constructor(message, status) { super(message); this.status = status; }
  }

  async function api(path, options = {}) {
    const headers = { Accept: 'application/json' };
    if (state.apiKey) headers.Authorization = `Bearer ${state.apiKey}`;
    if (options.body !== undefined) headers['Content-Type'] = 'application/json';
    const controller = new AbortController();
    const timer = options.timeout ? setTimeout(() => controller.abort(), options.timeout) : null;
    let response;
    try {
      response = await fetch(path, {
        method: options.body === undefined ? 'GET' : 'POST', headers,
        body: options.body === undefined ? undefined : JSON.stringify(options.body),
        cache: 'no-store', credentials: 'same-origin', redirect: 'error', signal: controller.signal,
      });
    } catch (error) {
      throw new ApiError(error.name === 'AbortError' ? 'The service did not respond in time. Check the terminal running Nextora.' : 'Cannot reach the service. Check that Nextora is still running, then refresh.', 0);
    } finally {
      if (timer) clearTimeout(timer);
    }
    let data;
    try { data = await response.json(); }
    catch (_) { throw new ApiError('The service returned an unreadable response.', response.status); }
    if (!response.ok) {
      if (response.status === 401) requireKey();
      throw new ApiError(data?.error?.message || `Request failed (${response.status}).`, response.status);
    }
    return data;
  }

  function notice(message, tone = 'success') {
    $('notice').textContent = message;
    $('notice').dataset.tone = tone;
    $('notice').hidden = !message;
  }

  function showError(error) {
    notice(error.message || String(error), error.status === 409 ? 'warning' : 'error');
    if (error.status === 0) setServiceState('offline', 'Service offline');
  }

  function requireKey() {
    state.connected = false;
    const wasHidden = $('auth-panel').hidden;
    $('auth-panel').hidden = false;
    setServiceState('locked', 'API key required');
    updateControls();
    if (wasHidden) $('api-key').focus();
  }

  function setServiceState(kind, label) {
    $('service-state').dataset.state = kind;
    $('service-state-text').textContent = label;
  }

  function node(tag, className, text) {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text !== undefined) item.textContent = String(text);
    return item;
  }

  function bytes(value) {
    if (!Number.isFinite(Number(value)) || value === null || value === undefined) return '—';
    const n = Number(value);
    if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GiB`;
    if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MiB`;
    if (n >= 1024) return `${(n / 1024).toFixed(1)} KiB`;
    return `${n} B`;
  }

  function modelPath(root, id) {
    if (!root || !id) return id || '';
    const windows = /^[a-z]:[\\/]/i.test(root) || root.startsWith('\\\\');
    const separator = windows ? '\\' : '/';
    return root.replace(/[\\/]$/, '') + separator + id.replace(/[\\/]/g, separator);
  }

  function isSameRoot(a, b) {
    if (!a || !b) return a === b;
    const clean = (value) => value.replace(/\\/g, '/').replace(/\/$/, '');
    return state.config?.platform === 'windows' ? clean(a).toLowerCase() === clean(b).toLowerCase() : clean(a) === clean(b);
  }

  function setDirty(dirty = true) {
    state.dirty = dirty;
    $('draft-state').textContent = dirty ? 'Unsaved changes' : 'Current settings';
    $('draft-state').dataset.dirty = String(dirty);
    updateControls();
  }

  function backendHelp() {
    const backend = $('backend').value;
    $('backend-help').textContent = backend === 'auto'
      ? 'Automatically select compatible GPU hardware, or CPU when none is available. Choosing an adapter or a nonzero GPU cap requires GPU execution; model-loading errors are reported.'
      : backend === 'cpu'
        ? 'Explicit CPU execution. GPU adapter and GPU allocation cap are unused.'
        : backend === 'hybrid'
          ? 'Diagnostic path: CPU state with GPU projections. Use Vulkan for the resident GPU graph.'
          : 'Native GPU graph with bounded expert placement. Hardware support is checked when loading; this mode does not switch to CPU.';
    $('device').disabled = backend === 'cpu' || state.applying || !state.connected;
    $('vram-mib').disabled = backend === 'cpu' || state.applying || !state.connected;
  }

  function updateControls() {
    const working = state.scanning || state.applying;
    $('scan-models').disabled = working || !state.connected;
    $('scan-models').textContent = state.scanning ? 'Scanning…' : 'Scan models';
    $('model-directory').disabled = state.applying || !state.connected;
    $('use-f-drive').disabled = working || !state.connected;
    $('runtime-fields').disabled = state.applying || !state.connected;
    $('apply-settings').disabled = working || !state.connected;
    $('apply-settings').textContent = state.applying ? 'Applying…' : state.selection ? 'Apply & load model' : 'Apply settings';
    $('reset-settings').disabled = working || !state.connected || !state.dirty;
    $('unload-model').disabled = working || !state.connected || !state.status?.model;
    $('export-settings').disabled = working || !state.connected;
    $('refresh-hardware').disabled = state.hardwareLoading || state.applying || !state.connected;
    $('refresh-status').disabled = state.refreshing || state.applying;
    document.querySelectorAll('input[name="model"]').forEach((input) => { input.disabled = working; });
    $('suggested-folders').querySelectorAll('button').forEach((button) => { button.disabled = working || !state.connected; });
    if (state.applying) {
      $('apply-title').textContent = 'Applying configuration';
      $('apply-help').textContent = 'Large models and cache warmup can take a while. Keep this page open.';
    } else if (state.selection) {
      $('apply-title').textContent = 'Ready to load your selection';
      $('apply-help').textContent = 'Applies the model and settings together. Existing work must finish first.';
    } else {
      $('apply-title').textContent = 'Ready when you are';
      $('apply-help').textContent = 'Choose a model above, or save runtime settings for later.';
    }
    backendHelp();
  }

  function renderSelection() {
    $('selected-model').hidden = !state.selection;
    $('selected-model-name').textContent = state.selection?.id || '';
    $('selected-model-path').textContent = state.selection ? modelPath(state.selection.root, state.selection.id) : '';
    document.querySelectorAll('input[name="model"]').forEach((input) => {
      const selected = !!state.selection && isSameRoot(state.selection.root, state.scan?.root) && input.value === state.selection.id;
      input.checked = selected;
      input.closest('.model-option').dataset.selected = String(selected);
    });
    updateControls();
  }

  function applyConfigToForm(config) {
    const settings = { ...defaults, ...config.settings };
    for (const [key, id] of Object.entries(fields)) {
      if (key === 'expert_warmup') $(id).checked = settings[key];
      else if (key === 'device') {
        state.pendingDevice = settings.device;
        renderDeviceOptions();
      } else $(id).value = settings[key] === null ? '' : String(settings[key]);
    }
    $('model-directory').value = config.models_dir || (config.platform === 'windows' ? 'F:\\' : config.in_wsl ? '/mnt/f' : '');
    const selectedModel = config.model || config.desired_model;
    state.selection = selectedModel ? { id: selectedModel, root: config.models_dir } : null;
    setDirty(false);
    renderSelection();
  }

  function rememberConfig(config, resetForm = false) {
    state.config = config;
    $('engine-version').textContent = `Forge native ${config.version || ''}`.trim();
    $('platform-label').textContent = config.in_wsl ? 'Running inside WSL' : `Running on ${config.platform || 'this machine'}`;
    const windows = config.platform === 'windows';
    $('use-f-drive').hidden = !(windows || config.in_wsl);
    $('use-f-drive').textContent = windows ? 'Use F: drive' : 'Use /mnt/f';
    $('model-directory').placeholder = windows ? 'F:\\Models' : config.in_wsl ? '/mnt/f/Models' : '/path/to/models';
    $('directory-help').textContent = windows
      ? 'This is the Windows service. Use a Windows path such as F:\\Models.'
      : config.in_wsl
        ? 'This is the Linux service in WSL. A mounted Windows F: drive is usually /mnt/f.'
        : 'Use a folder on the machine running this service, not the browser’s computer.';
    $('suggested-folders').replaceChildren();
    const suggested = Array.isArray(config.suggested_model_dirs) ? config.suggested_model_dirs : [];
    $('suggested-folders').hidden = !suggested.length;
    for (const folder of suggested.slice(0, 8)) {
      const button = node('button', 'folder-shortcut', folder.label || folder.path);
      button.type = 'button'; button.title = folder.path;
      button.addEventListener('click', () => {
        if (state.applying || state.scanning || !state.connected) return;
        $('model-directory').value = folder.path; setDirty(); $('model-directory').focus();
      });
      $('suggested-folders').append(button);
    }
    if (suggested.some((folder) => /f:\\|\/mnt\/f/i.test(folder.path))) $('use-f-drive').hidden = true;
    $('persistence-note').textContent = config.settings_file
      ? `Successful changes are saved to ${config.settings_file}. Host, port, API key, and CPU worker count still come from startup options.`
      : 'Changes apply to this running service. Export a settings file to keep them, then start with --settings-file. API keys are not included.';
    $('client-auth-hint').textContent = config.startup?.api_key_required
      ? 'API key required: use the same service key in your coding client.'
      : 'No service API key is configured. If your client requires a value, use any placeholder.';
    if (resetForm) applyConfigToForm(config);
    else updateControls();
    if (config.startup_error) {
      notice(`The saved model could not be loaded. The console is available so you can fix its path or runtime settings.\n${config.startup_error}`, 'warning');
    }
  }

  function readSettings() {
    if (!$('settings-form').reportValidity()) throw new ApiError('Check the highlighted setting.', 400);
    const out = {};
    for (const [key, id] of Object.entries(fields)) {
      const input = $(id);
      if (key === 'backend') out[key] = input.value;
      else if (out.backend === 'cpu' && key === 'device') out[key] = null;
      else if (out.backend === 'cpu' && key === 'vram_mib') out[key] = 0;
      else if (key === 'expert_warmup') out[key] = input.checked;
      else if ((key === 'expert_ram_mib' || key === 'device') && input.value.trim() === '') out[key] = null;
      else {
        out[key] = Number(input.value);
        if (!Number.isSafeInteger(out[key]) || out[key] < 0) throw new ApiError(`${input.labels?.[0]?.textContent || key} must be a whole, non-negative number.`, 400);
      }
    }
    if (out.max_tokens >= out.ctx) {
      $('max-tokens').focus();
      throw new ApiError('The output token cap must be smaller than the context per request.', 400);
    }
    return out;
  }

  function draftProfile() {
    return {
      directory: state.selection?.root || state.scan?.root || $('model-directory').value.trim() || null,
      model: state.selection?.id || null,
      settings: readSettings(),
    };
  }

  function emptyModels(title, description) {
    const empty = node('div', 'empty-state');
    empty.append(node('div', 'folder-symbol', '⌑'), node('h3', '', title), node('p', '', description));
    empty.firstChild.setAttribute('aria-hidden', 'true');
    $('model-results').replaceChildren(empty);
  }

  function renderScan(result) {
    state.scan = result;
    const models = Array.isArray(result.models) ? result.models : [];
    $('model-count').textContent = `${models.length} compatible`;
    const scanned = Number(result.scanned_entries || 0).toLocaleString();
    $('scan-summary').textContent = `${models.length} compatible model${models.length === 1 ? '' : 's'} found · ${scanned} entries checked${result.truncated ? ' · Scan limit reached' : ''}`;
    $('model-results').replaceChildren();
    if (!models.length) {
      const foundSafe = Number(result.safetensors_files || 0);
      emptyModels('No compatible GGUF models found', foundSafe
        ? `Found ${foundSafe} safetensors file${foundSafe === 1 ? '' : 's'}. This engine needs a supported GGUF checkpoint with all its shards. Check any file warnings below, or scan another folder.`
        : 'Try the folder where the weights were downloaded. Check the warnings below for missing shards or incompatible files. The source ZIP does not contain pretrained model weights.');
    }
    for (const model of models) {
      const option = node('label', 'model-option');
      const input = node('input');
      input.type = 'radio'; input.name = 'model'; input.value = model.id;
      input.dataset.modelId = model.id;
      const body = node('div', 'model-option-body');
      body.append(node('div', 'model-option-title', model.id));
      body.append(node('div', 'model-option-path', model.path || modelPath(result.root, model.id)));
      const meta = node('div', 'model-option-meta');
      meta.append(node('span', '', bytes(model.bytes)));
      if (model.architecture) meta.append(node('span', '', model.architecture));
      const shards = Array.isArray(model.shards) ? model.shards.length : Number(model.shards || 1);
      meta.append(node('span', '', `${shards} file${shards === 1 ? '' : 's'}`));
      body.append(meta); option.append(input, body);
      input.addEventListener('change', () => {
        state.selection = { id: model.id, root: result.root };
        setDirty(); renderSelection();
      });
      $('model-results').append(option);
    }
    const issues = [];
    for (const rejected of result.rejected || []) issues.push({ title: rejected.path || 'Skipped file', detail: rejected.reason || 'Not compatible with this engine.' });
    for (const warning of result.warnings || []) issues.push({ title: 'Scan warning', detail: typeof warning === 'string' ? warning : JSON.stringify(warning) });
    if (result.truncated) issues.unshift({ title: 'Scan limit reached', detail: 'This scan is bounded. Choose a more specific folder to check the remaining files.' });
    $('scan-issues').replaceChildren();
    $('scan-details').hidden = !issues.length;
    $('scan-details-summary').textContent = `${issues.length} file warning${issues.length === 1 ? '' : 's'} or skipped item${issues.length === 1 ? '' : 's'}`;
    $('scan-details').open = models.length === 0 && issues.length > 0;
    for (const issue of issues.slice(0, 100)) {
      const item = node('div', 'scan-issue');
      item.append(node('strong', '', issue.title), node('p', '', issue.detail));
      $('scan-issues').append(item);
    }
    if (issues.length > 100) $('scan-issues').append(node('p', 'field-help', `Showing the first 100 of ${issues.length} warnings.`));
    renderSelection();
  }

  function hardwareCompatible(adapter) {
    return !adapter.software && adapter.compatible === true;
  }

  function hardwareRejection(adapter) {
    const reasons = Array.isArray(adapter.rejection_reasons)
      ? adapter.rejection_reasons.filter((reason) => typeof reason === 'string') : [];
    if (reasons.length) return reasons.join('; ');
    if (adapter.software) return 'Software adapter; GPU hardware is required';
    if (adapter.compute_queue === null) return 'No compute queue';
    return 'Required GPU capabilities are unavailable';
  }

  function renderDeviceOptions() {
    const selected = state.pendingDevice;
    const automatic = node('option', '', 'Automatic hardware selection'); automatic.value = '';
    $('device').replaceChildren(automatic);
    for (const adapter of state.hardware) {
      const unavailable = !hardwareCompatible(adapter);
      const reason = unavailable ? hardwareRejection(adapter) : '';
      const option = node('option', '', `${adapter.index}: ${adapter.name}${unavailable ? ' · unavailable' : ''}`);
      option.value = String(adapter.index); option.disabled = unavailable; option.title = reason;
      $('device').append(option);
    }
    if (selected !== null && selected !== undefined && !state.hardware.some((adapter) => adapter.index === Number(selected))) {
      const option = node('option', '', `${selected}: configured adapter (not discovered)`);
      option.value = String(selected); $('device').append(option);
    }
    $('device').value = selected === null || selected === undefined ? '' : String(selected);
    backendHelp();
  }

  async function refreshHardware() {
    if (!state.connected || state.hardwareLoading) return;
    state.hardwareLoading = true; updateControls();
    $('hardware-status').textContent = 'Checking available adapters…';
    $('hardware-status').dataset.warning = 'false';
    try {
      const data = await api('/api/hardware', { timeout: 20000 });
      state.pendingDevice = $('device').value === '' ? null : Number($('device').value);
      state.hardware = Array.isArray(data.adapters) ? data.adapters : [];
      renderDeviceOptions();
      const physical = state.hardware.filter(hardwareCompatible);
      const cpuHint = data.cpu_available ? 'Automatic mode can use CPU.' : 'CPU execution is unavailable on this platform.';
      $('hardware-status').textContent = physical.length
        ? `${physical.length} hardware adapter${physical.length === 1 ? '' : 's'} meet${physical.length === 1 ? 's' : ''} the baseline Vulkan requirements.`
        : data.vulkan_discovery_error
          ? `Vulkan discovery is unavailable: ${data.vulkan_discovery_error}. ${cpuHint}`
          : state.config?.in_wsl
            ? `No compatible hardware Vulkan adapter is available to this WSL process. ${cpuHint} Run the Windows native service to use the Windows GPU driver, or fix WSL GPU exposure.`
            : `No compatible hardware Vulkan adapter was found. ${cpuHint} Check the GPU driver and the requirements below for GPU execution.`;
      $('hardware-status').dataset.warning = String(!physical.length);
      $('hardware-list').replaceChildren();
      for (const adapter of state.hardware) {
        const item = node('div', 'hardware-item'); item.dataset.software = String(!!adapter.software);
        item.append(node('strong', '', `${adapter.index}: ${adapter.name}`));
        const heap = Math.max(0, ...(adapter.device_local_heaps_bytes || []));
        item.append(node('p', '', `Vulkan ${adapter.vulkan_version || '—'}${heap ? ` · ${bytes(heap)} largest device heap` : ''}`));
        const compatible = hardwareCompatible(adapter);
        item.append(node('span', 'hardware-tag', compatible ? 'Capabilities compatible · compute not yet verified' : 'Unavailable for GPU execution'));
        if (!compatible) item.append(node('p', '', hardwareRejection(adapter)));
        $('hardware-list').append(item);
      }
    } catch (error) {
      state.hardware = [];
      renderDeviceOptions();
      $('hardware-list').replaceChildren();
      $('hardware-status').textContent = error.message;
      $('hardware-status').dataset.warning = 'true';
    } finally { state.hardwareLoading = false; updateControls(); }
  }

  function addRuntimeRow(label, value) {
    if (value === undefined || value === null) return;
    const row = node('div', 'runtime-stat-row');
    row.append(node('dt', '', label), node('dd', '', value)); $('runtime-stats').append(row);
  }


  function insightRate(value) {
    return Number.isFinite(Number(value)) && Number(value) >= 0
      ? Number(value).toFixed(1) + ' tok/s' : 'No samples';
  }
  function renderInsight(result) {
    const m = result.measurements || {};
    const host = result.host || {};
    const memory = result.memory || {};
    const samples = Array.isArray(m.recent_samples) ? m.recent_samples : [];
    $('insight-generation').textContent = insightRate(m.recent_aggregate_generation_tokens_per_second);
    $('insight-prefill').textContent = insightRate(m.recent_aggregate_prefill_tokens_per_second);
    $('insight-ram').textContent = bytes(host.available_host_ram_bytes);
    const gpuUsed = memory.gpu_live_owned_allocation_bytes;
    const gpuCap = memory.gpu_allocation_budget_bytes;
    $('insight-vram').textContent = gpuUsed === null || gpuUsed === undefined ? 'Not measured'
      : bytes(gpuUsed) + (gpuCap !== null && gpuCap !== undefined ? ' / ' + bytes(gpuCap) : '');
    $('insight-trend').replaceChildren();
    const visible = samples.slice(-24);
    const maximum = Math.max(1, ...visible.map(sample => Number(sample.generation_tokens_per_second) || 0));
    visible.forEach((sample, i) => {
      const speed = Math.max(0, Number(sample.generation_tokens_per_second) || 0);
      const item = node('span', 'insight-trend-bar');
      item.style.height = Math.max(2, Math.min(100, speed / maximum * 100)) + '%';
      item.title = 'Completed request ' + (i + 1) + ': ' + speed.toFixed(1) + ' output tok/s';
      $('insight-trend').append(item);
    });
    $('insight-trend').setAttribute('aria-label', visible.length
      ? visible.length + ' completed requests, latest ' + Number(visible[visible.length - 1].generation_tokens_per_second || 0).toFixed(1) + ' output tokens per second'
      : 'No completed request measurements yet');
    $('insight-sample-note').textContent = samples.length
      ? (m.completed_requests || 0) + ' successful requests · ' + (m.interrupted_or_failed_requests || 0) + ' interrupted/failed · last ' + samples.length + ' shown · ' + (result.admission?.queued || 0) + ' waiting'
      : 'No successful completed requests. Rates only appear after real inference.';
    const cacheHits = memory.host_expert_cache_hits_cumulative;
    const cacheMisses = memory.host_expert_cache_misses_cumulative;
    const cacheEvictions = memory.host_expert_cache_evictions_cumulative;
    const cacheBytes = memory.host_expert_payload_read_bytes_cumulative;
    $('insight-expert-cache').textContent = cacheHits === null || cacheHits === undefined
      ? 'Expert-cache counters are unavailable until a compatible model is loaded.'
      : 'Expert cache since load: ' + cacheHits + ' hits · ' + (cacheMisses ?? 0) +
        ' misses · ' + (cacheEvictions ?? 0) + ' evictions · ' + bytes(cacheBytes) + ' read';
    const rec = $('insight-recommendations');
    rec.replaceChildren();
    const recommendations = Array.isArray(result.recommendations) ? result.recommendations : [];
    if (!recommendations.length) rec.append(node('p', 'small-note',
      'No immediate recommendations from the currently available evidence.'));
    for (const recommendation of recommendations.slice(0, 5)) {
      const item = node('div', 'insight-recommendation');
      const heading = node('strong', '', (recommendation.id || 'Review').replace(/-/g, ' '));
      const evidence = node('p', 'insight-evidence', recommendation.evidence || '');
      const action = node('p', '', recommendation.action || '');
      const purchase = node('span', 'insight-purchase', recommendation.purchase === 'conditional_only'
        ? 'Upgrade only if measured' : 'Try before spending');
      item.append(heading, evidence, action, purchase);
      rec.append(item);
    }
    $('insight-status').textContent = 'Local only · ' + (result.backend || 'No active model') + ' · live snapshot';
  }
  async function refreshInsight() {
    if (!state.connected || state.insightLoading || document.hidden) return;
    state.insightLoading = true;
    try {
      renderInsight(await api('/api/insight/live', { timeout: 10000 }));
    } catch (error) {
      $('insight-status').textContent = 'Insight unavailable: ' + error.message;
    } finally { state.insightLoading = false; }
  }

  function renderStatus(status) {
    state.status = status;
    const admission = status.admission || {};
    const runtime = status.runtime;
    const changing = !!admission.model_changing || state.applying;
    setServiceState(changing ? 'loading' : status.model ? 'ready' : 'waiting', changing ? 'Loading / applying' : status.model ? 'Service ready' : 'Awaiting model');
    $('loaded-model-name').textContent = status.model || 'No model loaded';
    $('loaded-model-name').title = status.model || '';
    $('loaded-model-detail').textContent = status.model ? `${runtime?.config?.architecture || 'MoE checkpoint'} · ${bytes(runtime?.mapped_weight_bytes)} on disk` : 'Select a GGUF to get started';
    const execution = runtime?.execution || {};
    const selection = runtime?.backend_selection;
    $('execution-name').textContent = execution.backend === 'cpu' ? 'CPU' : execution.backend === 'vulkan' ? 'Vulkan' : execution.backend ? 'Hybrid' : '—';
    const adapterName = execution.adapter?.name || execution.statistics?.adapter?.name;
    $('execution-detail').textContent = selection?.automatic
      ? `Automatic selection${adapterName ? ` · ${adapterName}` : ''} · ${selection.reason}`
      : adapterName || (execution.backend === 'cpu' ? 'Explicit CPU execution' : execution.backend ? 'CPU state + Vulkan projections' : 'Runtime starts when a model loads');
    $('execution-detail').title = $('execution-detail').textContent;
    $('active-count').textContent = String(admission.active ?? 0);
    $('active-capacity').textContent = String(admission.capacity ?? state.config?.settings?.parallel ?? 1);
    $('queued-count').textContent = String(admission.queued ?? 0);
    $('queue-capacity').textContent = String(admission.queue_limit ?? state.config?.settings?.max_queued ?? 32);
    $('client-model').value = status.model || 'Load a model first';
    $('copy-model-id').disabled = !status.model;
    $('runtime-empty').hidden = !!runtime;
    $('runtime-stats').hidden = !runtime;
    $('runtime-details').hidden = !runtime;
    $('runtime-stats').replaceChildren();
    if (runtime) {
      if (selection) {
        addRuntimeRow('Backend choice', selection.automatic ? 'Automatic' : 'Explicit');
        addRuntimeRow('Selection reason', selection.reason);
      }
      addRuntimeRow('Weight files', bytes(runtime.mapped_weight_bytes));
      addRuntimeRow('Shards', Array.isArray(runtime.shards) ? runtime.shards.length : runtime.shards);
      if (runtime.expert_cache?.resident_bytes !== undefined) addRuntimeRow('Expert RAM used', bytes(runtime.expert_cache.resident_bytes));
      if (runtime.host_memory_plan?.expert_cache_capacity_bytes !== undefined) addRuntimeRow('Expert RAM cap', bytes(runtime.host_memory_plan.expert_cache_capacity_bytes));
      if (runtime.host_memory_plan?.state_reserve_bytes !== undefined) addRuntimeRow('Host state reserve', bytes(runtime.host_memory_plan.state_reserve_bytes));
      const memory = execution.memory || execution.statistics || {};
      if (memory.live_allocation_bytes !== undefined) addRuntimeRow('GPU allocations', bytes(memory.live_allocation_bytes));
      if (memory.allocation_budget_bytes !== undefined) addRuntimeRow('GPU allocation cap', bytes(memory.allocation_budget_bytes));
      if (memory.cpu_expert_fallbacks !== undefined) addRuntimeRow('CPU expert executions', memory.cpu_expert_fallbacks);
      if (memory.selected_experts_omitted !== undefined) addRuntimeRow('Selected experts omitted', memory.selected_experts_omitted);
      addRuntimeRow('Reusable sequences', status.idle_sequences ?? 0);
      addRuntimeRow('State precision', runtime.kv_precision);
      $('runtime-json').textContent = JSON.stringify(runtime, null, 2);
    } else $('runtime-json').textContent = '';
    $('last-updated').textContent = `Status updated ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;
    updateControls();
  }

  async function refreshStatus(silent = false) {
    if (!state.connected || state.polling) return;
    state.polling = true;
    try {
      renderStatus(await api('/v1/status', { timeout: 10000 }));
      state.lastError = '';
    } catch (error) {
      if (error.status !== 401) {
        setServiceState('offline', 'Service offline');
        $('last-updated').textContent = 'Status unavailable · check the service terminal';
        if (!silent && state.lastError !== error.message) showError(error);
        state.lastError = error.message;
      }
    } finally { state.polling = false; }
  }

  async function connect(resetForm = false) {
    if (state.refreshing) return;
    state.refreshing = true; updateControls();
    try {
      const config = await api('/api/config', { timeout: 10000 });
      state.connected = true; $('auth-panel').hidden = true; $('api-key').value = '';
      rememberConfig(config, resetForm || !state.config);
      await refreshStatus();
      void refreshHardware();
      void refreshInsight();
    } catch (error) {
      if (error.status !== 401) showError(error);
      else if (state.apiKey) notice('The service did not accept that API key. Check the key used to start the service and try again.', 'error');
    } finally { state.refreshing = false; updateControls(); }
  }

  async function scanModels(event) {
    event?.preventDefault();
    if (state.scanning || state.applying || !state.connected) return;
    const directory = $('model-directory').value.trim();
    if (!directory) { $('model-directory').focus(); notice('Enter a model folder or drive to scan.', 'warning'); return; }
    state.scanning = true; updateControls(); notice('');
    $('scan-summary').textContent = 'Scanning model headers and checking complete shard sets…';
    try {
      const result = await api('/api/models/scan', { body: { directory } });
      $('model-directory').value = result.root || directory;
      renderScan(result);
      if (state.selection && !isSameRoot(state.selection.root, result.root)) {
        notice('Scan complete. Your current model selection is unchanged. Select a result to use a model from this folder.', 'warning');
      }
    } catch (error) {
      $('scan-summary').textContent = 'Scan failed. Check the folder path and try again.';
      showError(error);
    } finally { state.scanning = false; updateControls(); }
  }

  async function applyProfile(event) {
    event?.preventDefault();
    if (state.applying || state.scanning || !state.connected) return;
    let profile;
    try { profile = draftProfile(); } catch (error) { showError(error); return; }
    state.applying = true; updateControls(); notice('');
    setServiceState('loading', 'Loading / applying');
    try {
      const result = await api('/api/config', { body: profile });
      const config = result.settings ? result : await api('/api/config');
      rememberConfig(config, true);
      notice(profile.model ? 'Model loaded and settings applied. Your service is ready for client requests.' : 'Settings applied. Select and load a compatible model when you are ready.');
      await refreshStatus();
    } catch (error) {
      showError(error);
      await refreshStatus(true);
    } finally {
      state.applying = false; updateControls();
      if (state.status && state.connected) renderStatus(state.status);
    }
  }

  async function unloadModel() {
    if (!state.config || !state.status?.model || state.applying || state.scanning) return;
    const wasDirty = state.dirty;
    state.applying = true; updateControls(); notice('');
    try {
      const result = await api('/api/config', { body: { directory: state.config.models_dir, model: null, settings: state.config.settings } });
      const config = result.settings ? result : await api('/api/config');
      rememberConfig(config, !wasDirty);
      if (wasDirty) { state.selection = null; renderSelection(); }
      notice(wasDirty ? 'Model unloaded and its runtime released. Your unsaved settings are still in the form.' : 'Model unloaded and its runtime released. The API and configuration console remain available.');
      await refreshStatus();
    } catch (error) { showError(error); await refreshStatus(true); }
    finally { state.applying = false; updateControls(); if (state.status && state.connected) renderStatus(state.status); }
  }

  async function copyText(inputId, buttonId) {
    const input = $(inputId), button = $(buttonId);
    try {
      await navigator.clipboard.writeText(input.value);
      button.textContent = 'Copied'; setTimeout(() => { button.textContent = 'Copy'; }, 1600);
    } catch (_) {
      input.focus(); input.select();
      notice('The value is selected. Press Ctrl+C (or Command+C) to copy it.', 'warning');
    }
  }

  function exportSettings() {
    try {
      const profile = draftProfile();
      const blob = new Blob([JSON.stringify(profile, null, 2) + '\n'], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = node('a'); link.href = url; link.download = 'nextora-service.json';
      document.body.append(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      notice('Settings exported as nextora-service.json. This exports the current form; it does not load or change the running model.');
    } catch (error) { showError(error); }
  }

  $('api-base').value = `${location.origin}/v1`;
  $('auth-form').addEventListener('submit', async (event) => {
    event.preventDefault(); state.apiKey = $('api-key').value.trim();
    $('unlock').disabled = true; notice('');
    try { await connect(!state.config); }
    finally { $('unlock').disabled = false; }
  });
  $('scan-form').addEventListener('submit', scanModels);
  $('settings-form').addEventListener('submit', applyProfile);
  $('unload-model').addEventListener('click', unloadModel);
  $('refresh-hardware').addEventListener('click', refreshHardware);
  $('insight-refresh').addEventListener('click', refreshInsight);
  $('refresh-status').addEventListener('click', () => connect(!state.config));
  $('export-settings').addEventListener('click', exportSettings);
  $('copy-api-base').addEventListener('click', () => copyText('api-base', 'copy-api-base'));
  $('copy-model-id').addEventListener('click', () => copyText('client-model', 'copy-model-id'));
  $('use-f-drive').addEventListener('click', () => { $('model-directory').value = state.config?.platform === 'windows' ? 'F:\\' : '/mnt/f'; setDirty(); $('model-directory').focus(); });
  $('model-directory').addEventListener('input', () => setDirty());
  $('reset-settings').addEventListener('click', () => { if (state.config) { applyConfigToForm(state.config); notice('Form reset to the applied service configuration.'); } });
  for (const id of Object.values(fields)) {
    $(id).addEventListener('input', () => {
      if (id === 'device') state.pendingDevice = $('device').value === '' ? null : Number($('device').value);
      setDirty(); if (id === 'backend') backendHelp();
    });
  }
  window.addEventListener('beforeunload', (event) => {
    if (state.applying) { event.preventDefault(); event.returnValue = ''; }
  });
  updateControls();
  void connect(true);
  setInterval(() => { if (!document.hidden) void refreshStatus(true); }, 2500);
  setInterval(() => { if (!document.hidden) void refreshInsight(); }, 5000);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) void refreshStatus(true); });
})();
