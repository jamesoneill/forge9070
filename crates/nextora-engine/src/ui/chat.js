'use strict';

// Read complete SSE events across arbitrary network and UTF-8 chunk boundaries.
async function readNextoraChatStream(body, onChunk) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let pending = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      let boundary;
      while ((boundary = /\r?\n\r?\n/.exec(pending))) {
        const event = pending.slice(0, boundary.index);
        pending = pending.slice(boundary.index + boundary[0].length);
        const data = event.split(/\r?\n/).filter(line => line.startsWith('data:'))
          .map(line => line.slice(5).trimStart()).join('\n');
        if (data === '[DONE]') return;
        if (data) {
          const chunk = JSON.parse(data);
          if (chunk.error) throw new Error(chunk.error.message || 'Generation failed.');
          onChunk(chunk);
        }
      }
      if (done) throw new Error('The response ended before generation finished.');
    }
  } finally {
    reader.releaseLock();
  }
}

function createNextoraChat(state, api, requireKey) {
  const el = id => document.getElementById(id);
  const log = el('chat-log'), prompt = el('chat-prompt'), limit = el('chat-limit');
  const status = el('chat-status'), preview = el('chat-context');
  const history = [];
  let controller = null, readingFiles = false, checking = false, model = null;
  let outputCap = 0, draftVersion = 0, benchmark = null;

  function invalidatePreview() {
    draftVersion++;
    preview.textContent = '';
  }

  function update() {
    const current = state.status?.model || (state.tuningActive ? state.config?.model : null) || null;
    if (current !== model) {
      model = current;
      history.length = 0;
      log.replaceChildren();
      controller?.abort();
      benchmark?.abort();
      invalidatePreview();
    }
    const maximum = state.config?.settings?.max_tokens || 128;
    if (outputCap !== maximum) {
      outputCap = maximum;
      limit.max = String(maximum);
      limit.value = String(Math.min(Number(limit.value) || 128, maximum));
      invalidatePreview();
    }
    const available = !!model && state.connected && !state.applying && !state.tuningActive && !state.status?.admission?.model_changing;
    el('chat-model').textContent = model ? model.split(/[\\/]/).pop() : 'Load a model to chat';
    const busy = !!controller || !!benchmark || readingFiles || checking;
    el('speed-run').disabled = !available || busy;
    el('speed-stop').hidden = !benchmark;
    el('chat-send').disabled = !available || busy || !prompt.value.trim();
    el('chat-count').disabled = el('chat-send').disabled;
    el('chat-stop').hidden = !controller;
    el('chat-clear').disabled = busy || !log.childElementCount;
    el('chat-files').disabled = busy;
    prompt.readOnly = busy;
    limit.disabled = busy;
  }

  function requestBody() {
    const draft = prompt.value.trim();
    if (!model || !draft || !limit.reportValidity()) return null;
    return { model, messages: [...history, { role: 'user', content: draft }], max_tokens: Number(limit.value) };
  }

  function addMessage(label, text) {
    const message = document.createElement('article');
    message.className = 'chat-message';
    const author = document.createElement('strong');
    author.textContent = label;
    const content = document.createElement('p');
    content.textContent = text;
    message.append(author, content);
    log.append(message);
    return content;
  }

  function measurements(chunk, elapsed) {
    const u = chunk.usage, t = chunk.timings;
    if (!u) return '';
    const generated = u.completion_tokens, cached = u.prompt_tokens_details?.cached_tokens || 0;
    const parts = [];
    if (t?.decode_seconds > 0) parts.push((generated / t.decode_seconds).toFixed(2) + ' tok/s decode');
    parts.push(generated + ' output tokens');
    if (t) parts.push(t.decode_seconds.toFixed(2) + ' s decode');
    parts.push(u.prompt_tokens + ' prompt tokens (' + cached + ' cached)');
    if (t?.prefill_seconds > 0) {
      parts.push(t.prefill_seconds.toFixed(2) + ' s prefill');
      if (u.prompt_tokens > cached) parts.push(((u.prompt_tokens - cached) / t.prefill_seconds).toFixed(2) + ' tok/s prefill');
    }
    parts.push(elapsed.toFixed(2) + ' s total');
    return parts.join(' · ');
  }

  async function send(event) {
    event.preventDefault();
    if (el('chat-send').disabled) return;
    const body = requestBody();
    if (!body) return;
    const active = new AbortController();
    controller = active;
    const started = performance.now();
    const user = body.messages.at(-1);
    const userText = addMessage('You', user.content);
    const reply = addMessage('Assistant', '');
    let answer = '', thought = '', thinking = null, usage = null, finish = '';
    prompt.value = '';
    invalidatePreview();
    update();
    status.textContent = 'Waiting for the model…';
    const tick = setInterval(() => {
      status.textContent = (answer || thought ? 'Generating' : 'Waiting for the model') + ' · ' + ((performance.now() - started) / 1000).toFixed(1) + ' s';
    }, 400);
    log.scrollTop = log.scrollHeight;
    try {
      const headers = { 'Content-Type': 'application/json', Accept: 'text/event-stream' };
      if (state.apiKey) headers.Authorization = 'Bearer ' + state.apiKey;
      const response = await fetch('/v1/chat/completions', {
        method: 'POST', headers, signal: active.signal, credentials: 'same-origin', redirect: 'error',
        body: JSON.stringify({ ...body, stream: true, stream_options: { include_usage: true } }),
      });
      if (!response.ok) {
        if (response.status === 401) requireKey();
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.error?.message || 'Chat request failed (' + response.status + ').');
      }
      if (!response.body || !response.headers.get('content-type')?.includes('text/event-stream')) throw new Error('The service did not return a chat stream.');
      await readNextoraChatStream(response.body, chunk => {
        const follow = log.scrollTop + log.clientHeight >= log.scrollHeight - 45;
        const choice = chunk.choices?.[0];
        if (choice?.delta?.reasoning_content) {
          if (!thinking) {
            const details = document.createElement('details'), summary = document.createElement('summary');
            summary.textContent = 'Reasoning';
            thinking = document.createElement('p');
            details.append(summary, thinking);
            reply.before(details);
          }
          thought += choice.delta.reasoning_content;
          thinking.append(document.createTextNode(choice.delta.reasoning_content));
        }
        if (choice?.delta?.content) {
          answer += choice.delta.content;
          reply.append(document.createTextNode(choice.delta.content));
        }
        if (choice?.finish_reason) finish = choice.finish_reason;
        if (chunk.usage) usage = chunk;
        if (follow) log.scrollTop = log.scrollHeight;
      });
      if (!finish) throw new Error('The model did not finish its response.');
      history.push(user, { role: 'assistant', content: answer });
      const metrics = document.createElement('small');
      metrics.className = 'chat-measurements';
      metrics.textContent = measurements(usage || {}, (performance.now() - started) / 1000);
      reply.after(metrics);
      status.textContent = finish === 'length' ? 'Output limit reached.' : 'Reply complete.';
    } catch (err) {
      active.abort();
      const message = err.name === 'AbortError' ? 'Stopped.' : (err.message || 'Chat failed.');
      if (body.model === model) {
        if (answer || thought) {
          const note = document.createElement('small');
          note.className = 'chat-measurements';
          note.textContent = 'Partial reply · excluded from conversation context';
          reply.after(note);
        } else {
          reply.parentNode.remove();
          userText.parentNode.remove();
        }
        prompt.value = user.content;
      }
      status.textContent = message;
    } finally {
      clearInterval(tick);
      controller = null;
      update();
      prompt.focus();
    }
  }

  el('speed-stop').addEventListener('click', () => benchmark?.abort());
  el('speed-run').addEventListener('click', async () => {
    if (el('speed-run').disabled) return;
    benchmark = new AbortController();
    update();
    const settings = state.config.settings;
    const body = { model, messages: [{role: 'user', content: 'Write a Python function that returns the squares of the first twenty integers. /no_think'}], temperature: 0, max_tokens: Math.min(settings.max_tokens,128) };
    const headers = {'Content-Type':'application/json'};
    if (state.apiKey) headers.Authorization = 'Bearer ' + state.apiKey;
    let tokens = 0, seconds = 0;
    try {
      for (let i = 0; i < 3; i++) {
        el('speed-status').textContent = i ? `Measuring reply ${i} of 2…` : 'Warming up…';
        const response = await fetch('/v1/chat/completions', {method:'POST', headers, signal:benchmark.signal, body:JSON.stringify(body), redirect:'error'});
        const result = await response.json();
        if (!response.ok) throw new Error(result.error?.message || `Request failed (${response.status})`);
        if (i) { tokens += result.usage.completion_tokens; seconds += result.timings.decode_seconds; }
      }
      if (!(tokens > 0 && seconds > 0)) throw new Error('No generated tokens were available to measure.');
      const row = document.createElement('li');
      const label = document.createElement('strong'); label.textContent = `${(tokens / seconds).toFixed(2)} tok/s`;
      row.append(label, document.createTextNode(` · ${body.model.split('/').pop()} · ${state.status.runtime.execution.backend} · ${settings.ctx.toLocaleString()} context · ${settings.parallel} slot(s) · ${settings.vram_mib || 'auto'} GPU MiB · ${tokens} generated tokens in ${seconds.toFixed(2)} s`));
      el('speed-results').prepend(row);
      el('speed-status').textContent = 'Test complete. Change a setting and run again to compare.';
    } catch (error) { el('speed-status').textContent = error.name === 'AbortError' ? 'Test stopped.' : error.message; }
    finally { benchmark = null; update(); }
  });
  el('chat-form').addEventListener('submit', send);
  el('chat-stop').addEventListener('click', () => controller?.abort());
  el('chat-clear').addEventListener('click', () => {
    history.length = 0;
    log.replaceChildren();
    invalidatePreview();
    status.textContent = '';
    update();
  });
  el('chat-count').addEventListener('click', async () => {
    const body = requestBody();
    if (!body) return;
    const version = draftVersion;
    checking = true;
    update();
    preview.textContent = 'Counting tokens…';
    try {
      const count = await api('/api/context', { body, timeout: 60000 });
      if (version === draftVersion) preview.textContent = count.prompt_tokens.toLocaleString() + ' prompt + ' +
        count.reserved_output_tokens.toLocaleString() + ' output reserved · ' +
        count.remaining_tokens.toLocaleString() + ' of ' + count.context.toLocaleString() + ' tokens remaining';
    } catch (err) {
      if (version === draftVersion) preview.textContent = err.message;
    } finally { checking = false; update(); }
  });
  el('chat-files').addEventListener('change', async event => {
    const input = event.target, files = Array.from(input.files);
    if (!files.length) return;
    readingFiles = true;
    update();
    try {
      // Leave space in the 8 MiB HTTP body for JSON encoding and conversation history.
      if (files.reduce((sum, file) => sum + file.size, 0) > 4 * 1024 * 1024) throw new Error('Choose up to 4 MiB of text files at a time.');
      const blocks = [];
      const decoder = new TextDecoder('utf-8', { fatal: true });
      for (const file of files) {
        let text;
        try { text = decoder.decode(await file.arrayBuffer()); }
        catch { throw new Error(file.name + ' could not be read as UTF-8 text.'); }
        if (text.includes('\0')) throw new Error(file.name + ' contains binary data.');
        blocks.push('File: ' + file.name + '\n' + text);
      }
      prompt.value = [prompt.value, ...blocks].filter(Boolean).join('\n\n');
      invalidatePreview();
      status.textContent = 'File text added to your draft.';
    } catch (err) { status.textContent = err.message; }
    finally { readingFiles = false; input.value = ''; update(); }
  });
  prompt.addEventListener('input', () => { invalidatePreview(); update(); });
  limit.addEventListener('input', invalidatePreview);
  prompt.addEventListener('keydown', event => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey) && !el('chat-send').disabled) {
      event.preventDefault(); el('chat-form').requestSubmit();
    }
  });
  return { update };
}
