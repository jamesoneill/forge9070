# Coding clients and the local API

**Scope:** this guide documents the existing compatibility application. For the
independent Forge 0.9 runtime, its Vulkan/CPU settings, coding API and queue,
use [NATIVE-ENGINE.md](NATIVE-ENGINE.md). Flags and endpoints differ.

Forge serves an OpenAI-compatible **Chat Completions** API at
`http://127.0.0.1:8080/v1`. A coding client can send several independent requests;
Forge admits work up to the loaded model's capacity and queues the excess.
**Qwen3.8-Flash-Next still has one active generation sequence in this build.**
Auto mode does not add the missing Qwen HC/PLE/QSA multi-sequence backend.

For admission limits, memory behavior and retry handling, see
[REQUEST-SCHEDULING.md](REQUEST-SCHEDULING.md). Uses RustingStrata by Vasyl Trefilov.

## Start the Windows Vulkan server

Run this in PowerShell from the project directory after building the Windows
executable. Enter the actual path to the first GGUF shard when prompted. Set
the same API key in your client; replace the example key before use.

```powershell
$Forge = (Resolve-Path '.\target\release\forge9070.exe').Path
$Model = (Resolve-Path -LiteralPath (Read-Host 'Full path to the first model GGUF shard')).Path
$env:STRATA_API_KEY = 'replace-with-your-own-local-api-key'

& $Forge serve --model $Model --preset code-long `
  --parallel auto --max-parallel 4 `
  --queue-capacity 32 --queue-timeout 120 --max-waiting 64 `
  --host 127.0.0.1 --port 8080
```

`code-long` provides a 32,768-token context and an 8,192-token output cap. Use
`--preset code` for a 16,384-token context and 4,096-token output cap. Context
includes the rendered prompt, tools, history, reasoning and generated answer;
the output cap is not additional context. Explicit flags override the preset.

The Windows process uses Windows Vulkan. A WSL client must use an address that
actually reaches that Windows process; do not assume localhost forwarding in
every WSL networking configuration. See [WSL.md](WSL.md) for the launcher and
environment distinctions. CPU mode requires explicit `--cpu` and generates
one reply at a time.

When `STRATA_API_KEY` is set, send `Authorization: Bearer YOUR_KEY`. The default
loopback server can run without a key; binding outside loopback requires a key.
`/health` is an unauthenticated readiness route. Other authentication and admin
rules still apply to their respective endpoints.

## Configure an OpenAI-compatible client

| Client setting | Value or behavior |
|---|---|
| Provider/API style | OpenAI-compatible **Chat Completions** |
| Base URL | `http://127.0.0.1:8080/v1` on the server's machine |
| API key | The server's `STRATA_API_KEY`, when configured |
| Model | Exact `id` returned by `GET /v1/models` |
| Streaming | Supported through Chat Completions SSE |
| Tools | Function tools, automatic choice; the client executes tools |
| Output limit | `max_completion_tokens` or `max_tokens`, within the server cap |
| Request timeout | Longer than possible queue wait, prefill and generation combined |
| Concurrent requests | Client may submit several; inspect `/v1/status` for actual capacity |

This is a protocol configuration guide, not a claim that a particular client
release has been tested. In particular, an application called **agentlooper**
needs to support these settings and this API subset; no agentlooper-specific
integration has been verified.

Do not configure the Responses API or the legacy text Completions API:
`POST /v1/responses` and `POST /v1/completions` return **501** with
`error.code = "unsupported_endpoint"`.

## Discover the loaded model and capacity

From a shell where the base address reaches Forge, these commands return JSON.
The key must match the server's key; omit the authorization header when running
a local server without authentication.

```bash
curl -i -H "Authorization: Bearer $STRATA_API_KEY" \
  http://127.0.0.1:8080/v1/models

curl -i -H "Authorization: Bearer $STRATA_API_KEY" \
  http://127.0.0.1:8080/v1/status
```

In PowerShell, use `curl.exe` and `$env:STRATA_API_KEY`, or:

```powershell
$Headers = @{ Authorization = "Bearer $env:STRATA_API_KEY" }
$Catalog = Invoke-RestMethod -Uri 'http://127.0.0.1:8080/v1/models' -Headers $Headers
$Catalog.data | Select-Object id
Invoke-RestMethod -Uri 'http://127.0.0.1:8080/v1/status' -Headers $Headers
```

`/v1/models` describes the currently loaded/loading model; it is not a search
of every Hugging Face repository or a list of every downloaded file. Check
`/v1/status` for readiness. Use the returned model ID verbatim, including any
filename or path-like components. A Hugging Face repository name is not a
substitute unless that exact string is the returned ID. An explicit ID that
does not match the loaded model gets **409**, including when a queued request
encounters a model switch. Omitting an ID chooses the currently loaded model;
coding clients should name it to prevent accidental model changes.

The status JSON includes `status`, `model`, `backend`,
`context_tokens_per_request`, `max_output_tokens`, `generating_requests`,
`concurrency`, `admission`, and supported API metadata. Read
`admission.capacity` for current admission capacity. `concurrency` explains
the configured policy and effective choice. See the scheduling guide for the
distinction between admitted, reserved, queued and generating requests.

## A Python example with no SDK dependency

This example discovers the exact loaded model ID and makes one JSON Chat
Completions request using the Python standard library. Set `STRATA_API_KEY`
in the client's environment when authentication is enabled. Set
`FORGE_BASE_URL` if the server is not on the same localhost address.

```python
import json
import os
import urllib.error
import urllib.request

base = os.environ.get("FORGE_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
key = os.environ.get("STRATA_API_KEY", "")

def request(route, payload=None, timeout=900):
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{base}/{route}", data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Forge HTTP {error.code}: {body}") from error

state = request("status", timeout=30)
if state["status"] != "ready":
    raise RuntimeError(f"Forge is not ready: {state['status']}")
models = request("models", timeout=30)["data"]
if not models:
    raise RuntimeError("Load a model in Forge before sending a chat request")

reply = request("chat/completions", {
    "model": models[0]["id"],
    "messages": [
        {"role": "system", "content": "Explain code carefully and identify uncertain assumptions."},
        {"role": "user", "content": "Explain how to test cancellation in a Rust task queue."},
    ],
    "max_completion_tokens": 2048,
    "stream": False,
})
print(reply["choices"][0]["message"].get("content", ""))
```

The 900-second timeout is an example, not a measured upper bound. This minimal
example deliberately leaves retries to the caller. A production client should
apply the bounded retry rules in the scheduling guide.

## JSON, streaming and request fields

Send `Content-Type: application/json`. The main route is
`POST /v1/chat/completions`. Supported core fields include:

- `model`, `messages`, `stream` and `stream_options.include_usage`.
- `max_completion_tokens` or `max_tokens`. If both are present, they must agree.
- `temperature`, `top_p`, `top_k`, `seed` and `stop`; only `n: 1` is supported.
- Function `tools`, `tool_choice: "auto"` or `"none"`, assistant `tool_calls`,
  and tool messages with matching `tool_call_id`.
- `chat_template_kwargs.enable_thinking` and
  `chat_template_kwargs.preserve_thinking`, both booleans.

Leave output and sampling fields absent to use the server defaults. The server
rejects an explicit output limit that exceeds its cap or remaining context;
it does not silently cut the source/history to make the request fit.
`POST /api/context` previews the rendered text-and-tool request budget without
inference. Its successful preview response can contain `fits: false`; inspect
that field. It can wait behind model work, so it is not a busy-server monitoring
endpoint. See [LARGE-CODE-PROJECTS.md](LARGE-CODE-PROJECTS.md).

With `stream: true`, read Server-Sent Events containing
`chat.completion.chunk` objects and finish at `data: [DONE]`. Accumulate
`choices[].delta.content`, any `reasoning_content`, and tool-call deltas.
When `stream_options.include_usage` is true, the final usage event has
`choices: []`; do not assume every event contains `choices[0]`. Preserve earlier
reasoning when continuing a conversation with a template that uses it.

## Tools belong to the client

Forge generates function-call suggestions. It does not execute shell commands,
edit repositories, search the web or run the functions declared in `tools`.
The coding client must validate the proposed arguments, execute its own tools
under its own permissions, and send the results back.

For each tool turn, retain the assistant's `tool_calls`, then append a `tool`
message for each call with the matching `tool_call_id`. All outstanding calls
need results before requesting the next completion. IDs must be unique within
the call group; argument strings must contain JSON objects. The parser matches
results by ID and restores call order for model templates that do not retain
the IDs. Put system/developer instructions before the conversation; leading
instruction messages are merged, and later system/developer messages are
rejected instead of silently discarded.

Schemas guide the prompt; **strict schema enforcement and forced function
generation are not implemented**. Do not interpret tool support as guaranteed
valid arguments or successful execution.

## Explicitly unsupported Chat Completions options

These requests return **400** rather than pretending to apply the option:

| Option | Supported alternative |
|---|---|
| `tool_choice: "required"` or a named forced function | `"auto"` or `"none"` |
| `parallel_tool_calls: false` | Omit it or allow `true`; the model may emit several calls |
| Function schema `strict: true` | Omit it or use `false`; validate arguments in the client |
| JSON/schema `response_format` | Omit it or use exactly `{"type":"text"}` |
| `logprobs: true`, `top_logprobs > 0` | Omit them or use neutral defaults |
| Output `modalities` other than `["text"]` | Text output |
| `store: true` | Keep conversation history in the client |
| Legacy `functions` / `function_call` | Function `tools` / `tool_calls` |
| `audio`, `prediction`, `reasoning_effort`, `verbosity`, `web_search_options` | Omit them |
| Nonneutral penalties or sampling bias | Omit unsupported settings |
| `stream_options.include_obfuscation: true` or unknown stream options | `include_usage`; obfuscation omitted or false |
| Unknown `chat_template_kwargs` | Only the two thinking booleans listed above |

Accepted neutral values for unsupported sampling controls are
`presence_penalty: 0`, `frequency_penalty: 0`, `repetition_penalty: 1`,
`min_p: 0`, `typical_p: 1`, and an empty `logit_bias` object. Unknown message
content parts are rejected. These Forge coding/model profiles are text paths;
generic parser support for some image fields does not establish vision support
for the selected model.

## Other endpoints and limits

The existing `POST /v1/messages` and `POST /v1/messages/count_tokens` routes
provide an **Anthropic Messages subset** through request translation. They are
not full Anthropic API compatibility or a guarantee for a particular coding
client. In particular, forced tool choices still reach the same unsupported
operation, and provider-hosted tools are not supplied. Prefer Chat Completions
when the client offers that option.

One checkpoint is loaded at a time. Requests naming another checkpoint do not
load it automatically. Model management is an admin operation; switching the
original/uncensored checkpoint changes the shared loaded model for subsequent
requests. API compatibility tests using the small CPU fixture do not establish
Qwen inference accuracy, actual tool-use quality, Windows Vulkan operation or
RX 9070 XT token rates.

Implementation references: [Chat Completions parsing and translation](../vendor/rusting-strata/src/serve.rs),
[conversation/tool validation](../vendor/rusting-strata/src/chat.rs), and
[HTTP routes and response streaming](../vendor/rusting-strata/src/cli.rs).
