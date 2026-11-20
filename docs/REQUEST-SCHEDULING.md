> **Historical compatibility guide:** The previous Strata-based application
> and port 8080 have been retired from Nextora. The original documentation
> below is retained solely as migration reference and may describe commands,
> cache settings or API behaviour that the native Rust service does not offer.
> For the active runtime, use [the native guide](NATIVE-ENGINE.md) and
> [migration details](COMPATIBILITY.md).

# Request capacity and queue behavior

**Scope:** this guide documents the existing compatibility application. For the
independent Forge 0.9 runtime, its Vulkan/CPU settings, coding API and queue,
use [NATIVE-ENGINE.md](NATIVE-ENGINE.md). Flags and endpoints differ.

Forge accepts multiple clients, admits generation up to the loaded model's
supported capacity, and holds excess requests in a bounded in-memory queue.
**Qwen3.8-Flash-Next remains limited to one active generation sequence.**
The Qwen HC/PLE/QSA changes required for four simultaneous sequences have not
been implemented. Four clients can submit requests to that model; those
requests are served one at a time with the rest waiting.

See [CLIENT-API.md](CLIENT-API.md) for client setup and request examples.

## Capacity policy

The default server policy is `--parallel auto --max-parallel 4`.

| Configuration | Capacity selected at model load |
|---|---|
| Auto, supported GPU model without single-sequence restrictions | The `--max-parallel` ceiling, from 1 to 4 |
| Auto, CPU | 1 |
| Auto, model with hyper-connections, per-layer embeddings or QSA | 1 |
| Auto with otherwise supported MTP enabled | 1 |
| Explicit `--parallel 1`, `2`, `3` or `4` | Exact requested count if compatible; otherwise an error |

`--max-parallel` limits auto only. For example, explicit `--parallel 4
--max-parallel 2` still requests four slots. Explicit CPU/multi-slot,
MTP/multi-slot and Qwen/multi-slot conflicts are errors. Auto does not disable
an incompatible MTP request or add an unsupported MTP head. Non-server
commands run one sequence; their default auto setting does not enable batching.

**Auto chooses compatible capacity, not the fastest or largest allocation
that fits the current GPU.** Weights are shared, but context and recurrent
state are allocated for the chosen slots when the model loads. Occupied slots
follow incoming demand; empty slots do not return their reserved state memory
to the expert cache. There is no continuous VRAM growth/shrink controller,
allocation-failure fallback to fewer slots, or automatic throughput benchmark
behind this setting.

Lower the ceiling or explicitly use one slot when additional context state
would leave too little room for experts. For example:

```powershell
& $Forge serve --model $Model --preset code --parallel auto --max-parallel 2
```

That command chooses at most two slots on a supported GPU model, and still
chooses one for Qwen3.8. `code` is 16,384 context / 4,096 maximum output;
`code-long` is 32,768 / 8,192. Context is **per request**, including output.
The browser's explicit code-preset action continues to stage fixed one-slot
settings. Increasing concurrency does not multiply the card's compute, host
RAM bandwidth or expert cache: per-request speed and aggregate throughput need
measurement on the real model and machine.

## Separate queue and connection limits

| Flag | Default | Meaning |
|---|---:|---|
| `--parallel` | `auto` | Compatible generation capacity policy |
| `--max-parallel` | 4 | Ceiling for auto, valid 1–4 |
| `--queue-capacity` | 32 | Waiting generation requests beyond occupied/reserved capacity, valid 0–4096 |
| `--queue-timeout` | 120 | Seconds from enqueue until admission is claimed, valid 1–86,400 |
| `--max-waiting` | 64 | Total accepted connections, including readers, admitted requests, queued requests and status/control clients |

The total connection limit must be at least queue capacity plus the maximum
configured generation capacity. Validation reports conflicting values instead
of silently increasing either limit. Auto validation conservatively allows for
a later compatible model using the full ceiling.

`--queue-capacity 0` disables waiting: a request needs a free capacity slot or
gets a queue-full response. The queue timeout must still be positive. Raising
queue capacity adds waiting clients, not generation slots or model throughput.
The default connection limit leaves some space beyond the generation queue for
other requests, but monitoring remains subject to connection and resource
limits; it is not guaranteed under arbitrary connection saturation.

Queued requests are promoted **FIFO at the admission gate**. A short-lived
reservation prevents a later arrival from taking a slot already offered to an
older waiter. An unclaimed reservation keeps its original deadline. Once a
request claims admission, its preparation, prefill and generation are outside
the queue timeout. Completion order need not match arrival order: requests can
have different lengths and the GPU scheduler interleaves supported sequences.

The multi-slot backend shares model weights and combines ready decode work in
groups of up to four. Long prompts are processed in chunks, with other active
sequences able to decode between chunks. This is separate from the outer queue;
an admitted request can still spend time preparing or awaiting scheduler work.

## Status and what the counters mean

`GET /v1/status` is available to authenticated client keys, when authentication
is configured. It reports the current model, readiness, effective concurrency
policy, context/output limits, generating requests and an `admission` object:

| Admission field | Meaning |
|---|---|
| `capacity` | Current maximum admitted/reserved generation requests; zero while unavailable |
| `active` | Claimed permits, including preparation, prefill, generation and response cleanup |
| `reserved` | Slots offered to waiting request handlers but not yet claimed |
| `queued` | Requests still waiting beyond available capacity |
| `queue_capacity` | Configured maximum waiting requests |
| `queue_timeout_seconds` | Configured admission deadline duration |
| `rejected_full_total` | Queue-full refusals since server start |
| `timed_out_total` | Admission timeouts since server start |
| `cancelled_total` | Queued or reserved requests cancelled before execution |

`active + reserved` consumes capacity. A queued request does not count as
active generation. `generating_requests` comes from the narrower model-work
statistics and can differ from admission `active`. Counters are snapshots, not
an event log or proof that every admitted request is executing a GPU kernel at
that instant. The browser's Requests panel shows these categories separately.

The following monitoring/discovery routes are answered independently of the
generation loop, so a long response does not itself hold them up:

| Route | Access and purpose |
|---|---|
| `GET /health` | Unauthenticated readiness; 200 when a model is ready, 503 when absent/loading |
| `GET /v1/status` | Client authentication when configured; capacity, queue and API capabilities |
| `GET /v1/models` | Client authentication when configured; current model ID and context metadata |
| `GET /api/stats` | Admin; detailed statistics including admission/concurrency |
| `GET /metrics` | Admin; Prometheus-style statistics and admission counters |

These routes still need an accepted connection. Other model-control, loading,
context-preview and token-counting operations can wait behind model work; do
not use them as a busy-server liveness probe. A ready health response means a
model is loaded, not that a generation slot is currently free.

## Errors, timing and retries

| Condition | HTTP / error detail | Client action |
|---|---|---|
| Waiting queue is full | 429 / `queue_full`, `Retry-After: 2` | Bounded backoff or reduce client parallelism |
| Admission deadline expired | 503 / `queue_timeout`, `Retry-After: 2` | Retry later, reduce load, or explicitly increase the deadline |
| No loaded model | 503 / `model_not_ready` | Wait for readiness or load a model |
| Explicit model ID differs from the loaded model | 409 / `model_not_loaded` | Rediscover/check the intended model; do not silently change models |
| Invalid request, unsupported option or context/output overflow | 400 | Correct the request; repeating it unchanged will not help |

Queue identifiers are returned in `error.code`; model mismatch is identified
by `error.type = "model_not_loaded"`.

Connection or per-key limits can also produce 503/429 responses with different
error details and `Retry-After` values. Inspect the status, error body and
header instead of interpreting every 503 as a queue timeout.

Generation responses expose `X-Forge-Queue-Ms` for **outer admission wait
only**. It does not include all HTTP reading time, tokenizer work, time inside
the model scheduler, prefill, token generation or transport latency. Use it
with client timing and server statistics; do not label it total first-token
latency. Preserve `X-Request-ID` when diagnosing a request.

Set client timeouts above the expected queue wait **plus** preparation,
prefill and generation time. The 120-second queue deadline is not a completion
deadline. Four cold, large prompts can take substantially longer than a short,
already-cached prompt; no wall-clock upper bound is promised.

Retry only a bounded number of times, honor `Retry-After`, add backoff/jitter,
and stop when the caller's overall deadline expires. Before streaming starts,
an explicit `queue_full` or `queue_timeout` response can be retried without
having started that generation. Once a stream starts, or a transport failure
leaves its outcome uncertain, **do not blindly replay the request**: there is
no idempotency or exactly-once tool-execution guarantee. Reconcile partial
output/tool actions in the client first.

## Lifetime and model switching

The queue belongs to live HTTP connections and process memory. It is not a
durable job service: there are no job IDs to resume after restart, persisted
queued work, or detached background results. A disconnected queued client is
cancelled when detected, releasing its place/reservation. Active disconnects
also require normal inference/response cleanup; they are not a guarantee of
instant GPU interruption.

One checkpoint is loaded at a time. Auto capacity is recomputed on each model
load/reload, and unavailable/loading state is reflected in status. A queued
OpenAI request naming the previous model is rejected rather than answered by
a different checkpoint. Keep the selected model ID explicit and let the client
decide how to handle that change.

This feature adds bounded admission, compatible capacity selection and client
visibility. It does not establish four-way Qwen decoding, new model accuracy,
RX 9070 XT performance or native WSL Vulkan availability. Those require the
corresponding backend work and hardware validation.

Implementation references: [capacity policy](../vendor/rusting-strata/src/serving_policy.rs),
[admission controller](../vendor/rusting-strata/src/admission.rs), and
[HTTP/server integration](../vendor/rusting-strata/src/cli.rs).
