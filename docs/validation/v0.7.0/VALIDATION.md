> Historical snapshot for v0.7.0. Published with path redaction and link repairs; the recorded test outcomes and limitations are retained. See [the evidence policy](../README.md).

# Forge9070 0.7.0 validation record

Uses RustingStrata by Vasyl Trefilov. Validation ran on 17 November 2026 UTC.
This release adds model-aware capacity selection, bounded request admission,
queue visibility and coding-client API corrections. It does not add Qwen3.8
multi-sequence GPU execution.

## Observed results

| Check | Observed result |
| --- | --- |
| Rust unit suites | **255 passed, 0 failed**: 38 frontend and 217 engine |
| Vendored serving suite | **28 reported passed**; 27 exercised their assertions, one hardware smoke returned early without `STRATA_MODEL` |
| Optimized Linux x86_64 release build | Passed with `--release --locked --offline` |
| Coding-client API on the release binary | **25 HTTP/API checks passed** |
| Request scheduling on the release binary | **10 HTTP checks passed** |
| Existing CPU generation / HTTP / SSE smoke | **9 checks passed** |
| Existing concurrency and configuration regression | **9 checks passed** |
| Code-context CLI / Git / HTTP regression | **33 commands and assertions passed**, plus a captured server log |
| Actual UI helper source and complete script syntax | **13 Node checks passed** with minimal element/storage stubs |
| Real browser rendering | Not run; the previous Chromium attempt was blocked before page creation by socket EPERM |
| Windows / RX 9070 XT / full Qwen3.8 | Not run in this environment |

Inherited CUDA availability tests also return early when CUDA/NVRTC is absent.
The Rust harness total does not establish CUDA or Vulkan model execution.
No token-rate, four-sequence Qwen, or particular external coding-client claim
is made from these results.

## Admission and scheduling

The admission module has 13 focused tests for FIFO reservations, immediate-only
queue mode, exact deadline boundaries, reserved-ticket expiry, cancellation and
release races, capacity shrink/recovery, Condvar wakeup and RAII ownership.
Policy tests distinguish auto selection from strict explicit counts and reject
inconsistent queue/connection budgets.

Real HTTP tests expand the existing untrained CPU Qwen3MoE fixture to four
layers in a temporary directory. They synchronize against status and response
headers while real decoding runs, continuously drain streaming responses, and
inspect generation-start logs. They establish:

- Authentication and zero capacity when no model is loaded.
- CPU auto capacity of one, with waiting requests counted separately.
- Queue-full HTTP 429 and timeout HTTP 503 with explicit codes and Retry-After.
- Status responsiveness during inference and a full queue.
- FIFO admission, queued disconnect cleanup, no generation for cancelled or
  expired queued work, and successful later generation after cleanup.

The queue is connection-bound process memory, not durable background work.
Capacity buffers are reserved at load. Auto selects compatible capacity;
it does not measure available VRAM or find an optimal throughput setting.
Qwen HC/PLE/QSA remains restricted to one generation sequence.

## Coding API and prompt correctness

The 25 API checks execute actual CPU inference with synthetic models. One
fixture deliberately emits a complete function-call token, allowing the real
JSON/SSE response path and subsequent tool-result turn to be tested. This is
transport verification, not evidence of useful real-model tool selection.

Coverage includes model discovery/identity, developer instructions, output
limit aliases, stable SSE IDs, optional final usage chunks, stop/length/tool-call
finish reasons, function IDs and JSON arguments, and explicit unsupported-option
errors. Image preparation without a usable projector now returns JSON errors
before SSE for both streaming and nonstreaming requests; count_tokens has the
same behavior and all permits/counters are released afterward.

The original raw-template golden fixture bytes remain unchanged. API-only
function validation is separate from those raw formatter comparisons. Tool
results are validated by ID and reordered into the assistant's original call
order before rendering. Leading system/developer messages are retained; later
instruction injection through unsupported template positions is rejected.

OpenAI Chat Completions is supported as documented in docs/CLIENT-API.md.
Forced tool choice, strict schema enforcement and the Responses API remain
unsupported. The existing Anthropic Messages adapter is a documented subset.
Clients execute tools themselves; this server never executes returned calls.

## Commands actually exercised

```bash
cargo test --locked --workspace --lib --bins --test serve --offline -- --nocapture
cargo build --release --locked --offline
python3 tests/api-coding-client.py --binary target/release/forge9070
python3 tests/scheduling_api.py --binary target/release/forge9070
python3 tests/smoke.py --binary target/release/forge9070
python3 tests/concurrency_cli.py --binary target/release/forge9070
python3 scripts/validate-code-context.py --exe target/release/forge9070 --output artifacts/code-context-integration-0.7.0.json
node scripts/validate-code-ui-helpers.cjs
```

Complete results, scope, commands, logs and final source hashes are in
[validation.json](validation.json) and its referenced versioned records. Captured logs
are excluded from assertion totals. The previous release's record is preserved
in [v0.6.0/](../v0.6.0). Historical failed development runs are labelled as such;
the final suites listed above are passing.

## Remaining validation on the target machine

A working hardware Vulkan adapter must be confirmed independently of WSL's
`/dev/dxg`. Test Windows compilation, actual checkpoint loading, memory use,
streaming and model tool quality on the 9070 XT. Eligible multi-slot GPU models
need hardware concurrency/performance measurement; Qwen3.8 still needs backend
state/dispatch development before that path can be enabled. The source archive
contains no Windows executable or pretrained model weights.
