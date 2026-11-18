> Historical snapshot for v0.9.0. Published with path redaction and link repairs; the recorded test outcomes and limitations are retained. See [the evidence policy](../README.md).

# Forge9070 0.9.0 validation record

Validated 17 November 2026. The independent package is `forge-engine` 0.2.0;
the encompassing source project is Forge9070 0.9.0.

**The independent CPU/runtime, native API and explicit software-Vulkan gates
pass. RX 9070 XT production performance and full pretrained-model quality
remain unverified.** The validation adapter was llvmpipe, LLVM 19.1.7, Mesa
25.0.7, Vulkan 1.4.305. It was explicitly permitted for numerical testing and
remains rejected by default in ordinary execution. No AMD card, Windows build
or native WSL GPU execution was available here.

## Release gates

| Gate | Result | Evidence |
|---|---|---|
| Normal Rust regression invocation | 391 passed, 0 failed; 7 explicitly gated tests ignored in this invocation | [native-core-tests-0.9.0.json](../native-core-tests-0.9.0.json) |
| Native unit tests within that total | 73 passed | Same record |
| Ordinary Qwen full-vector parity | 8 passed | Same record; `tests/new_engine_parity.rs` |
| All-format packed Rust decoding/dot parity | 3 passed, covering 14 formats | Same record; `tests/new_quant_parity.rs` |
| Qwen3.8 CPU semantics/state parity | 7 passed | Same record; `tests/new_qwen38_parity.rs` |
| Optimized Linux standalone and frontend build | Passed, 99.59 seconds | [build-release-0.9.0.json](../build-release-0.9.0.json) |
| Release CLI and frontend dispatch | 16 checks passed; 0 skipped | [forge-engine-cli-0.9.0.json](../forge-engine-cli-0.9.0.json) |
| Release native HTTP/SSE/admission | 28 checks passed | [native-api-0.9.0.json](../native-api-0.9.0.json) |
| Four simultaneous Qwen Vulkan HTTP requests | All 4 matched CPU answers and usage; high-water active count 4 | [native-gpu-api-0.9.0.json](../native-gpu-api-0.9.0.json) |
| Vulkan packed/resident/budget/128K fixture gates | 3 explicitly enabled tests passed; zero validation errors | [native-vulkan-validation-0.9.0.json](../native-vulkan-validation-0.9.0.json) |
| Resident Qwen GPU full-state/error recovery | 2 explicitly enabled tests passed, 3 model configurations | [qwen38-vulkan-resident-parity-0.9.0.json](../qwen38-vulkan-resident-parity-0.9.0.json) |
| Diagnostic GPU-projection bridge | 1 explicitly enabled test passed; 5,601 actual GPU projections | [qwen38-vulkan-projection-parity-0.9.0.json](../qwen38-vulkan-projection-parity-0.9.0.json) |
| Official Qwen chat template | 6 Python-Jinja byte comparisons passed in 1 explicitly enabled Rust test | [native-template-0.9.0.json](../native-template-0.9.0.json) |
| Selected real checkpoint layout | All 1,224 tensor descriptors consumed, zero errors | [qwen38-real-header-layout-0.9.0.json](../qwen38-real-header-layout-0.9.0.json) |

The normal test total includes the existing compatibility engine. Some inherited
GPU/CUDA availability tests return early without hardware, so this is not a
count of GPU executions. Actual new Vulkan dispatch is established by the
separate explicit gates. The seven tests ignored in the normal invocation were
run separately: three native Vulkan gates, three Qwen Vulkan gates and one
external-template gate.

The aggregate command was:

```bash
cargo test --locked --workspace --lib --bins \
  --test inference --test new_engine_parity --test new_quant_parity \
  --test new_qwen38_parity --test serve --test model_switch --test ops --offline
cargo build --release --locked --offline -p forge-engine -p forge9070
```

[native-release-harnesses-0.9.0.json](../native-release-harnesses-0.9.0.json) retains final command receipts.
The standalone dependency tree in `artifacts/native-dependencies-0.9.0.txt`
contains no RustingStrata runtime dependency. Root compatibility commands keep
their existing dependency and attribution. The source/binary hashes are in
[native-source-0.9.0.json](../native-source-0.9.0.json).

## What the numerical comparisons establish

All fixtures are original, deterministic and untrained. Comparisons force the
same input tokens through the independent implementation and a separate
reference. They check all vocabulary logits and normalized hidden values,
rather than relying only on a deliberately preferred output token.

Ordinary Qwen CPU comparisons include multihead GQA, dense/MoE variants,
quantized rows, token bounds, resets, prefill and interleaved states. Their
per-value bound is `2e-5 + 2e-5 * abs(reference)`.

Qwen3.8 checks HC streams/output, split or combined beta/alpha projections,
DeltaNet convolution and recurrent matrices, gated attention K/V, PLE history,
and QSA raw keys. CPU formats include F32, F16, BF16, Q4_0 and Q8_0. Resident
GPU fixtures include F32 HC/PLE/QSA, Q4_0 with combined beta/alpha, and F32 dense
attention with PLE. Their per-value bound is
`3e-5 + 5e-5 * abs(reference)`. Full state is compared over the 20-token traces,
including a second interleaved sequence, reset and prefill replay.

All 14 packed formats are compared against the existing format decoder and dot
reference over deterministic encoded patterns. CPU decoded values match exact
F32 bit patterns. Dot reductions use documented numerical bounds. The Vulkan
packed/ordinary gate uses `0.002 + 0.0003 * abs(reference)`; the Qwen resident
state gate retains its tighter bound above. These are tolerance-based numerical
checks, not bit-identical CPU/GPU arithmetic or trained-model accuracy scores.

## Actual Vulkan execution, pressure and concurrent requests

The native engine gate used Khronos validation 1.4.309, with layer insertion
confirmed by the loader and synchronization validation enabled. It reported
zero validation/VUID/synchronization-hazard messages. Shader compilation and
SPIR-V validation succeeded; the checked-in SPIR-V is included in the source
package, so a normal user build does not require a shader compiler.

The four-layer Qwen fixture needed exactly **five submissions per warm token**.
Cold selected-expert matrices were uploaded in bounded staging batches; a
controlled cold token used nine submissions, including four batch uploads.
This validates grouping and counters, not a target-card speedup. The native
runtime still reports layer-boundary readbacks and no transfer/compute overlap.

A controlled 8-MiB allocation-pressure test forced **32 exact CPU expert
contributions and omitted none**. It checked the allocation ceiling, expert
results, cache eviction/reload and prefill protection of cached entries and
LRU order. Dense weights, transient/cached expert weights, states and staging
are included in the owned-allocation budget.

A separate synthetic fixture raised its declared context and allocated a
128K indexed sequence. Its 20-token logits and hidden values matched the CPU
reference; the tiny fixture used 39,447,056 live allocation bytes. This is
an address/dispatch/allocation check. It does not establish that a full-size
Qwen model fits 128K on a 16-GB card or produces accurate long-context answers.

The release HTTP test observed four simultaneously active Qwen requests,
**116 completed model tokens**, **928 GPU expert executions**, zero CPU expert
fallbacks, zero omitted experts and exact CPU-reference responses/usage. All
permits returned to idle. The software test took **117.94 seconds**. Its first
30-second client deadline was insufficient; the explicit 120-second diagnostic
passed. Both receipts are retained. No RX 9070 XT latency or throughput is
inferred from this CPU software driver.

## API, code-output and memory regressions

The release HTTP checks cover actual inference, JSON and SSE identities,
terminal finish reasons and `[DONE]`, optional usage, model/option validation,
exact prompt-plus-output overflow, function-call/result transport, literal code
examples, Unicode, and stops spanning generated tokens. They also cover exact
executed-prefix counts, changed-history reset, idle model switching and
restoration after replacement failure, one/four active slots, full queues,
queued/active disconnection, deadlines and permit cleanup.

Reasoning/tool responses are buffered for complete parsing before their SSE
deltas are emitted. There is no claim of incremental tool-argument streaming,
strict schema decoding, Responses/Anthropic compatibility, or certification
of a particular external coding client.

Testing found two concrete numerical/output defects and the release includes
both fixes:

- A GPU RMS sum could overflow and become an apparently valid zero output.
  The guard now rejects the submitted computation, poisons that sequence until
  reset, clears recurrent/PLE state on reset, and then matches normal execution.
- `tokenizers` 0.20's streaming helper underflowed its history index on a Unicode
  regression. The native path now uses an independently implemented incremental
  byte decoder. Its packed token-byte table is shared; each stream retains at
  most three incomplete UTF-8 bytes. All 256 single-byte patterns, all 65,536
  two-byte patterns, 48 mixed valid/invalid corpora, merged/added/control tokens
  and invalid-ID recovery match full decoding. Per-token full-answer decoding
  and emitted-prefix copying have been removed. One final decode flushes any
  incomplete suffix.

The failing pre-fix regression record is preserved separately; the final
aggregate run and release API checks pass. The official Qwen template's six
text/tool-history cases also match Python Jinja using Transformers JSON
spelling/spacing defaults. Template source SHA-256 is
`af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375`.

Memory/storage checks cover shard consistency and bounds, missing shards,
immutable leased expert identity, hard payload capacity, parallel misses,
atomic triplet warmup, direct/buffered I/O fallback, prefill protection, required
metadata and overflow. Available-memory tests check one/four-slot planning and
clamp an explicit cache ceiling to actual expert payload before RAM fit checks.
Windows-specific direct I/O is implemented but was not executed on this Linux
host. RAM/VRAM plans remain estimates/owned limits, not reservations against
other programs.

## Real checkpoint header evidence and Avalon loading policy

The bounded inspection used the two IQ3_S shards of
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S),
revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`.
At most 16 MiB per shard was fetched; the parsed headers were 11,024,355 and
173 bytes. Prefix downloads included post-header bytes, but tensor payloads
were not interpreted. The full checkpoint was not downloaded or executed.

The actual descriptors contain 50,292,326,400 expert bytes, 28,800,138,240 lazy
PLE bytes and 4,514,173,440 other weight bytes. All 13 quantization IDs in those
descriptors are supported by the new 14-format implementation. The actual GGUF uses PLE layer index 1; this was checked against its tensor
presence and dimensions instead of inferring an offset from publisher metadata. Header acceptance is separate from
weight-value validation, memory fit and inference quality.

The independent loader retains the [Avalon](https://github.com/freriadoc/Avalon)
findings described in [docs/AVALON-COMPARISON.md](../../AVALON-COMPARISON.md):
bounded owned expert RAM, an active GPU expert tier, lazy PLE access, protected
leases and prefill-aware caching. It does not port Avalon's registered host
arena, shipped expert ranking, CPU/GPU overlap, MTP, IQ3_KS or elastic KV loans.
It computes every selected expert and preserves the checkpoint's quantization.

## Remaining target acceptance

No real 9070 XT tokens/second, real-model code accuracy, full-model 1/2/4-request
memory fit, long-context quality, Windows execution or native WSL Vulkan
support was established by this environment. Those need the user's actual
model files, driver and hardware. Begin with one request and a representative
code prompt, then compare two/four requests with identical model/context and
quality checks. Four active slots are supported by the scheduler and independent
states; their practical context and speed depend on shared device memory and
bandwidth. `llvmpipe` is not a workaround for exposing the physical AMD GPU.

The complete native commands and limits are in
[docs/NATIVE-ENGINE.md](NATIVE-ENGINE.md). Prior 0.8 validation is preserved
under [v0.8.0](../v0.8.0); older evidence is retained as history and not presented
as a fresh 0.9 run.
