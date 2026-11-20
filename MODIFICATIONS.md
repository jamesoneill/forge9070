> **Historical archive:** This document records modifications to the
> former Strata-derived compatibility distribution. The separate runtime and
> vendored source were removed from the current ForgeAI branch. See
> [runtime consolidation](docs/RUNTIME-CONSOLIDATION.md) for active code and
> [historical Strata licence](LICENSE-RUSTINGSTRATA-HISTORICAL.md).

# Forge9070 changes and provenance

Uses RustingStrata by Vasyl Trefilov.

This is a modified distribution of RustingStrata 1.2.4, with a new Rust
application tailored to an AMD Radeon RX 9070 XT. It is not an official release
or endorsement by RustingStudio, AMD, Microsoft, or Avalon.

The sections below are a chronological change record. Version 0.4 supersedes
the earlier Q8 default and user-facing approximate expert-tail option.

The full upstream source is retained in `vendor/rusting-strata`. `UPSTREAM.json`
records the exact upstream commit and downloaded source archive SHA-256. The
root `Cargo.lock` pins the resolved dependency versions. The root and vendored
`LICENSE.md` contain RustingStrata License 1.0, including its conditions on
redistribution and sale of the engine itself. Third-party files retain their
original notices. This distribution does not relicense them.

## Significant modifications

### Version 0.9.1: native service configuration UI

- Added an embedded browser control panel to the independent Rust server.
  It can start without a model and discover supported complete GGUF files in
  a chosen folder, including Windows F: or the corresponding WSL mount.
- Added idle-only runtime configuration, model loading/unloading, live status
  and explicit JSON profile persistence. API keys stay outside saved profiles.
- Kept model validation, bounded MoE memory policy, queued admission and
  transactional replacement in the native engine. The new UI does not dispatch
  to the older compatibility engine or loosen the software-GPU guard.
- Added focused configuration/API and desktop/mobile browser checks using
  synthetic CPU fixtures. Existing GPU numerical evidence remains recorded
  under 0.9.0; no new RX 9070 XT performance claim is implied.


### Version 0.9: independent GPU execution and MoE serving

- Implemented Qwen3.8/qwen4exp HC, DeltaNet, QSA, PLE and routed/shared experts
  in new Rust code, with independent sequence state and numerical comparisons.
- Added our own Vulkan kernels and graphs. Ordinary Qwen3 keeps its model on
  device; Qwen3.8 uses fixed dense matrices, lazy embedding rows and a bounded
  packed expert cache. All selected experts execute, with counted CPU expert
  execution when an individual device working set cannot fit.
- Implemented complete sharded GGUF loading and 14 stored quantization formats.
  GGML codebook tables are attributed with their original MIT notice in
  `crates/forge-engine/GGML-NOTICE.md`; no Strata shader was relabelled.
- Reimplemented the Avalon-derived owned expert-RAM policy, direct offset reads,
  immutable leases, bounded expert-major warmup and prefill protection in the
  independent package. Available RAM and concurrent state inform its budget.
- Added a native Chat Completions API, model-embedded Jinja templates, tool-call
  transport, prefix/state reuse, idle-only model switching and one-to-four
  active requests with a bounded FIFO queue and cancellation.
- Added CPU and software-Vulkan full-state gates, API transport/admission tests,
  source receipts and documented target-card acceptance limits. The standalone
  package has no Strata execution dependency. Legacy root commands retain
  their prior implementation and notices for compatibility.
- No target-card throughput, full-checkpoint accuracy, transfer/compute overlap,
  MTP, quantized KV or continuous batching is claimed for the new native engine.

### Version 0.8: independently implemented engine foundation

- Added `crates/forge-engine`, a separately licensed, independently implemented
  Rust runtime with no RustingStrata execution dependency. Its standalone binary
  and `forge9070 engine` entry point execute only its documented CPU paths.
- Added bounded GGUF parsing, mapped quantized tensor operations, text BPE,
  ordinary Qwen3 and Qwen3MoE forward computation, independent per-sequence KV
  state and reset, and a measured prefill/decode benchmark.
- Added full-vector numerical comparisons against the existing engine as a test
  reference. The original runtime and its license remain unchanged; no upstream
  crate or GPU shader has been relabelled as first-party code.
- Existing application and GPU/API commands still use RustingStrata. The new
  core has no Qwen3.8, Vulkan, sharded loading or HTTP serving capability yet.
  Those remain explicit gates before a production engine replacement.

### Version 0.7: demand-based admission and coding-client API

- Added `--parallel auto` and `--max-parallel` with a shared, deterministic
  compatibility policy. Auto resolves CPU, MTP and Qwen HC/PLE/QSA to one
  supported sequence; eligible GPU models use the configured ceiling up to
  four. Fixed counts remain strict. State memory is reserved at model load;
  this does not add live VRAM resizing, allocation fallback or Qwen multi-slot
  kernels.
- Added a bounded FIFO generation queue, independent of the total connection
  bound. Waiting requests have deadlines and connection cancellation; RAII
  permits retain and release execution capacity across handler/error paths.
  Model transitions close admission until a model is ready. Full queues return
  429 and expired waits return 503, both with machine-readable codes and
  `Retry-After`.
- Added authenticated `/v1/status`, queue counters in admin stats/Prometheus,
  request/outer-queue timing headers, and browser controls/reporting that keep
  chosen capacity, active requests and waiting work distinct.
- Hardened the existing OpenAI Chat Completions API for coding clients:
  leading developer instructions, validated tool-result IDs and ordering,
  explicit unsupported-option errors, correct optional SSE usage events and
  JSON preparation errors before SSE starts. Retained the existing limited
  Anthropic Messages adapter. Tool execution remains the client's job.
- Added real CPU HTTP tests for admission, overflow, expiry, disconnect/FIFO
  behavior and streaming function-call round trips using synthetic fixtures.
  Added client and scheduling guides. No real Qwen/9070 XT throughput or
  external coding-client compatibility is claimed by these tests.

### Earlier changes

1. Added a standalone `forge9070` Rust binary with a profile for this card:
   explicit Vulkan selection, no automatic CPU fallback, 4K context, 256-token
   prefill chunks, Q8 KV, MTP disabled initially, and memory margins.
2. Added read-only hardware diagnostics, including Vulkan capabilities, real
   adapter identity, backend index mapping, memory-budget estimates, WSL
   identification, and observed ROCm metadata. Discovery is labelled separately
   from actual compute validation.
3. Added metadata preflight for the primary Qwen/Flint path and unsupported
   quantization rejection, plus commands for inspection and measured benchmarks.
4. Exposed the upstream CLI as a Rust library function. The original main
   implementation moved from `src/main.rs` to `src/cli.rs`; the original binary
   now forwards to that function. Expert placement algorithms and GPU kernels
   remain upstream code; version 0.2 extends the HTTP model-management routes.
5. Added native Windows/Linux build helpers, documentation, an original tiny
   synthetic GGUF fixture, and correctness/integration tests.
6. Version 0.2 adds Hugging Face search, revision-pinned file selection, isolated
   per-repo/commit model storage, resumable downloads, complete shard handling,
   available SHA-256 verification, credential handling and provenance records.
   HTTPS uses platform certificate verification and protocol-specific proxy
   precedence while preserving proxy authentication and exclusions.
7. Added a recursive model catalogue, explicit model load/unload, unique model
   IDs, before-switch metadata validation, stale OpenAI model-ID errors, and
   browser controls for search/download/switching. Exactly one model is active.
8. Added native WSL launchers for explicit CPU or Vulkan mode, plus direct
   Windows executable interop from a WSL shell. Native WSL GPU use is conditional
   on driver capability; no HIP backend or ROCm installation is implied.
9. Corrected per-load synthesized tensor storage in `weights.rs`. Owned tensors
   now free with the model, and tensor views borrow their owner. LoRA overrides
   use the same ownership approach without leaked buffers or extended unsafe
   lifetimes. The estimated owned-RAM count includes these buffers.

The wrapper runs the upstream library in the same process. It does not spawn an
external inference executable and does not depend on a C++ inference engine.
Some third-party build dependencies compile C code, and the Vulkan path uses
the upstream precompiled SPIR-V shaders and their included source.

The library bridge retains the upstream process-global configuration and Rayon
pool. Call it once per process as this CLI does; it is not a reentrant server
factory for multiple independently configured engines in one process.

## Version 0.3: compatible MoE models and concurrent requests

Version 0.3 keeps the pinned RustingStrata engine and adds the following
distribution changes. This is a Vulkan integration with selected Avalon-derived
ideas, not a complete Avalon port. The source comparison and remaining gaps are
recorded in [docs/AVALON-COMPARISON.md](docs/AVALON-COMPARISON.md).

- Added a shared sparse-MoE metadata policy in
  `compatibility.rs`. Local inspection, the model catalogue, Hub listings,
  downloads and model loads check supported architecture, expert geometry,
  required tensors, tensor types, dimensions and complete shard groups. Dense
  models, unknown types and Avalon's custom IQ3_K/IQ3_KS formats are excluded.
  The earlier dense Qwen3-8B example is consequently outside this version's
  model-selection policy. Community abliterated/uncensored models use the same
  technical checks as other models.
- Added bounded remote GGUF header inspection against immutable Hub commits.
  HTTP Range responses, declared sizes and every shard are checked; successful
  metadata checks are cached using repository/revision/file identity. A timeout
  or unavailable header remains unverified and is excluded with a reason.
  Slow Hub search/file inspections now run on a separate, single-job worker so
  they do not perform network waits on the generation scheduler.
- Exposed `--parallel 1..4` GPU server slots sharing one active model, with
  separate sequence state. CPU generation stays serial. MTP, CPU and
  architecture-specific conflicts are rejected explicitly, including before a
  working model is released for a requested switch. The queue remains bounded;
  excess requests receive a rate-limit response. Multiple active sequences are
  different from loading several models at once.
- Exposed supported `--mtp 0..2`, a matching `--mtp-file`, and the `tune`
  calibration command. Added runtime and draft-head geometry preflight. The
  default remains MTP off with conservative prompt chunks; calibration is an
  actual measurement on the selected machine, not a supplied performance claim.
- Added model identities for routing profiles and calibration records.
  Canonical paths, lengths and modification times are combined across shards
  and companion files. Calibration also checks device/backend, context, KV,
  memory margin, thread count, slot count and approximation mode. These are
  metadata identities, not full weight-content checksums; replacing bytes while
  preserving all those file attributes requires clearing the saved profiles.
- Added optional approximate expert-tail omission in `expert_tail.rs`,
  `cpu_worker.rs`, `gpu_model.rs` and the short-prefill wrapper. It is off by
  default. When enabled, only uncached main-model decode/target-verification
  experts whose best one-based rank across every row is at least the threshold
  are eligible. Cached experts, prefill and the draft head are retained. Weights
  are not renormalized, and completely omitted CPU rows publish zero. The
  one-based CLI threshold 8 corresponds to Avalon's zero-based threshold 7.
  This changes computation and can change answers, including with different
  verification batches or co-scheduled clients.
- Added approximation counters to server and benchmark output, expanded the
  performance controls and status display, and added a concurrent API benchmark
  tool reporting actual completion tokens, elapsed time and per-request latency.
  Added an original tiny sparse-MoE fixture and regression coverage for metadata,
  remote probing, request handling, configuration and the tail policy.

The v0.3 expert-tail policy is a fresh Rust implementation of the published Avalon
algorithm. That release imported no Avalon HIP kernels, elastic VRAM loan allocator,
direct-I/O RAM arena or custom quantization kernels. Existing GPU shaders are
retained. This package does not add HIP support or repair the earlier ROCm/DKMS
installation. RX 9070 XT GPU execution and native WSL Vulkan execution remain
unverified here; [VALIDATION.md](VALIDATION.md) records the checks actually run.

## Version 0.4: generation speed with preserved model computation

The profile now prioritizes the speed of an individual reply while retaining
every expert selected by the model's router. This does not assert that a
quantized model or a generated answer is perfectly accurate.

- Selected F16 KV as the initial fidelity baseline and retained one GPU slot
  and MTP off as the unmeasured starting settings. Q8 KV and multiple slots are
  explicit alternatives. These choices are defaults for evaluation, not claims
  that they are universally the fastest configuration.
- Disabled expert-tail omission through the Forge9070 profile. Nonzero
  `--expert-tail-min-rank` values are rejected, and the normal application
  configures all selected experts to contribute. The v0.3 implementation and
  its attribution remain as source history; it is not an advertised speed mode
  in the supported workflow.
- Fixed greedy GPU input preparation in `gpu_model.rs`. Greedy and full-logit
  verification now share context/batch validation, token bounds and current
  KV/RoPE position uploads. Previously the greedy path left RoPE positions
  stale, affecting QSA queries and attention with a nonzero RoPE offset. The
  shared path preserves the GPU argmax fast path without full-logit readback.
- Tightened CPU sampling in `ops.rs`: empty or mixed non-finite logits now fail
  during the existing argmax/candidate scan, including deterministic-draft
  accept/reject. A finite winning token no longer hides a non-finite entry.
  This does not introduce a new GPU-wide non-finite detection shader.
- Tightened HTTP sampling-option validation. Unsupported non-neutral penalties,
  `min_p`, `typical_p` and nonempty `logit_bias` now receive HTTP 400 instead of
  being silently ignored; conventional neutral defaults remain accepted.
  Temperature and top-p must remain finite and valid after conversion to the
  engine's `f32` representation, covering overflow and range-changing underflow.
- Revised the MTP description: verification uses the target model's logits,
  but floating-point rounding can differ with batched kernels or CPU/GPU
  placement. A token comparison on measured prompts is a useful guard, not a
  universal proof of bit-identical output or factual answer accuracy.
- Changed hardware tuning to target measured generation throughput separately
  from prompt processing and to check candidate greedy token sequences against
  a baseline. The chosen settings must pass the tested-output comparison;
  rejected candidates remain part of the report. The result is scoped to the
  measured model, precision, runtime settings and workloads, not a general
  answer-quality score. Versioned calibration records prevent reuse of the old
  whole-request-throughput selection as a new decode-speed result. The baseline
  is retained unless an eligible candidate improves the measured median decode
  rate by at least 3%; this threshold is not a statistical confidence claim.
- Added excluded benchmark warmup and paired per-repetition records: each
  measured actual-token count stays with its own prefill/decode timings.
  Tuning also retains raw workload samples and excludes warmup from its median
  decode-rate selection. This avoids combining a token count from one response
  with the timing of another when their output lengths differ.
- Added tensor-derived compression reporting to compatible GGUF metadata,
  including element counts and sub-four-bit routed-expert formats. Hub probing
  prioritizes common Q4/Q5 choices, while the browser requires an explicit
  quantization selection and explains compression separately from memory fit
  and answer quality. Existing compatible low-bit files are not relabelled as
  unsupported solely because of precision.
- Added focused no-GPU tests for position advancement, batch/context/position
  bounds and mixed non-finite CPU sampler inputs. Actual GPU execution and
  performance remain separate hardware checks.

The practical distinction is between computation fidelity, task accuracy and
measured speed. Retaining all routed experts avoids deliberate omissions; it
does not undo weight/KV quantization or make an inaccurate base model factual.
The package supplies the evaluation procedure and records the checks actually
performed in `VALIDATION.md`.

## Version 0.4.1: Qwen3.8 GGUF discovery and metadata-only first shards

- Accepts a valid metadata-only first shard of a complete GGUF model, including
  the unpadded EOF used by the inspected Unsloth Qwen3.8 conversions. The first
  file is accepted provisionally only with a valid multi-shard declaration and
  positive combined tensor count. Complete-group checks still require the
  remaining files, supported tensors and correct byte bounds. Standalone empty
  and all-empty models remain rejected by Forge's compatibility layer.
- Normalizes the runtime mapper's unused empty-file data offset to exact EOF,
  validates alignment, and preserves the first map's index so later tensor
  views refer to their correct files. A synthetic split fixture compares every
  mapped tensor's bytes against the original unsplit model.
- Adds `qwen4_exp` to the rejection-only Hugging Face search hint. The actual
  GGUF architecture allowlist is unchanged; a misleading repository label
  cannot substitute for the real header checks.
- Adds optional `--file EXACT` to `models files REPO`. This inspects only the
  requested GGUF choice and its complete shard group at the resolved immutable
  revision, with the existing time/byte bounds, without starting a full model
  download. The JSON reports accepted metadata or an explicit exclusion reason.
- Documents original and community uncensored model sources, complete download
  sizes, separate catalogue IDs and the current one-slot/no-MTP restrictions
  for HC/PLE/QSA. Header compatibility is explicitly distinct from successful
  full loading, numerical accuracy and measured hardware speed.

The pinned engine already contained Qwen3.8's `qwen4exp` text computation paths.
This release fixes discovery and packaging compatibility; it does not add a
HIP backend, a new QSA kernel or a verified RX 9070 XT benchmark.

## Version 0.5: owned expert RAM and available-memory planning

- Adds a native Rust bounded expert cache with immutable leases, checked payload
  accounting, a bounded LRU index, allocation fallback, override invalidation
  and model-instance isolation. Hot hits avoid repeated tensor-name lookup.
- Routes single/multi CPU expert execution, GPU expert fills and prompt staging
  through the lease API. Owned memory is never passed to mapped-page discard
  advice. Prefill misses bypass admission and leave decode LRU order intact,
  including short prompts that use decode-shaped GPU work.
- Retains each mapped shard's source file object and adds aligned positional
  reads: Linux O_DIRECT and Windows FILE_FLAG_NO_BUFFERING attempts, with
  counted buffered fallback using that same object. An expert is published only
  after all three matrices have been read. Scratch allocations are bounded.
- Budgets expert residency from available physical memory, Linux cgroup-v2
  ancestor headroom, actual expert tensor bytes and separate OS/runtime/prompt
  reserves. Sparse PLE tensors remain lazy. Adds `memory` and tensor-category
  reporting instead of treating complete model file bytes as required RAM.
- Adds `--expert-ram auto|0|MiB`, bounded startup warmup, statistics and matching
  tuning configuration. Calibration schema changes reject old measurements
  made under a different memory policy or budget.
- Documents the standard GSQ-RCO IQ3_S source used before Avalon's custom
  re-encoding, with a 46.84 GiB expert payload, and preserves the separate
  original/uncensored model workflow.

The design is informed by Avalon's MIT-licensed host arena, RAM budget and lazy
PLE paths at commit `762a3dae8d78ff18acb38e351dfa940328485fd8`. The new cache and
I/O helper are Rust implementations; the package does not claim HIP memory
registration, custom IQ3_K/IQ3_KS support, Qwen4 MTP, elastic VRAM loans or the
maintainer's measured token rates. Selected checkpoint bytes and all routed
experts are preserved. See the current validation record for tests actually run.

## Version 0.6: large code context and conversation correctness

- Added explicit `code` (16,384 context / 4,096 output) and `code-long`
  (32,768 context / 8,192 output) presets, with 1 GiB parked-state cache and
  6 GiB application-growth margin. Flags override defaults; sampling follows
  Qwen3.8 thinking guidance, and no hardware speed is asserted.
- Added a bounded local Git context packer with lexical/path/identifier
  ranking, current working-tree lines, scoped monorepo inventory, explicit
  omissions, known secret/generated/binary exclusions and verified contained
  file handles. Optional GGUF tokenization counts the entire rendered pack.
  It is not a semantic index, remote upload or autonomous coding agent.
- Unified context bounds across CLI and model reloads, including training
  metadata limits. Added exact complete-request budget preview at
  `/api/context` and explicit oversize-output/context rejection before model
  state changes. Generation reuses the same already-checked token IDs.
- Updated browser context/output controls, explicit coding presets and
  pre-send budget checks. Fresh controls inherit server settings; earlier
  assistant reasoning and open-continuation whitespace are preserved.
- Exposed existing think/preserve-thinking controls and optional persistent
  GPU prefix-cache path. Replaced unsafe version-1 persistence with bounded,
  checksummed version-2 records keyed to model/shard metadata, actual adapters
  and runtime geometry. Snapshot admission checks budgets before allocation.
  Model metadata identity is not a fresh full-weight-content hash.
- Added targeted unit, CLI, local-Git, HTTP and browser checks. The validation
  report distinguishes executed tests from untested Windows/RX 9070 XT
  inference and from source-derived context-memory calculations.

## Public 0.9.1 source hardening — 18 November 2026

- Corrected the independent engine's Host validation for wildcard listeners.
  It checks the interface that accepted each connection and preserves
  authentication, exact Origin matching and rejection of unrelated Hosts.
- Expanded Linux and Windows CI to build both documented entry points, with
  real-browser and software-Vulkan gates on Linux. Release preparation requires
  an owner-merged main commit and successful CI for that exact commit.
- Added release-provenance and archive-integrity regression tests, portable
  binary packaging checks, and explicit owner-only repository ruleset templates.
- Excluded 21 optional captured upstream benchmark/transcript/metadata files
  from the public source tree. Original licence notices and all required
  numerical/template fixtures remain. Optional reference runners now take
  configurable paths. See the vendored modification notice for scope.
- Retained the experimental status and the outstanding full-model/RX 9070 XT
  acceptance work. Source-publication and synthetic test checks do not establish
  target-hardware production readiness.
