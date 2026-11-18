# Forge9070 integration changes

Uses RustingStrata by Vasyl Trefilov.

This is a modified distribution of RustingStrata. The original engine remains
under the RustingStrata License 1.0 in `LICENSE.md`.

This is a chronological notice. The v0.4 profile disables the earlier
user-facing expert-tail approximation and changes the initial KV format.

The original `src/main.rs` implementation was moved to the public `src/cli.rs`
module. `cli::run_from` accepts OS-native arguments and runs the native Rust
engine within the calling process. Argument parsing errors are returned;
help and version requests print their output and return success. The original
binary is now a small caller of this same library entry point. Model execution,
expert placement, GPU kernels and server behavior are unchanged by this bridge.

## Forge9070 0.2 additions

- `hf.rs`: repository search, immutable revision resolution, per-repo/commit
  storage, resumable/verified GGUF downloads, automatic shard groups and
  provenance. Tokens remain in memory and cross-host redirects strip them.
  HTTPS respects protocol-specific proxies and platform trust stores without
  disabling certificate or hostname verification.
- `serve.rs` and `cli.rs`: recursive model catalogue, unique IDs, before-switch
  metadata checks, sequential load/unload, and stale OpenAI model-ID rejection.
- `web/index.html`: Hugging Face search, pinned file selection, download
  progress/cancel, model switching and unload controls, with Forge9070 branding
  and the original acknowledgement retained.
- `weights.rs` and `model.rs`: synthesized tensors and LoRA copies now have
  model-owned storage and borrow-scoped views, eliminating per-reload leaks and
  extended unsafe lifetimes for these buffers.
- Added model-management regression tests. GPU kernels and expert placement
  algorithms remain unchanged. These changes do not add a HIP backend.

The complete distribution-level list is in the root `MODIFICATIONS.md`.

## Public-source packaging

The public Forge9070 distribution omits optional captured upstream runs:
`docs/V100-results.md`, all three captured `docs/parity/*.json` reports, the four
`tools/deepseek_ref/ref_*.json` captures, the three
`tools/glm_ref/p*_llama.json` captures, and the two `server_llama*.err` logs.
The generated `gguf_meta.txt`, `gguf_tensors.txt` and `long_prompt.txt` captures
under `tools/deepseek_ref`, `tools/prompts.json`, and the captured `p*_req.json`
and `strs.jsonl` files under `tools/glm_ref` are also omitted. Optional comparison
helpers generate or accept their own inputs; these captures are not runtime dependencies.
These contain environment-specific paths or generated transcripts and are not
inputs to the Rust build or test suites. Their generator tools, required
numerical and template fixtures, and all original licence notices remain.
Benchmark documentation links point to the pinned upstream revision where
appropriate. Optional runner scripts use configurable paths instead of an
upstream developer's workstation layout. This packaging choice makes no new
claim about the upstream benchmark results or Forge9070 performance.

## Forge9070 0.3 additions

- `compatibility.rs`: a shared conservative sparse-MoE policy, bounded GGUF
  header parser, complete shard validation, implemented quantization checks,
  required tensor/geometry validation and draft-head geometry checks. The model
  catalogue and all public download/load paths apply this policy. A successful
  metadata result does not establish memory fit, numerical correctness or speed.
- `hf.rs`: immutable-commit, bounded HTTP Range probes before listing/downloading
  a model; checks include response ranges, file boundaries and every shard.
  Positive results are cached by repository/revision/file identity. Listings
  distinguish rejected or unverified files. The policy rejects dense models and
  unsupported custom formats irrespective of repository names or content labels.
- `cli.rs` and `serve.rs`: exposed GPU sequence slots and MTP controls, early
  validation of incompatible modes, per-load options, and model-dependent
  parallel/MTP constraints. A separate metadata worker keeps Hub network waits
  off the decode scheduler and admits at most one such job at a time. Serving
  retains one loaded model; CPU responses are generated sequentially.
- `cache_identity.rs`, `gguf.rs` and `expert_cache.rs`: per-model routing-profile
  identities combine canonical file paths, lengths and modification times for
  every shard and optional companion. `cli.rs` stores and checks calibration
  against the model identity and the measured device/configuration. Historical
  basename-only calibration files are not trusted as current matches. These
  identities do not hash the complete weight contents.
- `expert_tail.rs`, `cpu_worker.rs`, `gpu_model.rs` and
  `gpu_model/prefill.rs`: optional approximate omission of low-ranked CPU expert
  misses, configured once per main-model worker through
  `STRATA_EXPERT_TAIL_MIN_RANK`. Zero disables it. An expert is retained if any
  row ranks it above the cutoff or observes it resident. Main-model decode and
  target verification alone are eligible; prefill and MTP draft-head computation
  retain all selected experts. Retained weights are unchanged. Omitted CPU rows are zeroed before
  the usual completion handshake. Per-assignment and per-expert-group counters
  expose the omitted work. The pass type travels with each worker sequence.
- `cli.rs`, `serve.rs` and `web/index.html`: mode/counter reporting, MoE-filtered
  model selection and explicit performance options. Added synthetic MoE and
  policy/transport/configuration/request regression coverage.

Tail omission changes model computation. Because the policy uses all rows in a
pass, a verification batch or another active client can affect which experts
are retained. Even with omission disabled, floating-point rounding may differ
between batched/single-token kernels or CPU/GPU expert placement. Speculative
accept/reject preserves the distribution of the target logits used to verify;
it is not a universal cross-kernel bit-identity claim.

The optional policy is a fresh Rust translation of `moe_hosted_tail_skip` in
[Avalon commit 762a3dae8d78ff18acb38e351dfa940328485fd8](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/src/ggml-cuda/moe-cache.cu).
Avalon is published under the MIT licence. Its complete engine, HIP kernels,
custom IQ3_K/IQ3_KS kernels and memory allocators were not imported. Forge9070
continues to use the Vulkan backend and the included RustingStrata licence.
GPU execution and native WSL Vulkan execution have not been validated on the
target RX 9070 XT; source changes and CPU tests do not establish GPU throughput.

## Forge9070 0.4 additions

- `gpu_model.rs`: greedy and full-logit verification share validated input
  preparation, including current RoPE positions. This fixes stale QSA/offset
  positions in the greedy path. Bounds are checked before starting a worker
  sequence or submitting the forward. The existing argmax-only readback,
  profiling boundaries and forward/finish/after-forward ordering are retained.
  Three no-GPU tests cover per-row positions, context/batch bounds and integer
  position overflow. No shader or split-GPU implementation was changed.
- `ops.rs`: CPU greedy selection, sampling and fixed-draft acceptance reject
  empty or mixed non-finite logits during their existing scans. Two focused
  unit tests cover the vectorized scan/tail, invalid values and finite ties.
  GPU argmax/non-finite signalling remains upstream behaviour.
- `serve.rs`: unsupported non-neutral sampling options return HTTP 400 rather
  than silently altering the requested behaviour. Neutral penalty/filter values
  and an empty `logit_bias` remain accepted. Temperature/top-p validation checks
  the actual `f32` value as well as the JSON number, rejecting conversion
  overflow and values that leave the permitted range after rounding/underflow.
- `compatibility.rs`, `hf.rs` and the browser: tensor-derived compression
  metadata distinguishes sub-four-bit formats, including routed expert banks,
  from format compatibility. Q4/Q5 probe priority is not an accuracy score or
  automatic download recommendation. The browser requires an explicit GGUF
  choice and explains compression, memory estimates and quality separately.
- `cli.rs`: calibration distinguishes generation time from prefill, compares
  candidate greedy outputs with a baseline and records candidate selection or
  rejection. The new versioned record describes the generation objective and
  tested-output gate; historical whole-request-rate calibrations are excluded.
  Selection requires at least 3% improvement in median measured decode rate
  over the baseline; otherwise the baseline remains selected. Benchmark and
  calibration records pair actual completion counts with their corresponding
  timing samples and exclude configured warmup from measured rates.
  A measured agreement on the calibration workloads is not universal numerical
  parity or a factual-accuracy evaluation, and the 3% threshold is not a
  statistical confidence guarantee.
- Distribution controls: F16 KV, one active GPU request and MTP off are the
  initial evaluation defaults; expert-tail omission is disabled by the Forge
  profile. Q8 and multiple request slots remain explicit tradeoffs. The tail
  implementation is retained for provenance, with its existing licence notice.

Preserving selected experts does not undo checkpoint quantization, eliminate
floating-point rounding or guarantee the correctness of generated answers.
The hardware tuning and validation records distinguish these claims.
