> **Historical comparison:** Strata references in this document describe
> earlier releases. Current Nextora no longer links or vendors Strata;
> the active native inference implementation is in `crates/nextora-engine`.

# Avalon comparison and optimisation scope

## Current implementation: Nextora 0.9 independent Rust engine

Updated 17 November 2026. The independent execution path lives in
[`crates/nextora-engine`](../crates/nextora-engine): it owns GGUF loading,
quantization, model execution, expert storage, and Vulkan work. The older
RustingStrata comparison is preserved under **Compatibility-engine history**
below. Its MTP, KV-precision and concurrency descriptions belong to that older
engine and must not be read as native 0.9 capabilities.

The native loader applies the useful findings from the previously reviewed
Avalon commit
[`762a3dae8d78ff18acb38e351dfa940328485fd8`](https://github.com/freriadoc/Avalon/tree/762a3dae8d78ff18acb38e351dfa940328485fd8):
keep a bounded owned expert working set, account for real available memory,
protect in-use weights, avoid filling RAM with the sparse embedding table,
and protect decode residents from prompt traffic. These are independently
implemented Rust mechanisms. They do not copy Avalon's complete allocator,
dispatcher, custom quantizations or published performance results. Separately
licensed GGML codebook data is identified in
[`GGML-NOTICE.md`](../crates/nextora-engine/GGML-NOTICE.md).

### Actual selected checkpoint, rather than its filename alone

The bounded header inspection is for
[`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S)
at revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, selecting
`IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`.
It declares `qwen4exp`, 48 layers, 512 experts per layer and 10 selected experts
per token per layer. There are two shards and 1,224 tensors. The retained
[header evidence](validation/qwen38-header-checks/ista-gsq-rco-iq3-s-v050.json)
records these payload counts:

| Payload category | Tensors | Bytes | GiB |
| --- | ---: | ---: | ---: |
| Routed gate/up/down expert tensors | 144 | 50,292,326,400 | 46.838 |
| Lazy PLE hash-embedding table | 1 | 28,800,138,240 | 26.822 |
| Other weights, including dense projections | 1,079 | 4,514,173,440 | 4.204 |
| Total tensor payload | 1,224 | 83,606,638,080 | 77.865 |

These are tensor bytes, excluding file headers and padding. The PLE table is
exactly `per_layer_token_embd.weight`, shape `[160, 320001536]`, stored as
IQ4_NL. Native Qwen3.8 reads only the needed hash rows for each token; its
projection, normalization and convolution weights remain ordinary dense
weights. The entire 77.865-GiB tensor payload is therefore not an owned-RAM
allocation requirement. Disk capacity, available RAM, per-sequence state,
and GPU capacity remain separate limits.

The checkpoint actually contains 13 formats: BF16, F16, F32, IQ2_S, IQ3_S,
IQ3_XXS, IQ4_NL, IQ4_XS, Q2_0, Q4_K, Q5_K, Q6_K and Q8_0. Native
[`quant.rs`](../crates/nextora-engine/src/quant.rs) and the packed
[`quant.glsl`](../crates/nextora-engine/shaders/quant.glsl) path cover these;
Q4_0 is also supported. The IQ3_S filename does not mean every tensor uses
IQ3_S. Header acceptance and format support do not by themselves establish
trained-model quality or target-card throughput.

### Implemented loader and cache mechanisms

| Mechanism | Native 0.9 implementation and actual policy |
| --- | --- |
| Complete sharded model lookup | [`Gguf::open`](../crates/nextora-engine/src/gguf.rs) maps each shard and exposes one tensor namespace without copying the whole weight set. It accepts a metadata-only first shard and selection of any part, and validates split counts, ordinals, shared metadata, tensor identities and byte ranges. |
| Owned, bounded host expert cache | [`ExpertCache::lease_with`](../crates/nextora-engine/src/expert_cache.rs) owns encoded expert bytes under a hard payload ceiling. Decode hits update LRU recency. Eviction considers only unleased entries. If an expert cannot fit, the caller receives its mapped view instead of an overflow allocation. |
| Available-RAM accounting | [`memory::plan`](../crates/nextora-engine/src/memory.rs) uses Linux/WSL MemAvailable with cgroup-v2 headroom, or Windows GlobalMemoryStatusEx, then allows for dense pages, configured sequence slots and an OS margin. Automatic capacity is capped by actual expert payload. An explicit ceiling is clamped to that payload and checked against the same headroom. Zero disables the owned cache. This plan is an estimate, not an OS reservation. |
| Immutable active leases and stable identity | [`ExpertLease` and `TensorKey`](../crates/nextora-engine/src/gguf.rs) preserve the model/shard/byte-range identity when weights move into owned RAM. An active Arc lease prevents its cache entry from being evicted. The reported `pinned_bytes` means retained by readers; it does not mean OS-locked or GPU-registered memory. |
| Offset/direct reads with observable fallback | [`expert_io::Source`](../crates/nextora-engine/src/expert_io.rs) attempts Linux O_DIRECT or Windows FILE_FLAG_NO_BUFFERING with a bounded 64-KiB aligned scratch buffer. Unsupported or failed direct reads use buffered positional reads. Owned fills do not copy from the mmap. Successful direct I/O avoids populating mapping pages; buffered reads and mapped bypasses can still populate OS cache. |
| Bounded startup warmup | [`Gguf::warm_expert_cache`](../crates/nextora-engine/src/gguf.rs) visits expert IDs in ascending order, round-robin across layers. It admits complete gate/up/down triplets only when the remaining budget fits them, never evicts existing entries, and never warms the whole PLE table. This is unranked warmup, not a prediction of the hottest experts. A failed triplet read publishes none of that triplet. |
| Prefill protection for decode residents | [`expert_cached_for_prefill`](../crates/nextora-engine/src/gguf.rs) and [`prefill_lease`](../crates/nextora-engine/src/expert_cache.rs) reuse existing host entries without admitting misses or refreshing decode recency. Native [`model.rs`](../crates/nextora-engine/src/model.rs) and [`qwen38.rs`](../crates/nextora-engine/src/qwen38.rs) use this policy for prompt tokens. The Qwen3.8 Vulkan [`prepare_batch`](../crates/nextora-engine/src/vulkan/hybrid.rs) policy also keeps prefill misses transient and prevents them from evicting cached decode weights. |
| Retention of every selected expert | Native CPU/hybrid [`qwen38.rs`](../crates/nextora-engine/src/qwen38.rs) executes the router's selected experts. The resident Vulkan [`QwenGpuModel`](../crates/nextora-engine/src/vulkan/qwen38.rs) keeps selected device buffers alive through submission and its fence; an expert whose packed triplet cannot fit the device budget is computed on the CPU and merged with its routing weight. CPU expert executions are counted. There is no expert-tail omission. |
| Visible cache pressure and I/O behavior | [`ExpertCacheStats`](../crates/nextora-engine/src/expert_cache.rs) reports resident/pinned bytes, hits, misses, evictions, mapped bypasses, prefill hits/misses, direct and buffered loads, fallback counts, payload bytes read and read errors. [`runtime.rs`](../crates/nextora-engine/src/runtime.rs) exposes the memory plan, cache statistics and warmup report. |

Avalon's corresponding design evidence remains the pinned
[RAM arena](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/src/ggml-cuda/moe-arena.h),
[RAM budget](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/src/ggml-cuda/moe-ram-budget.h),
[expert cache and hosted dispatcher](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/src/ggml-cuda/moe-cache.cu),
[transfer ring](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/src/ggml-cuda/moe-ring.h)
and [ahead reader](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/src/ggml-cuda/moe-ring-ahead.h).
The policies and scheduling are not identical.

For the actual Qwen3.8 geometry, native host-state planning allows
4,149,084,160 bytes for one 32,768-token slot and 16,596,336,640 bytes for four.
These allowances include KV/indexer capacity growth, recurrent working and
rollback buffers, and per-slot scratch allowance. More host slots therefore
reduce the RAM available for cached experts. Resident Vulkan KV/recurrent
state is accounted on the device instead; its host plan retains a smaller
per-slot host allowance. Native KV and recurrent arithmetic currently use F32.

### Features that native 0.9 does not yet implement

| Feature | Present limit |
| --- | --- |
| Registered or pinned host arena | Owned expert vectors are pageable. There is no HIP registration, imported host-memory arena or loan-and-restore prefill allocator. Immutable lease retention must not be presented as driver registration. |
| Shipped expert rankings | No measured or packaged per-layer expert ranking guides startup. Warmup is unranked round-robin; the host cache follows subsequent decode access recency. |
| Asynchronous transfer/CPU overlap | Selected packed GPU uploads are batched into bounded staging windows, but uploads use fences and CPU expert work completes before the layer's expert graph submission. There is no ahead-read pipeline like Avalon's or concurrent CPU-miss/GPU-hit dispatcher. |
| MTP/speculative decoding | The independent path has no draft-head execution or target verification. Historical compatibility-engine MTP options do not provide native MTP. |
| Avalon IQ3_K and IQ3_KS | Custom GGUF type IDs 43 and 44 are unsupported. They are not interchangeable with the standard IQ3_S checkpoint used here; see Avalon's pinned [quantization IDs](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/include/ggml.h). |
| Elastic KV/compute-buffer borrowing | The expert cache does not borrow an unused KV tail or reclaimable compute arena. See Avalon's pinned [context lending](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/src/llama-context.cpp) and [KV lending](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/src/llama-kv-cache.cpp) for that separate mechanism. |

### Evidence and performance limits

[`native-storage-validation-0.9.0.json`](validation/native-storage-validation-0.9.0.json)
records the previously executed focused gates: 17 storage/cache/I/O tests,
3 quantization parity tests, 7 memory-budget tests and 10 protocol tests.
These are receipts from their individual runs, not a claim that later edits
were covered by those same invocations. Quantization parity uses the existing
reference only in integration tests; it is not an inference-runtime dependency.

The retained checkpoint evidence contains bounded header data, not the trained
weights. These receipts do not establish full-checkpoint inference, answer
quality, Windows direct-I/O execution or RX 9070 XT throughput. Avalon's
published token rates depend on its hardware, custom checkpoint and additional
techniques. Native rates require measurement with the actual checkpoint,
context, cache warmth, slot count and backend recorded.

## Compatibility-engine history: Nextora 0.3–0.5

**Everything below this heading describes the earlier RustingStrata-backed
compatibility engine and its historical release policies.** It is retained for
provenance; the current independent implementation and its gaps are described
above.

Reviewed 16 November 2026. Avalon source was inspected at commit
[`762a3dae8d78ff18acb38e351dfa940328485fd8`](https://github.com/freriadoc/Avalon/tree/762a3dae8d78ff18acb38e351dfa940328485fd8).
The source includes the techniques below; this is more than a comparison of
project names or README headlines. Nextora embeds RustingStrata 1.2.4 at
[`f821c4435ea553eb8a49f542313da52caab75b90`](https://github.com/RustingStudio/RustingStrata/tree/f821c4435ea553eb8a49f542313da52caab75b90).
Neither source inspection nor a passing CPU test measures performance on
the target RX 9070 XT.

**Historical 0.5 profile: Nextora 0.5 retains all selected experts, starts with one
GPU request slot and F16 KV, and rejects expert-tail omission.** The v0.3 tail
policy below is retained as implementation/provenance history, not a supported
speed recommendation for the current fidelity-focused workflow. Q8 KV and
checkpoint quantization remain explicit precision/memory tradeoffs.

### What the existing engine already does

| Technique | Avalon implementation | Nextora / RustingStrata implementation |
| --- | --- | --- |
| Keep useful experts on the GPU | Per-expert pools, admission and replacement in `ggml/src/ggml-cuda/moe-cache.cu` | `src/expert_cache.rs`: `ExpertCache::new`, `record`, `adapt`; routing frequencies place hot experts and replace cold residents |
| Compute CPU misses alongside GPU hits | Hosted MoE mailbox and GPU merge inside the graph | `src/cpu_worker.rs` and `src/gpu_model.rs`: CPU worker receives routed misses while cached experts execute on the GPU |
| Prefetch and batch prompt experts | Ahead reads and transfer slots borrowed from the expert cache | `src/gpu_model/prefill.rs`: grouped expert GEMMs, three staged groups, transfer/compute overlap when the backend has a suitable transfer queue |
| Reuse captured GPU work | HIP/CUDA graphs and fused kernels | Vulkan command-buffer replay in `src/vulkan/mod.rs`, plus the engine's fused kernels |
| Reduce attention state memory | Quantized keys and values | Q8 or F16 KV cache; Q8 leaves more room for experts |
| Sparse attention for the relevant architecture | GPU QSA selection and gathered attention | `qsa_pool`/`qsa_select` and QSA buffers in the Rust GPU forward and prefill |
| Speculative decoding | Small MTP draft head sharing the main expert cache | `spec_step`/`mtp_run`, supported embedded or auxiliary draft heads; subject to the model and concurrency limits below |

These are the same broad optimisation ideas, with different policies, memory
allocators and kernels. They do not establish identical implementation or
identical throughput. The relevant Avalon sources are the pinned
[expert cache and hosted dispatcher](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/src/ggml-cuda/moe-cache.cu),
[prefetch ring](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/src/ggml-cuda/moe-ring.h),
[ahead reader](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/src/ggml-cuda/moe-ring-ahead.h)
and [QSA kernels](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/src/ggml-cuda/qsa-select.cu).
The corresponding RustingStrata sources are the pinned
[expert cache](https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/src/expert_cache.rs),
[CPU worker](https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/src/cpu_worker.rs),
[GPU forward and speculative decoding](https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/src/gpu_model.rs),
[prefill](https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/src/gpu_model/prefill.rs)
and [Vulkan graph implementation](https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/src/vulkan/mod.rs).

### Owned expert RAM implemented in Nextora 0.5

The earlier releases used mmap and filesystem cache for host experts. Version
0.5 adds a native Rust owned expert tier, available-memory budgeting and
positional direct reads. Immutable leases protect CPU readers and GPU uploads;
the payload ceiling includes live leased entries. Startup warmup loads only
bounded expert slots. Prefill can reuse existing entries but does not admit
misses or disturb the decode LRU. See [EXPERT-RAM.md](EXPERT-RAM.md) for exact
policy, flags, platform fallbacks and limits.

This addresses the central memory finding in Avalon's source. The sparse PLE
file remains lazy, so the entire GGUF download is not a RAM requirement. The
standard GSQ-RCO IQ3_S source has 46.84 GiB of expert weights; the previous
UD-IQ4_XS candidate has 56.24 GiB. Allocation uses actual available memory,
with reserves, rather than assuming a 62 GB host has 62 GB free.

The Rust tier uses pageable allocations. It does not copy HIP driver
registration limits, pinned/imported host buffers, Avalon's prefill slot loans
or its Windows-disabled GPU wait dispatcher. No new approximate arithmetic,
tail skipping or custom quantization kernels are enabled.

### Changes in Nextora 0.3

The model workflow now checks actual GGUF metadata, routed-expert geometry and
tensor types. The speed-focused catalogue admits sparse MoE models only.
Repository names and words such as "uncensored" are not proof of compatibility;
the full model's shards must satisfy the policy. Metadata acceptance does not
prove that a model fits available memory or reaches a particular token rate.

The earlier Qwen3-8B abliterated example is a dense checkpoint. It cannot use
the MoE hot-expert-cache strategy and is excluded from the MoE-only picker.
Compatible MoE community variants remain eligible. See that publisher's
[Qwen3-8B GGUF model card](https://huggingface.co/DevQuasar/huihui-ai.Qwen3-8B-abliterated-GGUF)
and the included `src/compatibility.rs` for the actual acceptance rules.

The update also exposes tuning and concurrency controls described in the main
README. The previous wrapper explicitly forced MTP off and fixed its prompt
chunk size, even though the underlying engine already provided calibration
and speculative decoding. Settings that conflict with a model or with each
other must be resolved explicitly; a model label does not override those
constraints.

#### Historical v0.3 approximate expert-tail omission

Version 0.3 exposed `--expert-tail-min-rank R` for a new Rust implementation of
Avalon's `moe_hosted_tail_skip`, off by default (`R = 0`). Version 0.4 rejects
nonzero values through the Nextora profile; the helper and attribution remain
in the source. The following describes the historical algorithm.
Ranks are one-based: `R = 8` corresponds to Avalon's zero-based threshold 7.
The source policy is `moe_hosted_tail_skip` in the pinned
[Avalon hosted expert dispatcher](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/src/ggml-cuda/moe-cache.cu).

An expert is omitted only when it is a CPU cache miss and every occurrence
across the current decode or target-verification pass has rank R or lower.
If any row ranks it higher, all occurrences are kept. Cached GPU experts are
never omitted. Retained routing weights are unchanged; there is no
renormalization. Omitting every CPU miss in a row explicitly publishes a zero
CPU contribution before the normal completion handshake, so a previous row's
output cannot be reused accidentally.

Prefill retains all selected experts, including short prompts that internally
use decode-shaped kernels. The MTP draft head also retains them. Counters distinguish omitted
token/expert assignments from distinct omitted expert groups per layer/pass.
Ordinary cache-miss counters continue to count routing misses, including ones
that this optional approximation omits.

**This changes model computation and may change answers.** Its selection is
made across the rows being computed together. Consequently, a target checked
with several speculative tokens, or decoded alongside another client, can
keep an expert that a single-row pass would omit. Speculative verification
against the target logits cannot undo this deliberate approximation. Differences
caused by batching are part of the defined policy, not evidence of an exact
performance optimisation.

The helper's tests cover rank ordering, a high-rank occurrence in another row,
cached experts, repeated expert groups, original weighting, complete omission,
and exact/prefill/draft eligibility. Hardware execution and the speed/quality
tradeoff still need measurement on the target card. The implementation changes
the CPU-miss selection and result publication, without new GPU shaders.

#### Fidelity and the v0.4 correction

MTP verifies drafts against the selected target model. For a deterministic
draft, accepting it with its probability under the configured target sampler
and otherwise sampling the renormalized remainder preserves that sampler's
distribution. This statement is
about the logits used for verification. Batched kernels and CPU/GPU expert
placement may round differently from single-token execution, so it is too
strong to promise bit-identical output for every model and backend. The pinned
[GPU model tests](https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/tests/gpu_model.rs)
use fixed expert placement for their MTP token comparison and tolerate some
logit differences when comparing batched verification with individual steps.

The v0.4 source audit found a concrete greedy-path problem: `verify_greedy`
updated the KV/worker positions but not the RoPE position buffer. QSA reads the
latter even when the ordinary attention offset is zero. The modified engine
now shares validated position preparation between greedy and full-logit
verification. It retains GPU argmax readback instead of copying the vocabulary
to the CPU. The associated no-GPU tests cover preparation/bounds; actual QSA
GPU execution still requires target hardware validation.

CPU sampler scans now reject mixed non-finite logits instead of discarding an
invalid entry while returning a finite token. This is a CPU-side check and
does not claim GPU-wide non-finite detection. No GPU shaders were changed.

The v0.4 tuning workflow targets measured generation throughput separately
from prefill and checks candidate greedy token sequences against a baseline
at fixed model/precision/context settings. Candidates that disagree on the
tested outputs are excluded. This is a workload-scoped numerical guard; it
does not certify factual answers, all possible prompts or every sampling mode.
Changing checkpoint or KV precision is a separate quality/speed comparison.

Retaining all experts and checking draft output are computation-fidelity
measures. Weight quantization, KV quantization and the base model's task
accuracy remain separate. F16 KV avoids additional int8 KV quantization;
`--llama-numerics` is an upstream reference-parity mode with a performance cost,
not a switch that makes answers more factual. See the pinned
[CLI mode descriptions](https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/src/cli.rs)
and [sampling implementation](https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/src/ops.rs).

### Avalon mechanisms that remain different or absent

| Mechanism | Actual gap |
| --- | --- |
| Elastic VRAM lending | Avalon lends unused compute buffers and the unused KV tail to its expert cache, then reclaims them. The Rust engine allocates KV/scratch first and budgets expert pools from the remainder; it does not implement that loan allocator. |
| Registered RAM and prefill loans | Forge 0.5 now has an owned expert LRU, available-memory/cgroup budgeting and native direct-read attempts. Its allocations are pageable, not HIP-registered. Prefill misses bypass admission instead of borrowing and restoring arena slots. |
| Per-pass cost-based CPU/GPU dispatch | Avalon estimates CPU, GPU and PCIe costs and may move selected misses to the GPU to shorten the current MoE window. RustingStrata adapts placement from routing frequency; it lacks the same fitted dispatcher. |
| Custom IQ3_K / IQ3_KS | Avalon uses custom GGUF type IDs 43 and 44 and matching CPU/HIP kernels. Those types remain unsupported here. They are not interchangeable with Q3_K or IQ3_S. |
| Q4 value cache | Avalon's headline configuration uses Q8 keys and Q4 values. This engine provides Q8/F16 KV modes. |
| Flagship-model MTP | RustingStrata disables MTP for hyper-connection models, including the qwen4exp path used by Avalon's flagship. Exposing an MTP option does not remove this limitation. |
| Backend-specific tuning | Avalon's published AMD path is HIP/ROCm. Nextora's RX 9070 XT path is Vulkan. A Rust wrapper does not translate HIP kernels or add ROCm support. |

See Avalon's pinned [RAM arena](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/src/ggml-cuda/moe-arena.h),
[RAM budgeting](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/src/ggml-cuda/moe-ram-budget.h),
[context/compute lending](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/src/llama-context.cpp),
[KV lending](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/src/llama-kv-cache.cpp)
and [quantization IDs](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/include/ggml.h).

### Concurrency and performance interpretation

Multiple HTTP connections can be accepted even when computation is serial.
GPU parallel slots provide simultaneous active sequences for supported models:
weights are shared, while each slot needs its own KV/recurrent state. The MoE
engine processes up to four decode rows in a shared forward. Its larger dense
batch path does not imply that dozens of MoE requests run in one forward.

The pinned engine rejects parallel slots for hyper-connections, PLE or QSA
models. This profile's ordinary MTP geometry preflight also rejects those
configurations. MTP and multiple active GPU slots are not supported together. CPU
reference inference handles one request at a time; accepting or queueing
several requests must not be described as simultaneous CPU generation.
These limits follow the pinned RustingStrata
[GPU model allocation and forwarding](https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/src/gpu_model.rs)
and [CLI flag resolution](https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/src/cli.rs).

Extra slots consume memory that would otherwise hold expert weights. More
aggregate tokens per second does not imply faster replies for each person.
Avalon's own headline command uses `--parallel 1` and explains the KV-memory
reason in its pinned [README](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/README.md).

Avalon's roughly 98 tokens/s headline was measured on an RX 7900 XT with 20 GB
VRAM, a Ryzen 7 3800XT, 64 GB DDR4, its custom Qwen3.8-Flash-Next quantization,
warm expert cache, greedy output, MTP drafts and approximate tail skipping.
Those conditions are not a benchmark of this Rust/Vulkan build on a 16 GB
RX 9070 XT. The "Strata v0.2.9" comparison in Avalon's README refers to
[Niko1221/Strata](https://github.com/Niko1221/Strata), a different project from
RustingStudio/RustingStrata.

Record the actual model/revision, quantization, context/prompt lengths, slot
count, draft setting, disabled expert omission in Nextora 0.4, cache warmness, CPU/RAM and GPU backend
when comparing results. Measure time to first token, per-request decode speed
and aggregate throughput separately. `VALIDATION.md` states which checks were
performed in this package; it does not claim target-hardware throughput.

### Attribution

Uses RustingStrata by Vasyl Trefilov, under its included RustingStrata License
1.0 ([pinned licence](https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/LICENSE.md)).
Avalon is a llama.cpp fork under the
[MIT licence](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/LICENSE),
copyright the ggml authors. The historical v0.3 tail policy is a fresh Rust translation
of the published algorithm; it is not a copy of Avalon's complete engine,
custom quantization kernels, memory allocator or benchmark results.
