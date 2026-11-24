# AMD memory and throughput findings

Measured on 21 November 2026 with Ryzen 9 9950X3D, 64 GB DDR5-6000 and Radeon RX 9070 XT.
The engine remains native Rust/Vulkan. No BIOS, driver, power-plan, affinity,
or model location were changed. The later logits fix explicitly enables Direct
for the existing saved model, leaving its other service settings intact.

## Findings from this machine

| Component | Verified observation | Consequence |
|---|---|---|
| GPU link | PCIe 5.0 x16 negotiated | No evidence of a reduced-width GPU link to fix. |
| GPU queues | Dedicated transfer queue plus compute queues exposed | Future independent transfers can be pipelined. Queue availability alone does not prove overlap or a throughput gain. |
| CPU | AVX-512F/BW/DQ supported; 16 cores / 32 threads | Direct packed Q4_K/Q6_K CPU kernels can replace scalar block reconstruction plus AVX2 accumulation. |
| CPU cache | Windows reports 96 MiB and 32 MiB L3 domains; one NUMA node | Affinity experiments should use native topology; WSL's cache view is incomplete. No affinity was changed. |
| F: models | Two Samsung 850 EVO SATA SSDs behind one PCIe 2.0 ×1 controller | Cold loading and disk misses share a theoretical 500 MB/s link before overhead. This is not the NVMe drive. |
| C: storage | WD_BLACK SN750 SE NVMe, PCIe 4.0 x4 | A future optional NVMe working cache could help large models that spill to disk. Model files remain on F:. |

The warmed Qwen3-30B run retained 17,553,162,240 expert bytes in RAM. Its measured
steady decode samples had zero disk reads, zero PCIe weight uploads and zero CPU
expert fallbacks. About 93% of host-wall decode time was inside GPU submission and
fence waits; this is not a measurement of 93% GPU utilization. Extra CPU threads
or faster SSD storage alone cannot remove that measured wait.

## Changes implemented

- Runtime-selected AVX-512 Q4_K/Q6_K row-dot kernels, with CPU/OS feature checks and
  AVX2/scalar fallbacks. Windows warm-memory row benchmarks measured 5.44–5.53×
  (Q4_K) and 4.42–4.49× (Q6_K) versus the previous AVX2 implementation. These are
  CPU kernel measurements, not whole-model token-rate claims. Tests compare
  independent bit layouts, complete synthetic model hidden/logit outputs, and
  an F64 numerical reference for ill-conditioned benchmark rows.
- For CPUs without the AVX-512 K-format kernel, Q4_K and Q6_K dot products
  accumulate in F64 lanes to retain small residuals after cancellation.
- Bounded expert-read scratch now defaults to 1 MiB and allocates only the aligned
  span needed for small reads. A typical 884,736-byte matrix requires one read
  instead of 14. On the existing F: volume, complete expert-cache warmup measured
  approximately 48–51 seconds versus 80.37 seconds with 64 KiB reads. Override
  `NEXTORA_EXPERT_READ_KIB=64` restores the earlier chunk size for controlled
  comparisons; accepted values are 4–1024 KiB in 4 KiB multiples.
- Optional benchmark phase-counter snapshots (`NEXTORA_PROFILE_SPARSE=1`) are
  sampled outside timed regions. They separate cold warmup, prompt processing
  and steady decode rather than attributing cumulative cold-read time to CPU
  work during a warmed sample.

## Direct GPU expert path

The initial descriptor-array prototype changed one greedy token in a 128-step
same-input real-checkpoint comparison. A pointwise hardware probe localized the
first difference to Q6_K fast integer-dot GEMVs: the old and array modules
rounded their floating-point scale products differently. Q8 activation bytes
and scales, Q4_K GEMVs, normalization and the tested RoPE rotations matched.
A full layer trace confirmed that tiny Q6_K differences could later cross an
activation-quantization threshold and amplify through the model.

With Direct and fast math enabled, the Q4_K and Q6_K expert kernels read
packed words and assign two of the original 128 lane accumulators to each of
64 workers. They preserve the reduction order. The Q6_K kernel applies the
activation scale, integer dot product and signed subscale in the original
rounding order, followed by a fused multiply-add for the block's global scale.
The SPIR-V build fixes the order of those intermediates.

On the validated Windows RX 9070 XT driver, the 128-row × 2,048-column F32
expert router uses fused accumulation and the original 128-lane reduction.
The reported counters `expert_word_dispatches` and `f32_router_dispatches`
show which specializations ran. If shader creation fails, the adapter reports
`specialized_fast_decode_fallback_reason` and uses the existing Direct kernels.

Both Q4_K and Q6_K experts stay batched and read the existing expert-cache buffers
directly, retaining source leases until their submission fence. Pipeline switches
are cached during recording, and graph reuse rebinds every selected expert source.

The existing local Qwen3-30B-A3B Q4_K_M checkpoint passed a stricter 128-step
comparison at the same 40,960-token context setting and Q8 activation mode:
**all 19,447,808 logits were bit-for-bit identical**, all 128 greedy choices
matched, and maximum absolute error and RMS error were both zero. Direct avoided
approximately 1.097 GB of GPU-to-GPU expert packing copies per token, with zero
CPU expert fallbacks. This is a short initialized-context test, not a speed or
quality verification with 40,960 occupied tokens or for every checkpoint/GPU.

Automatic continues to select Packed. Direct remains an explicit setting, and
preparation preserves that choice. The fixed Q6_K arithmetic is built into the fast Direct shader; it does not
depend on a launch-time environment variable. Historical all-array pipeline
modes remain test-only diagnostics.

A same-process packed/direct/packed comparison, with two warmups and three
128-step samples per model, measured medians of 33.05 / **35.85** / 33.65 tok/s.
Direct improved decode by 6.5–8.5% in that harness. All measured token checksums
matched; measured samples had zero weight uploads and CPU expert fallbacks.
These are controlled test-harness rates. Three live chat runs with a 31-token
coding prompt and 128-token output measured 23.65 / 34.46 / **37.61 tok/s** as
the GPU expert cache warmed, with Direct active and no expert packing copies.

The Rust tests `hardware_direct_experts_forced_trajectory_matches_packed` and
`hardware_direct_experts_paired_throughput` reproduce the correctness and
throughput comparisons with a local checkpoint.

## Resident decode on the RX 9070 XT

The short coding benchmark ran with the same Qwen3-30B-A3B Q4_K_M,
40,960-token capacity, F32 KV cache and 14,000 MiB Vulkan allocation budget.
GPU timestamps identified the routed Q4_K gate/up and Q6_K down projections as
major costs. Isolated hardware probes measured 1.43× faster Q4_K and 1.57×
faster Q6_K expert operations using packed word reads. The graph now groups
independent projections behind a single memory barrier. Windows AMD Direct
with fast math also keeps one CPU worker available while the GPU computes
short routing passes. Polling ends at 750 microseconds or when the fence
signals, whichever comes first. Batched prefill and positions from 2,048 use
the blocking wait.

Four interleaved 128-token samples began with 21 initialized coding tokens.
Median decode rose from 49.01 tok/s to 54.40 with just the expert word reads.
The specialized router and barrier grouping brought it to 55.90. Adding the
bounded wait yielded 59.35 tok/s (57.98–59.89). Every candidate matched all
19,599,744 reference logits, including the vector at the end of the prompt.
A repeat after a Windows restart measured 32.88 tok/s for the original and
38.27 tok/s for the complete optimized path on the same short prompt. All
19,599,744 logits matched again. Absolute rates varied across sessions.

For longer F32 histories, the score kernel evaluates four adjacent positions
per workgroup. It preserves the original 128-component dot-product reduction.
The weighted-value kernel shares each exponential across all 128 output
components and accumulates values in the original position order. The product
of the exponential and value rounds before the final fused multiply-add with
the reciprocal denominator. This order matched the tested Windows driver.
Selection requires Direct, fast math, the validated RX 9070 XT driver and a
32-query-head, four-KV-head model with 128 components per head. Runtime memory
statistics expose `tiled_score_dispatches` and `long_attention_dispatches`.

With 4,096 real project tokens already in cache, four interleaved 64-token
decode samples measured 5.96 tok/s for the previous Direct path and 14.59 tok/s
with both specialized attention stages. The weighted-value change contributed
most of the gain. Every candidate matched all 9,875,840 logits on the forced
trajectory. The timed passes performed zero weight uploads or CPU expert
fallbacks. The native weighted-value dispatch at 4,096 positions took a median
of 420 microseconds, down from 2,689 microseconds in an isolated GPU test.

At 40,960 populated positions a separate test compared all 1,310,720 F32
attention scores and all 4,096 weighted outputs with the original operations.
Every bit matched. The median weighted-value dispatch took 4.10 ms, down from
27.09 ms. The full-window score dispatch took approximately 12.52 ms on both
paths. Full-model decode throughput was measured through 4,160 occupied tokens.

A service-level pressure test exposed a separate arithmetic change when
decode-filled expert weights occupied the allocation budget. Prompt misses
could reach the CPU F32 fallback even when idle GPU weights could be evicted.
In two isolated 31-token prefills this produced 464 CPU expert calculations
per prompt. Direct fast mode now reclaims unleased, unpinned cache entries
before placing a supported prompt expert on CPU. Transient prompt weights
keep their existing admission priority. The genuine insufficient-capacity
fallback remains available.

## Integration benchmark, 24 November 2026

The Windows release build used the same RX 9070 XT and Qwen3-30B-A3B
Q4_K_M (`5eed9fdfc8dd2a614fa0f344befbb68d9e049978`), with a 40,960-token
window, 14,000 MiB Vulkan budget, F32 GPU KV and Q8 activations.

| Workload | Packed | Direct |
| --- | --- | --- |
| 21-token prompt, 128 decode steps | 30.03 and 30.18 tok/s | 38.01 tok/s |
| 4,096-token coding prompt, 64 decode steps | — | 15.61 tok/s |

The short run used Packed, Direct, then Packed on the same machine, with two
warmups and three measured samples per configuration. All 15 continuations
had the same token checksum. The longer run used the release CLI with one
warmup and one measured pass. Prefill took 170.06 seconds (24.09 tok/s), and
decode took 4.10 seconds. The GPU completed every selected expert with zero
CPU fallbacks. A separate 128-step forced trajectory with a 644-token prefix
matched all 19,447,808 Packed reference logits bit-for-bit.

## What AMD memory features mean here

Smart Access Memory/Resizable BAR expands CPU access to VRAM; it does not combine
64 GB RAM and 16 GB VRAM into one equally fast pool. GPU reads from system RAM and
CPU writes to VRAM still cross PCIe. The active firmware ReBAR/SAM state was not
established by this audit. Vulkan's host-visible VRAM types alone do not prove it.
[Microsoft ReBAR documentation](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/resizable-bar-support),
[AMD memory usage guidance](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html).

DirectStorage GPU destinations use D3D12 resources; adding them to this Vulkan
engine would require explicit interoperation and suitable data layout. GGUF
quantization is not a GPU decompression format. It may help future cold loading
or cache misses, but the current warmed samples perform neither disk reads nor
weight uploads.
[Microsoft DirectStorage guidance](https://github.com/microsoft/DirectStorage/blob/main/Docs/DeveloperGuidance.md).

Dedicated transfer queues and persistent staging rings remain useful for models
that stream weights. They require explicit ownership, synchronization and
measurements; they cannot speed up a zero-upload sample merely by being enabled.
[Vulkan queue guidance](https://docs.vulkan.org/guide/latest/queues.html).

To regenerate the Direct shaders, run `python3 scripts/build-direct-shaders.py`
with glslangValidator and SPIRV-Tools available, or supply `--tool-dir`. Use
`--output-dir` for an isolated reproducibility check. The script validates the
Q6 operation pattern and the hardware-tested output hash before replacing any
checked shader. The ordinary `ops.spv` and `ops_fast.spv` files are not rebuilt.
