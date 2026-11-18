> Historical snapshot for v0.5.0. Published with path redaction and link repairs; the recorded test outcomes and limitations are retained. See [the evidence policy](../README.md).

# Forge9070 0.5.0 validation record

Uses RustingStrata by Vasyl Trefilov. This source release adds an owned expert
RAM tier informed by Avalon's memory design. It preserves the existing Vulkan
backend and selected checkpoint arithmetic.

## Observed result

| Check | Result |
| --- | --- |
| Final Rust unit suite | **209 passed, 0 failed**: 186 engine and 23 frontend |
| Optimized Linux x86_64 release build | Passed, `--release --locked --offline` |
| Release CLI / CPU / local HTTP integration | **21 commands and assertions passed**, plus captured server log |
| Native direct-read coverage | Exercised: `native_direct_requests=1` |
| ISTA GSQ-RCO IQ3_S complete headers | Accepted: both shards, pinned revision |
| ToR-O3 uncensored UD-IQ4_XS complete headers | Accepted: all four shards, pinned revision |
| RX 9070 XT full inference / token rate | Not run; target hardware unavailable |
| Native Windows build and execution | Not run; Windows implementation included as source |

Exact commands, outputs and timestamps are in
[validation.json](validation.json).

## Storage and inference correctness checks

The cache tests cover byte and tensor-metadata preservation, live lease
protection, bounded payload accounting, pinned-entry fallback, eviction,
concurrent miss coalescing, model/override isolation, failed reads without
partial publication, bounded LRU bookkeeping, counter rollover, warmup
non-eviction and the owned-hit fast path. These are behavioral tests; no GPU
speed is inferred from them.

CPU integration tests compare cache-disabled and cache-enabled single-token,
multi-token and scalar-reference calculations. A selected-expert batch larger
than the cache exercises borrowed fallback while other leases remain live.
Full scalar prefill and precomputed embedding prefill produce the same logits
and state as their reference paths while preserving existing cache residency
and LRU ordering. Ordinary decode resumes admission afterward.

The native read helper is checked on unaligned ranges, chunk boundaries, EOF,
invalid/overflowing ranges, buffered operation, a failed native read after
partial output, retained source identity after pathname replacement, and split
GGUF maps. A separate test first probes a correctly aligned OS read. When it
succeeds, the unaligned logical wrapper must return `Direct` with exact bytes.
That path was exercised on this Linux filesystem; it was not skipped or
satisfied through the buffered fallback.

A native-direct counter describes a successfully completed O_DIRECT or
FILE_FLAG_NO_BUFFERING request. It does not independently measure internal
filesystem caching. The cache uses pageable owned storage, not GPU-registered
or permanently resident host memory.

The final suite command was:

```bash
cargo test --workspace --lib --bins --offline -- --nocapture
```

Some pre-existing CUDA availability tests return early without a CUDA runtime.
No CUDA or Vulkan model computation is claimed from the unit-test result.
The final run is identified by `final_command_index` in
[rust-tests-0.5.0.json](../rust-tests-0.5.0.json); earlier
intermediate runs are retained and are not added to the 209-test total.

## Actual release integration

The release executable was built with:

```bash
cargo build --release --locked --offline
python3 scripts/validate-expert-ram.py --exe target/release/forge9070
```

The script uses only the bundled tiny synthetic four-expert model. The
release reports version 0.5.0. The `memory` preview counts its actual 98,304
expert bytes, respects disabled mode and rejects an invalid ceiling. The
`tune` interface exposes the new memory controls.

Mapped, warmed-owned and on-demand-owned CPU generation returned identical
output on that deterministic fixture. The warmup log confirmed four experts
loaded. A private local HTTP server then:

1. Listed two independent catalogue entries.
2. Loaded and warmed the first entry.
3. Generated a chat response using the owned cache.
4. Reported cache hits without exceeding its payload ceiling.
5. Switched to the second entry with fresh cache counters.
6. Unloaded and cleared the active cache status.

The first load reported 98,304 resident bytes, four warm admissions and twelve
native direct matrix reads. After chat, it reported eight ordinary hits and
52 prefill hits. The second model began with four warm admissions and zero
ordinary hits, demonstrating that the former model's entries/counters were
not retained as the new model's cache.

Records:

- [build-release-0.5.0.json](../build-release-0.5.0.json)
- [expert-ram-integration-0.5.0.json](../expert-ram-integration-0.5.0.json)
- [scripts/validate-expert-ram.py](../../../scripts/validate-expert-ram.py)

These are synthetic functional checks in the validation environment. They are
not Qwen3.8 quality tests or measurements of Jamie's memory capacity or GPU.

## Real Hugging Face header checks

The **0.5.0 release executable** checked the selected file and every required
shard using bounded header requests. Neither complete checkpoint was downloaded.

| Model | Immutable revision | Shards | File bytes | Expert bytes |
| --- | --- | ---: | ---: | ---: |
| `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_S | `ed59f92082b1e93c0e96d60a8b11aab089b52f09` | 2 | 83,617,662,656 | 50,292,326,400 |
| `ToR-O3/Qwen3.8-Flash-Next-Uncensored-GGUF`, UD-IQ4_XS | `afa27ea7af1f83171a0aeafa8d5918eaa0dbfeee` | 4 | 93,672,359,738 | 60,384,870,400 |

Both report `qwen4exp`, 48 layers, 512 experts with 10 active per token,
1,224 tensors, supported tensor types and no metadata issues. The PLE payload
is 28,800,138,240 bytes in each, and is separate from owned expert residency.

Records:

- [ISTA complete header check](../qwen38-header-checks/ista-gsq-rco-iq3-s-v050.json)
- [ToR-O3 complete header check](../qwen38-header-checks/uncensored-tor-ud-iq4-xs-v050.json)
- [Memory inventory and source audit](../qwen38-memory-inventory.json)

The source audit also inspected Avalon's custom packed file. Its custom
IQ3_K/IQ3_KS types and layout remain unsupported. Standard-format GSQ-RCO IQ3_S
is the relevant compatible source used before Avalon's custom re-encoding;
allowing custom IDs without their arithmetic would not be a correct port.

## Practical limits

The automatic budget uses memory available at load time, with OS, runtime and
prompt-cache reserves. Tests include a simulated 62 GiB host with 53 GiB
available: its default expert ceiling is 47.5 GiB, enough for the inspected
46.84 GiB GSQ-RCO expert payload but not the 56.24 GiB UD-IQ4_XS payload. The
latter uses partial residency and disk-backed cold reads. Installed capacity
alone is not a reservation or proof of performance.

The complete Qwen3.8 checkpoints have not been loaded or decoded here. Windows
unbuffered reads, RX 9070 XT Vulkan mailbox behavior, GPU allocation, native WSL
Vulkan execution and real token rates still require execution on that machine.
The Windows code follows the documented native file APIs; source review and
Linux testing are not Windows runtime validation.

This release adds the RAM tier, direct-read attempts, warmup and accounting.
It does not implement Avalon's HIP-registered arena, packed custom quantization,
Qwen4 MTP, Q4 value cache, cost-fitted dispatcher or elastic VRAM lending.
Qwen3.8 continues to require one generation slot and MTP disabled. Clients may
queue multiple requests, and original/uncensored models can be kept on disk and
switched with one active model at a time.

The cache preserves the selected checkpoint's quantized bytes and all routed
experts. GPU/CPU rounding, quantization quality, factual accuracy and the effects
of a community uncensored modification remain separate evaluation questions.

## Historical records

The previous top-level validation record is archived in [v0.4.1/](../v0.4.1).
Earlier files explicitly labelled 0.4.1 or 0.4.0 remain historical evidence;
they are not presented as additional tests run for 0.5.0.
