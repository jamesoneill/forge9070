> **Historical compatibility guide:** The previous Strata-based application
> and port 8080 have been retired from ForgeAI. The original documentation
> below is retained solely as migration reference and may describe commands,
> cache settings or API behaviour that the native Rust service does not offer.
> For the active runtime, use [the native guide](NATIVE-ENGINE.md) and
> [migration details](COMPATIBILITY.md).

# Owned expert RAM cache in Forge9070 0.5

**Scope:** this guide documents the existing compatibility application. For the
independent Forge 0.9 runtime, its Vulkan/CPU settings, coding API and queue,
use [NATIVE-ENGINE.md](NATIVE-ENGINE.md). Flags and endpoints differ.

The large GGUF download is not a required RAM allocation. Forge now separates
three kinds of weight storage: dense weights placed on the GPU, routed experts
cached in owned host memory and on the GPU, and the large sparse PLE table whose
rows are read only when the model needs them.

This is a new Rust implementation informed by Avalon's
[host arena](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/src/ggml-cuda/moe-arena.h),
[RAM budgeting](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/src/ggml-cuda/moe-ram-budget.h)
and [lazy PLE loading](https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/src/models/qwen4exp.cpp).
It retains RustingStrata's Vulkan backend and existing quantized kernels.
Uses RustingStrata by Vasyl Trefilov.

## What is implemented

- An owned LRU cache groups each expert's gate, up and down matrices, preserving
  the original quantized bytes. CPU execution, GPU cache fills and prompt
  staging all obtain views through leases. Entries cannot be evicted or
  overwritten while a lease exists. If the remaining entries are busy, a miss
  uses the original mapped bytes instead of exceeding the payload ceiling.
- Cache hits avoid reconstructing tensor names or resolving the three source
  matrices. The LRU index is bounded, and normal insertion/replacement uses
  logarithmic index operations rather than scanning every expert.
- Each model owns its cache. Reloading releases it after inference workers have
  stopped; changing weight overrides invalidates previous entries. Live old
  leases remain counted until they can be released safely.
- Default startup warmup populates only the bounded expert working set. It uses
  the saved model-specific routing profile where available and does not evict
  already cached entries. It never scans or preloads the PLE table.
- Prompt processing can use owned hits without changing their LRU order.
  Prompt misses are transient mapped reads. This protects the generation
  working set, but is not Avalon's borrowed-slot prefill implementation and does
  not establish faster cold prefill. The previous coalesced prompt path remains
  available with `--expert-ram 0`.
- Memory-mapped-page advice is applied only to addresses belonging to retained
  model maps. Owned heap storage never receives `MADV_DONTNEED`, which could
  otherwise discard live weights. Asynchronous raw-address prefetch is bypassed
  on the owned-cache path.

## Read weights without an additional mandatory file-cache copy

Each GGUF shard retains the same file object that backs its mapping. Expert
cache admissions use positional reads into owned storage. Linux attempts
`O_DIRECT`; Windows attempts `FILE_FLAG_NO_BUFFERING`. Reads use aligned,
bounded scratch space and copy exactly the requested matrix bytes, including
unaligned offsets and a requested region ending at EOF.

If the native mode cannot open or complete a read, the helper retries using
buffered positional I/O on that same file object. It never reopens a changed
pathname after mapping. It publishes an expert only after all three matrices
have been read successfully. Sparse PLE lookups still use ordinary mmap access.

The `direct_reads` and `buffered_reads` counters count completed matrix-read
requests. A successful direct request does not prove that every filesystem
internally bypassed its cache. Owned storage is pageable Rust memory; it is not
HIP-registered, Vulkan-imported or locked against the operating system's pager.
Windows code has not been executed in this Linux validation environment.

## The budget

For each model load, Forge takes a new available-memory snapshot. Linux uses
`MemAvailable` and the smallest finite cgroup-v2 ancestor headroom, respecting
`memory.max` and `memory.high`; Windows uses available physical memory reported
by `GlobalMemoryStatusEx`. Unknown cgroup usage under a known ceiling does not
count as free memory. The installed total alone never sizes the automatic tier.

The default expert ceiling is the smaller of the actual expert payload and:

```text
available host RAM
  minus 4 GiB OS/application margin
  minus 1 GiB staging/runtime reserve
  minus 512 MiB prompt-state cache reserve
```

An explicit `--max-ram` further restricts the expert tier after subtracting
estimated owned model memory and future runtime/prompt reserves. Existing model allocations
already reduce the available-memory snapshot; they are not subtracted twice.
The small `memory` preview command runs before tokenizer/model allocation, so
the load-time ceiling can be lower than its preview.

All these numbers describe startup budgeting. Other applications can allocate
memory later, and the operating system can page owned storage. The payload cap
does not include every allocator header or runtime allocation; the separate
reserves account conservatively for those. It is not an operating-system hard
limit or a guarantee that all allocated pages stay physically resident.

| Option | Meaning |
| --- | --- |
| `--expert-ram auto` | Use detected available memory and the actual expert inventory; default |
| `--expert-ram 40960` | Request at most 40 GiB, still clamped by known available memory and other bounds |
| `--expert-ram 0` | Disable owned expert caching and keep the original mapped-weight path |
| `--expert-warmup true` | Fill the bounded working set at load time; default |
| `--expert-warmup false` | Fill on demand; loading is quicker but initial decode may need disk reads |
| `--ram-margin 8192` | Leave an 8 GiB OS/application margin |
| `--cache-ram 512` | Reserve 512 MiB for prompt-state reuse; separate from expert storage |
| `--max-ram 49152` | Add a 48 GiB soft ceiling for estimated owned process memory |

The same options are available to `tune`. Calibration records include the
effective expert budget and policy, so results measured with a different RAM
configuration are not silently reused.

## What approximately 62 GB means for Qwen3.8

Inspected immutable GGUF headers give these actual payloads:

| Quantization | Routed expert payload | Lazy PLE table | Other weight payload |
| --- | ---: | ---: | ---: |
| ISTA GSQ-RCO IQ3_S | 46.84 GiB | 26.82 GiB | 4.20 GiB |
| Unsloth / ToR-O3 UD-IQ4_XS | 56.24 GiB | 26.82 GiB | 4.17 GiB |

For example, 53 GiB **available at load time** leaves 47.5 GiB under the default
reserves. The ISTA expert payload can fit that ceiling; the larger UD-IQ4_XS
uses a partial cache. If less memory is available, both can use a smaller cache
and read cold experts from disk. This is a capacity calculation, not a token
rate or a guarantee that the GPU side loads on a particular driver.

GB and GiB differ: 62 decimal GB is approximately 57.74 GiB. Forge uses detected
bytes rather than guessing which unit was shown by a system information tool.

The GSQ-RCO file is the standard-format source of Avalon's expert quantization,
before its custom re-encoding. Its quantization still changes model precision.
The cache itself preserves the chosen checkpoint's bytes and retains every
selected expert; it does not make an uncensored derivative equally accurate.

## Inspect and measure on the target PC

```powershell
& $Forge inspect --model $Model
& $Forge memory --model $Model --expert-ram auto --json
& $Forge chat --model $Model --expert-ram auto --expert-warmup true --ctx 4096 --kv f16 --parallel 1 --mtp 0
& $Forge bench --model $Model --expert-ram auto --pp 128 --tg 128 --reps 3 --warmup 1 --json-out '.\ram-cache-bench.json'
```

Use a real model path and the executable built on that PC. The startup log
reports the payload, available RAM, ceiling and reserves. Run summaries,
benchmark JSON and server stats include owned-cache bytes, hits, misses,
evictions, fallback counts and completed I/O modes. `VALIDATION.md` distinguishes
CPU/storage tests performed here from unperformed RX 9070 XT execution.

Avalon's custom packed IQ3_K/IQ3_KS model, its cut-head Qwen4 MTP, HIP pinned
arena, elastic VRAM lending and Q4 value cache remain separate work. Enabling
their names without the matching arithmetic or layout implementation would
produce an invalid model, so those unsupported formats stay rejected.
