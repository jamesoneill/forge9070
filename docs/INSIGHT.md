# Nextora Insight — evidence-based hardware performance advice

**Status:** First implementation; read-only diagnostic CLI. Not a benchmark database, live profiler, tuning engine or upgrade-price service.

Nextora Insight helps identify the *next useful measurement* before suggesting an upgrade. It uses the native engine's existing Vulkan adapter discovery and benchmark output. It does **not** install drivers, change model settings, send telemetry, scrape prices or claim a faster GPU will deliver a particular token rate.

## Quick start

Build the native application:

```sh
cargo build --release --locked -p nextora-engine
./target/release/nextora insight
```

Windows PowerShell:

```powershell
.\target\release\nextora.exe insight
```

The default command reports logical CPU thread availability, currently available host RAM and the existing Vulkan adapter capability snapshot. It works without downloading a model, and can report a missing Vulkan loader without treating the host GPU as defective.

After **successfully** loading a compatible real GGUF checkpoint, record a repeatable baseline. Use the exact same checkpoint revision and quantization in comparisons:

```sh
./target/release/nextora bench --model /models/checkpoint.gguf \
  --backend auto --ctx 4096 --prompt "Explain this Rust compiler error." \
  --decode-steps 128 --reps 5 --warmup 1 --requests 1 \
  --json-out baseline.json

./target/release/nextora insight --bench-report baseline.json --json-out insight.json
```

You can pass multiple `--bench-report` values (up to 16); none is required. Reports must be regular files no larger than 8 MiB. If another process may edit the input files during analysis, run Insight on immutable copies. JSON output uses the existing atomic report writer, and is rejected when its destination aliases an input report.

A benchmark is only admitted if it is a native Nextora report with matching sample count, positive finite aggregate rates and latency, workload metadata and *passing identical greedy checksums*. These checks are **not** a substitute for independent model-quality validation. The command strips model paths, prompt text and raw per-request samples from the exported benchmark summaries.

## Interpreting recommendations

| Evidence | Possible advice | What it does **not** establish |
| --- | --- | --- |
| No eligible Vulkan hardware in the running process | Inspect the loader/driver; on WSL, try the Windows executable using the Windows GPU driver | That you need a new GPU |
| Eligible GPU but no physical Vulkan benchmark supplied | Measure a real supported checkpoint using `--backend vulkan` | That the GPU or shader path actually works |
| No measurements | Record a fixed five-repetition baseline | Any predicted token rate |
| One active request measured | Compare 1, 2 and 4 requests | Linear concurrency scaling |
| Tiny synthetic model | Repeat with real pretrained weights | Useful-model throughput or answer quality |
| Model mapped file larger than GPU heap | Investigate allocation, expert-cache misses and CPU fallback | That all mapped weights need to reside in VRAM |
| Less than 4 GiB available RAM at discovery | Inspect memory pressure under the real workload | That adding RAM will speed up generation |

All upgrade suggestions remain **conditional**. Neither host RAM availability nor the largest Vulkan memory heap measures live bandwidth, GPU occupancy, thermal throttling, disk throughput, PCIe transfers or sustained VRAM pressure.

## Hardware and privacy boundaries

- Linux native and WSL hardware discovery can differ drastically. `/dev/dxg` and a software Vulkan adapter such as `llvmpipe` do not establish access to a compute-capable hardware Vulkan device.
- Discovery reads GPU *capability* information, not a successful model execution benchmark.
- Insight does not capture usernames, serial numbers, model paths or prompts in its output. Hardware adapter names and an optional host timestamp are included, so review a report before sharing it.
- The benchmark output you provide to Insight can still contain paths and prompts on your local disk; Insight only emits a reduced summary.
- No external network calls or server-side benchmark uploads are added.
- No raw performance comparisons are attempted between reports lacking driver and checkpoint fingerprints.

## Next milestones

1. Measure physical AMD RX 9070 XT execution under Windows Vulkan and verify correctness against a deterministic CPU/reference run.
2. Record model hashes, driver revisions, OS build, context, quantization, backend and observed allocations to make A/B comparisons auditable.
3. Add live host/GPU memory, transfer and cache-miss counters with privacy-safe export controls.
4. Benchmark local tuning candidates only after correctness gates, comparing median and tail latencies on representative workloads.
5. Add an opt-in price-aware hardware upgrade planner based on measured outcomes, not vendor marketing estimates.

**Project principle:** Optimise first. Recommend upgrades second. Spend money last.

See [model loading and quality](MODEL-LOADING.md), [hardware support](HARDWARE-SUPPORT.md), [validation evidence](../VALIDATION.md) and [roadmap](ROADMAP.md).
