# Nextora Insight — evidence-based hardware performance advice

Use `nextora insight` for a hardware snapshot, `nextora insight-compare` to compare saved reports, and the [local console](#live-performance-dashboard) to see measurements from the running service.

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
- Comparisons without driver and checkpoint fingerprints are exploratory.

## Next milestones

1. Measure physical AMD RX 9070 XT execution under Windows Vulkan and verify correctness against a deterministic CPU/reference run.
2. Record model hashes, driver revisions, OS build, context, quantization, backend and observed allocations to make A/B comparisons auditable.
3. Extend the console’s available RAM, owned allocation and cache snapshots with transfer, memory-pressure and cache-miss measurements.
4. Benchmark local tuning candidates only after correctness gates, comparing median and tail latencies on representative workloads.
5. Add an opt-in price-aware hardware upgrade planner based on measured outcomes, not vendor marketing estimates.

**Project principle:** Optimise first. Recommend upgrades second. Spend money last.

See [model loading and quality](MODEL-LOADING.md), [hardware support](HARDWARE-SUPPORT.md), [validation evidence](../VALIDATION.md) and [roadmap](ROADMAP.md).

## Live performance dashboard

Start the single native service with `nextora serve --open-ui --backend auto`. The existing browser console at `http://127.0.0.1:9070/` includes a **Nextora Insight** section and obtains its readings from authenticated `GET /api/insight/live`. Its read-only panel shows:

- Recent completed request prefill and output throughput, with per-request bars and success/failure counts. Generation timing includes sampler and streaming overhead, not just GPU kernel execution.
- Currently available RAM and Nextora-owned GPU allocations; neither is installed RAM nor total/free device VRAM.
- Current queue depth, active request slots, expert-cache footprint and CPU expert-fallback observations (when supported by the loaded runtime).
- Free-first recommendations for additional measurement and conditional hardware-capacity suggestions.

The service keeps at most 64 completed request samples in memory, resetting on restart. Its inference route records incomplete requests separately and does not use them to claim throughput. UI data never contains prompts, generated text, checkpoint paths or personal identifiers. The GET endpoint uses the service's existing Host, Origin and Bearer-token authentication checks; do not expose a service without an adequate API key.

**Interpretation:** aggregate rates across recent completed requests are *not* a GPU utilisation percentage; trends need a representative workload. The endpoint samples current state, and does not yet capture thermal throttling, system paging, disk stalls or external GPU allocations.

## Exploratory comparisons

Use two completed native benchmark reports:

```sh
./target/release/nextora insight-compare --baseline baseline.json --candidate candidate.json --json-out comparison.json
```

Insight rejects differences in recorded request count, prompt-token count, decode-step count, mapped model bytes or execution backend. It displays the observed throughput ratios but labels the outcome **exploratory, not controlled**, because the existing reports lack full checkpoint checksums, input-token digests, driver identity and hardware metadata. Two models with identical file size can produce completely different results. A ratio must **not** be presented as an upgrade speedup or a quality guarantee.

## Future hardware upgrade simulator

A purchase calculator requires verified measurements from both candidate and current hardware, applicable compatibility checks, actual component pricing, power/cooling constraints, and common workload/model fingerprints. No live hardware prices or hypothetical performance predictions are implemented in this release. The upgrade recommendation remains **insufficient evidence for purchase** until those data exist.

## Host expert cache observations

The live endpoint now includes cumulative host expert-cache hits, misses, evictions, mapped bypasses, prefill hits/misses, direct/buffered reads, read errors and encoded payload bytes read. The console shows selected counters. These come from the engine's existing bounded expert-cache statistics; counters reset on model replacement and are not normalised by request. A high miss count is a profiling prompt, **not** evidence that a more expensive GPU or RAM kit will improve performance. Expert read errors are diagnostic. This release does not time transfers, profile PCIe utilisation, measure GPU compute occupancy, or tune expert placement automatically.
