# Native concurrency benchmark sweep

The local script `scripts/benchmark-concurrency.py` runs the **existing Rust `nextora bench`** with one, two and four active request slots. It is not a new inference backend and does not change service settings.

```powershell
python scripts/benchmark-concurrency.py --binary .\target\release\nextora.exe --model 'F:\Models\supported.gguf' --backend vulkan --prompt 'Explain Rust ownership.' --ctx 4096 --decode-steps 128 --reps 5 --warmup 1 --output concurrency.json
```

Use `--backend cpu` for a CPU-only baseline. If Vulkan is unavailable, the strict backend request fails rather than silently falling back to CPU. The checkpoint must already be supported and complete. Compare actual backend/adapter identity across runs; mismatched results are rejected.

## Interpretation

Each recorded run includes median aggregate prompt-processing rate, median aggregate decode-step rate, and p95 decode completion time. Output ratios are **observed**, not predictions or universal scalability claims. Model file name and size are recorded, but full checkpoint shard hashes, exact GPU driver version, operating-system build and thermal/power conditions are *not* captured. This is not a coding-task success benchmark.

No model downloading, external web calls or telemetry uploads occur. The script does not put prompt text into the JSON output, though prompts may appear in shell process arguments and history. Fix any real-quality issues before describing a faster setting as better.

Recommended RX 9070 XT acceptance: repeat after independent CPU/Vulkan output-quality tests; record actual driver, checkpoint digests, context, memory high-water, prefill/decode speed, and 1/2/4-slot tail latency on the same machine.
