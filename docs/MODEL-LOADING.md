# Model loading: quality and performance validation

Nextora uses a single Rust inference engine. Its supported GGUF format,
architecture, tokenizer and GPU requirements are defined by the actual code,
not by model or publisher names. This guide describes how to tune a real
checkpoint without trading away correctness silently.

## Startup path

1. **Map and validate the GGUF shards.** Confirm all parts exist, agree on
   metadata, are non-overlapping and fit addressable space. The loader uses
   memory maps for weight storage; mapping byte counts are not physical RAM use.
2. **Construct and validate the tokenizer before probing the GPU.**
   Unsupported tokenizer metadata fails before Vulkan device creation.
3. **Resolve execution policy.** `auto` picks eligible Vulkan hardware or CPU
   if capability discovery cannot find a usable adapter. An explicit device or
   Vulkan memory setting remains strict; model, shader, allocation or execution
   failures never silently retry on CPU.
4. **Plan host/cache memory.** Account for state slots, OS reserve and mapped
   versus owned expert bytes. A cache limit is a ceiling, not a promise of RAM
   availability. GPU expert storage has a separate bounded budget.
5. **Build the runtime and verify tokenizer/model vocabulary agreement.**
   Only a fully accepted model proceeds to optional expert-cache warmup.
6. **Warmup when appropriate.** Preloading routed experts can reduce later
   cold-cache stalls but increases startup and I/O. `--expert-warmup false`
   optimizes first-load latency on storage- or RAM-constrained hosts;
   `--expert-warmup true` may help repeated long-running decode jobs.
   Neither setting changes quantization precision or intentionally skips
   any routed expert.

The runtime report includes `model_load_timings_ms`: GGUF mapping, tokenizer
construction, backend selection, memory planning, runtime initialization,
expert warmup, and total load time. Times include platform and filesystem
effects and must not be represented as pure device-compute benchmarks.

## Validate quality before tuning speed

- Keep **the exact same revision and quantization** of a checkpoint in all
  comparisons; report its shard checksums and GGUF metadata.
- Compare fixed-token, deterministic CPU and Vulkan outputs, tensor kernels,
  prefill versus repeated decode, and sampled prompts at controlled seeds.
  Floating-point reductions may differ slightly across devices; define a
  numerical tolerance appropriate to the operation and check task-level output.
- Verify the tokenizer's chat template and control/stop IDs. A raw-prompt
  benchmark does not validate conversation formatting.
- Do not silently lower KV precision, context length, routed-expert count, or
  enable approximate decoding in a comparison advertised as equal quality.
- Distinguish passing synthetic fixtures and software-Vulkan tests from real
  pretrained-model quality on physical hardware.

## Measure representative workloads

Use the CLI with the **same prompt, model, context and backend policy** for
each run. Example on an already downloaded and validated checkpoint:

```sh
./target/release/nextora inspect --model /models/checkpoint-00001-of-00002.gguf
./target/release/nextora bench --model /models/checkpoint-00001-of-00002.gguf \
  --backend cpu --ctx 4096 --prompt "Explain a Rust borrow checker error." \
  --decode-steps 128 --reps 5 --warmup 1 --json-out cpu-bench.json
./target/release/nextora bench --model /models/checkpoint-00001-of-00002.gguf \
  --backend auto --ctx 4096 --prompt "Explain a Rust borrow checker error." \
  --decode-steps 128 --reps 5 --warmup 1 --json-out auto-bench.json
```

The separate CLI benchmark's `--warmup` option is benchmark-run repetitions,
not the same thing as `--expert-warmup`. Measure each supported cache mode
independently. Repeat cold- and warm-file-cache tests, record time to first
token, prefill tokens/s, decode tokens/s, peak RSS/VRAM, cache evictions and
device execution. Report median and tail latency, with at least five measured
repetitions and a stable baseline. Include concurrent requests only if
the intended workload uses them.

## Acceptance gates

- All hosted CI checks, including Windows and Linux CPU/shader/browser/software-
  Vulkan tests, pass on the exact commit being merged.
- Full checkpoint load and generation succeeds on real target hardware.
- Quality parity passes defined reference prompts/tolerances for that model.
- No unexpected memory growth, cache-lease invalidation, or silent backend
  switch under load and cancellation.
- Performance is compared to the prior shipped revision on the same hardware
  and driver, with complete numeric results. If a change is slower, document
  whether it improves quality or safety rather than calling it an optimisation.

No claim is made that Nextora is universally optimal for every GPU, model
architecture, quantization, storage device or workload.
