> **Historical compatibility guide:** The previous Strata-based application
> and port 8080 have been retired from Nextora. The original documentation
> below is retained solely as migration reference and may describe commands,
> cache settings or API behaviour that the native Rust service does not offer.
> For the active runtime, use [the native guide](NATIVE-ENGINE.md) and
> [migration details](COMPATIBILITY.md).

# Measuring generation speed without deliberate expert omission

**Scope:** this guide documents the existing compatibility application. For the
independent Forge 0.9 runtime, its Vulkan/CPU settings, coding API and queue,
use [NATIVE-ENGINE.md](NATIVE-ENGINE.md). Flags and endpoints differ.

Nextora 0.6 targets an RX 9070 XT through native Rust/Vulkan. It has not been
benchmarked on the target card. The objective is the best measured individual-reply
performance for a specified model and execution configuration, subject to the
included output-comparison gate. It is not a global hardware optimum or a
promise of factual model accuracy.

For large code projects, use [LARGE-CODE-PROJECTS.md](LARGE-CODE-PROJECTS.md).
The `code` / `code-long` presets increase context and output allowance, reserve
1 GiB for parked conversation states, preserve earlier reasoning and use the
publisher's Qwen3.8 thinking-mode sampling defaults. They remain chosen
starting settings. Measure representative repository excerpts before selecting
larger chunks, context or a different worker count.

## Starting configuration

| Setting | Baseline | Reason |
| --- | --- | --- |
| Active GPU requests | 1 | More state/expert-cache space for one reply |
| Expert omissions | Disabled | Every expert selected by the router contributes |
| KV representation | F16 | Avoid additional int8 KV quantization |
| MTP | 0 until measured | A valid head is necessary, but acceleration is workload-dependent |
| Context | 4096 | A moderate initial allocation; increase deliberately for longer work |
| Prompt chunk | 256 before tuning | Modest scratch allocation; measured candidates may be better |
| VRAM margin | 2048 MiB | Leave an initial allowance for the desktop and driver |

Quantized model weights remain quantized. F16 state also rounds floating-point
values. The word fidelity here refers to the selected model's computation;
it does not mean full-precision equivalence or guaranteed correct answers.
See [ACCURACY.md](ACCURACY.md) for checkpoint and task-quality evaluation.

## 1. Verify the execution path

Build on the PC that will run the program so the CPU expert path can use its
available vector instructions:

```bash
bash scripts/setup-wsl.sh --build
./target/release/nextora-cli doctor --require-gpu
./target/release/nextora-cli inspect --model /path/model.gguf
./target/release/nextora-cli check --model /path/model.gguf
```

`doctor` checks discovery/capabilities and `check` checks pipeline shared-memory
requirements. Neither runs a complete inference workload. Follow them with a
short real generation before tuning. Native WSL Vulkan depends on the installed
graphics stack; the Windows-executable launcher is a distinct route. See
[WSL.md](WSL.md).

The shared model policy checks actual architecture, routing, quantization,
tensor geometry, shards and byte bounds. It also reports low-bit tensors and
whether routed experts use them. Format acceptance and file size do not certify
GPU/host memory fit or answer quality. No pretrained checkpoints are included.

## 2. Establish a measured baseline

Use representative prompt text when available:

```bash
./target/release/nextora-cli bench --model /path/model.gguf \
  --ctx 4096 --pp 128,512 --tg 128 --reps 3 --warmup 1 \
  --kv f16 --mtp 0 --json-out baseline.json
```

`bench` separates prompt processing from generation, excludes the requested
warmup repetitions and records individual token/time samples as well as summary
rates. `--prompt-file` supplies raw representative text for this benchmark.
Short synthetic prompts are useful for checking wiring but not for judging a
large model's realistic performance or quality.

Model weights and cache warmth, CPU/RAM bandwidth, available VRAM, prompt length
and context all influence results. Paging model weights from disk can dominate
runtime. Do not attribute a rate change to one flag if the checkpoint, workload,
cache state or other settings also changed.

## 3. Tune decode settings with an output gate

Stop a separately running GPU server before using the CLI tuner, so both
processes do not compete for VRAM and compute during measurement.

```bash
./target/release/nextora-cli tune --model /path/model.gguf \
  --ctx 4096 --kv f16 --tg 128 --reps 3 --warmup 1 \
  --json-out tuning.json
```

The tuner holds the checkpoint, KV representation, context, memory margins and
CPU worker count fixed. It considers prompt chunk and supported embedded MTP
choices together: a large prompt buffer can reduce expert residency and harm
generation even if prompt processing itself is fast.

The procedure uses chat-formatted workload prompts, excluded warmup and at
least three measured repetitions. Every sample pairs its actual generated token
count with its own elapsed decode time. Median generation throughput drives
selection; prompt timings are reported separately. The tuner uses the same
adaptive expert-placement policy as ordinary server/generation/benchmark runs.
Thinking remains enabled for these workloads and is recorded; reasoning tokens
count as generated tokens. This is not a measurement of visible answer tokens alone.

A candidate must match the no-MTP reference's greedy token IDs on every tested
prompt and measured repetition. Unstable or mismatching candidates are rejected.
The reference is retained unless an eligible candidate improves median measured
generation throughput by at least 3%. Failed allocations or unsupported MTP heads are recorded as such,
not reported as successful acceleration.

Use `--prompt-file workload.txt` to replace the built-in workloads with one
representative user prompt, or `--threads N` to hold a different CPU worker
count. The global worker pool cannot be resized reliably during a process;
compare worker counts through separate invocations. The tuner does not sweep
checkpoint precision, KV precision, GPU drivers or every possible memory budget.

The JSON report retains the workload/configuration, raw measurements, output
comparisons and chosen settings. Use the printed flags explicitly. The browser's
calibrate action runs the same method with default workload options and reloads
with the accepted recommendation. Calibration can take several minutes on a
large model and occupies the inference device during that time.

A finite greedy comparison is a conservative regression check. It is not a
proof of identical outputs for all prompts, sampled decoding, long contexts or
other devices, and does not assess whether the reference answer is correct.
Greedy temperature zero is used for this comparison only; use the model
publisher's supported sampling guidance for actual work.

### Calibration cache scope

Version-5 records include model/shard/companion identities and runtime settings,
including expert RAM budget, policy, warmup and prompt-cache reserve.
Version-3 prefill/whole-request recommendations are ignored. Changed context,
KV, worker count, numerical mode, memory margin or decode-cache policy requires
matching calibration. Prompt identity and measurement details remain in the
report; a result from one workload need not generalize to another.

Identities use canonical paths, lengths and modification times, not complete
checkpoint checksums. Editing bytes while preserving that metadata requires
manual cache clearing. The GPU key records name, index and total VRAM rather
than a hardware UUID or driver version; recalibrate after a driver change.

### MTP and KV precision

Ordinary MTP needs a usable embedded head or a matching standalone GGUF supplied
through `--mtp-file` for normal generation/serving. Automatic tuning considers
embedded heads only. Matching geometry cannot establish training provenance or
useful acceptance rates. EAGLE3 heads and HC/PLE/QSA configurations are excluded
from this profile's ordinary MTP path.

Speculative acceptance verifies drafts against the configured target model.
Changed kernel batching or CPU/GPU expert placement can still change rounding
near tied logits. A matching calibration run is useful evidence for its tested
workload, not a universal bit-identity claim.

Q8 KV remains explicitly available with `--kv q8`; it can free VRAM for expert
weights. Compare its answer quality and performance against F16 on your tasks.
Tuning a Q8 run compares candidates against a Q8 baseline, not against F16, so
that run alone does not establish that reducing KV precision preserves quality.
The tuner never changes KV representation or checkpoint quantization silently.

## 4. Measure multi-client throughput separately

One checkpoint is shared by all slots; each active request has separate state.
The default is one slot. For several clients:

```bash
./target/release/nextora-cli serve --model /path/model.gguf \
  --parallel 2 --mtp 0 --kv f16 --ctx 4096 --max-tokens 512
```

Then, using the server's administrative API key if configured:

```bash
python3 scripts/bench-concurrency.py --clients 2 --requests 8 \
  --max-tokens 128 --warmup 1 --json-out concurrency-c2.json
```

This benchmark defaults to one client. Its aggregate rate is actual completion
tokens divided by measured batch wall time, including queuing, prefill and
transfer. Per-request latency and server decode-rate fields are separate.
Warmup is excluded and reused prompt-cache tokens are recorded. The script reads
resolved model/execution settings before and after the run, including model
identity, KV format, chunk size, threads and cache policy. Changed reported
settings, stale IDs or failed requests invalidate the result. The v0.4 server
and administrative metadata access are required for that configuration check.

CPU mode accepts overlapping connections but computes generation serially.
Supported GPU MoE paths permit up to four active sequences. Hyper-connections,
PLE and QSA require one slot, and MTP cannot be combined with multiple slots.
More context state can displace cached experts, so compare individual latency
and aggregate throughput before increasing slot count.

## Validation limits

The package tests include actual synthetic MoE CPU inference, model switching,
overlapping JSON/SSE clients, parameter rejection and measurement accounting.
Pure tests cover GPU input-position preparation, sampler error handling and
candidate selection. No target GPU, native WSL, large-checkpoint inference or
actual GPU calibration was available during package preparation.

The Forge profile no longer exposes deliberate expert-tail omission. Its
historical v0.3 implementation remains in the vendored source and attribution
record. The [Avalon comparison](AVALON-COMPARISON.md) identifies remaining kernel
and memory-management differences; this package does not promise Avalon parity
or a particular token rate. See [../VALIDATION.md](../VALIDATION.md) for the
observed checks and their precise scope.
