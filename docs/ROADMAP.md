# Roadmap

Nextora prioritizes measured token generation speed without silently changing
which routed experts contribute. This roadmap is a set of engineering priorities,
not a delivery schedule or hardware support guarantee.

## Release acceptance

| Priority | Work | Acceptance evidence |
| --- | --- | --- |
| 1 | Run the native engine on the actual RX 9070 XT under Windows | Driver/device identity, successful GPU computation, numerical checks and reproducible logs |
| 2 | Load the complete selected GSQ-RCO checkpoint | Exact file hashes, startup/peak memory, representative prompts and output comparisons |
| 3 | Measure coding workloads | Prefill, time to first token, decode speed, latency distribution and task correctness |
| 4 | Compare one to four active sequences | Aggregate throughput, per-request latency, memory use and queue behavior under cancellation |
| 5 | Evaluate long contexts and community variants | Explicit context/quantization budgets, correctness checks and failure-recovery results |

## Performance investigations

- Profile real checkpoint execution before selecting further kernel work.
- Investigate transfer/compute overlap only with independent correctness checks.
- Evaluate cache policy under real expert-routing traces and memory pressure.
- Compare speculative decoding against an exact baseline before exposing it.
- Evaluate additional quantization formats individually; reject unimplemented
  formats rather than guessing their layout.

These investigations are not claims that the features already exist. See the
[Avalon comparison](AVALON-COMPARISON.md) for current implementation boundaries.

## Project maintenance

Keep Windows and Linux builds repeatable, release archives self-contained with
notices, docs explicit about native versus compatibility commands, and CI
usable for fork pull requests. Only the repository owner merges into `main`;
see [governance](REPOSITORY-GOVERNANCE.md).

## Nextora Insight — hardware performance advisor

The read-only `nextora insight` command combines Vulkan discovery, available RAM and CPU thread counts with verified native benchmark reports. The local console shows recent request rates, queue depth and Nextora-owned allocations. Use `nextora insight-compare` to compare two reports with matching recorded workload fields. See [Nextora Insight](INSIGHT.md) for commands and measurement limits.

Future work includes driver and model fingerprinting, transfer and memory-pressure profiling, controlled tuning trials, thermal and power measurements, storage analysis, opt-in benchmark sharing and price-aware upgrade comparisons. Benchmark uploads and hardware purchases require separate approval.

## Memory-aware model-fit planning (initial implementation)

The `nextora model-fit` CLI estimates GGUF tensor-layout sizes for hypothetical offline quantisation and plans nominal GPU headroom and host memory needs from the existing memory planner. It **does not produce a quantised checkpoint**, validate accuracy, measure VRAM residency, or add Inkling support. The acceptance gate for an encoder is a new-file write, roundtrip data integrity, quantization error checks and real model quality parity. Support for looped transformer architectures and optimised expert offloading remains follow-up implementation work; neither is claimed here.

### Experimental offline quantizer (follow-up)

`nextora quantize` now encodes eligible single-shard floating point GGUF matrices to Q4_0 or Q8_0, retaining other tensors and preserving source files. Every tensor has a normalized RMS round-trip reconstruction gate, and a new output is parsed before publication. This **does not** implement an importance-aware optimiser, optimum mixed precision, benchmarked 16GB VRAM fit or whole-model quality parity. See `docs/OFFLINE-QUANT.md` and issue #19.

## Model-level source/candidate quality gates

The `nextora quality` command now evaluates fixed raw prompts on both actual models, using a CPU reference and CPU or hardware Vulkan candidate, with identical reference-token teacher forcing. It measures softmax KL divergence, absolute logit difference and top-1 token agreement, with explicit pass/fail thresholds. This is not a coding-task benchmark, model truthfulness test, chat-template evaluation or RX 9070 XT speed result. See [MODEL-QUALITY.md](MODEL-QUALITY.md). Before accepting production-ready model conversion, we still require independent pretrained model results, task accuracy, long contexts and actual GPU memory/performance data.
