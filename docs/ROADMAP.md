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

The first milestone is a read-only `nextora insight` command. It combines native Vulkan discovery, available RAM and logical CPU information with existing verified `nextora bench` reports. It suggests benchmark follow-ups and **conditional** capacity checks without pretending model files are resident VRAM or claiming a speedup from a new GPU.

Future milestones (not implemented yet): benchmark fingerprinting (driver, model digest, operating-system build), live memory/transfer and queue profiling, controlled A/B optimisation trials, thermal/power/storage measurements, a local UI, opt-in anonymous benchmark sharing and price-aware upgrade comparisons. No benchmark upload or hardware purchase is authorised by this roadmap.
