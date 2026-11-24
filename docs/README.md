# Nextora documentation

Start with the [Nextora launch page](../site/index.html), [rename/compatibility guide](RENAME.md), or [project overview](../README.md). For setup, go straight to the
[native engine guide](NATIVE-ENGINE.md).

## Run and configure

| Guide | Use it for |
| --- | --- |
| [Windows quick start](WINDOWS.md) | Build on Windows, open the configuration UI, repair common startup errors |
| [Native engine](NATIVE-ENGINE.md) | Independent Rust engine, configuration UI, model discovery, API, Windows and WSL |
| [Nextora Insight](INSIGHT.md) | Read-only hardware assessment, benchmark-backed advice, and measured upgrade decisions |
| [Offline GGUF quantizer](OFFLINE-QUANT.md) | Conservative single-file Q4_0/Q8_0 export with per-tensor error gates |
| [Model-fit planning](MODEL-FIT.md) | Read-only GGUF quantisation file-size estimates and GPU memory-budget advice |
| [Hardware support](HARDWARE-SUPPORT.md) | Capability-based Vulkan selection, automatic CPU fallback and GPU-validation limitations |
| [Loading quality and performance](MODEL-LOADING.md) | Startup stages, accuracy invariants, warmup trade-offs and reproducible benchmarks |
| [Model selection](QWEN3.8.md) | Exact Qwen3.8 GGUF revisions, shards, original and community variants |
| [Architecture](ARCHITECTURE.md) | Request lifecycle, model storage and component boundaries |
| [Compatibility migration](COMPATIBILITY.md) | Retired Strata commands and the unified native service on port 9070 |
| [WSL details](WSL.md) | Historical WSL notes; see current Windows and native engine guides for supported commands |

## Performance and correctness

| Guide | Scope |
| --- | --- |
| [Validation status](../VALIDATION.md) | Completed checks and remaining hardware/model acceptance |
| [Published evidence](validation/README.md) | Sanitized historical reports and their provenance |
| [Avalon comparison](AVALON-COMPARISON.md) | Implemented MoE mechanisms and remaining differences |
| [Expert RAM](EXPERT-RAM.md) | Bounded cache design and accounting |
| [Performance measurement](PERFORMANCE.md) | Compatibility-path tuning and reproducible measurements |
| [Accuracy](ACCURACY.md) | Quantization, sampling and output-comparison limits |

## Historical compatibility material

Some older guides describe the retired Strata application. They are kept for
migration reference only and do not establish live native CLI or API features.
Use the [native guide](NATIVE-ENGINE.md) for current functionality.

## Develop and maintain

- [Contributing](../CONTRIBUTING.md)
- [CI and releases](PIPELINES.md)
- [Repository governance](REPOSITORY-GOVERNANCE.md)
- [Roadmap](ROADMAP.md)
- [Security policy](../SECURITY.md)
- [Third-party notices](../THIRD_PARTY_NOTICES.md)
- [Brand and artwork](BRAND.md)
- [Change log](../CHANGELOG.md)
