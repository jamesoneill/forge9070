# ForgeAI documentation

Start with the [ForgeAI launch page](../site/index.html), [rename/compatibility guide](RENAME.md), or [project overview](../README.md) or go straight to the
[native engine guide](NATIVE-ENGINE.md).

## Run and configure

| Guide | Use it for |
| --- | --- |
| [Native engine](NATIVE-ENGINE.md) | Independent Rust engine, configuration UI, model discovery, API, Windows and WSL |
| [Hardware support](HARDWARE-SUPPORT.md) | Capability-based Vulkan selection, automatic CPU fallback and GPU-validation limitations |
| [Model selection](QWEN3.8.md) | Exact Qwen3.8 GGUF revisions, shards, original and community variants |
| [Architecture](ARCHITECTURE.md) | Request lifecycle, model storage and component boundaries |
| [Compatibility migration](COMPATIBILITY.md) | Retired Strata commands and the unified native service on port 9070 |
| [WSL details](WSL.md) | Environment diagnostics and compatibility-path launchers |

## Performance and correctness

| Guide | Scope |
| --- | --- |
| [Validation status](../VALIDATION.md) | Completed checks and remaining hardware/model acceptance |
| [Published evidence](validation/README.md) | Sanitized historical reports and their provenance |
| [Avalon comparison](AVALON-COMPARISON.md) | Implemented MoE mechanisms and remaining differences |
| [Expert RAM](EXPERT-RAM.md) | Bounded cache design and accounting |
| [Performance measurement](PERFORMANCE.md) | Compatibility-path tuning and reproducible measurements |
| [Accuracy](ACCURACY.md) | Quantization, sampling and output-comparison limits |

## Compatibility-path features

These guides describe the retained application. Consult the native guide for
independent-engine API behavior; options are not interchangeable.

- [Large code projects](LARGE-CODE-PROJECTS.md)
- [Coding-client API](CLIENT-API.md)
- [Request scheduling](REQUEST-SCHEDULING.md)

## Develop and maintain

- [Contributing](../CONTRIBUTING.md)
- [CI and releases](PIPELINES.md)
- [Repository governance](REPOSITORY-GOVERNANCE.md)
- [Roadmap](ROADMAP.md)
- [Security policy](../SECURITY.md)
- [Third-party notices](../THIRD_PARTY_NOTICES.md)
- [Brand and artwork](BRAND.md)
- [Change log](../CHANGELOG.md)
