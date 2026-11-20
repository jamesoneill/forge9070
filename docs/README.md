# Nextora documentation

Start with the [Nextora launch page](../site/index.html), [rename/compatibility guide](RENAME.md), or [project overview](../README.md). For setup, go straight to the
[native engine guide](NATIVE-ENGINE.md).

## Run and configure

| Guide | Use it for |
| --- | --- |
| [Native engine](NATIVE-ENGINE.md) | Independent Rust engine, configuration UI, model discovery, API, Windows and WSL |
| [Model selection](QWEN3.8.md) | Exact Qwen3.8 GGUF revisions, shards, original and community variants |
| [Architecture](ARCHITECTURE.md) | Request lifecycle, model storage and component boundaries |
| [Compatibility application](COMPATIBILITY.md) | Earlier `forge9070 serve` and Strata commands, port 8080 |
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
