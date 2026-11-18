# Change log

Project and native engine versions are independent. The initial public source
snapshot is Forge9070 **0.9.1**, including `forge-engine` **0.2.1**.

## 0.9.1 — configuration and public-source preparation

- Added the native browser configuration panel, complete-shard model discovery,
  idle-only settings changes and optional profile persistence.
- Added Windows and WSL configuration launchers and startup recovery when a
  saved model or runtime is unavailable.
- Added original Forge9070 branding, a focused native quick start, contributor
  guides, issue forms, component licensing and published validation evidence.
- Added native Linux/Windows CI and draft release packaging, plus owner-only
  main-branch ruleset templates and a checked administration helper.

Publication preparation changes documentation, packaging and repository
configuration. It does not establish additional GPU performance or pretrained
model quality. See [validation](VALIDATION.md) for recorded checks.

## 0.9.0 — independent native engine

Introduced the independent Rust runtime, Vulkan state, supported Qwen GGUF
loading, exact selected-expert execution and native coding API. Earlier
compatibility development and upstream modifications remain documented in
[MODIFICATIONS.md](MODIFICATIONS.md) and the
[compatibility guide](docs/COMPATIBILITY.md).
