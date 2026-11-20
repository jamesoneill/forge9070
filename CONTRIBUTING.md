# Contributing to Forge9070

Forge9070 is an experimental Rust inference project. Contributions should improve
measured behavior while preserving model validation, bounded memory use and
explicit backend selection. Read the [native engine guide](docs/NATIVE-ENGINE.md)
and [validation record](VALIDATION.md) before changing execution behavior.

## Work on the intended engine

| Area | Location | Entry point |
|---|---|---|
| Native engine, scheduler, UI and runtime cache | `crates/forge-engine/` | `forgeai`, `forge-engine` or `forge9070` |
| Historical command alias | `src/main.rs` | Forwards to the same native Rust engine |
| Bounded repository context packer | `src/code_context.rs` | Compiled into the native crate; `forgeai context` |

The workspace pins Rust in `rust-toolchain.toml`. Install Rust with rustup and
the platform's C/C++ build tools. On Windows, build in a local Windows folder;
see the native guide for Windows/WSL setup. No pretrained download is needed
for the synthetic CPU checks below.

## Submit a change

1. Fork the repository and create a focused branch from `main`.
2. Explain the problem and resulting behavior. Keep unrelated formatting and
   vendored changes out of the patch.
3. Run the checks relevant to the change and record their actual results.
4. Open a pull request against `main`. Include documentation and attribution
   updates when behavior or provenance changes.

From the repository root, the native baseline is:

```bash
cargo build --release --locked -p forge-engine -p forge9070
cargo test --locked -p forge-engine --lib --bins
python3 scripts/validate-forge-engine.py \
  --binary target/release/forge-engine --frontend target/release/forge9070 \
  --output artifacts/forge-engine-cli-local.json
python3 scripts/validate-native-api.py \
  --binary target/release/forge-engine --output artifacts/native-api-local.json
python3 scripts/validate-native-ui.py \
  --binary target/release/forge-engine --output artifacts/native-ui-local.json
```

On Windows use `python` and append `.exe` to binary paths. For browser and GPU checks, follow [VALIDATION.md](VALIDATION.md) and
[tests/README.md](tests/README.md). Report skipped or unavailable checks explicitly;
a synthetic or software-Vulkan pass does not establish physical GPU performance.

## Correctness and performance evidence

Keep every expert selected by the router in the computation. Preserve numerical
checks when changing kernels, quantization, caching or scheduling. Performance
reports should identify the commit, OS, GPU/driver, model revision and quantization,
context/output sizes, concurrency, command and measurement method. Separate
prefill, decode, per-request latency and aggregate throughput. Compare output
quality or numerical behavior as appropriate; do not infer full-model quality
from a small fixture.

Do not commit credentials, personal prompts, downloaded checkpoints, local
service profiles or build outputs. Redact diagnostics before sharing them.

## Review, ownership and licenses

**Only @jamesoneill may merge pull requests into `main`.** Contributors submit
pull requests; maintainers do not enable automatic merging. The intended branch rules must enforce this policy, but the currently installed
GitHub integration bypass requires separate ruleset review: `CODEOWNERS` requests review but does
not itself restrict merges or grant permissions.

Native project code is MIT-licensed. Preserve every applicable dependency and
reference notice in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The historical RustingStrata licence text is retained for provenance, but the
old inference runtime is no longer included.

Follow the [code of conduct](CODE_OF_CONDUCT.md). Report vulnerabilities through
the [security policy](SECURITY.md), not a public bug report.
