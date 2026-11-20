# Native inference tests and fixtures

Nextora has one Rust inference runtime. Run its core tests and numerical
regressions with:

```bash
cargo test --locked -p nextora-engine --lib --bins
cargo test --locked -p nextora-cli --bin nextora-cli
cargo build --release --locked -p nextora-engine -p nextora-cli
python scripts/validate-nextora-engine.py --binary target/release/nextora-engine --frontend target/release/nextora-cli
python scripts/validate-native-api.py --binary target/release/nextora-engine
```

The native library has tiny, original test models and reference outputs.
`tests/fixtures` contains small untrained GGUF checkpoints for HTTP, CLI
and numerical regression. These are not language-quality benchmarks.

Legacy Strata cross-implementation tests were removed with the retired
runtime. Future native graph additions need independent reference vectors,
full model-file validation, and hardware tests separately.
