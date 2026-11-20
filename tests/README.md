# Native inference tests and fixtures

ForgeAI has one Rust inference runtime. Run its core tests and numerical
regressions with:

```bash
cargo test --locked -p forge-engine --lib --bins
cargo test --locked -p forge9070 --bin forge9070
cargo build --release --locked -p forge-engine -p forge9070
python scripts/validate-forge-engine.py --binary target/release/forge-engine --frontend target/release/forge9070
python scripts/validate-native-api.py --binary target/release/forge-engine
```

The native library has tiny, original test models and reference outputs.
`tests/fixtures` contains small untrained GGUF checkpoints for HTTP, CLI
and numerical regression. These are not language-quality benchmarks.

Legacy Strata cross-implementation tests were removed with the retired
runtime. Future native graph additions need independent reference vectors,
full model-file validation, and hardware tests separately.
