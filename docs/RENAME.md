# ForgeAI naming and backwards compatibility

The public project name is ForgeAI. The GitHub repository remains
`jamesoneill/forge9070` to preserve existing links.

- **Canonical executable:** `forgeai`, built from the `forge-engine` Rust crate.
- **Native executable alias:** `forge-engine`, same Rust entry point.
- **Historical executable alias:** `forge9070`, a thin wrapper invoking the
  same native Rust CLI; it no longer selects a separate Strata runtime.
- **Unified command:** `serve` (port 9070), `generate`, `bench`, `doctor`,
  `inspect`, `context`, and `capabilities` are provided by the native CLI.
- Models and native service profiles are not deleted or automatically rewritten.
  Incompatible legacy Strata profiles require deliberate migration.

No source or runtime from RustingStrata is linked in the native applications.
Credit for its role in the project history remains in documentation.
See [compatibility migration](COMPATIBILITY.md).
