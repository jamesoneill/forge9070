# Nextora compatibility and migration

**One inference engine:** `nextora`, `forgeai`, `forge-engine`, and the historical
`forge9070` command names now enter the same native Rust engine. They are
alternate executable names, not separate inference implementations.

The RustingStrata compatibility server has been retired from the application.
The old `forge9070 engine` prefix is retired: use native commands directly,
for example `forge9070 serve`. Strata model-provider download, browser and
profile-management commands were removed; use a local supported GGUF and
the native configuration panel.
The primary server command is `nextora serve --open-ui --backend auto`
(or `forge-engine serve` with the same flags) on port 9070.

Existing settings files and local GGUF model files are not deleted.
A saved legacy compatibility profile might need migration to the native
settings schema. The legacy port 8080 is no longer an inference endpoint.

The secure, tracked-only `context` command now runs in the native Rust CLI,
and accepts an optional model path for exact token counts.

Prior attribution and third-party notices remain in the repository. Runtime
and release binaries do not link against the RustingStrata inference runtime.
