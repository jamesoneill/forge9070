# ForgeAI naming and compatibility

ForgeAI is the public project name. **Forge9070** is the original project name and a retained compatibility identifier.

## Stable interfaces

| Existing interface | Policy |
| --- | --- |
| `https://github.com/jamesoneill/forge9070` | Remains the canonical repository address until an intentional GitHub repository rename with redirects is separately verified. Existing clone URLs keep working. |
| Cargo package `forge9070` | Retained to avoid breaking Cargo.lock, package selectors and dependent scripts. This is the legacy CLI and hardware profile. |
| Cargo package `forge-engine` | Retained as the independent native engine library and executable. |
| New executable `forgeai` | Alias for the same `crates/forge-engine/src/main.rs` entry point. |
| `forge9070 engine ...`, `forge9070 serve` | Retained; latter is the older compatibility service. |
| API base URL and ports | Unchanged. Native service remains on port 9070. |
| Files, model directories and settings | Do not rename automatically; existing paths/configuration continue to work. |
| Existing logo at `docs/assets/forge9070-logo.svg` | Retained for historic links; `forgeai-logo.svg` is the new public asset. |

This rebrand makes **no new hardware-support claims**. Native Vulkan requires a capable adapter and suitable driver. Some paths and documentation still describe RX 9070 XT validation because that is their historical context.

## Build and run

```sh
git clone https://github.com/jamesoneill/forge9070.git
cd forge9070
cargo build --release --locked -p forge-engine
./target/release/forgeai serve --help
```

The `forge-engine` binary remains valid. To build only the new alias: `cargo build --release --locked -p forge-engine --bin forgeai`.

**Migration guidance:** Existing scripts should keep running. New scripts may use `forgeai` for the native runtime. Do not rename the root Cargo package or release artefacts until downstream build scripts and CI/release contracts are deliberately migrated and tested.

## Launch page

The static [ForgeAI launch page](../site/index.html) links to source, docs, releases, and the status evidence. It is designed to work as an ordinary static file and can be deployed with GitHub Pages after the site's deployment workflow is reviewed.
