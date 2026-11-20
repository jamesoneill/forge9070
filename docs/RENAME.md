# Nextora — naming and migration

**Nextora** is the public product name. **ForgeAI** and **Forge9070** are historical
identities. All current executable names invoke one native Rust inference engine.
The earlier Strata compatibility runtime has been retired.

| Interface | Migration policy |
| --- | --- |
| GitHub `jamesoneill/forge9070` | Kept until an administrator renames the repository and verifies redirects; existing links remain valid |
| Cargo package `forge-engine` | Preserved; the `nextora`, `forgeai`, and `forge-engine` executables use the same Rust entry point |
| Root Cargo package / binary `forge9070` | Preserved as a thin alias invoking the same native Rust CLI |
| `forge9070 engine ...` | Retired prefix; use `forge9070 ...` or `nextora ...` directly |
| Native API paths, default port, settings files | Native port remains 9070; existing native service profiles are preserved |
| Former Strata service and port 8080 | Retired; use the native service on port 9070 |
| Former Strata download, browser and model-provider settings | Retired; select a local supported GGUF through the native configuration panel and migrate legacy profiles deliberately |
| `docs/assets/forgeai-logo.svg`, `forge9070-logo.svg` | Preserved; current public artwork is `nextora-logo.svg` |
| Historical validation reports, prior release names, upstream credits | Retained as originally published |

Model files and native settings are not deleted or automatically rewritten.
The alias preserves the executable name, not every historical Strata command
or configuration schema. Native subcommands include `serve`, `generate`,
`bench`, `doctor`, `inspect`, `context`, and `capabilities`.
See [compatibility migration](COMPATIBILITY.md) and each command's `--help`.

## Build and run

```sh
git clone https://github.com/jamesoneill/forge9070.git
cd forge9070
cargo build --release --locked -p forge-engine
./target/release/nextora serve --open-ui --backend auto
```

Windows: `.\target\release\nextora.exe serve --open-ui --backend auto`.
To build the historical root executable as well, add `-p forge9070` to the build.
Existing callers can use `forgeai`, `forge-engine`, or `forge9070` with the same
native CLI arguments. Released archive names and the packaged `forge-engine`
executable remain unchanged until the release tooling is deliberately migrated.

The Rust package name remains `forge-engine` to preserve Cargo selectors and
release tooling. Branding the runtime as Nextora does not itself establish
support for every GPU, model architecture, or execution path.

## Launch site

The static [Nextora site](../site/index.html) points to current documentation,
source, releases and validation evidence. Until deployed, it is a versioned page
in source control, not a published production deployment.

The native applications do not link the retired RustingStrata runtime.
Historical attribution, licence text and original GGML/Avalon credits remain.
Other businesses use "Nextora"; this repository does not claim trademark clearance.
