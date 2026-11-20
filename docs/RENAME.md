# Nextora — naming and migration

**Nextora** is the public product name. **ForgeAI** and **Forge9070** are historical identities. The rename changes presentation and introduces a command alias without silently breaking shipped interfaces.

| Interface | Migration policy |
| --- | --- |
| GitHub `jamesoneill/forge9070` | Kept until an administrator explicitly renames the repository and validates automatic redirects; all current links remain valid |
| Cargo package `forge-engine` | Preserved; new public executable `nextora` is built from the same Rust entry point |
| Existing binaries `forgeai`, `forge-engine` | Preserved |
| Root Cargo package / binary `forge9070` | Preserved as legacy profile/compatibility entry point |
| `forge9070 engine` | Preserved as the native engine bridge |
| API paths, default ports, settings files | Unchanged; native service port remains 9070 |
| `docs/assets/forgeai-logo.svg`, `forge9070-logo.svg` | Preserved; new public art is `nextora-logo.svg` |
| Historical validation reports, prior release names, upstream credits | Never rewritten retroactively |

## Build

```sh
git clone https://github.com/jamesoneill/forge9070.git
cd forge9070
cargo build --release --locked -p forge-engine
./target/release/nextora serve --help
```

Windows: `.\target\release\nextora.exe serve --help`.

Existing callers can continue to invoke `forgeai` or `forge-engine`. The root `forge9070 serve` command is the older compatibility path, not the same backend.

**Do not rename `forge-engine` as a Cargo package until downstream dependency and release tooling is migrated and independently verified.** Branding the runtime as Nextora does not itself add support for all GPUs, models, or execution paths.

## Launch site

The static [Nextora site](../site/index.html) points to current documentation, source, releases and validation evidence. Before GitHub Pages deploys it, this remains a versioned page in source control rather than a claimed live production deployment.

**Name clearance:** Other businesses use "Nextora"; check relevant trademarks, project-name availability, and domains before formal commercial branding or acquiring a domain.
