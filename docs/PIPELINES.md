# Continuous integration and releases

The workflows validate the code that is present in this repository. Their status
comes from GitHub Actions; adding a workflow does not establish that it has
already passed on a hosted runner or on an RX 9070 XT.

## Pull requests and main

[`ci.yml`](../.github/workflows/ci.yml) runs on pull requests, pushes to `main`,
and manual dispatch. It uses the Rust version in `rust-toolchain.toml`, the
committed Cargo lockfile, and official GitHub actions pinned to full commit SHAs.

| Job | What it checks |
| --- | --- |
| Source and shader checks | Python, JavaScript and shell syntax; UI helper and release-provenance/archive regressions; GLSL compilation and SPIR-V validation of both freshly compiled and committed shader bytecode |
| Native engine checks: Linux | Native unit tests, CPU numerical parity, both release entry points, frontend dispatch, HTTP/SSE/admission, persisted configuration, desktop/mobile Chromium interactions, and software-Vulkan numerical and concurrency checks |
| Native engine checks: Windows | Native unit tests, both release entry points, frontend dispatch, CPU CLI/API/configuration harnesses, and PowerShell launcher syntax |
| `required` | Succeeds only when source checks and **both** operating-system jobs succeed; failure, cancellation and skipped dependencies do not pass |

Configure the exact check-run context **`required`** as the required status check
on `main`. GitHub's interface may display it under the workflow as
**CI / required**. Keep the job name stable when changing the workflow.

PR jobs have `contents: read`, do not use user secrets, and do not persist checkout
credentials. There is no `pull_request_target`, deployment, merge bot or
auto-merge step. Dependabot opens reviewable weekly dependency-update PRs; it
cannot approve or merge them. Repository rules must enforce the owner's
merge authority separately from the CI status gate.

Reports are uploaded as `validation-ubuntu-24.04` and
`validation-windows-2025` artifacts for 14 days. A failed harness can stop before
writing its report; consult the corresponding job log in that case.

### Scope of a green check

These checks use small untrained fixtures. They do not establish full-model
loading, pretrained-model accuracy, hardware GPU execution, or token rates.
The Linux Vulkan gate explicitly selects Mesa's software driver and exercises
the real compute path; it is numerical evidence, not Radeon performance data.
The shader job checks source compilation and bytecode validity, not byte-for-byte
identity across compiler versions. Chromium exercises the actual served UI at
desktop and mobile sizes; screenshots and overflow checks support layout review.
The browser test dependency is pinned in `.github/scripts/requirements.txt`.

## Run the native checks locally

From the repository root on Linux:

```bash
export RUSTFLAGS='-C target-cpu=x86-64'
cargo test --locked -p forge-engine --lib --bins
cargo test --locked -p forge9070 --test new_engine_parity --test new_quant_parity --test new_qwen38_parity
cargo build --release --locked -p forge-engine -p forge9070
python3 scripts/validate-forge-engine.py --binary target/release/forge-engine --frontend target/release/forge9070 --output artifacts/local-cli.json
python3 scripts/validate-native-api.py --binary target/release/forge-engine --output artifacts/local-api.json
python3 scripts/validate-native-ui.py --binary target/release/forge-engine --output artifacts/local-ui.json
```

In Windows PowerShell:

```powershell
$env:RUSTFLAGS = '-C target-cpu=x86-64'
$env:PYTHONUTF8 = '1'
cargo test --locked -p forge-engine --lib --bins
cargo build --release --locked -p forge-engine -p forge9070
python scripts/validate-forge-engine.py --binary target/release/forge-engine.exe --frontend target/release/forge9070.exe --output artifacts/local-cli.json
python scripts/validate-native-api.py --binary target/release/forge-engine.exe --output artifacts/local-api.json
python scripts/validate-native-ui.py --binary target/release/forge-engine.exe --output artifacts/local-ui.json
```

Use Python 3.12 or later for these pipeline helpers. Rust and a working native
C/C++ build toolchain are required. The hosted runners provide the C/C++ tools;
Windows development installations need the Visual Studio C++ build tools.

## Native release packages

[`release.yml`](../.github/workflows/release.yml) runs when a `v*.*.*` tag is pushed,
or through **Actions → Release → Run workflow** with an existing tag. The tag
must match the root `Cargo.toml` version and belong to the history of
`origin/main`. Before any build, a read-only gate loaded from trusted `main`
verifies that its exact commit is the result of a PR merged by `@jamesoneill`
and has successful push-triggered main CI. Both the initiating actor and the
actor rerunning a release must be `@jamesoneill`. Manual dispatch runs from
`main`. Builds and draft preparation then use the verified commit SHA.
The engine's independent version is recorded separately; for example project
`v0.9.1` contains engine `0.2.1`.

For the first publication, the owner merges the source PR, waits for main CI,
then tags that merge result. An unmerged publication branch cannot produce a
release. No workflow creates a tag or merges a PR on the owner's behalf.

Each platform tests and builds the native engine, then runs the CPU CLI/API/UI
harnesses against the **exact executable being packaged**. Both platforms must
succeed before the draft-release job runs.

| Asset | Build target and scope |
| --- | --- |
| `forge9070-vVERSION-x86_64-pc-windows-msvc.zip` | Windows x64; x86-64 CPU baseline and static C runtime |
| `forge9070-vVERSION-x86_64-unknown-linux-gnu.tar.gz` | Linux x64; x86-64 CPU baseline, built on Ubuntu 24.04 for glibc systems |

Archives include the standalone engine, embedded UI/shaders, a usage guide,
engine MIT license, GGML attribution, transitive dependency license texts,
Rust standard-library notices and a build manifest. Model weights and the
RustingStrata compatibility runtime are not bundled. The full repository keeps
its separate licensing scopes.

The workflow checks both archive manifests against the requested tag/commit,
verifies the executable and archive SHA-256 hashes, and publishes checksum
sidecars plus `SHA256SUMS`. It creates or refreshes a **draft release** for the
owner to review. It refuses to modify an already published release. Only that
final job receives `contents: write`; no workflow merges a PR or pushes to main.

Distributable builds override `.cargo/config.toml` with
`RUSTFLAGS='-C target-cpu=x86-64'`. Windows release builds additionally enable
`-C target-feature=+crt-static`. These avoid inheriting the hosted runner's CPU
instruction set. Linux archives are not musl/static or universal-distro builds;
Vulkan drivers are a separate prerequisite on either platform.
The workflow does not code-sign Windows executables; systems enforcing an
enterprise signing policy may require their administrator's approved process.

For a build tuned to your own CPU, remove the CI `RUSTFLAGS` override and build
locally using the repository's `target-cpu=native` configuration. Do not
redistribute that locally tuned binary to machines with older CPU features.

## Dependency maintenance

[`dependabot.yml`](../.github/dependabot.yml) checks Cargo, GitHub Actions and
the browser-test dependency weekly. Action references stay pinned to immutable SHAs, and updates remain
ordinary PRs subject to the owner's review and the required CI gate. No
auto-approval or auto-merge is configured.
