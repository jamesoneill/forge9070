<p align="center">
  <img src="docs/assets/nextora-logo.svg" alt="Nextora logo" width="96" height="96">
</p>

<h1 align="center">Nextora</h1>

<p align="center">
  <strong>Local models. Your hardware. A Rust engine you can inspect.</strong>
</p>

<p align="center">
  <a href="https://github.com/jamesoneill/forge9070/actions/workflows/ci.yml"><img src="https://github.com/jamesoneill/forge9070/actions/workflows/ci.yml/badge.svg?branch=main" alt="Build and test status"></a>
  <a href="crates/nextora-engine/LICENSE-MIT"><img src="https://img.shields.io/badge/native_engine-MIT-23967f?style=flat-square" alt="Native engine licence: MIT"></a>
  <a href="VALIDATION.md"><img src="https://img.shields.io/badge/status-experimental-f46a3c?style=flat-square" alt="Experimental; read validation status"></a>
  <a href="docs/NATIVE-ENGINE.md"><img src="https://img.shields.io/badge/backend-Vulkan-59677b?style=flat-square" alt="Vulkan backend"></a>
</p>

<p align="center">
  <a href="#watch-the-architecture-walkthrough">Video walkthrough</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#choose-a-model">Models</a> ·
  <a href="docs/NATIVE-ENGINE.md">Engine guide</a> ·
  <a href="docs/README.md">Documentation</a> ·
  <a href="CONTRIBUTING.md">Contribute</a>
</p>

---

**Nextora** is a local-first inference project built in Rust. Its independent Vulkan engine targets compatible GPU devices and includes a CPU execution path. The old Strata runtime is retired; legacy executable names forward to the native engine.
Its independent Rust engine selects eligible hardware Vulkan adapters by capabilities (not vendor or model name), and supports explicit automatic CPU fallback when no compatible adapter is available. See [hardware support](docs/HARDWARE-SUPPORT.md) for the validated scope.
Its independent Rust engine combines Vulkan compute, bounded mixture-of-experts
caches, a browser configuration panel and an OpenAI-compatible coding API.
Windows and Linux launchers are included, with an explicit Windows-host route
for WSL users.

> [!IMPORTANT]
> **Experimental, with evidence attached.** CPU and software-Vulkan tests are
> recorded in [VALIDATION.md](VALIDATION.md). Full pretrained-model quality,
> Windows GPU execution and RX 9070 XT throughput still require validation on
> the target hardware. No token-rate claim is implied by a passing CI badge.

## Nextora launch page

The public-facing [Nextora landing page](site/index.html) links to the repository, documentation, releases and validation status. Until GitHub Pages is deployed, the page can be opened locally.

**Compatibility:** The repository remains at `jamesoneill/forge9070` so existing clone links continue working. The native crate remains `nextora-engine`; `nextora`, `nextora`, and `nextora-engine` are executable aliases. The historical `nextora-cli` executable now invokes that **same** native Rust engine. Legacy Strata commands and provider profiles require migration. See [the rename guide](docs/RENAME.md).

## Watch the architecture walkthrough

[**Nextora Explained: Inside a Rust & Vulkan Local AI Engine for AMD** (archived video title)](https://www.youtube.com/watch?v=NLr7lbs6Iwg)
is a 14:25 technical walkthrough from [**the1darkj Tech Briefing**](https://www.youtube.com/@the1darkj).
It explains expert caching across disk, RAM and VRAM, memory budgets, the native
coding API and Windows/WSL launch paths, with upstream credits and source references.

The episode covers the project as reviewed on **18 November 2026**. Read the
[current architecture](docs/ARCHITECTURE.md) and [validation record](VALIDATION.md)
alongside it: full pretrained-model quality and target RX 9070 XT performance
remain to be measured. Passing CI does not establish those results.

## What makes Nextora different

| Capability | What is implemented |
| :--- | :--- |
| **Independent Rust engine** | Model loading, sampling, CPU operations and native Vulkan shaders in `crates/nextora-engine`; no Strata execution dependency for the standalone binary. |
| **Capability-based hardware selection** | One Vulkan path for eligible devices, explicit CPU/Vulkan/hybrid choices and `--backend auto` with a reported selection reason. |
| **Bounded MoE storage** | Owned expert RAM, a GPU expert cache, lazy PLE embedding rows and protected expert leases, informed by the Avalon source audit. |
| **Every routed expert retained** | The native path computes every expert selected by the model. It does not omit expert tails for a speed headline. |
| **A service you can configure** | Start without a model, scan complete GGUF shard sets, set context and memory limits, then save the configuration. |
| **Coding-client API** | Chat Completions, streaming, model templates, function-call output, prefix reuse and a bounded request queue. Clients execute tools. |
| **Explicit resource limits** | One active model, one to four independent native request slots, configurable queue capacity and context/output budgets. |

[Architecture](docs/ARCHITECTURE.md) · [Avalon comparison](docs/AVALON-COMPARISON.md) · [Validation evidence](VALIDATION.md)

## Quick start

### 1. Build the native engine

Install the [Rust toolchain](https://rustup.rs/) and your platform's C/C++ build
tools. The repository pins Rust **1.100.0**. Windows uses the MSVC toolchain and
[Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
GPU execution also needs a working Vulkan driver exposing the required device
features. Ordinary builds use the checked-in shader binaries.

```sh
git clone https://github.com/jamesoneill/forge9070.git
cd nextora-cli
cargo build --release --locked -p nextora-engine
# produces nextora, nextora and nextora-engine executable entry points
```

On Windows, clone onto a **local Windows drive**, for example
`C:\code\nextora-cli`, and build in Windows PowerShell. Executing Windows build
scripts from a WSL network share can be blocked by Application Control.

### 2. Open the configuration panel

**Windows PowerShell**

```powershell
.\target\release\nextora-engine.exe serve --open-ui --backend auto `
  --models-dir 'F:\Models' --settings-file '.\forge-service.json'
```

**Linux**

```sh
./target/release/nextora-engine serve --open-ui --backend auto \
  --models-dir /path/to/models --settings-file ./forge-service.json
```

The native `--backend auto` policy selects an eligible Vulkan adapter or CPU if discovery cannot supply one. An explicit device, nonzero VRAM cap, or software-Vulkan permission prevents silent CPU fallback. Model, shader, device creation and allocation failures still propagate. `nextora-engine doctor` reports eligibility, not validated throughput.

Open **<http://127.0.0.1:9070/>**. Choose your actual model folder, select
**Scan models**, choose the first shard of a complete supported model, and use
**Apply & load**. The panel can start before a model is available. The settings
file saves successful configuration changes; an existing profile takes
precedence over model/runtime command-line options.

<details>
<summary><strong>Using WSL, or seeing llvmpipe?</strong></summary>

A Windows process launched from WSL still uses the Windows GPU driver. Build
the Windows binary first, then run:

```sh
bash scripts/configure-native-wsl.sh \
  --windows-exe /mnt/c/code/nextora-cli/target/release/nextora-engine.exe \
  --models-dir /mnt/f/Models
```

Native Linux execution inside WSL needs an actual usable Vulkan hardware
adapter. `llvmpipe` is a software renderer; `/dev/dxg` alone does not establish
Vulkan support. The launchers do not change GPU drivers, DKMS or Windows
Application Control. See the [native guide](docs/NATIVE-ENGINE.md) for path,
network and platform details.

</details>

### 3. Connect a coding client

| Setting | Value |
| :--- | :--- |
| API style | OpenAI Chat Completions |
| Base URL | `http://127.0.0.1:9070/v1` |
| Model ID | Read the active ID from `GET /v1/models` |
| Streaming | Server-sent events through `/v1/chat/completions` |
| Configuration panel | `http://127.0.0.1:9070/` |

Start with **one active request** and measure your workload. More request slots
consume more state memory and share the same compute resources. Extra requests
wait in the bounded queue; a full queue returns HTTP 429. This is not continuous
batching or a four-times speed guarantee. See the
[native API and admission guide](docs/NATIVE-ENGINE.md) for supported parameters,
authentication, context limits and client behavior.

> [!TIP]
> **Use the native command.** `nextora serve`, `nextora serve`,
> `nextora-engine serve` and `nextora-cli serve` enter the same service on port
> **9070**, with `--open-ui`. The former `nextora-cli engine` prefix is retired.
> With `cargo run`, put `--` before `serve`.

## Choose a model

The native service validates **GGUF architecture, tensor formats, geometry,
tokenizer and complete shard sets**. Model names and publisher labels are not
compatibility tests. Original and community-modified models go through the same
checks. The repository contains only tiny untrained fixtures, not pretrained
weights.

| Previously selected checkpoint | Download | Files | Disk space |
| :--- | :--- | :---: | ---: |
| Standard Qwen3.8-Flash-Next, GSQ-RCO IQ3_S | [Pinned IQ3_S folder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S) | 2 | 83.6 GB |
| Community uncensored Qwen3.8-Flash-Next, UD-IQ4_XS | [Publisher and download instructions](https://huggingface.co/ToR-O3/Qwen3.8-Flash-Next-Uncensored-GGUF) | 4 | 93.7 GB |

Keep **all shards of one checkpoint together** and select the `00001` file.
Keep different variants in separate folders. These are publisher downloads,
not models trained by Nextora. The GSQ-RCO layout has real-header inspection
evidence; neither complete checkpoint has been accepted end to end on the
target RX 9070 XT. The community modification has its own evaluation limits.

Disk size does not specify the required working-set size. Bounded caches allow
weights to move through host memory and GPU memory, while cold reads and large
contexts can still make a configuration impractical. Start with the
[model guide](docs/QWEN3.8.md) and the native memory controls.

## One Rust application

All three documented executable names enter the **same native Rust inference
service**, sharing a single GGUF loader, cache manager, scheduler and API.
`nextora` is the public-facing name, `nextora-engine` is the native package's
original executable, and `nextora-cli` is a thin historical command alias.
There is no separate Strata serving backend or port 8080.

```sh
cargo build --release --locked -p nextora-engine -p nextora-cli
./target/release/nextora serve --open-ui --backend auto
```

The native `context` command preserves the local tracked-file source packer.
See [migration](docs/COMPATIBILITY.md) for unsupported former Strata commands,
and [native runtime](docs/NATIVE-ENGINE.md) for the supported API and model
architectures. Only the native configurations are supported.

## Engineering principles

**Measure on the real workload.** Report checkpoint revision, quantization,
driver, hardware, context, prompt/generation token counts, warmup and memory
use alongside latency and throughput. Keep numerical parity and model quality
checks separate from speed measurements.

**Keep budgets explicit.** Separate model-file size, owned expert RAM, mapped
embedding rows, GPU expert storage and per-request state. All allocations need
headroom; an estimate is not a reservation.

**Show what is still missing.** The native engine does not claim Avalon's HIP
kernels, custom IQ3_KS format, transfer/compute overlap, speculative decoding or
elastic reuse of unused KV memory. The [comparison](docs/AVALON-COMPARISON.md)
records the actual relationship to the reference project.

## Project map

| Path | Purpose |
| :--- | :--- |
| [`crates/nextora-engine/`](crates/nextora-engine) | Independent inference runtime, Vulkan kernels and embedded service UI |
| [`src/`](src) | Historical CLI alias and shared local repository context packer |
| [`vendor/`](vendor) | Historical Avalon policy notice; Strata runtime source removed |
| [`scripts/`](scripts) | Launchers, validators and release helpers |
| [`tests/`](tests) | Original fixtures and repeatable regression checks |
| [`docs/`](docs/README.md) | Architecture, platforms, model formats and operational guides |
| [`.github/`](.github) | CI, release automation, issue forms and ownership rules |

## Contributing and releases

Small, well-evidenced contributions are welcome. Begin with
[CONTRIBUTING.md](CONTRIBUTING.md), use an [issue template](https://github.com/jamesoneill/forge9070/issues/new/choose),
and include relevant checks with a pull request.

CI builds and tests the native engine on Linux and Windows, with a stable
aggregate check for branch protection. Tagged/manual release automation prepares
**draft** native binary releases with checksums and notices. A successful build
does not establish GPU or large-checkpoint performance. See
[the pipeline guide](docs/PIPELINES.md) for the exact gates and release process.

**Merge policy requires owner review.** The live GitHub ruleset currently also
permits an integration to bypass update restrictions through pull requests;
that permission must be reviewed separately before claiming strict owner-only
merges. Automatic merging is disabled. Ownership and
required checks are configured separately; see
[repository governance](docs/REPOSITORY-GOVERNANCE.md). CODEOWNERS routes review
requests but is not, by itself, a merge-permission control.

[Security policy](SECURITY.md) · [Code of conduct](CODE_OF_CONDUCT.md) · [Roadmap](docs/ROADMAP.md)

## Credits and licences

The single native runtime and historical command alias are licensed under
[MIT](LICENSE.md). Third-party components retain their own notices.

**Inspired by and credited to RustingStrata by Vasyl Trefilov.** The retired
compatibility engine's original licence is preserved as
[historical licence text](LICENSE-RUSTINGSTRATA-HISTORICAL.md) alongside
[upstream provenance](UPSTREAM.json); **no RustingStrata inference runtime is
included or linked**. The project does not claim authorship of RustingStrata.

[Avalon](https://github.com/freriadoc/Avalon) informed the MoE storage
architecture, and [GGML / llama.cpp](https://github.com/ggml-org/llama.cpp)
informed GGUF and quantization support; retain their applicable notices.
See [third-party notices](THIRD_PARTY_NOTICES.md) and
[source references](SOURCES.md). Check every redistributable component's
licence individually, especially after dependencies change.

Nextora is an independent project. It is not endorsed by AMD, Qwen, Avalon or
RustingStrata. The original Nextora artwork is documented in the
[brand guide](docs/BRAND.md).

