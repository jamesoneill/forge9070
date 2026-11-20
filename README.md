<p align="center">
  <img src="docs/assets/nextora-logo.svg" alt="Nextora logo" width="96" height="96">
</p>

<h1 align="center">Nextora</h1>

<p align="center">
  <strong>Local models. Your hardware. A Rust engine you can inspect.</strong>
</p>

<p align="center">
  <a href="https://github.com/jamesoneill/forge9070/actions/workflows/ci.yml"><img src="https://github.com/jamesoneill/forge9070/actions/workflows/ci.yml/badge.svg?branch=main" alt="Build and test status"></a>
  <a href="crates/forge-engine/LICENSE-MIT"><img src="https://img.shields.io/badge/native_engine-MIT-23967f?style=flat-square" alt="Native engine licence: MIT"></a>
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

**Nextora** is a local-first inference project built in Rust. Its independent Vulkan engine targets compatible GPU devices and includes a CPU execution path. The legacy RX 9070 XT-specific profile remains for compatibility.
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

**Compatibility:** The repository remains at `jamesoneill/forge9070` so existing clone links continue working. The native crate stays `forge-engine`, while the executable aliases `nextora`, `forgeai` and `forge-engine` are supported. Existing `forge9070` and `forge-engine` commands remain available. See [the rename guide](docs/RENAME.md).

## Watch the architecture walkthrough

[**Forge9070 Explained: Inside a Rust & Vulkan Local AI Engine for AMD** (archived video title)](https://www.youtube.com/watch?v=NLr7lbs6Iwg)
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
| **Independent Rust engine** | Model loading, sampling, CPU operations and native Vulkan shaders in `crates/forge-engine`; no Strata execution dependency for the standalone binary. |
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
cd forge9070
cargo build --release --locked -p forge-engine
# produces nextora, forgeai and forge-engine executable entry points
```

On Windows, clone onto a **local Windows drive**, for example
`C:\code\forge9070`, and build in Windows PowerShell. Executing Windows build
scripts from a WSL network share can be blocked by Application Control.

### 2. Open the configuration panel

**Windows PowerShell**

```powershell
.\target\release\forge-engine.exe serve --open-ui --backend vulkan `
  --models-dir 'F:\Models' --settings-file '.\forge-service.json'
```

**Linux**

```sh
./target/release/forge-engine serve --open-ui --backend vulkan \
  --models-dir /path/to/models --settings-file ./forge-service.json
```

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
  --windows-exe /mnt/c/code/forge9070/target/release/forge-engine.exe \
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
> **Use the correct command.** `forge-engine serve` and `forge9070 engine serve`
> are the native service on port **9070**, with `--open-ui`.
> `forge9070 serve` is the retained compatibility service on port **8080**.
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

## Two clearly separated paths

| | Native engine — primary path | Compatibility application |
| :--- | :--- | :--- |
| Package | `forge-engine` **0.2.1** | `forge9070` **0.9.1** |
| Source | [`crates/forge-engine`](crates/forge-engine) | [`src`](src) and [`vendor/rusting-strata`](vendor/rusting-strata) |
| Start | `forge-engine serve --open-ui` | `forge9070 serve` |
| Default port | **9070** | **8080** |
| License | MIT, with retained third-party notices | RustingStrata License 1.0 |
| Guide | [Native guide](docs/NATIVE-ENGINE.md) | [Compatibility guide](docs/COMPATIBILITY.md) |

The compatibility path retains the earlier Hugging Face browser/downloader,
repository-context tooling and Strata features. Its architecture restrictions,
API options and concurrency behavior differ from the independent engine.

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
| [`crates/forge-engine/`](crates/forge-engine) | Independent inference runtime, Vulkan kernels and embedded service UI |
| [`src/`](src) | RX 9070 XT profile and retained compatibility commands |
| [`vendor/`](vendor) | Pinned third-party source and attribution |
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

**Authorised maintainers and the configured GitHub integration may merge pull requests into
`main`.** Automatic merging is disabled by the repository setup. Ownership and
required checks are configured separately; see
[repository governance](docs/REPOSITORY-GOVERNANCE.md). CODEOWNERS routes review
requests but is not, by itself, a merge-permission control.

[Security policy](SECURITY.md) · [Code of conduct](CODE_OF_CONDUCT.md) · [Roadmap](docs/ROADMAP.md)

## Credits and licences

The independent native engine is [MIT-licensed](crates/forge-engine/LICENSE-MIT).
The root compatibility application and vendored engine retain
[RustingStrata License 1.0](LICENSE.md), including its commercial-resale
restriction. **The repository as a whole is not MIT-licensed.**

**Uses RustingStrata by Vasyl Trefilov.**

[Avalon](https://github.com/freriadoc/Avalon) informed the MoE storage work;
[GGML / llama.cpp](https://github.com/ggml-org/llama.cpp) material retains its
notices. Qwen and the model publishers own their respective model releases.
See [third-party notices](THIRD_PARTY_NOTICES.md),
[modification history](MODIFICATIONS.md), [sources](SOURCES.md) and
[pinned upstream metadata](UPSTREAM.json).

Nextora is an independent project. It is not endorsed by AMD, Qwen, Avalon or
RustingStrata. The original Nextora artwork is documented in the
[brand guide](docs/BRAND.md).

