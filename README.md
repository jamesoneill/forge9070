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

**Nextora** is an experimental, local-first AI inference engine written in
Rust. It supports validated GGUF checkpoints with CPU execution and eligible
Vulkan devices, a browser configuration panel, bounded mixture-of-experts caches
and a local coding-client API. Hardware selection is based on Vulkan
capabilities rather than the GPU brand or model name.

The older Strata provider runtime has been retired. Both the public
`nextora` executable and optional `nextora-cli` wrapper use the same native
inference implementation. For the exact validated hardware and model scope,
see [hardware support](docs/HARDWARE-SUPPORT.md) and
[validation evidence](VALIDATION.md).

> [!IMPORTANT]
> **Experimental, with evidence attached.** CPU and software-Vulkan tests are
> recorded in [VALIDATION.md](VALIDATION.md). For Windows RX 9070 XT results,
> including the checkpoint, context occupancy and packed-versus-Direct logits
> comparison, see [AMD measurements](docs/AMD-PERFORMANCE.md).

## Nextora launch page

The public-facing [Nextora landing page](site/index.html) links to the repository, documentation, releases and validation status. Until GitHub Pages is deployed, the page can be opened locally.

**Compatibility:** The repository remains at `jamesoneill/forge9070` so existing clone links continue working. The public executable is `nextora` and its Rust package is `nextora-engine`. The auxiliary `nextora-cli` command forwards to the same native runtime. Legacy Strata commands and provider profiles require migration. See [the rename guide](docs/RENAME.md).

## Watch the architecture walkthrough

[**Early architecture walkthrough** (archived, recorded before the rebrand)](https://www.youtube.com/watch?v=NLr7lbs6Iwg)
is a 14:25 technical walkthrough from [**the1darkj Tech Briefing**](https://www.youtube.com/@the1darkj).
It explains expert caching across disk, RAM and VRAM, memory budgets, the native
coding API and Windows/WSL launch paths, with upstream credits and source references.

The episode covers the project as reviewed on **18 November 2026**. Read the
[current architecture](docs/ARCHITECTURE.md) and [validation record](VALIDATION.md)
for changes since the recording.

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

### Windows: recommended first run

The simplest way to start the local configuration UI on Windows is to run
the Windows launcher from a **Windows PowerShell** terminal. You do not need
WSL, Python, ROCm, a model download, or a separate frontend server just to
open the UI.

Install [Rustup](https://rustup.rs/) and [Visual Studio C++ Build
Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the
**Desktop development with C++** workload, including MSVC and the Windows SDK.
Nextora pins its Rust toolchain in `rust-toolchain.toml`.

```powershell
New-Item -ItemType Directory -Force C:\code | Out-Null
Set-Location C:\code
git clone https://github.com/jamesoneill/forge9070.git nextora
Set-Location .\nextora
.\scripts\configure-native-windows.ps1 -Build
```

This builds the `nextora-engine` Rust package, which produces
**`target\release\nextora.exe`** (not `nextora-engine.exe`). It starts the
foreground server and attempts to open **http://127.0.0.1:9070/**. Keep
PowerShell open while using the configuration UI. An initial model is optional.

To start it again after building:

```powershell
.\scripts\configure-native-windows.ps1
```

Or launch it directly:

```powershell
.\target\release\nextora.exe serve --backend auto --open-ui
```

The launcher saves the current configuration under
`%LOCALAPPDATA%\Nextora\native-service.json`. Existing saved model/runtime
settings take precedence over CLI values. To diagnose a stale model or GPU
setting without deleting your profile, use:

```powershell
.\scripts\configure-native-windows.ps1 -Backend cpu -Port 9071 `
  -SettingsFile "$env:TEMP\nextora-troubleshoot.json"
```

Then open **http://127.0.0.1:9071/**. The model folder can be selected and
scanned in the UI; specify `-ModelsDir 'F:\Models'` only if that directory
exists on your computer.

If `nextora.exe` is missing, Cargo fails, the browser cannot connect, or GPU
discovery doesn't work, follow the
**[Windows setup and troubleshooting guide](docs/WINDOWS.md)**. It includes
the exact Rust/MSVC requirements, validation commands, saved-profile recovery,
port conflicts and Windows Application Control limitations.

### Linux and WSL

For native Linux, build the Rust package and run the Linux binary:

```sh
cargo build --release --locked -p nextora-engine
./target/release/nextora serve --backend auto --open-ui
```

If you're working in WSL but want to use the **Windows GPU driver**, build
and launch the Windows `nextora.exe` on a local Windows drive. WSL's
`llvmpipe` Vulkan renderer is not a substitute for GPU hardware.
See [Windows from WSL](docs/WINDOWS.md#working-with-wsl) and the
[native guide](docs/NATIVE-ENGINE.md) for details.

### Configure a model

Open the running UI, select the actual folder containing your **complete
GGUF checkpoint**, click **Scan models**, choose the first shard of a supported
model and then **Apply & load**. Both the CPU backend and supported hardware
Vulkan adapters are available; `auto` selects compatible Vulkan hardware
during discovery, or CPU if no suitable adapter is found. Explicit GPU settings
are strict: model, shader and device failures are not silently retried on CPU.
Native quality and performance remain subject to checkpoint and hardware
validation.

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
> **Use the native command.** `nextora serve` or
> `nextora-cli serve` enter the same service on port
> **9070**, with `--open-ui`. The old nested `engine` prefix is no longer supported.
> With `cargo run`, put `--` before `serve`.

You can also search public **Hugging Face models** from the console. Nextora uses
`curl` to inspect pinned GGUF headers, including all shards, and lists a bounded
set of results that pass its architecture, tensor, tokenizer and chat-template
checks. Choose **Download** to save a model to `F:\Models` on Windows or
`/mnt/f/Models` under WSL. Nextora checks disk space and verifies the published
SHA-256 hashes and local model layout. Choose **Prepare model** to fit context
and benchmark the downloaded model in a separate native process. Its profile is
saved and the previous service is restored. **Use prepared model** then loads
the selected model and performs normal cache warmup. Manual configuration is
also available. See [native model setup](docs/NATIVE-MODEL-SETUP.md).

In **Chat**, attach source files and use **Check context** to count the full
conversation and reserve space for a reply. **Use model maximum** sets the
context field to the limit declared by the selected checkpoint. Apply the
settings to use the new window. Each reply reports prompt and generation speed.
**Compare generation speed** repeats a coding prompt and excludes the first
warmup reply so you can compare settings on your own machine.


Choose **Run tuning** under **Performance tuning** to measure a loaded model in a
separate process. Serving pauses while the worker compares execution backends
and CPU worker counts at your selected context and memory limits. The fastest
validated configuration is loaded and saved. **Cancel tuning** stops the worker
and restores the previous configuration. Results are saved beside the service
profile in `.nextora-tuning` and reused for the same checkpoint, machine, context
and memory limits. **Run tuning again** starts a fresh comparison when wanted.
The benchmark uses a short coding prompt, with warmup excluded and two measured
runs per candidate. Preparation can shorten slow candidates and reports the
actual token counts; legacy fixed-context tuning uses 128-token runs. With Fast GPU math enabled, the study compares GPU
worker counts within that numerical mode. The panel records prompt and
generation speed for each case.

**Fast GPU math** is an optional setting for Qwen3 dense and MoE models on the
Vulkan backend. It is off by default. Enable it in the configuration panel and
choose **Apply & load**, or pass `--fast-math true` when launching with the CLI.
Saved service settings take precedence over CLI values. The selected GPU must
support Vulkan `shaderIntegerDotProduct`; an unsupported device reports an
error when this setting is requested.

This mode dynamically quantizes supported matrix inputs to Q8 and uses packed
integer dot products with Q4_K and Q6_K weights. The GGUF weights retain their
original format; context-cache precision is a separate setting. Other tensor formats use the F32
activation path. Quantizing activations introduces numerical differences, so
answers may change; this mode does not promise numerical equivalence or higher
speed. Use **Compare generation speed** with the same model, context and memory
limits to measure its effect on your machine.

### Large context windows

Choose **Find 512K models** to search for compatible GGUF files whose headers
declare at least 524,288 tokens. Model selection, download and loading stay under
your control. **Use 512K** becomes available after selecting a suitable model;
it selects compact context storage, one active request and a larger bounded GPU
allocation cap for review before **Apply & load**. The engine checks the full
context memory requirement before accepting compact settings. A short-context
checkpoint cannot gain a longer validated range by changing the context field.

**Context memory → Compact · Q4** stores every key/value token using signed
4-bit values and a half-precision scale per 64 values. Attention calculations
remain F32, and no history is dropped. This differs from **Fast GPU math**, which
quantizes matrix inputs. Either setting can change numerical results. Explicit
GGUF YaRN metadata is supported for ordinary Qwen3; its embedded scaling factor
and original context are preserved even when a smaller window is selected.
Unsupported position-scaling variants still fail validation. Compact or extended
contexts use CPU-generated position tables shared across all GPU layers to keep
long-position trigonometry consistent with the reference path. Compact context
storage currently permits one active request.

For a 48-layer Qwen3 MoE with four KV heads of width 128, compact storage at
524,288 tokens uses **12.75 GiB for K/V alone**. Weights, bounded attention scratch,
allocation alignment and a temporary old layer during cache growth need more
space. Expert GPU caching shrinks as context grows, with all selected experts
still evaluated. The UI reports the full-context memory estimate. Full-window
speed and retrieval quality require separate validation with the chosen model;
short-prompt token/s measurements do not establish full-window performance.

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

Both executables use the **same native Rust inference service**, GGUF loader,
cache manager, scheduler and API. `nextora` is the public executable,
`nextora-engine` is the Cargo package, and `nextora-cli` is an optional
thin wrapper.
There is no separate Strata serving backend or port 8080.

```sh
cargo build --release --locked -p nextora-engine -p nextora-cli
./target/release/nextora serve --open-ui --backend auto
```

The native `context` command preserves the local tracked-file source packer.
See [migration](docs/COMPATIBILITY.md) for unsupported former Strata commands,
and [native runtime](docs/NATIVE-ENGINE.md) for the supported API and model
architectures. Only the native configurations are supported.

## Plan model fit before changing hardware

`nextora model-fit --model /models/checkpoint.gguf --target-vram-mib 16384 --target-file-mib 12288 --ctx 4096 --requests 1` inspects a complete GGUF shard set and returns **hypothetical** size estimates for eligible floating-point matrix weights, a headroom-aware nominal GPU budget and conservative host-memory evidence. It does not create new weights, improve model quality, measure actual VRAM residency or enable previously unsupported architectures. Read [model-fit planning](docs/MODEL-FIT.md) and run real model benchmarks before hardware purchases.

## Loading quality and performance

Nextora checks GGUF and tokenizer metadata before probing or allocating Vulkan resources,
and verifies vocabulary agreement before warming expert caches. Load reports expose
per-stage timing so cold-start mapping, tokenizer work and optional warmup can
be measured separately. See the [model-loading and quality guide](docs/MODEL-LOADING.md)
for reproducible CPU/Vulkan comparisons, baseline acceptance and tuning trade-offs.

For Qwen3 MoE on Vulkan, ordinary model tensors remain resident on the GPU while
routed expert weights use a bounded GPU cache. Every selected expert contributes
to the result; expert batches that cannot fit can execute on the CPU. This path
allocates a small initial F32 KV cache and grows it as the conversation fills the
requested context, making room in the expert cache when needed. Selecting a
large context therefore does not reserve its full KV capacity at load time, but
long conversations still need the corresponding memory.

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
RustingStrata. The Nextora artwork is documented in the
[brand guide](docs/BRAND.md).

