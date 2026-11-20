# ForgeAI compatibility — legacy Forge9070 commands

The former project name remains in CLI, package and filesystem identifiers to avoid breaking existing integrations. See [the rename guide](RENAME.md).

This guide documents the retained `forge9070` compatibility commands, using
RustingStrata. For the independent engine and configuration UI, start with the
[project README](../README.md) and [native guide](NATIVE-ENGINE.md).

## Compatibility engine and existing application

**Native Rust GGUF inference with an RX 9070 XT profile, based on RustingStrata.**

Uses RustingStrata by Vasyl Trefilov.

This project adapts an existing native Rust inference engine for the target
**AMD Radeon RX 9070 XT, RDNA 4 / gfx1201, 16 GiB VRAM**. It provides an
Avalon-like workflow for supported Qwen mixture-of-experts models: keep useful
expert weights on the GPU, compute nonresident experts on the CPU, generate
text, and serve a local chat API. It is a tailored build, not a claim of
feature parity or equal speed with Avalon. Version 0.4 prioritises individual
reply generation speed while retaining all selected experts. It starts with
F16 KV and one request slot, measures decode speed with output-comparison gates,
and makes model compression visible. Compatible MoE model switching, optional
MTP/concurrent slots and WSL launchers remain included.

Version 0.5 adds the important Avalon memory mechanism: a **bounded, owned RAM
cache of routed experts**, beneath the GPU cache, sized from available memory.
CPU misses and GPU uploads share immutable expert leases. Native direct reads
are attempted on Linux and Windows, with an explicitly counted buffered
fallback. Startup warmup fills the bounded expert working set; sparse PLE
embeddings stay mapped and are read only as needed. Prompt misses do not evict
the generation working set. This does not port Avalon's HIP registration,
custom IQ3_K/IQ3_KS kernels, Qwen4 MTP or elastic VRAM allocator.

For a roughly 62 GB host, start by inspecting the standard **GSQ-RCO IQ3_S**
Qwen3.8 conversion: its 83.62 GB download contains **46.84 GiB of expert
weights**, plus a separate **26.82 GiB lazy embedding table**. The whole file
does not need to be resident. The previous UD-IQ4_XS choice has 56.24 GiB of
experts and will more often use a partial cache. See
[docs/QWEN3.8.md](QWEN3.8.md) for exact revisions, original and uncensored
choices, and [docs/EXPERT-RAM.md](EXPERT-RAM.md) for the new memory policy.

## Large code projects

Version 0.6 adds **`--preset code` (16K context / 4K output)** and
**`--preset code-long` (32K context / 8K output)**, each with a 1 GiB parked
conversation cache, F16 KV, one generation slot, MTP disabled and a 6 GiB
host-memory growth allowance. Explicit flags override preset values. These
are chosen starting points, not hardware-calibrated speed claims.

```powershell
& $Forge serve --models-dir 'C:\Models' --preset code-long
& $Forge context --repo 'C:\code\MyProject' --query 'Trace cancellation and refund retries' --model $Model --preset code-long --output '.\repo-context.txt'
```

`$Forge` is the built Windows executable and `$Model` is the actual local first
GGUF shard. The new `context` command selects bounded, line-labelled excerpts
from the current Git working tree using lexical query/path matching. A model
tokenizer gives exact pack counts; it does not run inference or execute project
code. It is not a complete semantic code index or autonomous editing agent.

The server and browser check the **rendered prompt plus requested output**
before inference. An explicit oversize request gets an actionable error rather
than silent truncation. `POST /api/context` previews that exact text-request
budget. Browser history preserves reasoning and code whitespace for more
faithful continuation. Optional persistent GPU prefix caches now validate model,
adapter and runtime identity and bound stored lengths before allocation.

See **[docs/LARGE-CODE-PROJECTS.md](LARGE-CODE-PROJECTS.md)** for commands,
memory estimates, context selection, prefix reuse, output reserves and measured
tuning. Full Qwen inference, Windows operation and RX 9070 XT throughput remain
hardware validation steps; source/CPU tests do not establish those results.

## Concurrent clients and coding APIs

The server now defaults to **`--parallel auto --max-parallel 4`**, selects
compatible capacity when a model loads, and admits requests as demand arrives.
**Qwen3.8-Flash-Next still runs one generation sequence**; additional requests
queue. Supported GPU architectures can use up to four shared-model slots.
Auto does not tune memory or speed, and idle slots do not release the context
state allocated at load.

The bounded queue defaults to **32 waiting requests**, a **120-second admission
deadline**, and **64 total connections**. `/v1/status` reports actual capacity
and queue state while generation is busy. Coding clients can use the existing
OpenAI Chat Completions API with JSON/SSE and client-executed function tools;
the Responses API is not implemented.

See **[docs/CLIENT-API.md](CLIENT-API.md)** for startup and client examples,
supported options and model-ID discovery, and
**[docs/REQUEST-SCHEDULING.md](REQUEST-SCHEDULING.md)** for capacity,
memory behavior, queue errors, deadlines and retry handling.

## The ROCm decision

The GPU backend in this build is **Vulkan**. ROCm is not linked, initialized,
required, upgraded, or removed. Your existing ROCm 6.3.4 installation can stay
where it is, but this is not a working ROCm 6.3.4 backend for the 9070 XT.
AMD's 6.3.4 WSL support matrix does not include this card. A Rust rewrite cannot
add that missing runtime support. If you specifically need HIP/ROCm in WSL,
AMD documents the card in the ROCm 7.2.x / ROCDXG 1.2.0 compatibility table;
that is a separate driver/runtime migration, outside this build.

**WSL can run the Linux application directly.** CPU mode needs no GPU runtime;
native WSL Vulkan mode requires a real adapter passing the capability checks.
RustingStrata documents RX 9070 XT Vulkan runs on Windows and native Linux,
together with known issues. WSL's `/dev/dxg` does not itself establish that
compute path. An additional launcher lets the WSL shell run the Windows GPU
executable directly when native WSL Vulkan is unavailable.

The old `amdgpu-dkms` package failure is separate: this project does not repair
APT/dpkg state. Its scripts never install kernel modules or change drivers.

## What is included

- Complete pinned RustingStrata source and original license.
- Rust frontend: `doctor`, `inspect`, `memory`, `profile`, `check`, `generate`, `chat`,
  `serve`, `bench`, `tune`, and `models search/files/pull/list/load/unload/status`.
- Search and choose metadata-checked Hugging Face MoE GGUFs in the browser, including compatible
  community, uncensored and abliterated variants. Downloads are isolated by
  repository and immutable revision, resumable and verified before publication.
- Real GGUF loading, tokenizer, transformer inference, adaptive expert placement,
  and CPU/GPU execution supplied by the engine.
- OpenAI-compatible `/v1/chat/completions`, streamed responses, `/v1/models`,
  health endpoint, and the engine's browser chat page.
- Read-only GPU/environment report with JSON output. CPU/software adapters
  cannot silently satisfy the RX 9070 XT selection.
- Windows/Linux build scripts and a tiny deterministic synthetic model for
  repeatable CPU and API tests.

There are no pretrained model weights, Windows executable, ROCm packages, or
GPU performance claims in this source archive. See `VALIDATION.md` for the
checks actually completed and their limits.

## Start in WSL

Place the project in the WSL Linux filesystem, for example
`~/code/forge9070`, then run:

```bash
bash scripts/setup-wsl.sh --check
bash scripts/setup-wsl.sh --build
```

If Rust is missing, install the Linux toolchain from <https://rustup.rs/>.
The check reports missing C/build dependencies; `--install-deps` is an explicit
optional step for ordinary Ubuntu build tools and the Vulkan loader. It does
not install DKMS or change ROCm. If the old failed AMD package is still pending,
it reports that state before attempting an APT operation.

Start the complete application with explicit CPU execution:

```bash
bash scripts/serve-wsl.sh --backend cpu
```

Or attempt native WSL GPU execution, which stops if GPU discovery fails:

```bash
bash scripts/serve-wsl.sh --backend vulkan
```

Open **http://localhost:8080**. No initial model is required: use the model
manager to download and load one. The server remains available while unloaded;
`/health` reports HTTP 503 until a model is ready, while management endpoints
continue to work.

For GPU execution on Windows controlled from the WSL terminal, build the Windows
executable and use:

```bash
bash scripts/serve-wsl-windows.sh \
  --exe '/mnt/c/code/forge9070/target/release/forge9070.exe' \
  --models-dir '/mnt/d/Models'
```

That route executes inference on Windows. Native WSL GPU execution has not been
verified on the target machine. See **[docs/WSL.md](WSL.md)** for paths,
networking, token forwarding and exact distinctions between the three routes.

## Browse Hugging Face and switch models

In the chat page, open the model controls and use **Hugging Face models**:

1. Search for a repository, such as `Qwen3-30B-A3B-abliterated`, or paste `owner/repo`.
2. Select a repository and revision, then inspect its GGUF choices and sizes.
3. Download the desired quantization. The listed revision is resolved to a
   commit and the download stays pinned to that snapshot.
4. Use **load downloaded model**, or select an existing local model and load it.
5. Use **unload** to release its resources without stopping the application.

There is **one active model at a time**. Switching releases the previous model
before allocating the next. Metadata/format/tokenizer rejection preserves the
currently loaded model. A later allocation or full-loading failure can leave
the server unloaded; select another model to recover without restarting.

An OpenAI chat request naming a different model gets HTTP 409 with the active
model ID. Load that model explicitly first. The `model` field does not silently
trigger a download or an automatic swap. Query `/v1/models` for the active ID;
omitting the field uses the current model. Anthropic aliases retain the
underlying engine's behaviour.

### CLI equivalents

Use `./target/release/forge9070` on Linux/WSL or
`.\target\release\forge9070.exe` on Windows:

```bash
./target/release/forge9070 models search 'Qwen3-30B-A3B-abliterated'
./target/release/forge9070 models files mradermacher/Qwen3-30B-A3B-abliterated-GGUF

# Choose a verified file deliberately. This Q4_K_M example is ~18.56 GB and
# uses hybrid GPU/host-memory placement on a 16 GiB card.
./target/release/forge9070 models pull mradermacher/Qwen3-30B-A3B-abliterated-GGUF \
  --file Qwen3-30B-A3B-abliterated.Q4_K_M.gguf

./target/release/forge9070 models list
./target/release/forge9070 models status
./target/release/forge9070 models load 'EXACT_ID_FROM_MODELS_LIST'
./target/release/forge9070 models unload
```

`pull --load` downloads, checks the format, then asks the running server to load
that model. The CLI and server must use the same model directory. `--revision`
accepts a branch, tag or commit, and `--json` provides machine-readable output.
For another server use `--server http://HOST:PORT` on load/unload/status or
pull-with-load. Give the root URL, without `/v1`. Model-loading calls have a
600-second default timeout, configurable with `--timeout`.

### Storage and authentication

The model directory is selected in this order:

1. Explicit `--models-dir` on serve, list or pull.
2. `FORGE9070_MODELS_DIR`.
3. Linux/WSL: `${XDG_DATA_HOME:-$HOME/.local/share}/forge9070/models`.
   Windows: `%LOCALAPPDATA%\forge9070\models`.

Hugging Face downloads live under a per-repository/per-commit subdirectory and
retain their original file paths. Identical filenames from two repositories
cannot overwrite each other. All shards are downloaded together. Interrupted
downloads keep `.part` files for resumption; only complete GGUF files become
catalogue entries. Size and available LFS SHA-256 metadata are checked, and a
sidecar JSON records provenance. No repository Python or custom loader code
executes.

Public repositories normally need no token. For a gated/private repository,
obtain the required account access and set `HF_TOKEN` **before starting the
server or pull command**. The application reads it from the environment,
does not save it to model metadata and does not forward it to other redirect
hosts. `STRATA_API_KEY` is separate: it authenticates local management/chat
requests. Keep the same `STRATA_API_KEY` in the server and CLI environments.

Hugging Face requests respect the protocol-specific `HTTPS_PROXY`/`https_proxy`
setting before the general `ALL_PROXY`, preserve `NO_PROXY`, and verify HTTPS
using the platform certificate trust configuration. On Unix, explicitly
configured `SSL_CERT_FILE`/`SSL_CERT_DIR` roots are supported. Invalid proxy
configuration returns an error; it does not disable certificate verification.

### Compatible MoE models only

The catalogue now requires a supported **sparse mixture-of-experts** architecture,
valid routed-expert geometry, required tensors and implemented quantization
types. It inspects actual GGUF metadata, including every shard, before offering
a download. Search names, publisher labels and file extensions alone are not
accepted as proof. Unknown, incomplete, dense or unsupported choices are hidden;
direct download and load requests enforce the same policy.

The current primary paths include Qwen3MoE, Qwen3-Next, Qwen3.5MoE, qwen4exp and
Flint when their actual configuration and tensors satisfy the checks. Individual
features have additional restrictions: hyper-connections/PLE/QSA need one slot,
and this profile's ordinary MTP path rejects those configurations. Metadata acceptance
still requires subsequent allocation and actual inference to validate a machine.

Community models labelled uncensored or abliterated are eligible under exactly
the same checks. The dense Qwen3-8B example from version 0.2 is now excluded.
Avalon's custom IQ3_K/IQ3_KS files remain unsupported and are rejected rather
than interpreted as a different quantization. Ordinary compatible Q3_K/Q4_K
MoE GGUFs are distinct formats.

Header probes are bounded and pinned to immutable commits. A server refusing
byte ranges, a missing access grant, or a timeout makes a file unverified and
unselectable; the file listing reports the reason. It does not mean the model
has been proved incompatible. Successful pinned metadata checks are cached.
Model authors' licence and access conditions still apply.

## Generation speed and fidelity

The default objective is fast generation for one reply. Every expert selected
by the model's router contributes; the Forge frontend rejects the legacy
expert-tail omission option when it is nonzero. F16 KV avoids adding int8 KV
quantization to the selected checkpoint. Quantized weights and floating-point
kernels still have numerical limits. This profile does not make a checkpoint
factually correct.

The engine retains its adaptive expert cache, CPU/GPU overlap, prompt prefetch,
fused GPU work and prompt-state reuse. The source comparison in
[docs/AVALON-COMPARISON.md](AVALON-COMPARISON.md) identifies the separate
Avalon mechanisms that are still absent.

Start with:

```bash
./target/release/forge9070 serve --model /path/model.gguf \
  --parallel 1 --mtp 0 --kv f16 --ctx 4096
```

The model picker requires an explicit GGUF choice and displays actual tensor
formats, including low-bit expert weights. A size estimate is not a quality
recommendation. Q4_K_M and Q5 variants are useful candidates for comparison;
formats, architecture, checkpoint quality and the tasks being answered all
matter. The selected model may need host RAM as well as VRAM. See
[docs/ACCURACY.md](ACCURACY.md) for model and sampling guidance.

### Tune generation on the actual GPU

```bash
./target/release/forge9070 tune --model /path/model.gguf \
  --ctx 4096 --kv f16 --tg 128 --reps 3 --warmup 1 \
  --json-out tuning.json
```

Calibration considers small prompt chunks and supported embedded MTP budgets
jointly, because prompt scratch allocation also affects expert residency during
generation. It excludes warmup, keeps each token count paired with its own
elapsed decode time and compares median generation rates. A faster candidate
must match the no-MTP baseline's greedy token IDs on every tested prompt/run;
it must improve median measured generation speed by at least 3% to displace the baseline.
The report retains raw measurements and reasons for rejected candidates.

By default the tuner uses three chat-formatted workloads. Use `--prompt-file`
with representative text to measure a workload closer to your own, and
`--threads` to hold the CPU worker count fixed. Thread counts need separate
process runs; the tuner does not sweep hardware, model precision or KV formats.
Only matching version-4 calibration records can be applied. Older records are
ignored.

Stop a separately running GPU server before invoking the CLI tuner so the two
processes do not compete for memory or compute. Use the printed flags explicitly,
or use the browser's calibrate action. The
comparison checks selected-model execution on finite test prompts; it does not
score factual answers or prove universal numerical equivalence. Greedy decoding
is used to make that comparison reproducible, not because it always gives
better model answers.

### Multiple requests remain available

```bash
# Several clients share ONE active checkpoint, with separate request state.
./target/release/forge9070 serve --model /path/model.gguf \
  --parallel 2 --mtp 0 --kv f16
```

`--parallel auto` selects compatible capacity up to `--max-parallel` (default 4);
explicit `--parallel 1`–`4` selects an exact count where the architecture permits it. CPU mode
accepts concurrent connections but computes generation serially. MTP and
multiple GPU slots cannot be combined. Hyper-connections, PLE and QSA require
one slot and are excluded from this profile's ordinary MTP path.

More slots use more context memory and leave less VRAM for expert weights.
Use them for aggregate multi-client throughput; they do not imply faster
individual replies. The concurrency benchmark defaults to one client and can
be run explicitly with more:

```bash
python3 scripts/bench-concurrency.py --clients 2 --requests 8 \
  --max-tokens 128 --json-out concurrency-c2.json
```

The report counts actual completion tokens and includes queuing/prefill in its
aggregate wall-time metric. It separately reports latency and checks resolved
execution settings before and after the run, including KV, chunk, worker count
and model identity. Use the server's administrative API key when configured.

Q8 KV remains an explicit option through `--kv q8`. It can free memory for
experts, but may change outputs; evaluate it against the F16 baseline on your
workload. The tuner never silently changes KV format or checkpoint precision.
See [docs/PERFORMANCE.md](PERFORMANCE.md) for the measurement procedure.

## Build on Windows

Use **native Windows PowerShell**, preferably Developer PowerShell for Visual
Studio. Extract this archive to a normal Windows directory, for example
`C:\code\forge9070`.

Prerequisites:

1. A working AMD Windows driver exposing Vulkan 1.3 for the RX 9070 XT.
2. [Rust with the MSVC toolchain](https://rustup.rs/). The project pins the tested
   Rust 1.100.0 toolchain in `rust-toolchain.toml`; rustup can install it when
   Cargo first runs.
3. [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   with the Desktop development with C++ workload and Windows SDK. Some Rust
   dependencies compile C code and need a linker.

Run from the extracted project directory:

```powershell
cargo build --release --locked
.\target\release\forge9070.exe doctor
.\target\release\forge9070.exe profile
```

The first build downloads the pinned Cargo dependencies. A Vulkan SDK or shader
compiler is not required for the ordinary build: upstream includes the SPIR-V
shader binaries. The optional `scripts\build-windows.ps1` performs the same
build and prints the diagnostic report.

The build targets the local CPU's available instructions for efficient hybrid
expert computation. Build on the PC that will run the program. Rebuild before
moving the executable to an older CPU.

### Verify the card before loading a large model

```powershell
.\target\release\forge9070.exe doctor --require-gpu
.\target\release\forge9070.exe doctor --json > doctor.json
```

Look for the real `AMD Radeon RX 9070 XT`, the selected `backend_index`, Vulkan
1.3, and present required features. The backend index is the engine's filtered
and sorted Vulkan index, not the Windows Task Manager GPU number. If several
9070 XT cards are present, select one with `--device N`.

The doctor only queries physical-device capabilities. A positive report is
not evidence that a kernel or a model has executed correctly. It does not
allocate large buffers or run a benchmark. The engine's `check` command adds a
pipeline/shared-memory preflight; actual generation is the compute test.

## Choose a supported model

This frontend accepts supported sparse MoE configurations in the following
architecture paths. A tag alone is insufficient: every model must pass the
expert-routing, tensor-shape and quantization checks described above.

| Architecture tag | Family |
| --- | --- |
| `qwen3`, `qwen3moe` | Qwen3 with supported sparse MoE tensors; dense models are excluded |
| `qwen3next` | Qwen3-Next |
| `qwen35`, `qwen35moe` | Supported Qwen3.5 MoE configurations |
| `qwen4exp` | Qwen3.8-Flash-Next |
| `flint` | Supported Flint configuration |

Other upstream experimental model paths are outside this frontend.
Supported tensor formats include F32, F16, BF16, Q4_0/Q4_1, Q5_0/Q5_1, Q8_0,
Q2_K–Q6_K, and the IQ formats implemented in the vendored engine. `inspect`
prints the actual types and rejects unsupported ones before loading weights.

**Avalon's custom IQ3_K/IQ3_KS and packed layouts are not interchangeable with
these models.** A filename containing `iq3_k` is conservatively rejected too;
renaming a file does not convert its format. Use a standard supported GGUF
export. For a sharded model, keep every shard together and pass the
`-00001-of-0000N.gguf` file.

The full model does not have to fit into 16 GiB of VRAM or physical RAM. Hot
experts are placed on the GPU; the owned host tier holds as much of the expert
working set as its budget permits. Cold experts come from the model files.
Sparse PLE rows remain mapped independently of the expert tier. The owned
payload stays within its configured ceiling, including entries leased by
active CPU work or uploads. Storage is pageable, and runtime/state allocations
still need headroom. Preview the actual categories and this machine's current
budget with `forge9070 memory --model FIRST-SHARD.gguf --json`.

## Load, generate, and serve

Replace the example path with your compatible GGUF model:

```powershell
$Model = 'C:\models\your-supported-model.gguf'
.\target\release\forge9070.exe inspect --model $Model
.\target\release\forge9070.exe check --model $Model
.\target\release\forge9070.exe chat --model $Model
```

For a raw completion rather than chat-template formatting:

```powershell
.\target\release\forge9070.exe generate --model $Model --prompt 'The capital of France is' --max-tokens 64
```

Start the local server with this initial model, or omit `--model` to start empty:

```powershell
.\target\release\forge9070.exe serve --model $Model --port 8080
```

Open **http://127.0.0.1:8080** for the included chat page, or configure a client
with API base **http://127.0.0.1:8080/v1**. Query `/v1/models` for the model ID.
The server defaults to model-compatible auto capacity, a separate queue of
32 waiting generation requests with a 120-second admission deadline, and a
total connection cap of 64. Qwen3.8 currently resolves to one active sequence.
The `general` preset permits at most 1024 output tokens per
request; `code` permits 4096 and `code-long` 8192. `--max-tokens` overrides that
startup cap. Explicit requests above the cap, or whose rendered prompt plus
requested completion exceeds context, are rejected before inference. The
`/api/context` route previews the complete text-request budget without generation.

PowerShell API example in a second terminal:

```powershell
$Models = Invoke-RestMethod 'http://127.0.0.1:8080/v1/models'
$Body = @{
    model = $Models.data[0].id
    messages = @(@{ role = 'user'; content = 'Write a short hello.' })
    max_tokens = 128
    temperature = 0.7
    top_p = 0.8
    top_k = 20
    stream = $false
    chat_template_kwargs = @{ enable_thinking = $false }
} | ConvertTo-Json -Depth 8
Invoke-RestMethod 'http://127.0.0.1:8080/v1/chat/completions' -Method Post -ContentType 'application/json' -Body $Body
```

For Qwen3, the example uses supported non-thinking sampling settings and
disables a separate thinking phase so visible text is returned in
`message.content`. With thinking enabled, some supported models also use
`reasoning_content`, and a short completion budget can end during that phase.

For local authentication, set `$env:STRATA_API_KEY` before starting the process
and send `Authorization: Bearer YOUR_KEY`. The frontend requires that environment
variable if you deliberately bind outside loopback with `--host`. Localhost is
the default. The upstream browser's administrative model-loading controls may
change model and run options; the Forge profile describes startup defaults,
not immutable resource limits or all later `/api/load` choices.

## WSL clients of the Windows process

When using the Windows GPU launcher, WSL applications can act as clients.
With WSL mirrored networking, Microsoft
documents Linux-to-Windows access through `127.0.0.1`:

```bash
curl http://127.0.0.1:8080/v1/models
```

In WSL's default NAT mode, `127.0.0.1` does not generally reach a service running
on Windows. Either use mirrored networking on a supported Windows/WSL setup,
or use the Windows host IP with an explicit server bind and appropriate firewall
configuration. This project does not silently alter networking or open ports.
Use Microsoft's linked networking documentation for your environment.

## Initial RX 9070 XT settings

These are conservative chosen defaults, **not measurements from your card**.

| Setting | Default | Purpose |
| --- | --- | --- |
| Backend | Vulkan | Avoid dependency on the unsupported ROCm 6.3.4/WSL combination |
| Context | 4096 tokens | Modest state/KV allocation |
| Prefill chunk | 256 tokens | Shorter GPU submissions |
| VRAM margin | 2048 MiB | Leave room for desktop/driver use |
| RAM margin | 4096 MiB below available RAM | Leave room for the OS and other applications |
| Expert RAM | `auto` | Bound owned expert weights using available RAM and the actual tensor payload |
| Expert warmup | `true` | Fill the bounded expert working set during loading; `false` fills on demand |
| Expert runtime reserve | 1024 MiB | Leave additional room for staging/work buffers and cache bookkeeping |
| Prompt-state cache | 512 MiB | Limit this separate host cache |
| KV format | F16 | Avoid additional int8 quantization of attention state |
| MTP drafts | 0 initially | Establish basic inference before speculation |
| Concurrent requests | 1 initially | Prioritise individual generation and expert residency |

The GPU expert cache uses the engine's remaining GPU budget. `--expert-ram`
controls the separate owned host tier (`auto`, `0`, or a ceiling in MiB).
`--cache-ram` reserves prompt-state caching separately. Expert budgeting uses
Linux `MemAvailable` and cgroup-v2 ancestor headroom, or Windows available
physical memory, and also respects an explicit `--max-ram` cap. These are
startup ceilings, not operating-system reservations. Driver-reported free VRAM
is also an estimate; discovery does not guarantee a large allocation.

Override settings explicitly, for example:

```powershell
.\target\release\forge9070.exe serve --model $Model --ctx 8192 --chunk 128 --vram-margin 3072
.\target\release\forge9070.exe bench --model $Model --pp 128,512 --tg 128 --reps 3 --warmup 1 --json-out bench.json
```

The benchmark records actual results. It does not insert estimated throughput.
Each prompt length plus generation count must fit the context. Use a real
`--prompt-file` to exercise realistic expert routing.

If the GPU resets during prefill, try `--chunk 64`, then `--chunk 0` for
token-at-a-time prefill. If you run out of VRAM, lower context, increase the VRAM
margin, or use a smaller model. If the engine reports non-finite logits, capture
the error; `STRATA_NAN_TRACE=1` enables its per-layer tracing. See upstream
`vendor/rusting-strata/docs/AMD.md` for the documented AMD coherent-memory and
driver issues. Neither the doctor nor smaller chunks can guarantee correctness
for every model/driver combination.

## Native Linux and explicit CPU mode

With a working native Linux AMD Vulkan driver and Rust/C build toolchain:

```bash
cargo build --release --locked
./target/release/forge9070 doctor --require-gpu
./target/release/forge9070 serve --model /path/to/model.gguf
```

The same binary can run a CPU reference path with `--cpu`. This is explicit and
may be slow for large models. Failure to find the GPU never selects it silently:

```bash
./target/release/forge9070 generate --cpu --model tests/fixtures/tiny-qwen3moe.gguf --prompt hello --max-tokens 4 --ctx 128 --ram-margin 0
```

The synthetic fixture emits `AAAA`. It is a numerical/integration test, not a
language model for useful conversation or a speed benchmark.

## Reproduce validation

```bash
cargo test --locked --workspace --lib --bins --test inference --test model_switch --test serve --test ops
python3 tests/smoke.py --binary target/debug/forge9070
python3 tests/models_cli.py --binary target/debug/forge9070
python3 tests/concurrency_cli.py --binary target/debug/forge9070
python3 tests/bench_cli.py --binary target/debug/forge9070
python3 scripts/test-wsl.py
```

On Windows use `python` and `target\debug\forge9070.exe`. Build the debug binary
first with `cargo build --locked` if you only built release. The HTTP test
starts its own loopback server, executes real CPU inference, checks streaming
and authentication, and terminates its own process. The tests do not download
model weights or change drivers. The engine and new downloader tests can be run with
`cargo test --locked -p rusting_strata --lib`. The two-model switching
regression is `cargo test --locked -p rusting_strata --test model_switch`.

## Source and license

Read `MODIFICATIONS.md`, `UPSTREAM.json`, `LICENSE.md`, `SOURCES.md` and
`VALIDATION.md`. RustingStrata License 1.0 allows modification and distribution
subject to its conditions; it is not MIT and restricts selling the engine
itself as a substitute product. Required acknowledgement:

**Uses RustingStrata by Vasyl Trefilov.**

