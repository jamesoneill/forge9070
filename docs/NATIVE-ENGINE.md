# Nextora — native engine 0.2.1

The standalone `nextora-engine` package runs our Rust model implementation and
Vulkan shaders without a RustingStrata execution dependency. It supports checked
`qwen3`, `qwen3moe` and Qwen3.8 `qwen4exp` text paths. The native HTTP catalogue
admits validated MoE checkpoints only. The ordinary dense Qwen3 path remains
available to the CLI and numerical tests.

Use the `nextora` executable for the native engine. The auxiliary `nextora-cli` wrapper invokes the same native implementation. The earlier Strata compatibility server was retired. Its old model provider settings and port 8080 do not carry over automatically. The native runtime keeps the existing MIT and GGML attribution notices.

## Implemented scope

| Component | Native implementation |
|---|---|
| Storage | Complete standard GGUF shard sets, including metadata-only first shards; checked tensor ranges and geometry |
| Quantization | F32, F16, BF16, Q4_0, Q8_0, Q2_0, Q4_K, Q5_K, Q6_K, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS |
| Qwen3.8 | Hyper-connections, gated attention, DeltaNet recurrent state, QSA pooling/selection, lazy PLE histories, routed and shared experts |
| CPU | Direct quantized rows, existing AVX2/FMA/F16C selection where implemented, reusable per-sequence working memory |
| Vulkan | Capability-selected adapters, packed-weight kernels, device-owned model state, reusable command resources, independent sequences |
| Native API | Chat Completions JSON/SSE, embedded model template, text/function-call histories, model catalogue and idle switching |
| Configuration UI | Embedded browser panel, startup without a model, bounded folder scan, idle-only settings/load/unload and optional saved JSON profile |
| Admission | One to four active requests, FIFO bounded wait queue, aggregate queued-prompt token budget, disconnect cancellation and deadlines |
| Reuse | Exact executed-token prefix retains sequence state; optional request-level cache salt isolates agent namespaces; unrelated histories reset state |
| Numerical state | F32 KV and recurrent state; checkpoint quantization is retained |

Generation and benchmarking require an explicit policy: `--backend auto`,
`--backend cpu`, `--backend vulkan`, or `--backend hybrid`. `--cpu` is an alias
for CPU. `auto` selects eligible Vulkan hardware or CPU when discovery finds
none. Explicit device, VRAM budget or software-Vulkan settings disable silent
CPU fallback, and load/creation/shader errors remain errors. The status reports
both the policy and the actual execution backend. See [hardware support](HARDWARE-SUPPORT.md). The hybrid backend is a
diagnostic bridge with CPU state and cached GPU projections; its per-projection
synchronization makes it unsuitable as the default performance route. The
Vulkan Qwen3.8 path keeps state operations on the GPU and uses a bounded expert
cache. A device-budget miss that cannot accommodate an expert's packed working
set computes that expert on the CPU and increments an explicit counter. It
never drops the expert or falls back to an external engine.

Unsupported geometry, unknown tensor layout and missing required metadata are
errors. Model names, marketing labels and “uncensored” are not compatibility
checks. The server accepts original and modified model weights under the same
architecture/format validation. A full downloaded model must include all its
shards and its tokenizer/template metadata.

## Serving controls informed by vLLM

The native service retains **independent sequence slots**, not vLLM's continuous GPU batching or paged KV cache. The following safe serving controls allow measured tuning without changing model equations:

- `--prefill-chunk-tokens N`: 1–4096, default **32**. Larger chunks reduce the frequency of request-disconnect checks; they may change throughput and time to first token, so benchmark rather than assuming larger is faster.
- `--max-queued-prompt-tokens N`: aggregate prompt-token budget for **waiting** requests (not active slots). Default **0** derives `context × parallel × 4`; queued requests exceeding the budget receive HTTP 429. This complements the existing request-count limit and adjusts on idle configuration changes.
- `cache_salt` in `/v1/chat/completions`: optional, nonempty string of at most 256 bytes. Reuse of an idle, executed prompt prefix requires an exact match between cache namespaces. Requests without a salt can reuse only other unsalted prefixes. This is an isolation aid, not authentication.

Example startup with a larger prefill chunk and an explicit waiting budget:

```powershell
.\target\release\nextora.exe serve --backend auto --open-ui --prefill-chunk-tokens 128 --max-queued-prompt-tokens 65536
```

Include `"cache_salt": "my-agent"` on each request from an agent that should reuse its own history but not that of other agents. The cache salt is not included in the model prompt or written to the saved profile. Keep tenant authentication and access control separate.

Inspect `GET /v1/status` for queue prompt-token accounting and the configured prefill chunk size. Neither control alone improves token computation throughput; physical-GPU benchmarking is required.

## Routing-guided, bounded host expert prefetch

The Qwen3.8 native Vulkan path now observes **actual decode routing IDs** by
layer and uses a conservative, frequency-based prediction to warm **host RAM**
expert triplets for future decode requests. Only repeated hot selections qualify
(at least 16 observations for the layer and at least 25% selection frequency).
One candidate is scheduled every 32 decoded tokens using a per-layer
round-robin policy. Work is handled by one background worker with a queue of
capacity one; if the queue is busy the hint is dropped, not accumulated.
Set `NEXTORA_MOE_PREFETCH=0` in the Nextora process environment to
disable background host prefetch and compare latency with the baseline. The
default is enabled. Restart the service after changing this environment variable.

Key invariants:

- **Prediction never changes routing or skips selected experts.** A mistaken
  prediction only consumes spare host-cache space and disk I/O.
- Existing non-evicting atomic `warm_batch` admission loads gate/up/down
  together, or declines without evicting live or hot entries.
- The host cache remains bounded by the same configured memory capacity.
  This does not prefill GPU VRAM or borrow KV/recurrent state memory.
- Prefill routing is not used for training the decode predictor or scheduling
  hints, so large prompts do not thrash the decode cache.
- No network traffic, model weights modification, or external inference engine.
- Metrics appear under
  `runtime.execution.memory.predictive_host_prefetch`: scheduled,
  completed, loaded bytes, busy skips, budget skips, and errors.
  A completed hint may load zero bytes if the triplet is already resident.

**Important:** The worker uses the same safe expert cache and its admission
lock, which can still compete with active inference. This first implementation
is deliberately conservative, not a proof of reduced latency or true
asynchronous GPU expert transfers. Benchmark cold and warm caches and watch
host memory pressure on the actual GPU before increasing hint frequency.
Disabling background prefetch under contention and GPU-resident prefetch
are separate future work.

## Bounded host/GPU overlap for MoE expert misses

The native Vulkan Qwen3.8 path can overlap **CPU expert fallback calculations**
with the **final in-flight batch of packed expert uploads**. Previously it
waited for all transfers to finish before starting CPU fallback work. The
transfer now submits, CPU-only fallback runs while the GPU copy is in flight,
and the fence must complete before any transferred expert buffer is used.

This design is deliberately conservative:

- Bounded host expert caching and Vulkan allocation budgets still apply.
- Each staging window is reused only after its Vulkan fence completes.
  Earlier chunks still wait synchronously; this is not a dedicated async
  transfer queue.
- CPU host work runs outside the shared GPU-cache mutex.
- A host calculation error still waits for the transfer fence before buffers
  are released.
- Prefill does not evict decode-hot experts, and no routed expert is omitted.
- The Qwen3.8 memory report includes
  `cpu_gpu_transfer_overlap_batches` to count batches with both an outstanding
  transfer and some CPU fallback work, not measured overlapping milliseconds.

**Scope:** This is CPU calculation against GPU transfer of other selected
experts in the same layer. It does **not** yet overlap GPU compute with GPU
transfers, use a separate transfer queue, predict future routing, or establish
a token-rate improvement on the RX 9070 XT. If all selected experts fit the
GPU or all fall back to CPU, there may be no overlap opportunity.

Use identical weights, prompts and context settings to compare cold and warm
caches, memory pressure, CPU fallbacks, prefill/decode latency and numerical
correctness with the prior release.

## Configure the service in a browser

The native server embeds its own HTML, CSS and JavaScript. It needs no frontend
build, CDN or separate UI process. All executable names enter one native service, exposed at `/`. A model is optional at startup, so
you can find the files and configure the runtime before allocating GPU state.

### Windows

For a reliable first run, use **[Windows setup and troubleshooting](WINDOWS.md)**. In Windows PowerShell on a local Windows drive, from the repository root:

```powershell
.\scripts\configure-native-windows.ps1 -Build
```

This creates `target\release\nextora.exe`, starts a server without requiring a model, and opens the configuration UI. It does not automatically search `F:\` or allocate GPU memory until you choose a compatible model.

The URL is **http://127.0.0.1:9070/**. Keep the terminal running; stopping this
process stops the UI and API. If the browser does not open, visit that address
manually. The Windows helper `scripts/configure-native-windows.ps1` launches
the already-built independent executable with a per-user settings file.
It does not build, download weights, install a background Windows service or
change Application Control policy.

### Find the model on F:

Enter `F:\` in the model-folder field, then select **Scan models**. When you
know a likely folder, scan that instead: `F:\Models` or the actual Hugging
Face download directory. The scan is read-only. It reports accepted complete
GGUF models, rejection reasons, inaccessible folders and bounded-scan warnings.
It checks metadata, tensor layout and the complete shard set; it does not run
inference or prove model quality.

The package does not contain pretrained weights. No `.gguf` result can mean
the model was downloaded elsewhere, is still incomplete, or has not been
downloaded. A scan that hit an entry/depth limit has not searched the entire
drive. `.safetensors` files are reported as a format hint; the native loader
does not load the original Transformers weights directly. Download all GGUF
shards into the same folder and select the first shard. Original and
abliterated/uncensored models pass the same architecture and format checks.

If you used the earlier `nextora-cli models pull` command without
`--models-dir`, check its default download folder as well:

| Downloader process | Default model folder |
|---|---|
| Windows | `%LOCALAPPDATA%\nextora-cli\models` |
| Linux / native WSL | `$XDG_DATA_HOME/nextora-cli/models`, or `$HOME/.local/share/nextora-cli/models` when unset |
| Explicit override | `NEXTORA_MODELS_DIR`, if that environment variable was set |

These were defaults of the retired download command, not confirmation that
a download exists on your computer. A different downloader can use another
cache location. Scans skip symbolic links and report skipped paths; put a
complete, regular-file GGUF set in a chosen model folder when necessary.

Select the desired model, review the settings, then choose **Apply & load**.
Load and configuration changes require an idle service. If requests are active,
queued or preparing, the request returns 409 and your form remains available.
The previous model is released before a replacement loads; failures attempt
to restore the previous configuration and model.

### What the panel controls

- **Model and execution:** folder, selected GGUF checkpoint, backend and Vulkan
  adapter. CPU and diagnostic hybrid execution are explicitly identified.
- **Context and output:** context window and default maximum response length.
  Responses must still fit inside the rendered prompt plus output budget.
- **Admission:** one to four active requests, maximum queued requests and queue
  deadline. Additional requests share the same weights and compete for memory
  and compute; this is not continuous batching or a speed multiplier.
- **Memory:** native GPU allocation cap, owned expert-RAM cap, OS memory margin
  and bounded expert warmup. Zero GPU cap selects the existing automatic
  budget. Blank expert RAM means automatic; zero disables that cache.
- **Runtime status:** loaded model, execution backend, active/queued requests,
  API address and available runtime counters. The panel does not invent a
  token-rate measurement.

Choose one active request initially for latency evaluation. Increase context
and concurrency after observing the actual model's memory use and throughput.
The full model's metadata and available memory still govern acceptance; the UI
cannot make an unsupported checkpoint or an unavailable Vulkan adapter work.

### Save and reuse settings

`--settings-file PATH.json` enables persistence. If the file exists, its model,
model folder and runtime settings override those command-line values. Host,
port, API key, connection ceiling and CPU worker count remain startup options.
If the file does not exist, the first successful application creates it. A
failed load or invalid configuration does not replace the saved profile.

If a saved checkpoint, model drive or GPU backend is unavailable at startup,
the UI starts with no model loaded and shows the failure. You can correct the
path or settings there; the saved file stays unchanged until a successful
apply. An explicit invalid `--model` supplied directly on the command line
continues to report a startup error.

Only the explicit JSON file is written. The API key is neither stored there
nor returned in configuration responses. Without `--settings-file`, settings
live for the current process; **Export settings** downloads a profile you can
pass on the next launch. The UI's API-key input is held in page memory and is
not saved to browser storage.

The UI shell is accessible even when an API key is configured, so it can ask
for that key. Configuration, scans, hardware discovery and inference use the
same API authorization. Browser requests must come from the same origin;
command-line clients can continue using requests without an Origin header.

### Launch from WSL

The process that opens the files determines the path format. A **Windows
executable**, even when invoked from WSL, uses `F:\Models`. A **Linux executable
inside WSL** uses the mounted path, normally `/mnt/f/Models`.

For the Windows GPU process from WSL:

```bash
/mnt/c/code/nextora/target/release/nextora.exe serve \
  --open-ui --backend vulkan --models-dir 'F:\' \
  --settings-file 'C:\code\nextora\nextora-service.json'
```

For native Linux execution in WSL:

```bash
./target/release/nextora serve --open-ui --backend auto \
  --models-dir /mnt/f --settings-file ./nextora-service.json
```

The native WSL UI can start without a GPU model, but Vulkan inference still
requires a real exposed hardware adapter. If discovery shows only `llvmpipe`,
use the Windows executable for the Windows GPU path. Seeing `/dev/dxg` does
not establish Vulkan support. The UI does not install or repair drivers.
Microsoft documents the path and interop rules in
[Working across Windows and Linux file systems](https://learn.microsoft.com/en-us/windows/wsl/filesystems).

The helper `scripts/configure-native-wsl.sh` accepts Linux/WSL paths and can
launch either process explicitly:

```bash
# Native Linux process; default profile under XDG_CONFIG_HOME or ~/.config.
bash scripts/configure-native-wsl.sh --models-dir /mnt/f/Models

# Windows process, with path conversion performed by wslpath.
bash scripts/configure-native-wsl.sh \
  --windows-exe /mnt/c/code/nextora/target/release/nextora.exe \
  --models-dir /mnt/f/Models
```

The Windows-through-WSL helper stores a portable profile beside that executable
unless `--settings-file` is supplied. Both helpers accept `--no-browser` (or
`-NoBrowser` in PowerShell) for manual browser opening. They only launch an
already-built independent executable.

## The Avalon loading findings are retained

The original investigation identified a useful distinction between file size,
resident dense weights, routed experts and the large sparse PLE table. The new
loader implements that distinction directly, independently of the old runtime.

1. **Map complete shards without decoding the entire model.** Quantized tensor
   bytes remain in their stored format. The loader validates split counts,
   metadata consistency, duplicate tensors and data ranges.
2. **Keep an owned expert-RAM tier.** Its payload ceiling is calculated from
   currently available memory, an OS margin, dense-page allowance and concurrent
   sequence-state reserves. Linux accounts for a cgroup v2 limit where present;
   Windows queries available physical memory. Installed RAM is not assumed free.
3. **Read cache misses by file offset.** Linux attempts `O_DIRECT`; Windows
   attempts `FILE_FLAG_NO_BUFFERING`. Alignment scratch is bounded. Unsupported
   or failed direct reads use an explicitly counted buffered fallback. This is
   pageable owned memory, not Avalon's registered host arena.
4. **Protect active expert leases.** Immutable leases keep bytes valid during
   CPU computation and GPU upload. Capacity includes pinned payloads. If no
   reclaimable slot fits, execution bypasses admission instead of allocating an
   unlimited second cache. Per-model/shard/range keys prevent stale GPU hits.
5. **Warm only a bounded expert working set.** Gate/up/down triplets are admitted
   together. Round-robin layer coverage avoids filling only the first layers.
   This is not a pretrained popularity ranking or a full PLE-table scan.
6. **Protect decode reuse during prompts.** Prefill may hit cached experts but
   does not admit cold misses or refresh the decode LRU. The Qwen GPU path also
   prevents prefill misses from evicting decode experts.
7. **Keep PLE sparse and lazy.** Token embeddings and the selected PLE hash rows
   are gathered as needed; the 26.82-GiB table is not copied wholesale into RAM
   or VRAM. Recurrent/PLE histories belong to each request.
8. **Retain every routed expert.** The native path has no approximate expert-tail
   omission. CPU/GPU placement can change; the selected computation remains.

For the pinned GSQ-RCO IQ3_S checkpoint, the independently checked headers give:

| Category | Bytes | GiB | Policy |
|---|---:|---:|---|
| Routed experts | 50,292,326,400 | 46.838 | Owned bounded RAM tier and bounded hot GPU tier |
| Sparse PLE table | 28,800,138,240 | 26.822 | Lazy mapped rows |
| Other weights | 4,514,173,440 | 4.204 | Dense-page allowance; fixed GPU projections where applicable |

The source is `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`, its two `IQ3_S` shards. All 1,224
expected tensor descriptors were checked. This was a bounded header check,
not generation from the full pretrained checkpoint. The table does not mean
a 62-GB machine always has enough free RAM to cache all experts. Other programs,
WSL limits, dense pages and request state reduce the actual cache ceiling.

Avalon's async transfer/compute overlap, registered host buffers, expert ranking,
MTP, custom IQ3_KS format and elastic borrowing of unused KV memory are **not**
implemented in this native release. See [AVALON-COMPARISON.md](AVALON-COMPARISON.md)
for the mechanism-by-mechanism comparison and original sources.

## GPU execution and performance boundaries

Ordinary Qwen3 runs a resident device graph. Its prompt submissions group up to
32 tokens; generation uses one graph submission per token. Qwen3.8 retains
HC, DeltaNet, QSA, KV and nonlinear working state on the device. A layer returns
routing decisions and a small FFN input to the host for expert placement. Expert
work is combined with the next layer's dense/state work. A warm, fully fitting
expert set targets **layer count plus one submissions per token**, rather than
one synchronous operation for every matrix projection. Cache misses require
additional upload work and submissions; counters report that work.

The QSA path caches completed pooled keys and gathers selected attention cells.
The expert cache stores packed weights. GPU allocations for fixed matrices,
sequence state, staging and cached/transient experts share one tracked ceiling.
The default ceiling is **70% of the largest device-local heap**, not a claim
about currently free VRAM. `--vram-mib` sets an explicit owned-allocation limit.
Allocation failure is an error; it does not establish a hardware performance
result. Additional request states can evict unleased cache entries to fit.

This release has no continuous batching, async CPU/GPU expert overlap, MTP or
quantized KV. GPU queues share the physical card. Four independent requests can
be active, but they compete for compute, memory bandwidth and cache capacity.
Software-Vulkan parity validates computation, not AMD hardware speed.

Native WSL execution requires a compatible Linux Vulkan adapter. If only llvmpipe is available, `--backend auto` selects CPU; using the Windows executable may expose an actual Windows GPU instead. Physical performance remains unverified.

## Build on Windows and use from WSL

On Windows use a normal local Windows checkout such as `C:\code\nextora`
and Developer PowerShell with the MSVC/C++ toolchain installed:

```powershell
Set-Location C:\code\nextora
cargo build --release --locked -p nextora-engine
$Nextora = (Resolve-Path .\target\release\nextora.exe).Path
& $Nextora doctor
& $Nextora capabilities
```

The earlier Windows error executed a generated Windows build program through
`\\wsl.localhost\...` and Windows Code Integrity blocked it. Keep Windows
build outputs on a local Windows volume. If policy also blocks the local
executable, the policy owner must approve the build workflow; these scripts do
not disable application control. A Linux Cargo build belongs in the WSL
filesystem and produces a Linux binary, not a Windows executable.

For a Windows GPU server launched from WSL, use the actual Windows executable
and Windows model paths:

```bash
/mnt/c/code/nextora/target/release/nextora.exe serve \
  --backend vulkan --model 'C:\Models\your-model-00001-of-00002.gguf' \
  --models-dir 'C:\Models' --ctx 16384 --max-tokens 4096 \
  --parallel 1 --max-queued 32 --port 9070
```

Inference in this route runs on Windows. For a native WSL build:

```bash
cargo build --release --locked -p nextora-engine
./target/release/nextora doctor
./target/release/nextora serve --backend vulkan \
  --model "$HOME/models/your-model-00001-of-00002.gguf" \
  --models-dir "$HOME/models" --ctx 16384 --max-tokens 4096 \
  --parallel 1 --max-queued 32 --port 9070
```

Native WSL GPU execution requires a real Vulkan device with the needed compute
features. `/dev/dxg` alone is not proof. `llvmpipe` is rejected by default;
`--allow-software-vulkan` exists only for explicit numerical diagnostics. It
must not be used as a 9070 XT performance workaround. The native package does
not initialize ROCm/HIP or install `amdgpu-dkms`.

## Large coding projects and concurrency

Start with 16,384 context tokens and one active request for latency evaluation.
Both are explicit workload choices, not calibrated 9070 XT limits. Reserve
output inside the context: a 16,384-token window with 4,096 output tokens leaves
12,288 for the rendered system prompt, tool schemas, history and code. The
`/api/context` endpoint counts the actual rendered template and tokenizer.
Oversized requests fail before admission; there is no silent history truncation.

The existing `nextora-cli context` repository excerpt tool can still prepare
bounded query-focused source excerpts. It does not run repository code or make
the native server understand an entire repository automatically. Send relevant
files and preserve conversation prefixes to benefit from exact state reuse.

To accept four active requests with more work waiting:

```powershell
& $Nextora serve --backend vulkan --model $Model --models-dir C:\Models `
  --ctx 16384 --max-tokens 4096 --parallel 4 --max-queued 32 `
  --queue-timeout-secs 120 --ram-margin-mib 8192 --port 9070
```

States are allocated on demand and reused. Four is an admission ceiling;
allocation may still fail if that model/context cannot fit all states. A larger
context or another active request can reduce expert-cache residency. The host
planner reserves all configured slots. For CPU/hybrid Qwen3.8, its conservative
32,768-token estimate is 4,149,084,160 bytes per slot, including state growth and
rollback allowances. Four slots reserve 16,596,336,640 bytes before expert RAM.
Device mode accounts state in the device allocator instead. These estimates are
not memory reservations against competing programs or a promise of fit.

Use `--expert-ram-mib N` to cap the expert RAM payload, `0` to disable it, or
omit the option for automatic sizing. `--ram-margin-mib` defaults to 4096;
8192 is a reasonable starting allowance while builds and editors are active,
subject to the actual machine. `--expert-warmup false` disables bounded startup
warmup. Runtime/status JSON exposes the plan, residency, reads, cache hits,
evictions, fallback counts, device allocations and queue occupancy.

## Coding-client API

Base URL: `http://127.0.0.1:9070/v1`. Configure a client for OpenAI **Chat
Completions**, use a model ID returned by `/v1/models`, and match the API key if
`--api-key` is set. Binding beyond loopback requires a key of at least 16
characters. The API does not execute tool calls; the coding client executes
requested tools and sends their results back.

| Route | Purpose |
|---|---|
| `GET /` | Native configuration UI; no model required |
| `GET /api/config` | Effective runtime configuration, selected model and persistence status |
| `POST /api/config` | Validate and apply settings/model changes while idle; optionally persist |
| `POST /api/models/scan` | Read-only bounded folder scan with compatible models and rejection reasons |
| `GET /api/hardware` | Vulkan discovery; no inference submitted |
| `GET /health` | Loaded/ready status |
| `GET /v1/models` | Validated local MoE catalogue IDs |
| `GET /v1/status` | Runtime, memory/cache counters and active/queued requests |
| `POST /v1/chat/completions` | Text/tool completion, JSON or SSE |
| `POST /api/context` | Exact prompt/output context check |
| `POST /api/models/load` | Idle-only model switch using a catalogue ID |

The catalogue checks complete shards and supported geometry. The configuration
UI can scan again after a download and apply a newly selected folder/model
without restarting. The existing Hugging Face model-management commands can download GGUF
files; the native server itself does not download or execute repository code.
Original and abliterated/uncensored variants use the same compatibility checks.
Switch with `{"model":"relative/catalogue-model.gguf"}`. A model name in a
chat request cannot silently trigger replacement. Busy switching returns 409.
The old model is released before another large model loads; a failed replacement
attempt reloads the previous model when possible.

The template is the checkpoint's embedded Jinja template. Plain text fallback
has a limited Qwen ChatML form; tools require an embedded template. Request
messages support system/developer/user/assistant/tool text roles, tool-call
histories and matching tool-result IDs. Generated JSON or Qwen XML tool calls
become OpenAI function-call objects. Code-fenced and inline examples remain
text. Reasoning is separated only when an actual reasoning prefix is present.

Supported sampling controls include temperature, top-p, seed, frequency and
presence penalties, stop strings, and one completion per request. Greedy
sampling is the default. The API supports `tool_choice: auto` or `none` and
`parallel_tool_calls`; forced/strict-schema decoding is rejected. Images,
Responses API, Anthropic Messages and arbitrary JSON-schema constrained output
are not native endpoints. Unknown options fail clearly rather than being
silently ignored. A particular external client still needs its own integration
check; passing protocol fixtures is not certification of every coding client.

SSE uses stable IDs, a terminal finish reason, optional usage and `[DONE]`.
Ordinary text uses an incremental tokenizer decoder and suffix-only stop checks,
avoiding repeated decoding/copying of the entire answer. Tool/reasoning answers are buffered for
complete parsing before their deltas are emitted; incremental tool-argument
streaming is not claimed. Full queues return 429, queued timeouts 504, unavailable
models 503, and invalid requests 400. Disconnected active or queued requests
release their permits. Health/status requests do not acquire generation slots.

## Measuring this card without substituting a synthetic result

Use a representative code prompt and the actual downloaded GGUF:

```powershell
Get-Content -Raw .\code-prompt.txt | & $Nextora --threads 8 bench `
  --backend vulkan --model $Model --ctx 16384 --prompt - `
  --decode-steps 256 --requests 1 --warmup 1 --reps 5 `
  --json-out .\native-one-request.json
```

Repeat with two and four requests only after the one-request case works. The
JSON separates load/tokenization, prompt processing and measured decode, records
samples and the actual adapter, and exposes CPU expert fallback/cache counters.
Aggregate throughput is total executed decode steps divided by phase wall time;
per-request completion latency is separate. The fixed-workload benchmark runs
through EOS, while normal generation stops at EOS. A first generated token
comes from prefill, so generated-token count can exceed decode calls by one.

The synthetic tests check all logits, hidden/state vectors, quantized arithmetic,
resets and independent requests. Header acceptance and those comparisons do not
establish trained-model accuracy. Before calling this a fast production
replacement, run the chosen model on the RX 9070 XT, compare full-vector outputs
and useful code tasks, and measure one/two/four-request latency, memory and
throughput after long prompts. See [VALIDATION.md](../VALIDATION.md) for exactly
what was executed in this build.
