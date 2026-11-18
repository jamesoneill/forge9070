> Historical snapshot for v0.3.0. Published with path redaction and link repairs; the recorded test outcomes and limitations are retained. See [the evidence policy](../README.md).

# Forge9070 0.3 validation record

Date: 16 November 2026. This describes the environment used to prepare the
source package, not Jamie's PC. Earlier release records are retained under
[v0.1.0/](../v0.1.0) and [v0.2.0/](../v0.2.0).

## Environment and limits

- Linux x86_64, kernel `6.18.57`; this is not a WSL kernel.
- Rust `1.100.0 (0ef1bc386 2026-11-09)`.
- No RX 9070 XT or other usable Vulkan GPU driver is available.
- The Vulkan loader is present, but instance/device discovery fails without a
  usable driver. The doctor records `compute_status: not_tested`.
- Windows, MSVC, Windows/WSL interoperability and the target graphics stack are
  unavailable in this environment.

## Build and automated checks

| Check | Observed result |
| --- | --- |
| Development build (`cargo build --locked`) | Passed |
| Release build (`cargo build --release --locked`) | Passed in 166.96 seconds |
| Root unit tests | 22 passed |
| Actual CPU inference/ownership tests | 5 passed |
| Engine library tests | 138 passed, including 19 HF and 9 compatibility tests |
| Actual model-switch/catalogue tests | 2 passed |
| Server regression tests | 25 passed |
| Release executable/HTTP smoke | 8 checks passed |
| Release native Linux launcher/model CLI | 9 checks passed |
| Release overlapping clients/admission/accounting | 8 checks passed |
| WSL launcher argument/environment mocks | 6 passed |
| Browser JavaScript and shell syntax | Passed |
| Live bounded Hugging Face metadata checks | Search and two repository listings passed; see qualifications below |

**192 Rust tests passed, 0 failed.** The final modified-scope test command is:

```bash
cargo test --locked --workspace --lib --bins \
  --test inference --test model_switch --test serve
```

[rust-tests.json](rust-tests.json) records the command, exit status and group totals.
[builds.json](builds.json) records the build results. [validation.json](validation.json)
provides the combined machine-readable summary. `smoke-release.json`,
`models-cli-release.json` and `concurrency-release.json` record the final release
executable checks. The separate `concurrency-debug.json` records an earlier
development build. Some upstream GPU-dependent
cases return without GPU execution when the relevant runtime is unavailable;
the Rust test count does not certify any GPU kernel.

The initial regression runs exposed an implicit prompt chunk larger than a
short context after loading, which prevented a later reload, and new expert
metrics remaining present without an active expert cache. Both were corrected
before the final passing Rust run. The tests were not weakened to accept these
behaviours.

To reproduce the executable checks:

```bash
cargo build --release --locked
python3 tests/smoke.py --binary target/release/forge9070
python3 tests/models_cli.py --binary target/release/forge9070
python3 tests/concurrency_cli.py --binary target/release/forge9070
python3 scripts/test-wsl.py
```

All established inference in this environment uses the explicit CPU path.

## Real MoE inference, model switching and compatibility

The new fixture is a genuine one-layer Qwen3MoE computation with four
nonidentical experts and top-2 routing, nonzero F32 weights, a 32-dimensional
hidden state and a byte-level tokenizer. Its deliberate output projection
selects `A`. It contains 15 tensors, 182,656 tensor-payload bytes and is 188,256
bytes on disk. The routing test changes the active-expert count from two to
one and observes changed hidden states; constant output text alone would not
establish executed expert routing.

MoE fixture SHA-256:

```text
a5c3df3b15a2d24017176a8d5ff742c5d8ca537aef599933bf2d6147392deba7
```

The original 113,696-byte dense fixture remains for lower-level inference and
negative policy tests. It is excluded from the user-facing MoE-only workflow.
The fixtures are generated from original synthetic data, contain no pretrained
weights and establish neither language quality nor realistic model speed.

The model-switch integration tests derive a second checkpoint whose projection
emits `BBBB` instead of `AAAA`. They use identical basenames under different
repository/revision directories, load and unload through the real HTTP server,
and verify the changed generated output. They also check:

- Invalid metadata, dense models and unsupported execution options fail before
  the existing working model is released.
- Direct load requests cannot bypass the catalogue's MoE policy.
- A stale explicit OpenAI model ID receives HTTP 409.
- Authentication, catalogue containment, complete shards and symlink exclusion
  remain enforced.
- Management remains available without a loaded model, and reload succeeds.
- Model-owned synthesized weights remain valid after LoRA overrides are cleared.

The shared compatibility tests check actual architecture/expert metadata,
required tensors, shapes, block alignment and byte bounds, including bounded
remote headers. Unsupported Avalon custom quantization IDs and malformed
metadata are rejected. MTP tests validate ordinary standalone heads, projection
and attention geometry, normalization aliases and matching target metadata/token
IDs. A matching shape still cannot prove training/checkpoint provenance.

Metadata acceptance is a format preflight. It does not establish numerical
correctness, model quality, host-RAM/VRAM fit or hardware performance.

## Concurrent request handling

The concurrency harness submits four overlapping clients, two with JSON
responses and two with SSE streams, to a real CPU inference server. It checks
complete isolated 64-token outputs, four counted requests and 256 actual
completion tokens. It also fills the eight-connection admission limit, checks
HTTP 503 with Retry-After, disconnects incomplete clients and confirms recovery.

The benchmark-accounting check uses four measured requests plus one excluded
warmup, counts actual returned completion usage, and records the execution
configuration. It distinguishes total tokens divided by batch wall time from
server decode rates. Configuration changes, failed requests and stale model IDs
make a benchmark fail rather than silently produce a throughput claim.

**CPU computation is serial.** The server correctly reports one compute slot.
These checks establish overlapping network handling, queueing and isolation;
they do not establish GPU multi-sequence execution or acceleration. The GPU
scheduler's up-to-four-slot path was reviewed in source, but needs measurement
on the target card with a compatible model. MTP and multiple GPU slots cannot be
combined, and certain model features require one slot.

## Hugging Face metadata and transport

The 19 local HTTP fixture tests exercise immutable revision selection, shard
completeness, actual metadata policy checks, bounded byte-range header probes,
streaming downloads, resumed ranges and restart when ranges are ignored, file
size and LFS checksum checks, provenance, path containment, redirects and proxy
selection. Successful pinned probes are cached without weight data. Downloads
stay on the exact checked commit. These tests use local synthetic payloads.

Live Rust CLI checks also used the configured HTTPS proxy and platform trust,
with certificate and hostname verification enabled. They inspected real GGUF
headers, not just repository names. The stored reports are:

- [compat-hf-live-search.json](compat-hf-live-search.json): a compatible abliterated Qwen3MoE
  search result with 128 experts and 8 active experts per token.
- [compat-hf-live-abliterated.json](compat-hf-live-abliterated.json):
  `mradermacher/Qwen3-30B-A3B-abliterated-GGUF`, commit
  `5eed9fdfc8dd2a614fa0f344befbb68d9e049978`. Nine variants passed during the
  first bounded listing. Q3_K_M was 14,711,846,880 bytes; Q4_K_M was
  18,556,686,304 bytes. Two variants remained unverified when the time budget
  expired.
- [compat-hf-live-hybrid.json](compat-hf-live-hybrid.json):
  `unsloth/Qwen3.5-35B-A3B-GGUF`, commit
  `bc014a17be43adabd7066b7a86075ff935c6a4e2`. Three variants passed during the
  first bounded listing, including the 16,356,375,168-byte Q3_K_M, with 256
  experts and 8 active per token. MXFP4 type 39 was explicitly rejected;
  other choices remained unverified when the budget expired.

Those two initial listings each reached the 180-second operation budget.
Unverified choices are excluded with a reason, rather than guessed compatible.
Their reports predate the final header-read batching and file-priority changes;
the final search report was refreshed after the last development build and
passed in 22.02 seconds. No claim
is made that every variant in either repository was validated.

No complete pretrained checkpoint was downloaded or executed. Local transport
fixtures cover complete transfers; live headers do not certify a model's
inference quality or quantify CDN download performance.

## Tuning, cache identity and approximation

Unit tests cover model/cache namespace separation across paths, revisions,
shards, auxiliary heads and metadata changes. Calibration results must match
the model and recorded runtime configuration; old basename-based records are
ignored. Unsupported draft-head configurations cannot be selected by tuning.
No actual GPU tuning run took place here.

The model identity is based on canonical paths, sizes and modification times,
not a full weight checksum. The device identity uses the reported name, index
and total VRAM, not a hardware UUID or driver version. Edits that preserve all
recorded metadata and driver changes require manual cache clearing/recalibration.

Seven pure policy tests cover the new optional expert-tail omission: one-based
rank, keeping cached experts, retaining an expert ranked higher in another row,
grouping repeated experts, unchanged weights, explicit zero contributions when
all misses are omitted, and exact prefill/draft eligibility. The default is off.
The implementation changes CPU miss selection and publication, not GPU shaders.

Approximation can change output and can depend on other requests/speculative
rows in the same batch. Neither its GPU execution nor speed/quality tradeoff
was validated here. Benchmark output labels approximation explicitly and does
not present inferred routed bytes as measured hardware bandwidth.

## WSL and browser checks

Six mock-launcher tests cover argument/path preservation, explicit CPU mode,
GPU preflight ordering/failure, ports, Windows path conversion, directory
precedence and environment-token forwarding. The model CLI harness also uses
the actual native Linux launcher and executable in CPU mode. This is not an
actual WSL or Windows interoperability test.

The browser's inline JavaScript passes `node --check`; all four shell launch/
build/setup scripts pass `bash -n`. No visual browser inspection is claimed.
No packages, graphics drivers, ICDs, WSL settings or firewall rules were changed.

## Work still requiring target hardware

- Native Windows build/execution and actual Windows/WSL interoperability.
- Native WSL Vulkan compute and the installed graphics driver.
- RX 9070 XT kernels, GPU multi-sequence execution, numerical parity and MTP.
- Quantized large models, expert-cache behaviour, long contexts and RAM pressure.
- Actual single-request/aggregate token rates, latency and tuning choices.
- Output quality and speed when approximate expert omission is enabled.

The engine uses Vulkan, with no HIP/ROCm backend. This package does not repair
the earlier APT/DKMS installation or turn ROCm 6.3.4 into a supported runtime for
the card. GPU discovery and pipeline checks must be followed by real generation
on the target machine. See `docs/PERFORMANCE.md` for the measurement procedure
and `docs/AVALON-COMPARISON.md` for the source audit and unimplemented Avalon
mechanisms. The archive does not claim full Avalon parity or its token rate.
