> Historical snapshot for v0.4.0. Published with path redaction and link repairs; the recorded test outcomes and limitations are retained. See [the evidence policy](../README.md).

# Forge9070 0.4 validation record

Date: 16 November 2026. This describes the environment used to prepare the
source package, not Jamie's PC. Earlier release records are retained under
[v0.1.0/](../v0.1.0), [v0.2.0/](../v0.2.0) and [v0.3.0/](../v0.3.0).

## Environment and limits

- Linux x86_64, kernel `6.18.57`; this is not a WSL kernel.
- Rust `1.100.0 (0ef1bc386 2026-11-09)`.
- No RX 9070 XT or other usable Vulkan GPU driver is available.
- The Vulkan loader is present, but device discovery fails without a usable
  driver. The doctor records that no compute was submitted.
- Windows, Windows/WSL interoperability and the target graphics stack are
  unavailable in this environment.

## Build and automated checks

| Check | Observed result |
| --- | --- |
| Development build (`cargo build --locked`) | Passed in 29.48 seconds |
| Release build (`cargo build --locked --release`) | Passed in 146.41 seconds |
| Root unit tests | 22 passed |
| Actual CPU inference/ownership tests | 5 passed |
| Engine library tests | 150 passed |
| Actual model-switch/catalogue tests | 2 passed |
| Sampler integration tests | 10 passed |
| Server regression tests | 27 passed |
| Release executable/HTTP smoke | 9 checks passed |
| Release native Linux launcher/model CLI | 9 checks passed |
| Release overlapping clients/admission/accounting | 9 checks passed |
| Release benchmark accounting | 4 checks passed |
| WSL launcher argument/environment mocks | 6 passed |
| Browser JavaScript and four shell scripts | Syntax passed |

**216 Rust tests passed, 0 failed.** The final modified-scope command was:

```bash
cargo test --locked --workspace --lib --bins \
  --test inference --test model_switch --test serve --test ops
```

[rust-tests.json](../rust-tests.json) records the command, exit status and group totals.
[builds.json](../builds.json) records both builds. [validation.json](validation.json) provides
an aggregate machine-readable summary. The final release executable reports
are `smoke-release.json`, `models-cli-release.json`, `concurrency-release.json`
and `bench-release.json` under `artifacts/`. These contain 31 passed checks.

Some upstream GPU-dependent tests return without GPU execution when the
runtime is unavailable. A passing Rust test count does not certify a GPU kernel.
All established inference here uses the explicit CPU path.

To reproduce the executable checks:

```bash
cargo build --locked --release
python3 tests/smoke.py --binary target/release/forge9070
python3 tests/models_cli.py --binary target/release/forge9070
python3 tests/concurrency_cli.py --binary target/release/forge9070
python3 tests/bench_cli.py --binary target/release/forge9070
python3 scripts/test-wsl.py
```

## Version 0.4 correctness and measurement changes

The Forge profile now defaults to F16 KV, one request slot and MTP off, and
rejects nonzero expert-tail omission. The actual model's quantized weights and
floating-point arithmetic remain in use; this is not full-precision equivalence
or a guarantee of factual answers.

Greedy GPU verification formerly left the RoPE position buffer stale. It now
shares current KV/RoPE uploads and input bounds with full-logit verification,
while retaining GPU argmax-only readback. Three pure tests cover advancing
positions, offsets, context/batch bounds and overflow. These validate input
preparation, not GPU numerical execution. CPU greedy, sampling and draft
acceptance reject empty or mixed non-finite logits during their existing scans.
Unit and sampler integration tests exercise that behaviour. GPU argmax has no
new GPU-wide non-finite detection shader.

HTTP tests confirm that non-neutral unsupported penalties, min-p, typical-p and
logit bias are rejected rather than silently ignored. Neutral defaults remain
accepted. Sampling values that cannot retain their required finite/range
properties when narrowed to f32 are also rejected.

Actual GGUF tensor types now provide compression counts, including sub-four-bit
weights and routed-expert subsets. Mixed-format tests and remote/local agreement
checks exercise that report. The UI requires explicit weight-file selection
and does not infer a model-quality score from file size or compatibility.

## Calibration and benchmark accounting

The new tuner compares chunk/MTP candidates to a no-MTP baseline while keeping
model, KV, context, memory margins and worker count fixed. It uses the same
adaptive expert-placement policy as normal generation/serving, fresh candidate
allocations and an unchanged saved routing profile. Version-3 recommendations
are ignored.

Warmup is excluded. Each measured sample pairs actual output tokens with its
own decode time, and selection uses the median across measured rounds. Every
measured token sequence and stopping behaviour must match the baseline for the
same prompt. At least 32 generated tokens are required per workload; unstable
or insufficient baselines invalidate earlier usable v4 results. An eligible
candidate must improve the baseline by at least 3% to replace it. Thinking is
enabled and recorded; reasoning tokens count toward generation speed.

Pure tests cover paired accounting, missing/duplicate/non-finite samples,
output/stopping comparisons, bounds and the selection threshold. The release
benchmark test executes a synthetic MoE model, excludes warmup, accounts for
128 actual output tokens across four measured samples, reconciles the rates
and records prompt/checkpoint identities. No actual GPU calibration ran here.

Finite greedy-output agreement is a regression check, not proof of identical
sampled outputs or factual correctness across all prompts. Changing placement
or batched kernels can alter floating-point rounding. The tuner does not choose
checkpoint or KV precision. A Q8 run compares candidates against its own Q8
baseline and cannot establish equivalence to F16.

Cache identities use canonical paths, sizes and modification times rather than
full weight checksums. GPU identity uses reported name, index and total VRAM,
not a hardware UUID or driver version. Preserved-metadata weight edits and
driver changes require manual cache clearing/recalibration.

## Real MoE inference and model management

The original synthetic MoE fixture has one Qwen3MoE layer, four nonidentical
experts, top-2 routing, nonzero F32 weights, 32 hidden dimensions and a byte-level
tokenizer. Its deliberate output projection emits `A`. A routing test changes
the selected expert count and observes changed hidden states, which establishes
executed expert routing independently of constant output text.

The fixture is 188,256 bytes, with 15 tensors and 182,656 tensor-payload bytes.
Its SHA-256 is:

```text
a5c3df3b15a2d24017176a8d5ff742c5d8ca537aef599933bf2d6147392deba7
```

The 113,696-byte dense fixture remains for lower-level inference and negative
MoE-policy tests. Neither fixture contains pretrained weights or establishes
language quality, realistic speed or quantized-kernel correctness.

Model-switch tests derive a second checkpoint whose projection emits `BBBB`
instead of `AAAA`, use identical basenames in different catalogue directories,
and verify changed generated output through the real HTTP server. They cover
unload/reload, invalid metadata and option rejection before release of the
working model, stale explicit model IDs, authorization, path containment,
complete shards and symlink exclusion. Synthesized weights remain model-owned
and valid after LoRA overrides are cleared.

Shared compatibility checks inspect architecture, expert geometry, tensor names,
types, shapes, alignment and byte bounds. Unsupported Avalon custom types and
malformed metadata are rejected. MTP geometry checks cover ordinary heads and
matching token/target metadata; matching shapes do not prove training provenance.
Acceptance is a format preflight, not proof of GPU correctness or memory fit.

## Concurrent requests

The final harness submits four overlapping clients, including JSON and SSE
responses, to real CPU inference. It verifies four isolated 64-token replies,
256 counted completion tokens, an eight-connection admission limit, HTTP 503
with Retry-After under saturation, disconnect recovery and responsive health.
CPU compute is serial and correctly reports one active compute slot.

The concurrency benchmark excludes warmup and uses actual completion usage.
It distinguishes aggregate tokens divided by full batch wall time from per-run
decode rates. A new test reloads the same model ID with a different KV setting
and verifies that the configuration snapshot detects the change. Failed
requests, stale IDs and changed reported configuration invalidate a benchmark.

This proves overlapping network handling and isolation, not GPU multi-sequence
execution or acceleration. Supported GPU paths allow up to four slots, with
architecture restrictions; MTP cannot be combined with multiple active slots.
Those paths still require target-hardware evaluation.

## Hugging Face checks

The 19 local HTTP fixture tests exercise immutable revisions, shards, bounded
header probes, actual metadata policy, streaming downloads, resume/restart,
size/hash checks, provenance, containment, redirects and proxy selection. They
use synthetic local payloads. TLS verification was not bypassed.

[compat-hf-live-search.json](../compat-hf-live-search.json) records the bounded v0.4 live search on
the finished development build, including observed compression metadata.
The earlier full repository listings and v0.3 source scope are preserved in
[v0.3.0/](../v0.3.0); those historical results do not certify the final build.
Unverified variants are excluded rather than presumed compatible. No complete
pretrained checkpoint was downloaded or executed during v0.4 validation.

## WSL, browser and remaining hardware work

Six launcher mocks check argument/path preservation, explicit CPU mode, GPU
preflight ordering/failure, ports, Windows path conversion, directory precedence
and token forwarding. The model CLI harness also runs the native Linux launcher
and CPU executable. Neither establishes actual Windows/WSL interoperability.
The inline browser JavaScript and four shell scripts pass syntax checks; no
visual browser inspection is claimed.

The following remain untested here: native Windows and WSL GPU execution;
RX 9070 XT kernels, numerical parity, multi-sequence execution and MTP;
large quantized checkpoints, long contexts, expert-cache/RAM pressure; actual
token rates, calibration choices and task-answer quality. Expert omission is
disabled in the supported Forge workflow; its retained historical source tests
do not establish an advertised approximation mode.

No packages, drivers, WSL settings or firewall rules were changed. The backend
is Vulkan, with no HIP/ROCm runtime. This package does not repair the earlier
APT/DKMS failure. Discovery and pipeline checks must be followed by actual
generation on the target machine. See `docs/PERFORMANCE.md`, `docs/ACCURACY.md`
and `docs/AVALON-COMPARISON.md` for the measurement procedure, quality limits
and remaining Avalon differences. No Avalon-equivalent token rate is claimed.
