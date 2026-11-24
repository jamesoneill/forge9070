# Nextora validation record

## Integrated model setup and Vulkan decode, 24 November 2026

The integrated source on top of `5ead89a` passed 197 Linux and 196 Windows
Rust tests. The Linux release build passed 21 CLI checks, 30 HTTP checks and
79 desktop/mobile UI checks. The model setup suites exercised Hub downloads
against a local fixture server, per-model tuning in a separate worker, service
restoration, explicit activation, chat and live Insight. Seven long-context
UI checks covered model eligibility and saved settings.

On Windows the RX 9070 XT matched the original F32 attention kernels at
40,959 and 40,960 populated positions. At 40,960 positions the comparison
covered all 1,310,720 scores and 4,096 weighted outputs. All eight shipped shader
modules rebuilt byte-for-byte from the checked-in shader sources. For
checkpoint, driver, cache and throughput details, see the
[AMD performance record](docs/AMD-PERFORMANCE.md).

## Historical checks

The records below include results obtained before the independent runtime
replaced the former two-runtime implementation.

Validated 17 November 2026. The independent package is `nextora-engine` 0.2.1;
the encompassing project is Nextora 0.9.1.

**The optimized Linux build, native unit tests, coding API regression checks
and new configuration API checks passed. Browser rendering and interaction
were not verified because neither available browser route could open the
local test service. Windows, native WSL GPU execution and RX 9070 XT
performance remain untested here.**

The linked reports below are published copies of the original validation
records, with ephemeral workspace paths redacted. See the
[evidence policy and history](docs/validation/README.md). These are historical
results, separate from subsequent GitHub Actions runs.

## Native GPU portability — 19 November 2026

The capability-based selection update was tested on Linux x86_64 with the
pinned Rust 1.100.0 toolchain and locked, offline dependencies. It adds an
explicit `auto` policy, capability-filtered vendor-neutral Vulkan selection,
CPU-only diagnostics and configuration/reporting support. The local standalone
binary was a development build, not a release-performance benchmark.

| Check | Local result |
| --- | --- |
| Native library and binary tests | 90 passed, 0 failed; 4 explicit external-template/software-Vulkan gates ignored |
| New adapter and backend-selection tests within that total | 15 passed |
| Standalone CLI with real synthetic CPU inference | 18 checks passed, including automatic CPU selection with Vulkan disabled |
| HTTP/SSE, model switching, queues and cancellation | 28 checks passed |
| Configuration API and persisted automatic policy | 56 checks passed; optional browser group skipped |
| Existing UI helper checks | 13 passed |
| Release provenance/archive regression checks | 13 passed |

Automatic CPU inference produced the same tokens as explicit CPU and reserved
the same host-state budget. Explicit GPU settings failed instead of silently
selecting CPU, and malformed checkpoints remained errors. Adapter tests use
synthetic capability snapshots; their AMD/NVIDIA/Intel names are not hardware
execution evidence.

The local machine exposes no Vulkan ICD. A separate Chromium attempt passed
47 API checks before the environment denied the browser's process-singleton
socket creation; no browser page or screenshot was produced. The existing CI
gates cover Windows/Linux, Chromium interaction and software Vulkan. Read their
actual result for the PR revision; a configured gate is not a passing result.
Physical GPU execution, speed and full pretrained-model acceptance remain
unverified. See [hardware support](docs/HARDWARE-SUPPORT.md) and the concise
[portability validation record](docs/validation/gpu-portability-2026-11-19.json).

The older records below are retained as historical evidence.

## Publication hardening — 18 November 2026

The publication review corrected native Host validation for wildcard listeners:
an accepted connection is now checked against its actual local interface address.
Authenticated LAN clients can use `--host 0.0.0.0` or `--host ::` while unrelated
Host values remain rejected. Focused unit cases and three real-HTTP regression
assertions were added. This change does not alter model arithmetic.

Local source checks passed for Python, Bash, JavaScript and structured data.
The three tiny GGUF fixtures and all 134 historical validation-manifest entries
matched their recorded hashes. The existing UI helper suite passed 13 checks,
the WSL argument harness passed six stub-based checks, and 13 new release
authorization/archive-integrity regression tests passed.

The README masthead uses the committed SVG mark and live text. Four raster
brand assets remain pending publication; the [brand guide](docs/BRAND.md)
records their filenames and status without presenting download links.

The initial publication pass did not have a Rust toolchain. During the
18 November continuation, Rust 1.100.0 was installed in an isolated workspace
and the current source was rebuilt with the locked dependency set and portable
`-C target-cpu=x86-64` settings. Both release entry points built successfully.

Fresh continuation checks passed. The [continuation record](docs/validation/publication-linux-2026-11-18.json)
includes fixture and executable hashes and records the unexecuted gates:

| Check | Result |
| --- | --- |
| Native library and binary tests | 75 passed; 4 explicitly gated tests ignored |
| Engine, quantization and Qwen3.8 parity suites | 18 passed; 3 Vulkan-gated tests ignored |
| CLI and frontend dispatch | 16 checks passed |
| HTTP, SSE, admission, queueing and cancellation | 28 checks passed |
| Configuration and discovery API | 53 checks passed; browser group not run |
| Synthetic model regeneration | All three fixtures reproduced byte for byte from the published generators |
| Required vendored binary files | 88 files verified against the pinned public upstream Git revision |

These are Linux CPU results with synthetic, untrained models. Windows checks,
Chromium interaction and software-Vulkan execution remain delegated to the
expanded [CI workflow](.github/workflows/ci.yml); read their live result from
[GitHub Actions](https://github.com/jamesoneill/forge9070/actions/workflows/ci.yml).
Configured gates are not a claim that they have already passed. The historical
17 November reports below remain unchanged. Target-hardware and pretrained-model
acceptance still need the checks described at the end of this file.

## Checks executed for this update

| Gate | Result | Evidence |
|---|---|---|
| Optimized standalone and frontend build on Linux x86_64 | Passed, 62.60 seconds | [build-release-0.9.1.json](docs/validation/build-release-0.9.1.json) |
| Native Rust library and binary unit tests | 75 passed, 0 failed; 4 separately gated tests ignored | [native-core-tests-0.9.1.json](docs/validation/native-core-tests-0.9.1.json) |
| Native CLI and `nextora-cli engine` dispatch | 16 checks passed, none skipped | [nextora-engine-cli-0.9.1.json](docs/validation/nextora-engine-cli-0.9.1.json) |
| Existing native HTTP/SSE/model switching/admission | 28 checks passed | [native-api-0.9.1.json](docs/validation/native-api-0.9.1.json) |
| New configuration and model discovery API | 50 checks passed; one browser group skipped | [native-ui-0.9.1.json](docs/validation/native-ui-0.9.1.json) |
| Launcher arguments and Python/Bash syntax | 6 checks passed; Windows conversion used a stub | [native-ui-launchers-0.9.1.json](docs/validation/native-ui-launchers-0.9.1.json) |
| Final HTML IDs, JavaScript/Python syntax, CSP structure and exact embedded asset bytes | 9 checks passed; no browser execution implied | [native-ui-source-checks-0.9.1.json](docs/validation/native-ui-source-checks-0.9.1.json) |
| Automated Chromium workflow | Blocked before opening a page by denied browser socket operation | [native-ui-browser-attempt-0.9.1.json](docs/validation/native-ui-browser-attempt-0.9.1.json) |
| Separate browser interface | Local test service connection refused; no page rendered | [native-ui-browser-preview-0.9.1.json](docs/validation/native-ui-browser-preview-0.9.1.json) |

The four ignored native tests are explicit Vulkan or external-template gates.
Their existing 0.9.0 evidence remains available; they were not rerun for this
UI/server update. The test count above does not include the previous release's
391-test aggregate as though it were newly executed.

## Configuration API behavior actually exercised

The focused harness launches the built release executable as a child process
with a temporary profile, a private loopback port and an original, untrained
CPU fixture. It performs actual inference after loading; it is not an HTTP
mock.

The checks cover:

- Starting without a model; serving the public UI shell/assets while API
  authentication remains enforced; health reports not-ready until loading.
- Exact same-origin and Host checks, JSON mutation transport, CSP and asset
  content types, and keeping the API key out of the configuration and profile.
- Scanning a folder with spaces and literal HTML-like filenames; accepting
  supported complete sparse MoE models; explaining dense, malformed and
  incomplete checkpoints; reporting safetensors and skipping directory links.
- Applying real context, output, concurrency and queue settings; returning
  the expected output from the loaded synthetic model.
- Rejecting invalid settings, missing files and traversal outside the selected
  model root without changing the prior runtime or saved profile.
- Rejecting configuration changes with HTTP 409 during actual active generation.
- Preserving the old runtime/settings when persistence cannot be written.
- Unloading, rejecting subsequent generation while unloaded, and retaining the
  configured service settings.
- Restarting from a saved profile with and without a loaded model.
- Keeping the configuration UI available when a saved checkpoint or its entire
  directory is missing, while leaving the saved profile unchanged.

The existing 28-check API harness also rechecks exact prefix reuse, unrelated
history reset, prompt/output bounds, Unicode SSE, stop boundaries, tool-call
transport, idle model switching, real one/four-active admission, queued and
active cancellation, queue capacity and queue deadlines.

## Browser verification limitation

The optional Playwright/Chromium run completed its preceding API checks, then
Chromium failed before creating a page:

```text
FATAL:chrome/browser/process_singleton_posix.cc:297
Check failed: . socket() failed: Operation not permitted (1)
```

No browser policy or sandbox protection was altered. A separate supported
browser interface was then tried against the running test service and
returned `net::ERR_CONNECTION_REFUSED`. The test service was stopped
afterwards.

Consequently there are **no verified desktop/mobile screenshots, no browser
overflow result and no passed real-browser interaction assertions** for this
release. HTML/JavaScript source checks and successful API requests do not
substitute for those checks.

The included browser harness is ready to test the actual served UI in a
browser-capable environment. It exercises authentication, scan/select/apply,
unload, profile export, literal filenames, draft preservation during status
polling, folder errors, and desktop/mobile overflow.

## What changed, and what was retained

The native server now has optional-model startup, embedded UI assets, hardware
discovery, read-only bounded model scanning, runtime configuration and optional
JSON profile persistence. Runtime changes use the same idle admission barrier
as model replacement. A model's default output limit is held with that
immutable loaded model so request preparation cannot mix configurations.

Scans check complete GGUF shards, supported MoE layout, tokenizer vocabulary
and chat-template construction. They do not submit GPU compute, warm expert
caches or allocate request state. They stop at 20,000 entries, 16 directory
levels or 1,024 eligible models and report partial results and skipped links.

This update does not change the model arithmetic, tokenizer, quantization
kernels, Vulkan shaders or bounded expert cache implementation from 0.9.0.
All selected experts are still computed. The Avalon-derived owned expert-RAM
tier, bounded GPU tier, lazy PLE rows and prefill cache policy are retained.

The 0.9.0 validation record is preserved at
[v0.9.0/VALIDATION.md](docs/validation/v0.9.0/VALIDATION.md), alongside the
original detailed artifacts. It contains the software-Vulkan numerical gates,
four-active-request GPU fixture test, official-template comparisons and
1,224 real-checkpoint descriptor checks. Those historical results are
numerical/layout evidence, not a current RX 9070 XT speed or pretrained-quality
measurement.

## Reproduce the current checks

From the project root:

```bash
cargo build --release --locked -p nextora-engine -p nextora-cli
cargo test --locked -p nextora-engine --lib --bins
python3 scripts/validate-nextora-engine.py \
  --binary target/release/nextora-engine --frontend target/release/nextora-cli \
  --output artifacts/nextora-engine-cli-local.json
python3 scripts/validate-native-api.py \
  --binary target/release/nextora-engine --output artifacts/native-api-local.json
python3 scripts/validate-native-ui.py \
  --binary target/release/nextora-engine --output artifacts/native-ui-local.json
```

For browser testing, install/use the test environment's Python Playwright and
Chromium, then add `--browser` and, when needed, `--chromium /path/to/chromium`.
Screenshots are optional test outputs; they are not needed by the product.
The embedded UI has no Node/frontend build or external asset dependency.

## Target-machine acceptance remains outstanding

No Windows build, PowerShell launcher execution, real RX 9070 XT inference,
native WSL Vulkan execution, complete pretrained Qwen3.8 load, long-context
quality test or external coding-client integration was performed here. The
WSL launcher argument tests used a path-conversion stub and do not establish
Windows interoperability.

The UI does not expose a hardware Vulkan adapter to WSL or repair drivers.
The earlier llvmpipe-only report remains a software adapter. The Windows
executable is a separate supported launch route for using the Windows driver;
actual device discovery and model execution still need to succeed on the PC.

Large context and four active requests consume additional state memory and
share the GPU. No continuous batching, speculative decoding, asynchronous
CPU/GPU expert overlap or new token-rate improvement is claimed by this UI
update. The source package does not include pretrained model weights.

