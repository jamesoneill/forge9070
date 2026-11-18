> Historical snapshot for v0.6.0. Published with path redaction and link repairs; the recorded test outcomes and limitations are retained. See [the evidence policy](../README.md).

# Forge9070 0.6.0 validation record

Uses RustingStrata by Vasyl Trefilov. This release adds coding context presets,
bounded repository excerpts, exact request budgets and corrected conversation-state
persistence. Validation ran on 17 November 2026 UTC.

## Observed results

| Check | Observed result |
| --- | --- |
| Rust unit suite | **231 passed, 0 failed**: 34 frontend and 197 engine |
| Optimized Linux x86_64 release build | Passed with `--release --locked --offline` |
| Code-context CLI / Git / HTTP integration | **33 commands and assertions passed**, plus a captured server log |
| Expert RAM regression integration | **21 commands and assertions passed**, plus a captured server log |
| CPU generation / authenticated HTTP / SSE smoke tests | **9 checks passed** |
| Actual UI JavaScript helpers in Node VM | **7 checks passed** with minimal element/storage stubs |
| Complete inline JavaScript syntax | Passed `node --check` |
| Rendered Chromium browser integration | **Blocked before page creation** by environment `socket()` EPERM; zero browser assertions ran |
| RX 9070 XT / Windows / full Qwen inference | Not executed; no target hardware or Windows runtime here |

Complete outputs and source hashes are recorded in [validation.json](validation.json)
and the versioned records below. Browser setup/cleanup records are not counted
as successful browser assertions.

## Context and repository checks

Frontend tests check preset values and explicit overrides. Repository tests
check deterministic lexical ranking, original line labels, complete UTF-8
lines, whole-pack token counts, byte/scan bounds, visible omitted tails,
tracked/untracked behavior, scoped monorepo inventory, include-prefix priority,
unsafe paths, symlinks and common secret/generated/binary exclusions. A fixture
also checks that Git fsmonitor code is not executed during inventory.

The release integration creates a private Git worktree and exercises actual
CLI output, exact tokenizer budgets, deterministic repetition, exclusions,
untracked opt-in, output files, answer reserves and model-context errors.
Tokenizer-only packing uses a derived **untrained synthetic fixture** with
larger metadata context; this is not evidence of Qwen long-context quality.

The CPU HTTP test uses a 512-token synthetic model and a 64-token output cap.
Its hello request rendered to 30 prompt tokens; generation reported exactly
the same count. It verifies that:

- Preview requires the API key and returns without changing request totals or expert counters.
- An explicit output request above the cap is rejected before normal or streaming generation.
- Prompt-plus-output overflow returns an actionable budget error rather than truncation.
- A reload exceeding model training context leaves the active model/context unchanged.
- Valid generation after those failures still returns the expected `AAAA` output.

Records: [rust-tests-0.6.0.json](../rust-tests-0.6.0.json),
[code-context-integration-0.6.0.json](../code-context-integration-0.6.0.json), and
`scripts/validate-code-context.py`.

The unit command was:

```bash
cargo test --locked --workspace --lib --bins --offline -- --nocapture
```

Some inherited CUDA availability tests return early when CUDA/NVRTC is absent.
No CUDA or Vulkan model execution is inferred from the unit-test total.

## Conversation-cache correctness

Pure version-2 persistence tests check exact payload/owner round-trip, old
versions, incompatible identity, checksum failure, truncation, trailing bytes,
excessive counts/lengths, pre-allocation budgets, exact buffer geometry, token
ranges and LoRA matrix values/scales. They do not restore state onto a real GPU.

Identity combines every checkpoint/auxiliary shard's existing metadata identity
with model configuration, actual adapters and runtime/precision layout.
Checkpoint identity is **not a fresh full-weight checksum**. Changing weights
while preserving canonical path, length and mtime requires clearing caches.
Invalid persistence is discarded rather than blocking model startup.

Context tests cover whole rendered requests, tools, reasoning settings and
open assistant continuation. Literal continuation whitespace is retained.
Generation uses the same token IDs that passed the request-budget check.

## Release and regression checks

```bash
cargo build --release --locked --offline
python3 scripts/validate-code-context.py --exe target/release/forge9070
python3 scripts/validate-expert-ram.py --exe target/release/forge9070
python3 tests/smoke.py --binary target/release/forge9070
node scripts/validate-code-ui-helpers.cjs
```

Expert RAM regression repeats mapped/warmed/on-demand CPU parity, bounded
warmup, cache counters, independent model switching and unloading. Smoke tests
repeat API-key enforcement, ordinary/SSE generation, token counts, finish
reasons and invalid sampling rejection. The completion-cap expectation now
requires an explicit error rather than silent clamping.

Records: [build-release-0.6.0.json](../build-release-0.6.0.json),
[expert-ram-integration-0.6.0.json](../expert-ram-integration-0.6.0.json), and [smoke-0.6.0.json](../smoke-0.6.0.json).

## UI logic and browser limit

The Node VM harness extracts actual helper blocks from the application HTML,
records its SHA-256, and checks reasoning/whitespace preservation, initial
server defaults, saved user overrides, context/output bounds, immutable-cap
refusal, CPU/training-context guards and explicit preset staging. **These are
helper unit checks with minimal stubs, not DOM, layout or browser-event tests.**

Records: `scripts/validate-code-ui-helpers.cjs`,
[code-ui-helpers-0.6.0.json](../code-ui-helpers-0.6.0.json), and [code-ui-syntax-0.6.0.json](../code-ui-syntax-0.6.0.json).

The full browser harness starts an isolated CPU server and would exercise
real controls and captured HTTP requests. Chromium aborted at process-singleton
socket creation with Operation not permitted before creating a page. Five
setup/cleanup records succeeded; **zero browser assertions ran**. No screenshot
was produced. No runtime policy was changed, no browser-launch workaround was
attempted after the block, and no additional DOM dependency was installed.

`scripts/validate-code-browser.py` remains available for a normal local
Chromium environment. [code-browser-integration-0.6.0.json](../code-browser-integration-0.6.0.json) preserves
the blocked run and must not be described as a browser pass.

## Memory estimates and practical limits

[context-memory-estimates-0.6.0.json](../context-memory-estimates-0.6.0.json) records source-derived Qwen3.8
F16 geometry. At 16K/32K, attention K/V payload is 384/768 MiB and a complete
parked host prefix is approximately 521/929 MiB. These are not total VRAM/RSS
figures or measured allocations. A 1 GiB coding cache budget can fit one such
32K snapshot by payload size, not unlimited long conversations.

Presets are chosen starting settings. No real-hardware throughput optimum was
measured. Context, prefill buffers, dense weights, driver reserve and expert
caches compete for memory. Startup estimates and ceilings are not OS reservations.

Full Qwen checkpoints were not downloaded, loaded or decoded for this release.
Version 0.5's pinned original/uncensored complete-header checks remain historical
evidence in [qwen38-header-checks/](../qwen38-header-checks). They were not rerun or presented
as 0.6 full-inference validation. Windows build/execution, native WSL Vulkan,
RX 9070 XT allocation/cache restoration, numerical behavior and token rates
still require that environment.

Qwen3.8 continues to require one generation slot and MTP disabled. This Vulkan
release does not add HIP, Avalon's packed IQ3_K/IQ3_KS kernels, elastic VRAM
lending or simultaneous generation for this architecture.

Previous top-level validation records are retained in [v0.5.0/](../v0.5.0) and
[v0.4.1/](../v0.4.1). Earlier versioned files describe their own runs.

