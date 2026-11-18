> Historical snapshot for v0.8.0. Published with path redaction and link repairs; the recorded test outcomes and limitations are retained. See [the evidence policy](../README.md).

# Forge9070 0.8.0 validation record

## Release boundary

This release adds the independently implemented `forge-engine` 0.1.0 package.
It executes ordinary `qwen3` and `qwen3moe` on CPU, using single-file GGUF and
F32/F16/BF16/Q4_0/Q8_0 weights. Its standalone runtime has no RustingStrata
dependency. `forge9070 engine` exposes the same core; existing application,
GPU and HTTP commands still use the attributed RustingStrata compatibility
engine. **The requested RX 9070 XT / Qwen3.8 production replacement is not
complete.**

Native Vulkan, Qwen3.8/qwen4exp HC/PLE/QSA and recurrent semantics, sharded
loading and a native HTTP adapter are unimplemented. Explicit `--cpu` is
required for native generation/benchmarking and unsupported models are
rejected without fallback. No real-model token rate is claimed.

## Environment and build

- Linux x86_64, AMD EPYC 9V74 80-Core Processor, nine available logical CPUs.
- Rust rustc 1.100.0 (0ef1bc386 2026-11-09); release optimization level 3, thin LTO, one codegen unit.
- AVX2, FMA and F16C detected and exercised on this host.
- Optimized `forge-engine` and `forge9070` builds passed in 53.04 seconds.
- No Windows build, WSL Vulkan, AMD GPU or CUDA compute was validated here.

```sh
cargo build --release --locked --offline -p forge9070 -p forge-engine --bins
```

See [build-release-0.8.0.json](../build-release-0.8.0.json) and [native-host-0.8.0.json](../native-host-0.8.0.json).
The Cargo warning about ignored profile settings in the vendored non-root
package is inherited; the workspace root profile controls this build.

## Executed Rust checks

```sh
cargo test --locked --workspace --lib --bins --test inference --test new_engine_parity --test serve --offline
```

| Suite | Harness-reported passed |
|---|---:|
| Independent engine unit tests | 36 |
| Independent/reference numerical integration tests | 8 |
| Forge frontend unit tests | 38 |
| Existing inference integration tests | 5 |
| Compatibility engine unit tests | 217 |
| Serving integration tests | 28 |
| Total | 332 |

Zero failures. One serving GPU smoke test returns early without suitable
hardware: 27 serving tests were exercised. Inherited CUDA/NVRTC availability
checks also return early when their runtime is absent. The harness totals
must not be presented as that many hardware executions.

The eight parity tests feed identical token IDs into the independent and
reference engines and compare **every logit, final normalized hidden value
and attention K/V value**. They cover dense/MoE, all five formats, grouped
query attention with four query heads/two KV heads, independent histories,
reset, invalid tokens, context and model ownership, and explicit Qwen3.8
rejection. Output projections vary in sign and magnitude so a favored fixture
token cannot conceal an incorrect forward pass. The numerical gate is:

```text
abs(new - reference) <= 2e-5 + 2e-5 * abs(reference)
```

The fixtures are tiny untrained models. These checks establish implementation
agreement for the covered operations, not pretrained answer quality, Qwen3.8
correctness or long-context GPU accuracy. SIMD tests separately cover multiple
storage blocks, unaligned reads, odd float tails, exact lengths and non-finite
values. F16 tests include normal/subnormal values, signed zero, maximum finite
half values, and NaN/infinity in both vector and scalar-tail positions.

Full command/output: [native-core-tests-0.8.0.json](../native-core-tests-0.8.0.json).

## Standalone and frontend CLI checks

```sh
python3 scripts/validate-forge-engine.py --binary target/release/forge-engine --frontend target/release/forge9070 --output artifacts/forge-engine-cli-0.8.0.json
```

All **16 checks passed**, none skipped, against the final optimized binaries.
They execute actual CPU model loading and generation; check Unicode/stdin,
explicit CPU selection, unsupported/truncated models and context reservations;
execute four independent CPU sequences within two workers; validate timing,
checksums and saved JSON; preserve the source model when report destinations
alias it directly or through a hard link; and exercise `forge9070 engine`.
Four CPU benchmark states do not establish four concurrent Qwen3.8 GPU requests.

## Measured CPU kernels

```sh
target/release/forge-engine kernel-bench --width 4096 --rows 256 --iterations 64 --reps 7 --json-out artifacts/native-kernels-0.8.0.json
```

The benchmark compares the **new engine's selected dot kernel against its own
portable scalar reference** on the host above. Each sample executes 256 rows
of width 4,096 for 64 iterations. Seven paired samples follow a warmup;
execution order alternates. Both paths include public row validation. Rows are
processed serially with warm synthetic weights; attention, model orchestration,
Rayon scheduling, storage stalls and GPU execution are outside this measurement.

| Format | Selected kernel | Scalar median | Selected median | Scalar time / selected time |
|---|---|---:|---:|---:|
| F32 | `avx2_fma` | 87.046 ms | 49.233 ms | 1.77x |
| F16 | `avx2_fma_f16c` | 213.768 ms | 47.740 ms | 4.48x |
| BF16 | `avx2_fma` | 81.902 ms | 47.297 ms | 1.73x |
| Q4_0 | `avx2_fma` | 93.136 ms | 48.871 ms | 1.91x |
| Q8_0 | `avx2_fma` | 99.390 ms | 52.679 ms | 1.89x |

All per-row correctness checks passed. Maximum scaled error across formats was
4.9471855e-05, using
`abs(selected - scalar) / max(abs(scalar), 1)`, below the 0.0002 kernel gate.
FMA/reduction order can change rounding; these paths are not claimed bit-identical.

These are **CPU row-kernel ratios**, not whole-model token speedups, a comparison
with Strata/Avalon, or an RX 9070 XT result. Raw samples and exact errors are in
[native-kernels-0.8.0.json](../native-kernels-0.8.0.json); execution receipt is in
[native-kernels-run-0.8.0.json](../native-kernels-run-0.8.0.json). Earlier pre-F16C measurements are
retained under the `before-f16c` filenames and are not the final release result.

Separate one/four-request benchmark records in
[native-1-requests-0.8.0.json](../native-1-requests-0.8.0.json) and
[native-4-requests-0.8.0.json](../native-4-requests-0.8.0.json) demonstrate phase timing and repeatable
CPU states using the tiny MoE fixture. They predate the final F16C change and
use F32 small rows. Their synthetic token rates are not useful-model estimates.

## Existing coding API and queue regression checks

Against the optimized 0.8 application binary, the existing CPU HTTP tests passed:

| Harness | Passed |
|---|---:|
| `tests/api-coding-client.py` | 25 |
| `tests/scheduling_api.py` | 10 |
| `tests/smoke.py` | 9 |

These checks cover actual loopback HTTP/SSE, model identity, API-key handling,
completion limits, generated synthetic tool calls and tool-result turns,
invalid-parameter rejection, FIFO admission, overload, timeout and cancellation.
They passed before the last metadata/F16 changes confined to the independent
core; the compatibility implementation is unchanged. This release did not
re-run every historical UI/context/model-switch harness. Earlier results remain
historical evidence rather than new 0.8 executions.

The raw 0.8 command receipts are [api-coding-client-0.8.0.json](../api-coding-client-0.8.0.json),
[scheduling-api-0.8.0.json](../scheduling-api-0.8.0.json) and [smoke-0.8.0.json](../smoke-0.8.0.json).
Fresh scheduling evidence is preserved as
[scheduling-api-0.8.0-evidence.json](../scheduling-api-0.8.0-evidence.json). That inherited harness hardcodes
a 0.7 label/output path; its raw metadata is retained rather than silently
rewritten. The tested binary was the 0.8 release build. Fixture tool selection
is scripted by its weights; no trained coding model or external coding client
was evaluated.

The existing Qwen3.8 path still allows **one active generation**, with waiting
requests handled by the existing queue. Native CPU benchmarks do not change
that production capability. Eligible other GPU architectures retain their
existing ceiling of four; no new GPU batching is claimed.

## Reproducibility and remaining acceptance work

- `artifacts/native-dependencies-0.8.0.txt` records the standalone package's
  runtime dependency tree, with no RustingStrata entry.
- [native-source-0.8.0.json](../native-source-0.8.0.json) pins SHA-256 values for the 15 relevant
  source/manifest/test files; they were checked unchanged after the final build
  and CLI/kernel runs.
- [validation.json](validation.json) is the machine-readable release summary.
- The previous release record is retained under [v0.7.0/](../v0.7.0).
- Target-card acceptance still requires real checkpoint numerical comparisons,
  useful code tasks, GPU prefill/decode, long-context memory, cancellation and
  one/two/four-request measurements after the native GPU/model paths exist.

See `docs/NATIVE-ENGINE.md` for the implemented design and remaining migration
work. Original third-party provenance/licenses remain in the source archive.
