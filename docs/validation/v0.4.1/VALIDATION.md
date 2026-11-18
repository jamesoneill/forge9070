> Historical snapshot for v0.4.1. Published with path redaction and link repairs; the recorded test outcomes and limitations are retained. See [the evidence policy](../README.md).

# Forge9070 0.4.1 validation record

Date: 16 November 2026. These results describe the source preparation
workspace, not Jamie's PC. This patch fixes GGUF discovery and split-file
handling; it does not establish a GPU benchmark.

## Environment

Linux x86_64, kernel `6.18.57`, Rust `1.100.0`. This is not WSL. No usable
RX 9070 XT or Windows graphics environment is available. Some upstream tests
skip GPU work when a suitable device is absent; the passing test total must
not be interpreted as GPU execution.

## Completed checks on the final source

| Check | Observed result |
| --- | --- |
| Development build | Passed, 9.96 seconds |
| Release build, `cargo build --locked --release` | Passed, 133.84 seconds |
| Engine library unit tests | 157 passed, 0 failed |
| Forge frontend unit tests | 22 passed, 0 failed |
| Release command/inspection checks | 4 passed, 0 failed |
| Original Unsloth UD-IQ4_XS metadata | All four shards passed; 54.02 seconds |
| Uncensored ToR-O3 UD-IQ4_XS metadata | All four shards passed; 46.73 seconds |

**179 Rust unit tests passed.** The commands were:

```bash
cargo test --locked --offline -p rusting_strata --lib
cargo test --locked -p forge9070 --bin forge9070
```

Seven added regressions cover exact-file HF selection, immutable commit use,
actual-header enforcement despite a Transformers architecture hint, and
metadata-only first shards. The split fixture retains real tiny-model tensor
payloads and compares every mapped tensor byte with the unsplit model. Missing
shards, empty combined models, wrong counts, malformed alignment, unexpected
trailing bytes and truncated tensor data remain rejected by Forge preflight.
The low-level mapper still permits generic empty auxiliary maps; these are not
accepted as complete Forge inference models.

Release checks verified version 0.4.1, the new `models files --file` interface,
unsafe-path rejection before a network call, and inspection of the tiny MoE
fixture. No inference is implied by those four checks.

## Actual remote model metadata

The live checks used the final development executable and immutable revisions.
No complete model payload was downloaded or evaluated.

| Repository | Revision | Observed structure |
| --- | --- | --- |
| `unsloth/Qwen3.8-Flash-Next-GGUF` | `f5d175de2cf7f720d45d6064019a81fcf056e986` | 4 shards, 1,224 tensors, 93,672,359,738 file bytes |
| `ToR-O3/Qwen3.8-Flash-Next-Uncensored-GGUF` | `afa27ea7af1f83171a0aeafa8d5918eaa0dbfeee` | 4 shards, 1,224 tensors, 93,672,359,738 file bytes |

Each choice is
`UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf` in its own
repository. Both headers describe `qwen4exp`, 48 layers, 512 experts with 10
routed experts active per token, and no unsupported tensor type IDs.

The original first shard has zero tensors and 10,946,618 bytes of metadata
ending exactly at EOF. The former parser rejected it immediately. The patched
parser accepts that first file only provisionally and then verifies the
complete, nonempty model across the remaining shards. The initial failed
probe is retained separately from the successful final probe.

The selected UD-IQ4_XS files use mixed formats: 64 routed-expert tensors use
IQ2_S, IQ3_XXS or Q2_0. The name does not mean uniform four-bit weights. These
format observations do not establish the factual quality of either model.

A separate bounded audit read the PLE and QSA metadata arrays. Unsloth and
ToR-O3 have the expected `[0,0,0,4]` compression pattern and matching PLE array
geometry. The alternate mradermacher IQ4_XS header passes implemented Forge
checks but has a stale all-zero compression array. The pinned engine ignores
that array and fixes QSA pooling at four; this is a portability discrepancy,
not proof of incorrect Forge execution. That alternative is not the primary
paired example in the guide.

The ToR-O3 publisher describes a 146-tensor main-model transplant into the
Unsloth quantization. It is not a full reproduction of OrcaRouter's checkpoint
or MTP changes. The publisher's limited evaluation and original model's
benchmarks do not certify this derivative's general accuracy.

## Records

- [validation.json](validation.json): current aggregate record.
- [rust-tests-0.4.1.json](../rust-tests-0.4.1.json): commands, totals and test scope.
- [build-release-0.4.1.json](../build-release-0.4.1.json): final release build result.
- [release-cli-0.4.1.json](../release-cli-0.4.1.json): actual release command outputs.
- [qwen38-header-summary.json](../qwen38-header-summary.json): paired live metadata results.
- [qwen38-header-checks/](../qwen38-header-checks): complete live probe reports and initial failure.
- [qwen38-source-metadata.json](../qwen38-source-metadata.json): publisher metadata, sizes,
  provenance, additional bounded array inspection and source caveats.
- `docs/QWEN3.8.md`: Windows inspection, download and switching commands.

## What remains unverified

Neither full Qwen3.8 model has been loaded or used for inference here. Host RAM
working-set fit, GPU allocations, Windows AMD mailbox behavior, answer quality,
numerical correctness and token rates need execution on the target machine.
For this family the current runtime requires one generation slot, MTP off and
text input. Multiple clients can queue requests; this is not concurrent
multi-sequence generation.

The engine uses Vulkan. Installing a newer ROCm package does not change its
backend or make the previously observed WSL software adapter a hardware GPU.
The native Windows Vulkan route remains a separate hardware validation step.

## Historical release checks

The earlier 0.4.0 validation and aggregate record are preserved in
[v0.4.0/](../v0.4.0). Earlier 0.1–0.3 records remain in their existing folders.
Root-level reports without a 0.4.1 suffix may belong to the preceding release.
Its 216-test full Rust run and 31 release executable/HTTP checks were not
rerun for this focused patch. Do not report those historical totals as new
0.4.1 results.

Uses RustingStrata by Vasyl Trefilov.
