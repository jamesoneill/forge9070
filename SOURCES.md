# Primary sources

Reviewed 16 November 2026. The source snapshot used by this project is pinned in
`UPSTREAM.json`; external documentation may evolve.

- RustingStrata engine and primary README:
  https://github.com/RustingStudio/RustingStrata
- Pinned engine source:
  https://github.com/RustingStudio/RustingStrata/tree/f821c4435ea553eb8a49f542313da52caab75b90
- RX 9070 XT and AMD-specific Vulkan notes:
  https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/docs/AMD.md
- Reported GPU configurations:
  https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/docs/SUPPORTED_GPUS.md
- Engine distribution license:
  https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/LICENSE.md
- AMD ROCm 6.3.4 WSL compatibility matrix (RX 9070 XT absent):
  https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.3.4/docs/compatibility/wsl/wsl_compatibility.html
- AMD ROCDXG compatibility table, including RX 9070 XT with ROCm 7.2.x and bridge 1.2.0:
  https://github.com/ROCm/librocdxg/blob/develop/README.md
  The repository now points development to ROCm/rocm-systems; the historical table
  is cited for this specific version pairing, not as a claim about the latest SDK.
- AMD RX 9070 XT architecture identification:
  https://rocm.docs.amd.com/projects/install-on-windows/en/docs-7.2/reference/system-requirements.html
- Microsoft WSL networking (mirrored localhost and default NAT differences):
  https://learn.microsoft.com/en-us/windows/wsl/networking
- Rust installation and Windows build tool requirements:
  https://www.rust-lang.org/tools/install
  https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Avalon architecture/version checks and custom format context:
  https://github.com/freriadoc/Avalon

No benchmark from another card or checkpoint is presented as a measurement of
this build on the target machine. Defaults are engineering choices for initial
validation. See the measured local checks in `VALIDATION.md`.

## Version 0.2 model management and WSL

- Hugging Face revision-aware downloads and cache concepts:
  https://huggingface.co/docs/huggingface_hub/guides/download
  https://huggingface.co/docs/huggingface_hub/package_reference/file_download
- Example community GGUF repository, publisher-reported `qwen3` architecture
  and Q4_K_M size approximately 5.03 GB:
  https://huggingface.co/DevQuasar/huihui-ai.Qwen3-8B-abliterated-GGUF
  Read-only Hub metadata on 16 November 2026 identified
  `huihui-ai.Qwen3-8B-abliterated.Q4_K_M.gguf`, 5,027,779,840 bytes, at commit
  `8585a13b1f8798bffa863c2db012f48dd6be7b3b`.
  This identifies an example download; its inference quality/speed has not been
  measured with this build. It is a dense model and is excluded by version 0.3's
  sparse-MoE selection policy; the reference is retained as version 0.2 history.
- Microsoft WSL maintainer guidance on the experimental Dozen (`dzn`) Vulkan
  route, including the need for an appropriate Mesa build:
  https://github.com/microsoft/WSL/issues/41410
- Microsoft WSL file placement and executable interoperability:
  https://learn.microsoft.com/en-us/windows/wsl/filesystems

## Version 0.3 Avalon source audit

Avalon was inspected at commit
`762a3dae8d78ff18acb38e351dfa940328485fd8`, dated 16 November 2026. The source
audit compares implementations rather than treating project names or benchmark
headlines as evidence of feature parity:

- Pinned repository and published benchmark configuration:
  <https://github.com/freriadoc/Avalon/tree/762a3dae8d78ff18acb38e351dfa940328485fd8>
  <https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/README.md>
- Expert cache, CPU/GPU dispatch, memory lending and `moe_hosted_tail_skip`:
  <https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/src/ggml-cuda/moe-cache.cu>
- Owned expert RAM arena and memory-budget calculation:
  <https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/src/ggml-cuda/moe-arena.h>
  <https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/src/ggml-cuda/moe-ram-budget.h>
- Custom quantization IDs, including IQ3_K = 43 and IQ3_KS = 44:
  <https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/ggml/include/ggml.h>
- Avalon MIT licence:
  <https://github.com/freriadoc/Avalon/blob/762a3dae8d78ff18acb38e351dfa940328485fd8/LICENSE>

The relevant pinned RustingStrata implementations are its
[expert cache](https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/src/expert_cache.rs),
[CPU worker](https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/src/cpu_worker.rs),
[GPU model](https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/src/gpu_model.rs),
[prefill](https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/src/gpu_model/prefill.rs)
and [Vulkan backend](https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/src/vulkan/mod.rs).
Avalon's README comparison with [Niko1221/Strata](https://github.com/Niko1221/Strata)
refers to a different project from RustingStudio/RustingStrata.

See [docs/AVALON-COMPARISON.md](docs/AVALON-COMPARISON.md) for the mechanism-by-
mechanism findings. Forge9070 v0.3's optional tail policy is a fresh Rust translation;
the Vulkan integration does not import Avalon's HIP kernels or establish the
same throughput. The other differences and target-hardware validation limits
are stated in that document.

## Version 0.3 sparse-MoE examples and metadata evidence

The following publisher repositories are examples whose actual GGUF headers
were checked by the application's policy on 16 November 2026. They are not
blanket endorsements of every file in each repository. Full checkpoints were
not downloaded for these checks, and no inference or speed measurement of these
checkpoints was performed on the target GPU.

| Publisher snapshot | File checked | Captured metadata result |
| --- | --- | --- |
| [mradermacher/Qwen3-30B-A3B-abliterated-GGUF, commit 5eed9fdf](https://huggingface.co/mradermacher/Qwen3-30B-A3B-abliterated-GGUF/tree/5eed9fdfc8dd2a614fa0f344befbb68d9e049978) | `Qwen3-30B-A3B-abliterated.Q3_K_M.gguf`, 14,711,846,880 bytes | `qwen3moe`; 48 layers; 128 experts per MoE layer, 8 active; accepted |
| [unsloth/Qwen3.5-35B-A3B-GGUF, commit bc014a17](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/tree/bc014a17be43adabd7066b7a86075ff935c6a4e2) | `Qwen3.5-35B-A3B-Q3_K_S.gguf`, 15,265,856,128 bytes | `qwen35moe`; 40 layers; 256 experts per MoE layer, 8 active; accepted |

The historical v0.3 evidence is in [v0.3.0/compat-hf-live-abliterated.json](docs/validation/v0.3.0/compat-hf-live-abliterated.json) and
[v0.3.0/compat-hf-live-hybrid.json](docs/validation/v0.3.0/compat-hf-live-hybrid.json). Those files include the immutable commit,
full-file size, tensor-type histogram and verification caveat. Some other
choices exhausted the bounded verification time; that is recorded as
unverified, not proof that those weights are inherently unsupported. The hybrid
repository's MXFP4 choice was explicitly rejected because its type 39 tensors
are unsupported by this engine.

The examples illustrate why filenames alone cannot define compatibility.
Quantization labels can describe mixed tensor types, while metadata acceptance
still leaves RAM/VRAM allocation, numerical correctness, quality and throughput
to actual loading and measurement. The accepted metadata does not imply that
the whole checkpoint plus context and scratch space fits in 16 GiB of VRAM.

Publisher repositories and model cards are primary sources for their releases;
the included metadata reports are the evidence for this program's acceptance
decision. [VALIDATION.md](VALIDATION.md) distinguishes these read-only checks
from synthetic inference tests and from GPU/WSL execution that remains untested.

## Version 0.4 computation fidelity audit

The audit used the same pinned RustingStrata source plus the included Forge9070
modifications. Relevant primary implementation and test references are:

- Target-sampler candidate selection and deterministic-draft accept/reject:
  <https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/src/ops.rs>
- Greedy/verification paths, positions, expert placement and speculative state:
  <https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/src/gpu_model.rs>
- Upstream MTP/verification tests and their fixed-placement/tolerance scope:
  <https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/tests/gpu_model.rs>
- Finite-sampler distribution tests:
  <https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/tests/ops.rs>
- KV precision, reference-numerics controls and original calibration method:
  <https://github.com/RustingStudio/RustingStrata/blob/f821c4435ea553eb8a49f542313da52caab75b90/src/cli.rs>

Version 0.4's greedy-position and CPU sampler corrections are changes to those
sources, not claims made by the upstream project. The included modified files,
tests and validation record establish what changed locally. No upstream
benchmark is presented as an RX 9070 XT result for this distribution.
