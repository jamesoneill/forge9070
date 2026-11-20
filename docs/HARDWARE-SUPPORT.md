# Native hardware support

The native `nextora-engine` uses CPU execution and a shared Vulkan implementation.
GPU selection checks capabilities rather than a vendor or card-model name. This
lets eligible AMD, NVIDIA and Intel devices use the same execution path while
retaining explicit backend and adapter overrides.

This guide applies to `nextora-engine` and `nextora-cli`. The retained
`nextora-cli serve` and `nextora-cli doctor` commands use the RX 9070 XT compatibility
profile; their options and device indices are not interchangeable with native
commands.

## Platform and evidence matrix

| Environment | Native execution path | Validation boundary |
| --- | --- | --- |
| Windows / Linux, eligible AMD GPU | Vulkan | Physical GPU execution and target-card performance need hardware validation. |
| Windows / Linux, eligible NVIDIA GPU | Vulkan | No CUDA dependency; physical GPU execution and performance need hardware validation. |
| Windows / Linux, eligible Intel discrete or integrated GPU | Vulkan | Device limits and model fit still apply; physical GPU execution and performance need hardware validation. |
| Windows / Linux without eligible Vulkan hardware | CPU, explicitly or through `auto` | CPU/service tests are distinct from full pretrained-model acceptance. |
| Native Linux process in WSL | Vulkan if exposed and eligible; otherwise CPU with `auto` | `/dev/dxg` alone does not establish usable Vulkan hardware. |
| Windows executable launched from WSL | Windows Vulkan or CPU | Uses Windows drivers and Windows paths; the process remains on Windows. |
| Software Vulkan such as llvmpipe | Explicit diagnostic execution only | Numerical checks do not establish hardware acceleration or throughput. |
| macOS / Apple Silicon | Not a supported target in this change | A suitable GPU backend, platform memory handling and hardware validation remain work to do. |

CUDA, HIP/ROCm and native Metal are not implemented backends. Different GPU
vendors can produce different performance and small floating-point differences;
passing a capability check is not a benchmark or a guarantee that a model fits.
See the [validation record](../VALIDATION.md) for completed checks.

## Select an execution policy

```sh
nextora-engine doctor
nextora-engine serve --open-ui --backend auto --models-dir /path/to/models
```

On Windows use `nextora-engine.exe` and Windows model paths. The equivalent
frontend command is `nextora-cli doctor` or `nextora-cli serve ...`.

| Policy | Behaviour |
| --- | --- |
| `--backend auto` | Select eligible hardware Vulkan; use CPU if hardware discovery cannot supply an eligible adapter. Report the actual selection and reason. |
| `--backend vulkan` | Require native Vulkan execution; return an error when requirements cannot be met. |
| `--backend cpu` or `--cpu` | Run the CPU implementation. Vulkan-specific options are rejected. |
| `--backend hybrid` | Require the diagnostic CPU-state/Vulkan-projection path; it is not the default performance route. |

Generation and benchmarks require an explicit policy. A fresh service started
without a model or policy opens its configuration panel with `auto` selected.
An existing settings file keeps its saved policy and overrides model/runtime
CLI settings.

Automatic CPU selection is deliberately narrow:

- An explicit `--device`, nonzero `--vram-mib`, or
  `--allow-software-vulkan` requires strict device selection, including under
  `auto`; CPU execution must not hide an invalid or unmet GPU setting.
- A malformed or unsupported model remains an error.
- Device creation, shader/pipeline creation and allocation failures remain
  errors. They do not trigger a silent reload on CPU.
- Execution does not switch backends midway through a request.

The reported actual backend is the one used for memory planning and model
loading. The configured `auto` policy remains visible separately. The existing
Qwen3.8 policy may compute an individual expert on CPU when its packed working
set cannot fit the GPU expert budget; that counted operation is separate from
selecting the whole-model CPU backend.

## Vulkan eligibility and model fit

The native kernels require Vulkan 1.1 or later, a compute queue, 128-lane
workgroups, 512 bytes of shared memory, 64-byte push constants, at least seven
compute storage-buffer bindings, usable device-local memory and host-visible
staging memory. The adapter report exposes eligibility and rejection reasons.
Selection filters unsuitable adapters before ranking discrete, integrated,
virtual and other hardware; larger usable heap capacity then enumeration index
break ties. This is a selection heuristic, not a measured performance ranking.
Software adapters are excluded unless explicitly enabled for diagnostics.

Use `--device N` with the physical-device index from **native** `doctor` to
select a particular card. Enumeration order can change with the environment;
inspect the reported name when changing machines or drivers. Adapter choice
does not pool memory or split inference across multiple GPUs.

Successful discovery has not submitted a compute kernel. The selected model
must still satisfy tensor-buffer limits and fit its weights, sequence state,
staging and expert-cache working set. Automatic native allocation budgeting uses
70% of the largest device-local heap. This is an owned-allocation ceiling,
not a measurement of currently free VRAM. Integrated GPUs can share physical
memory with the CPU, so heap capacity must not be added to host RAM as if it
were separate memory. Use explicit budgets and leave room for other applications.

## Relevant specifications

- [Khronos: What is Vulkan?](https://docs.vulkan.org/guide/latest/what_is_vulkan.html)
  describes the cross-platform, cross-vendor compute API.
- [Khronos: querying properties, features and limits](https://docs.vulkan.org/guide/latest/querying_extensions_features.html)
  explains why applications must inspect each device's capabilities.
