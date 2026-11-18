# Running Forge9070 from WSL2

**Scope:** this guide documents the existing compatibility application. For the
independent Forge 0.9 runtime, its Vulkan/CPU settings, coding API and queue,
use [NATIVE-ENGINE.md](NATIVE-ENGINE.md). Flags and endpoints differ.

Forge9070 now has a native Linux build and launch workflow for WSL. The model
catalogue, Hugging Face downloads, browser interface and model switching use the
same application in Linux and Windows. Select the compute route explicitly:

| Route | Where inference executes | GPU requirement |
| --- | --- | --- |
| Native WSL CPU | Linux process inside WSL | None; choose `--backend cpu` |
| Native WSL Vulkan | Linux process inside WSL | Real RX 9070 XT Vulkan device passing the engine's capability checks |
| WSL-controlled Windows GPU | Windows process launched from the WSL shell | Windows RX 9070 XT Vulkan driver; native WSL GPU access is not used |

The Linux build and CPU execution can be tested without the card. **Native WSL
Vulkan execution on the target RX 9070 XT has not been verified.** The Windows-host
launcher is provided as an explicit alternative if the Linux GPU check fails.
It does not disguise a Windows process as native WSL computation.

## 1. Build the native Linux application

Extract or copy the project into the WSL filesystem, for example
`~/code/forge9070`. Microsoft recommends keeping Linux build files in the Linux
filesystem for performance. Open the WSL shell in that directory:

```bash
bash scripts/setup-wsl.sh --check
```

The check reads the environment and looks for Cargo and C build tools. If
ordinary Ubuntu build dependencies are missing, explicitly install them with:

```bash
bash scripts/setup-wsl.sh --install-deps
```

This installs `build-essential`, `pkg-config`, `ca-certificates`, `libvulkan1`
and `vulkan-tools`. It does not install a Linux GPU kernel module, a replacement
Mesa stack, ROCm, or DKMS. It also does not change WSL configuration. The Vulkan
loader and `vulkaninfo` by themselves do not supply hardware compute support.

If the earlier `amdgpu-dkms` error left packages unconfigured, the optional
dependency installer stops and displays `dpkg --audit`. That package-manager
state needs separate repair; the script does not repeatedly invoke its failed
post-install step or remove packages automatically.

If Rust is missing, install the Linux toolchain from <https://rustup.rs/> and
open a new WSL shell. The project uses its pinned toolchain. Then build:

```bash
bash scripts/setup-wsl.sh --build
```

The executable is `target/release/forge9070`. The script prints the hardware
doctor report after compilation. To build a development executable, add
`--debug` and pass `--bin target/debug/forge9070` to the launchers below.

## 2. Native WSL CPU mode

This runs the full application as a Linux process. Start with an empty catalogue
and choose/download models through the browser:

```bash
bash scripts/serve-wsl.sh --backend cpu
```

Set `FORGE9070_MODELS_DIR` to use a shared default catalogue for the model CLI
and native WSL launcher. An explicit `--models-dir` takes precedence. Without
that environment override, the default catalogue is:

```text
${XDG_DATA_HOME:-$HOME/.local/share}/forge9070/models
```

Open <http://localhost:8080> in a Windows browser. Microsoft documents Windows
access to WSL-hosted services through localhost in the usual WSL networking
configuration.

To choose another directory or load a model initially:

```bash
bash scripts/serve-wsl.sh --backend cpu \
  --models-dir "$HOME/models" \
  --model "$HOME/models/your-compatible-model.gguf"
```

Additional engine arguments follow `--`:

```bash
bash scripts/serve-wsl.sh --backend cpu --models-dir "$HOME/models" \
  -- --ctx 4096 --threads 8
```

CPU mode is useful for development, small models and reference checks. Large
MoE models can be slow and still need substantial RAM. WSL's VM memory limit
can also be lower than the machine's installed RAM; the scripts do not increase
that limit automatically.

## 3. Native WSL Vulkan mode

Run the actual discovery check inside WSL:

```bash
./target/release/forge9070 doctor --json
./target/release/forge9070 doctor --require-gpu
```

If a real RX 9070 XT exposes the necessary Vulkan 1.3 features, start:

```bash
bash scripts/serve-wsl.sh --backend vulkan --models-dir "$HOME/models"
```

The launcher runs `doctor --require-gpu` first and stops on failure. It never
falls back to a CPU/software Vulkan adapter or selects CPU inference silently.
Discovery is only a prerequisite check; short actual generation is still
needed to validate the driver/model combination.

### Why GPU discovery can fail even though `/dev/dxg` exists

WSL exposes Windows GPU access through a different interface from native Linux.
The presence of `/dev/dxg`, working OpenGL, or a working HIP application does
not establish the Vulkan features that RustingStrata requires. Microsoft WSL
maintainer guidance identifies Mesa's Dozen (`dzn`) Vulkan-on-D3D12 driver as
the WSL Vulkan path and describes it as experimental. A distribution's Mesa
package may not include it; installing the generic Vulkan loader does not
change that.

The engine additionally needs subgroup and arithmetic capabilities and, for
hybrid expert workloads on AMD, suitable coherent-memory behaviour. An adapter
appearing in `vulkaninfo` is therefore insufficient evidence that the complete
engine works. This package does not replace the distribution's Mesa stack,
change ICD files, or bypass the engine's feature checks.

ROCm 6.3.4 does not officially support this RX 9070 XT WSL combination. AMD's
ROCDXG table includes the card with newer ROCm versions, including the
7.2.x/ROCDXG 1.2.0 pairing. **Forge9070's inference backend is Vulkan; updating
ROCm does not add a HIP backend to this build.**

## 4. Control the Windows GPU process from WSL

If native WSL Vulkan is unavailable, build the Windows executable as described
in the main README. Keep Windows-served weights on a Windows drive where
practical. For example, use `D:\Models` (seen as `/mnt/d/Models` in WSL):

```bash
bash scripts/serve-wsl-windows.sh \
  --exe '/mnt/c/code/forge9070/target/release/forge9070.exe' \
  --models-dir '/mnt/d/Models'
```

An initial model is optional:

```bash
bash scripts/serve-wsl-windows.sh \
  --exe '/mnt/c/code/forge9070/target/release/forge9070.exe' \
  --models-dir '/mnt/d/Models' \
  --model '/mnt/d/Models/your-compatible-model.gguf' \
  -- --ctx 4096
```

The launcher converts paths with `wslpath -w`, checks the Windows GPU, then
executes `forge9070.exe` directly. It never constructs a PowerShell or cmd
command string from model names, so spaces and punctuation remain literal
argument data. The process stays attached to this terminal and can be stopped
with Ctrl+C.

For gated/private Hugging Face models, export `HF_TOKEN` in the WSL shell. If
`HF_TOKEN` or `STRATA_API_KEY` is set, the launcher forwards those values through
`WSLENV` without printing them. It preserves unrelated `WSLENV` entries.

This route runs the model on **Windows**. The terminal, development tools and
clients may remain in WSL. Using Windows weights through a Linux share is
possible, but reading across the Windows/WSL filesystem boundary can be slower;
the documented Windows-drive layout avoids that for the inference process.

### Networking from a WSL client

With mirrored networking enabled, Windows and WSL can connect to each other's
services through `127.0.0.1`:

```bash
curl http://127.0.0.1:8080/v1/models
```

Under default NAT networking, WSL-to-Windows access normally uses the Windows
host IP, and the Windows server must bind an appropriate interface. That also
requires the relevant firewall configuration and an API key when binding
outside loopback. See Microsoft's networking guide. These launchers leave
existing networking settings and firewall rules unchanged.

## Compatible MoE models and multiple requests

Version 0.4 offers only sparse-MoE GGUFs passing the shared architecture,
expert-geometry, tensor and quantization checks. The browser, local catalogue,
download and load commands apply the same policy, including complete shard
groups. Community abliterated/uncensored MoE variants are eligible under these
checks. The earlier dense Qwen3-8B example and Avalon's custom IQ3_K/IQ3_KS
files are excluded. A metadata pass is not a promise of memory fit or speed;
remote files that cannot be verified are not offered as compatible choices.

After the relevant GPU route passes its capability check, two clients can use
one supported model concurrently:

```bash
# Native WSL process, conditional on its actual Vulkan capabilities.
bash scripts/serve-wsl.sh --backend vulkan \
  --model "$HOME/models/your-compatible-moe.gguf" \
  -- --parallel 2 --mtp 0
```

The Windows launcher also forwards `-- --parallel 2 --mtp 0` to the Windows
process. GPU mode accepts 1–4 active slots with shared weights and separate
per-request state. CPU mode queues generation one request at a time.
Hyper-connection/PLE/QSA models currently require one slot, and MTP cannot run
with multiple slots. More slots leave less VRAM for expert weights; measure
aggregate throughput and each client's latency separately.

For the fastest individual-reply baseline, keep `--parallel 1 --mtp 0 --kv f16`.
Use `forge9070 tune --model /path/model.gguf` on a working GPU to compare
generation settings and tested output fidelity. Every selected expert is
retained; the current Forge profile rejects nonzero expert-tail omission.
See [PERFORMANCE.md](PERFORMANCE.md) for
measurement and [AVALON-COMPARISON.md](AVALON-COMPARISON.md) for the source audit.

All these GPU routes use **Vulkan**, not HIP/ROCm. This remains a RustingStrata
integration with selected Avalon-derived techniques, not a complete Avalon
port. Neither native WSL Vulkan nor Windows GPU inference on the target RX 9070 XT
has been validated in this workspace.

## Checks included in this package

```bash
python3 scripts/test-wsl.py
```

The tests use fake executables to verify path/argument preservation, explicit
CPU selection, GPU preflight ordering, stopping after GPU failure, port
validation, Windows path conversion, and environment-variable forwarding. They
do not establish real Windows interoperability or GPU execution. Native Linux
inference and model-switching checks are recorded separately in `VALIDATION.md`.

## Primary references

- Microsoft, filesystem performance and executable interoperability:
  <https://learn.microsoft.com/en-us/windows/wsl/filesystems>
- Microsoft, WSL networking and mirrored versus NAT localhost behaviour:
  <https://learn.microsoft.com/en-us/windows/wsl/networking>
- Microsoft WSL issue 41410, maintainer guidance on experimental Dozen Vulkan:
  <https://github.com/microsoft/WSL/issues/41410>
- AMD, ROCDXG runtime compatibility table:
  <https://github.com/ROCm/librocdxg>
- AMD, ROCm 6.3.4 WSL compatibility matrix:
  <https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.3.4/docs/compatibility/wsl/wsl_compatibility.html>
- Pinned engine requirements and AMD caveats:
  `vendor/rusting-strata/docs/AMD.md`.

Reviewed 16 November 2026. No WSL or GPU capability is inferred solely from an
installed package version.
