> **Current Nextora note:** Use `nextora serve` on port 9070.
> Historical references to Strata/8080 and the legacy download workflow below
> do not describe the current native engine. Confirm each GGUF against
> `nextora capabilities` before loading.

# Qwen3.8-Flash-Next: original and uncensored models

Nextora can keep both versions in its model catalogue and switch between
them. There is one active model at a time. A model's original, uncensored or
abliterated label does not change the technical compatibility policy.

## Format and architecture

The official [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
repository contains Transformers/Safetensors weights. Forge's loader needs a
compatible **GGUF conversion**. Renaming a Safetensors file to GGUF does not
convert it.

The engine implements the GGUF `qwen4exp` architecture, including
hyper-connections, per-layer n-gram embeddings (PLE) and Qwen Sparse Attention
(QSA). Transformers calls the family `qwen4_exp`. Version 0.4.1 recognizes that
spelling as a search hint; it still checks the actual GGUF before acceptance.

Architecture support does not certify an arbitrary conversion. Every shard,
tensor type and required tensor must pass the metadata checks. Allocation,
full loading, numerical behavior and useful speed need a real execution test.
The Forge frontend supports **text** for this family; do not assume that the
base model's image capability is exposed here.

## Public conversions for a roughly 62 GB host

These are publication details checked on 16 November 2026. GB means decimal
bytes, while GiB is bytes divided by 1,073,741,824. Sizes include every shard.

| Version | Repository | Choice | File bytes | Size |
| --- | --- | --- | ---: | ---: |
| Original, smaller expert working set | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | GSQ-RCO IQ3_S | 83,617,662,656 | 83.62 GB / 77.88 GiB |
| Original | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | UD-IQ4_XS | 93,672,359,738 | 93.67 GB / 87.24 GiB |
| Uncensored community derivative | [ToR-O3/Qwen3.8-Flash-Next-Uncensored-GGUF](https://huggingface.co/ToR-O3/Qwen3.8-Flash-Next-Uncensored-GGUF) | UD-IQ4_XS | 93,672,359,738 | 93.67 GB / 87.24 GiB |

The GSQ-RCO original and ToR-O3 uncensored choice need about **177.3 GB on disk
together**, before other files. Keeping both on disk does not load them both.
The original and uncensored UD-IQ4_XS pair instead needs about 187.3 GB.

The relevant memory division is:

| Choice | Routed expert payload | Sparse PLE payload | Other weights |
| --- | ---: | ---: | ---: |
| GSQ-RCO IQ3_S original | 46.84 GiB | 26.82 GiB | 4.20 GiB |
| Unsloth UD-IQ4_XS original | 56.24 GiB | 26.82 GiB | 4.17 GiB |
| ToR-O3 UD-IQ4_XS uncensored | 56.24 GiB | 26.82 GiB | 4.17 GiB |

These are counts from actual GGUF tensor headers. The sparse PLE table is read
by row and stays mapped. Dense weights, expert VRAM slots, KV state and work
buffers have separate GPU allocations. Forge 0.5 now caches experts in owned
host memory, so the complete download is not the host RAM demand. The ToR-O3
derivative uses the same bounded tier and can use a partial cache when its
experts do not all fit the available-memory ceiling.

**Start with the GSQ-RCO IQ3_S original.** It is the standard-format expert
source Avalon uses before custom re-encoding, retaining the 512-expert model.
It is not the separate 256-expert Coder/pruned variant. This choice gives a
roughly 62 GB machine more headroom than the earlier UD-IQ4_XS selection.
Quantization remains a precision choice; the publisher's quality evaluations
do not replace checking answers on your own tasks.

The actual Forge **0.5.0 release executable** accepted both shards of the
pinned GSQ-RCO IQ3_S choice and all four shards of the pinned ToR-O3 choice.
Both report 48 layers, 512 experts with 10 active, and 1,224 tensors. This was a
bounded header verification, not a full checkpoint download or GPU inference.
The exact records are linked from [VALIDATION.md](../VALIDATION.md).

Both exact UD-IQ4_XS choices passed Forge 0.4.1's complete four-shard
header checks: `qwen4exp`, 48 layers, 512 experts with 10 routed experts active,
and 1,224 tensors. This is a format check, not a full model load or GPU run.
The actual mixed quantization includes 64 routed-expert tensors using IQ2_S,
IQ3_XXS or Q2_0. The name does not mean that every weight is stored at four bits.
Use the reported tensor formats and an answer-quality comparison when choosing
between this compact option and a larger quantization.

The Unsloth and ToR-O3 UD-IQ4_XS first shards are approximately 10.9 MB and contain metadata. Their
remaining three shards contain the model tensors, including the large PLE
table. The first shard alone is not a usable model.

The ToR-O3 publisher describes replacing 146 main-model tensors in the Unsloth
quantization with quantized abliterated OrcaRouter weights. It is an independent
community derivative, does not reproduce the entire Orca checkpoint or its
MTP modifications, and has limited evaluation. The original model's benchmark
scores do not establish this derivative's accuracy.

The Unsloth and ToR-O3 repositories use the same GGUF basenames. Forge stores them under
different repository/revision directories, so downloading both does not
overwrite one with the other. Select the full catalogue ID when switching.

An alternative static IQ4_XS conversion from mradermacher is 98.42 GB and
passed Forge's implemented header checks. Our additional metadata audit found
48 zero compression-ratio entries where the source has `[0,0,0,4]` repeated.
The pinned Forge engine uses a fixed QSA pool of four and ignores that array,
so the discrepancy does not establish wrong Forge execution, but it is stale
metadata and is not the primary example here. Some larger mradermacher choices
are raw `.gguf.part1of3` pieces requiring reassembly; those are not directly
selectable standard GGUF shards.

The publisher's own [OrcaRouter GGUF repository](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF)
is gated. Access requires accepting its conditions on Hugging Face and using
an authorized `HF_TOKEN` in the Forge process. A public file listing is not an
access grant. The public mradermacher conversion is a separate published
source. Licence labels differ between repositories and metadata endpoints;
retain the model's supplied upstream terms. The source archive contains no
pretrained model weights.

## Check one choice before downloading

In **Windows PowerShell**, point `$Forge` at the Windows executable you built.
The path below follows the separate Windows build directory used in the
troubleshooting instructions; change it if your executable is elsewhere.

```powershell
$Forge = 'C:\code\nextora-cli-windows\target-windows-msvc\x86_64-pc-windows-msvc\release\nextora-cli.exe'
& $Forge --version

$NormalRepo = 'ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF'
$NormalRevision = 'ed59f92082b1e93c0e96d60a8b11aab089b52f09'
$NormalFile = 'IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'
& $Forge models files $NormalRepo --revision $NormalRevision --file $NormalFile --json

$UncensoredRepo = 'ToR-O3/Qwen3.8-Flash-Next-Uncensored-GGUF'
$UncensoredRevision = 'afa27ea7af1f83171a0aeafa8d5918eaa0dbfeee'
$UncensoredFile = 'UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf'
& $Forge models files $UncensoredRepo --revision $UncensoredRevision --file $UncensoredFile --json
```

`--file` was added in 0.4.1. It probes only the selected choice and its complete
shard set, then stops before full weight transport. Checks have a 180-second
overall limit, a 256 MiB aggregate read budget and a 64 MiB per-header bound.
They do not download the complete 94 GB models. Use the `models` array for
accepted choices and `excluded[].reason` for the reason a choice could not be
verified. A timeout or access error is not proof of an unsupported architecture.

The exact commits above make the checks reproducible. To inspect a later
publisher revision, use `--revision main`, review its returned immutable commit
and selected file, then use that commit for the download.

## Download and switch after choosing the storage and memory tradeoff

The following commands **do download the full models**. Run them only after
checking available disk space and deciding that these large checkpoints suit
your machine. Use the variables from the preceding section.

```powershell
& $Forge models pull $NormalRepo --revision $NormalRevision --file $NormalFile --models-dir 'C:\Models'
& $Forge models pull $UncensoredRepo --revision $UncensoredRevision --file $UncensoredFile --models-dir 'C:\Models'
& $Forge models list --models-dir 'C:\Models' --json
```

Start the server with this family's supported settings:

```powershell
& $Forge serve --models-dir 'C:\Models' --ctx 4096 --kv f16 --parallel 1 --mtp 0 --expert-ram auto --expert-warmup true --ram-margin 4096 --cache-ram 512
```

Open `http://localhost:8080`, select the downloaded original or uncensored
model, and load it. For CLI control, use the exact catalogue ID from
`models list` or `models status`:

```powershell
& $Forge models status --json
& $Forge models load 'COPY-THE-EXACT-CATALOGUE-ID-HERE'
```

Replace that placeholder with the real ID. Switching frees the previous model
before allocating the next. An allocation/loading failure can leave the server
unloaded, so another model may need to be selected.

For this architecture **one generation slot and MTP disabled are required**.
Multiple clients can queue requests; concurrent token generation across
several sequences is not supported for HC/PLE/QSA in this build. Other
compatible architectures have different slot/MTP capabilities.

## What the RX 9070 XT can and cannot tell us

The official checkpoint contains approximately 180 billion parameters: 125B
language-model parameters (6B active), 51B n-gram embeddings and 4B MTP. A
6B active count does not mean a 6B memory footprint. The selected GGUFs omit
some auxiliary material and use mixed tensor quantizations.

These choices do not fit completely into 16 GiB VRAM. Forge places useful
experts on the GPU and computes misses on the CPU during decode, using the new
owned RAM tier. Its startup ceiling uses available memory with OS, runtime and
prompt-cache reserves. The PLE table remains lazy on the CPU. Cold experts or
PLE rows can still require disk reads; useful speed depends on the working set,
CPU, available RAM, SSD and graphics driver. See [EXPERT-RAM.md](EXPERT-RAM.md).
No token-per-second result for the target machine is claimed in this release.

For a first hardware check, start with 4,096 context tokens and F16 KV.
For large code work after that check, version 0.6 provides `--preset code`
(16K / 4K output) and `--preset code-long` (32K / 8K output), exact request
budget checks and bounded repository context packs. See
[LARGE-CODE-PROJECTS.md](LARGE-CODE-PROJECTS.md). Raising context reserves more
state and reduces the space available for expert weights. For Windows AMD
drivers lacking device-coherent memory, the engine's CPU-expert mailbox can
be slow or fail visibility checks; a real load/decode test is necessary.

Native WSL Vulkan discovery must expose the actual RX 9070 XT. The previous
`llvmpipe` result is software execution. Use the Windows Vulkan executable
(which can be launched from WSL) if that remains the only native WSL adapter.
Changing ROCm cannot enable a HIP backend that Forge does not implement.

To report host RAM from Windows PowerShell:

```powershell
[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
```

For a useful diagnosis, retain the exact load error, GGUF filename, Forge
version, available RAM and hardware doctor output. Check the same workload on
both versions before deciding which one gives the desired answer quality.

## Evidence

- [Official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Original UD-IQ4_XS file group](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/UD-IQ4_XS)
- [Uncensored community derivative and provenance](https://huggingface.co/ToR-O3/Qwen3.8-Flash-Next-Uncensored-GGUF)
- [PLE table packaging](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/engram/README.md)
- [qwen38-source-metadata.json](validation/qwen38-source-metadata.json): publication metadata and byte totals.
- [qwen38-header-checks/](validation/qwen38-header-checks): observed bounded binary-header checks.
- `VALIDATION.md`: final release validation and execution limits.

Uses RustingStrata by Vasyl Trefilov.
