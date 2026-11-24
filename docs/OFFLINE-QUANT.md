# Nextora offline GGUF quantizer

**Experimental, correctness gated.** The first native Rust exporter supports only **single-file GGUF**, floating-point source matrix rows (F16/F32/BF16), and Q4_0 or Q8_0 block encodings. It refuses to overwrite existing files. It is not yet importance-matrix-aware variable-bit quantisation.

## Usage

Windows PowerShell:

    .\target\release\nextora.exe quantize --model 'F:\Models\my-f16.gguf' --output 'F:\Models\my-q4.gguf' --target q4-0 --max-nrmse 0.25

Linux:

    ./target/release/nextora quantize --model /models/model-f16.gguf --output /models/model-q8.gguf --target q8-0 --max-nrmse 0.25

This is an offline CPU operation. A large model may take considerable time and temporary disk space. Each encoded row is decoded and checked; each converted tensor must meet the normalised RMS reconstruction error limit. The source and destination must be distinct; the destination cannot already exist, including symlinks. The exporter writes a unique temporary file next to the destination, checks exact tensor geometry, metadata count and entire-output GGUF parsing, then publishes via a create-new hard link.

## Boundaries

- Only single-shard GGUF with eligible original floating-point matrix weights is supported; never re-quantise an already-quantised checkpoint.
- Other weights, including embeddings and norms, remain unchanged. Original GGUF metadata types (including nested arrays) are copied unchanged except a known uint32 `general.file_type` field, if present.
- This is uniform Q4_0/Q8_0, not optimally allocated tensor-by-tensor mixed precision and not the upstream `--target-size` feature.
- Tensor reconstruction error is **not** coding accuracy or task-level model-quality parity. No new model or GPU architecture is supported by the exporter.
- Even when output is smaller, actual VRAM fit, context, transfer and token speed need physical hardware benchmarks.

A full acceptance benchmark uses a supported real checkpoint, exact source/converted references, fixed prompts and token/logit checks, then measures load, VRAM, time-to-first-token, prefill, decode and concurrent request throughput. Conversion fails rather than silently accepting errors above the selected limit.

See [model-fit](MODEL-FIT.md), [hardware support](HARDWARE-SUPPORT.md), [accuracy](ACCURACY.md) and [validation](../VALIDATION.md).

## Automatic file-size target (experimental mixed precision)

Instead of selecting a uniform Q4_0 or Q8_0 format, use an output file-size ceiling in MiB:

    .\target\release\nextora.exe quantize --model 'F:\Models\source-f16.gguf' --output 'F:\Models\source-budgeted.gguf' --target-size-mib 12288 --max-nrmse 0.25

The command makes a **first offline assessment pass** over eligible original floating-point matrix weights. It independently quantizes and decodes candidate Q8_0 and Q4_0 rows, accumulates measured per-tensor squared reconstruction error, then greedily chooses the lower measured incremental error per byte saved until the requested output size fits. Unsupported matrices, existing quantized weights, embeddings and normalization tensors remain unchanged. The exporter makes a **second pass** to write the chosen formats to a new file, checks every converted tensor against the same NRMSE threshold, reparses the completed GGUF, and rejects a final output above the requested size.

Only one of `--target` and `--target-size-mib` may be given. If no combination of the supported candidate formats meets both the file-size ceiling and reconstruction threshold, no output is published. A target larger than the source file is rejected as unnecessary. Only **single-shard original floating-point** GGUF is supported.

This is a greedy tensor-error heuristic, **not** an importance-matrix optimiser, optimal bit-allocation solver, or proof of model/task accuracy. It does not model VRAM use, KV-cache residency, PCIe transfers or sustained tokens/sec. Your real RX 9070 XT still needs workload and quality-parity benchmarks before any performance claim.
