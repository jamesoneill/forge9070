# Nextora model-fit: memory-aware checkpoint planning

**Status:** Read-only native Rust implementation. No quantized checkpoint is generated.

This command estimates what a *new offline quantized GGUF* might occupy based on the original file's validated tensor shapes. It also returns a conservative nominal GPU allocation ceiling and current host memory-planning evidence, making it easier to identify which experiments are worth attempting on consumer GPUs.

## Usage

```powershell
.\target\release\nextora.exe model-fit `
  --model "F:\Models\MyModel\model-00001-of-00002.gguf" `
  --ctx 4096 --requests 1 `
  --target-vram-mib 16384 `
  --target-file-mib 12288 `
  --gpu-headroom-percent 20 `
  --json-out model-fit.json
```

For Linux, use `./target/release/nextora model-fit ...`. If a model has multiple shards, keep all shards together and specify a valid shard path. Run `nextora doctor` and `nextora insight` for actual process-level hardware evidence, then `nextora bench` on a separately validated compatible checkpoint.

## What is measured, estimated and unknown?

| Output | Meaning | Not established |
| --- | --- | --- |
| Parsed original tensor payload | Actual encoded bytes declared by a complete valid GGUF | Peak resident RAM or VRAM |
| Candidate F16, Q8_0, Q6_K, Q5_K, Q4_K, Q4_0, IQ4_XS, IQ3_S, IQ2_S sizes | Layout-based hypothetical file footprints for eligible source floating-point matrix weights | Quality, conversion support, exact output file size or available encoder |
| Unchanged tensors | Already-quantized weights, norms, embeddings, positional data and other excluded tensors remain at their original byte footprint in every estimate | That those choices minimise quality loss |
| Nominal GPU budget | User-specified nominal capacity or eligible Vulkan heap, less an explicit headroom percentage | Free VRAM, model fit or successful GPU execution |
| Host memory planner | Capacity estimates under the existing Qwen host-state planner at requested context/slots | Real usage or paging under load |
| Architecture | GGUF metadata checked against the native Qwen text architecture set | Inkling support, tensor semantics, tokenizer compatibility or complete model validation |

The layout size estimates are **not** calibrated quality/performance predictions. Because the engine does not yet provide an offline encoder with an importance-matrix path, the planner does not automatically convert the file, select an optimal precision layout, or suggest purchasing hardware. A file target is distinct from the memory required to run the model.

## Hardware-specific development workflow

1. Collect an exact supported model/checkpoint revision and record shard hashes.
2. Run `model-fit` with the proposed context, number of request slots and nominal VRAM capacity.
3. Quantize a separate **source floating-point checkpoint** using a validated offline encoder when one exists. Do not repeatedly quantize pre-quantized files to satisfy a size number.
4. Verify tensor/file metadata, tokenizer and output quality against the source, then compare CPU/Vulkan inference on identical prompts.
5. Profile actual model loading, resident RAM/VRAM, transfers, prefill and generation speed; benchmark 1–4 request slots.
6. Only recommend hardware upgrades when repeatable measurements demonstrate a bottleneck and a controlled comparison provides evidence of benefit.

## Other model architectures

Nextora currently implements `qwen3`, `qwen3moe`, `qwen35` and `qwen4exp` text paths. **Inkling 2 Mini's looped architecture is not supported**, even if its GGUF metadata is readable. Repeated layer execution requires independent loop scheduling, attention/recurrent-state correctness, expert-caching validation and CPU/Vulkan parity. Do not relabel an Inkling GGUF to bypass model compatibility checks.

## Next implementation work

- An offline, verified quantizer for a strictly supported source format, written to a **new** file with integrity checks, precision policy and per-tensor quality gates.
- Exact VRAM residency and context-dependent working-set estimation for native Qwen architectures, validated on real GPUs and backed by measured allocator high-water marks.
- Native support for additional model architectures, starting with a small reproducible CPU correctness test before implementing Vulkan.
- Improved expert offload/transfer instrumentation and bounded scheduling benchmarks; no unchecked omission of routed experts.
- Hardware-aware tuning with repeatable baseline acceptance and opt-in hardware-price comparisons.

No remote upload, model download or settings change is performed by `model-fit`.
