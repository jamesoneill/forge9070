> Historical snapshot for v0.8.0. Published with path redaction and link repairs; the recorded test outcomes and limitations are retained. See [the evidence policy](../README.md).

# Independent Forge engine and performance work

Forge9070 0.8 contains a new Rust package, `forge-engine`, whose inference path
does not call RustingStrata. This is the first executable migration stage:
an independent CPU reference and performance foundation. It is not yet the
replacement for the target RX 9070 XT / Qwen3.8 production path.

## What executes today

| Component | Independent Forge core | Existing Forge compatibility path |
|---|---|---|
| Entry point | `forge-engine`, or `forge9070 engine` | Existing `forge9070 serve/generate/chat/bench` |
| Runtime | New Rust code in `crates/forge-engine` | Modified RustingStrata, attributed to Vasyl Trefilov |
| Architectures | Ordinary `qwen3`, `qwen3moe` | Existing supported sparse-MoE policy including `qwen4exp` |
| Weight storage | Single-file GGUF, mapped and quantized | Existing sharded loader and expert tiers |
| Weight formats | F32, F16, BF16, Q4_0, Q8_0 | Existing supported mixed quantizations |
| Compute | CPU, explicit `--cpu` | CPU or Vulkan according to existing commands |
| Attention state | Per-sequence F32 reference KV | Existing F16/Q8 GPU choices |
| HTTP, tools, queue | Not yet connected to the independent runtime | Existing 0.7 API and admission system |
| Qwen3.8 / RX 9070 XT | Not implemented in the new runtime | Existing compatibility path; real hardware validation still required |

The new runtime has no fallback. An unsupported checkpoint or omitted `--cpu`
returns an error. It must never report native support while routing to Strata.
Use `forge-engine capabilities` to inspect the executable's actual scope.

The existing source distribution still includes and uses RustingStrata by
Vasyl Trefilov. Its original license and provenance are retained. The new
component is independently implemented under its own MIT license; it does
not relicense any existing engine code or third-party dependencies.

## Performance decisions already represented in the design

- **Mapped weights remain quantized.** Tensor views address mapped storage;
  the loader does not convert the entire checkpoint to a floating-point copy.
- **Direct block dot products.** Matrix-vector execution consumes stored
  F32/F16/BF16/Q4_0/Q8_0 rows without allocating a decoded row for every dot.
- **Runtime SIMD selection.** On x86_64 CPUs advertising both AVX2 and FMA,
  rows of at least 128 values use the new vector kernels for F32, BF16, Q4_0
  and Q8_0. F16 additionally requires F16C and converts eight stored half values
  per iteration without allocating a decoded row. Unsupported instruction sets
  and smaller rows use the portable scalar path. This is CPU acceleration;
  the GPU backend remains unimplemented.
- **Reusable sequence scratch.** Attention, FFN and router work areas belong
  to the sequence. Independent requests share immutable weights and retain
  separate KV histories.
- **Prompt projection only when needed.** Prefill still executes each token's
  transformer computation, but skips the vocabulary projection for prompt
  tokens whose logits will not be used. This is not a batched GPU prefill path.
- **Cached rotary frequencies.** Inverse frequencies are model data, and each
  token's rotary values can be shared across attention layers.
- **Bounded CPU parallel work.** Matrix row parallelism uses a threshold;
  the benchmark's one-to-four sequences share one Rayon worker budget. Adding
  requests does not create a fresh complete weight copy or worker pool each.

These design choices are not a measured claim of beating another engine.
Every performance statement must identify its model, quantization, context,
backend, hardware and concurrency. Synthetic fixture speed cannot predict a
useful 512-expert model's speed on the RX 9070 XT.

## Build and execute independently

From the source root, with Rust 1.100 and the platform C/C++ build tools:

```bash
cargo build --release --locked -p forge-engine
cargo tree --locked -p forge-engine --edges normal
./target/release/forge-engine capabilities
```

On Windows the output is `target\release\forge-engine.exe`. This command builds
the independent package; it does not build the Strata compatibility package.
The root Forge binary still links both packages to preserve existing commands.
The CPU core uses the Hugging Face `tokenizers` library for BPE and its regular
expression dependency; it is not claiming that every third-party build input is
Rust or that the whole application has no other dependencies.

Raw-prompt generation with a synthetic model:

```bash
./target/release/forge-engine generate --cpu \
  --model tests/fixtures/tiny-qwen3moe.gguf --ctx 128 \
  --prompt 'hello' --max-tokens 8 --json
```

Generation currently uses deterministic greedy sampling. The prompt is raw
text; no chat template, tool schema, system message or reasoning policy is
silently inserted. Model and tokenizer compatibility are validated at load. The model must explicitly
state its positive RoPE frequency base and RMSNorm epsilon; absent values are
rejected. Scaled RoPE, including the legacy linear-scaling key, is unsupported.
The optional ChatML formatting helper is limited to its documented text roles;
it does not make the new core a tool-calling server.

## Measure performance without hiding the workload

```bash
./target/release/forge-engine --threads 4 bench --cpu \
  --model tests/fixtures/tiny-qwen3moe.gguf --ctx 256 \
  --prompt 'Measure independent sequence state.' \
  --decode-steps 64 --requests 1 --warmup 1 --reps 5 \
  --json-out artifacts/native-one-request.json

./target/release/forge-engine --threads 4 bench --cpu \
  --model tests/fixtures/tiny-qwen3moe.gguf --ctx 256 \
  --prompt 'Measure independent sequence state.' \
  --decode-steps 64 --requests 4 --warmup 1 --reps 5 \
  --json-out artifacts/native-four-requests.json
```

Use an actual supported model and representative code prompt for useful
throughput measurements. The committed tiny fixtures are for validation only.

The JSON reports load and tokenization time separately, excludes warmup, and
records every measured sample. Prefill and decode phases have separate clocks.
Each request uses an independent sequence; the aggregate decode rate is total
decode steps across requests divided by the phase's wall time. Per-request
completion times include scheduling delay within that phase. The p95 is the
nearest-rank sample percentile, not a statistical guarantee about production.

Benchmark decoding deliberately continues through EOS for a fixed workload;
normal generation stops at EOS. Decode-step counts count actual model calls.
For normal generation the first sampled token comes from prefill, so the number
of decode calls can be one less than the number of generated tokens. This is
made explicit in the JSON rather than inflating a generation speed figure.

Repeated greedy checksums are a basic reproducibility gate, not an accuracy
evaluation. Full-vector parity tests are the correctness gate.

The standalone kernel benchmark compares the actual selected CPU dot kernel
with its portable scalar reference, using the same encoded weights and inputs:

```bash
./target/release/forge-engine kernel-bench --width 2048 --rows 128 \
  --iterations 8 --reps 5 --json-out artifacts/native-kernels.json
```

It checks every output before timing, warms both paths, alternates execution
order and records every sample. It measures the checked row API with warm
synthetic weights; it excludes model orchestration, attention and GPU work.
The scalar/selected time ratio must not be described as a whole-model speedup.
Both benchmarks use the middle sample, or the mean of the two middle samples,
for medians. Model benchmarks reject report paths that alias the input model,
including hard links; reports are written by atomic file replacement.

## Accuracy gates before performance optimizations are retained

The test reference may import the existing engine; the new runtime cannot.
The same input token IDs are forced into both implementations. Tests compare
all logits, hidden states and attention K/V values, including a varied output
projection so that the deliberately favored synthetic `A` token cannot conceal
a broken forward pass. They also cover reset, interleaved independent sequences,
invalid tokens, context bounds and tokenizer round trips.

Quantized arithmetic is checked against independently decoded known blocks.
SIMD or parallel reduction may change floating-point rounding; tolerances must
be measured and reported, not described as bit-identical arithmetic. Model
architecture extensions need their own fixtures before entering capabilities.

## Remaining gates for the requested fast GPU replacement

1. **Native Vulkan execution.** Implement device-owned state and persistent
   weights, packed-quantization matrix kernels, fused normalization/activation,
   attention and expert routing. Reuse command buffers and avoid CPU round trips
   between individual tensor operations.
2. **Qwen3.8 semantics and formats.** Add the actual `qwen4exp` path, including
   recurrent layers, hyper-connections, PLE histories, QSA selection and the mixed
   quantizations of the selected checkpoint. Ordinary Qwen3 support is not this.
3. **Large-model memory placement.** Add shard-aware maps, lazy sparse PLE rows,
   bounded host expert storage, GPU hot-expert residency, transfer/compute overlap
   and safe expert leases during eviction. Preserve all selected experts.
4. **Request execution.** Connect a typed sequence interface to the existing
   coding API and queue. Implement real GPU batching with separate state for
   every supported architecture before increasing its concurrency capability.
5. **Target-card acceptance.** Run per-layer/logit checks and useful code tasks
   on the actual checkpoint, followed by 1/2/4-request prefill/decode, memory,
   cancellation and long-context tests on the RX 9070 XT. Compare aggregate
   throughput and per-request latency; four requests do not imply four times
   one-request speed.

The priority is useful decode throughput with preserved model computation.
Removing Strata from the production path is gated on these results. The CPU
reference is a component of that migration, not a substitute speed claim.

## Primary format and model references

- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [GGML quantization layouts](https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c)
- [Qwen3 model family](https://github.com/QwenLM/Qwen3)

The existing comparison tests reference the pinned runtime recorded in
`UPSTREAM.json`. References describe formats or supply a numerical oracle;
they are not evidence that the new GPU path already exists.
