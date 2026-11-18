# Generation speed with output fidelity

Forge9070 0.4 prioritizes the rate of one generated response while retaining
the selected model's expert computation. It does not promise a particular
tokens/second figure or guarantee that a model's answers are correct. The
hardware, model, quantization and workload must be measured together.

## Three different questions

| Question | What establishes it |
| --- | --- |
| Can this engine load the file? | Architecture, tensor, quantization and shard checks |
| Does an optimization preserve the chosen model's behaviour? | Numerical and output comparisons under the same checkpoint and settings |
| Is the model answering the user's task correctly? | Trusted answers, task tests and evaluation of the actual model variant |

An optimization can faithfully reproduce a wrong answer. A model can also be
compatible with the engine while its quantization or fine-tuning reduces
quality on a particular task. These questions are deliberately kept separate.

## Default execution choices

- **One active request.** Extra slots remain available for aggregate throughput,
  but allocate additional context state and can reduce expert-cache residency.
- **All selected experts retained.** The Forge9070 workflow rejects nonzero
  expert-tail omission. The model still uses its normal trained top-k routing;
  this does not mean evaluating every expert in the model.
- **f16 KV cache.** Attention state uses floating-point storage. Explicit
  `--kv q8` remains a memory/precision tradeoff to evaluate on the workload;
  it is not described as a lossless replacement for f16.
- **No automatic precision downgrade or context reduction.** A speed result
  obtained by changing the model, quantization or task budget is a different
  configuration, not an implementation-only speedup.

These are conservative starting choices, not proof that they are the fastest
configuration on every driver or model. f16 itself is finite precision, and
quantized model kernels can use other finite-precision intermediate operations.
This profile is not a full-float reference implementation.

## Choosing weights consciously

The catalogue reports actual GGUF tensor formats. Its new `compression` report
counts tensor elements, identifies nominally sub-4-bit formats and separately
counts tensors storing routed experts. Mixed quantizations are inspected
tensor by tensor. A misleading filename cannot hide low-bit expert weights.
Counts include all declared tensors, including any embedded draft head; they
are metadata observations rather than a measured quality score.

The browser no longer marks the largest file that appears to fit VRAM, or the
smallest file that appears to fit RAM, as universally "recommended". A download
requires an explicit file selection. Sub-4-bit choices remain compatible when
the engine supports their formats, but their compression tradeoff is visible.
Four-bit and higher choices are not labelled guaranteed accurate either.

For a given supported checkpoint, **Q4_K_M and Q5_K_M are reasonable comparison
points** when balancing footprint and model fidelity. This is a starting
experiment, not a ranking across different model families. The official
[Qwen3-30B-A3B GGUF release](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF)
provides Q4_K_M, Q5_0, Q5_K_M, Q6_K and Q8_0 variants. Its Q4_K_M file is about
18.6 GB, already larger than the card's 16 GiB before context and work buffers;
this engine's hybrid expert placement is therefore relevant. A higher-precision
variant can require more CPU expert work. Do not infer a token rate from size.

Published [Qwen3 quantization research](https://arxiv.org/abs/2505.02214)
finds stronger degradation at very low precision and shows that results depend
on the quantization method and task. That study is not a validation of this
engine, these GGUF kernels, every MoE variant, or a universal four-bit threshold.
The correct comparison uses the exact files and workloads being considered.

Community uncensored or abliterated variants remain selectable. Their
compatibility checks are the same, but the original model's published task
scores do not establish the modified checkpoint's scores. Evaluate the chosen
variant independently.

## Sampling and reasoning affect answer quality

Greedy generation is useful for repeatable engine comparisons. It is not a
general prescription for the best answers. The official
[Qwen3 model card's best practices](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF#best-practices)
recommend temperature 0.6, top-p 0.95 and top-k 20 for thinking mode, and 0.7,
0.8 and 20 for non-thinking mode. It explicitly discourages greedy decoding
for that thinking model because it can degrade performance and repeat.

Those recommendations apply to that model. Other checkpoints, including
Instruct-2507 or a community modification, can have different instructions.
The engine supports temperature, top-p, top-k and seed. The chat API rejects
non-neutral unsupported presence/frequency/repetition penalties, min-p,
typical-p and logit bias with HTTP 400 instead of silently ignoring them;
neutral defaults are accepted. It also rejects sampling numbers that overflow
the engine's finite f32 representation. The publisher's complete sampling
recipe can therefore exceed this engine's supported subset. In particular,
the card's nonzero presence penalty is not implemented here.

Provide enough context and output budget for the task. Disabling reasoning or
shortening a response can reduce time to an answer, but can also change the
quality of reasoning. Generated-token throughput, time to first token and time
to a correct final answer are separate measurements.

## Tuning without turning quality into a speed label

Use the same checkpoint, prompt set, KV format, context and sampling conditions
when comparing execution settings. Keep all selected experts enabled. Exclude
warmup and use repeated measured runs; report actual generated tokens and
decode time separately from prompt processing and model loading.

The tuner compares candidate output with an ordinary decoding baseline on its
test prompts. Rejecting a mismatch is a useful fidelity gate. Passing those
prompts does not prove equality on all future inputs, sampling-distribution
equivalence, CPU/GPU numerical parity, or factual accuracy. Speculative decoding
and different batching can expose floating-point near-ties, so retain the
ordinary path as the reference when diagnosing differences.

Keep a separate set of representative tasks with trusted checks. For code,
compile and run meaningful tests; for extraction, compare expected fields; for
factual tasks, verify against sources. Use the same set when comparing
quantizations or community variants, and keep it separate from tuning prompts.

See [PERFORMANCE.md](PERFORMANCE.md) for the measurement workflow and
[../VALIDATION.md](../VALIDATION.md) for checks actually completed. This package's
CPU fixtures verify engine wiring and API behaviour. They do not establish the
accuracy or throughput of a pretrained model on the target RX 9070 XT.
