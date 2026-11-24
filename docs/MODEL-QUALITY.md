# Nextora model-level quality acceptance

Use `nextora quality` to compare a source pretrained model and a separate candidate checkpoint **after offline quantisation**, with identical raw prompts. This is a true inference test, not an estimate from tensor sizes or reconstructed weights.

```powershell
.\target\release\nextora.exe quality `
  --reference 'F:\Models\source-f16.gguf' `
  --candidate 'F:\Models\candidate-q4.gguf' `
  --prompt 'Explain the Rust borrow checker in two sentences.' `
  --prompt 'Write a test for a queue.' `
  --ctx 4096 --steps 16 `
  --candidate-backend cpu `
  --max-mean-kl 0.1 --min-top1-agreement 0.8 `
  --json-out quality.json
```

For hardware parity use `--candidate-backend vulkan` with a detected compatible adapter; the reference remains CPU. This command is read-only and does not change or load your running server model.

## Acceptance

The command checks model architecture, all tensor shapes, tokenizer metadata and vocabulary size. Each prompt is encoded through both tokenizers and must produce identical token IDs. Nextora computes the complete next-token logits for the source and candidate, checks finite values, and measures the KL divergence between their softmax distributions, absolute logit differences and top-1 token agreement. Each subsequent step feeds the **reference model's top token to both models**, preventing free-generation divergence from invalidating a same-input comparison. Scoring stops at reference EOS or the requested step limit.

The report records aggregate and per-case statistics, passes only if the configured mean KL and top-1 agreement thresholds pass, and omits prompt text and model paths. A failed threshold returns a nonzero CLI exit status; when `--json-out` is supplied the measured report is saved first.

## Important limitations

- These are **teacher-forced raw-prompt inference comparisons**, not independent model truthfulness, coding-agent task success, instruction-following, chat-template or tool-call correctness tests.
- The reference itself might be wrong. Matching its output does not validate accuracy.
- CPU reference and Vulkan candidate may have small floating-point differences, requiring separately justified tolerances.
- Source/candidate checkpoint revisions and quality baselines must be controlled; the tool verifies tokenizer and geometry but does not yet hash every shard.
- `--prompt` may appear in shell history; the **report** does not export prompt text.
- Actual sustained speed, free VRAM, thermal throttling and hardware cost require separate benchmarking.

Use this as a quality gate before accepting quantised-model performance improvements. See [Offline quantisation](OFFLINE-QUANT.md), [model-fit](MODEL-FIT.md), [accuracy](ACCURACY.md), and [validation](../VALIDATION.md).
