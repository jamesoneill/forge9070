# Inkling 2 Mini looped architecture — Nextora foundation

**Status: fixed loop scheduling and dependency plan only. Native Inkling model loading/inference remains unsupported.**

Official model architecture: [Thinking Machines release](https://thinkingmachines.ai/news/inkling-2-mini/), [Hugging Face architecture guide](https://huggingface.co/docs/transformers/main/model_doc/inkling2_loop).

The released fixed-loop model stores 24 layers. Layers 0–5 run once, layers 6–17 are reused for three passes, and layers 18–23 run once, yielding 48 layer applications per generated token. Passes two and three consume trained messages from the preceding loop pass. These are **learned residual gates**, not ordinary repeat-forward operations.

The new Rust `inkling_plan` module defines an exact schedule and a compatibility report with missing prerequisites. Tests verify layer reuse, pass ordering, mandatory prior-pass message dependencies and rejection of unsupported pass counts. These tests do not load a checkpoint or validate numbers.

## Required before enabling native inference

- Reproduce official tensor naming, architecture metadata, tokenization and quantised tensor layouts in a tiny independent CPU test.
- Implement trained residual/message projections, gating, proper attention and experts, and pass-aware KV/cache lifetimes. Looping Qwen layers without these operations is mathematically incorrect.
- Validate CPU reference logits/token outputs with a known fixture; then implement Vulkan kernels with numerical parity.
- Validate a real quantised GGUF and run performance tests with a 16GB RX 9070 XT, including actual allocated VRAM and full memory overhead.
- Keep unsupported architecture failure explicit until each gate passes; do not silently use an external runtime or claim a compatible model by name.

The official NVFP4/MXFP8 checkpoint is approximately 20GB and recommends at least 32GB on other GPU platforms. A community GGUF size does not establish fit or quality on 16GB.
