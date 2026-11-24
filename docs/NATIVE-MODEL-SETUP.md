# Native model setup and memory placement

Nextora uses its independent Rust execution paths. Preparing a model runs a
separate native worker; it does not start llama.cpp, Strata or another inference
runtime. Here, tuning means choosing runtime settings, not training weights.

## User workflow

1. Search Hugging Face. Results must pass the native architecture, tensor shape,
   quantization, tokenizer and chat-template checks against pinned GGUF headers.
2. Download the complete shard set to the configured models directory (normally
   `F:\Models` on Windows). Verify files before offering setup.
3. Select **Prepare model**. The worker checks memory fit within the checkpoint's
   declared context and the native backend's limits, then measures candidate
   runtime settings. Expert-cache filling is disabled during fitting.
4. Save the profile for this model, engine build and machine. Preparation restores
   the model that was previously serving, including an unloaded service.
5. Select **Use prepared model**. This loads the selected model and performs its
   normal cache warmup. Profiles persist; warmed RAM and GPU caches do not.

Progress distinguishes downloading, verification, memory fitting, benchmark,
profile saving, restoration and normal model loading. Preparation can be cancelled;
the worker exits before the previous service is restored.

## Capacity and performance are different measurements

The maximum-fit option seeks the largest context that passes memory checks on a
bounded search grid. It respects the context declared by the checkpoint. It does
not extend RoPE limits just because RAM is available. A full-capacity allocation
probe and a short inference probe establish capacity and basic execution, not
quality or throughput with an occupied full context.

The speed benchmark uses a short coding prompt, excludes warmup and compares
repeated decodes at the chosen allocation context. Slow preparation candidates
use shorter samples; the report records their actual lengths. Reports retain
the actual prompt length and the memory-fit ceiling separately. Neither a
512K-capacity test nor a short-prompt token rate establishes 512K-prompt speed.

## Context storage

Cache precision and placement are independent choices. F32 retains ordinary
floating-point K/V values. Q4_64 quantizes groups of 64 values and retains every
initialized token. GPU placement provides device-local cache storage. RAM
placement uses host-visible system-memory storage buffers and a separate budget;
it does not count the RAM cache against the owned GPU allocation limit.

The planner reserves the complete future RAM context, growth overlap, working
memory and an OS margin before sizing the bounded expert cache. This prevents a
large expert warmup from consuming memory promised to future context growth.
The running cache controller protects remaining future context capacity too.

RAM placement increases capacity on a discrete GPU but attention still has to
read the history across PCIe. Large occupied contexts can therefore be much
slower. Prefer GPU cache storage when it fits the requested workload; use RAM
when the extra context capacity is worth measuring that tradeoff.

## Supported model families

Compatibility is an explicit native capability check, not a promise that every
Hugging Face model or every GGUF file runs. Ordinary Qwen3 dense and MoE use the
native CPU, hybrid or resident Vulkan paths. The Qwen3.5/Qwen3.8 dense adapter
(`qwen35`, including the 27B text model) initially uses CPU or hybrid execution;
its recurrent state, partial RoPE, query/output gates and MTP validation differ
from ordinary Qwen3. Qwen3.8 Flash-Next (`qwen4exp`) uses its own adapter and backend
context limits. Precision and fast-math options are checked per family.

No full Qwen3.8 checkpoint was downloaded as part of adding these adapters.
Header validation and small independent numerical fixtures do not establish
pretrained-model quality or a throughput result for either new model.

## Ideas studied in llama.cpp

The following upstream designs informed the implementation choices; Nextora
implements its own code and does not link the upstream runtime:

- Separate context-memory layout, buffer placement and execution state, with
  full-size planning before serving. Reference:
  [llama-memory.h](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-memory.h).
- Treat K/V precision and host/device placement as explicit settings. Reference:
  [llama-kv-cache.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp).
- Query actual Vulkan capabilities and driver-reported memory usage rather than
  relying only on the GPU's advertised capacity. Preserve a separate owned
  allocation limit. Reference:
  [ggml-vulkan.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp).
- Validate the GGUF block layout independently for each quantization format.
  Native Q3_K decoding extends the existing CPU and Vulkan format checks.

The existing optional activation integer-dot fast path, grouped expert execution,
GPU weight reuse and tiled full-history attention complement these changes.
Batched prompt processing and further fused kernels remain performance work;
this setup flow does not claim exhaustive hardware optimization.
