# Native runtime cache control — implementation review

This revision adds **in-process Rust** adjustment of owned expert weight bytes.
The service samples available host memory once per second, including while idle.
It shrinks promptly, grows by at most 64 MiB after three healthy observations,
and never preloads experts just because its target grew.

Pinned expert leases remain valid when capacity shrinks below resident bytes;
future admissions bypass owned caching until those leases are released. The
cache's resize operation synchronizes with admissions under the existing mutex.

The initial model-load memory plan remains a hard ceiling. Runtime headroom
uses at least 512 MiB plus any larger configured OS margin. The monitor does not
claim to measure free VRAM; the native bounded Vulkan expert cache is unchanged.

## Acceptance
- Added unit regressions for active leases during shrink, lower-than-pinned
  capacity, warmup and recovery.
- Re-run locked Rust CI on Windows/Linux and existing numerical gates.
- Real memory pressure, GPU telemetry, model quality, and full native
  GLM/Llama/Qwen2 support remain separate pending validation.
- The former local unpublished implementation was **not imported**; this is a
  fresh, narrower change in the published runtime.
