# Native-only application consolidation

ForgeAI now exposes **one native Rust inference implementation**. The
`forgeai` and `forge-engine` binaries share the native crate entry point.
The `forge9070` binary is a thin CLI shim calling that same crate.

There is no Strata runtime dependency, provider execution path or second
HTTP server. The default native service runs on port 9070 and manages
Vulkan/CPU selection and bounded expert caching in Rust. The old Strata
service on port 8080 and its CLI-only features were intentionally retired.

The bounded, Git-tracked local source-context packer is compiled into the
native library and available as `forgeai context --repo PATH --query TEXT`.
An optional `--model` supplies the native GGUF tokenizer for exact counts.

Historical Strata attribution and upstream licence receipts are preserved,
but the vendored source tree is no longer part of the repository.

**Limits:** The native-model work from the earlier unpublished local branch
(Qwen2, Llama, GLM-5.2) has not been incorporated. Supported native GGUF
architectures remain those explicitly listed by `forgeai capabilities`.
A passing CI run does not certify pretrained model output or physical GPU
throughput. The live repository rulesets require separate audit to enforce
exclusive owner-only merge authority.
