# Runtime consolidation status

The root `forge9070` Cargo package is now a thin native-engine CLI alias. Its
runtime no longer imports `rusting_strata`. The `forge-engine` crate owns
the only execution pipeline, model state, queue, API and Vulkan/CPU backends.

The previous standalone repository-context utility is available as
`forgeai context --repo <path> --query <text>` and from either alias.
Its source remains `src/code_context.rs` but is compiled into the native
library and shares Forge's GGUF tokenizer for exact counts.

**Pending:** The vendored RustingStrata source tree remains as historical
source until the lockfile, third-party licence inventory, and build/test
dependencies are verified and it can be removed safely. It is not a workspace
member or runtime dependency. The earlier unpublished GLM, Llama, Qwen2 model
graphs are not imported here; only existing native architectures are supported.

Do not claim release readiness until branch-specific Windows/Linux CI and the
release-notice audit succeed.
