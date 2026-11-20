# Nextora integration review

The single-runtime refactor is on draft PR #9. It includes the GPU portability
logic from PR #7, the Nextora branding already merged through PR #8,
automatic Rust host-cache resizing, and migration of the old root command
alias and context packer onto the independent native engine.

The root no longer links RustingStrata; the vendored source, retired
launcher/test files and old Cargo dependencies have been removed.
Original upstream licence text and attribution are retained separately.

## Validation and release blockers

1. The required Windows/Linux source, native, API, Vulkan, UI and release
   gates must pass **on this exact head**. CI of an earlier head is not enough.
2. Complete pretrained checkpoint acceptance, physical GPU throughput and
   target Windows/WSL hardware execution remain unverified.
3. The earlier native GLM/Llama/Qwen2 graphs were written in an unpublished
   prior workspace; those exact sources and their numerical evidence have not
   been integrated. Only advertised architectures are supported.
4. Auditing old documentation and model-profile migrations is required before
   telling existing users that all old provider options still work.
5. Live main-branch ruleset permits a GitHub integration PR bypass in addition
   to the owner. The connector cannot update this ruleset; exclusive owner-only
   merge authority must be configured separately.

Do **not** auto-merge or mark the draft ready until the owner reviews the
completed checks. All executable names route to the same Rust engine;
the 8080 Strata service is gone.
