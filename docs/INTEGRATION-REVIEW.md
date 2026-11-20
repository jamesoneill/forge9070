# Integration review — ForgeAI staged GPU portability

This is a **merge reconciliation** against main commit `9478e6b45c6e47442e57f05dfedadff1d74536d4`, which already includes PR #8. The staging PR incorporates the GPU-portability diff from #7 while preserving ForgeAI branding, the executable alias and governance content already on main. It does **not** claim to have unified the application or imported the earlier unpublished native GLM/cache changes.

## Included
- Source/runtime/CLI/UI/scripts/CI evidence from PR #7.
- ForgeAI brand and static page **from main**, not duplicate content from #8.
- Hardware and architecture documentation merged by hand to preserve the new project name.
- No additional merge permission, build target or C++ inference backend.

## Required before merge
1. Recover and integrate the original native single-service, cache controller and GLM/Llama/Qwen2 graph sources; currently unavailable on the public repository.
2. Eliminate the retained Strata compatibility service, preserve applicable licence/notice obligations, and converge on one executable/runtime.
3. Reconcile auto-device selection with architecture-specific CPU-only graphs, and validate failed load rollback, concurrent state, eviction and strict security.
4. Run branch-specific CI (Windows, Linux, source/shader, native numerical and service suites) and then execute real model/hardware acceptance separately.
5. Confirm the live branch ruleset: it currently permits a GitHub integration to bypass update restrictions through pull requests as well as the repository owner. Restoring owner-only bypass requires a separately approved ruleset change.

The reconciled branch is a **staging PR**, not merge-ready. Do not merge this independently alongside #7.
