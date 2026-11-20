# ForgeAI integration review — pending native consolidation

This branch stages tested GPU-portability code from PR #7 with presentation-only assets from PR #8. It deliberately does **not** introduce a second `forgeai` executable, change repository merge permissions, or merge either original PR.

## Preserved work

- Capability-based native Vulkan selection and explicit CPU fallback from PR #7.
- ForgeAI launch page and vector logo from PR #8, with build commands corrected to the **existing** `forge-engine` native executable.

## Required before merging to main

1. Obtain the actual local unified-application, native-model and automatic-cache commits/patches; they are not available on GitHub. Do not reconstruct or claim their tests have been independently reproduced.
2. Consolidate the root Strata compatibility service into a single native Rust runtime and check licences and attribution after removing it.
3. Choose the final executable identity (`forgeai`) only when the one-binary migration is implemented and release/CI scripts and the site agree.
4. Reconcile automatic Vulkan selection with architecture-aware CPU-only native GLM/Llama/Qwen2 graphs. Explicit GPU settings must fail before replacing an active model.
5. Run full CI again on the combined revision, including Windows, Linux, native numerical regression, secure offline networking, plugin API and cache-lease/pressure tests. Validate actual hardware separately.
6. Preserve owner-only merges to `main`; no integration bypass authority added by this branch.

## Source PRs

- GPU portability: https://github.com/jamesoneill/forge9070/pull/7
- Branding: https://github.com/jamesoneill/forge9070/pull/8

**Status:** Integration staging only. Not yet ready to merge.
