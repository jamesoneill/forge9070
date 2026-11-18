# Optional reference generation tools

These upstream helpers can generate local llama.cpp comparison results. Set
`LLAMA_BIN_DIR` to the directory containing `llama-server` and `STRATA_MODEL` to
the first shard of the complete local GGUF checkpoint. `run_ref.sh` uses the
three synthetic prompts in `prompts.txt`. `run_long_ref.sh` requires an explicit
`STRATA_LONG_PROMPT_FILE` containing your own UTF-8 prompt.

Use `STRATA_REF_OUTPUT_DIR` to choose a local output directory and
`STRATA_GPU_LOCK` to choose a shared lock file when coordinating GPU use.
Generated results and logs can contain model paths and generated text; they
are intentionally omitted from this public source distribution.

The `dequant/` binary fixtures and `tpl_tools.json` are retained because the
vendored numerical and template tests consume them. They do not establish
Forge9070 performance on any hardware.
