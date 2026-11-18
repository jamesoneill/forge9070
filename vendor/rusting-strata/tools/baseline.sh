#!/usr/bin/env bash
# Sweep llama.cpp --n-cpu-moe on the target file; print llama-bench markdown rows.
set -uo pipefail
M=${STRATA_MODEL:?Set STRATA_MODEL to a local GGUF checkpoint}
B=${LLAMA_BIN:-$HOME/Clones/llama.cpp/build-cuda/bin}
for n in ${SWEEP:-4 6 8 10 12 16 20}; do
  echo "== n-cpu-moe $n"
  "$B/llama-bench" -m "$M" -ngl 99 -ncmoe "$n" -fa 1 -p 4096 -n 128 -r 2 -o md 2>&1 | grep -E "^\| *qwen|error:" || echo "failed"
done
