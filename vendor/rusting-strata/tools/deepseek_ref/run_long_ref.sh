#!/usr/bin/env bash
# Greedy reference for the long prompt (exercises compressed-attention blocks / indexer past 128 tokens).
set -euo pipefail
: "${LLAMA_BIN_DIR:?Set LLAMA_BIN_DIR to the directory containing llama-server}"
: "${STRATA_MODEL:?Set STRATA_MODEL to the first shard of a complete GGUF checkpoint}"
: "${STRATA_LONG_PROMPT_FILE:?Set STRATA_LONG_PROMPT_FILE to your local UTF-8 prompt file}"
REF_DIR=$(cd "$(dirname "$0")" && pwd)
OUTPUT_DIR=${STRATA_REF_OUTPUT_DIR:-"$REF_DIR"}
mkdir -p "$OUTPUT_DIR"
exec 9>"${STRATA_GPU_LOCK:-"$OUTPUT_DIR/.gpu.lock"}"
flock 9
"$LLAMA_BIN_DIR/llama-server" -m "$STRATA_MODEL" -ngl 99 --cpu-moe -c 4096 --port 8199 --temp 0 \
  > "$OUTPUT_DIR/long_server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
ready=0
for ((attempt = 0; attempt < 300; attempt++)); do
  if curl -fsS http://127.0.0.1:8199/health | grep -q ok; then ready=1; break; fi
  kill -0 "$SERVER_PID"
  sleep 5
done
if (( !ready )); then printf '%s\n' 'llama-server startup timed out' >&2; exit 1; fi
jq -Rs '{prompt:.,n_predict:32,temperature:0,top_k:1,n_probs:5,cache_prompt:false}' "$STRATA_LONG_PROMPT_FILE" \
  | curl -fsS http://127.0.0.1:8199/completion -H 'Content-Type: application/json' -d @- \
    > "$OUTPUT_DIR/ref_long.json"
echo DONE
