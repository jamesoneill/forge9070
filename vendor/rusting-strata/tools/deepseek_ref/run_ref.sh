#!/usr/bin/env bash
# Greedy reference from llama.cpp: 3 prompts, 64 tokens, top-5 logprobs per token.
set -euo pipefail
: "${LLAMA_BIN_DIR:?Set LLAMA_BIN_DIR to the directory containing llama-server}"
: "${STRATA_MODEL:?Set STRATA_MODEL to the first shard of a complete GGUF checkpoint}"
REF_DIR=$(cd "$(dirname "$0")" && pwd)
OUTPUT_DIR=${STRATA_REF_OUTPUT_DIR:-"$REF_DIR"}
mkdir -p "$OUTPUT_DIR"
exec 9>"${STRATA_GPU_LOCK:-"$OUTPUT_DIR/.gpu.lock"}"
flock 9
"$LLAMA_BIN_DIR/llama-server" -m "$STRATA_MODEL" -ngl 99 --cpu-moe -c 4096 --port 8199 --temp 0 \
  > "$OUTPUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
printf '%s\n' "$SERVER_PID" > "$OUTPUT_DIR/server.pid"
ready=0
for ((attempt = 0; attempt < 300; attempt++)); do
  if curl -fsS http://127.0.0.1:8199/health | grep -q ok; then ready=1; break; fi
  kill -0 "$SERVER_PID"
  sleep 5
done
if (( !ready )); then printf '%s\n' 'llama-server startup timed out' >&2; exit 1; fi
i=0
while IFS= read -r prompt; do
  i=$((i+1))
  jq -n --arg p "$prompt" '{prompt:$p,n_predict:64,temperature:0,top_k:1,n_probs:5,cache_prompt:false}' \
    | curl -fsS http://127.0.0.1:8199/completion -H 'Content-Type: application/json' -d @- \
      > "$OUTPUT_DIR/ref_$i.json"
done < "$REF_DIR/prompts.txt"
echo DONE
