#!/usr/bin/env bash
# Run the native Linux binary; CPU use is always explicit.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/serve-wsl.sh --backend cpu|vulkan [options] [-- engine-options]

  --bin PATH          Linux forge9070 executable (default: target/release/forge9070)
  --models-dir PATH   Model catalogue (default: FORGE9070_MODELS_DIR, else XDG data)
  --model PATH        Optional model to load initially; otherwise start unloaded
  --port NUMBER       HTTP port (default: 8080)

Examples:
  bash scripts/serve-wsl.sh --backend cpu --models-dir "$HOME/models"
  bash scripts/serve-wsl.sh --backend vulkan --models-dir "$HOME/models" -- --ctx 4096

Vulkan must pass doctor --require-gpu. Failure stops the launch; it does not
switch to CPU. All Hugging Face/model-switching features work in either mode.
EOF
}

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
binary="$project_root/target/release/forge9070"
models_dir="${FORGE9070_MODELS_DIR:-${XDG_DATA_HOME:-${HOME}/.local/share}/forge9070/models}"
model=''
backend=''
port=8080
extra=()
need_value() { [[ $# -ge 2 && -n "$2" ]] || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }; }
while (( $# )); do
    case "$1" in
        --bin) need_value "$@"; binary=$2; shift 2 ;;
        --models-dir) need_value "$@"; models_dir=$2; shift 2 ;;
        --model) need_value "$@"; model=$2; shift 2 ;;
        --backend) need_value "$@"; backend=$2; shift 2 ;;
        --port) need_value "$@"; port=$2; shift 2 ;;
        --) shift; extra=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "$backend" == cpu || "$backend" == vulkan ]] || { printf '%s\n' 'Choose --backend cpu or --backend vulkan explicitly.' >&2; exit 2; }
[[ "$port" =~ ^[0-9]{1,5}$ ]] && (( 10#$port >= 1 && 10#$port <= 65535 )) || { printf '%s\n' '--port must be between 1 and 65535.' >&2; exit 2; }
[[ -x "$binary" ]] || { printf 'Executable not found: %s\nBuild with bash scripts/setup-wsl.sh --build\n' "$binary" >&2; exit 1; }
[[ "$binary" != *.exe ]] || { printf '%s\n' 'Use serve-wsl-windows.sh for a Windows executable.' >&2; exit 2; }
binary=$(realpath -- "$binary")
if [[ -n "$model" && ! -f "$model" ]]; then printf 'Model file not found: %s\n' "$model" >&2; exit 1; fi
mkdir -p -- "$models_dir"
args=(serve --models-dir "$models_dir" --port "$port")
if [[ -n "$model" ]]; then args+=(--model "$model"); fi
if [[ "$backend" == cpu ]]; then
    args+=(--cpu)
    printf '%s\n' 'Starting native Linux inference with the explicit CPU backend.'
else
    printf '%s\n' 'Checking the native Linux Vulkan path before starting the server.'
    if ! "$binary" doctor --require-gpu; then
        printf '%s\n' 'Native Vulkan prerequisites failed. Use explicit CPU mode, or the Windows-host GPU launcher described in docs/WSL.md.' >&2
        exit 1
    fi
    printf '%s\n' 'Starting native Linux Vulkan inference; actual model execution still needs validation on this GPU/driver.'
fi
exec "$binary" "${args[@]}" "${extra[@]}"

