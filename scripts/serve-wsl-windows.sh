#!/usr/bin/env bash
# WSL-controlled Windows inference. Launch the executable directly; never build
# a PowerShell/cmd command string from model paths or user arguments.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/serve-wsl-windows.sh --exe PATH --models-dir PATH [options] [-- engine-options]

  --exe PATH          Windows forge9070.exe, using its WSL path (/mnt/c/...)
  --models-dir PATH   Shared model folder, using its WSL path
  --model PATH        Optional initial model, using its WSL path
  --port NUMBER       HTTP port (default: 8080)

Example:
  bash scripts/serve-wsl-windows.sh \
    --exe '/mnt/c/code/forge9070/target/release/forge9070.exe' \
    --models-dir '/mnt/d/Models' -- --ctx 4096

This controls a WINDOWS inference process from WSL. It is not native WSL GPU
computation. HF_TOKEN and STRATA_API_KEY are forwarded through WSLENV if set.
Use mirrored networking for WSL-to-Windows localhost access.
EOF
}

binary=''
models_dir=''
model=''
port=8080
extra=()
need_value() { [[ $# -ge 2 && -n "$2" ]] || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }; }
while (( $# )); do
    case "$1" in
        --exe) need_value "$@"; binary=$2; shift 2 ;;
        --models-dir) need_value "$@"; models_dir=$2; shift 2 ;;
        --model) need_value "$@"; model=$2; shift 2 ;;
        --port) need_value "$@"; port=$2; shift 2 ;;
        --) shift; extra=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ -n "$binary" && -n "$models_dir" ]] || { usage >&2; exit 2; }
[[ "$binary" == *.exe && -x "$binary" ]] || { printf 'Windows executable not found: %s\n' "$binary" >&2; exit 1; }
binary=$(realpath -- "$binary")
[[ "$port" =~ ^[0-9]{1,5}$ ]] && (( 10#$port >= 1 && 10#$port <= 65535 )) || { printf '%s\n' '--port must be between 1 and 65535.' >&2; exit 2; }
command -v wslpath >/dev/null 2>&1 || { printf '%s\n' 'wslpath is unavailable. Run this launcher in WSL with Windows interoperability enabled.' >&2; exit 1; }
if [[ -n "$model" && ! -f "$model" ]]; then printf 'Model file not found: %s\n' "$model" >&2; exit 1; fi
mkdir -p -- "$models_dir"
models_absolute=$(cd -- "$models_dir" && pwd -P)
windows_models=$(wslpath -w "$models_absolute")
args=(serve --models-dir "$windows_models" --port "$port")
if [[ -n "$model" ]]; then
    model_absolute=$(realpath -- "$model")
    windows_model=$(wslpath -w "$model_absolute")
    args+=(--model "$windows_model")
fi

# Share only the explicitly relevant variables. Preserve unrelated WSLENV rules
# and never print the token/key values. /w means Win32 processes launched by WSL.
forge_wslenv=${WSLENV-}
for variable in HF_TOKEN STRATA_API_KEY; do
    if [[ -n ${!variable-} ]]; then
        IFS=: read -r -a entries <<< "$forge_wslenv"
        kept=()
        for entry in "${entries[@]}"; do
            [[ -z "$entry" || "${entry%%/*}" == "$variable" ]] || kept+=("$entry")
        done
        kept+=("$variable/w")
        forge_wslenv=$(IFS=:; printf '%s' "${kept[*]}")
    fi
done
export WSLENV="$forge_wslenv"
printf '%s\n' 'Checking the Windows-host RX 9070 XT Vulkan path.'
if ! "$binary" doctor --require-gpu; then
    printf '%s\n' 'Windows GPU prerequisites failed. No CPU fallback was selected.' >&2
    exit 1
fi
printf '%s\n' 'Starting Windows GPU inference, controlled from this WSL terminal.'
printf '%s\n' 'For a WSL client, use mirrored localhost networking or an explicitly configured Windows host address.'
exec "$binary" "${args[@]}" "${extra[@]}"

