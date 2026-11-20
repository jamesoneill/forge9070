#!/usr/bin/env bash
# Launch the independent Nextora configuration UI. No build/download or driver changes.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/configure-native-wsl.sh [options]

  --bin PATH           Native Linux nextora-engine (default: target/release/nextora)
  --windows-exe PATH   Windows nextora.exe, given as a WSL path (/mnt/c/...)
  --models-dir PATH    Optional model folder; defaults to /mnt/f when mounted
  --settings-file PATH Persistent JSON profile, given as a Linux/WSL path
  --port NUMBER        Loopback port (default: 9070)
  --no-browser         Print the address without opening a browser

Native Linux stores settings in $XDG_CONFIG_HOME/nextora/native-service.json
(or $HOME/.config/nextora/native-service.json). A Windows executable launched
from WSL uses native-service.json beside the executable unless --settings-file
is supplied. Windows mode converts the model/profile paths with wslpath.

Examples:
  bash scripts/configure-native-wsl.sh --models-dir /mnt/f/Models
  bash scripts/configure-native-wsl.sh \
    --windows-exe /mnt/c/code/nextora/target/release/nextora.exe \
    --models-dir /mnt/f/Models

New profiles use auto: a compatible hardware Vulkan adapter, otherwise CPU
with the reason reported in the UI. Existing profiles keep their settings.
Windows mode runs inference on Windows; native Linux mode uses only devices
exposed to WSL. Auto on an llvmpipe-only WSL setup selects real CPU execution.
EOF
}

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
binary="$project_root/target/release/nextora"
windows_mode=false
models_dir=''
settings_file=''
port=9070
open_browser=true
native_binary_chosen=false
need_value() { [[ $# -ge 2 && -n "$2" ]] || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }; }
while (( $# )); do
    case "$1" in
        --bin) need_value "$@"; binary=$2; native_binary_chosen=true; shift 2 ;;
        --windows-exe) need_value "$@"; binary=$2; windows_mode=true; shift 2 ;;
        --models-dir) need_value "$@"; models_dir=$2; shift 2 ;;
        --settings-file) need_value "$@"; settings_file=$2; shift 2 ;;
        --port) need_value "$@"; port=$2; shift 2 ;;
        --no-browser) open_browser=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
if $windows_mode && $native_binary_chosen; then
    printf '%s\n' 'Choose either --bin or --windows-exe.' >&2
    exit 2
fi
[[ "$port" =~ ^[0-9]{1,5}$ ]] && (( 10#$port >= 1 && 10#$port <= 65535 )) || {
    printf '%s\n' '--port must be between 1 and 65535.' >&2; exit 2;
}
[[ -f "$binary" && -x "$binary" ]] || {
    printf 'Executable not found: %s\nBuild the independent engine with: cargo build --release --locked -p nextora-engine\n' "$binary" >&2
    exit 1
}
binary=$(realpath -- "$binary")
if [[ -z "$models_dir" && -d /mnt/f ]]; then models_dir=/mnt/f; fi
if [[ -n "$models_dir" ]]; then
    [[ -d "$models_dir" ]] || { printf 'Model folder does not exist: %s\n' "$models_dir" >&2; exit 1; }
    models_dir=$(realpath -- "$models_dir")
fi

if $windows_mode; then
    [[ "$binary" == *.exe ]] || { printf '%s\n' '--windows-exe must name a Windows .exe.' >&2; exit 2; }
    command -v wslpath >/dev/null 2>&1 || { printf '%s\n' 'Run Windows mode in WSL with Windows interoperability enabled.' >&2; exit 1; }
    windows_binary=$(wslpath -w "$binary")
    [[ "$windows_binary" != \\\\* ]] || {
        printf '%s\n' 'Use an executable built in a Windows-local folder, for example /mnt/c/code/nextora-cli, rather than the WSL UNC share.' >&2
        exit 1
    }
    if [[ -z "$settings_file" ]]; then settings_file="$(dirname -- "$binary")/native-service.json"; fi
else
    [[ "$binary" != *.exe ]] || { printf '%s\n' 'Use --windows-exe for a Windows executable.' >&2; exit 2; }
    if [[ -z "$settings_file" ]]; then
        settings_file="${XDG_CONFIG_HOME:-${HOME}/.config}/nextora/native-service.json"
    fi
fi
settings_file=$(realpath -m -- "$settings_file")
mkdir -p -- "$(dirname -- "$settings_file")"
printf 'Nextora configuration: http://127.0.0.1:%s/\nSettings: %s\n' "$port" "$settings_file"
if $windows_mode; then
    settings_file=$(wslpath -w "$settings_file")
    if [[ -n "$models_dir" ]]; then models_dir=$(wslpath -w "$models_dir"); fi
    printf '%s\n' 'Running the Windows inference process. Use the Windows browser address above.'
else
    printf '%s\n' 'Running the native Linux inference process. Scan for models and choose a backend in the UI.'
fi
engine_args=(serve --backend auto --settings-file "$settings_file" --port "$port")
if [[ -n "$models_dir" ]]; then engine_args+=(--models-dir "$models_dir"); fi
if $open_browser; then engine_args+=(--open-ui); fi
exec "$binary" "${engine_args[@]}"
