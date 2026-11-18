#!/usr/bin/env bash
# Prepare a native Linux build inside WSL. No GPU/ROCm/DKMS driver installation.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/setup-wsl.sh [--check] [--build] [--debug] [--install-deps]

Without options, check prerequisites only. --build compiles the native Linux
binary. --install-deps explicitly installs ordinary Ubuntu build tools, the
Vulkan loader and vulkaninfo; it never installs ROCm, DKMS or a GPU driver.
Install Rust from https://rustup.rs/ if it is missing.
EOF
}

build=0
debug=0
install=0
for option in "$@"; do
    case "$option" in
        --check) ;;
        --build) build=1 ;;
        --debug) debug=1 ;;
        --install-deps) install=1 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$option" >&2; usage >&2; exit 2 ;;
    esac
done

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
printf 'Forge9070 native Linux/WSL preparation\nProject: %s\n' "$project_root"
if [[ -r /proc/sys/kernel/osrelease ]]; then
    IFS= read -r kernel < /proc/sys/kernel/osrelease
    printf 'Kernel: %s\n' "$kernel"
fi
if [[ -e /dev/dxg ]]; then
    printf '%s\n' '/dev/dxg exists; this alone does not prove Vulkan compute support.'
fi

if (( install )); then
    command -v apt-get >/dev/null 2>&1 || { printf '%s\n' '--install-deps is for Ubuntu/Debian; install equivalent build tools manually.' >&2; exit 1; }
    if command -v dpkg >/dev/null 2>&1; then
        audit=$(dpkg --audit 2>&1) || { printf '%s\n' "$audit" >&2; exit 1; }
        if [[ -n "$audit" ]]; then
            printf '%s\n' 'dpkg has unfinished packages. Repair that state before installing build dependencies:' "$audit" >&2
            printf '%s\n' 'This script does not retry or remove a failed amdgpu-dkms package.' >&2
            exit 1
        fi
    fi
    elevation=()
    if (( EUID != 0 )); then
        command -v sudo >/dev/null 2>&1 || { printf '%s\n' 'sudo is required for --install-deps.' >&2; exit 1; }
        elevation=(sudo)
    fi
    "${elevation[@]}" apt-get update
    "${elevation[@]}" apt-get install -y --no-install-recommends \
        build-essential pkg-config ca-certificates libvulkan1 vulkan-tools
fi

cargo_bin=$(command -v cargo || true)
if [[ -z "$cargo_bin" && -x "${HOME}/.cargo/bin/cargo" ]]; then
    cargo_bin="${HOME}/.cargo/bin/cargo"
fi
missing=0
if [[ -z "$cargo_bin" ]]; then
    printf '%s\n' 'Missing Rust/Cargo: install Rust from https://rustup.rs/ and open a new shell.' >&2
    missing=1
fi
for tool in cc pkg-config; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Missing build tool: %s\n' "$tool" >&2
        missing=1
    fi
done
if (( missing )); then
    printf '%s\n' 'To install Ubuntu build dependencies explicitly, rerun with --install-deps.' >&2
    exit 1
fi
printf 'Cargo: %s\n' "$cargo_bin"
if ! command -v vulkaninfo >/dev/null 2>&1; then
    printf '%s\n' 'vulkaninfo is optional for diagnostics; Forge9070 doctor performs its own Vulkan discovery.'
fi
if (( ! build )); then
    printf '%s\n' 'Build prerequisites found. Use --build to compile; GPU support is not established by this check.'
    exit 0
fi

cd -- "$project_root"
build_args=(build --locked)
profile=debug
if (( ! debug )); then build_args+=(--release); profile=release; fi
"$cargo_bin" "${build_args[@]}"
"$project_root/target/$profile/forge9070" doctor
printf 'Built native Linux executable: %s/target/%s/forge9070\n' "$project_root" "$profile"
printf '%s\n' 'Next: scripts/serve-wsl.sh --backend cpu or --backend vulkan.'

