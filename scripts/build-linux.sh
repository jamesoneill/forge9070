#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if ! command -v cargo >/dev/null 2>&1; then
    printf '%s\n' 'Install Rust from https://rustup.rs and a C build toolchain, then retry.' >&2
    exit 1
fi
cargo build --release --locked
./target/release/forge9070 doctor

