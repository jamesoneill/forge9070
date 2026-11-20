#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if ! command -v cargo >/dev/null 2>&1; then
    printf '%s\n' 'Install the pinned Rust toolchain first.' >&2
    exit 1
fi
cargo build --release --locked -p nextora-engine -p nextora-cli
./target/release/nextora doctor
