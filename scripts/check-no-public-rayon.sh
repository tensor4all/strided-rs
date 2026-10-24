#!/usr/bin/env bash
# Fails when a scheduler (rayon) path appears in the public API of strided-exec.
# Uses `cargo public-api` (cargo install cargo-public-api; needs a nightly
# toolchain for rustdoc JSON). Skips with a notice when it is not installed.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! cargo public-api --version >/dev/null 2>&1; then
    echo "cargo-public-api not installed; skipping (install: cargo install cargo-public-api)" >&2
    exit 0
fi
api=$(cargo public-api -p strided-exec --all-features)
if grep -n 'rayon::' <<<"$api"; then
    echo "error: rayon appears in the public API of strided-exec" >&2
    exit 1
fi
echo "ok: no rayon in the public API of strided-exec"
