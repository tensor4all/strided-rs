#!/usr/bin/env bash
# Fails when a scheduler (rayon) path appears in the public API of strided-exec.
# Uses `cargo public-api` (cargo install cargo-public-api; needs a nightly
# toolchain for rustdoc JSON).
set -euo pipefail
cd "$(dirname "$0")/.."
api=$(cargo public-api -p strided-exec --all-features)
if grep -n 'rayon::' <<<"$api"; then
    echo "error: rayon appears in the public API of strided-exec" >&2
    exit 1
fi
echo "ok: no rayon in the public API of strided-exec"
