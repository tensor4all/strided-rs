#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo " Single-threaded benchmarks (strided-einsum2 vs OMEinsum)"
echo "============================================================"
echo ""

echo "--- Rust (strided-einsum2, single-threaded) ---"
RAYON_NUM_THREADS=1 cargo bench --manifest-path "$PROJECT_DIR/Cargo.toml" 2>&1 \
    | grep -v "^$\|Compiling\|Finished\|Running"
echo ""

echo "--- Julia (OMEinsum, single-threaded) ---"
if command -v julia >/dev/null 2>&1; then
    for jl in "$SCRIPT_DIR"/julia_*.jl; do
        [ -f "$jl" ] || continue
        name="$(basename "$jl" .jl)"
        echo "  [$name]"
        OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project="$SCRIPT_DIR" "$jl" 2>&1 || true
        echo ""
    done
else
    echo "julia not found; skipping Julia benchmarks."
fi
echo ""

echo "============================================================"
echo " Done"
echo "============================================================"
