#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo " Single-threaded benchmarks"
echo "============================================================"
echo ""

echo "--- Rust (strided-rs, single-threaded) ---"
RAYON_NUM_THREADS=1 cargo bench --bench rust_compare \
    --manifest-path "$PROJECT_DIR/Cargo.toml" 2>&1 \
    | grep -v "^$\|Compiling\|Finished\|Running\|Benchmarking"
echo ""

echo "--- Julia (Strided.jl, single-threaded) ---"
if command -v julia >/dev/null 2>&1; then
    JULIA_NUM_THREADS=1 julia --project="$SCRIPT_DIR" "$SCRIPT_DIR/julia_compare.jl"
else
    echo "julia not found; skipping Julia benchmarks."
fi
echo ""

echo "============================================================"
echo " Rank-25 tensor permutation benchmarks"
echo "============================================================"
echo ""

echo "--- Rust (strided-rs, single-threaded) ---"
RAYON_NUM_THREADS=1 cargo bench --bench rank25_permute \
    --manifest-path "$PROJECT_DIR/Cargo.toml" 2>&1 \
    | grep -v "^$\|Compiling\|Finished\|Running\|Benchmarking"
echo ""

echo "--- Julia (Strided.jl, single-threaded) ---"
if command -v julia >/dev/null 2>&1; then
    JULIA_NUM_THREADS=1 julia --project="$SCRIPT_DIR" "$SCRIPT_DIR/julia_rank25_compare.jl"
else
    echo "julia not found; skipping Julia benchmarks."
fi
echo ""

echo "============================================================"
echo " Multi-threaded benchmarks (Rust parallel feature)"
echo "============================================================"
echo ""

THREADS="${@:-1 2 4}"

cargo build --release --features parallel --manifest-path "$PROJECT_DIR/Cargo.toml" 2>&1 \
    | grep -v "^$"
echo ""

for T in $THREADS; do
    echo "------------------------------------------------------------"
    echo " Threads: $T"
    echo "------------------------------------------------------------"

    echo "--- Rust (strided-rs, parallel feature) ---"
    RAYON_NUM_THREADS=$T cargo bench --features parallel --bench threaded_compare \
        --manifest-path "$PROJECT_DIR/Cargo.toml" 2>&1 \
        | grep -v "^$\|Compiling\|Finished\|Running\|Benchmarking"
    echo ""

    echo "--- Julia (Strided.jl) ---"
    if command -v julia >/dev/null 2>&1; then
        JULIA_NUM_THREADS=$T julia --project="$SCRIPT_DIR" "$SCRIPT_DIR/julia_threaded_compare.jl"
    else
        echo "julia not found; skipping Julia benchmarks."
    fi
    echo ""
done
