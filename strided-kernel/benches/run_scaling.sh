#!/bin/bash
# Run scaling benchmarks for Rust and Julia at 1, 2, and 4 threads.
#
# Usage:
#   bash benches/run_scaling.sh          # default: 1 2 4 threads
#   bash benches/run_scaling.sh 1 8      # custom thread counts
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

THREADS="${@:-1 2 4}"

echo "============================================================"
echo " Scaling benchmarks: Rust (strided-rs) vs Julia (Strided.jl)"
echo " Thread counts: $THREADS"
echo "============================================================"
echo ""

# Pre-build Rust with parallel feature
cargo build --release --features parallel --manifest-path "$PROJECT_DIR/Cargo.toml" 2>&1 \
    | grep -v "^$"

for T in $THREADS; do
    echo ""
    echo "============================================================"
    echo " Threads: $T"
    echo "============================================================"
    echo ""

    echo "--- Rust (strided-rs, parallel feature) ---"
    RAYON_NUM_THREADS=$T cargo bench --features parallel --bench scaling_compare \
        --manifest-path "$PROJECT_DIR/Cargo.toml" 2>&1 \
        | grep -v "^$\|Compiling\|Finished\|Running\|Benchmarking"
    echo ""

    echo "--- Julia (Strided.jl) ---"
    JULIA_NUM_THREADS=$T julia --project="$PROJECT_DIR" "$SCRIPT_DIR/julia_scaling_compare.jl"
    echo ""
done
