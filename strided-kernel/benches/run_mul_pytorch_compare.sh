#!/usr/bin/env bash
set -euo pipefail

THREAD_COUNTS=("$@")
if (( ${#THREAD_COUNTS[@]} == 0 )); then
    THREAD_COUNTS=(1 4)
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CRATE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$CRATE_DIR/.." && pwd)"
PROFILE="${STRIDED_KERNEL_MUL_BENCH_PROFILE:-full}"
RUNS="${STRIDED_KERNEL_MUL_BENCH_RUNS:-15}"
WARMUPS="${STRIDED_KERNEL_MUL_BENCH_WARMUPS:-3}"
DTYPES="${STRIDED_KERNEL_MUL_BENCH_DTYPES:-f64}"
THREAD_LABEL="$(IFS=_; echo "${THREAD_COUNTS[*]}")"
DTYPE_LABEL="${DTYPES//,/_}"
OUT="${STRIDED_KERNEL_MUL_BENCH_OUTPUT:-$CRATE_DIR/target/mul_pytorch_compare_t${THREAD_LABEL}_${PROFILE}_${DTYPE_LABEL}.csv}"

export STRIDED_KERNEL_MUL_BENCH_PROFILE="$PROFILE"
export STRIDED_KERNEL_MUL_BENCH_RUNS="$RUNS"
export STRIDED_KERNEL_MUL_BENCH_WARMUPS="$WARMUPS"
export STRIDED_KERNEL_MUL_BENCH_DTYPES="$DTYPES"

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

configure_cpu_thread_env() {
    local threads="${1:?threads required}"
    local xla_multi_thread="false"
    if [[ "$threads" =~ ^[0-9]+$ ]] && (( threads > 1 )); then
        xla_multi_thread="true"
    fi

    export OMP_NUM_THREADS="$threads"
    export OMP_THREAD_LIMIT="$threads"
    export OMP_DYNAMIC=FALSE
    export RAYON_NUM_THREADS="$threads"
    export OPENBLAS_NUM_THREADS="$threads"
    export GOTO_NUM_THREADS="$threads"
    export MKL_NUM_THREADS="$threads"
    export VECLIB_MAXIMUM_THREADS="$threads"
    export VECLIB_NUM_THREADS="$threads"
    export NUMEXPR_NUM_THREADS="$threads"
    export BLIS_NUM_THREADS="$threads"
    export XLA_FLAGS="--xla_cpu_multi_thread_eigen=${xla_multi_thread} intra_op_parallelism_threads=${threads}"
}

run_rust() {
    local threads="${1:?threads required}"
    local rust_bench_log
    rust_bench_log="$(mktemp)"
    cargo bench \
        --manifest-path "$WORKSPACE_DIR/Cargo.toml" \
        -p strided-kernel \
        --bench mul_pytorch_compare \
        --features parallel \
        >"$rust_bench_log"

    if [[ -s "$OUT" ]]; then
        awk '/^mul,/' "$rust_bench_log" >>"$OUT"
    else
        awk '/^suite,/ || /^mul,/' "$rust_bench_log" >"$OUT"
    fi
    rm -f "$rust_bench_log"
}

run_pytorch() {
    local threads="${1:?threads required}"
    if command -v uv >/dev/null 2>&1; then
        uv run --with torch --with numpy python "$SCRIPT_DIR/mul_pytorch_compare.py" \
            --num-threads "$threads" \
            --profile "$PROFILE" \
            --runs "$RUNS" \
            --warmups "$WARMUPS" \
            --dtypes "$DTYPES" \
            --output "$OUT"
    else
        python3 "$SCRIPT_DIR/mul_pytorch_compare.py" \
            --num-threads "$threads" \
            --profile "$PROFILE" \
            --runs "$RUNS" \
            --warmups "$WARMUPS" \
            --dtypes "$DTYPES" \
            --output "$OUT"
    fi
}

for num_threads in "${THREAD_COUNTS[@]}"; do
    configure_cpu_thread_env "$num_threads"
    run_rust "$num_threads"
    run_pytorch "$num_threads"
done

echo "mul comparison results saved to: $OUT"
