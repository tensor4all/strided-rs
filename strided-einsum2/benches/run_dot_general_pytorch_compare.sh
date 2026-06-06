#!/usr/bin/env bash
set -euo pipefail

THREAD_COUNTS=("$@")
if (( ${#THREAD_COUNTS[@]} == 0 )); then
    THREAD_COUNTS=(1 4)
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CRATE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$CRATE_DIR/.." && pwd)"
PROFILE="${STRIDED_EINSUM2_DOT_GENERAL_BENCH_PROFILE:-full}"
RUNS="${STRIDED_EINSUM2_DOT_GENERAL_BENCH_RUNS:-15}"
WARMUPS="${STRIDED_EINSUM2_DOT_GENERAL_BENCH_WARMUPS:-3}"
DTYPES="${STRIDED_EINSUM2_DOT_GENERAL_BENCH_DTYPES:-f64}"
RUST_FEATURES="${STRIDED_EINSUM2_DOT_GENERAL_RUST_FEATURES:-}"
RUST_NO_DEFAULT_FEATURES="${STRIDED_EINSUM2_DOT_GENERAL_RUST_NO_DEFAULT_FEATURES:-}"
THREAD_LABEL="$(IFS=_; echo "${THREAD_COUNTS[*]}")"
DTYPE_LABEL="${DTYPES//,/_}"
OUT="${STRIDED_EINSUM2_DOT_GENERAL_BENCH_OUTPUT:-$CRATE_DIR/target/dot_general_pytorch_compare_t${THREAD_LABEL}_${PROFILE}_${DTYPE_LABEL}.csv}"
REQUIRE_ACCELERATE="${STRIDED_EINSUM2_DOT_GENERAL_REQUIRE_ACCELERATE:-}"

export STRIDED_EINSUM2_DOT_GENERAL_BENCH_PROFILE="$PROFILE"
export STRIDED_EINSUM2_DOT_GENERAL_BENCH_RUNS="$RUNS"
export STRIDED_EINSUM2_DOT_GENERAL_BENCH_WARMUPS="$WARMUPS"
export STRIDED_EINSUM2_DOT_GENERAL_BENCH_DTYPES="$DTYPES"

if [[ -z "$RUST_FEATURES" ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
        RUST_FEATURES="parallel,blas-accelerate"
        RUST_NO_DEFAULT_FEATURES=1
    else
        RUST_FEATURES="parallel"
    fi
fi

if [[ -z "$REQUIRE_ACCELERATE" ]]; then
    REQUIRE_ACCELERATE=0
    RUST_FEATURES_COMPACT="${RUST_FEATURES//[[:space:]]/}"
    if [[ "$(uname -s)" == "Darwin" ]] \
        && ([[ ",$RUST_FEATURES_COMPACT," == *",blas-accelerate,"* ]] \
            || ([[ ",$RUST_FEATURES_COMPACT," == *",blas,"* ]] \
                && [[ ",$RUST_FEATURES_COMPACT," != *",blas-openblas,"* ]] \
                && [[ ",$RUST_FEATURES_COMPACT," != *",blas-mkl,"* ]] \
                && [[ ",$RUST_FEATURES_COMPACT," != *",blas-inject,"* ]])); then
        REQUIRE_ACCELERATE=1
    fi
fi

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
    local rust_bench_bin
    local rust_build_log
    local rust_bench_log
    local cargo_args=(
        bench
        --manifest-path "$WORKSPACE_DIR/Cargo.toml"
        -p strided-einsum2
        --bench dot_general_pytorch_compare
        --features "$RUST_FEATURES"
    )
    if [[ -n "$RUST_NO_DEFAULT_FEATURES" ]]; then
        cargo_args+=(--no-default-features)
    fi

    rust_build_log="$(mktemp)"
    cargo "${cargo_args[@]}" --no-run >"$rust_build_log" 2>&1
    rust_bench_bin="$(awk -F'[()]' '/Executable .*dot_general_pytorch_compare/ {print $(NF - 1)}' "$rust_build_log" | tail -n 1)"
    rm -f "$rust_build_log"

    if [[ "$REQUIRE_ACCELERATE" == "1" ]]; then
        if [[ -z "$rust_bench_bin" || ! -x "$rust_bench_bin" ]]; then
            echo "error: could not locate dot_general_pytorch_compare bench binary for Accelerate verification" >&2
            exit 1
        fi
        if ! command -v otool >/dev/null 2>&1; then
            echo "error: otool is required to verify Accelerate linkage on macOS" >&2
            exit 1
        fi
        if ! otool -L "$rust_bench_bin" | grep -q 'Accelerate.framework'; then
            echo "error: macOS benchmark requires Accelerate, but the binary is not linked against it; this is a regression" >&2
            otool -L "$rust_bench_bin" >&2
            exit 1
        fi
    fi

    rust_bench_log="$(mktemp)"
    cargo "${cargo_args[@]}" >"$rust_bench_log"

    if [[ -s "$OUT" ]]; then
        awk '/^dot_general,/' "$rust_bench_log" >>"$OUT"
    else
        awk '/^suite,/ || /^dot_general,/' "$rust_bench_log" >"$OUT"
    fi
    rm -f "$rust_bench_log"
}

run_pytorch() {
    local threads="${1:?threads required}"
    if command -v uv >/dev/null 2>&1; then
        uv run --with torch --with numpy python "$SCRIPT_DIR/dot_general_pytorch_compare.py" \
            --num-threads "$threads" \
            --profile "$PROFILE" \
            --runs "$RUNS" \
            --warmups "$WARMUPS" \
            --dtypes "$DTYPES" \
            --output "$OUT"
    else
        python3 "$SCRIPT_DIR/dot_general_pytorch_compare.py" \
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
    run_rust
    run_pytorch "$num_threads"
done

echo "dot_general comparison results saved to: $OUT"
