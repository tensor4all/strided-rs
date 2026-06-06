
## strided-einsum2 (binary einsum)

The `strided-einsum2` crate provides `einsum2_into` for binary tensor contractions.

Most Rust benchmark runners were moved to `strided-opteinsum/benches/`.
Latest broad benchmark results are documented in `strided-opteinsum/README.md`.

The `dot_general` batched matmul diagnostic compares the tenferro-benchmark
`bij,bjk->bik` cases against PyTorch `bmm` with memory-matched row-major and
col-major layouts. Allocation and setup are outside the timed loop: Rust
prepares GEMM operands once, and PyTorch uses `torch.bmm(..., out=...)`.
On macOS, the runner uses `blas-accelerate` by default and verifies the
benchmark binary with `otool -L`. Losing this Accelerate path is a benchmark
regression. Use `STRIDED_EINSUM2_DOT_GENERAL_RUST_FEATURES` to test another
provider such as `parallel,blas-openblas` or `parallel,blas-mkl`.
Set `STRIDED_EINSUM2_DOT_GENERAL_BENCH_DIAGNOSTICS=1` to emit extra Rust-side
diagnostic rows (`raw-cblas-dgemm`, `raw-trait-dgemm`, and on macOS
`raw-fortran-dgemm`) for separating BLAS call overhead from PyTorch `bmm`.
On macOS without MKL, PyTorch CPU `bmm` falls back to per-batch `addmm` rather
than MKL batched GEMM; current 1T Accelerate runs do not show a stable TN-layout
loss once measured with enough repetitions.

```bash
STRIDED_EINSUM2_DOT_GENERAL_BENCH_PROFILE=full \
STRIDED_EINSUM2_DOT_GENERAL_BENCH_DTYPES=f64,c64,c128 \
bash strided-einsum2/benches/run_dot_general_pytorch_compare.sh 1 4
```

Latest local allocation-free Accelerate run (`runs=30`, `warmups=5`):

| benchmark | dtype | threads | strided-einsum2-accelerate-prepared ms | pytorch-bmm ms | ratio |
|---|---:|---:|---:|---:|---:|
| `bin_batched_matmul_b32_m64_n64_k64` | f64 | 1 | 0.046375 | 0.077375 | 0.60x |
| `bin_batched_matmul_b32_m64_n64_k64` | f64 | 4 | 0.046417 | 0.078521 | 0.59x |
| `bin_batched_matmul_b32_m64_n64_k64` | c64 | 1 | 0.343666 | 0.332021 | 1.04x |
| `bin_batched_matmul_b32_m64_n64_k64` | c64 | 4 | 0.342417 | 0.332604 | 1.03x |
| `bin_batched_matmul_b32_m64_n64_k64` | c128 | 1 | 0.748334 | 0.760542 | 0.98x |
| `bin_batched_matmul_b32_m64_n64_k64` | c128 | 4 | 0.755875 | 0.758333 | 1.00x |
| `bin_batched_matmul_b32_m128_n128_k128` | f64 | 1 | 0.301042 | 0.376313 | 0.80x |
| `bin_batched_matmul_b32_m128_n128_k128` | f64 | 4 | 0.307916 | 0.378687 | 0.81x |
| `bin_batched_matmul_b32_m128_n128_k128` | c64 | 1 | 1.224958 | 1.251854 | 0.98x |
| `bin_batched_matmul_b32_m128_n128_k128` | c64 | 4 | 1.238375 | 1.243312 | 1.00x |
| `bin_batched_matmul_b32_m128_n128_k128` | c128 | 1 | 2.981833 | 2.979667 | 1.00x |
| `bin_batched_matmul_b32_m128_n128_k128` | c128 | 4 | 2.999375 | 2.977917 | 1.01x |

The faer-prepared path is useful for isolating non-BLAS behavior, but macOS
regression checks should use the Accelerate path above.

**Julia reference scripts** (e.g. `julia_matmul.jl`, `julia_dot.jl`) use OMEinsum. Run single-threaded for comparison (from repo root):

```bash
OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=strided-einsum2/benches strided-einsum2/benches/julia_<name>.jl
```

Example: `julia_matmul.jl`, `julia_dot.jl`, `julia_trace.jl`, `julia_tcontract.jl`, `julia_outer.jl`, etc.
