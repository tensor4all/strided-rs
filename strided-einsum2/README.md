
## strided-einsum2 (binary einsum)

The `strided-einsum2` crate provides `einsum2_into` for binary tensor contractions.

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
than MKL batched GEMM.

```bash
STRIDED_EINSUM2_DOT_GENERAL_BENCH_PROFILE=full \
STRIDED_EINSUM2_DOT_GENERAL_BENCH_DTYPES=f64,c64,c128 \
bash strided-einsum2/benches/run_dot_general_pytorch_compare.sh 1 4
```

Published benchmark programs and current measured results live in
[`strided-rs-benchmark-suite`](https://github.com/tensor4all/strided-rs-benchmark-suite).

**Julia reference scripts** (e.g. `julia_matmul.jl`, `julia_dot.jl`) use OMEinsum. Run single-threaded for comparison (from repo root):

```bash
OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=strided-einsum2/benches strided-einsum2/benches/julia_<name>.jl
```

Example: `julia_matmul.jl`, `julia_dot.jl`, `julia_trace.jl`, `julia_tcontract.jl`, `julia_outer.jl`, etc.
