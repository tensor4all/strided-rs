
## strided-einsum2 (binary einsum)

The `strided-einsum2` crate provides `einsum2_into` for binary tensor contractions.

Rust benchmark runners were moved to `strided-opteinsum/benches/`.
Latest benchmark results are documented in `strided-opteinsum/README.md`.

**Julia reference scripts** (e.g. `julia_matmul.jl`, `julia_dot.jl`) use OMEinsum. Run single-threaded for comparison (from repo root):

```bash
OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=strided-einsum2/benches strided-einsum2/benches/julia_<name>.jl
```

Example: `julia_matmul.jl`, `julia_dot.jl`, `julia_trace.jl`, `julia_tcontract.jl`, `julia_outer.jl`, etc.
