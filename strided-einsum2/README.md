
## strided-einsum2 (binary einsum)

The `strided-einsum2` crate provides `einsum2_into` for binary tensor contractions. Benchmarks (Rust and Julia OMEinsum reference scripts) live in `strided-einsum2/benches/`.

**Run all benchmarks (Rust + Julia, single-threaded)** from repo root:

```bash
bash strided-einsum2/benches/run_all.sh
```

**Rust benchmarks** (from repo root; use `--manifest-path` so only strided-einsum2 benches run):

```bash
RAYON_NUM_THREADS=1 cargo bench
# or individually, e.g.:
cargo bench --bench matmul
cargo bench --bench dot
# ... (manyinds, batchmul, trace, ptrace, diag, perm, tcontract, star, starandcontract, indexsum, hadamard, outer)
```

**Julia reference scripts** (e.g. `julia_matmul.jl`, `julia_dot.jl`) use OMEinsum. Run single-threaded for comparison (from repo root):

```bash
OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=strided-einsum2/benches strided-einsum2/benches/julia_<name>.jl
```

Example: `julia_matmul.jl`, `julia_dot.jl`, `julia_trace.jl`, `julia_tcontract.jl`, `julia_outer.jl`, etc.

### Benchmark results

Environment: Apple Silicon M2, single-threaded. Rust: `RAYON_NUM_THREADS=1 cargo bench --bench <name> --manifest-path strided-einsum2/Cargo.toml`. Julia: `OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=strided-einsum2/benches strided-einsum2/benches/julia_<name>.jl`. Mean time (ms).

| Case | Julia OMEinsum (ms) | Rust strided-einsum2 (ms) |
|---|---:|---:|
| matmul (1) square 1000×1000 Float64 | 39.3 | 40.0 |
| matmul (1) square 1000×1000 ComplexF64 | 195.4 | 214 |
| matmul (2) (2000,50)×(50,2000) Float64 | 8.77 | 8.5 |
| matmul (3) (50,2000)×(2000,50) Float64 | 0.26 | 0.27 |
| batchmul (1) square b=3 1000³ Float64 | 119 | 119 |
| batchmul (1) square b=3 1000³ ComplexF64 | 597 | 632 |
| dot (1) square 100³ Float64 | 5.44 | 0.21 |
| dot (1) square 100³ ComplexF64 | 4.82 | 0.57 |
| trace 1000×1000 Float64 | 0.0028 | 0.006 |
| trace 1000×1000 ComplexF64 | 0.0036 | 0.006 |
| ptrace (100,100,100)→(100) Float64 | 0.019 | 0.062 |
| ptrace (100,100,100)→(100) ComplexF64 | 0.028 | 0.064 |
| diag (100,100,100)→(100,100) Float64 | 0.015 | 0.062 |
| diag (100,100,100)→(100,100) ComplexF64 | 0.021 | 0.066 |
| perm (30,30,30,30) Float64 | 0.63 | 0.52 |
| perm (30,30,30,30) ComplexF64 | 0.73 | 0.85 |
| tcontract (1) square 30³ Float64 | 0.068 | 0.078 |
| tcontract (1) square 30³ ComplexF64 | 0.23 | 0.31 |
| star (50,50) Float64 | 2.48 | 111 |
| star (50,50) ComplexF64 | 6.26 | 111 |
| starandcontract (50,50) Float64 | 0.103 | 4.4 |
| starandcontract (50,50) ComplexF64 | 0.122 | 3.3 |
| indexsum (100,100,100)→(100,100) Float64 | 0.16 | 7.1 |
| indexsum (100,100,100)→(100,100) ComplexF64 | 1.26 | 7.4 |
| hadamard (100,100,100) Float64 | 0.20 | 0.25 |
| hadamard (100,100,100) ComplexF64 | 0.67 | 0.82 |
| outer (1) square 40⁴ Float64 | 1.06 | 0.48 |
| outer (1) square 40⁴ ComplexF64 | 2.18 | 0.95 |
| manyinds (12+12→13 indices, dim 2) Float64 | 0.063 | 0.12 |
| manyinds (12+12→13 indices, dim 2) ComplexF64 | 0.126 | 0.18 |

Notes:
- Julia results from the corresponding `julia_<name>.jl` scripts (BenchmarkTools mean). Rust from `cargo bench --bench <name>` (mean of 3–5 runs).
- All use column-major layout for parity. "—" means the Julia script was not run for that row; run the matching `julia_<name>.jl` to compare.
- Some benchmarks (matmul, batchmul, dot, tcontract, outer) define three shape cases: (1) square, (2) n1=n3≫n2, (3) n1=n3≪n2; only representative cases are shown above.
- Outer product uses smaller sizes (40⁴, (80,20)×(20,80), (20,80)×(80,20)) to keep memory use moderate.
