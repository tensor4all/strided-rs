# strided-opteinsum

N-ary einsum frontend built on `strided-view`, `strided-kernel`, and `strided-einsum2`.

## Scope

- String parser for einsum with nested notation (example: `"(ij,jk),kl->il"`)
- Mixed `f64` / `Complex64` operands with promotion to complex when needed
- Single-tensor ops (permute/trace/partial-trace/diagonal extraction)
- 3+ tensor contraction-order optimization via `omeco`

## Quick Example

```rust
use strided_opteinsum::{einsum, EinsumOperand};
use strided_view::StridedArray;

let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| (idx[0] * 2 + idx[1] + 1) as f64);
let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| (idx[0] * 2 + idx[1] + 5) as f64);
let out = einsum("ij,jk->ik", vec![EinsumOperand::from(a), EinsumOperand::from(b)]).unwrap();
assert!(out.is_f64());
```

## Benchmarks

Run Rust benchmarks (single-threaded) from repo root:

```bash
RAYON_NUM_THREADS=1 cargo bench -p strided-opteinsum
# or one bench:
RAYON_NUM_THREADS=1 cargo bench -p strided-opteinsum --bench matmul
```

Run Julia OMEinsum references (single-threaded):

```bash
OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=strided-einsum2/benches strided-einsum2/benches/julia_matmul.jl
```

Benchmark results were re-measured on February 7, 2026.
Environment: Apple Silicon M2, single-threaded. Mean time (ms).

| Case | Julia OMEinsum (ms) | Rust strided-opteinsum (ms) |
|---|---:|---:|
| matmul (1) square 1000×1000 Float64 | 40.121 | 42.859 |
| matmul (1) square 1000×1000 ComplexF64 | 201.803 | 230.800 |
| matmul (2) (2000,50)×(50,2000) Float64 | 9.613 | 9.966 |
| matmul (2) (2000,50)×(50,2000) ComplexF64 | 44.214 | 46.754 |
| matmul (3) (50,2000)×(2000,50) Float64 | 0.262 | 0.351 |
| matmul (3) (50,2000)×(2000,50) ComplexF64 | 1.098 | 1.476 |
| batchmul (1) square b=3 1000³ Float64 | 120.510 | 130.148 |
| batchmul (1) square b=3 1000³ ComplexF64 | 606.124 | 683.173 |
| batchmul (2) b=3 (2000,50)×(50,2000) Float64 | 28.576 | 36.425 |
| batchmul (2) b=3 (2000,50)×(50,2000) ComplexF64 | 130.857 | 158.332 |
| batchmul (3) b=3 (50,2000)×(2000,50) Float64 | 0.778 | 1.060 |
| batchmul (3) b=3 (50,2000)×(2000,50) ComplexF64 | 3.344 | 4.223 |
| dot (1) square 100³ Float64 | 5.459 | 0.581 |
| dot (1) square 100³ ComplexF64 | 4.872 | 1.629 |
| dot (2) (2000,50,2000) Float64 | 1096.416 | 217.776 |
| dot (2) (2000,50,2000) ComplexF64 | 978.969 | 443.016 |
| dot (3) (50,2000,50) Float64 | 27.319 | 5.133 |
| dot (3) (50,2000,50) ComplexF64 | 24.567 | 9.589 |
| trace 1000×1000 Float64 | 0.003 | 0.356 |
| trace 1000×1000 ComplexF64 | 0.004 | 0.772 |
| ptrace (100,100,100)→(100) Float64 | 0.020 | 0.327 |
| ptrace (100,100,100)→(100) ComplexF64 | 0.029 | 0.700 |
| diag (100,100,100)→(100,100) Float64 | 0.016 | 0.495 |
| diag (100,100,100)→(100,100) ComplexF64 | 0.023 | 0.684 |
| perm (30,30,30,30) Float64 | 0.567 | 0.955 |
| perm (30,30,30,30) ComplexF64 | 0.835 | 2.540 |
| tcontract (1) square 30³ Float64 | 0.073 | 0.080 |
| tcontract (1) square 30³ ComplexF64 | 0.225 | 0.281 |
| tcontract (2) (2000,50,50)×(50,2000,50) Float64 | 405.114 | 429.890 |
| tcontract (2) (2000,50,50)×(50,2000,50) ComplexF64 | 1985.169 | 2264.214 |
| tcontract (3) (50,2000,2000)×(2000,50,2000) Float64 | 719.871 | 889.128 |
| tcontract (3) (50,2000,2000)×(2000,50,2000) ComplexF64 | 4017.383 | 3591.857 |
| star (50,50) Float64 | 2.540 | 0.544 |
| star (50,50) ComplexF64 | 6.403 | 1.741 |
| starandcontract (50,50) Float64 | 0.111 | 0.032 |
| starandcontract (50,50) ComplexF64 | 0.126 | 0.034 |
| indexsum (100,100,100)→(100,100) Float64 | 0.163 | 0.747 |
| indexsum (100,100,100)→(100,100) ComplexF64 | 1.270 | 1.078 |
| hadamard (100,100,100) Float64 | 0.196 | 45.906 |
| hadamard (100,100,100) ComplexF64 | 0.755 | 65.349 |
| outer (1) square 40⁴ Float64 | 1.455 | 0.599 |
| outer (1) square 40⁴ ComplexF64 | 2.269 | 1.476 |
| outer (2) (80,20)×(20,80) Float64 | 1.309 | 0.654 |
| outer (2) (80,20)×(20,80) ComplexF64 | 2.272 | 1.467 |
| outer (3) (20,80)×(80,20) Float64 | 1.217 | 0.654 |
| outer (3) (20,80)×(80,20) ComplexF64 | 2.281 | 1.488 |
| manyinds (12+12→13 indices, dim 2) Float64 | 0.067 | 0.089 |
| manyinds (12+12→13 indices, dim 2) ComplexF64 | 0.130 | 0.138 |

## Notes

- Current parser accepts ASCII lowercase index labels.
- Generative output notation such as `->ii` is tracked in GitHub issues.
