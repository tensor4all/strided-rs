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
| matmul (1) square 1000×1000 Float64 | 40.121 | 45.426 |
| matmul (1) square 1000×1000 ComplexF64 | 201.803 | 240.058 |
| matmul (2) (2000,50)×(50,2000) Float64 | 9.613 | 9.716 |
| matmul (2) (2000,50)×(50,2000) ComplexF64 | 44.214 | 48.040 |
| matmul (3) (50,2000)×(2000,50) Float64 | 0.262 | 0.306 |
| matmul (3) (50,2000)×(2000,50) ComplexF64 | 1.098 | 1.399 |
| batchmul (1) square b=3 1000³ Float64 | 120.510 | 134.531 |
| batchmul (1) square b=3 1000³ ComplexF64 | 606.124 | 720.396 |
| batchmul (2) b=3 (2000,50)×(50,2000) Float64 | 28.576 | 30.488 |
| batchmul (2) b=3 (2000,50)×(50,2000) ComplexF64 | 130.857 | 142.431 |
| batchmul (3) b=3 (50,2000)×(2000,50) Float64 | 0.778 | 0.972 |
| batchmul (3) b=3 (50,2000)×(2000,50) ComplexF64 | 3.344 | 4.720 |
| dot (1) square 100³ Float64 | 5.459 | 0.252 |
| dot (1) square 100³ ComplexF64 | 4.872 | 0.663 |
| dot (2) (2000,50,2000) Float64 | 1096.416 | 60.504 |
| dot (2) (2000,50,2000) ComplexF64 | 978.969 | 137.299 |
| dot (3) (50,2000,50) Float64 | 27.319 | 1.552 |
| dot (3) (50,2000,50) ComplexF64 | 24.567 | 3.525 |
| trace 1000×1000 Float64 | 0.003 | 0.002 |
| trace 1000×1000 ComplexF64 | 0.004 | 0.001 |
| ptrace (100,100,100)→(100) Float64 | 0.020 | 0.008 |
| ptrace (100,100,100)→(100) ComplexF64 | 0.029 | 0.013 |
| diag (100,100,100)→(100,100) Float64 | 0.016 | 0.010 |
| diag (100,100,100)→(100,100) ComplexF64 | 0.023 | 0.015 |
| perm (30,30,30,30) Float64 | 0.567 | 0.002 |
| perm (30,30,30,30) ComplexF64 | 0.835 | 0.001 |
| tcontract (1) square 30³ Float64 | 0.073 | 0.070 |
| tcontract (1) square 30³ ComplexF64 | 0.225 | 0.220 |
| tcontract (2) (2000,50,50)×(50,2000,50) Float64 | 405.114 | 457.123 |
| tcontract (2) (2000,50,50)×(50,2000,50) ComplexF64 | 1985.169 | 2416.238 |
| tcontract (3) (50,2000,2000)×(2000,50,2000) Float64 | 719.871 | 786.455 |
| tcontract (3) (50,2000,2000)×(2000,50,2000) ComplexF64 | 4017.383 | 3056.063 |
| star (50,50) Float64 | 2.540 | 0.551 |
| star (50,50) ComplexF64 | 6.403 | 1.847 |
| starandcontract (50,50) Float64 | 0.111 | 0.037 |
| starandcontract (50,50) ComplexF64 | 0.126 | 0.026 |
| indexsum (100,100,100)→(100,100) Float64 | 0.163 | 0.718 |
| indexsum (100,100,100)→(100,100) ComplexF64 | 1.270 | 1.226 |
| hadamard (100,100,100) Float64 | 0.196 | 0.537 |
| hadamard (100,100,100) ComplexF64 | 0.755 | 1.510 |
| outer (1) square 40⁴ Float64 | 1.455 | 0.667 |
| outer (1) square 40⁴ ComplexF64 | 2.269 | 1.531 |
| outer (2) (80,20)×(20,80) Float64 | 1.309 | 0.661 |
| outer (2) (80,20)×(20,80) ComplexF64 | 2.272 | 1.551 |
| outer (3) (20,80)×(80,20) Float64 | 1.217 | 0.714 |
| outer (3) (20,80)×(80,20) ComplexF64 | 2.281 | 1.499 |
| manyinds (12+12→13 indices, dim 2) Float64 | 0.067 | 0.206 |
| manyinds (12+12→13 indices, dim 2) ComplexF64 | 0.130 | 0.148 |

## Notes

- Current parser accepts ASCII lowercase index labels.
- Generative output notation such as `->ii` is tracked in GitHub issues.
