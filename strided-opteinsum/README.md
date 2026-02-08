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

Benchmark results were re-measured on February 8, 2026.
Environment: Apple Silicon M2, single-threaded. Mean time (ms).

| Case | Julia OMEinsum (ms) | Rust strided-opteinsum (ms) |
|---|---:|---:|
| matmul (1) square 1000×1000 Float64 | 41.433 | 41.520 |
| matmul (1) square 1000×1000 ComplexF64 | 204.183 | 225.949 |
| matmul (2) (2000,50)×(50,2000) Float64 | 11.446 | 9.519 |
| matmul (2) (2000,50)×(50,2000) ComplexF64 | 44.618 | 44.645 |
| matmul (3) (50,2000)×(2000,50) Float64 | 0.267 | 0.309 |
| matmul (3) (50,2000)×(2000,50) ComplexF64 | 1.418 | 1.297 |
| batchmul (1) square b=3 1000³ Float64 | 127.743 | 126.615 |
| batchmul (1) square b=3 1000³ ComplexF64 | 603.465 | 675.871 |
| batchmul (2) b=3 (2000,50)×(50,2000) Float64 | 30.294 | 27.938 |
| batchmul (2) b=3 (2000,50)×(50,2000) ComplexF64 | 129.160 | 134.802 |
| batchmul (3) b=3 (50,2000)×(2000,50) Float64 | 0.780 | 0.905 |
| batchmul (3) b=3 (50,2000)×(2000,50) ComplexF64 | 3.280 | 4.082 |
| dot (1) square 100³ Float64 | 5.382 | 0.325 |
| dot (1) square 100³ ComplexF64 | 4.805 | 0.974 |
| dot (2) (2000,50,2000) Float64 | 1073.075 | 56.802 |
| dot (2) (2000,50,2000) ComplexF64 | 961.889 | 151.283 |
| dot (3) (50,2000,50) Float64 | 26.836 | 2.320 |
| dot (3) (50,2000,50) ComplexF64 | 24.051 | 3.509 |
| trace 1000×1000 Float64 | 0.003 | 0.002 |
| trace 1000×1000 ComplexF64 | 0.004 | 0.002 |
| ptrace (100,100,100)→(100) Float64 | 0.019 | 0.009 |
| ptrace (100,100,100)→(100) ComplexF64 | 0.029 | 0.008 |
| diag (100,100,100)→(100,100) Float64 | 0.015 | 0.007 |
| diag (100,100,100)→(100,100) ComplexF64 | 0.021 | 0.012 |
| perm (30,30,30,30) Float64 | 0.576 | 0.002 |
| perm (30,30,30,30) ComplexF64 | 0.758 | 0.001 |
| tcontract (1) square 30³ Float64 | 0.072 | 0.069 |
| tcontract (1) square 30³ ComplexF64 | 0.220 | 0.232 |
| tcontract (2) (2000,50,50)×(50,2000,50) Float64 | 399.641 | 423.274 |
| tcontract (2) (2000,50,50)×(50,2000,50) ComplexF64 | 1987.545 | 2273.855 |
| tcontract (3) (50,2000,2000)×(2000,50,2000) Float64 | 690.430 | 739.603 |
| tcontract (3) (50,2000,2000)×(2000,50,2000) ComplexF64 | 5005.792 | 3524.387 |
| star (50,50) Float64 | 2.552 | 0.526 |
| star (50,50) ComplexF64 | 6.400 | 1.986 |
| starandcontract (50,50) Float64 | 0.112 | 0.029 |
| starandcontract (50,50) ComplexF64 | 0.127 | 0.041 |
| indexsum (100,100,100)→(100,100) Float64 | 0.162 | 0.825 |
| indexsum (100,100,100)→(100,100) ComplexF64 | 1.265 | 0.930 |
| hadamard (100,100,100) Float64 | 0.195 | 0.451 |
| hadamard (100,100,100) ComplexF64 | 0.695 | 1.311 |
| outer (1) square 40⁴ Float64 | 1.470 | 0.679 |
| outer (1) square 40⁴ ComplexF64 | 2.262 | 1.553 |
| outer (2) (80,20)×(20,80) Float64 | 1.224 | 0.671 |
| outer (2) (80,20)×(20,80) ComplexF64 | 2.260 | 1.523 |
| outer (3) (20,80)×(80,20) Float64 | 1.164 | 0.685 |
| outer (3) (20,80)×(80,20) ComplexF64 | 2.262 | 1.508 |
| manyinds (12+12→13 indices, dim 2) Float64 | 0.065 | 0.096 |
| manyinds (12+12→13 indices, dim 2) ComplexF64 | 0.128 | 0.155 |

## Notes

- Current parser accepts ASCII lowercase index labels.
- Generative output notation such as `->ii` is tracked in GitHub issues.
