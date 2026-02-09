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
| matmul (1) square 1000×1000 Float64 | 41.433 | 42.311 |
| matmul (1) square 1000×1000 ComplexF64 | 204.183 | 238.002 |
| matmul (2) (2000,50)×(50,2000) Float64 | 11.446 | 9.059 |
| matmul (2) (2000,50)×(50,2000) ComplexF64 | 44.618 | 44.121 |
| matmul (3) (50,2000)×(2000,50) Float64 | 0.267 | 0.312 |
| matmul (3) (50,2000)×(2000,50) ComplexF64 | 1.418 | 1.325 |
| batchmul (1) square b=3 1000³ Float64 | 127.743 | 132.710 |
| batchmul (1) square b=3 1000³ ComplexF64 | 603.465 | 717.366 |
| batchmul (2) b=3 (2000,50)×(50,2000) Float64 | 30.294 | 28.087 |
| batchmul (2) b=3 (2000,50)×(50,2000) ComplexF64 | 129.160 | 138.139 |
| batchmul (3) b=3 (50,2000)×(2000,50) Float64 | 0.780 | 0.929 |
| batchmul (3) b=3 (50,2000)×(2000,50) ComplexF64 | 3.280 | 4.173 |
| dot (1) square 100³ Float64 | 5.382 | 0.295 |
| dot (1) square 100³ ComplexF64 | 4.805 | 0.639 |
| dot (2) (2000,50,2000) Float64 | 1073.075 | 60.099 |
| dot (2) (2000,50,2000) ComplexF64 | 961.889 | 134.398 |
| dot (3) (50,2000,50) Float64 | 26.836 | 1.418 |
| dot (3) (50,2000,50) ComplexF64 | 24.051 | 3.688 |
| trace 1000×1000 Float64 | 0.003 | 0.002 |
| trace 1000×1000 ComplexF64 | 0.004 | 0.004 |
| ptrace (100,100,100)→(100) Float64 | 0.019 | 0.009 |
| ptrace (100,100,100)→(100) ComplexF64 | 0.029 | 0.009 |
| diag (100,100,100)→(100,100) Float64 | 0.015 | 0.009 |
| diag (100,100,100)→(100,100) ComplexF64 | 0.021 | 0.011 |
| perm (30,30,30,30) Float64 | 0.576 | 0.004 |
| perm (30,30,30,30) ComplexF64 | 0.758 | 0.003 |
| tcontract (1) square 30³ Float64 | 0.072 | 0.065 |
| tcontract (1) square 30³ ComplexF64 | 0.220 | 0.236 |
| tcontract (2) (2000,50,50)×(50,2000,50) Float64 | 399.641 | 455.236 |
| tcontract (2) (2000,50,50)×(50,2000,50) ComplexF64 | 1987.545 | 2515.189 |
| tcontract (3) (50,2000,2000)×(2000,50,2000) Float64 | 690.430 | 723.605 |
| tcontract (3) (50,2000,2000)×(2000,50,2000) ComplexF64 | 5005.792 | 3047.210 |
| star (50,50) Float64 | 2.552 | 0.513 |
| star (50,50) ComplexF64 | 6.400 | 1.644 |
| starandcontract (50,50) Float64 | 0.112 | 0.030 |
| starandcontract (50,50) ComplexF64 | 0.127 | 0.028 |
| indexsum (100,100,100)→(100,100) Float64 | 0.162 | 0.183 |
| indexsum (100,100,100)→(100,100) ComplexF64 | 1.265 | 0.325 |
| hadamard (100,100,100) Float64 | 0.195 | 0.399 |
| hadamard (100,100,100) ComplexF64 | 0.695 | 0.819 |
| outer (1) square 40⁴ Float64 | 1.470 | 0.430 |
| outer (1) square 40⁴ ComplexF64 | 2.262 | 1.276 |
| outer (2) (80,20)×(20,80) Float64 | 1.224 | 0.427 |
| outer (2) (80,20)×(20,80) ComplexF64 | 2.260 | 1.033 |
| outer (3) (20,80)×(80,20) Float64 | 1.164 | 0.429 |
| outer (3) (20,80)×(80,20) ComplexF64 | 2.262 | 0.992 |
| manyinds (12+12→13 indices, dim 2) Float64 | 0.065 | 0.086 |
| manyinds (12+12→13 indices, dim 2) ComplexF64 | 0.128 | 0.162 |

## Notes

- Current parser accepts ASCII lowercase index labels.
- Generative output notation such as `->ii` is tracked in GitHub issues.

See also [benchmark suite](https://github.com/tensor4all/strided-rs-benchmark-suite) for strided-rs based on the einsum benchmark
