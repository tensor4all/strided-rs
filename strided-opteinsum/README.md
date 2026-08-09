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

Published benchmark programs and current measured results live in
[`strided-rs-benchmark-suite`](https://github.com/tensor4all/strided-rs-benchmark-suite).

## Notes

- Current parser accepts ASCII lowercase index labels.
- Generative output notation such as `->ii` is tracked in GitHub issues.
