# strided-kernel

Cache-optimized compute kernels over `strided-view` tensors.

## Scope

- Unary/Binary/N-ary map kernels (`map_into`, `zip_map*_into`)
- Reductions (`reduce`, `reduce_axis`)
- Utility ops (`copy_into`, `add`, `dot`, `sum`, `symmetrize_into`)
- Optional parallel execution via `parallel` feature (Rayon)

## Quick Example

```rust
use strided_kernel::{map_into, StridedArray};

let src = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1]) as f64);
let mut dst = StridedArray::<f64>::row_major(&[2, 3]);
map_into(&mut dst.view_mut(), &src.view(), |x| 2.0 * x).unwrap();
assert_eq!(dst.get(&[1, 2]), 10.0);
```

## Parallel Feature

```toml
[dependencies]
strided-kernel = { path = "../strided-kernel", features = ["parallel"] }
```
