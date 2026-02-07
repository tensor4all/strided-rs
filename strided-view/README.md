# strided-view

Core dynamic-rank strided tensor view types and metadata-only transformations.

## Scope

- `StridedView` / `StridedViewMut`
- `StridedArray`
- Stride/layout helpers (`row_major_strides`, `col_major_strides`)
- Zero-copy metadata ops (`permute`, `transpose_2d`, `broadcast`, `diagonal_view`)

`strided-view` does not implement heavy compute kernels by itself.
Use `strided-kernel` for map/reduce operations.

## Quick Example

```rust
use strided_view::StridedArray;

let a = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 10 + idx[1]) as f64);
let v = a.view();
let vt = v.permute(&[1, 0]).unwrap();
assert_eq!(vt.dims(), &[3, 2]);
assert_eq!(vt.get(&[2, 1]), 12.0);
```
