# strided-view

Core dynamic-rank strided tensor view types and metadata-only transformations.

## Scope

- `StridedView` / `StridedViewMut`
- `StridedArray`
- Stride/layout helpers (`row_major_strides`, `col_major_strides`)
- Zero-copy metadata ops (`permute`, `transpose_2d`, `broadcast`, `diagonal_view`)

`strided-view` does not implement heavy compute kernels by itself.
Use `strided-kernel` for map/reduce operations.

## Core Types

### `StridedView<'a, T, Op>` / `StridedViewMut<'a, T>`

Dynamic-rank immutable/mutable views over strided data:
- `T`: Element type
- `Op`: Element operation (Identity, Conj, Transpose, Adjoint) -- applied lazily on access

### `StridedArray<T>`

Owned strided multidimensional array with `view()` and `view_mut()` methods.

## Quick Example

```rust
use strided_view::StridedArray;

let a = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 10 + idx[1]) as f64);
let v = a.view();
let vt = v.permute(&[1, 0]).unwrap();
assert_eq!(vt.dims(), &[3, 2]);
assert_eq!(vt.get(&[2, 1]), 12.0);
```

## View Operations

```rust
use strided_view::{StridedArray, StridedView};

let a = StridedArray::<f64>::from_fn_row_major(&[3, 4], |idx| {
    (idx[0] * 10 + idx[1]) as f64
});
let v = a.view();

// Transpose (zero-copy, swaps strides)
let vt = v.transpose_2d().unwrap();
assert_eq!(vt.dims(), &[4, 3]);

// General permutation (zero-copy)
let vp = v.permute(&[1, 0]).unwrap();
assert_eq!(vp.get(&[2, 1]), v.get(&[1, 2]));

// Broadcast (stride-0 for size-1 dims)
let row_data = vec![1.0, 2.0, 3.0];
let row = StridedView::<f64>::new(&row_data, &[1, 3], &[3, 1], 0).unwrap();
let broad = row.broadcast(&[4, 3]).unwrap();
```
