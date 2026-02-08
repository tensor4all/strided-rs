# strided-rs

`strided-rs` is a Rust workspace for strided tensor views, kernels, and einsum.
It is inspired by Julia's [Strided.jl](https://github.com/Jutho/Strided.jl),
[StridedViews.jl](https://github.com/Jutho/StridedViews.jl), and
[OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl).

## Workspace Layout

- [`strided-view`](strided-view/README.md): core dynamic-rank strided view/array types and metadata ops
- [`strided-kernel`](strided-kernel/README.md): cache-optimized elementwise/reduction kernels over strided views
- [`strided-einsum2`](strided-einsum2/README.md): binary einsum (`einsum2_into`) on strided tensors
- [`strided-opteinsum`](strided-opteinsum/README.md): N-ary einsum frontend with nested notation and contraction-order optimization

## Features

- **Dynamic-rank strided views** (`StridedView` / `StridedViewMut`) over contiguous memory
- **Owned strided arrays** (`StridedArray`) with row-major and column-major constructors
- **Lazy element operations** (conjugate, transpose, adjoint) with type-level composition
- **Zero-copy transformations**: permuting, transposing, broadcasting
- **Cache-optimized iteration** with automatic blocking and loop reordering
- **Optional multi-threading** via Rayon (`parallel` feature) with recursive dimension splitting

## Installation

These crates are currently **not published to crates.io** (`publish = false`).
Use workspace path dependencies:

```toml
[dependencies]
strided-view = { path = "../strided-rs/strided-view" }
strided-kernel = { path = "../strided-rs/strided-kernel" }
strided-einsum2 = { path = "../strided-rs/strided-einsum2" }
strided-opteinsum = { path = "../strided-rs/strided-opteinsum" }
```

## Documentation

Generate API docs locally:

```bash
cargo doc --workspace --no-deps
```

Open docs locally:

```bash
open target/doc/index.html
```

CI also builds rustdoc on PRs and deploys workspace docs to GitHub Pages on `main`.

## Quick Start

```rust
use strided_kernel::{StridedArray, StridedView, map_into};

// Create a row-major 2D array
let src = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| {
    (idx[0] * 10 + idx[1]) as f64
});
let mut dest = StridedArray::<f64>::row_major(&[2, 3]);

// Element-wise map: dest[i] = src[i] * 2
map_into(&mut dest.view_mut(), &src.view(), |x| x * 2.0).unwrap();
assert_eq!(dest.get(&[1, 2]), 24.0); // (1*10 + 2) * 2
```

## Core Types

### `StridedView<'a, T, Op>` / `StridedViewMut<'a, T>`

Dynamic-rank immutable/mutable views over strided data:
- `T`: Element type
- `Op`: Element operation (Identity, Conj, Transpose, Adjoint) -- applied lazily on access

### `StridedArray<T>`

Owned strided multidimensional array with `view()` and `view_mut()` methods.

## View Operations

```rust
use strided_kernel::StridedArray;

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
let row = strided_kernel::StridedView::<f64>::new(&row_data, &[1, 3], &[3, 1], 0).unwrap();
let broad = row.broadcast(&[4, 3]).unwrap();
```

## Map and Reduce Operations

```rust
use strided_kernel::{StridedArray, map_into, zip_map2_into, zip_map4_into, reduce};

let a = StridedArray::<f64>::from_fn_row_major(&[4, 5], |idx| idx[0] as f64);
let b = StridedArray::<f64>::from_fn_row_major(&[4, 5], |idx| idx[1] as f64);
let mut out = StridedArray::<f64>::row_major(&[4, 5]);

// Unary map: dest[i] = f(src[i])
map_into(&mut out.view_mut(), &a.view(), |x| x * 2.0).unwrap();

// Binary zip map: dest[i] = f(a[i], b[i])
zip_map2_into(&mut out.view_mut(), &a.view(), &b.view(), |x, y| x + y).unwrap();

// Full reduction
let total = reduce(&a.view(), |x| x, |a, b| a + b, 0.0).unwrap();
```

## High-Level Operations

```rust
use strided_kernel::{StridedArray, copy_into, add, dot, symmetrize_into};

let a = StridedArray::<f64>::from_fn_row_major(&[4, 4], |idx| (idx[0] * 10 + idx[1]) as f64);
let mut out = StridedArray::<f64>::row_major(&[4, 4]);

// Copy
copy_into(&mut out.view_mut(), &a.view()).unwrap();

// Element-wise add: dest[i] += src[i]
add(&mut out.view_mut(), &a.view()).unwrap();

// Dot product
let d = dot(&a.view(), &a.view()).unwrap();

// Symmetrize: dest = (src + src^T) / 2
symmetrize_into(&mut out.view_mut(), &a.view()).unwrap();
```

## Cache Optimization

The library automatically optimizes iteration order for cache efficiency:

1. **Dimension Fusion**: Contiguous dimensions are fused to reduce loop overhead
2. **Dimension Reordering**: Dimensions are sorted by stride magnitude for optimal memory access
3. **Tiled Iteration**: Operations are blocked to fit in L1 cache (32KB)
4. **Contiguous Fast Paths**: Contiguous arrays bypass blocking for direct iteration

## Parallel Feature

Enable Rayon-based multi-threading with the `parallel` feature:

```toml
[dependencies]
strided-kernel = { path = "../strided-rs/strided-kernel", features = ["parallel"] }
```

When enabled, `map_into`, `zip_map*_into`, `reduce`, and all high-level ops
(`copy_into`, `add`, `sum`, `dot`, etc.) automatically parallelize when the
total element count exceeds 32768. The implementation faithfully ports Julia
Strided.jl's `_mapreduce_threaded!` recursive dimension-splitting strategy via
`rayon::join`. The pipeline orders dimensions before fusing (order → fuse →
block), which enables threading for any memory layout, not just column-major.

## Benchmarks

See [`strided-kernel/README.md`](strided-kernel/README.md#benchmarks) for benchmark results, run commands, and detailed analysis comparing Rust strided-rs vs Julia Strided.jl.

## Acknowledgments

This crate is inspired by and ports functionality from:
- [Strided.jl](https://github.com/Jutho/Strided.jl) by Jutho
- [StridedViews.jl](https://github.com/Jutho/StridedViews.jl) by Jutho
- [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) for
  `strided-opteinsum` design ideas and reference test-case patterns

## License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

See `NOTICE` for upstream attribution (Strided.jl / StridedViews.jl are MIT-licensed).
