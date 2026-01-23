# mdarray-strided

Cache-optimized kernels for strided multidimensional array operations in Rust.

This crate is a port of Julia's [Strided.jl](https://github.com/Jutho/Strided.jl) and [StridedViews.jl](https://github.com/Jutho/StridedViews.jl) libraries.

## Features

- **Zero-copy strided views** over contiguous memory
- **Lazy element operations** (conjugate, transpose, adjoint) with type-level composition
- **Zero-copy transformations**: slicing, reshaping, permuting, transposing
- **Broadcasting** with stride-0 for size-1 dimensions
- **Cache-optimized iteration** with automatic blocking and loop reordering
- **Parallel iteration** with rayon (optional)
- **BLAS integration** for optimized linear algebra (optional)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mdarray-strided = "0.1"
```

### Optional Features

```toml
[dependencies]
mdarray-strided = { version = "0.1", features = ["parallel", "blas"] }
```

- `parallel`: Enable rayon-based parallel iteration (`par_iter()`)
- `blas`: Enable BLAS-backed linear algebra operations (`blas_axpy`, `blas_dot`, `blas_gemm`)

## Quick Start

```rust
use mdarray_strided::{StridedArrayView, Identity};

// Create a 2D view over existing data
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let view: StridedArrayView<'_, f64, 2, Identity> =
    StridedArrayView::new(&data, [2, 3], [3, 1], 0).unwrap();

// Access elements
assert_eq!(view.get([0, 0]), 1.0);
assert_eq!(view.get([1, 2]), 6.0);

// Transpose (zero-copy)
let transposed = view.t();
assert_eq!(transposed.size(), &[3, 2]);
assert_eq!(transposed.get([0, 1]), 4.0);
```

## Core Types

### `StridedArrayView<'a, T, N, Op>`

An immutable view over strided data with:
- `T`: Element type
- `N`: Number of dimensions (const generic)
- `Op`: Element operation (Identity, Conj, Transpose, Adjoint)

### `StridedArrayViewMut<'a, T, N, Op>`

A mutable version of `StridedArrayView`.

## View Operations

### Slicing

```rust
use mdarray_strided::{StridedArrayView, Identity, StridedRange};

let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
let view: StridedArrayView<'_, f64, 3, Identity> =
    StridedArrayView::new(&data, [2, 3, 4], [12, 4, 1], 0).unwrap();

// Slice with range
let sliced = view.slice([0..1, 1..3, 0..4]).unwrap();
assert_eq!(sliced.size(), &[1, 2, 4]);

// Slice with stride
let strided = view.slice([
    StridedRange::new(0, 2, 1),   // all rows
    StridedRange::new(0, 3, 2),   // every other column
    StridedRange::new(0, 4, 1),   // all depth
]).unwrap();
```

### Transpose and Permute

```rust
// 2D transpose
let transposed = view_2d.t();

// Hermitian adjoint (transpose + conjugate)
let adjoint = complex_view.h();

// General permutation
let permuted = view_3d.permute([2, 0, 1]);
```

### Reshape

```rust
let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
let view_1d: StridedArrayView<'_, f64, 1, Identity> =
    StridedArrayView::new(&data, [12], [1], 0).unwrap();

// Reshape to 2D (only works if contiguous)
let view_2d = view_1d.reshape_2d([3, 4]).unwrap();
```

### Broadcasting

```rust
// Broadcast a row vector [3] to a matrix [4, 3]
let row: StridedArrayView<'_, f64, 1, Identity> =
    StridedArrayView::new(&data, [3], [1], 0).unwrap();
let matrix = row.broadcast([4, 3]).unwrap();
// row is repeated 4 times (stride-0 broadcasting)
```

## Element Operations

Element operations are applied lazily and compose at the type level:

```rust
use mdarray_strided::{StridedArrayView, Identity, Conj};
use num_complex::Complex64;

let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
let view: StridedArrayView<'_, Complex64, 1, Identity> =
    StridedArrayView::new(&data, [2], [1], 0).unwrap();

// Apply conjugate (lazy)
let conj_view = view.conj();
assert_eq!(conj_view.get([0]), Complex64::new(1.0, -2.0));

// Double conjugate returns to identity (type-level optimization)
let double_conj = conj_view.conj(); // type: StridedArrayView<..., Identity>
```

## Iteration

```rust
// Sequential iteration
for value in view.iter() {
    println!("{}", value);
}

// Enumerated iteration
for (indices, value) in view.enumerate() {
    println!("{:?}: {}", indices, value);
}
```

### Parallel Iteration (requires `parallel` feature)

```rust
use rayon::prelude::*;

let sum: f64 = view.par_iter().sum();
let max: f64 = view.par_iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
```

## Map and Reduce Operations

```rust
use mdarray_strided::{map_into, zip_map2_into, zip_map3_into, zip_map4_into, reduce};

// Unary map: dest[i] = f(src[i])
map_into(&mut dest, &src, |x| x * 2.0).unwrap();

// Binary zip map: dest[i] = f(a[i], b[i])
zip_map2_into(&mut dest, &a, &b, |x, y| x + y).unwrap();

// Ternary zip map: dest[i] = f(a[i], b[i], c[i])
zip_map3_into(&mut dest, &a, &b, &c, |x, y, z| x * y + z).unwrap();

// Quaternary zip map: dest[i] = f(a[i], b[i], c[i], d[i])
zip_map4_into(&mut dest, &a, &b, &c, &d, |w, x, y, z| w * x + y * z).unwrap();

// Reduce with map
let total = reduce(&src, |x| *x, |a, b| a + b, 0.0).unwrap();
```

## High-Level Operations

```rust
use mdarray_strided::{copy_into, copy_scale, copy_conj, add, mul, axpy, fma, sum, dot};

// Copy operations
copy_into(&mut dest, &src).unwrap();           // dest = src
copy_scale(&mut dest, &src, 2.0).unwrap();     // dest = 2.0 * src
copy_conj(&mut dest, &src).unwrap();           // dest = conj(src)

// Element-wise arithmetic
add(&mut dest, &a, &b).unwrap();               // dest = a + b
mul(&mut dest, &a, &b).unwrap();               // dest = a * b

// BLAS-like operations
axpy(&mut y, 2.0, &x).unwrap();                // y = 2.0 * x + y
fma(&mut dest, &a, &b, &c).unwrap();           // dest = a * b + c

// Reductions
let s = sum(&array).unwrap();                  // sum of all elements
let d = dot(&x, &y).unwrap();                  // dot product
```

## BLAS Integration (requires `blas` feature)

```rust
use mdarray_strided::{blas_axpy, blas_dot, blas_gemm, is_blas_matrix};

// Check if a matrix is BLAS-compatible
if let Some(info) = is_blas_matrix(&matrix_view) {
    println!("Layout: {:?}, ld: {}", info.layout, info.ld);
}

// BLAS axpy: y = alpha * x + y
blas_axpy(2.0, &x, &mut y).unwrap();

// BLAS dot: result = x · y
let dot_product = blas_dot(&x, &y).unwrap();

// BLAS gemm: C = alpha * A * B + beta * C
blas_gemm(1.0, &a, &b, 0.0, &mut c).unwrap();
```

## Cache Optimization

The library automatically optimizes iteration order for cache efficiency:

1. **Dimension Reordering**: Dimensions are sorted by stride magnitude
2. **Tiled Iteration**: Operations are blocked to fit in L1 cache (32KB)
3. **Contiguous Fast Paths**: Contiguous arrays bypass blocking for direct iteration

```rust
// Check contiguity information
let inner_dims = view.contiguous_inner_dims();
let inner_len = view.contiguous_inner_len();

// Get contiguous slice (if applicable)
if let Some(slice) = view_1d.as_slice() {
    // Direct slice access for SIMD/BLAS
}
```

## Performance Tips

1. **Use contiguous arrays when possible** - they get fast-path optimization
2. **Enable the `parallel` feature** for large arrays
3. **Use `is_blas_matrix`** to check BLAS compatibility before calling BLAS functions
4. **Prefer row-major layout** (stride[N-1] == 1) for better cache performance

## License

MIT

## Acknowledgments

This crate is inspired by and ports functionality from:
- [Strided.jl](https://github.com/Jutho/Strided.jl) by Jutho
- [StridedViews.jl](https://github.com/Jutho/StridedViews.jl) by Jutho
