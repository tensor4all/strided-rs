# strided-rs

Cache-optimized kernels for strided multidimensional array operations in Rust.

This crate is a port of Julia's [Strided.jl](https://github.com/Jutho/Strided.jl) and [StridedViews.jl](https://github.com/Jutho/StridedViews.jl) libraries.

## Features

- **Dynamic-rank strided views** (`StridedView` / `StridedViewMut`) over contiguous memory
- **Owned strided arrays** (`StridedArray`) with row-major and column-major constructors
- **Lazy element operations** (conjugate, transpose, adjoint) with type-level composition
- **Zero-copy transformations**: permuting, transposing, broadcasting
- **Cache-optimized iteration** with automatic blocking and loop reordering

## Installation

This crate is currently **not published to crates.io** (`publish = false` in `Cargo.toml`).

For local development, add a path dependency:

```toml
[dependencies]
strided-rs = { path = "../strided-rs" }
```

## Quick Start

```rust
use strided_rs::{StridedArray, StridedView, map_into};

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
use strided_rs::StridedArray;

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
let row = strided_rs::StridedView::<f64>::new(&row_data, &[1, 3], &[3, 1], 0).unwrap();
let broad = row.broadcast(&[4, 3]).unwrap();
```

## Map and Reduce Operations

```rust
use strided_rs::{StridedArray, map_into, zip_map2_into, zip_map4_into, reduce};

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
use strided_rs::{StridedArray, copy_into, add, dot, symmetrize_into};

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

## Benchmarks

Run the Rust benchmark runner:

```bash
cargo bench --bench rust_compare
```

Run the Julia comparison script:

```bash
JULIA_NUM_THREADS=1 julia --project=. benches/julia_compare.jl
```

### Benchmark Results (2026-01-29 12:18)

Environment: Apple Silicon M2, single-threaded (`JULIA_NUM_THREADS=1`).

| Case | Julia Strided (ms) | Rust strided (ms) | Rust naive (ms) |
|---|---:|---:|---:|
| symmetrize_4000 | 20.52 | 25.21 | 96.65 |
| scale_transpose_1000 | 0.66 | 0.79 | 2.15 |
| mwe_stridedview_scale_transpose_1000 | 0.64 | 0.65 | 2.01 |
| complex_elementwise_1000 | 7.77 | 13.02 | 16.42 |
| permute_32_4d | 1.12 | 4.77 | 3.61 |
| multiple_permute_sum_32_4d | 2.92 | 6.54 | 9.97 |

Notes:
- Julia results from `benches/julia_compare.jl` using BenchmarkTools (mean time). Rust results from `benches/rust_compare.rs`.
- The Rust naive baseline uses `StridedView::get`/`StridedArray::set` per-element indexing with bounds checks.
- `permute_32_4d` and `multiple_permute_sum_32_4d`: the 4D strided kernel is slower than Julia; this is a known area for optimization.

### Algorithm Comparison: Julia Strided.jl vs Rust strided-rs

Both implementations share the same core algorithm ported from Strided.jl:
1. **Dimension fusion** — merge contiguous dimensions to reduce loop depth
2. **Importance-weighted ordering** — bit-pack stride orders with output array weighted 2× to determine optimal iteration order
3. **L1 cache blocking** — iteratively halve block sizes until the working set fits in 32 KB

The key architectural differences are:

| Feature | Julia | Rust |
|---------|-------|------|
| **Kernel generation** | `@generated` unrolls loops per (rank, num\_arrays) at compile time | Handwritten 1D/2D/3D/4D specializations + generic N-D fallback |
| **Loop order** | Generated code nests loops in the computed optimal order | Fixed nesting order (0→1→2→3) regardless of computed order |
| **Inner-loop SIMD** | Explicit `@simd` pragma on innermost loop | Raw pointer arithmetic; relies on LLVM auto-vectorization |
| **Threading** | Recursive work-stealing via `Threads.@spawn` (disabled in benchmarks) | Single-threaded only |

#### Per-case analysis

**symmetrize\_4000** (Julia 20.5 ms, Rust 25.2 ms) —
Julia runs `dest .= (src .+ src') ./ 2` through the general mapreduce kernel with `@simd` on the inner loop. Rust uses a dedicated 2D tiled transpose path (`zip_map2_2d_tiled_transpose`) with tile size √(32 KB / 2·8 B) ≈ 45. The Rust 2D fast path avoids the full blocking machinery but uses higher-level element access (`get`/`set_unchecked`) inside tiles, which gives LLVM less vectorization opportunity than Julia's `@simd` stride loop.

**scale\_transpose\_1000** (Julia 0.66 ms, Rust 0.79 ms) —
Both follow the same importance-weighted ordering for a 2-array (dest + transposed src) operation. Julia's `@simd` inner loop vectorizes the strided copy more effectively. Rust goes through `build_plan_fused` → `kernel_2d_inner` with pointer-stride advancement; LLVM can vectorize contiguous cases but strided access limits auto-vectorization.

**mwe\_stridedview\_scale\_transpose\_1000** (Julia 0.64 ms, Rust 0.65 ms) —
Same algorithm as scale\_transpose\_1000. Near-parity here because the operation is simple enough (single unary map on a transposed view) that both compilers generate similar code.

**complex\_elementwise\_1000** (Julia 7.8 ms, Rust 13.0 ms) —
Julia broadcasts `3a + 2conj(b) + ab` lazily via `CaptureArgs`, fusing all operations into a single pass with `@simd`. Rust calls `zip_map2_into` with a closure `|a, b| a * 3.0 + b.conj() * 2.0 + a * b`. The gap likely comes from: (1) Julia's `@simd` enabling wider vectorization of complex arithmetic, and (2) Rust's `num_complex::Complex64` arithmetic generating more conservative LLVM IR than Julia's native complex type.

**permute\_32\_4d** (Julia 1.1 ms, Rust 4.8 ms) —
This is the largest gap and reflects a fundamental architectural difference. Julia's `@generated` kernel nests loops in the *computed* optimal order, adapting to any permutation pattern. Rust's `kernel_4d_inner` always iterates dimensions in order 0→1→2→3, ignoring the importance-based ordering. For a (4,3,2,1) permutation, this fixed order causes cache misses because the innermost Rust loop strides through non-contiguous memory. Note that even the naive baseline (3.6 ms) beats the strided kernel here, confirming the loop order issue.

**multiple\_permute\_sum\_32\_4d** (Julia 2.9 ms, Rust 6.5 ms) —
Same root cause as permute\_32\_4d, amplified by 4 differently-permuted input arrays. Julia computes a combined importance score over all 5 arrays (output + 4 inputs) and generates loops in the optimal compromise order. Rust computes the same importance scores but does not apply the resulting order to its loop nesting, so cache behavior degrades further with conflicting stride patterns.

## Acknowledgments

This crate is inspired by and ports functionality from:
- [Strided.jl](https://github.com/Jutho/Strided.jl) by Jutho
- [StridedViews.jl](https://github.com/Jutho/StridedViews.jl) by Jutho

## License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

See `NOTICE` for upstream attribution (Strided.jl / StridedViews.jl are MIT-licensed).
