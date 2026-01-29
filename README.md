# strided-rs

Cache-optimized kernels for strided multidimensional array operations in Rust.

This crate is a port of Julia's [Strided.jl](https://github.com/Jutho/Strided.jl) and [StridedViews.jl](https://github.com/Jutho/StridedViews.jl) libraries.

## Features

- **Dynamic-rank strided views** (`StridedView` / `StridedViewMut`) over contiguous memory
- **Owned strided arrays** (`StridedArray`) with row-major and column-major constructors
- **Lazy element operations** (conjugate, transpose, adjoint) with type-level composition
- **Zero-copy transformations**: permuting, transposing, broadcasting
- **Cache-optimized iteration** with automatic blocking and loop reordering
- **Optional multi-threading** via Rayon (`parallel` feature) with recursive dimension splitting

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

## Parallel Feature

Enable Rayon-based multi-threading with the `parallel` feature:

```toml
[dependencies]
strided-rs = { path = "../strided-rs", features = ["parallel"] }
```

When enabled, `map_into`, `zip_map*_into`, `reduce`, and all high-level ops
(`copy_into`, `add`, `sum`, `dot`, etc.) automatically parallelize when the
total element count exceeds 32768. The implementation faithfully ports Julia
Strided.jl's `_mapreduce_threaded!` recursive dimension-splitting strategy via
`rayon::join`, with an additional **double-fusion pass** (fuse → order →
fuse-again → block) that enables threading for any memory layout, not just
column-major.

## Benchmarks

Run all benchmarks (single-threaded + multi-threaded, Rust + Julia):

```bash
bash benches/run_all.sh        # default thread counts: 1 2 4
bash benches/run_all.sh 1 2 4 8  # custom thread counts
```

Or individually:

```bash
# Single-threaded Rust
cargo bench --bench rust_compare

# Single-threaded Julia
JULIA_NUM_THREADS=1 julia --project=. benches/julia_compare.jl

# Multi-threaded Rust (N threads)
RAYON_NUM_THREADS=N cargo bench --features parallel --bench threaded_compare

# Multi-threaded comparison script
bash benches/run_threaded.sh 1 2 4
```

### Single-Threaded Results

Environment: Apple Silicon M2, single-threaded.

| Case | Julia Strided (ms) | Rust strided (ms) | Rust naive (ms) |
|---|---:|---:|---:|
| symmetrize_4000 | 20.52 | 23.18 | 95.11 |
| scale_transpose_1000 | 0.66 | 0.88 | 2.13 |
| mwe_stridedview_scale_transpose_1000 | 0.64 | 0.79 | 2.00 |
| complex_elementwise_1000 | 7.77 | 12.86 | 16.30 |
| permute_32_4d | 1.12 | 1.54 | 4.55 |
| multiple_permute_sum_32_4d | 2.92 | 3.06 | 9.30 |

Notes:
- Julia results from `benches/julia_compare.jl` using BenchmarkTools (mean time). Rust results from `benches/rust_compare.rs`.
- The Rust naive baseline uses `StridedView::get`/`StridedArray::set` per-element indexing with bounds checks.

### Multi-Threaded Scaling (Rust, `parallel` feature)

Environment: Apple Silicon M2 (4 performance + 4 efficiency cores).

| Case | 1T (ms) | 2T (ms) | 4T (ms) | Speedup (4T) |
|---|---:|---:|---:|---:|
| symmetrize_4000 | 42.2 | 23.3 | 16.4 | 2.6x |
| scale_transpose_1000 | 1.8 | 1.8 | 0.5 | 3.9x |
| mwe_scale_transpose_1000 | 1.4 | 2.4 | 0.5 | 2.6x |
| complex_elementwise_1000 | 13.6 | 6.8 | 3.7 | 3.7x |
| permute_32_4d | 1.9 | 0.9 | 0.6 | 2.9x |
| multiple_permute_sum_32_4d | 9.5 | 4.6 | 3.4 | 2.8x |
| sum_1m | 1.0 | 0.5 | 0.4 | 2.4x |

### Algorithm Comparison: Julia Strided.jl vs Rust strided-rs

Both implementations share the same core algorithm ported from Strided.jl:
1. **Dimension fusion** — merge contiguous dimensions to reduce loop depth
2. **Importance-weighted ordering** — bit-pack stride orders with output array weighted 2× to determine optimal iteration order
3. **L1 cache blocking** — iteratively halve block sizes until the working set fits in 32 KB
4. **Reversed loop nesting** — innermost loop operates on the highest-importance dimension (smallest stride) for optimal cache access

The key architectural differences are:

| Feature | Julia | Rust |
|---------|-------|------|
| **Kernel generation** | `@generated` unrolls loops per (rank, num\_arrays) at compile time | Handwritten 1D/2D/3D/4D specializations + generic N-D fallback |
| **Inner-loop SIMD** | Explicit `@simd` pragma on innermost loop | Stride-specialized inner loops: slice-based when stride=1, raw pointer otherwise; relies on LLVM auto-vectorization |
| **Threading** | Recursive dimension-splitting via `Threads.@spawn` | Recursive dimension-splitting via `rayon::join` with double-fusion for layout-agnostic parallelization |

> **Note: Strided.jl threading bug for non-column-major views.**
> Julia's `_mapreduce_fuse!` only detects column-major contiguity, so permuted
> views (e.g. `PermutedDimsArray(A, (2,1))`) with row-major strides are never
> fused. This causes `_mapreduce_threaded!` to fall through to the single-threaded
> kernel because all individual dimensions are ≤1024. strided-rs fixes this with
> a **double-fusion pass** after ordering. See
> [docs/strided\_jl\_threading\_bug.md](docs/strided_jl_threading_bug.md) for a
> minimal reproduction and root cause analysis.

#### Per-case analysis

**symmetrize\_4000** (Julia 20.5 ms, Rust 23.2 ms) —
Both use the general mapreduce kernel: dimension fusion → importance ordering → L1 cache blocking. Julia applies `@simd` on the innermost loop. Rust uses stride-specialized inner loops (slice-based when stride=1).

**scale\_transpose\_1000** (Julia 0.66 ms, Rust 0.88 ms) —
Both follow the same importance-weighted ordering for a 2-array (dest + transposed src) operation. When the inner stride is 1, Rust's stride-specialized inner loop uses slice-based iteration for better auto-vectorization.

**mwe\_stridedview\_scale\_transpose\_1000** (Julia 0.64 ms, Rust 0.79 ms) —
Same algorithm as scale\_transpose\_1000 using `map_into` with a transposed view.

**complex\_elementwise\_1000** (Julia 7.8 ms, Rust 12.9 ms) —
Julia broadcasts `3a + 2conj(b) + ab` lazily via `CaptureArgs`, fusing all operations into a single pass with `@simd`. Rust calls `zip_map2_into` with a closure. The gap likely comes from Rust's `num_complex::Complex64` arithmetic generating more conservative LLVM IR than Julia's native complex type.

**permute\_32\_4d** (Julia 1.1 ms, Rust 1.5 ms) —
Both nest loops with the highest-importance dimension innermost. The stride=1 specialization allows LLVM to vectorize the contiguous inner dimension effectively.

**multiple\_permute\_sum\_32\_4d** (Julia 2.9 ms, Rust 3.1 ms) —
Near parity. Both compute a combined importance score over all 5 arrays (output + 4 inputs) and iterate in the optimal compromise order.

## Acknowledgments

This crate is inspired by and ports functionality from:
- [Strided.jl](https://github.com/Jutho/Strided.jl) by Jutho
- [StridedViews.jl](https://github.com/Jutho/StridedViews.jl) by Jutho

## License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

See `NOTICE` for upstream attribution (Strided.jl / StridedViews.jl are MIT-licensed).
