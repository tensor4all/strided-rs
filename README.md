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

Run all benchmarks (single-threaded + multi-threaded, Rust + Julia):

```bash
bash strided-kernel/benches/run_all.sh        # default thread counts: 1 2 4
bash strided-kernel/benches/run_all.sh 1 2 4 8  # custom thread counts
```

Or individually:

```bash
# Single-threaded Rust
cargo bench --bench rust_compare --manifest-path strided-kernel/Cargo.toml

# Single-threaded Julia
JULIA_NUM_THREADS=1 julia --project=strided-kernel/benches strided-kernel/benches/julia_compare.jl

# Multi-threaded Rust (N threads)
RAYON_NUM_THREADS=N cargo bench --features parallel --bench threaded_compare --manifest-path strided-kernel/Cargo.toml

# Multi-threaded comparison script
bash strided-kernel/benches/run_threaded.sh 1 2 4

# Scaling benchmarks (sum + permute, 1/2/4 threads)
bash strided-kernel/benches/run_scaling.sh
bash strided-kernel/benches/run_scaling.sh 1 2 4 8  # custom thread counts

# Rank-25 tensor permutation (quantum circuit simulation workload)
RAYON_NUM_THREADS=1 cargo bench --bench rank25_permute --manifest-path strided-kernel/Cargo.toml

# Rank-25 Julia comparison
JULIA_NUM_THREADS=1 julia --project=strided-kernel/benches strided-kernel/benches/julia_rank25_compare.jl
```

### Single-Threaded Results

Environment: Apple Silicon M2, single-threaded.

| Case | Julia Strided (ms) | Rust strided (ms) | Rust naive (ms) |
|---|---:|---:|---:|
| symmetrize_4000 | 22.22 | 21.41 | 43.09 |
| scale_transpose_1000 | 0.67 | 0.95 | 0.47 |
| mwe_stridedview_scale_transpose_1000 | 0.78 | 0.99 | 0.40 |
| complex_elementwise_1000 | 8.10 | 12.79 | 12.21 |
| permute_32_4d | 1.04 | 1.32 | 2.03 |
| multiple_permute_sum_32_4d | 2.44 | 3.24 | 2.48 |

Notes:
- Julia results from `strided-kernel/benches/julia_compare.jl` (mean time). Rust results from `strided-kernel/benches/rust_compare.rs` (mean time).
- All benchmarks use column-major layout for parity with Julia.
- The Rust naive baseline uses raw pointer arithmetic with `unsafe` and precomputed strides (no bounds checks, no library overhead).
- `scale_transpose` and `multiple_permute_sum`: the naive baseline is faster because the ordering/blocking pipeline overhead is not recovered on these relatively small, simple access patterns. Julia Strided shows the same trend.

### Multi-Threaded Scaling (Rust, `parallel` feature)

Environment: Apple Silicon M2 (4 performance + 4 efficiency cores). Mean time.

| Case | 1T (ms) | 2T (ms) | 4T (ms) | Speedup (4T) |
|---|---:|---:|---:|---:|
| symmetrize_4000 | 26.30 | 18.95 | 12.29 | 2.1x |
| scale_transpose_1000 | 1.03 | 0.49 | 0.35 | 2.9x |
| mwe_scale_transpose_1000 | 0.88 | 0.39 | 0.29 | 3.1x |
| complex_elementwise_1000 | 13.11 | 6.82 | 3.56 | 3.7x |
| permute_32_4d | 1.40 | 0.70 | 0.51 | 2.8x |
| multiple_permute_sum_32_4d | 3.34 | 1.96 | 1.34 | 2.5x |
| sum_1m | 0.96 | 0.47 | 0.27 | 3.5x |

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
| **Threading** | Recursive dimension-splitting via `Threads.@spawn` | Recursive dimension-splitting via `rayon::join`; order-before-fuse pipeline enables layout-agnostic parallelization |

> **Note: Strided.jl threading bug for non-column-major views.**
> Julia's pipeline fuses before ordering (`fuse → order → block`), so
> `_mapreduce_fuse!` only detects column-major contiguity. Permuted views
> (e.g. `PermutedDimsArray(A, (2,1))`) with row-major strides are never fused,
> causing `_mapreduce_threaded!` to fall through to the single-threaded kernel.
> strided-rs fixes this by simply reordering the pipeline to `order → fuse →
> block`: ordering first puts smallest-stride dimensions adjacent, and a single
> fusion pass then catches contiguity regardless of memory layout. See
> [docs/strided\_jl\_threading\_bug.md](docs/strided_jl_threading_bug.md) for a
> minimal reproduction and root cause analysis.

### Scaling Benchmarks (Strided.jl benchtests.jl suite)

These benchmarks measure performance across exponentially-scaled array sizes, showing the
crossover point where the ordering/blocking pipeline overhead pays off.

Run with: `bash strided-kernel/benches/run_scaling.sh` (default: 1 2 4 threads)

Environment: Apple Silicon M2 (4P + 4E cores). Median timing, adaptive iteration count.

#### 1D Sum

| size | Rust naive (μs) | Rust strided 1T | 2T | 4T | Julia base (μs) | Julia Strided 1T | 2T | 4T |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.04 | 0.04 | 0.00 | 0.00 | 0.00 | 0.04 | 0.04 | 0.04 |
| 12 | 0.04 | 0.04 | 0.00 | 0.00 | 0.00 | 0.04 | 0.04 | 0.04 |
| 32 | 0.04 | 0.04 | 0.00 | 0.00 | 0.00 | 0.04 | 0.04 | 0.04 |
| 91 | 0.08 | 0.04 | 0.04 | 0.04 | 0.00 | 0.08 | 0.08 | 0.08 |
| 256 | 0.21 | 0.04 | 0.04 | 0.04 | 0.04 | 0.17 | 0.17 | 0.17 |
| 725 | 0.75 | 0.12 | 0.08 | 0.08 | 0.08 | 0.38 | 0.38 | 0.38 |
| 2048 | 2.04 | 0.29 | 0.25 | 0.25 | 0.21 | 0.92 | 0.96 | 0.96 |
| 5793 | 5.12 | 0.62 | 0.67 | 0.67 | 0.50 | 2.62 | 2.58 | 2.58 |
| 16384 | 13.96 | 1.83 | 1.83 | 1.83 | 1.54 | 7.08 | 7.25 | 7.25 |
| 46341 | 39.62 | 40.58 | 36.58 | 38.46 | 4.62 | 20.46 | 26.67 | 16.71 |
| 131072 | 112.17 | 113.21 | 77.12 | 77.12 | 12.50 | 57.75 | 46.85 | 24.33 |
| 370728 | 334.54 | 320.54 | 186.08 | 189.25 | 35.25 | 163.33 | 101.56 | 61.71 |
| 1048576 | 907.58 | 933.54 | 500.00 | 431.79 | 102.54 | 469.21 | 251.29 | 160.67 |

Notes:
- For 1D contiguous sum, Rust strided can beat the naive loop at small/medium sizes due to an explicit SIMD reduction kernel.
- With the Rust `parallel` feature enabled, sizes above the threading threshold route through the threaded kernel path (even at 1T), so 1T can be slightly slower than the naive loop until multi-threading is enabled.
- Julia's `Base.sum` uses hand-tuned SIMD reduction; Julia's `@strided sum` includes kernel-dispatch overhead compared to `Base.sum`.

#### 4D Permute: (4,3,2,1) — full reversal

| s | s⁴ | Rust naive (μs) | Rust strided 1T | 2T | 4T | Julia copy (μs) | Julia Strided 1T | 2T | 4T |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 256 | 0.21 | 1.33 | 1.33 | 1.33 | 0.04 | 0.29 | 0.33 | 0.29 |
| 8 | 4096 | 2.04 | 4.00 | 4.29 | 4.04 | 0.42 | 1.79 | 1.83 | 1.79 |
| 12 | 20736 | 14.33 | 14.25 | 14.21 | 14.21 | 2.50 | 8.58 | 8.54 | 8.54 |
| 16 | 65536 | 53.71 | 50.71 | 45.75 | 47.00 | 7.88 | 31.92 | 34.33 | 40.50 |
| 24 | 331776 | 292.25 | 201.33 | 126.00 | 149.50 | 40.62 | 127.12 | 86.38 | 70.08 |
| 32 | 1048576 | 1206.62 | 1512.17 | 637.92 | 491.54 | 188.00 | 934.00 | 468.08 | 333.88 |
| 48 | 5308416 | 19000.92 | 9917.50 | 6321.96 | 3604.54 | 1163.71 | 8151.92 | 5123.25 | 3238.73 |
| 64 | 16777216 | 98992.71 | 52719.83 | 28936.25 | 17362.04 | 4016.40 | 50167.04 | 26492.71 | 16612.62 |

#### 4D Permute: (2,3,4,1)

| s | s⁴ | Rust naive (μs) | Rust strided 1T | 2T | 4T | Julia copy (μs) | Julia Strided 1T | 2T | 4T |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 256 | 0.21 | 1.29 | 1.29 | 1.29 | 0.04 | 0.25 | 0.25 | 0.29 |
| 8 | 4096 | 2.04 | 3.96 | 3.96 | 3.96 | 0.42 | 1.62 | 1.67 | 1.62 |
| 12 | 20736 | 17.33 | 13.50 | 13.50 | 13.50 | 2.50 | 8.88 | 8.88 | 8.88 |
| 16 | 65536 | 46.12 | 38.00 | 39.12 | 40.21 | 7.83 | 25.79 | 29.42 | 31.08 |
| 24 | 331776 | 229.50 | 172.42 | 114.25 | 135.08 | 42.17 | 129.46 | 86.17 | 77.42 |
| 32 | 1048576 | 759.46 | 743.42 | 466.25 | 358.88 | 187.92 | 760.98 | 348.23 | 254.29 |
| 48 | 5308416 | 19792.42 | 6904.42 | 4785.92 | 3156.54 | 1178.33 | 7229.12 | 4260.04 | 2895.08 |
| 64 | 16777216 | 85280.88 | 24360.12 | 17059.75 | 11069.58 | 4157.96 | 21393.33 | 14701.90 | 9304.46 |

#### 4D Permute: (3,4,1,2)

| s | s⁴ | Rust naive (μs) | Rust strided 1T | 2T | 4T | Julia copy (μs) | Julia Strided 1T | 2T | 4T |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 256 | 0.21 | 1.29 | 1.29 | 1.33 | 0.04 | 0.25 | 0.29 | 0.29 |
| 8 | 4096 | 2.04 | 4.00 | 4.00 | 4.00 | 0.46 | 1.50 | 1.50 | 1.50 |
| 12 | 20736 | 10.00 | 14.00 | 14.00 | 14.00 | 2.46 | 7.50 | 7.46 | 7.46 |
| 16 | 65536 | 47.67 | 39.17 | 37.12 | 39.29 | 7.92 | 26.58 | 30.71 | 46.88 |
| 24 | 331776 | 230.71 | 200.21 | 127.46 | 149.33 | 42.17 | 117.21 | 78.67 | 92.96 |
| 32 | 1048576 | 1114.75 | 959.54 | 523.00 | 401.04 | 173.79 | 1191.96 | 629.08 | 404.88 |
| 48 | 5308416 | 9951.62 | 7670.17 | 4815.21 | 3053.54 | 1118.29 | 6055.00 | 4137.88 | 2787.50 |
| 64 | 16777216 | 70490.25 | 29309.25 | 19775.17 | 14502.79 | 4003.54 | 27776.96 | 18433.83 | 12759.00 |

#### Scaling observations

- **Crossover point**: Strided's ordering/blocking overhead is recovered at ~20K elements (s≥12 for 4D permute). Below this, the naive loop is faster.
- **Large arrays (s≥48)**: Rust strided achieves 0.3-0.6x of naive single-threaded, and further improves with threading.
- **Rust vs Julia strided (1T, s=64)**:
  - (4,3,2,1): Rust 52.7ms vs Julia 50.2ms — near parity
  - (2,3,4,1): Rust 24.4ms vs Julia 21.4ms — Rust 1.1x slower
  - (3,4,1,2): Rust 29.3ms vs Julia 27.8ms — near parity
- **Multi-threaded (4T, s=64)**: Both achieve ~2-4x speedup over single-threaded strided. Rust and Julia reach comparable absolute performance at large sizes.
- **Julia copy vs Rust naive**: Julia's `copy!` is ~10-25x faster than Rust's raw pointer loop for large s because Julia uses optimized `memcpy`; Rust's naive loop does element-by-element copy for parity with the permute benchmark.

#### Per-case analysis

**symmetrize\_4000** (Julia 22.2 ms, Rust 21.4 ms) —
Both use the general mapreduce kernel: dimension fusion → importance ordering → L1 cache blocking. Julia applies `@simd` on the innermost loop. Rust uses stride-specialized inner loops (slice-based when stride=1). Near parity.

**scale\_transpose\_1000** (Julia 0.67 ms, Rust 0.95 ms) —
Both follow the same importance-weighted ordering for a 2-array (dest + transposed src) operation. The naive baseline (0.42 ms) is faster because it writes contiguously without blocking overhead; the strided version pays for the ordering/blocking pipeline on a small array.

**mwe\_stridedview\_scale\_transpose\_1000** (Julia 0.78 ms, Rust 0.99 ms) —
Same operation as scale\_transpose\_1000 using `map_into` with a transposed view.

**complex\_elementwise\_1000** (Julia 8.1 ms, Rust 12.8 ms) —
Both arrays are contiguous, so the operation is compute-bound. The gap comes from Julia's `@simd` enabling aggressive auto-vectorization of transcendental functions (`exp`, `sin`), while Rust's LLVM generates more conservative code for the same operations.

**permute\_32\_4d** (Julia 1.0 ms, Rust 1.3 ms) —
Both nest loops with the highest-importance dimension innermost. The stride=1 specialization allows LLVM to vectorize the contiguous inner dimension effectively, but Rust is still ~1.3x slower here.

**multiple\_permute\_sum\_32\_4d** (Julia 2.4 ms, Rust 3.2 ms) —
Both compute a combined importance score over all 5 arrays (output + 4 inputs) and iterate in the optimal compromise order, but Rust is still ~1.3x slower.

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
