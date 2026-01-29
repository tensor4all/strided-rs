# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

strided-rs is a Rust library providing cache-optimized kernels for strided multidimensional array operations. It is a **port of Julia's Strided.jl/StridedViews.jl libraries**, currently built on top of the `mdarray` crate.

**Current Status (v0.1):**
- Broadcasting with `CaptureArgs` for lazy evaluation (stride-0 for size-1 dims)
- Zero-copy transformations: slice, reshape, permute, transpose
- Lazy element operations with type-level composition (`Identity`, `Conj`, `Transpose`, `Adjoint`)
- Cache-optimized map/reduce/broadcast kernels
- Overlapping src/dest memory is not supported

## Pre-Push / PR Checklist

Before pushing or creating a pull request, **all** of the following must pass:

```bash
cargo fmt --check   # formatting
cargo test          # all tests
```

If `cargo fmt --check` fails, run `cargo fmt` to fix formatting automatically.

## Build Commands

```bash
# Build
cargo build

# Run all tests
cargo test

# Run a single test
cargo test test_map_into_transposed

# Check formatting
cargo fmt --check

# Run benchmarks
cargo bench

# Run a specific benchmark
cargo bench -- copy_permuted
```

## Benchmarking Notes (Rust)

When adding or modifying benchmarks (especially "naive" baselines), optimize the baseline as well:
- Avoid per-element high-level indexing (`a[[i, j]]`) inside hot loops when the data is contiguous; prefer pointer-based loops or precomputed strides so the "naive" number reflects math + memory traffic, not indexing overhead.
- Keep setup/allocation out of the timed region and use `black_box` to prevent dead-code elimination.
- For parity with Julia scripts, run single-threaded (`RAYON_NUM_THREADS=1` / `JULIA_NUM_THREADS=1`) unless explicitly testing threading.

### Benchmark with native CPU features

By default `rustc` targets a generic `x86-64` baseline (SSE2 only). To enable AVX2/AVX-512 auto-vectorization for the host CPU:

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench
```

This can yield significant improvements for contiguous inner loops that LLVM auto-vectorizes.

## Architecture

### Core Types

- **`StridedArrayView<'a, T, N, Op>`** / **`StridedArrayViewMut<'a, T, N, Op>`**: Const-generic strided views where:
  - `T`: Element type
  - `N`: Number of dimensions (const generic)
  - `Op`: Element operation (Identity, Conj, Transpose, Adjoint) - applied lazily on access

### Module Organization

| Module | Purpose | Julia Equivalent |
|--------|---------|------------------|
| `view.rs` | `StridedArrayView`/`StridedArrayViewMut` types with slicing, permutation, reshape, broadcast | `stridedview.jl` |
| `element_op.rs` | Element operations (`Identity`, `Conj`, `Transpose`, `Adjoint`) with type-level composition | `FN`, `FC`, `FT`, `FA` |
| `kernel.rs` | `StridedView`/`StridedViewMut` internal wrappers, `_mapreduce_kernel!` implementation | `mapreduce.jl` |
| `map.rs` | `map_into`, `zip_map2_into`, `zip_map3_into`, `zip_map4_into` | `Base.map!` |
| `reduce.rs` | `reduce`, `reduce_axis`, `mapreducedim_into` | `Base.mapreduce`, `Base.mapreducedim!` |
| `broadcast.rs` | `CaptureArgs`, `promoteshape`, `broadcast_into`, `Arg`, `Scalar` | `broadcast.jl` |
| `ops.rs` | High-level operations: `copy_into`, `add`, `mul`, `axpy`, `fma`, `sum`, `dot`, `symmetrize_into` | Various |
| `order.rs` | Dimension ordering algorithm - sorts dimensions by stride magnitude | `indexorder` |
| `block.rs` | Block size computation to fit within L1 cache (`_computeblocks`) | `_computeblocks` |
| `fuse.rs` | Dimension fusion for contiguous dimensions | `_mapreduce_fuse!` |
| `auxiliary.rs` | Helper functions: `index_order`, `normalize_strides`, `simplify_dims` | `auxiliary.jl` |

### Cache Optimization Strategy

The library uses a blocking strategy faithful to Strided.jl:
1. **Dimension Fusion**: Contiguous dimensions are fused to reduce loop overhead (`_mapreduce_fuse!`)
2. **Dimension Reordering**: Dimensions are sorted by stride importance for optimal cache access (`_mapreduce_order!`)
3. **Tiled Iteration**: Operations are blocked into tiles fitting L1 cache (`_computeblocks`)
4. **Contiguous Fast Paths**: Contiguous arrays bypass blocking for direct iteration

### Key Constants

| Constant | Value | Julia Equivalent |
|----------|-------|------------------|
| `BLOCK_MEMORY_SIZE` | 32KB | `BLOCKMEMORYSIZE` |
| `CACHE_LINE_SIZE` | 64 bytes | `_cachelinelength` |

### Dependencies

- `mdarray` (v0.7.2): Base multidimensional array type
- `num-traits`/`num-complex`: Numeric trait bounds
- `thiserror`: Error type derivation
- `bytemuck`: POD trait for byte-copy fast paths

## Julia Port Status

### Fully Ported (98%)

| Julia Module | Rust Module | Status |
|--------------|-------------|--------|
| `StridedViews.jl/stridedview.jl` | `view.rs` | ✅ Complete |
| `StridedViews.jl/auxiliary.jl` | `auxiliary.rs` | ✅ Complete |
| `Strided.jl/mapreduce.jl` | `kernel.rs`, `map.rs`, `reduce.rs`, `fuse.rs`, `block.rs`, `order.rs` | ✅ Complete |
| `Strided.jl/broadcast.jl` | `broadcast.rs` | ✅ Complete |
| `Strided.jl/convert.jl` | (via `copy_into`) | ✅ Complete |
| `Strided.jl/macros.jl` | N/A | ⚠️ Not needed (Rust type system) |

### Key Julia Functions → Rust Equivalents

| Julia | Rust | Notes |
|-------|------|-------|
| `StridedView(array)` | `StridedArrayView::new()` | Const-generic N |
| `sview(a, indices...)` | `view.slice()` | Zero-copy slicing |
| `sreshape(a, dims)` | `view.sreshape_strided()` | Stride-preserving reshape |
| `permutedims(a, perm)` | `view.permute()` | Zero-copy permutation |
| `Base.map!(f, dest, srcs...)` | `map_into`, `zip_map*_into` | Up to 4 sources |
| `Base.mapreducedim!(f, op, dest, src)` | `mapreducedim_into` | Dimension reduction |
| `promoteshape(dims, arrays...)` | `promoteshape`, `promoteshape2`, `promoteshape3` | Broadcasting |
| `CaptureArgs` | `CaptureArgs<F, A>` | Lazy broadcast |

## Reference Materials

The `extern/` directory contains reference implementations:
- `extern/Strided.jl/`: Original Julia implementation (fully ported)
- `extern/StridedViews.jl/`: Julia package defining the `StridedView` type (fully ported)
- `extern/mdarray/`: The Rust mdarray crate source for reference

Design documentation:
- `docs/STRIDED_DESIGN.md`: Detailed analysis of Julia implementations and Rust porting guide

## Key Design Decisions

1. **Const generics for dimension count**: `StridedView<T, N>` where `N` is const
2. **Type-level element operations**: Avoid runtime dispatch for conj/transpose via the `ElementOp` trait
3. **Result-based error handling**: Return `Result<_, StridedError>` for invalid operations
4. **Trait-based extensibility**: `ElementOp`, `Reduce`, `Map` traits for customization

## Remaining Work (TODO)

- Explore explicit SIMD intrinsics to close remaining gap with Julia's `@simd`

## Performance Notes

See `README.md` for benchmark results comparing Rust strided vs naive baselines and Julia Strided.jl.
