# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mdarray-strided is a Rust library providing cache-optimized kernels for strided multidimensional array operations. It is a **complete port of Julia's Strided.jl/StridedViews.jl libraries**, built on top of the `mdarray` crate.

**Current Status (v0.1):**
- Julia Strided.jl/StridedViews.jl port: **~98% complete**
- Broadcasting with `CaptureArgs` for lazy evaluation (stride-0 for size-1 dims)
- Zero-copy transformations: slice, reshape, permute, transpose
- Lazy element operations with type-level composition (`Identity`, `Conj`, `Transpose`, `Adjoint`)
- Linear algebra: `matmul`, `generic_matmul`, `axpy`, `axpby`, `lmul`, `rmul`
- Optional features: `parallel` (rayon), `blas` (CBLAS)
- Overlapping src/dest memory is not supported

## Build Commands

```bash
# Build
cargo build

# Build with all features
cargo build --all-features

# Run all tests
cargo test

# Run a single test
cargo test test_map_into_transposed

# Run tests with features
cargo test --features parallel
cargo test --features blas
cargo test --all-features

# Run benchmarks
cargo bench

# Run a specific benchmark
cargo bench -- copy_permuted
```

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
| `map.rs` | `map_into`, `zip_map2_into`, `zip_map3_into`, `zip_map4_into`, `par_zip_map2_into` | `Base.map!` |
| `reduce.rs` | `reduce`, `reduce_axis`, `mapreducedim_into` | `Base.mapreduce`, `Base.mapreducedim!` |
| `broadcast.rs` | `CaptureArgs`, `promoteshape`, `broadcast_into`, `Arg`, `Scalar` | `broadcast.jl` |
| `linalg.rs` | `matmul`, `generic_matmul`, `axpy`, `axpby`, `lmul`, `rmul`, `isblasmatrix` | `linalg.jl` |
| `ops.rs` | High-level operations: `copy_into`, `add`, `mul`, `axpy`, `fma`, `sum`, `dot`, `symmetrize_into` | Various |
| `blas.rs` | BLAS integration: `is_blas_matrix`, `BlasFloat`, generic and BLAS-backed implementations | BLAS dispatch |
| `order.rs` | Dimension ordering algorithm - sorts dimensions by stride magnitude | `indexorder` |
| `block.rs` | Block size computation to fit within L1 cache (`_computeblocks`) | `_computeblocks` |
| `fuse.rs` | Dimension fusion for contiguous dimensions | `_mapreduce_fuse!` |
| `threading.rs` | Parallel divide-and-conquer threading | `_mapreduce_threaded!` |
| `auxiliary.rs` | Helper functions: `index_order`, `normalize_strides`, `simplify_dims` | `auxiliary.jl` |

### Cache Optimization Strategy

The library uses a blocking strategy faithful to Strided.jl:
1. **Dimension Fusion**: Contiguous dimensions are fused to reduce loop overhead (`_mapreduce_fuse!`)
2. **Dimension Reordering**: Dimensions are sorted by stride importance for optimal cache access (`_mapreduce_order!`)
3. **Tiled Iteration**: Operations are blocked into tiles fitting L1 cache (`_computeblocks`)
4. **Contiguous Fast Paths**: Contiguous arrays bypass blocking for direct iteration
5. **Parallel Divide-and-Conquer**: Large arrays split recursively across threads (`_mapreduce_threaded!`)

### Key Constants

| Constant | Value | Julia Equivalent |
|----------|-------|------------------|
| `BLOCK_MEMORY_SIZE` | 32KB | `BLOCKMEMORYSIZE` |
| `CACHE_LINE_SIZE` | 64 bytes | `_cachelinelength` |
| `MIN_THREAD_LENGTH` | 32768 | `MINTHREADLENGTH` |

### Dependencies

- `mdarray` (v0.7.2): Base multidimensional array type
- `num-traits`/`num-complex`: Numeric trait bounds
- `thiserror`: Error type derivation
- `rayon` (optional): Parallel iteration
- `cblas` (optional): BLAS backend

## Julia Port Status

### Fully Ported (98%)

| Julia Module | Rust Module | Status |
|--------------|-------------|--------|
| `StridedViews.jl/stridedview.jl` | `view.rs` | ✅ Complete |
| `StridedViews.jl/auxiliary.jl` | `auxiliary.rs` | ✅ Complete |
| `Strided.jl/mapreduce.jl` | `kernel.rs`, `map.rs`, `reduce.rs`, `fuse.rs`, `block.rs`, `order.rs` | ✅ Complete |
| `Strided.jl/broadcast.jl` | `broadcast.rs` | ✅ Complete |
| `Strided.jl/linalg.jl` | `linalg.rs` | ✅ Complete |
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
| `LinearAlgebra.mul!(C, A, B, α, β)` | `matmul(c, a, b, alpha, beta)` | BLAS-compatible |
| `promoteshape(dims, arrays...)` | `promoteshape`, `promoteshape2`, `promoteshape3` | Broadcasting |
| `CaptureArgs` | `CaptureArgs<F, A>` | Lazy broadcast |

## Reference Materials

The `extern/` directory contains reference implementations:
- `extern/Strided.jl/`: Original Julia implementation (fully ported)
- `extern/StridedViews.jl/`: Julia package defining the `StridedView` type (fully ported)
- `extern/mdarray/`: The Rust mdarray crate source for reference

Design documentation:
- `docs/STRIDED_DESIGN.md`: Detailed analysis of Julia implementations and Rust porting guide
- `docs/report.md`: Benchmark report comparing strided vs naive implementations

## Key Design Decisions

1. **Const generics for dimension count**: `StridedView<T, N>` where `N` is const
2. **Type-level element operations**: Avoid runtime dispatch for conj/transpose via the `ElementOp` trait
3. **Result-based error handling**: Return `Result<_, StridedError>` for invalid operations
4. **Trait-based extensibility**: `ElementOp`, `Reduce`, `Map` traits for customization
5. **Feature-gated optional backends**: `parallel` for rayon, `blas` for CBLAS

## Remaining Work (TODO)

- Optimize blocking strategy for 4D arrays ([Issue #5](https://github.com/AtelierArith/strided-rs-private/issues/5))
- Dynamic thread count control (`set_num_threads()` / `get_num_threads()` equivalent)
- Further false-sharing avoidance optimization for parallel reductions

## Performance Notes

See `docs/report.md` for detailed benchmark results. Summary:

| Operation | 2D Arrays | 4D Arrays |
|-----------|-----------|-----------|
| `zip_map` (contiguous) | 3-4x faster | Not tested |
| `zip_map` (mixed stride) | 2-2.6x faster | - |
| `symmetrize_into` | 1.5x faster | N/A |
| `permutedims` | ~same | **slower** (needs optimization) |
| `zip_map4_into` | Recommended | **slower** (needs optimization) |
