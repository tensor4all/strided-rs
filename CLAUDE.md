# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mdarray-strided is a Rust library providing cache-optimized kernels for strided multidimensional array operations. It is a port of Julia's Strided.jl/StridedViews.jl libraries, built on top of the `mdarray` crate.

**Current Status (v0.1):**
- Broadcasting supported via `.broadcast()` method (stride-0 for size-1 dims)
- Zero-copy transformations: slice, reshape, permute, transpose
- Lazy element operations with type-level composition
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

| Module | Purpose |
|--------|---------|
| `view.rs` | `StridedArrayView`/`StridedArrayViewMut` types with slicing, permutation, reshape, broadcast |
| `element_op.rs` | Element operations (`Identity`, `Conj`, `Transpose`, `Adjoint`) with type-level composition |
| `kernel.rs` | `StridedView`/`StridedViewMut` internal wrappers, `KernelPlan` for iteration planning |
| `order.rs` | Dimension ordering algorithm - sorts dimensions by stride magnitude |
| `block.rs` | Block size computation to fit within L1 cache |
| `map.rs` | `map_into`, `zip_map2_into`, `zip_map3_into`, `zip_map4_into` |
| `reduce.rs` | `reduce`, `reduce_axis` |
| `ops.rs` | High-level operations: `copy_into`, `add`, `mul`, `axpy`, `fma`, `sum`, `dot`, `symmetrize_into` |
| `blas.rs` | BLAS integration: `is_blas_matrix`, `BlasFloat`, generic and BLAS-backed implementations |

### Cache Optimization Strategy

The library uses a blocking strategy inspired by Strided.jl:
1. **Dimension Reordering**: Dimensions are sorted by stride (largest first) to maximize cache locality
2. **Tiled Iteration**: Operations are blocked into tiles that fit in L1 cache
3. **Contiguous Fast Paths**: Contiguous arrays bypass the blocking machinery for direct iteration

### Key Constants

- `BLOCK_MEMORY_SIZE`: 32KB (L1 cache target)
- `CACHE_LINE_SIZE`: 64 bytes
- `TRANSPOSE_TILE`: 16 (tile size for transpose operations)

### Dependencies

- `mdarray` (v0.7.2): Base multidimensional array type
- `num-traits`/`num-complex`: Numeric trait bounds
- `thiserror`: Error type derivation
- `rayon` (optional): Parallel iteration
- `cblas` (optional): BLAS backend

## Reference Materials

The `extern/` directory contains reference implementations:
- `extern/Strided.jl/`: Original Julia implementation being ported
- `extern/StridedViews.jl/`: Julia package defining the `StridedView` type
- `extern/mdarray/`: The Rust mdarray crate source for reference

Design documentation:
- `docs/STRIDED_DESIGN.md`: Detailed analysis of Julia implementations and Rust porting guide
- `docs/report.md`: Benchmark report comparing strided vs naive implementations

## Key Design Decisions

1. **Const generics for dimension count**: `StridedView<T, N>` where `N` is const
2. **Type-level element operations**: Avoid runtime dispatch for conj/transpose via the `ElementOp` trait
3. **Result-based error handling**: Return `Result<_, StridedError>` for invalid operations
4. **Trait-based extensibility**: `ElementOp`, `Reduce`, `Map` traits for customization

## Remaining Work (TODO)

- BLAS `matmul_into` with automatic backend selection
- `CaptureArgs` equivalent for lazy broadcast evaluation
- False-sharing avoidance for parallel reductions
- Optimize blocking strategy for 4D arrays ([Issue #5](https://github.com/AtelierArith/strided-rs-private/issues/5))

## Performance Notes

See `docs/report.md` for detailed benchmark results. Summary:

| Operation | 2D Arrays | 4D Arrays |
|-----------|-----------|-----------|
| `zip_map` (contiguous) | 3-4x faster | Not tested |
| `zip_map` (mixed stride) | 2-2.6x faster | - |
| `symmetrize_into` | 1.5x faster | N/A |
| `permutedims` | ~same | **slower** (needs optimization) |
| `zip_map4_into` | Recommended | **slower** (needs optimization) |
