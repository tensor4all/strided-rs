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

# Run tests
cargo test

# Run a single test
cargo test test_map_into_transposed

# Run benchmarks
cargo bench

# Run a specific benchmark
cargo bench -- copy_permuted
```

## Architecture

### Core Modules

- **lib.rs**: Public API exports and error types (`StridedError`)
- **element_op.rs**: Element-wise operations (`Identity`, `Conj`, `Transpose`, `Adjoint`) with type-level composition
- **view.rs**: `StridedArrayView` / `StridedArrayViewMut` - const-generic strided views with:
  - Lazy element operations (conj, transpose, adjoint)
  - Zero-copy slicing (`slice`, `slice_row`, `slice_col`)
  - Zero-copy permutation (`permute`, `t`, `h`)
  - Stride-preserving reshape (`reshape_1d`, `reshape_2d`)
- **kernel.rs**: Core kernel infrastructure
  - `StridedView`/`StridedViewMut`: Internal view wrappers over mdarray `Slice`
  - `KernelPlan`: Computed iteration order and block sizes for cache optimization
  - `for_each_offset`/`for_each_inner_block`: Block-based iteration primitives
- **order.rs**: Dimension ordering algorithm - sorts dimensions by stride magnitude to optimize cache access patterns
- **block.rs**: Block size computation - determines tile sizes to fit within L1 cache (`BLOCK_MEMORY_SIZE = 32KB`)
- **map.rs**: Mapping operations (`map_into`, `zip_map2_into`, `zip_map3_into`)
- **reduce.rs**: Reduction operations (`reduce`, `reduce_axis`)
- **ops.rs**: High-level operations built on map/reduce (`copy_into`, `add`, `mul`, `axpy`, `fma`, `sum`, `dot`, `symmetrize_into`, `copy_transpose_scale_into`)

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

## Reference Materials

The `extern/` directory contains reference implementations:
- `extern/Strided.jl/`: Original Julia implementation being ported
- `extern/StridedViews.jl/`: Julia package defining the `StridedView` type
- `extern/mdarray/`: The Rust mdarray crate source for reference

Design documentation:
- `docs/STRIDED_DESIGN.md`: Detailed analysis of Julia implementations and Rust porting guide

---

## Porting Goal: Strided.jl / StridedViews.jl → Rust

This project aims to port Julia's `Strided.jl` and `StridedViews.jl` packages to Rust, providing:

1. **Zero-copy strided views** over contiguous memory
2. **Cache-optimized iteration** with automatic blocking and loop reordering
3. **Multi-threaded execution** using rayon for large arrays
4. **BLAS integration** for BlasFloat types when beneficial

### Julia → Rust Feature Mapping

| Julia Feature | Rust Target | Status |
|--------------|-------------|--------|
| `StridedView` type | `StridedArrayView<T, N>` / `StridedArrayViewMut<T, N>` | **Done** |
| Element operations (identity/conj/transpose/adjoint) | `ElementOp` trait with type-level composition | **Done** |
| `sview` (slicing) | `.slice()`, `.slice_row()`, `.slice_col()` methods | **Done** |
| `sreshape` (stride-preserving reshape) | `.reshape_1d()`, `.reshape_2d()` methods | **Done** |
| `permutedims` | `.permute()`, `.t()`, `.h()` methods | **Done** |
| `map!` / `map` | `map_into` | Done |
| `mapreduce` | `reduce`, `reduce_axis` | Partial |
| Broadcasting (stride=0 for size-1 dims) | `.broadcast()`, `broadcast_shape` | **Done** |
| Dimension fusion (`_mapreduce_fuse!`) | `can_fuse_dims()`, `contiguous_inner_dims()` | **Done** |
| Loop reordering (`_mapreduce_order!`) | `order.rs` | Done |
| Block size computation (`_computeblocks`) | `block.rs` | Done |
| Multi-threading (`_mapreduce_threaded!`) | rayon `par_iter()` (feature-gated) | **Done** |
| `@strided` macro | N/A (Rust API design) | N/A |
| BLAS `mul!` | `matmul_into` with BLAS backend | TODO |

### Implementation Phases

#### Phase 1: Core StridedView Type ✅
- [x] Basic `StridedArrayView` / `StridedArrayViewMut` with const-generic dimensions
- [x] `KernelPlan` for iteration planning
- [x] Block-based iteration primitives
- [x] Negative stride support
- [x] Element operation traits (`Identity`, `Conj`, `Transpose`, `Adjoint`)
- [x] Type-level operation composition (group structure)

#### Phase 2: View Operations ✅
- [x] `sview`: Slicing with ranges, strided ranges, and dimension reduction
- [x] `sreshape`: Stride-preserving reshape with validation
- [x] `permutedims`: Zero-copy dimension permutation (`.permute()`)
- [x] 2D-specific: `.t()` (transpose), `.h()` (adjoint)

#### Phase 3: Broadcasting ✅
- [x] Stride-0 broadcasting for size-1 dimensions (`.broadcast()` method)
- [x] `broadcast_shape` / `broadcast_shape3` for size promotion
- [x] `can_broadcast_to` for compatibility checking
- [ ] `CaptureArgs` equivalent for lazy broadcast evaluation (future)

#### Phase 4: Advanced Optimizations ✅
- [x] Dimension fusion for reduced loop overhead (`can_fuse_dims`, `contiguous_inner_dims`, `contiguous_inner_len`)
- [x] SIMD-friendly contiguous inner loop detection (`as_slice` for contiguous views)
- [x] Iterators (`iter()`, `enumerate()`) with correct lifetime handling
- [x] Multi-threaded execution with rayon (`par_iter()`, feature-gated under `parallel`)
- [ ] False-sharing avoidance for reductions (future)

#### Phase 5: Linear Algebra Integration ✅
- [x] BLAS-compatible matrix detection (`is_blas_matrix`, `BlasLayout`, `BlasMatrix`)
- [x] `BlasFloat` trait for type dispatch (f32, f64, Complex32, Complex64)
- [x] Generic implementations (`generic_axpy`, `generic_dot`, `generic_gemm`)
- [x] BLAS-backed operations (feature-gated under `blas`):
  - [x] `blas_axpy`: y = alpha * x + y
  - [x] `blas_dot`: x · y
  - [x] `blas_gemm`: C = alpha * A * B + beta * C
- [x] `as_ptr` / `as_mut_ptr` methods for BLAS interop

### Key Design Decisions

1. **Const generics for dimension count**: `StridedView<T, N>` where `N` is const
2. **Type-level element operations**: Avoid runtime dispatch for conj/transpose
3. **Builder pattern for views**: `StridedViewBuilder::new(data).size(...).strides(...).build()`
4. **Result-based error handling**: Return `Result<_, StridedError>` for invalid operations
5. **Trait-based extensibility**: `ElementOp`, `Reduce`, `Map` traits for customization

### Performance Targets

- Match or exceed Julia's `Strided.jl` performance for:
  - `permutedims` on non-contiguous arrays
  - Broadcasting with transposed operands
  - Large-scale reductions
- Achieve near-BLAS performance for supported operations on BlasFloat types
