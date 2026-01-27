//! Cache-optimized kernels for strided mdarray views.
//!
//! This crate is a Rust port of Julia's [Strided.jl](https://github.com/Jutho/Strided.jl)
//! and [StridedViews.jl](https://github.com/Jutho/StridedViews.jl) libraries, providing
//! efficient operations on strided multidimensional array views.
//!
//! # Feature Flags
//!
//! - **`parallel`**: Enable rayon-based parallel iteration via [`par_zip_map2_into`].
//!   Uses Julia-style divide-and-conquer threading for large arrays.
//!
//! - **`blas`**: Enable CBLAS-backed linear algebra operations ([`blas_axpy`], [`blas_dot`],
//!   [`blas_gemm`]). Requires a CBLAS implementation to be available.
//!
//! # Core Types
//!
//! - [`StridedArrayView`] / [`StridedArrayViewMut`]: Zero-copy strided views over existing data
//! - [`ElementOp`] trait and implementations ([`Identity`], [`Conj`], [`Transpose`], [`Adjoint`]):
//!   Type-level element operations applied lazily on access
//!
//! # Primary API (Julia-compatible)
//!
//! ## Map Operations
//!
//! - [`map_into`]: Apply a function element-wise from source to destination
//! - [`zip_map2_into`], [`zip_map3_into`], [`zip_map4_into`]: Multi-array element-wise operations
//! - [`par_zip_map2_into`] (with `parallel` feature): Parallel binary map
//!
//! ## Reduce Operations
//!
//! - [`reduce`]: Full reduction with map function
//! - [`reduce_axis`]: Reduce along a single axis
//! - [`mapreducedim_into`]: Map-reduce along dimensions (Julia's `mapreducedim!`)
//!
//! ## Broadcast Operations
//!
//! - [`broadcast_into`], [`broadcast3_into`]: Broadcasting with automatic shape promotion
//! - [`promoteshape`], [`promoteshape2`], [`promoteshape3`]: Explicit shape promotion for broadcasting
//! - [`CaptureArgs`]: Lazy broadcast expression builder (Julia's `CaptureArgs`)
//!
//! ## Linear Algebra
//!
//! - [`matmul`], [`generic_matmul`]: Matrix multiplication with alpha/beta scaling
//! - [`linalg::axpy`](linalg_axpy), [`axpby`]: Vector operations (y = alpha*x + y, y = alpha*x + beta*y)
//! - [`lmul`], [`rmul`]: Scalar multiplication
//! - [`isblasmatrix`], [`isblasmatrix_identity`], [`isblasmatrix_conj`]: BLAS compatibility checks
//!
//! ## Basic Operations
//!
//! - [`copy_into`]: Copy array contents
//! - [`add`], [`mul`]: Element-wise arithmetic
//! - [`axpy`]: y = alpha*x + y (array version)
//! - [`fma`]: Fused multiply-add
//! - [`sum`], [`dot`]: Reductions
//! - [`symmetrize_into`], [`symmetrize_conj_into`]: Matrix symmetrization
//!
//! # Example
//!
//! ```rust
//! use mdarray_strided::{StridedArrayView, Identity};
//!
//! // Create a 2D view over existing data
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let view: StridedArrayView<'_, f64, 2, Identity> =
//!     StridedArrayView::new(&data, [2, 3], [3, 1], 0).unwrap();
//!
//! // Access elements
//! assert_eq!(view.get([0, 0]), 1.0);
//! assert_eq!(view.get([1, 2]), 6.0);
//!
//! // Transpose (zero-copy)
//! let transposed = view.t();
//! assert_eq!(transposed.size(), &[3, 2]);
//! ```
//!
//! # Broadcasting Example
//!
//! ```rust
//! use mdarray_strided::{StridedArrayView, StridedArrayViewMut, Identity, broadcast_into};
//!
//! // Broadcast [1, 3] to [4, 3] and add element-wise
//! let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
//! let b_data = vec![10.0, 20.0, 30.0]; // Row vector
//! let mut dest_data = vec![0.0; 12];
//!
//! let a: StridedArrayView<'_, f64, 2, Identity> =
//!     StridedArrayView::new(&a_data, [4, 3], [3, 1], 0).unwrap();
//! let b: StridedArrayView<'_, f64, 2, Identity> =
//!     StridedArrayView::new(&b_data, [1, 3], [3, 1], 0).unwrap();
//! let mut dest: StridedArrayViewMut<'_, f64, 2, Identity> =
//!     StridedArrayViewMut::new(&mut dest_data, [4, 3], [3, 1], 0).unwrap();
//!
//! broadcast_into(&mut dest, |x, y| x + y, &a, &b).unwrap();
//! ```
//!
//! # Cache Optimization
//!
//! The library uses Julia's blocking strategy for cache efficiency:
//! - Dimensions are sorted by stride magnitude for optimal memory access
//! - Operations are blocked into tiles fitting L1 cache ([`BLOCK_MEMORY_SIZE`] = 32KB)
//! - Contiguous arrays use fast paths bypassing the blocking machinery

mod auxiliary;
pub mod blas;
mod block;
pub mod broadcast;
mod element_op;
mod fuse;
mod kernel;
pub mod linalg;
mod map;
mod ops;
mod order;
mod promote;
mod reduce;
mod threading;
pub mod view;

// ============================================================================
// BLAS module exports
// ============================================================================
pub use blas::{generic_axpy, generic_dot, generic_gemm};
pub use blas::{is_blas_matrix, is_contiguous_1d, BlasFloat, BlasLayout, BlasMatrix};

// ============================================================================
// Element operations
// ============================================================================
pub use element_op::{Adjoint, Compose, Conj, ElementOp, ElementOpApply, Identity, Transpose};

// ============================================================================
// Map operations
// ============================================================================
#[cfg(feature = "parallel")]
pub use map::par_zip_map2_into;
pub use map::{map_into, zip_map2_into, zip_map3_into, zip_map4_into};

// ============================================================================
// High-level operations
// ============================================================================
pub use ops::{
    add, axpy, copy_conj, copy_into, copy_into_uninit, copy_scale, copy_transpose_scale_into,
    copy_transpose_scale_into_fast, dot, fma, mul, sum, symmetrize_conj_into, symmetrize_into,
};

// ============================================================================
// Reduce operations
// ============================================================================
pub use reduce::{
    mapreducedim_capture1_into, mapreducedim_capture2_into, mapreducedim_capture3_into,
    mapreducedim_capture4_into, mapreducedim_capture2_views_into,
    mapreducedim_capture3_views_into, mapreducedim_capture4_views_into,
    mapreducedim_capture_views_into,
    mapreducedim_into, reduce, reduce_axis,
};

// ============================================================================
// View types and utilities
// ============================================================================
pub use view::{
    broadcast_shape, broadcast_shape3, Idx, SliceIndex, StridedArrayView, StridedArrayViewMut,
    StridedRange,
};

// ============================================================================
// Broadcast operations (CaptureArgs for lazy evaluation)
// ============================================================================
pub use broadcast::{
    broadcast3_into, broadcast_into, promoteshape, promoteshape2, promoteshape3, Arg,
    BroadcastBuilder, CaptureArgs, Consume, Scalar,
};

// ============================================================================
// Linear algebra operations
// ============================================================================
/// AXPY operation for 1D arrays (y = alpha*x + y).
///
/// This is an alias for [`linalg::axpy`] to avoid name collision with the
/// array-based [`axpy`] function in ops module.
pub use linalg::axpy as linalg_axpy;
pub use linalg::{
    axpby, generic_matmul, getblasmatrix_identity, isblasmatrix, isblasmatrix_conj,
    isblasmatrix_identity, lmul, matmul, rmul, BlasTranspose,
};

// ============================================================================
// BLAS-backed operations (requires "blas" feature)
// ============================================================================
#[cfg(feature = "blas")]
pub use blas::{blas_axpy, blas_dot, blas_gemm};

// ============================================================================
// Constants
// ============================================================================

/// Block memory size for cache-optimized iteration (L1 cache target).
///
/// Operations are blocked into tiles that fit within this size to maximize cache hits.
/// Default: 32KB (typical L1 data cache size).
pub const BLOCK_MEMORY_SIZE: usize = 32 * 1024;

/// Cache line size in bytes.
///
/// Used for memory region calculations in block size computation.
pub const CACHE_LINE_SIZE: usize = 64;

/// Minimum array length before threading is applied.
///
/// Arrays smaller than this will be processed single-threaded even with the
/// `parallel` feature enabled. This avoids thread overhead for small arrays.
///
/// Julia equivalent: `MINTHREADLENGTH`
pub const MIN_THREAD_LENGTH: usize = 1 << 15; // 32768

// ============================================================================
// Error types
// ============================================================================

/// Errors that can occur during strided array operations.
#[derive(Debug, thiserror::Error)]
pub enum StridedError {
    /// Array ranks do not match.
    #[error("rank mismatch: {0} vs {1}")]
    RankMismatch(usize, usize),

    /// Array shapes are incompatible for the operation.
    #[error("shape mismatch: {0:?} vs {1:?}")]
    ShapeMismatch(Vec<usize>, Vec<usize>),

    /// Invalid axis index for the given array rank.
    #[error("invalid axis {axis} for rank {rank}")]
    InvalidAxis { axis: usize, rank: usize },

    /// Zero stride is not allowed for the specified dimension.
    #[error("invalid stride 0 for dim {dim}")]
    ZeroStride { dim: usize },

    /// Stride array length doesn't match dimensions.
    #[error("stride and dims length mismatch")]
    StrideLengthMismatch,

    /// Integer overflow while computing array offset.
    #[error("offset overflow while computing pointer")]
    OffsetOverflow,

    /// Failed to convert a scalar value for scaling operation.
    #[error("failed to convert scalar for scaling")]
    ScalarConversion,

    /// Matrix is not square when a square matrix was required.
    #[error("non-square matrix: rows={rows}, cols={cols}")]
    NonSquare { rows: usize, cols: usize },
}

/// Result type for strided array operations.
pub type Result<T> = std::result::Result<T, StridedError>;
