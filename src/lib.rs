//! Cache-optimized kernels for strided multidimensional array operations.
//!
//! This crate is a Rust port of Julia's [Strided.jl](https://github.com/Jutho/Strided.jl)
//! and [StridedViews.jl](https://github.com/Jutho/StridedViews.jl) libraries, providing
//! efficient operations on strided multidimensional array views.
//!
//! # Core Types
//!
//! - [`StridedView`] / [`StridedViewMut`]: Dynamic-rank strided views over existing data
//! - [`StridedArray`]: Owned strided multidimensional array
//! - [`ElementOp`] trait and implementations ([`Identity`], [`Conj`], [`Transpose`], [`Adjoint`]):
//!   Type-level element operations applied lazily on access
//!
//! # Primary API (view-based, Julia-compatible)
//!
//! ## Map Operations
//!
//! - [`map_into`]: Apply a function element-wise from source to destination
//! - [`zip_map2_into`], [`zip_map3_into`], [`zip_map4_into`]: Multi-array element-wise operations
//!
//! ## Reduce Operations
//!
//! - [`reduce`]: Full reduction with map function
//! - [`reduce_axis`]: Reduce along a single axis
//!
//! ## Basic Operations
//!
//! - [`copy_into`]: Copy array contents
//! - [`add`], [`mul`]: Element-wise arithmetic
//! - [`axpy`]: y = alpha*x + y (array version)
//! - [`sum`], [`dot`]: Reductions
//! - [`symmetrize_into`], [`symmetrize_conj_into`]: Matrix symmetrization
//!
//! # Example
//!
//! ```rust
//! use strided_rs::{StridedView, StridedViewMut, StridedArray, Identity, map_into};
//!
//! // Create a column-major array (Julia default)
//! let src = StridedArray::<f64>::from_fn_col_major(&[2, 3], |idx| {
//!     (idx[0] * 10 + idx[1]) as f64
//! });
//! let mut dest = StridedArray::<f64>::col_major(&[2, 3]);
//!
//! // Map with view-based API
//! map_into(&mut dest.view_mut(), &src.view(), |x| x * 2.0).unwrap();
//! assert_eq!(dest.get(&[1, 2]), 24.0); // (1*10 + 2) * 2
//! ```
//!
//! # Cache Optimization
//!
//! The library uses Julia's blocking strategy for cache efficiency:
//! - Dimensions are sorted by stride magnitude for optimal memory access
//! - Operations are blocked into tiles fitting L1 cache ([`BLOCK_MEMORY_SIZE`] = 32KB)
//! - Contiguous arrays use fast paths bypassing the blocking machinery

mod auxiliary;
mod block;
mod element_op;
mod fuse;
mod kernel;
mod order;
pub mod strided_view;
#[cfg(feature = "parallel")]
mod threading;

// View-based operation modules
mod map_view;
mod ops_view;
mod reduce_view;

// ============================================================================
// Element operations
// ============================================================================
pub use element_op::{Adjoint, Compose, Conj, ElementOp, ElementOpApply, Identity, Transpose};

// ============================================================================
// View-based types
// ============================================================================
pub use strided_view::{
    col_major_strides, row_major_strides, StridedArray, StridedView, StridedViewMut,
};

// ============================================================================
// Map operations
// ============================================================================
pub use map_view::{map_into, zip_map2_into, zip_map3_into, zip_map4_into};

// ============================================================================
// High-level operations
// ============================================================================
pub use ops_view::{
    add, axpy, copy_conj, copy_into, copy_scale, copy_transpose_scale_into, dot, fma, mul, sum,
    symmetrize_conj_into, symmetrize_into,
};

// ============================================================================
// Reduce operations
// ============================================================================
pub use reduce_view::{reduce, reduce_axis};

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
