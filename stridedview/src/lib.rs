//! Device-agnostic strided view types and metadata operations.
//!
//! This crate is a Rust port of Julia's [StridedViews.jl](https://github.com/Jutho/StridedViews.jl),
//! providing strided multidimensional array view types with zero-copy metadata transformations.
//!
//! # Core Types
//!
//! - [`StridedView`] / [`StridedViewMut`]: Dynamic-rank strided views over existing data
//! - [`StridedArray`]: Owned strided multidimensional array
//! - [`ElementOp`] trait and implementations ([`Identity`], [`Conj`], [`Transpose`], [`Adjoint`]):
//!   Type-level element operations applied lazily on access
//!
//! # Metadata Transformations
//!
//! These operate only on dims/strides/offset and never access the underlying data:
//! - `permute`: Reorder dimensions
//! - `transpose_2d`, `adjoint_2d`: 2D matrix transformations
//! - `conj`: Compose conjugation operation
//! - `broadcast`: Expand size-1 dimensions

pub mod auxiliary;
mod element_op;
pub mod strided_view;

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
