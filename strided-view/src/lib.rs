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
mod raw;
pub mod view;

// ============================================================================
// Element operations
// ============================================================================
pub use element_op::{
    Adjoint, ComposableElementOp, Compose, Conj, ElementOp, ElementOpApply, Identity, Transpose,
};

// ============================================================================
// View-based types
// ============================================================================
pub use raw::{
    ErasedRawStridedMut, ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut,
    KernelDType, KernelStorageElement, RawStridedMut, RawStridedRef,
};
pub use view::{col_major_strides, row_major_strides, StridedArray, StridedView, StridedViewMut};

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

    /// Mutable output layout maps multiple logical elements to the same memory offset.
    #[error("mutable output layout is not injective")]
    NonInjectiveOutputLayout,

    /// Runtime view layout does not match the layout a prepared plan was compiled for.
    #[error("view layout does not match the compiled plan")]
    PlanLayoutMismatch,

    /// Runtime dtype does not match the dtype a prepared plan was compiled for.
    #[error("dtype mismatch: expected {expected}, got {actual}")]
    DTypeMismatch {
        expected: &'static str,
        actual: &'static str,
    },

    /// A byte buffer length is not a whole number of dtype elements.
    #[error(
        "byte length {byte_len} is not a multiple of element size {element_size} for dtype {dtype}"
    )]
    ByteLengthMismatch {
        dtype: &'static str,
        byte_len: usize,
        element_size: usize,
    },

    /// A byte buffer pointer does not satisfy the dtype alignment.
    #[error("data pointer for dtype {dtype} is not aligned to {alignment} bytes")]
    DataAlignmentMismatch {
        dtype: &'static str,
        alignment: usize,
    },

    /// A byte buffer for `bool` contains a value that is not a valid Rust bool.
    #[error("invalid bool byte value {value}")]
    InvalidBoolByte { value: u8 },

    /// An execution context requested an invalid worker-thread budget.
    #[error("invalid thread budget {max_threads}")]
    InvalidThreadBudget { max_threads: usize },

    /// The dtype is recognized but unsupported by the selected operation.
    #[error("unsupported dtype {dtype}")]
    UnsupportedDType { dtype: &'static str },

    /// The dtype is recognized, but the selected op is not defined for it.
    #[error("unsupported op {op} for dtype {dtype}")]
    UnsupportedOp {
        op: &'static str,
        dtype: &'static str,
    },

    /// A mutable destination overlaps one of the operation inputs.
    #[error("destination overlaps input {input}")]
    OverlappingInputOutput { input: usize },

    /// Integer division or remainder encountered a zero divisor.
    #[error("integer {op} encountered a zero divisor")]
    IntegerDivisionByZero { op: &'static str },

    /// The operation arity is unsupported by this entry point.
    #[error("unsupported arity {arity}; maximum supported arity is {max}")]
    UnsupportedArity { arity: usize, max: usize },
}

/// Result type for strided array operations.
pub type Result<T> = std::result::Result<T, StridedError>;
