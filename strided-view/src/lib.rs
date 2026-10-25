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
pub mod injectivity;
mod raw;
pub mod view;

// ============================================================================
// Inline metadata capacity
// ============================================================================

/// Maximum rank whose metadata is stored inline without a heap allocation.
///
/// [`StridedView`], [`StridedViewMut`] and [`StridedArray`] keep up to this
/// many dimensions and strides inline in a fixed-size `Copy` buffer. Derived
/// views,
/// clones and array constructors are allocation-free at ranks 1 through
/// `INLINE_RANK`; a higher rank spills to the heap gracefully (never an
/// error).
///
/// # Examples
///
/// ```
/// use strided_view::{StridedView, INLINE_RANK};
///
/// assert_eq!(INLINE_RANK, 8);
/// let data = [0.0f64; 8];
/// let view = StridedView::<f64>::new(&data, &[2, 2, 2], &[1, 2, 4], 0).unwrap();
/// assert_eq!(view.ndim(), 3);
/// ```
pub const INLINE_RANK: usize = 8;

/// Stride metadata with inline storage for up to [`INLINE_RANK`] dimensions.
///
/// Returned by [`view::col_major_strides`] and [`view::row_major_strides`],
/// so a computed layout at rank `<= INLINE_RANK` does not allocate.
///
/// # Examples
///
/// ```
/// use strided_view::StrideVec;
///
/// let strides: StrideVec = [1isize, 2, 4].into_iter().collect();
/// assert_eq!(&strides[..], &[1, 2, 4]);
/// ```
pub type StrideVec = tinyvec::TinyVec<[isize; INLINE_RANK]>;

/// Axis metadata with inline storage for up to [`INLINE_RANK`] dimensions.
///
/// Returned by [`auxiliary::index_order`].
///
/// # Examples
///
/// ```
/// use strided_view::AxisVec;
///
/// let axes: AxisVec = [3usize, 1, 2].into_iter().collect();
/// assert_eq!(&axes[..], &[3, 1, 2]);
/// ```
pub type AxisVec = tinyvec::TinyVec<[usize; INLINE_RANK]>;

// ============================================================================
// Element operations
// ============================================================================
pub use element_op::{
    Adjoint, ComposableElementOp, Compose, Conj, ElementOp, ElementOpApply, Identity, Transpose,
};

// ============================================================================
// Layout validation
// ============================================================================
pub use injectivity::is_injective_layout;
#[doc(hidden)]
pub use injectivity::is_injective_layout_without_alloc;

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
