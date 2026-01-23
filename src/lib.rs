//! Cache-optimized kernels for strided mdarray views.
//!
//! This crate is a Rust port of Julia's Strided.jl/StridedViews.jl libraries.
//!
//! # Features
//!
//! - `parallel`: Enable rayon-based parallel iteration (`par_iter()`)
//! - `blas`: Enable BLAS-backed linear algebra operations
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

mod auxiliary;
pub mod blas;
mod block;
mod element_op;
mod fuse;
mod kernel;
mod map;
mod ops;
mod order;
mod reduce;
mod threading;
pub mod view;

pub use blas::{generic_axpy, generic_dot, generic_gemm};
pub use blas::{is_blas_matrix, is_contiguous_1d, BlasFloat, BlasLayout, BlasMatrix};
pub use element_op::{Adjoint, Compose, Conj, ElementOp, ElementOpApply, Identity, Transpose};
pub use map::{map_into, zip_map2_into, zip_map3_into, zip_map4_into};
#[cfg(feature = "parallel")]
pub use map::par_zip_map2_into;
pub use ops::{
    add, axpy, copy_conj, copy_into, copy_into_uninit, copy_scale, copy_transpose_scale_into,
    copy_transpose_scale_into_fast, dot, fma, mul, sum, symmetrize_conj_into, symmetrize_into,
};
pub use reduce::{reduce, reduce_axis};
pub use view::{
    broadcast_shape, broadcast_shape3, Idx, SliceIndex, StridedArrayView, StridedArrayViewMut,
    StridedRange,
};

#[cfg(feature = "blas")]
pub use blas::{blas_axpy, blas_dot, blas_gemm};

pub const BLOCK_MEMORY_SIZE: usize = 32 * 1024;
pub const CACHE_LINE_SIZE: usize = 64;
/// Minimum array length before threading is applied (Julia: MINTHREADLENGTH)
pub const MIN_THREAD_LENGTH: usize = 1 << 15; // 32768

#[derive(Debug, thiserror::Error)]
pub enum StridedError {
    #[error("rank mismatch: {0} vs {1}")]
    RankMismatch(usize, usize),
    #[error("shape mismatch: {0:?} vs {1:?}")]
    ShapeMismatch(Vec<usize>, Vec<usize>),
    #[error("invalid axis {axis} for rank {rank}")]
    InvalidAxis { axis: usize, rank: usize },
    #[error("invalid stride 0 for dim {dim}")]
    ZeroStride { dim: usize },
    #[error("stride and dims length mismatch")]
    StrideLengthMismatch,
    #[error("offset overflow while computing pointer")]
    OffsetOverflow,
    #[error("failed to convert scalar for scaling")]
    ScalarConversion,
    #[error("non-square matrix: rows={rows}, cols={cols}")]
    NonSquare { rows: usize, cols: usize },
}

pub type Result<T> = std::result::Result<T, StridedError>;
