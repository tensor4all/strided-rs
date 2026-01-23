//! Cache-optimized kernels for strided mdarray views.
//!
//! v0 behavior:
//! - All inputs must have identical shapes; broadcasting is not supported.
//! - Strides of 0 on dimensions with length > 1 are rejected.
//! - Overlapping src/dest memory is not supported; results are undefined if aliased.

mod block;
mod kernel;
mod map;
mod ops;
mod order;
mod reduce;

pub use map::{map_into, zip_map2_into, zip_map3_into};
pub use ops::{
    add, axpy, copy_conj, copy_into, copy_into_uninit, copy_scale, copy_transpose_scale_into,
    copy_transpose_scale_into_tiled, dot, fma, mul, sum, symmetrize_into,
};
pub use reduce::{reduce, reduce_axis};

pub const BLOCK_MEMORY_SIZE: usize = 32 * 1024;
pub const CACHE_LINE_SIZE: usize = 64;

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
