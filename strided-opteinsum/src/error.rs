#[derive(Debug, thiserror::Error)]
pub enum EinsumError {
    #[error("parse error: {0}")]
    ParseError(String),

    #[error(transparent)]
    Strided(#[from] strided_view::StridedError),

    #[error(transparent)]
    Einsum2(#[from] strided_einsum2::EinsumError),

    #[error("dimension mismatch for axis '{axis}': {dim_a} vs {dim_b}")]
    DimensionMismatch {
        axis: String,
        dim_a: usize,
        dim_b: usize,
    },

    #[error("output axis '{0}' not found in any input")]
    OrphanOutputAxis(String),
}

pub type Result<T> = std::result::Result<T, EinsumError>;
