/// Errors that can occur during einsum parsing or evaluation.
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

    #[error("operand count mismatch: expected {expected}, found {found}")]
    OperandCountMismatch { expected: usize, found: usize },

    #[error("type mismatch: output is {output_type} but computation requires {computed_type}")]
    TypeMismatch {
        output_type: &'static str,
        computed_type: &'static str,
    },

    #[error("output shape mismatch: expected {expected:?}, got {got:?}")]
    OutputShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience alias for `Result<T, EinsumError>`.
pub type Result<T> = std::result::Result<T, EinsumError>;
