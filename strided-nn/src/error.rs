//! Errors a fused kernel returns for a caller-shape contract violation.

/// A softmax call whose arguments do not describe a valid axis-0 reduction, or
/// whose scratch is too small for the requested worker count.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SoftmaxError {
    /// `axis_len` is zero, so no column exists.
    #[error("softmax axis length must be positive")]
    EmptyAxis,

    /// The input element count is not a whole number of `axis_len` columns.
    #[error("input length {len} is not a multiple of axis length {axis_len}")]
    UnevenColumns { len: usize, axis_len: usize },

    /// The output does not have one slot per input element.
    #[error("input and output lengths differ: {input} vs {output}")]
    LengthMismatch { input: usize, output: usize },

    /// The scratch holds less than one cache-sized group of columns.
    #[error("softmax scratch holds {got} elements; {need} needed")]
    ScratchTooSmall { need: usize, got: usize },

    /// The column length overflowed the element-size arithmetic.
    #[error("axis length {axis_len} overflows the element size {element_size}")]
    SizeOverflow {
        axis_len: usize,
        element_size: usize,
    },
}

/// A normalization call whose arguments do not describe a valid `[rows, width]`
/// row layout, or whose scratch is too small.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum NormError {
    /// `width` is zero, so no row exists.
    #[error("normalization width must be positive")]
    EmptyWidth,

    /// The input element count is not a whole number of `width`-element rows.
    #[error("input length {len} is not a multiple of width {width}")]
    UnevenRows { len: usize, width: usize },

    /// A slice has the wrong length for the call's `[rows, width]` shape.
    #[error("`{name}` has {got} elements; {expected} expected")]
    LengthMismatch {
        name: &'static str,
        expected: usize,
        got: usize,
    },

    /// The scratch holds fewer elements than the parameter reduction needs.
    #[error("normalization scratch holds {got} elements; {need} needed")]
    ScratchTooSmall { need: usize, got: usize },

    /// The scratch size overflowed `usize`.
    #[error("scratch size for {rows} rows of width {width} overflows")]
    SizeOverflow { rows: usize, width: usize },
}

/// An activation call whose slices or scratch do not fit together.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ActivationError {
    /// A slice has the wrong length.
    #[error("`{name}` has {got} elements; {expected} expected")]
    LengthMismatch {
        name: &'static str,
        expected: usize,
        got: usize,
    },

    /// The scratch holds less than one chunk of elements.
    #[error("activation scratch holds {got} elements; {need} needed")]
    ScratchTooSmall { need: usize, got: usize },
}
