//! N-ary Einstein summation with nested contraction notation.
//!
//! This crate provides an einsum frontend that parses nested string
//! notation (e.g. `"(ij,jk),kl->il"`), supports mixed `f64` / `Complex64`
//! operands, and delegates pairwise contractions to [`strided_einsum2`].
//! For three or more tensors in a single contraction node the
//! [`omeco`] greedy optimizer is used to find an efficient pairwise order.
//!
//! # Quick start
//!
//! ```ignore
//! use strided_opteinsum::{einsum, EinsumOperand};
//!
//! let result = einsum("(ij,jk),kl->il", vec![a.into(), b.into(), c.into()], None)?;
//! ```

use std::collections::HashMap;

/// Error types for einsum operations.
pub mod error;
/// Recursive contraction-tree evaluation.
pub mod expr;
/// Type-erased einsum operands (`f64` / `Complex64`, owned / borrowed).
pub mod operand;
/// Nested einsum string parser.
pub mod parse;
/// Single-tensor operations (permute, trace, diagonal extraction).
pub mod single_tensor;
/// Runtime type dispatch over `f64` and `Complex64` tensors.
pub mod typed_tensor;

pub use error::{EinsumError, Result};
pub use operand::{EinsumOperand, EinsumScalar, StridedData};
pub use parse::{parse_einsum, EinsumCode, EinsumNode};
pub use typed_tensor::{needs_c64_promotion, TypedTensor};

/// Parse and evaluate an einsum expression in one call.
///
/// Pass `size_dict` to specify sizes for output indices not present in any
/// input (generative outputs like `"->ii"` or `"i->ij"`).
///
/// # Example
/// ```ignore
/// let result = einsum("(ij,jk),kl->il", vec![a.into(), b.into(), c.into()], None)?;
/// ```
pub fn einsum<'a>(
    notation: &str,
    operands: Vec<EinsumOperand<'a>>,
    size_dict: Option<&HashMap<char, usize>>,
) -> Result<EinsumOperand<'a>> {
    let code = parse_einsum(notation)?;
    code.evaluate(operands, size_dict)
}

/// Parse and evaluate an einsum expression, writing the result into a
/// pre-allocated output buffer with alpha/beta scaling.
///
/// `output = alpha * einsum(operands) + beta * output`
///
/// Pass `size_dict` to specify sizes for output indices not present in any
/// input (generative outputs like `"->ii"` or `"i->ij"`).
///
/// # Example
/// ```ignore
/// use strided_opteinsum::{einsum_into, EinsumOperand};
///
/// let mut c = StridedArray::<f64>::col_major(&[2, 2]);
/// einsum_into("ij,jk->ik", vec![a.into(), b.into()], c.view_mut(), 1.0, 0.0, None)?;
/// ```
pub fn einsum_into<T: EinsumScalar>(
    notation: &str,
    operands: Vec<EinsumOperand<'_>>,
    output: strided_view::StridedViewMut<T>,
    alpha: T,
    beta: T,
    size_dict: Option<&HashMap<char, usize>>,
) -> Result<()> {
    let code = parse_einsum(notation)?;
    code.evaluate_into(operands, output, alpha, beta, size_dict)
}
