pub mod error;
pub mod expr;
pub mod operand;
pub mod parse;
pub mod single_tensor;
pub mod typed_tensor;

pub use error::{EinsumError, Result};
pub use operand::{EinsumOperand, StridedData};
pub use parse::{parse_einsum, EinsumCode, EinsumNode};
pub use typed_tensor::{needs_c64_promotion, TypedTensor};

/// Parse and evaluate an einsum expression in one call.
///
/// # Example
/// ```ignore
/// let result = einsum("(ij,jk),kl->il", vec![a.into(), b.into(), c.into()])?;
/// ```
pub fn einsum(notation: &str, operands: Vec<EinsumOperand<'_>>) -> Result<EinsumOperand<'static>> {
    let code = parse_einsum(notation)?;
    code.evaluate(operands)
}
