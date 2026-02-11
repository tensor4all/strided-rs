//! N-ary Einstein summation for `mdarray` arrays.
//!
//! This crate provides a thin wrapper over [`strided_opteinsum`] that accepts
//! `mdarray` `Array<T, DynRank>` and `View<T, DynRank>` types directly.
//! Row-major / column-major layout conversion is handled transparently.
//!
//! # Example
//!
//! ```ignore
//! use mdarray::{Array, DynRank};
//! use mdarray_opteinsum::einsum;
//!
//! let a: Array<f64, DynRank> = /* 3x4 matrix */;
//! let b: Array<f64, DynRank> = /* 4x5 matrix */;
//! let c: Array<f64, DynRank> = einsum("ij,jk->ik", vec![(&a).into(), (&b).into()])?;
//! ```

pub mod convert;

use mdarray::{Array, DynRank, View, ViewMut};
use num_complex::Complex64;
use strided_opteinsum::{EinsumOperand, EinsumScalar};

use crate::convert::{
    array_to_strided_view, reverse_notation, strided_array_to_mdarray,
    view_mut_to_strided_view_mut, view_to_strided_view,
};

/// Error type for mdarray-opteinsum operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Einsum(#[from] strided_opteinsum::EinsumError),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A type-erased einsum operand wrapping mdarray types.
///
/// Construct via `From` impls on owned arrays or views.
pub enum MdOperand<'a> {
    F64Array(&'a Array<f64, DynRank>),
    C64Array(&'a Array<Complex64, DynRank>),
    F64View(View<'a, f64, DynRank>),
    C64View(View<'a, Complex64, DynRank>),
}

impl<'a> From<&'a Array<f64, DynRank>> for MdOperand<'a> {
    fn from(arr: &'a Array<f64, DynRank>) -> Self {
        MdOperand::F64Array(arr)
    }
}

impl<'a> From<&'a Array<Complex64, DynRank>> for MdOperand<'a> {
    fn from(arr: &'a Array<Complex64, DynRank>) -> Self {
        MdOperand::C64Array(arr)
    }
}

impl<'a> From<View<'a, f64, DynRank>> for MdOperand<'a> {
    fn from(view: View<'a, f64, DynRank>) -> Self {
        MdOperand::F64View(view)
    }
}

impl<'a> From<View<'a, Complex64, DynRank>> for MdOperand<'a> {
    fn from(view: View<'a, Complex64, DynRank>) -> Self {
        MdOperand::C64View(view)
    }
}

/// Convert an `MdOperand` into a `strided_opteinsum::EinsumOperand` with reversed dims.
fn to_einsum_operand<'a>(op: &'a MdOperand<'a>) -> EinsumOperand<'a> {
    match op {
        MdOperand::F64Array(arr) => {
            let sv = array_to_strided_view(*arr);
            EinsumOperand::from_view(&sv)
        }
        MdOperand::C64Array(arr) => {
            let sv = array_to_strided_view(*arr);
            EinsumOperand::from_view(&sv)
        }
        MdOperand::F64View(view) => {
            let sv = view_to_strided_view(view);
            EinsumOperand::from_view(&sv)
        }
        MdOperand::C64View(view) => {
            let sv = view_to_strided_view(view);
            EinsumOperand::from_view(&sv)
        }
    }
}

/// Parse and evaluate an einsum expression on mdarray operands.
///
/// Returns the result as an owned `Array<T, DynRank>`.
///
/// # Example
///
/// ```ignore
/// let c: Array<f64, DynRank> = einsum("ij,jk->ik", vec![(&a).into(), (&b).into()])?;
/// ```
pub fn einsum<T: EinsumScalar>(
    notation: &str,
    operands: Vec<MdOperand<'_>>,
) -> Result<Array<T, DynRank>> {
    let reversed = reverse_notation(notation);
    let einsum_ops: Vec<EinsumOperand<'_>> =
        operands.iter().map(|op| to_einsum_operand(op)).collect();
    let result = strided_opteinsum::einsum(&reversed, einsum_ops)?;
    let data = T::extract_data(result)?;
    let strided_arr = data.into_array();
    Ok(strided_array_to_mdarray(strided_arr))
}

/// Parse and evaluate an einsum expression, writing the result into a
/// pre-allocated mdarray output with alpha/beta scaling.
///
/// `output = alpha * einsum(operands) + beta * output`
pub fn einsum_into<'a, T: EinsumScalar>(
    notation: &str,
    operands: Vec<MdOperand<'_>>,
    output: &'a mut ViewMut<'a, T, DynRank>,
    alpha: T,
    beta: T,
) -> Result<()> {
    let reversed = reverse_notation(notation);
    let einsum_ops: Vec<EinsumOperand<'_>> =
        operands.iter().map(|op| to_einsum_operand(op)).collect();
    let strided_out = view_mut_to_strided_view_mut(output);
    strided_opteinsum::einsum_into(&reversed, einsum_ops, strided_out, alpha, beta)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::mem::ManuallyDrop;

    /// Helper: create a row-major DynRank array from a Vec and shape.
    fn make_array(data: Vec<f64>, dims: &[usize]) -> Array<f64, DynRank> {
        let shape: DynRank = mdarray::Shape::from_dims(dims);
        let mut data = ManuallyDrop::new(data);
        let capacity = data.capacity();
        let ptr = data.as_mut_ptr();
        unsafe { Array::from_raw_parts(ptr, shape, capacity) }
    }

    /// Helper: read element at row-major indices from a DynRank array.
    fn get_elem(arr: &Array<f64, DynRank>, indices: &[usize]) -> f64 {
        let dims = arr.dims();
        let mut offset = 0usize;
        for (i, &idx) in indices.iter().enumerate() {
            let stride: usize = dims[i + 1..].iter().product();
            offset += idx * stride;
        }
        unsafe { *arr.as_ptr().add(offset) }
    }

    #[test]
    fn test_matmul_2x3_times_3x2() {
        // A = [[1,2,3],[4,5,6]]  (2x3, row-major)
        let a = make_array(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        // B = [[7,8],[9,10],[11,12]]  (3x2, row-major)
        let b = make_array(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);

        let c: Array<f64, DynRank> = einsum("ij,jk->ik", vec![(&a).into(), (&b).into()]).unwrap();

        assert_eq!(c.dims(), &[2, 2]);
        // C[0,0] = 1*7 + 2*9 + 3*11 = 58
        assert_abs_diff_eq!(get_elem(&c, &[0, 0]), 58.0, epsilon = 1e-10);
        // C[0,1] = 1*8 + 2*10 + 3*12 = 64
        assert_abs_diff_eq!(get_elem(&c, &[0, 1]), 64.0, epsilon = 1e-10);
        // C[1,0] = 4*7 + 5*9 + 6*11 = 139
        assert_abs_diff_eq!(get_elem(&c, &[1, 0]), 139.0, epsilon = 1e-10);
        // C[1,1] = 4*8 + 5*10 + 6*12 = 154
        assert_abs_diff_eq!(get_elem(&c, &[1, 1]), 154.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trace() {
        // A = [[1,2],[3,4]]  (2x2)
        let a = make_array(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let c: Array<f64, DynRank> = einsum("ii->", vec![(&a).into()]).unwrap();

        // Scalar result: trace = 1 + 4 = 5
        assert_eq!(c.dims(), &[] as &[usize]);
        assert_abs_diff_eq!(unsafe { *c.as_ptr() }, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_transpose() {
        // A = [[1,2,3],[4,5,6]]  (2x3)
        let a = make_array(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

        let c: Array<f64, DynRank> = einsum("ij->ji", vec![(&a).into()]).unwrap();

        assert_eq!(c.dims(), &[3, 2]);
        // C[0,0] = A[0,0] = 1
        assert_abs_diff_eq!(get_elem(&c, &[0, 0]), 1.0, epsilon = 1e-10);
        // C[1,0] = A[0,1] = 2
        assert_abs_diff_eq!(get_elem(&c, &[1, 0]), 2.0, epsilon = 1e-10);
        // C[0,1] = A[1,0] = 4
        assert_abs_diff_eq!(get_elem(&c, &[0, 1]), 4.0, epsilon = 1e-10);
        // C[2,1] = A[1,2] = 6
        assert_abs_diff_eq!(get_elem(&c, &[2, 1]), 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dot_product() {
        // a = [1,2,3], b = [4,5,6]
        let a = make_array(vec![1.0, 2.0, 3.0], &[3]);
        let b = make_array(vec![4.0, 5.0, 6.0], &[3]);

        let c: Array<f64, DynRank> = einsum("i,i->", vec![(&a).into(), (&b).into()]).unwrap();

        // dot = 1*4 + 2*5 + 3*6 = 32
        assert_eq!(c.dims(), &[] as &[usize]);
        assert_abs_diff_eq!(unsafe { *c.as_ptr() }, 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_outer_product() {
        // a = [1,2], b = [3,4,5]
        let a = make_array(vec![1.0, 2.0], &[2]);
        let b = make_array(vec![3.0, 4.0, 5.0], &[3]);

        let c: Array<f64, DynRank> = einsum("i,j->ij", vec![(&a).into(), (&b).into()]).unwrap();

        assert_eq!(c.dims(), &[2, 3]);
        // C[0,0] = 1*3 = 3
        assert_abs_diff_eq!(get_elem(&c, &[0, 0]), 3.0, epsilon = 1e-10);
        // C[0,2] = 1*5 = 5
        assert_abs_diff_eq!(get_elem(&c, &[0, 2]), 5.0, epsilon = 1e-10);
        // C[1,1] = 2*4 = 8
        assert_abs_diff_eq!(get_elem(&c, &[1, 1]), 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_three_operand_chain() {
        // A[2,3] * B[3,4] * C[4,2] -> result[2,2]
        let a = make_array((1..=6).map(|x| x as f64).collect(), &[2, 3]);
        let b = make_array((1..=12).map(|x| x as f64).collect(), &[3, 4]);
        let c = make_array((1..=8).map(|x| x as f64).collect(), &[4, 2]);

        let result: Array<f64, DynRank> =
            einsum("ij,jk,kl->il", vec![(&a).into(), (&b).into(), (&c).into()]).unwrap();

        assert_eq!(result.dims(), &[2, 2]);

        // Verify against manual computation:
        // AB = A * B (2x4), then ABC = AB * C (2x2)
        // AB[0,0] = 1*1 + 2*5 + 3*9 = 38
        // AB[0,1] = 1*2 + 2*6 + 3*10 = 44
        // AB[0,2] = 1*3 + 2*7 + 3*11 = 50
        // AB[0,3] = 1*4 + 2*8 + 3*12 = 56
        // ABC[0,0] = 38*1 + 44*3 + 50*5 + 56*7 = 38+132+250+392 = 812
        // ABC[0,1] = 38*2 + 44*4 + 50*6 + 56*8 = 76+176+300+448 = 1000
        assert_abs_diff_eq!(get_elem(&result, &[0, 0]), 812.0, epsilon = 1e-10);
        assert_abs_diff_eq!(get_elem(&result, &[0, 1]), 1000.0, epsilon = 1e-10);
    }

    #[test]
    fn test_view_input() {
        // Test that View inputs work
        let a = make_array(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = make_array(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);

        let va = a.expr();
        let vb = b.expr();

        let c: Array<f64, DynRank> = einsum("ij,jk->ik", vec![va.into(), vb.into()]).unwrap();

        assert_eq!(c.dims(), &[2, 2]);
        assert_abs_diff_eq!(get_elem(&c, &[0, 0]), 58.0, epsilon = 1e-10);
        assert_abs_diff_eq!(get_elem(&c, &[1, 1]), 154.0, epsilon = 1e-10);
    }

    #[test]
    fn test_einsum_into_matmul() {
        let a = make_array(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = make_array(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let mut c: Array<f64, DynRank> = Array::zeros(&[2usize, 2]);

        {
            let mut view = c.expr_mut();
            einsum_into(
                "ij,jk->ik",
                vec![(&a).into(), (&b).into()],
                &mut view,
                1.0,
                0.0,
            )
            .unwrap();
        }

        assert_abs_diff_eq!(get_elem(&c, &[0, 0]), 58.0, epsilon = 1e-10);
        assert_abs_diff_eq!(get_elem(&c, &[1, 1]), 154.0, epsilon = 1e-10);
    }
}
