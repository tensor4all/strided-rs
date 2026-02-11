//! N-ary Einstein summation for `ndarray` arrays.
//!
//! This crate provides a thin wrapper over [`strided_opteinsum`] that accepts
//! `ndarray` `ArrayD<T>` and `ArrayViewD<T>` types directly.
//! Dims and strides are passed through without reversal (ndarray has explicit strides).
//!
//! # Example
//!
//! ```ignore
//! use ndarray::ArrayD;
//! use ndarray_opteinsum::einsum;
//!
//! let a = ArrayD::from_shape_vec(vec![2, 3], (1..=6).map(|x| x as f64).collect()).unwrap();
//! let b = ArrayD::from_shape_vec(vec![3, 2], (7..=12).map(|x| x as f64).collect()).unwrap();
//! let c: ArrayD<f64> = einsum("ij,jk->ik", vec![(&a).into(), (&b).into()]).unwrap();
//! ```

pub mod convert;

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use num_complex::Complex64;
use strided_opteinsum::{EinsumOperand, EinsumScalar};

use crate::convert::{
    array_to_strided_view, strided_array_to_ndarray, view_mut_to_strided_view_mut,
    view_to_strided_view,
};

/// Error type for ndarray-opteinsum operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Einsum(#[from] strided_opteinsum::EinsumError),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A type-erased einsum operand wrapping ndarray types.
///
/// Construct via `From` impls on owned arrays or views.
pub enum NdOperand<'a> {
    F64Array(&'a ArrayD<f64>),
    C64Array(&'a ArrayD<Complex64>),
    F64View(ArrayViewD<'a, f64>),
    C64View(ArrayViewD<'a, Complex64>),
}

impl<'a> From<&'a ArrayD<f64>> for NdOperand<'a> {
    fn from(arr: &'a ArrayD<f64>) -> Self {
        NdOperand::F64Array(arr)
    }
}

impl<'a> From<&'a ArrayD<Complex64>> for NdOperand<'a> {
    fn from(arr: &'a ArrayD<Complex64>) -> Self {
        NdOperand::C64Array(arr)
    }
}

impl<'a> From<ArrayViewD<'a, f64>> for NdOperand<'a> {
    fn from(view: ArrayViewD<'a, f64>) -> Self {
        NdOperand::F64View(view)
    }
}

impl<'a> From<ArrayViewD<'a, Complex64>> for NdOperand<'a> {
    fn from(view: ArrayViewD<'a, Complex64>) -> Self {
        NdOperand::C64View(view)
    }
}

/// Convert an `NdOperand` into a `strided_opteinsum::EinsumOperand`.
fn to_einsum_operand<'a>(op: &NdOperand<'a>) -> EinsumOperand<'a> {
    match op {
        NdOperand::F64Array(arr) => {
            let sv = array_to_strided_view(arr);
            EinsumOperand::from_view(&sv)
        }
        NdOperand::C64Array(arr) => {
            let sv = array_to_strided_view(arr);
            EinsumOperand::from_view(&sv)
        }
        NdOperand::F64View(view) => {
            let sv = view_to_strided_view(view);
            EinsumOperand::from_view(&sv)
        }
        NdOperand::C64View(view) => {
            let sv = view_to_strided_view(view);
            EinsumOperand::from_view(&sv)
        }
    }
}

/// Parse and evaluate an einsum expression on ndarray operands.
///
/// Returns the result as an owned `ArrayD<T>`.
///
/// # Example
///
/// ```ignore
/// let c: ArrayD<f64> = einsum("ij,jk->ik", vec![(&a).into(), (&b).into()])?;
/// ```
pub fn einsum<T: EinsumScalar>(notation: &str, operands: Vec<NdOperand<'_>>) -> Result<ArrayD<T>> {
    let einsum_ops: Vec<EinsumOperand<'_>> =
        operands.iter().map(|op| to_einsum_operand(op)).collect();
    let result = strided_opteinsum::einsum(notation, einsum_ops)?;
    let data = T::extract_data(result)?;
    let strided_arr = data.into_array();
    Ok(strided_array_to_ndarray(strided_arr))
}

/// Parse and evaluate an einsum expression, writing the result into a
/// pre-allocated ndarray output with alpha/beta scaling.
///
/// `output = alpha * einsum(operands) + beta * output`
pub fn einsum_into<'a, T: EinsumScalar>(
    notation: &str,
    operands: Vec<NdOperand<'_>>,
    output: &'a mut ArrayViewMutD<'a, T>,
    alpha: T,
    beta: T,
) -> Result<()> {
    let einsum_ops: Vec<EinsumOperand<'_>> =
        operands.iter().map(|op| to_einsum_operand(op)).collect();
    let strided_out = view_mut_to_strided_view_mut(output);
    strided_opteinsum::einsum_into(notation, einsum_ops, strided_out, alpha, beta)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::ArrayD;

    fn make_array(data: Vec<f64>, shape: Vec<usize>) -> ArrayD<f64> {
        ArrayD::from_shape_vec(shape, data).unwrap()
    }

    #[test]
    fn test_matmul_2x3_times_3x2() {
        let a = make_array(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = make_array(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

        let c: ArrayD<f64> = einsum("ij,jk->ik", vec![(&a).into(), (&b).into()]).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        assert_abs_diff_eq!(c[[0, 0]], 58.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[0, 1]], 64.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[1, 0]], 139.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[1, 1]], 154.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trace() {
        let a = make_array(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let c: ArrayD<f64> = einsum("ii->", vec![(&a).into()]).unwrap();

        assert_eq!(c.shape(), &[] as &[usize]);
        assert_abs_diff_eq!(c[[]], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_transpose() {
        let a = make_array(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let c: ArrayD<f64> = einsum("ij->ji", vec![(&a).into()]).unwrap();

        assert_eq!(c.shape(), &[3, 2]);
        assert_abs_diff_eq!(c[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[1, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[0, 1]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[2, 1]], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dot_product() {
        let a = make_array(vec![1.0, 2.0, 3.0], vec![3]);
        let b = make_array(vec![4.0, 5.0, 6.0], vec![3]);

        let c: ArrayD<f64> = einsum("i,i->", vec![(&a).into(), (&b).into()]).unwrap();

        assert_eq!(c.shape(), &[] as &[usize]);
        assert_abs_diff_eq!(c[[]], 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_outer_product() {
        let a = make_array(vec![1.0, 2.0], vec![2]);
        let b = make_array(vec![3.0, 4.0, 5.0], vec![3]);

        let c: ArrayD<f64> = einsum("i,j->ij", vec![(&a).into(), (&b).into()]).unwrap();

        assert_eq!(c.shape(), &[2, 3]);
        assert_abs_diff_eq!(c[[0, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[0, 2]], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[1, 1]], 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_three_operand_chain() {
        let a = make_array((1..=6).map(|x| x as f64).collect(), vec![2, 3]);
        let b = make_array((1..=12).map(|x| x as f64).collect(), vec![3, 4]);
        let c = make_array((1..=8).map(|x| x as f64).collect(), vec![4, 2]);

        let result: ArrayD<f64> =
            einsum("ij,jk,kl->il", vec![(&a).into(), (&b).into(), (&c).into()]).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_abs_diff_eq!(result[[0, 0]], 812.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], 1000.0, epsilon = 1e-10);
    }

    #[test]
    fn test_view_input() {
        let a = make_array(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = make_array(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

        let va = a.view();
        let vb = b.view();

        let c: ArrayD<f64> = einsum(
            "ij,jk->ik",
            vec![
                va.into_dimensionality().unwrap().into(),
                vb.into_dimensionality().unwrap().into(),
            ],
        )
        .unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        assert_abs_diff_eq!(c[[0, 0]], 58.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[1, 1]], 154.0, epsilon = 1e-10);
    }

    #[test]
    fn test_einsum_into_matmul() {
        let a = make_array(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = make_array(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let mut c = ArrayD::<f64>::zeros(vec![2, 2]);

        {
            let mut view = c.view_mut();
            einsum_into(
                "ij,jk->ik",
                vec![(&a).into(), (&b).into()],
                &mut view,
                1.0,
                0.0,
            )
            .unwrap();
        }

        assert_abs_diff_eq!(c[[0, 0]], 58.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c[[1, 1]], 154.0, epsilon = 1e-10);
    }
}
