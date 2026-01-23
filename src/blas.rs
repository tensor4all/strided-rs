//! BLAS integration for optimized linear algebra operations.
//!
//! This module provides BLAS-backed implementations for common operations
//! when working with contiguous or stride-1 arrays.

use crate::element_op::{ElementOp, Identity};
use crate::view::{StridedArrayView, StridedArrayViewMut};
use crate::{Result, StridedError};

/// BLAS matrix layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlasLayout {
    /// Row-major (C-style): rows are contiguous, stride[1] == 1
    RowMajor,
    /// Column-major (Fortran-style): columns are contiguous, stride[0] == 1
    ColMajor,
}

/// Information about a BLAS-compatible matrix.
#[derive(Debug, Clone, Copy)]
pub struct BlasMatrix {
    pub layout: BlasLayout,
    pub rows: usize,
    pub cols: usize,
    pub ld: usize, // leading dimension
}

/// Check if a 2D view is compatible with BLAS operations.
///
/// A matrix is BLAS-compatible if either:
/// - Row-major: stride[1] == 1 and stride[0] >= cols
/// - Column-major: stride[0] == 1 and stride[1] >= rows
///
/// Returns `None` if the view is not BLAS-compatible.
pub fn is_blas_matrix<'a, T, Op: ElementOp>(
    view: &StridedArrayView<'a, T, 2, Op>,
) -> Option<BlasMatrix> {
    let size = view.size();
    let strides = view.strides();
    let rows = size[0];
    let cols = size[1];

    // Check for row-major layout (stride[1] == 1)
    if strides[1] == 1 && strides[0] >= cols as isize {
        return Some(BlasMatrix {
            layout: BlasLayout::RowMajor,
            rows,
            cols,
            ld: strides[0] as usize,
        });
    }

    // Check for column-major layout (stride[0] == 1)
    if strides[0] == 1 && strides[1] >= rows as isize {
        return Some(BlasMatrix {
            layout: BlasLayout::ColMajor,
            rows,
            cols,
            ld: strides[1] as usize,
        });
    }

    None
}

/// Check if a 1D view is contiguous (stride == 1 or stride == -1).
pub fn is_contiguous_1d<'a, T, Op: ElementOp>(view: &StridedArrayView<'a, T, 1, Op>) -> bool {
    let stride = view.strides()[0];
    stride == 1 || stride == -1
}

/// Trait for types that can be used with BLAS operations.
pub trait BlasFloat: Copy + Default + 'static {
    /// Check if this type is single precision (f32).
    fn is_single() -> bool;
    /// Check if this type is double precision (f64).
    fn is_double() -> bool;
    /// Check if this type is complex single precision.
    fn is_complex_single() -> bool;
    /// Check if this type is complex double precision.
    fn is_complex_double() -> bool;
}

impl BlasFloat for f32 {
    fn is_single() -> bool {
        true
    }
    fn is_double() -> bool {
        false
    }
    fn is_complex_single() -> bool {
        false
    }
    fn is_complex_double() -> bool {
        false
    }
}

impl BlasFloat for f64 {
    fn is_single() -> bool {
        false
    }
    fn is_double() -> bool {
        true
    }
    fn is_complex_single() -> bool {
        false
    }
    fn is_complex_double() -> bool {
        false
    }
}

impl BlasFloat for num_complex::Complex32 {
    fn is_single() -> bool {
        false
    }
    fn is_double() -> bool {
        false
    }
    fn is_complex_single() -> bool {
        true
    }
    fn is_complex_double() -> bool {
        false
    }
}

impl BlasFloat for num_complex::Complex64 {
    fn is_single() -> bool {
        false
    }
    fn is_double() -> bool {
        false
    }
    fn is_complex_single() -> bool {
        false
    }
    fn is_complex_double() -> bool {
        true
    }
}

// ============================================================================
// BLAS-backed operations (feature-gated)
// ============================================================================

#[cfg(feature = "blas")]
mod blas_impl {
    use super::*;
    use cblas::{Layout, Transpose};

    /// BLAS axpy: y = alpha * x + y
    ///
    /// Both x and y must be contiguous 1D views.
    pub fn blas_axpy<T: BlasFloat>(
        alpha: T,
        x: &StridedArrayView<'_, T, 1, Identity>,
        y: &mut StridedArrayViewMut<'_, T, 1, Identity>,
    ) -> Result<()> {
        let n = x.len();
        if n != y.len() {
            return Err(StridedError::ShapeMismatch(vec![n], vec![y.len()]));
        }

        let x_stride = x.strides()[0];
        let y_stride = y.strides()[0];

        // Get raw pointers
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_mut_ptr();

        unsafe {
            if T::is_double() {
                let alpha = std::mem::transmute_copy::<T, f64>(&alpha);
                let x_ptr = x_ptr as *const f64;
                let y_ptr = y_ptr as *mut f64;
                cblas::daxpy(
                    n as i32,
                    alpha,
                    std::slice::from_raw_parts(x_ptr, n),
                    x_stride as i32,
                    std::slice::from_raw_parts_mut(y_ptr, n),
                    y_stride as i32,
                );
            } else if T::is_single() {
                let alpha = std::mem::transmute_copy::<T, f32>(&alpha);
                let x_ptr = x_ptr as *const f32;
                let y_ptr = y_ptr as *mut f32;
                cblas::saxpy(
                    n as i32,
                    alpha,
                    std::slice::from_raw_parts(x_ptr, n),
                    x_stride as i32,
                    std::slice::from_raw_parts_mut(y_ptr, n),
                    y_stride as i32,
                );
            } else {
                // Fall back to generic implementation for complex types
                return Err(StridedError::ScalarConversion);
            }
        }

        Ok(())
    }

    /// BLAS dot: result = x · y
    ///
    /// Both x and y must be contiguous 1D views.
    pub fn blas_dot<T: BlasFloat>(
        x: &StridedArrayView<'_, T, 1, Identity>,
        y: &StridedArrayView<'_, T, 1, Identity>,
    ) -> Result<T> {
        let n = x.len();
        if n != y.len() {
            return Err(StridedError::ShapeMismatch(vec![n], vec![y.len()]));
        }

        let x_stride = x.strides()[0];
        let y_stride = y.strides()[0];

        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();

        unsafe {
            if T::is_double() {
                let x_ptr = x_ptr as *const f64;
                let y_ptr = y_ptr as *const f64;
                let result = cblas::ddot(
                    n as i32,
                    std::slice::from_raw_parts(x_ptr, n),
                    x_stride as i32,
                    std::slice::from_raw_parts(y_ptr, n),
                    y_stride as i32,
                );
                Ok(std::mem::transmute_copy(&result))
            } else if T::is_single() {
                let x_ptr = x_ptr as *const f32;
                let y_ptr = y_ptr as *const f32;
                let result = cblas::sdot(
                    n as i32,
                    std::slice::from_raw_parts(x_ptr, n),
                    x_stride as i32,
                    std::slice::from_raw_parts(y_ptr, n),
                    y_stride as i32,
                );
                Ok(std::mem::transmute_copy(&result))
            } else {
                Err(StridedError::ScalarConversion)
            }
        }
    }

    /// BLAS gemm: C = alpha * A * B + beta * C
    ///
    /// All matrices must be BLAS-compatible (row-major or column-major).
    pub fn blas_gemm<T: BlasFloat>(
        alpha: T,
        a: &StridedArrayView<'_, T, 2, Identity>,
        b: &StridedArrayView<'_, T, 2, Identity>,
        beta: T,
        c: &mut StridedArrayViewMut<'_, T, 2, Identity>,
    ) -> Result<()> {
        let a_info = is_blas_matrix(a).ok_or(StridedError::StrideLengthMismatch)?;
        let b_info = is_blas_matrix(b).ok_or(StridedError::StrideLengthMismatch)?;
        let c_info = is_blas_matrix_mut(c).ok_or(StridedError::StrideLengthMismatch)?;

        // Validate dimensions: A(m,k) * B(k,n) = C(m,n)
        let m = a_info.rows;
        let k = a_info.cols;
        let n = b_info.cols;

        if b_info.rows != k {
            return Err(StridedError::ShapeMismatch(
                vec![m, k],
                vec![b_info.rows, b_info.cols],
            ));
        }
        if c_info.rows != m || c_info.cols != n {
            return Err(StridedError::ShapeMismatch(
                vec![m, n],
                vec![c_info.rows, c_info.cols],
            ));
        }

        // Determine layout and transpose flags
        // CBLAS uses row-major by default, so we need to handle column-major carefully
        let layout = Layout::RowMajor;

        let (trans_a, lda) = match a_info.layout {
            BlasLayout::RowMajor => (Transpose::None, a_info.ld),
            BlasLayout::ColMajor => (Transpose::Ordinary, a_info.ld),
        };

        let (trans_b, ldb) = match b_info.layout {
            BlasLayout::RowMajor => (Transpose::None, b_info.ld),
            BlasLayout::ColMajor => (Transpose::Ordinary, b_info.ld),
        };

        let ldc = match c_info.layout {
            BlasLayout::RowMajor => c_info.ld,
            BlasLayout::ColMajor => {
                // For column-major C, we need to compute C^T = B^T * A^T
                // This is more complex, so for now we require row-major C
                return Err(StridedError::StrideLengthMismatch);
            }
        };

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.as_mut_ptr();

        unsafe {
            if T::is_double() {
                let alpha = std::mem::transmute_copy::<T, f64>(&alpha);
                let beta = std::mem::transmute_copy::<T, f64>(&beta);
                cblas::dgemm(
                    layout,
                    trans_a,
                    trans_b,
                    m as i32,
                    n as i32,
                    k as i32,
                    alpha,
                    std::slice::from_raw_parts(a_ptr as *const f64, m * k),
                    lda as i32,
                    std::slice::from_raw_parts(b_ptr as *const f64, k * n),
                    ldb as i32,
                    beta,
                    std::slice::from_raw_parts_mut(c_ptr as *mut f64, m * n),
                    ldc as i32,
                );
            } else if T::is_single() {
                let alpha = std::mem::transmute_copy::<T, f32>(&alpha);
                let beta = std::mem::transmute_copy::<T, f32>(&beta);
                cblas::sgemm(
                    layout,
                    trans_a,
                    trans_b,
                    m as i32,
                    n as i32,
                    k as i32,
                    alpha,
                    std::slice::from_raw_parts(a_ptr as *const f32, m * k),
                    lda as i32,
                    std::slice::from_raw_parts(b_ptr as *const f32, k * n),
                    ldb as i32,
                    beta,
                    std::slice::from_raw_parts_mut(c_ptr as *mut f32, m * n),
                    ldc as i32,
                );
            } else {
                return Err(StridedError::ScalarConversion);
            }
        }

        Ok(())
    }

    /// Check if a mutable 2D view is compatible with BLAS operations.
    fn is_blas_matrix_mut<'a, T, Op: ElementOp>(
        view: &StridedArrayViewMut<'a, T, 2, Op>,
    ) -> Option<BlasMatrix> {
        let size = view.size();
        let strides = view.strides();
        let rows = size[0];
        let cols = size[1];

        // Check for row-major layout (stride[1] == 1)
        if strides[1] == 1 && strides[0] >= cols as isize {
            return Some(BlasMatrix {
                layout: BlasLayout::RowMajor,
                rows,
                cols,
                ld: strides[0] as usize,
            });
        }

        // Check for column-major layout (stride[0] == 1)
        if strides[0] == 1 && strides[1] >= rows as isize {
            return Some(BlasMatrix {
                layout: BlasLayout::ColMajor,
                rows,
                cols,
                ld: strides[1] as usize,
            });
        }

        None
    }
}

#[cfg(feature = "blas")]
pub use blas_impl::{blas_axpy, blas_dot, blas_gemm};

// ============================================================================
// Generic fallback implementations (always available)
// ============================================================================

use crate::element_op::ElementOpApply;
use std::ops::{Add, Mul};

/// Generic axpy: y = alpha * x + y
///
/// Works with any strided views.
pub fn generic_axpy<'a, T>(
    alpha: T,
    x: &StridedArrayView<'a, T, 1, Identity>,
    y: &mut StridedArrayViewMut<'a, T, 1, Identity>,
) -> Result<()>
where
    T: ElementOpApply + Copy + Add<Output = T> + Mul<Output = T>,
{
    let n = x.len();
    if n != y.len() {
        return Err(StridedError::ShapeMismatch(vec![n], vec![y.len()]));
    }

    for i in 0..n {
        let xi = x.get([i]);
        let yi = y.get([i]);
        y.set([i], alpha * xi + yi);
    }

    Ok(())
}

/// Generic dot product: result = x · y
///
/// Works with any strided views.
pub fn generic_dot<'a, T>(
    x: &StridedArrayView<'a, T, 1, Identity>,
    y: &StridedArrayView<'a, T, 1, Identity>,
) -> Result<T>
where
    T: ElementOpApply + Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    let n = x.len();
    if n != y.len() {
        return Err(StridedError::ShapeMismatch(vec![n], vec![y.len()]));
    }

    let mut sum = T::default();
    for i in 0..n {
        sum = sum + x.get([i]) * y.get([i]);
    }

    Ok(sum)
}

/// Generic matrix multiplication: C = alpha * A * B + beta * C
///
/// Works with any strided views.
pub fn generic_gemm<'a, T>(
    alpha: T,
    a: &StridedArrayView<'a, T, 2, Identity>,
    b: &StridedArrayView<'a, T, 2, Identity>,
    beta: T,
    c: &mut StridedArrayViewMut<'a, T, 2, Identity>,
) -> Result<()>
where
    T: ElementOpApply + Copy + Add<Output = T> + Mul<Output = T> + Default,
{
    let (m, k1) = (a.size()[0], a.size()[1]);
    let (k2, n) = (b.size()[0], b.size()[1]);
    let (cm, cn) = (c.size()[0], c.size()[1]);

    if k1 != k2 {
        return Err(StridedError::ShapeMismatch(vec![m, k1], vec![k2, n]));
    }
    if cm != m || cn != n {
        return Err(StridedError::ShapeMismatch(vec![m, n], vec![cm, cn]));
    }

    let k = k1;

    for i in 0..m {
        for j in 0..n {
            let mut sum = T::default();
            for l in 0..k {
                sum = sum + a.get([i, l]) * b.get([l, j]);
            }
            let cij = c.get([i, j]);
            c.set([i, j], alpha * sum + beta * cij);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_blas_matrix_row_major() {
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        // Row-major 3x4 matrix: stride = [4, 1]
        let view: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&data, [3, 4], [4, 1], 0).unwrap();

        let info = is_blas_matrix(&view).unwrap();
        assert_eq!(info.layout, BlasLayout::RowMajor);
        assert_eq!(info.rows, 3);
        assert_eq!(info.cols, 4);
        assert_eq!(info.ld, 4);
    }

    #[test]
    fn test_is_blas_matrix_col_major() {
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        // Column-major 3x4 matrix: stride = [1, 3]
        let view: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&data, [3, 4], [1, 3], 0).unwrap();

        let info = is_blas_matrix(&view).unwrap();
        assert_eq!(info.layout, BlasLayout::ColMajor);
        assert_eq!(info.rows, 3);
        assert_eq!(info.cols, 4);
        assert_eq!(info.ld, 3);
    }

    #[test]
    fn test_is_blas_matrix_non_contiguous() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        // Non-contiguous: stride = [8, 2] (skip elements)
        let view: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&data, [3, 4], [8, 2], 0).unwrap();

        assert!(is_blas_matrix(&view).is_none());
    }

    #[test]
    fn test_generic_dot() {
        let x_data = vec![1.0, 2.0, 3.0, 4.0];
        let y_data = vec![5.0, 6.0, 7.0, 8.0];

        let x: StridedArrayView<'_, f64, 1, Identity> =
            StridedArrayView::new(&x_data, [4], [1], 0).unwrap();
        let y: StridedArrayView<'_, f64, 1, Identity> =
            StridedArrayView::new(&y_data, [4], [1], 0).unwrap();

        let result = generic_dot(&x, &y).unwrap();
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert_eq!(result, 70.0);
    }

    #[test]
    fn test_generic_axpy() {
        let x_data = vec![1.0, 2.0, 3.0];
        let mut y_data = vec![10.0, 20.0, 30.0];

        let x: StridedArrayView<'_, f64, 1, Identity> =
            StridedArrayView::new(&x_data, [3], [1], 0).unwrap();
        let mut y: StridedArrayViewMut<'_, f64, 1, Identity> =
            StridedArrayViewMut::new(&mut y_data, [3], [1], 0).unwrap();

        generic_axpy(2.0, &x, &mut y).unwrap();
        // y = 2 * x + y = [2, 4, 6] + [10, 20, 30] = [12, 24, 36]
        assert_eq!(y_data, vec![12.0, 24.0, 36.0]);
    }

    #[test]
    fn test_generic_gemm() {
        // A = [[1, 2], [3, 4]] (2x2)
        // B = [[5, 6], [7, 8]] (2x2)
        // C = A * B = [[19, 22], [43, 50]]
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![5.0, 6.0, 7.0, 8.0];
        let mut c_data = vec![0.0; 4];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [2, 2], [2, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [2, 2], [2, 1], 0).unwrap();
        let mut c: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut c_data, [2, 2], [2, 1], 0).unwrap();

        generic_gemm(1.0, &a, &b, 0.0, &mut c).unwrap();

        assert_eq!(c_data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_generic_gemm_with_alpha_beta() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![5.0, 6.0, 7.0, 8.0];
        let mut c_data = vec![1.0, 1.0, 1.0, 1.0];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [2, 2], [2, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [2, 2], [2, 1], 0).unwrap();
        let mut c: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut c_data, [2, 2], [2, 1], 0).unwrap();

        // C = 2 * A * B + 3 * C = 2 * [[19, 22], [43, 50]] + 3 * [[1, 1], [1, 1]]
        //   = [[38, 44], [86, 100]] + [[3, 3], [3, 3]] = [[41, 47], [89, 103]]
        generic_gemm(2.0, &a, &b, 3.0, &mut c).unwrap();

        assert_eq!(c_data, vec![41.0, 47.0, 89.0, 103.0]);
    }
}
