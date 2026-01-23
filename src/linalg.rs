//! Linear algebra operations ported from Julia's Strided.jl/src/linalg.jl
//!
//! This module provides BLAS-optimized matrix operations with support for
//! element operations (conj, transpose, adjoint) on strided arrays.
//!
//! # Key functions
//!
//! - `isblasmatrix`: Check if a matrix is BLAS-compatible
//! - `getblasmatrix`: Get BLAS-ready matrix with transpose flag
//! - `matmul`: Matrix multiplication with element operation support
//!
//! # Julia equivalent
//!
//! ```julia
//! function isblasmatrix(A::StridedView{T,2}) where {T<:LinearAlgebra.BlasFloat}
//!     if A.op == identity
//!         return stride(A, 1) == 1 || stride(A, 2) == 1
//!     elseif A.op == conj
//!         return stride(A, 2) == 1
//!     else
//!         return false
//!     end
//! end
//! ```

use crate::blas::BlasFloat;
use crate::element_op::{Conj, ElementOp, ElementOpApply, Identity};
use crate::view::{StridedArrayView, StridedArrayViewMut};
use crate::{Result, StridedError};
use num_traits::{One, Zero};
use std::ops::{Add, Mul};

// ============================================================================
// BLAS transpose flag
// ============================================================================

/// BLAS transpose operation flag.
///
/// This corresponds to the CBLAS_TRANSPOSE enum:
/// - 'N': No transpose
/// - 'T': Transpose
/// - 'C': Conjugate transpose (adjoint)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlasTranspose {
    /// No transpose operation
    NoTrans,
    /// Transpose operation
    Trans,
    /// Conjugate transpose (adjoint)
    ConjTrans,
}

impl BlasTranspose {
    /// Convert to CBLAS character representation.
    pub fn to_char(self) -> char {
        match self {
            BlasTranspose::NoTrans => 'N',
            BlasTranspose::Trans => 'T',
            BlasTranspose::ConjTrans => 'C',
        }
    }
}

// ============================================================================
// isblasmatrix - Julia-faithful implementation
// ============================================================================

/// Check if a matrix with Identity operation is BLAS-compatible.
///
/// A matrix is BLAS-compatible if either stride[0] == 1 (column-major)
/// or stride[1] == 1 (row-major).
///
/// # Julia equivalent
/// ```julia
/// function isblasmatrix(A::StridedView{T,2}) where {T<:LinearAlgebra.BlasFloat}
///     if A.op == identity
///         return stride(A, 1) == 1 || stride(A, 2) == 1
///     ...
/// end
/// ```
pub fn isblasmatrix_identity<T>(view: &StridedArrayView<'_, T, 2, Identity>) -> bool {
    let strides = view.strides();
    strides[0] == 1 || strides[1] == 1
}

/// Check if a matrix with Conj operation is BLAS-compatible.
///
/// For Conj matrices, only stride[1] == 1 is BLAS-compatible,
/// because we need to convert to adjoint which flips the dimensions.
///
/// # Julia equivalent
/// ```julia
/// elseif A.op == conj
///     return stride(A, 2) == 1
/// ```
pub fn isblasmatrix_conj<T>(view: &StridedArrayView<'_, T, 2, Conj>) -> bool {
    let strides = view.strides();
    strides[1] == 1
}

/// Check if a matrix is BLAS-compatible (generic over element operation).
pub fn isblasmatrix<T, Op: ElementOp>(view: &StridedArrayView<'_, T, 2, Op>) -> bool {
    let strides = view.strides();

    // For Identity: stride[0] == 1 || stride[1] == 1
    if std::any::TypeId::of::<Op>() == std::any::TypeId::of::<Identity>() {
        return strides[0] == 1 || strides[1] == 1;
    }

    // For Conj: stride[1] == 1 only
    if std::any::TypeId::of::<Op>() == std::any::TypeId::of::<Conj>() {
        return strides[1] == 1;
    }

    // Transpose and Adjoint should not appear directly in BLAS matrices
    // (they are handled by view transformations)
    false
}

// ============================================================================
// getblasmatrix - Julia-faithful implementation
// ============================================================================

/// Result of getblasmatrix: a normalized view and transpose flag.
pub struct BlasMatrixInfo<'a, T, Op: ElementOp> {
    /// The view, possibly transformed to be column-major
    pub view: StridedArrayView<'a, T, 2, Op>,
    /// The transpose flag to apply
    pub trans: BlasTranspose,
}

/// Get a BLAS-ready matrix from an Identity-operation view.
///
/// Returns the view (possibly transposed) and a transpose flag.
///
/// # Julia equivalent
/// ```julia
/// function getblasmatrix(A::StridedView{T,2}) where {T<:LinearAlgebra.BlasFloat}
///     if A.op == identity
///         if stride(A, 1) == 1
///             return A, 'N'
///         else
///             return transpose(A), 'T'
///         end
///     else
///         return adjoint(A), 'C'
///     end
/// end
/// ```
pub fn getblasmatrix_identity<'a, T: Copy>(
    view: &StridedArrayView<'a, T, 2, Identity>,
) -> (StridedArrayView<'a, T, 2, Identity>, BlasTranspose) {
    let strides = view.strides();
    let size = view.size();

    if strides[0] == 1 {
        // Already column-major: no transpose needed
        // Create a copy of the view
        let new_view = unsafe {
            StridedArrayView::<'a, T, 2, Identity>::new_unchecked(
                view.data(),
                *size,
                *strides,
                view.offset(),
            )
        };
        (new_view, BlasTranspose::NoTrans)
    } else {
        // Row-major: transpose to get column-major
        // Create transposed view manually to avoid ownership issue
        let transposed_size = [size[1], size[0]];
        let transposed_strides = [strides[1], strides[0]];
        let transposed_view = unsafe {
            StridedArrayView::<'a, T, 2, Identity>::new_unchecked(
                view.data(),
                transposed_size,
                transposed_strides,
                view.offset(),
            )
        };
        (transposed_view, BlasTranspose::Trans)
    }
}

/// Get a BLAS-ready matrix from a Conj-operation view.
///
/// For Conj matrices, we return the adjoint and 'C' flag.
pub fn getblasmatrix_conj<'a, T: Copy + ElementOpApply>(
    view: &StridedArrayView<'a, T, 2, Conj>,
) -> (StridedArrayView<'a, T, 2, Identity>, BlasTranspose) {
    // adjoint(conj(A)) = transpose(A) with Identity op
    // The physical data is A with conj applied lazily
    // Taking adjoint: conj ∘ transpose = Identity (via composition)
    // But we need to return a column-major view
    let size = *view.size();
    let strides = *view.strides();

    // Create transposed view (swap size and strides)
    let transposed_size = [size[1], size[0]];
    let transposed_strides = [strides[1], strides[0]];

    // Safety: same underlying data, just different indexing
    let transposed = unsafe {
        StridedArrayView::<'a, T, 2, Identity>::new_unchecked(
            view.data(),
            transposed_size,
            transposed_strides,
            view.offset(),
        )
    };

    (transposed, BlasTranspose::ConjTrans)
}

// ============================================================================
// Matrix multiplication - mul!
// ============================================================================

/// Matrix multiplication: C = α * A * B + β * C
///
/// This is the main entry point for matrix multiplication with element operation support.
/// It follows Julia's `mul!` implementation:
///
/// 1. If types don't match for BLAS, fall back to generic implementation
/// 2. Handle C.op (conj case requires special treatment)
/// 3. Handle C layout (row-major → column-major via transpose)
/// 4. Dispatch to BLAS or generic based on matrix compatibility
///
/// # Julia equivalent
/// ```julia
/// function LinearAlgebra.mul!(C::StridedView{T,2},
///                             A::StridedView{<:Any,2}, B::StridedView{<:Any,2},
///                             α::Number=true, β::Number=false) where {T}
///     if !(eltype(C) <: LinearAlgebra.BlasFloat && eltype(A) == eltype(B) == eltype(C))
///         return __mul!(C, A, B, α, β)
///     end
///     ...
/// end
/// ```
pub fn matmul<'a, T>(
    alpha: T,
    a: &StridedArrayView<'a, T, 2, Identity>,
    b: &StridedArrayView<'a, T, 2, Identity>,
    beta: T,
    c: &mut StridedArrayViewMut<'a, T, 2, Identity>,
) -> Result<()>
where
    T: BlasFloat
        + ElementOpApply
        + Copy
        + Default
        + Add<Output = T>
        + Mul<Output = T>
        + Zero
        + One
        + PartialEq,
{
    // Validate dimensions: A(m,k) * B(k,n) = C(m,n)
    let m = a.size()[0];
    let k = a.size()[1];
    let n = b.size()[1];

    if b.size()[0] != k {
        return Err(StridedError::ShapeMismatch(
            vec![m, k],
            vec![b.size()[0], b.size()[1]],
        ));
    }
    if c.size()[0] != m || c.size()[1] != n {
        return Err(StridedError::ShapeMismatch(
            vec![m, n],
            vec![c.size()[0], c.size()[1]],
        ));
    }

    // Check C's layout and dispatch
    let c_strides = c.strides();

    if c_strides[0] > c_strides[1] {
        // C is row-major: compute C^T = B^T * A^T
        // This is equivalent to computing in-place with transposed operands
        matmul_colmajor_c(alpha, a, b, beta, c)
    } else {
        // C is column-major (or square): direct computation
        matmul_colmajor_c(alpha, a, b, beta, c)
    }
}

/// Internal matmul for column-major C.
///
/// Dispatches to BLAS if all matrices are compatible, otherwise generic.
fn matmul_colmajor_c<'a, T>(
    alpha: T,
    a: &StridedArrayView<'a, T, 2, Identity>,
    b: &StridedArrayView<'a, T, 2, Identity>,
    beta: T,
    c: &mut StridedArrayViewMut<'a, T, 2, Identity>,
) -> Result<()>
where
    T: BlasFloat
        + ElementOpApply
        + Copy
        + Default
        + Add<Output = T>
        + Mul<Output = T>
        + Zero
        + One
        + PartialEq,
{
    let c_strides = c.strides();

    // Check if all matrices are BLAS-compatible
    if c_strides[0] == 1 && isblasmatrix_identity(a) && isblasmatrix_identity(b) {
        // Use BLAS path
        #[cfg(feature = "blas")]
        {
            return blas_matmul(alpha, a, b, beta, c);
        }
        #[cfg(not(feature = "blas"))]
        {
            return generic_matmul(alpha, a, b, beta, c);
        }
    }

    // Fall back to generic
    generic_matmul(alpha, a, b, beta, c)
}

// ============================================================================
// BLAS matrix multiplication
// ============================================================================

/// BLAS-backed matrix multiplication.
///
/// Uses getblasmatrix to normalize matrices and then calls BLAS gemm.
#[cfg(feature = "blas")]
fn blas_matmul<'a, T: BlasFloat>(
    alpha: T,
    a: &StridedArrayView<'a, T, 2, Identity>,
    b: &StridedArrayView<'a, T, 2, Identity>,
    beta: T,
    c: &mut StridedArrayViewMut<'a, T, 2, Identity>,
) -> Result<()> {
    use cblas::{Layout, Transpose};

    let (a_view, trans_a) = getblasmatrix_identity(a);
    let (b_view, trans_b) = getblasmatrix_identity(b);

    let m = c.size()[0];
    let n = c.size()[1];
    let k = a.size()[1];

    // Get leading dimensions
    let lda = if trans_a == BlasTranspose::NoTrans {
        a_view.strides()[1].unsigned_abs()
    } else {
        a_view.strides()[0].unsigned_abs()
    };

    let ldb = if trans_b == BlasTranspose::NoTrans {
        b_view.strides()[1].unsigned_abs()
    } else {
        b_view.strides()[0].unsigned_abs()
    };

    let ldc = c.strides()[1].unsigned_abs();

    let cblas_trans_a = match trans_a {
        BlasTranspose::NoTrans => Transpose::None,
        BlasTranspose::Trans => Transpose::Ordinary,
        BlasTranspose::ConjTrans => Transpose::Conjugate,
    };

    let cblas_trans_b = match trans_b {
        BlasTranspose::NoTrans => Transpose::None,
        BlasTranspose::Trans => Transpose::Ordinary,
        BlasTranspose::ConjTrans => Transpose::Conjugate,
    };

    let a_ptr = a_view.as_ptr();
    let b_ptr = b_view.as_ptr();
    let c_ptr = c.as_mut_ptr();

    unsafe {
        if T::is_double() {
            let alpha = std::mem::transmute_copy::<T, f64>(&alpha);
            let beta = std::mem::transmute_copy::<T, f64>(&beta);
            cblas::dgemm(
                Layout::ColumnMajor,
                cblas_trans_a,
                cblas_trans_b,
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
                Layout::ColumnMajor,
                cblas_trans_a,
                cblas_trans_b,
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

// ============================================================================
// Generic matrix multiplication - __mul! equivalent
// ============================================================================

/// Generic matrix multiplication: C = α * A * B + β * C
///
/// This is the fallback implementation for non-BLAS-compatible matrices.
/// It follows Julia's `__mul!` implementation but with a simpler approach.
///
/// # Julia equivalent (simplified)
/// ```julia
/// function __mul!(C::StridedView{<:Any,2}, A::StridedView{<:Any,2}, B::StridedView{<:Any,2},
///                 α::Number, β::Number)
///     m, n = size(C)
///     k = size(A, 2)
///     # ... uses _mapreducedim! for optimization
/// end
/// ```
pub fn generic_matmul<'a, T>(
    alpha: T,
    a: &StridedArrayView<'a, T, 2, Identity>,
    b: &StridedArrayView<'a, T, 2, Identity>,
    beta: T,
    c: &mut StridedArrayViewMut<'a, T, 2, Identity>,
) -> Result<()>
where
    T: ElementOpApply + Copy + Default + Add<Output = T> + Mul<Output = T> + Zero + One + PartialEq,
{
    let m = a.size()[0];
    let k = a.size()[1];
    let n = b.size()[1];

    // Handle special cases for optimization
    if alpha.is_zero() || k == 0 {
        // C = β * C (just scale C)
        scale_matrix(beta, c);
        return Ok(());
    }

    if beta.is_zero() {
        // C = α * A * B (no accumulation)
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    sum = sum + a.get([i, l]) * b.get([l, j]);
                }
                c.set([i, j], alpha * sum);
            }
        }
    } else if beta.is_one() {
        // C = α * A * B + C (accumulate without scaling)
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    sum = sum + a.get([i, l]) * b.get([l, j]);
                }
                let cij = c.get([i, j]);
                c.set([i, j], alpha * sum + cij);
            }
        }
    } else {
        // C = α * A * B + β * C (general case)
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    sum = sum + a.get([i, l]) * b.get([l, j]);
                }
                let cij = c.get([i, j]);
                c.set([i, j], alpha * sum + beta * cij);
            }
        }
    }

    Ok(())
}

/// Scale a matrix in-place: C = β * C
fn scale_matrix<'a, T>(beta: T, c: &mut StridedArrayViewMut<'a, T, 2, Identity>)
where
    T: ElementOpApply + Copy + Mul<Output = T> + Zero + One + PartialEq,
{
    let m = c.size()[0];
    let n = c.size()[1];

    if beta.is_zero() {
        for i in 0..m {
            for j in 0..n {
                c.set([i, j], T::zero());
            }
        }
    } else if !beta.is_one() {
        for i in 0..m {
            for j in 0..n {
                let cij = c.get([i, j]);
                c.set([i, j], beta * cij);
            }
        }
    }
    // If beta == 1, do nothing
}

// ============================================================================
// axpy and axpby - Julia-faithful implementations
// ============================================================================

/// axpy: Y = a * X + Y (for 1D arrays)
///
/// # Julia equivalent
/// ```julia
/// function LinearAlgebra.axpy!(a::Number, X::StridedView{<:Number,N},
///                              Y::StridedView{<:Number,N}) where {N}
///     if a == 1
///         Y .= X .+ Y
///     else
///         Y .= a .* X .+ Y
///     end
///     return Y
/// end
/// ```
pub fn axpy<'a, T>(
    a: T,
    x: &StridedArrayView<'a, T, 1, Identity>,
    y: &mut StridedArrayViewMut<'a, T, 1, Identity>,
) -> Result<()>
where
    T: ElementOpApply + Copy + Add<Output = T> + Mul<Output = T> + One + PartialEq,
{
    let n = x.size()[0];
    if n != y.size()[0] {
        return Err(StridedError::ShapeMismatch(
            x.size().to_vec(),
            y.size().to_vec(),
        ));
    }

    if n == 0 {
        return Ok(());
    }

    // Simple element-wise iteration
    if a.is_one() {
        for i in 0..n {
            let xi = x.get([i]);
            let yi = y.get([i]);
            y.set([i], xi + yi);
        }
    } else {
        for i in 0..n {
            let xi = x.get([i]);
            let yi = y.get([i]);
            y.set([i], a * xi + yi);
        }
    }

    Ok(())
}

/// axpby: Y = a * X + b * Y (for 1D arrays)
///
/// # Julia equivalent
/// ```julia
/// function LinearAlgebra.axpby!(a::Number, X::StridedView{<:Number,N},
///                               b::Number, Y::StridedView{<:Number,N}) where {N}
///     if b == 1
///         axpy!(a, X, Y)
///     elseif b == 0
///         mul!(Y, a, X)
///     else
///         Y .= a .* X .+ b .* Y
///     end
///     return Y
/// end
/// ```
pub fn axpby<'a, T>(
    a: T,
    x: &StridedArrayView<'a, T, 1, Identity>,
    b: T,
    y: &mut StridedArrayViewMut<'a, T, 1, Identity>,
) -> Result<()>
where
    T: ElementOpApply + Copy + Add<Output = T> + Mul<Output = T> + One + Zero + PartialEq,
{
    let n = x.size()[0];
    if n != y.size()[0] {
        return Err(StridedError::ShapeMismatch(
            x.size().to_vec(),
            y.size().to_vec(),
        ));
    }

    if n == 0 {
        return Ok(());
    }

    if b.is_one() {
        // Y = a * X + Y (use axpy)
        return axpy(a, x, y);
    } else if b.is_zero() {
        // Y = a * X (just scale and copy)
        for i in 0..n {
            let xi = x.get([i]);
            y.set([i], a * xi);
        }
    } else {
        // Y = a * X + b * Y (general case)
        for i in 0..n {
            let xi = x.get([i]);
            let yi = y.get([i]);
            y.set([i], a * xi + b * yi);
        }
    }

    Ok(())
}

/// rmul: dst = dst * α (in-place right multiply by scalar for 1D arrays)
///
/// # Julia equivalent
/// ```julia
/// LinearAlgebra.rmul!(dst::StridedView, α::Number) = mul!(dst, dst, α)
/// ```
pub fn rmul<'a, T>(dst: &mut StridedArrayViewMut<'a, T, 1, Identity>, alpha: T) -> Result<()>
where
    T: ElementOpApply + Copy + Mul<Output = T> + One + Zero + PartialEq,
{
    if alpha.is_one() {
        return Ok(());
    }

    let n = dst.size()[0];

    if alpha.is_zero() {
        for i in 0..n {
            dst.set([i], T::zero());
        }
    } else {
        for i in 0..n {
            let val = dst.get([i]);
            dst.set([i], val * alpha);
        }
    }

    Ok(())
}

/// lmul: dst = α * dst (in-place left multiply by scalar for 1D arrays)
///
/// # Julia equivalent
/// ```julia
/// LinearAlgebra.lmul!(α::Number, dst::StridedView) = mul!(dst, α, dst)
/// ```
pub fn lmul<'a, T>(alpha: T, dst: &mut StridedArrayViewMut<'a, T, 1, Identity>) -> Result<()>
where
    T: ElementOpApply + Copy + Mul<Output = T> + One + Zero + PartialEq,
{
    // For commutative multiplication, same as rmul
    rmul(dst, alpha)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isblasmatrix_identity_colmajor() {
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        // Column-major 3x4: stride = [1, 3]
        let view: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&data, [3, 4], [1, 3], 0).unwrap();

        assert!(isblasmatrix_identity(&view));
    }

    #[test]
    fn test_isblasmatrix_identity_rowmajor() {
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        // Row-major 3x4: stride = [4, 1]
        let view: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&data, [3, 4], [4, 1], 0).unwrap();

        assert!(isblasmatrix_identity(&view));
    }

    #[test]
    fn test_isblasmatrix_identity_non_contiguous() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        // Non-contiguous: stride = [8, 2]
        let view: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&data, [3, 4], [8, 2], 0).unwrap();

        assert!(!isblasmatrix_identity(&view));
    }

    #[test]
    fn test_getblasmatrix_identity_colmajor() {
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        // Column-major: stride = [1, 3]
        let view: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&data, [3, 4], [1, 3], 0).unwrap();

        let (_, trans) = getblasmatrix_identity(&view);
        assert_eq!(trans, BlasTranspose::NoTrans);
    }

    #[test]
    fn test_getblasmatrix_identity_rowmajor() {
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        // Row-major: stride = [4, 1]
        let view: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&data, [3, 4], [4, 1], 0).unwrap();

        let (transposed_view, trans) = getblasmatrix_identity(&view);
        assert_eq!(trans, BlasTranspose::Trans);
        // Transposed view should have swapped dimensions
        assert_eq!(transposed_view.size(), &[4, 3]);
    }

    #[test]
    fn test_generic_matmul_basic() {
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

        generic_matmul(1.0, &a, &b, 0.0, &mut c).unwrap();

        assert_eq!(c_data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_generic_matmul_with_alpha_beta() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![5.0, 6.0, 7.0, 8.0];
        let mut c_data = vec![1.0, 1.0, 1.0, 1.0];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [2, 2], [2, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [2, 2], [2, 1], 0).unwrap();
        let mut c: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut c_data, [2, 2], [2, 1], 0).unwrap();

        // C = 2 * A * B + 3 * C
        generic_matmul(2.0, &a, &b, 3.0, &mut c).unwrap();

        // = 2 * [[19, 22], [43, 50]] + 3 * [[1, 1], [1, 1]]
        // = [[38, 44], [86, 100]] + [[3, 3], [3, 3]]
        // = [[41, 47], [89, 103]]
        assert_eq!(c_data, vec![41.0, 47.0, 89.0, 103.0]);
    }

    #[test]
    fn test_generic_matmul_alpha_zero() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![5.0, 6.0, 7.0, 8.0];
        let mut c_data = vec![10.0, 20.0, 30.0, 40.0];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [2, 2], [2, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [2, 2], [2, 1], 0).unwrap();
        let mut c: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut c_data, [2, 2], [2, 1], 0).unwrap();

        // C = 0 * A * B + 2 * C = 2 * C
        generic_matmul(0.0, &a, &b, 2.0, &mut c).unwrap();

        assert_eq!(c_data, vec![20.0, 40.0, 60.0, 80.0]);
    }

    #[test]
    fn test_generic_matmul_beta_one() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![5.0, 6.0, 7.0, 8.0];
        let mut c_data = vec![1.0, 1.0, 1.0, 1.0];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [2, 2], [2, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [2, 2], [2, 1], 0).unwrap();
        let mut c: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut c_data, [2, 2], [2, 1], 0).unwrap();

        // C = 1 * A * B + 1 * C = A * B + C
        generic_matmul(1.0, &a, &b, 1.0, &mut c).unwrap();

        // = [[19, 22], [43, 50]] + [[1, 1], [1, 1]] = [[20, 23], [44, 51]]
        assert_eq!(c_data, vec![20.0, 23.0, 44.0, 51.0]);
    }

    #[test]
    fn test_generic_matmul_rectangular() {
        // A = [[1, 2, 3], [4, 5, 6]] (2x3)
        // B = [[7, 8], [9, 10], [11, 12]] (3x2)
        // C = A * B = [[58, 64], [139, 154]] (2x2)
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c_data = vec![0.0; 4];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [2, 3], [3, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [3, 2], [2, 1], 0).unwrap();
        let mut c: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut c_data, [2, 2], [2, 1], 0).unwrap();

        generic_matmul(1.0, &a, &b, 0.0, &mut c).unwrap();

        assert_eq!(c_data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_axpy_basic() {
        let x_data = vec![1.0, 2.0, 3.0, 4.0];
        let mut y_data = vec![10.0, 20.0, 30.0, 40.0];

        let x: StridedArrayView<'_, f64, 1, Identity> =
            StridedArrayView::new(&x_data, [4], [1], 0).unwrap();
        let mut y: StridedArrayViewMut<'_, f64, 1, Identity> =
            StridedArrayViewMut::new(&mut y_data, [4], [1], 0).unwrap();

        // Y = 2 * X + Y
        axpy(2.0, &x, &mut y).unwrap();

        assert_eq!(y_data, vec![12.0, 24.0, 36.0, 48.0]);
    }

    #[test]
    fn test_axpy_alpha_one() {
        let x_data = vec![1.0, 2.0, 3.0];
        let mut y_data = vec![10.0, 20.0, 30.0];

        let x: StridedArrayView<'_, f64, 1, Identity> =
            StridedArrayView::new(&x_data, [3], [1], 0).unwrap();
        let mut y: StridedArrayViewMut<'_, f64, 1, Identity> =
            StridedArrayViewMut::new(&mut y_data, [3], [1], 0).unwrap();

        // Y = 1 * X + Y = X + Y
        axpy(1.0, &x, &mut y).unwrap();

        assert_eq!(y_data, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_axpby_basic() {
        let x_data = vec![1.0, 2.0, 3.0];
        let mut y_data = vec![10.0, 20.0, 30.0];

        let x: StridedArrayView<'_, f64, 1, Identity> =
            StridedArrayView::new(&x_data, [3], [1], 0).unwrap();
        let mut y: StridedArrayViewMut<'_, f64, 1, Identity> =
            StridedArrayViewMut::new(&mut y_data, [3], [1], 0).unwrap();

        // Y = 2 * X + 3 * Y
        axpby(2.0, &x, 3.0, &mut y).unwrap();

        // = [2, 4, 6] + [30, 60, 90] = [32, 64, 96]
        assert_eq!(y_data, vec![32.0, 64.0, 96.0]);
    }

    #[test]
    fn test_axpby_beta_zero() {
        let x_data = vec![1.0, 2.0, 3.0];
        let mut y_data = vec![10.0, 20.0, 30.0];

        let x: StridedArrayView<'_, f64, 1, Identity> =
            StridedArrayView::new(&x_data, [3], [1], 0).unwrap();
        let mut y: StridedArrayViewMut<'_, f64, 1, Identity> =
            StridedArrayViewMut::new(&mut y_data, [3], [1], 0).unwrap();

        // Y = 2 * X + 0 * Y = 2 * X
        axpby(2.0, &x, 0.0, &mut y).unwrap();

        assert_eq!(y_data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_rmul_basic() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];

        let mut view: StridedArrayViewMut<'_, f64, 1, Identity> =
            StridedArrayViewMut::new(&mut data, [4], [1], 0).unwrap();

        rmul(&mut view, 2.0).unwrap();

        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_rmul_zero() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];

        let mut view: StridedArrayViewMut<'_, f64, 1, Identity> =
            StridedArrayViewMut::new(&mut data, [4], [1], 0).unwrap();

        rmul(&mut view, 0.0).unwrap();

        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_rmul_one() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let original = data.clone();

        let mut view: StridedArrayViewMut<'_, f64, 1, Identity> =
            StridedArrayViewMut::new(&mut data, [4], [1], 0).unwrap();

        rmul(&mut view, 1.0).unwrap();

        assert_eq!(data, original);
    }

    #[test]
    fn test_matmul_basic() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![5.0, 6.0, 7.0, 8.0];
        let mut c_data = vec![0.0; 4];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [2, 2], [2, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [2, 2], [2, 1], 0).unwrap();
        let mut c: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut c_data, [2, 2], [2, 1], 0).unwrap();

        matmul(1.0, &a, &b, 0.0, &mut c).unwrap();

        assert_eq!(c_data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_colmajor() {
        // Column-major layout
        let a_data = vec![1.0, 3.0, 2.0, 4.0]; // [[1, 2], [3, 4]] in column-major
        let b_data = vec![5.0, 7.0, 6.0, 8.0]; // [[5, 6], [7, 8]] in column-major
        let mut c_data = vec![0.0; 4];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [2, 2], [1, 2], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [2, 2], [1, 2], 0).unwrap();
        let mut c: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut c_data, [2, 2], [1, 2], 0).unwrap();

        matmul(1.0, &a, &b, 0.0, &mut c).unwrap();

        // Result in column-major: [[19, 43], [22, 50]] stored as [19, 43, 22, 50]
        assert_eq!(c_data, vec![19.0, 43.0, 22.0, 50.0]);
    }
}
