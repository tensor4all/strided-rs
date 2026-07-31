//! CBLAS-backed batched GEMM kernel on contiguous operands.
//!
//! Uses `cblas_dgemm` / `cblas_zgemm` for hardware-optimized matrix multiplication.
//! Operands must already have contiguous inner dimensions (prepared via
//! `prepare_input_*` and `prepare_output_*` in the `contiguous` module).

use crate::backend::{Backend, BlasBackend, OverwriteBackend};
use crate::contiguous::{ContiguousOperand, ContiguousOperandMut};
use crate::util::{try_fuse_group, MultiIndex};
use crate::ScalarBase;
use strided_kernel::ExecContext;

#[cfg(all(feature = "blas-inject", not(feature = "blas")))]
mod inject_fallback {
    use std::ffi::c_char;
    use std::sync::Once;

    use num_complex::{Complex32, Complex64};
    use num_traits::Zero;

    static REGISTER_ONCE: Once = Once::new();

    #[inline]
    fn trans_flag(t: c_char) -> u8 {
        (t as u8).to_ascii_uppercase()
    }

    #[inline]
    unsafe fn gemm_real<T>(
        transa: u8,
        transb: u8,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        a: *const T,
        lda: usize,
        b: *const T,
        ldb: usize,
        beta: T,
        c: *mut T,
        ldc: usize,
    ) where
        T: Copy + Zero + PartialEq + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
    {
        for j in 0..n {
            for i in 0..m {
                let mut sum = T::zero();
                for p in 0..k {
                    let a_val = if transa == b'N' {
                        *a.add(i + p * lda)
                    } else {
                        *a.add(p + i * lda)
                    };
                    let b_val = if transb == b'N' {
                        *b.add(p + j * ldb)
                    } else {
                        *b.add(j + p * ldb)
                    };
                    sum = sum + a_val * b_val;
                }
                let c_ptr = c.add(i + j * ldc);
                *c_ptr = if beta == T::zero() {
                    alpha * sum
                } else {
                    alpha * sum + beta * *c_ptr
                };
            }
        }
    }

    #[inline]
    unsafe fn gemm_complex(
        transa: u8,
        transb: u8,
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex64,
        a: *const Complex64,
        lda: usize,
        b: *const Complex64,
        ldb: usize,
        beta: Complex64,
        c: *mut Complex64,
        ldc: usize,
    ) {
        for j in 0..n {
            for i in 0..m {
                let mut sum = Complex64::new(0.0, 0.0);
                for p in 0..k {
                    let mut a_val = if transa == b'N' {
                        *a.add(i + p * lda)
                    } else {
                        *a.add(p + i * lda)
                    };
                    let mut b_val = if transb == b'N' {
                        *b.add(p + j * ldb)
                    } else {
                        *b.add(j + p * ldb)
                    };
                    if transa == b'C' {
                        a_val = a_val.conj();
                    }
                    if transb == b'C' {
                        b_val = b_val.conj();
                    }
                    sum += a_val * b_val;
                }
                let c_ptr = c.add(i + j * ldc);
                *c_ptr = if beta == Complex64::new(0.0, 0.0) {
                    alpha * sum
                } else {
                    alpha * sum + beta * *c_ptr
                };
            }
        }
    }

    unsafe extern "C" fn dgemm_fallback(
        transa: *const c_char,
        transb: *const c_char,
        m: *const cblas_sys::blasint,
        n: *const cblas_sys::blasint,
        k: *const cblas_sys::blasint,
        alpha: *const f64,
        a: *const f64,
        lda: *const cblas_sys::blasint,
        b: *const f64,
        ldb: *const cblas_sys::blasint,
        beta: *const f64,
        c: *mut f64,
        ldc: *const cblas_sys::blasint,
    ) {
        let transa = trans_flag(*transa);
        let transb = trans_flag(*transb);
        unsafe {
            gemm_real(
                transa,
                transb,
                *m as usize,
                *n as usize,
                *k as usize,
                *alpha,
                a,
                *lda as usize,
                b,
                *ldb as usize,
                *beta,
                c,
                *ldc as usize,
            );
        }
    }

    unsafe extern "C" fn sgemm_fallback(
        transa: *const c_char,
        transb: *const c_char,
        m: *const cblas_sys::blasint,
        n: *const cblas_sys::blasint,
        k: *const cblas_sys::blasint,
        alpha: *const f32,
        a: *const f32,
        lda: *const cblas_sys::blasint,
        b: *const f32,
        ldb: *const cblas_sys::blasint,
        beta: *const f32,
        c: *mut f32,
        ldc: *const cblas_sys::blasint,
    ) {
        unsafe {
            gemm_real(
                trans_flag(*transa),
                trans_flag(*transb),
                *m as usize,
                *n as usize,
                *k as usize,
                *alpha,
                a,
                *lda as usize,
                b,
                *ldb as usize,
                *beta,
                c,
                *ldc as usize,
            );
        }
    }

    unsafe extern "C" fn zgemm_fallback(
        transa: *const c_char,
        transb: *const c_char,
        m: *const cblas_sys::blasint,
        n: *const cblas_sys::blasint,
        k: *const cblas_sys::blasint,
        alpha: *const Complex64,
        a: *const Complex64,
        lda: *const cblas_sys::blasint,
        b: *const Complex64,
        ldb: *const cblas_sys::blasint,
        beta: *const Complex64,
        c: *mut Complex64,
        ldc: *const cblas_sys::blasint,
    ) {
        let transa = trans_flag(*transa);
        let transb = trans_flag(*transb);
        unsafe {
            gemm_complex(
                transa,
                transb,
                *m as usize,
                *n as usize,
                *k as usize,
                *alpha,
                a,
                *lda as usize,
                b,
                *ldb as usize,
                *beta,
                c,
                *ldc as usize,
            );
        }
    }

    unsafe extern "C" fn cgemm_fallback(
        transa: *const c_char,
        transb: *const c_char,
        m: *const cblas_sys::blasint,
        n: *const cblas_sys::blasint,
        k: *const cblas_sys::blasint,
        alpha: *const Complex32,
        a: *const Complex32,
        lda: *const cblas_sys::blasint,
        b: *const Complex32,
        ldb: *const cblas_sys::blasint,
        beta: *const Complex32,
        c: *mut Complex32,
        ldc: *const cblas_sys::blasint,
    ) {
        let transa = trans_flag(*transa);
        let transb = trans_flag(*transb);
        unsafe {
            for j in 0..*n as usize {
                for i in 0..*m as usize {
                    let mut sum = Complex32::new(0.0, 0.0);
                    for p in 0..*k as usize {
                        let mut av = if transa == b'N' {
                            *a.add(i + p * *lda as usize)
                        } else {
                            *a.add(p + i * *lda as usize)
                        };
                        let mut bv = if transb == b'N' {
                            *b.add(p + j * *ldb as usize)
                        } else {
                            *b.add(j + p * *ldb as usize)
                        };
                        if transa == b'C' {
                            av = av.conj();
                        }
                        if transb == b'C' {
                            bv = bv.conj();
                        }
                        sum = sum + av * bv;
                    }
                    let out = c.add(i + j * *ldc as usize);
                    *out = if *beta == Complex32::new(0.0, 0.0) {
                        *alpha * sum
                    } else {
                        *alpha * sum + *beta * *out
                    };
                }
            }
        }
    }

    pub(super) fn ensure_registered() {
        REGISTER_ONCE.call_once(|| unsafe {
            if !cblas_sys::is_dgemm_registered() {
                cblas_sys::register_dgemm(dgemm_fallback);
            }
            if !cblas_sys::is_sgemm_registered() {
                cblas_sys::register_sgemm(sgemm_fallback);
            }
            if !cblas_sys::is_cgemm_registered() {
                cblas_sys::register_cgemm(cgemm_fallback);
            }
            if !cblas_sys::is_zgemm_registered() {
                cblas_sys::register_zgemm(zgemm_fallback);
            }
        });
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::mem::MaybeUninit;

        fn args<T>(
            alpha: &T,
            a: *const T,
            beta: &T,
            c: *mut T,
        ) -> (
            *const c_char,
            *const c_char,
            *const cblas_sys::blasint,
            *const cblas_sys::blasint,
            *const cblas_sys::blasint,
            *const T,
            *const T,
            *const cblas_sys::blasint,
            *const T,
            *mut T,
            *const cblas_sys::blasint,
        ) {
            static N: c_char = b'N' as c_char;
            static ONE: cblas_sys::blasint = 1;
            (&N, &N, &ONE, &ONE, &ONE, alpha, a, &ONE, beta, c, &ONE)
        }

        #[test]
        fn all_registered_fallbacks_skip_poisoned_c_for_zero_beta() {
            let alpha32 = 2.0f32;
            let beta32 = 0.0f32;
            let a32 = 3.0f32;
            let b32 = 4.0f32;
            let poison32 = f32::NAN;
            let mut c32 = MaybeUninit::new(poison32);
            let (ta, tb, m, n, k, alpha, a, lda, beta, c, ldc) =
                args(&alpha32, &a32, &beta32, c32.as_mut_ptr().cast());
            unsafe {
                sgemm_fallback(ta, tb, m, n, k, alpha, a, lda, &b32, lda, beta, c, ldc);
            }
            assert_eq!(unsafe { c32.assume_init() }, 24.0);

            let alpha64 = 2.0f64;
            let beta64 = 0.0f64;
            let a64 = 3.0f64;
            let b64 = 4.0f64;
            let mut c64 = MaybeUninit::new(f64::NAN);
            let (ta, tb, m, n, k, alpha, a, lda, beta, c, ldc) =
                args(&alpha64, &a64, &beta64, c64.as_mut_ptr().cast());
            unsafe {
                dgemm_fallback(ta, tb, m, n, k, alpha, a, lda, &b64, lda, beta, c, ldc);
            }
            assert_eq!(unsafe { c64.assume_init() }, 24.0);

            let alpha_c = Complex32::new(2.0, 0.0);
            let beta_c = Complex32::new(0.0, 0.0);
            let a_c = Complex32::new(3.0, 0.0);
            let b_c = Complex32::new(4.0, 0.0);
            let mut c_c = MaybeUninit::new(Complex32::new(f32::NAN, f32::NAN));
            let (ta, tb, m, n, k, alpha, a, lda, beta, c, ldc) =
                args(&alpha_c, &a_c, &beta_c, c_c.as_mut_ptr().cast());
            unsafe {
                cgemm_fallback(ta, tb, m, n, k, alpha, a, lda, &b_c, lda, beta, c, ldc);
            }
            assert_eq!(unsafe { c_c.assume_init() }, Complex32::new(24.0, 0.0));

            let alpha_z = Complex64::new(2.0, 0.0);
            let beta_z = Complex64::new(0.0, 0.0);
            let a_z = Complex64::new(3.0, 0.0);
            let b_z = Complex64::new(4.0, 0.0);
            let mut c_z = MaybeUninit::new(Complex64::new(f64::NAN, f64::NAN));
            let (ta, tb, m, n, k, alpha, a, lda, beta, c, ldc) =
                args(&alpha_z, &a_z, &beta_z, c_z.as_mut_ptr().cast());
            unsafe {
                zgemm_fallback(ta, tb, m, n, k, alpha, a, lda, &b_z, lda, beta, c, ldc);
            }
            assert_eq!(unsafe { c_z.assume_init() }, Complex64::new(24.0, 0.0));
        }
    }
}

/// Type-level dispatch trait for CBLAS GEMM.
///
/// Implemented for `f32`/`f64` and `Complex32`/`Complex64`.
/// The `trans_a` and `trans_b` parameters accept `cblas_sys::CBLAS_TRANSPOSE` values.
pub trait BlasGemm: Sized {
    /// Call the appropriate CBLAS GEMM routine.
    ///
    /// Computes `C = alpha * op(A) * op(B) + beta * C` where:
    /// - A is stored as an lda-by-? matrix in col-major layout
    /// - op(A) is m-by-k, op(B) is k-by-n, C is m-by-n
    ///
    /// # Safety
    ///
    /// Pointers `a`, `b`, `c` must point to valid memory of sufficient size
    /// for the given dimensions and leading dimensions.
    unsafe fn gemm(
        trans_a: cblas_sys::CBLAS_TRANSPOSE,
        trans_b: cblas_sys::CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        k: i32,
        alpha: Self,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        beta: Self,
        c: *mut Self,
        ldc: i32,
    );
}

impl BlasGemm for f32 {
    unsafe fn gemm(
        trans_a: cblas_sys::CBLAS_TRANSPOSE,
        trans_b: cblas_sys::CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    ) {
        unsafe {
            cblas_sys::cblas_sgemm(
                cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                trans_a,
                trans_b,
                m,
                n,
                k,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c,
                ldc,
            );
        }
    }
}

impl BlasGemm for f64 {
    unsafe fn gemm(
        trans_a: cblas_sys::CBLAS_TRANSPOSE,
        trans_b: cblas_sys::CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    ) {
        unsafe {
            cblas_sys::cblas_dgemm(
                cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                trans_a,
                trans_b,
                m,
                n,
                k,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c,
                ldc,
            );
        }
    }
}

impl BlasGemm for num_complex::Complex32 {
    unsafe fn gemm(
        trans_a: cblas_sys::CBLAS_TRANSPOSE,
        trans_b: cblas_sys::CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        k: i32,
        alpha: num_complex::Complex32,
        a: *const num_complex::Complex32,
        lda: i32,
        b: *const num_complex::Complex32,
        ldb: i32,
        beta: num_complex::Complex32,
        c: *mut num_complex::Complex32,
        ldc: i32,
    ) {
        unsafe {
            cblas_sys::cblas_cgemm(
                cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                trans_a,
                trans_b,
                m,
                n,
                k,
                (&alpha) as *const _ as *const _,
                a as *const _ as *const _,
                lda,
                b as *const _ as *const _,
                ldb,
                (&beta) as *const _ as *const _,
                c as *mut _ as *mut _,
                ldc,
            );
        }
    }
}

impl BlasGemm for num_complex::Complex64 {
    unsafe fn gemm(
        trans_a: cblas_sys::CBLAS_TRANSPOSE,
        trans_b: cblas_sys::CBLAS_TRANSPOSE,
        m: i32,
        n: i32,
        k: i32,
        alpha: num_complex::Complex64,
        a: *const num_complex::Complex64,
        lda: i32,
        b: *const num_complex::Complex64,
        ldb: i32,
        beta: num_complex::Complex64,
        c: *mut num_complex::Complex64,
        ldc: i32,
    ) {
        unsafe {
            cblas_sys::cblas_zgemm(
                cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                trans_a,
                trans_b,
                m,
                n,
                k,
                (&alpha) as *const _ as *const _,
                a as *const _ as *const _,
                lda,
                b as *const _ as *const _,
                ldb,
                (&beta) as *const _ as *const _,
                c as *mut _ as *mut _,
                ldc,
            );
        }
    }
}

/// Flip a CBLAS transpose flag: NoTrans ↔ Trans.
///
/// Used when C is row-major and we rewrite C = A·B as C^T = B^T · A^T.
fn flip_transpose(t: cblas_sys::CBLAS_TRANSPOSE) -> cblas_sys::CBLAS_TRANSPOSE {
    match t {
        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans => cblas_sys::CBLAS_TRANSPOSE::CblasTrans,
        cblas_sys::CBLAS_TRANSPOSE::CblasTrans => cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
        other => other,
    }
}

/// Determine transpose flag and leading dimension for a contiguous operand.
///
/// CBLAS CblasColMajor expects:
/// - NoTrans: matrix stored col-major, lda >= nrows (= m or k depending on operand)
/// - Trans: matrix stored row-major, lda >= ncols (= k or n depending on operand)
///
/// `nrows` and `ncols` are the logical matrix dimensions (before any transpose).
/// They are needed because when one dimension is 1, the corresponding stride may
/// be 0 (since it's never used for address computation), but CBLAS still requires
/// the leading dimension to be >= the relevant matrix dimension.
///
/// Returns `(transpose_flag, leading_dimension)`.
fn operand_layout(
    row_stride: isize,
    col_stride: isize,
    nrows: usize,
    ncols: usize,
) -> (cblas_sys::CBLAS_TRANSPOSE, i32) {
    if row_stride == 1 || row_stride == 0 {
        // Col-major: lda = col_stride, but must be >= nrows
        let lda = col_stride.max(nrows as isize).max(1) as i32;
        (cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans, lda)
    } else if col_stride == 1 || col_stride == 0 {
        // Row-major = transposed col-major: lda = row_stride, but must be >= ncols
        let lda = row_stride.max(ncols as isize).max(1) as i32;
        (cblas_sys::CBLAS_TRANSPOSE::CblasTrans, lda)
    } else {
        // Neither row- nor col-major. This shouldn't happen after contiguous preparation.
        panic!(
            "bgemm_blas: operand has non-unit strides (row={}, col={}). \
             This indicates a bug in contiguous preparation.",
            row_stride, col_stride
        );
    }
}

/// Batched GEMM on pre-contiguous operands using CBLAS.
///
/// Operands must already have contiguous inner dimensions (prepared via
/// `prepare_input_*` and `prepare_output_*` in the `contiguous` module).
///
/// - `batch_dims`: sizes of the batch dimensions
/// - `m`: fused lo dimension size (number of rows of A/C)
/// - `n`: fused ro dimension size (number of cols of B/C)
/// - `k`: fused sum dimension size (inner dimension)
///
/// Handles both col-major (row_stride=1) and row-major (col_stride=1) operands
/// by mapping them to CblasNoTrans / CblasTrans respectively. When C is row-major,
/// the computation is rewritten as C^T = B^T * A^T via dimension/pointer swapping.
///
/// CBLAS handles `beta` internally, so no pre-scaling loop is needed
/// (unlike the faer backend which requires explicit pre-scaling for beta not in {0, 1}).
pub(crate) fn bgemm_contiguous_into<T: ScalarBase + strided_view::ElementOpApply + BlasGemm>(
    c: &mut ContiguousOperandMut<T>,
    a: &ContiguousOperand<T>,
    b: &ContiguousOperand<T>,
    batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    beta: T,
) -> strided_view::Result<()> {
    #[cfg(all(feature = "blas-inject", not(feature = "blas")))]
    inject_fallback::ensure_registered();

    // Conjugation must be resolved before reaching this function
    // (handled during contiguous preparation).
    debug_assert!(!a.conj());
    debug_assert!(!b.conj());

    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let c_ptr = c.ptr();

    // A is m×k, B is k×n, C is m×n
    let (trans_a, lda) = operand_layout(a.row_stride(), a.col_stride(), m, k);
    let (trans_b, ldb) = operand_layout(b.row_stride(), b.col_stride(), k, n);
    let c_is_col_major = c.row_stride() == 1 || c.row_stride() == 0;

    let m_i32 = m as i32;
    let n_i32 = n as i32;
    let k_i32 = k as i32;

    // Per-batch GEMM dispatch closure (individual cblas_dgemm calls).
    // Using individual calls instead of cblas_dgemm_batch avoids Vec allocation
    // overhead and is faster for many small GEMMs (e.g. str_mps, matrix_chain).
    let do_batch = |a_off: isize, b_off: isize, c_off: isize| unsafe {
        if c_is_col_major {
            let ldc = c.col_stride().max(m as isize).max(1) as i32;
            T::gemm(
                trans_a,
                trans_b,
                m_i32,
                n_i32,
                k_i32,
                alpha,
                a_ptr.offset(a_off),
                lda,
                b_ptr.offset(b_off),
                ldb,
                beta,
                c_ptr.offset(c_off),
                ldc,
            );
        } else {
            // C is row-major: rewrite as C^T = alpha * B^T * A^T + beta * C^T
            let ldc = c.row_stride().max(n as isize).max(1) as i32;
            T::gemm(
                flip_transpose(trans_b),
                flip_transpose(trans_a),
                n_i32,
                m_i32,
                k_i32,
                alpha,
                b_ptr.offset(b_off),
                ldb,
                a_ptr.offset(a_off),
                lda,
                beta,
                c_ptr.offset(c_off),
                ldc,
            );
        }
    };

    // Fast path: when batch dims are contiguous for all operands, use pointer
    // increments instead of MultiIndex carry-based iteration.
    let fused_a = try_fuse_group(batch_dims, a.batch_strides());
    let fused_b = try_fuse_group(batch_dims, b.batch_strides());
    let fused_c = try_fuse_group(batch_dims, c.batch_strides());

    if let (Some((total, a_step)), Some((_, b_step)), Some((_, c_step))) =
        (fused_a, fused_b, fused_c)
    {
        let mut a_off = 0isize;
        let mut b_off = 0isize;
        let mut c_off = 0isize;
        for _ in 0..total {
            do_batch(a_off, b_off, c_off);
            a_off += a_step;
            b_off += b_step;
            c_off += c_step;
        }
    } else {
        let mut batch_iter = MultiIndex::new(batch_dims);
        while batch_iter.next().is_some() {
            let a_off = batch_iter.offset(a.batch_strides());
            let b_off = batch_iter.offset(b.batch_strides());
            let c_off = batch_iter.offset(c.batch_strides());
            do_batch(a_off, b_off, c_off);
        }
    }

    Ok(())
}

fn checked_operand_layout(
    row_stride: isize,
    col_stride: isize,
    nrows: usize,
    ncols: usize,
) -> strided_view::Result<(cblas_sys::CBLAS_TRANSPOSE, i32)> {
    let (trans, lda) = if row_stride == 1 || row_stride == 0 {
        (
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            col_stride.max(nrows as isize).max(1),
        )
    } else if col_stride == 1 || col_stride == 0 {
        (
            cblas_sys::CBLAS_TRANSPOSE::CblasTrans,
            row_stride.max(ncols as isize).max(1),
        )
    } else {
        return Err(strided_view::StridedError::PlanLayoutMismatch);
    };
    Ok((
        trans,
        i32::try_from(lda).map_err(|_| strided_view::StridedError::OffsetOverflow)?,
    ))
}

/// CBLAS overwrite path. The literal zero beta is part of the private
/// contract; cblas-inject 0.1.2 guarantees that exact zero never reads C.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bgemm_contiguous_overwrite<T>(
    c: &mut crate::contiguous::UninitContiguousOperand<'_, '_, T>,
    a: &ContiguousOperand<T>,
    b: &ContiguousOperand<T>,
    batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    _ctx: &ExecContext,
) -> strided_view::Result<()>
where
    T: ScalarBase + strided_view::ElementOpApply + BlasGemm,
{
    #[cfg(all(feature = "blas-inject", not(feature = "blas")))]
    inject_fallback::ensure_registered();
    debug_assert!(!a.conj() && !b.conj());
    let (trans_a, lda) = checked_operand_layout(a.row_stride(), a.col_stride(), m, k)?;
    let (trans_b, ldb) = checked_operand_layout(b.row_stride(), b.col_stride(), k, n)?;
    let m_i32 = i32::try_from(m).map_err(|_| strided_view::StridedError::OffsetOverflow)?;
    let n_i32 = i32::try_from(n).map_err(|_| strided_view::StridedError::OffsetOverflow)?;
    let k_i32 = i32::try_from(k).map_err(|_| strided_view::StridedError::OffsetOverflow)?;
    let c_is_col_major = c.row_stride() == 1 || c.row_stride() == 0;
    let ldc_value = if c_is_col_major {
        c.col_stride().max(m as isize).max(1)
    } else {
        c.row_stride().max(n as isize).max(1)
    };
    let ldc = i32::try_from(ldc_value).map_err(|_| strided_view::StridedError::OffsetOverflow)?;
    let zero = T::zero();
    let mut batch = MultiIndex::new(batch_dims);
    while batch.next().is_some() {
        let a_off = batch.offset(a.batch_strides());
        let b_off = batch.offset(b.batch_strides());
        let c_off = batch.offset(c.batch_strides());
        unsafe {
            if c_is_col_major {
                T::gemm(
                    trans_a,
                    trans_b,
                    m_i32,
                    n_i32,
                    k_i32,
                    alpha,
                    a.ptr().offset(a_off),
                    lda,
                    b.ptr().offset(b_off),
                    ldb,
                    zero,
                    c.ptr().offset(c_off).cast(),
                    ldc,
                );
            } else {
                T::gemm(
                    flip_transpose(trans_b),
                    flip_transpose(trans_a),
                    n_i32,
                    m_i32,
                    k_i32,
                    alpha,
                    b.ptr().offset(b_off),
                    ldb,
                    a.ptr().offset(a_off),
                    lda,
                    zero,
                    c.ptr().offset(c_off).cast(),
                    ldc,
                );
            }
        }
    }
    Ok(())
}

impl<T> Backend<T> for BlasBackend
where
    T: ScalarBase + strided_view::ElementOpApply + BlasGemm,
{
    const MATERIALIZES_CONJ: bool = true;
    const REQUIRES_UNIT_STRIDE: bool = true;

    fn bgemm_contiguous_into(
        c: &mut ContiguousOperandMut<T>,
        a: &ContiguousOperand<T>,
        b: &ContiguousOperand<T>,
        batch_dims: &[usize],
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        beta: T,
    ) -> strided_view::Result<()> {
        // Delegate to the existing free function in this module.
        // Use explicit module path to disambiguate from the trait method.
        self::bgemm_contiguous_into(c, a, b, batch_dims, m, n, k, alpha, beta)
    }
}

impl<T> OverwriteBackend<T> for BlasBackend
where
    T: ScalarBase + strided_view::ElementOpApply + BlasGemm,
{
    fn bgemm_contiguous_overwrite(
        c: &mut crate::contiguous::UninitContiguousOperand<'_, '_, T>,
        a: &ContiguousOperand<T>,
        b: &ContiguousOperand<T>,
        batch_dims: &[usize],
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        ctx: &ExecContext,
    ) -> strided_view::Result<()> {
        bgemm_contiguous_overwrite(c, a, b, batch_dims, m, n, k, alpha, ctx)
    }
}
