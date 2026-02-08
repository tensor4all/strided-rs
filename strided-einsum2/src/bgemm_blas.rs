//! CBLAS-backed batched GEMM kernel on contiguous operands.
//!
//! Uses `cblas_dgemm` / `cblas_zgemm` for hardware-optimized matrix multiplication.
//! Operands must already have contiguous inner dimensions (prepared via
//! `prepare_input_*` and `prepare_output_*` in the `contiguous` module).

use crate::backend::{BgemmBackend, BlasBackend};
use crate::contiguous::{ContiguousOperand, ContiguousOperandMut};
use crate::util::{try_fuse_group, MultiIndex};
use crate::Scalar;

#[cfg(all(feature = "blas-inject", not(feature = "blas")))]
mod inject_fallback {
    use std::ffi::c_char;
    use std::sync::Once;

    use num_complex::Complex64;

    static REGISTER_ONCE: Once = Once::new();

    #[inline]
    fn trans_flag(t: c_char) -> u8 {
        (t as u8).to_ascii_uppercase()
    }

    #[inline]
    unsafe fn gemm_real(
        transa: u8,
        transb: u8,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a: *const f64,
        lda: usize,
        b: *const f64,
        ldb: usize,
        beta: f64,
        c: *mut f64,
        ldc: usize,
    ) {
        for j in 0..n {
            for i in 0..m {
                let mut sum = 0.0f64;
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
                    sum += a_val * b_val;
                }
                let c_ptr = c.add(i + j * ldc);
                *c_ptr = alpha * sum + beta * *c_ptr;
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
                *c_ptr = alpha * sum + beta * *c_ptr;
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

    pub(super) fn ensure_registered() {
        REGISTER_ONCE.call_once(|| unsafe {
            if !cblas_sys::is_dgemm_registered() {
                cblas_sys::register_dgemm(dgemm_fallback);
            }
            if !cblas_sys::is_zgemm_registered() {
                cblas_sys::register_zgemm(zgemm_fallback);
            }
        });
    }
}

/// Type-level dispatch trait for CBLAS GEMM.
///
/// Implemented for `f64` (via `cblas_dgemm`) and `Complex64` (via `cblas_zgemm`).
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
pub(crate) fn bgemm_contiguous_into<T: Scalar + BlasGemm>(
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

    let a_batch_strides = a.batch_strides();
    let b_batch_strides = b.batch_strides();
    let c_batch_strides = c.batch_strides();

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

    // Inline closure for per-batch GEMM (shared between fast and slow paths)
    let mut do_batch = |a_off: isize, b_off: isize, c_off: isize| unsafe {
        if c_is_col_major {
            // Standard: C (col-major) = alpha * op(A) * op(B) + beta * C
            // ldc must be >= m for col-major C
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
            // C is row-major (col_stride=1, row_stride>=n).
            // Rewrite as: C^T = alpha * B^T * A^T + beta * C^T
            // where C^T is col-major n×m with ldc >= n.
            let ldc = c.row_stride().max(n as isize).max(1) as i32;
            let flip = |t: cblas_sys::CBLAS_TRANSPOSE| -> cblas_sys::CBLAS_TRANSPOSE {
                match t {
                    cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans => {
                        cblas_sys::CBLAS_TRANSPOSE::CblasTrans
                    }
                    cblas_sys::CBLAS_TRANSPOSE::CblasTrans => {
                        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans
                    }
                    other => other,
                }
            };
            T::gemm(
                flip(trans_b),
                flip(trans_a),
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
    let fused_a = try_fuse_group(batch_dims, a_batch_strides);
    let fused_b = try_fuse_group(batch_dims, b_batch_strides);
    let fused_c = try_fuse_group(batch_dims, c_batch_strides);

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
            let a_off = batch_iter.offset(a_batch_strides);
            let b_off = batch_iter.offset(b_batch_strides);
            let c_off = batch_iter.offset(c_batch_strides);
            do_batch(a_off, b_off, c_off);
        }
    }

    Ok(())
}

impl<T> BgemmBackend<T> for BlasBackend
where
    T: Scalar + BlasGemm,
{
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
