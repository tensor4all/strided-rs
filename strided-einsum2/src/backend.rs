//! Backend abstraction for batched GEMM dispatch.
//!
//! This module defines the [`Backend`] trait, marker structs for each backend,
//! and the `ActiveBackend` type alias that serves as the single point of
//! backend selection based on Cargo features.

use strided_kernel::ExecContext;
use strided_view::ElementOp;

/// Trait for backends that can execute batched GEMM on contiguous operands.
///
/// Each backend declares its configuration (conjugation materialization,
/// stride requirements) and provides a GEMM implementation.
///
/// Implementations are provided by each backend module (faer, blas).
/// External crates can implement this trait for custom scalar types
/// (e.g., tropical semiring) and pass the backend to [`einsum2_with_backend_into`].
///
/// [`einsum2_with_backend_into`]: crate::einsum2_with_backend_into
pub trait Backend<T: crate::ScalarBase> {
    /// Whether the backend needs conjugation materialized into the data
    /// before GEMM (e.g., CBLAS has no conjugation flag for `?gemm`).
    const MATERIALIZES_CONJ: bool;

    /// Whether the backend requires at least one unit stride per matrix
    /// dimension (row or column stride must be 1). CBLAS `?gemm` requires
    /// this; faer does not.
    const REQUIRES_UNIT_STRIDE: bool;

    /// Execute batched GEMM: `C = alpha * A * B + beta * C` for each batch.
    ///
    /// - `c`: mutable output operand (batch x m x n)
    /// - `a`: input operand (batch x m x k)
    /// - `b`: input operand (batch x k x n)
    /// - `batch_dims`: sizes of the batch dimensions
    /// - `m`, `n`, `k`: fused matrix dimensions
    /// - `alpha`, `beta`: scaling factors
    fn bgemm_contiguous_into(
        c: &mut crate::contiguous::ContiguousOperandMut<T>,
        a: &crate::contiguous::ContiguousOperand<T>,
        b: &crate::contiguous::ContiguousOperand<T>,
        batch_dims: &[usize],
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        beta: T,
    ) -> strided_view::Result<()>;
}

/// Private overwrite-only backend contract. The initialized `Backend` trait
/// remains unchanged for beta-bearing callers.
#[allow(dead_code)]
pub(crate) trait OverwriteBackend<T: crate::ScalarBase> {
    fn bgemm_contiguous_overwrite(
        c: &mut crate::contiguous::UninitContiguousOperand<'_, '_, T>,
        a: &crate::contiguous::ContiguousOperand<T>,
        b: &crate::contiguous::ContiguousOperand<T>,
        batch_dims: &[usize],
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        ctx: &ExecContext,
    ) -> strided_view::Result<()>;
}

// ---------------------------------------------------------------------------
// Marker structs
// ---------------------------------------------------------------------------

/// Batched GEMM backend using the [`faer`] library.
///
/// `Backend<T>` is implemented in `bgemm_faer.rs`.
#[cfg(feature = "faer")]
pub struct FaerBackend;

/// Batched GEMM backend using CBLAS (via `cblas-sys` or `cblas-inject`).
///
/// `Backend<T>` is implemented in `bgemm_blas.rs`.
#[cfg(any(feature = "blas", feature = "blas-inject"))]
pub struct BlasBackend;

/// Fallback batched GEMM backend using explicit loops (no external library).
///
/// This backend is used as `ActiveBackend` when no GEMM feature is enabled.
/// The GEMM dispatch in `einsum2_into` calls `bgemm_naive` directly rather
/// than going through the `Backend` trait, so `bgemm_contiguous_into` is
/// unreachable.
#[allow(dead_code)]
pub struct NaiveBackend;

impl<T> Backend<T> for NaiveBackend
where
    T: crate::ScalarBase + strided_view::ElementOpApply,
{
    const MATERIALIZES_CONJ: bool = false;
    const REQUIRES_UNIT_STRIDE: bool = false;

    fn bgemm_contiguous_into(
        c: &mut crate::contiguous::ContiguousOperandMut<T>,
        a: &crate::contiguous::ContiguousOperand<T>,
        b: &crate::contiguous::ContiguousOperand<T>,
        batch_dims: &[usize],
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        beta: T,
    ) -> strided_view::Result<()> {
        let a_ptr = a.ptr();
        let b_ptr = b.ptr();
        let c_ptr = c.ptr();
        let a_rs = a.row_stride();
        let a_cs = a.col_stride();
        let b_rs = b.row_stride();
        let b_cs = b.col_stride();
        let c_rs = c.row_stride();
        let c_cs = c.col_stride();

        let mut batch_iter = crate::util::MultiIndex::new(batch_dims);
        while batch_iter.next().is_some() {
            let a_base = batch_iter.offset(a.batch_strides());
            let b_base = batch_iter.offset(b.batch_strides());
            let c_base = batch_iter.offset(c.batch_strides());

            for i in 0..m {
                for j in 0..n {
                    let mut acc = T::zero();
                    for l in 0..k {
                        let mut a_val = unsafe {
                            *a_ptr.offset(a_base + i as isize * a_rs + l as isize * a_cs)
                        };
                        let mut b_val = unsafe {
                            *b_ptr.offset(b_base + l as isize * b_rs + j as isize * b_cs)
                        };
                        if a.conj() {
                            a_val = strided_view::Conj::apply(a_val);
                        }
                        if b.conj() {
                            b_val = strided_view::Conj::apply(b_val);
                        }
                        acc = acc + a_val * b_val;
                    }
                    unsafe {
                        let c_elem = c_ptr.offset(c_base + i as isize * c_rs + j as isize * c_cs);
                        if beta == T::zero() {
                            *c_elem = alpha * acc;
                        } else {
                            *c_elem = alpha * acc + beta * (*c_elem);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(not(any(feature = "blas", feature = "blas-inject")))]
impl<T> OverwriteBackend<T> for NaiveBackend
where
    T: crate::ScalarBase + strided_view::ElementOpApply,
{
    fn bgemm_contiguous_overwrite(
        c: &mut crate::contiguous::UninitContiguousOperand<'_, '_, T>,
        a: &crate::contiguous::ContiguousOperand<T>,
        b: &crate::contiguous::ContiguousOperand<T>,
        batch_dims: &[usize],
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        ctx: &ExecContext,
    ) -> strided_view::Result<()> {
        crate::uninit::bgemm_contiguous_naive(c, a, b, batch_dims, m, n, k, alpha, ctx)
    }
}

// ---------------------------------------------------------------------------
// ActiveBackend type alias -- the SINGLE point of backend selection
// ---------------------------------------------------------------------------

/// The active GEMM backend, selected by Cargo features.
///
/// - `blas` or `blas-inject` -> `BlasBackend`
/// - `faer` without BLAS -> `FaerBackend`
/// - no backend feature -> `NaiveBackend`
/// - invalid combos -> `NaiveBackend` (placeholder; `compile_error!` fires first)
#[cfg(any(
    all(feature = "blas", not(feature = "blas-inject")),
    all(feature = "blas-inject", not(feature = "blas"))
))]
pub type ActiveBackend = BlasBackend;

#[cfg(all(feature = "faer", not(any(feature = "blas", feature = "blas-inject"))))]
pub type ActiveBackend = FaerBackend;

#[cfg(not(any(feature = "faer", feature = "blas", feature = "blas-inject")))]
pub type ActiveBackend = NaiveBackend;

/// Placeholder for invalid mutually-exclusive feature combinations.
///
/// The crate emits `compile_error!` for these combinations (in `lib.rs`), so this
/// alias only suppresses cascading type-resolution errors.
#[cfg(all(feature = "blas", feature = "blas-inject"))]
pub type ActiveBackend = NaiveBackend;
