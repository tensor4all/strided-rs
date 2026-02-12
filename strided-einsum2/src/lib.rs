//! Binary Einstein summation on strided views.
//!
//! Provides `einsum2_into` for computing binary tensor contractions with
//! accumulation semantics: `C = alpha * A * B + beta * C`.
//!
//! # Example
//!
//! ```
//! use strided_view::StridedArray;
//! use strided_einsum2::einsum2_into;
//!
//! // Matrix multiply: C_ik = A_ij * B_jk
//! let a = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
//! let b = StridedArray::<f64>::from_fn_row_major(&[3, 2], |idx| (idx[0] * 2 + idx[1] + 1) as f64);
//! let mut c = StridedArray::<f64>::row_major(&[2, 2]);
//!
//! einsum2_into(
//!     c.view_mut(), &a.view(), &b.view(),
//!     &['i', 'k'], &['i', 'j'], &['j', 'k'],
//!     1.0, 0.0,
//! ).unwrap();
//! ```

#[cfg(all(feature = "faer", feature = "blas"))]
compile_error!("Features `faer` and `blas` are mutually exclusive. Use one or the other.");

#[cfg(all(feature = "faer", feature = "blas-inject"))]
compile_error!("Features `faer` and `blas-inject` are mutually exclusive.");

#[cfg(all(feature = "blas", feature = "blas-inject"))]
compile_error!("Features `blas` and `blas-inject` are mutually exclusive.");

#[cfg(all(feature = "blas-inject", not(feature = "blas")))]
extern crate cblas_inject as cblas_sys;
#[cfg(all(feature = "blas", not(feature = "blas-inject")))]
extern crate cblas_sys;

#[cfg(all(
    not(feature = "faer"),
    any(
        all(feature = "blas", not(feature = "blas-inject")),
        all(feature = "blas-inject", not(feature = "blas"))
    )
))]
pub mod bgemm_blas;

#[cfg(feature = "faer")]
/// Batched GEMM backend using the [`faer`] library.
pub mod bgemm_faer;
/// Batched GEMM fallback using explicit loops.
pub mod bgemm_naive;
/// GEMM-ready operand types and preparation functions for contiguous data.
pub mod contiguous;
/// Contraction planning: axis classification and permutation computation.
pub mod plan;
/// Trace-axis reduction (summing axes that appear only in one operand).
pub mod trace;
/// Shared helpers (permutation inversion, multi-index iteration, dimension fusion).
pub mod util;

/// Backend abstraction for batched GEMM dispatch.
pub mod backend;

use std::any::TypeId;
use std::fmt::Debug;
use std::hash::Hash;

use strided_kernel::zip_map2_into;
#[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
use strided_view::StridedArray;
use strided_view::{Adjoint, Conj, ElementOp, ElementOpApply, StridedView, StridedViewMut};

pub use strided_traits::ScalarBase;

pub use backend::{BackendConfig, BgemmBackend};
pub use plan::Einsum2Plan;

/// Trait alias for axis label types.
pub trait AxisId: Clone + Eq + Hash + Debug {}
impl<T: Clone + Eq + Hash + Debug> AxisId for T {}

// ScalarBase is re-exported from strided_traits (see above).
// It no longer requires ElementOpApply, enabling custom scalar types
// (e.g., tropical semiring) to work with Identity-only views.

/// Trait alias for element types supported by einsum operations.
///
/// When the `faer` feature is enabled, this additionally requires `faer::ComplexField`
/// so that the faer GEMM backend can be used.
#[cfg(all(feature = "faer", not(any(feature = "blas", feature = "blas-inject"))))]
pub trait Scalar: ScalarBase + ElementOpApply + faer_traits::ComplexField {}

#[cfg(all(feature = "faer", not(any(feature = "blas", feature = "blas-inject"))))]
impl<T> Scalar for T where T: ScalarBase + ElementOpApply + faer_traits::ComplexField {}

/// Trait alias for element types (with `blas` or `blas-inject` feature).
///
/// Includes `BlasGemm` so that all `Scalar` types can be dispatched to CBLAS.
#[cfg(all(
    not(feature = "faer"),
    any(
        all(feature = "blas", not(feature = "blas-inject")),
        all(feature = "blas-inject", not(feature = "blas"))
    )
))]
pub trait Scalar: ScalarBase + ElementOpApply + bgemm_blas::BlasGemm {}

#[cfg(all(
    not(feature = "faer"),
    any(
        all(feature = "blas", not(feature = "blas-inject")),
        all(feature = "blas-inject", not(feature = "blas"))
    )
))]
impl<T> Scalar for T where T: ScalarBase + ElementOpApply + bgemm_blas::BlasGemm {}

/// Trait alias for element types (without `faer` or BLAS features).
#[cfg(not(any(feature = "faer", feature = "blas", feature = "blas-inject")))]
pub trait Scalar: ScalarBase + ElementOpApply {}

#[cfg(not(any(feature = "faer", feature = "blas", feature = "blas-inject")))]
impl<T> Scalar for T where T: ScalarBase + ElementOpApply {}

/// Placeholder trait definition for invalid mutually-exclusive feature combinations.
///
/// The crate emits `compile_error!` above for these combinations. This trait only
/// avoids cascading type-resolution errors so users see the intended diagnostics.
#[cfg(any(
    all(feature = "faer", any(feature = "blas", feature = "blas-inject")),
    all(feature = "blas", feature = "blas-inject")
))]
pub trait Scalar: ScalarBase + ElementOpApply {}

#[cfg(any(
    all(feature = "faer", any(feature = "blas", feature = "blas-inject")),
    all(feature = "blas", feature = "blas-inject")
))]
impl<T> Scalar for T where T: ScalarBase + ElementOpApply {}

/// Errors specific to einsum operations.
#[derive(Debug, thiserror::Error)]
pub enum EinsumError {
    #[error("duplicate axis label: {0}")]
    DuplicateAxis(String),
    #[error("output axis {0} not found in any input")]
    OrphanOutputAxis(String),
    #[error("dimension mismatch for axis {axis:?}: {dim_a} vs {dim_b}")]
    DimensionMismatch {
        axis: String,
        dim_a: usize,
        dim_b: usize,
    },
    #[error(transparent)]
    Strided(#[from] strided_view::StridedError),
}

/// Convenience alias for `Result<T, EinsumError>`.
pub type Result<T> = std::result::Result<T, EinsumError>;

/// Returns `true` if the given `ElementOp` type represents conjugation.
///
/// - `Identity` / `Transpose` → `false` (no per-element conjugation)
/// - `Conj` / `Adjoint` → `true` (per-element conjugation needed)
///
/// For scalar types, `Transpose::apply(x) = x` (identity) and the dimension
/// swap is already reflected in the view's strides/dims.  Similarly,
/// `Adjoint::apply(x) = x.conj()` with the dimension swap in the view.
fn op_is_conj<Op: 'static>() -> bool {
    TypeId::of::<Op>() == TypeId::of::<Conj>() || TypeId::of::<Op>() == TypeId::of::<Adjoint>()
}

/// Binary einsum contraction: `C = alpha * contract(A, B) + beta * C`.
///
/// `ic`, `ia`, `ib` are axis labels for C, A, B respectively.
/// Axes are classified as:
/// - **batch**: in A, B, and C
/// - **lo** (left-output): in A and C, not B
/// - **ro** (right-output): in B and C, not A
/// - **sum** (contraction): in A and B, not C
/// - **left_trace**: only in A (summed out before contraction)
/// - **right_trace**: only in B (summed out before contraction)
pub fn einsum2_into<T: Scalar, OpA, OpB, ID: AxisId>(
    c: StridedViewMut<T>,
    a: &StridedView<T, OpA>,
    b: &StridedView<T, OpB>,
    ic: &[ID],
    ia: &[ID],
    ib: &[ID],
    alpha: T,
    beta: T,
) -> Result<()>
where
    OpA: ElementOp<T> + 'static,
    OpB: ElementOp<T> + 'static,
{
    // 1. Build plan
    let plan = Einsum2Plan::new(ia, ib, ic)?;

    // 2. Validate dimension consistency across operands
    validate_dimensions::<ID>(&plan, a.dims(), b.dims(), c.dims(), ia, ib, ic)?;

    // 3. Reduce trace axes if present; determine conjugation flags.
    //    When trace reduction occurs, Op is already applied during the reduction,
    //    so conj flag is false. Otherwise, we strip the Op and pass a conj flag
    //    to the GEMM kernel (avoiding materialization).
    let (a_buf, conj_a) = if !plan.left_trace.is_empty() {
        let trace_indices = plan.left_trace_indices(ia);
        (Some(trace::reduce_trace_axes(a, &trace_indices)?), false)
    } else {
        (None, op_is_conj::<OpA>())
    };

    let a_view: StridedView<T> = match a_buf.as_ref() {
        Some(buf) => buf.view(),
        None => StridedView::new(a.data(), a.dims(), a.strides(), a.offset())
            .expect("strip_op_view: metadata already validated"),
    };

    let (b_buf, conj_b) = if !plan.right_trace.is_empty() {
        let trace_indices = plan.right_trace_indices(ib);
        (Some(trace::reduce_trace_axes(b, &trace_indices)?), false)
    } else {
        (None, op_is_conj::<OpB>())
    };

    let b_view: StridedView<T> = match b_buf.as_ref() {
        Some(buf) => buf.view(),
        None => StridedView::new(b.data(), b.dims(), b.strides(), b.offset())
            .expect("strip_op_view: metadata already validated"),
    };

    // 4. Dispatch to GEMM
    #[cfg(any(
        all(feature = "faer", not(any(feature = "blas", feature = "blas-inject"))),
        all(
            not(feature = "faer"),
            any(
                all(feature = "blas", not(feature = "blas-inject")),
                all(feature = "blas-inject", not(feature = "blas"))
            )
        )
    ))]
    einsum2_gemm_dispatch(c, &a_view, &b_view, &plan, alpha, beta, conj_a, conj_b)?;

    #[cfg(not(any(feature = "faer", feature = "blas", feature = "blas-inject")))]
    {
        let a_perm = a_view.permute(&plan.left_perm)?;
        let b_perm = b_view.permute(&plan.right_perm)?;
        let mut c_perm = c.permute(&plan.c_to_internal_perm)?;

        if plan.sum.is_empty() && plan.lo.is_empty() && plan.ro.is_empty() && beta == T::zero() {
            let mul_fn = move |a_val: T, b_val: T| -> T {
                let a_c = if conj_a { Conj::apply(a_val) } else { a_val };
                let b_c = if conj_b { Conj::apply(b_val) } else { b_val };
                alpha * a_c * b_c
            };
            zip_map2_into(&mut c_perm, &a_perm, &b_perm, mul_fn)?;
            return Ok(());
        }

        bgemm_naive::bgemm_strided_into(
            &mut c_perm,
            &a_perm,
            &b_perm,
            plan.batch.len(),
            plan.lo.len(),
            plan.ro.len(),
            plan.sum.len(),
            alpha,
            beta,
            conj_a,
            conj_b,
        )?;
    }

    Ok(())
}

/// Binary einsum for custom scalar types using naive GEMM.
///
/// Like [`einsum2_into`] but works with any `ScalarBase` type (no `ElementOpApply` required).
/// Uses closures `map_a` and `map_b` for per-element transformation instead of
/// conjugation flags. Always dispatches to the naive GEMM kernel.
///
/// Views must use `Identity` element operations (the default).
pub fn einsum2_naive_into<T, ID, MapA, MapB>(
    c: StridedViewMut<T>,
    a: &StridedView<T>,
    b: &StridedView<T>,
    ic: &[ID],
    ia: &[ID],
    ib: &[ID],
    alpha: T,
    beta: T,
    map_a: MapA,
    map_b: MapB,
) -> Result<()>
where
    T: ScalarBase,
    ID: AxisId,
    MapA: Fn(T) -> T,
    MapB: Fn(T) -> T,
{
    let plan = Einsum2Plan::new(ia, ib, ic)?;
    validate_dimensions::<ID>(&plan, a.dims(), b.dims(), c.dims(), ia, ib, ic)?;

    // Reduce trace axes if present.
    // When trace reduction occurs, map is applied via map_into before reduction,
    // so we use identity map for GEMM. Otherwise, pass through the original map.
    let (a_buf, use_map_a) = if !plan.left_trace.is_empty() {
        let trace_indices = plan.left_trace_indices(ia);
        let mut mapped = unsafe { strided_view::StridedArray::<T>::col_major_uninit(a.dims()) };
        strided_kernel::map_into(&mut mapped.view_mut(), a, &map_a)?;
        let reduced = trace::reduce_trace_axes(&mapped.view(), &trace_indices)?;
        (Some(reduced), false)
    } else {
        (None, true)
    };
    let a_view: StridedView<T> = match a_buf.as_ref() {
        Some(buf) => buf.view(),
        None => a.clone(),
    };

    let (b_buf, use_map_b) = if !plan.right_trace.is_empty() {
        let trace_indices = plan.right_trace_indices(ib);
        let mut mapped = unsafe { strided_view::StridedArray::<T>::col_major_uninit(b.dims()) };
        strided_kernel::map_into(&mut mapped.view_mut(), b, &map_b)?;
        let reduced = trace::reduce_trace_axes(&mapped.view(), &trace_indices)?;
        (Some(reduced), false)
    } else {
        (None, true)
    };
    let b_view: StridedView<T> = match b_buf.as_ref() {
        Some(buf) => buf.view(),
        None => b.clone(),
    };

    let a_perm = a_view.permute(&plan.left_perm)?;
    let b_perm = b_view.permute(&plan.right_perm)?;
    let mut c_perm = c.permute(&plan.c_to_internal_perm)?;

    // Element-wise fast path
    if plan.sum.is_empty() && plan.lo.is_empty() && plan.ro.is_empty() && beta == T::zero() {
        let mul_fn = move |a_val: T, b_val: T| -> T {
            let a_c = if use_map_a { map_a(a_val) } else { a_val };
            let b_c = if use_map_b { map_b(b_val) } else { b_val };
            alpha * a_c * b_c
        };
        zip_map2_into(&mut c_perm, &a_perm, &b_perm, mul_fn)?;
        return Ok(());
    }

    let final_map_a: Box<dyn Fn(T) -> T> = if use_map_a {
        Box::new(map_a)
    } else {
        Box::new(|x| x)
    };
    let final_map_b: Box<dyn Fn(T) -> T> = if use_map_b {
        Box::new(map_b)
    } else {
        Box::new(|x| x)
    };

    bgemm_naive::bgemm_strided_into_with_map(
        &mut c_perm,
        &a_perm,
        &b_perm,
        plan.batch.len(),
        plan.lo.len(),
        plan.ro.len(),
        plan.sum.len(),
        alpha,
        beta,
        final_map_a,
        final_map_b,
    )?;

    Ok(())
}

/// Binary einsum with a pluggable GEMM backend.
///
/// Like [`einsum2_into`] but works with any `ScalarBase` type and dispatches
/// to the caller-provided GEMM backend `B`. Views must use `Identity` element
/// operations (the default).
///
/// External crates can implement [`BgemmBackend`] for custom scalar types
/// (e.g., tropical semiring) and pass the backend here.
pub fn einsum2_with_backend_into<T, B, ID>(
    c: StridedViewMut<T>,
    a: &StridedView<T>,
    b: &StridedView<T>,
    ic: &[ID],
    ia: &[ID],
    ib: &[ID],
    alpha: T,
    beta: T,
) -> Result<()>
where
    T: ScalarBase,
    B: BgemmBackend<T> + BackendConfig,
    ID: AxisId,
{
    let plan = Einsum2Plan::new(ia, ib, ic)?;
    validate_dimensions::<ID>(&plan, a.dims(), b.dims(), c.dims(), ia, ib, ic)?;

    // Trace reduction (plain sum, Identity views)
    let a_buf = if !plan.left_trace.is_empty() {
        let trace_indices = plan.left_trace_indices(ia);
        Some(trace::reduce_trace_axes(a, &trace_indices)?)
    } else {
        None
    };
    let a_view: StridedView<T> = match a_buf.as_ref() {
        Some(buf) => buf.view(),
        None => a.clone(),
    };

    let b_buf = if !plan.right_trace.is_empty() {
        let trace_indices = plan.right_trace_indices(ib);
        Some(trace::reduce_trace_axes(b, &trace_indices)?)
    } else {
        None
    };
    let b_view: StridedView<T> = match b_buf.as_ref() {
        Some(buf) => buf.view(),
        None => b.clone(),
    };

    // Permute to canonical order
    let a_perm = a_view.permute(&plan.left_perm)?;
    let b_perm = b_view.permute(&plan.right_perm)?;
    let mut c_perm = c.permute(&plan.c_to_internal_perm)?;

    let n_batch = plan.batch.len();
    let n_lo = plan.lo.len();
    let n_ro = plan.ro.len();
    let n_sum = plan.sum.len();

    // Element-wise fast path (all batch, no contraction)
    if plan.sum.is_empty() && plan.lo.is_empty() && plan.ro.is_empty() && beta == T::zero() {
        if alpha == T::one() {
            zip_map2_into(&mut c_perm, &a_perm, &b_perm, |a_val, b_val| a_val * b_val)?;
        } else {
            let mul_fn = move |a_val: T, b_val: T| -> T { alpha * a_val * b_val };
            zip_map2_into(&mut c_perm, &a_perm, &b_perm, mul_fn)?;
        }
        return Ok(());
    }

    // Prepare contiguous operands for GEMM
    let a_op = contiguous::prepare_input_view_for_backend::<T, B>(&a_perm, n_batch, n_lo, n_sum)?;
    let b_op = contiguous::prepare_input_view_for_backend::<T, B>(&b_perm, n_batch, n_sum, n_ro)?;
    let mut c_op = contiguous::prepare_output_view_for_backend::<T, B>(
        &mut c_perm,
        n_batch,
        n_lo,
        n_ro,
        beta,
    )?;

    // Dimension sizes (batch-last canonical order)
    let lo_dims = &a_perm.dims()[..n_lo];
    let sum_dims = &a_perm.dims()[n_lo..n_lo + n_sum];
    let batch_dims = &a_perm.dims()[n_lo + n_sum..];
    let ro_dims = &b_perm.dims()[n_sum..n_sum + n_ro];
    let m: usize = lo_dims.iter().product::<usize>().max(1);
    let k: usize = sum_dims.iter().product::<usize>().max(1);
    let n: usize = ro_dims.iter().product::<usize>().max(1);

    // GEMM via backend
    B::bgemm_contiguous_into(&mut c_op, &a_op, &b_op, batch_dims, m, n, k, alpha, beta)?;

    // Finalize (copy-back if needed)
    c_op.finalize_into(&mut c_perm)?;

    Ok(())
}

/// Internal GEMM dispatch using ContiguousOperand types.
///
/// Called after trace reduction and Op stripping. Handles:
/// 1. Permutation to canonical order
/// 2. Element-wise fast path (if applicable)
/// 3. Contiguous preparation via `prepare_input_view`
/// 4. GEMM via `ActiveBackend::bgemm_contiguous_into`
/// 5. Finalize (copy-back if needed)
#[cfg(any(
    all(feature = "faer", not(any(feature = "blas", feature = "blas-inject"))),
    all(
        not(feature = "faer"),
        any(
            all(feature = "blas", not(feature = "blas-inject")),
            all(feature = "blas-inject", not(feature = "blas"))
        )
    )
))]
fn einsum2_gemm_dispatch<T: Scalar>(
    c: StridedViewMut<T>,
    a: &StridedView<T>,
    b: &StridedView<T>,
    plan: &Einsum2Plan<impl AxisId>,
    alpha: T,
    beta: T,
    conj_a: bool,
    conj_b: bool,
) -> Result<()>
where
    backend::ActiveBackend: BgemmBackend<T>,
{
    // 1. Permute to canonical order
    let a_perm = a.permute(&plan.left_perm)?;
    let b_perm = b.permute(&plan.right_perm)?;
    let mut c_perm = c.permute(&plan.c_to_internal_perm)?;

    // 2. Fast path: element-wise (all batch, no contraction)
    if plan.sum.is_empty() && plan.lo.is_empty() && plan.ro.is_empty() && beta == T::zero() {
        if alpha == T::one() && !conj_a && !conj_b {
            zip_map2_into(&mut c_perm, &a_perm, &b_perm, |a_val, b_val| a_val * b_val)?;
        } else if alpha == T::one() {
            let mul_fn = move |a_val: T, b_val: T| -> T {
                let a_c = if conj_a { Conj::apply(a_val) } else { a_val };
                let b_c = if conj_b { Conj::apply(b_val) } else { b_val };
                a_c * b_c
            };
            zip_map2_into(&mut c_perm, &a_perm, &b_perm, mul_fn)?;
        } else {
            let mul_fn = move |a_val: T, b_val: T| -> T {
                let a_c = if conj_a { Conj::apply(a_val) } else { a_val };
                let b_c = if conj_b { Conj::apply(b_val) } else { b_val };
                alpha * a_c * b_c
            };
            zip_map2_into(&mut c_perm, &a_perm, &b_perm, mul_fn)?;
        }
        return Ok(());
    }

    // 3. Prepare contiguous operands
    let n_batch = plan.batch.len();
    let n_lo = plan.lo.len();
    let n_ro = plan.ro.len();
    let n_sum = plan.sum.len();

    let a_op = contiguous::prepare_input_view(&a_perm, n_batch, n_lo, n_sum, conj_a)?;
    let b_op = contiguous::prepare_input_view(&b_perm, n_batch, n_sum, n_ro, conj_b)?;
    let mut c_op = contiguous::prepare_output_view(&mut c_perm, n_batch, n_lo, n_ro, beta)?;

    // Compute fused dimension sizes
    let lo_dims = &a_perm.dims()[..n_lo];
    let sum_dims = &a_perm.dims()[n_lo..n_lo + n_sum];
    let batch_dims = &a_perm.dims()[n_lo + n_sum..];
    let ro_dims = &b_perm.dims()[n_sum..n_sum + n_ro];
    let m: usize = lo_dims.iter().product::<usize>().max(1);
    let k: usize = sum_dims.iter().product::<usize>().max(1);
    let n: usize = ro_dims.iter().product::<usize>().max(1);

    // 4. GEMM — dispatched through trait
    backend::ActiveBackend::bgemm_contiguous_into(
        &mut c_op, &a_op, &b_op, batch_dims, m, n, k, alpha, beta,
    )?;

    // 5. Finalize
    c_op.finalize_into(&mut c_perm)?;

    Ok(())
}

/// Binary einsum accepting owned inputs for zero-copy optimization.
///
/// Same semantics as [`einsum2_into`] but accepts owned `StridedArray` inputs.
/// When inputs have non-contiguous strides after permutation, ownership
/// transfer avoids allocating separate buffers. For contiguous inputs,
/// the behavior is identical.
///
/// `conj_a` and `conj_b` indicate whether to conjugate elements of A/B.
#[cfg(any(
    all(feature = "faer", not(any(feature = "blas", feature = "blas-inject"))),
    all(
        not(feature = "faer"),
        any(
            all(feature = "blas", not(feature = "blas-inject")),
            all(feature = "blas-inject", not(feature = "blas"))
        )
    )
))]
pub fn einsum2_into_owned<T: Scalar, ID: AxisId>(
    c: StridedViewMut<T>,
    a: StridedArray<T>,
    b: StridedArray<T>,
    ic: &[ID],
    ia: &[ID],
    ib: &[ID],
    alpha: T,
    beta: T,
    conj_a: bool,
    conj_b: bool,
) -> Result<()>
where
    backend::ActiveBackend: BgemmBackend<T>,
{
    // 1. Build plan
    let plan = Einsum2Plan::new(ia, ib, ic)?;

    // 2. Validate dimensions
    validate_dimensions::<ID>(&plan, a.dims(), b.dims(), c.dims(), ia, ib, ic)?;

    // 3. Trace reduction: reduce trace axes if present.
    //    When trace reduction occurs, conjugation is applied during reduction,
    //    so the conj flag becomes false. Otherwise keep the caller's flag.
    let (a_for_gemm, conj_a_final) = if !plan.left_trace.is_empty() {
        let trace_indices = plan.left_trace_indices(ia);
        (trace::reduce_trace_axes(&a.view(), &trace_indices)?, false)
    } else {
        (a, conj_a)
    };

    let (b_for_gemm, conj_b_final) = if !plan.right_trace.is_empty() {
        let trace_indices = plan.right_trace_indices(ib);
        (trace::reduce_trace_axes(&b.view(), &trace_indices)?, false)
    } else {
        (b, conj_b)
    };

    // 4. Permute to canonical order (metadata-only on owned arrays)
    let a_perm = a_for_gemm.permuted(&plan.left_perm)?;
    let b_perm = b_for_gemm.permuted(&plan.right_perm)?;
    let mut c_perm = c.permute(&plan.c_to_internal_perm)?;

    let n_batch = plan.batch.len();
    let n_lo = plan.lo.len();
    let n_ro = plan.ro.len();
    let n_sum = plan.sum.len();

    // 5. Fast path: element-wise (all batch, no contraction)
    if plan.sum.is_empty() && plan.lo.is_empty() && plan.ro.is_empty() && beta == T::zero() {
        let mul_fn = move |a_val: T, b_val: T| -> T {
            let a_c = if conj_a_final {
                Conj::apply(a_val)
            } else {
                a_val
            };
            let b_c = if conj_b_final {
                Conj::apply(b_val)
            } else {
                b_val
            };
            alpha * a_c * b_c
        };
        zip_map2_into(&mut c_perm, &a_perm.view(), &b_perm.view(), mul_fn)?;
        return Ok(());
    }

    // 6. Extract dimension sizes BEFORE consuming arrays via prepare_input_owned
    let a_dims_perm = a_perm.dims().to_vec();
    let b_dims_perm = b_perm.dims().to_vec();

    let lo_dims = &a_dims_perm[..n_lo];
    let sum_dims = &a_dims_perm[n_lo..n_lo + n_sum];
    let batch_dims = a_dims_perm[n_lo + n_sum..].to_vec();
    let ro_dims = &b_dims_perm[n_sum..n_sum + n_ro];
    let m: usize = lo_dims.iter().product::<usize>().max(1);
    let k: usize = sum_dims.iter().product::<usize>().max(1);
    let n: usize = ro_dims.iter().product::<usize>().max(1);

    // 7. Prepare contiguous operands (owned path -- avoids extra copies)
    let a_op = contiguous::prepare_input_owned(a_perm, n_batch, n_lo, n_sum, conj_a_final)?;
    let b_op = contiguous::prepare_input_owned(b_perm, n_batch, n_sum, n_ro, conj_b_final)?;
    let mut c_op = contiguous::prepare_output_view(&mut c_perm, n_batch, n_lo, n_ro, beta)?;

    // 8. GEMM — dispatched through trait
    backend::ActiveBackend::bgemm_contiguous_into(
        &mut c_op,
        &a_op,
        &b_op,
        &batch_dims,
        m,
        n,
        k,
        alpha,
        beta,
    )?;

    // 9. Finalize
    c_op.finalize_into(&mut c_perm)?;

    Ok(())
}

/// Validate that dimensions match across operands for each axis group.
fn validate_dimensions<ID: AxisId>(
    plan: &Einsum2Plan<ID>,
    a_dims: &[usize],
    b_dims: &[usize],
    c_dims: &[usize],
    ia: &[ID],
    ib: &[ID],
    ic: &[ID],
) -> Result<()> {
    let find_dim = |labels: &[ID], dims: &[usize], id: &ID| -> usize {
        labels
            .iter()
            .position(|x| x == id)
            .map(|i| dims[i])
            .unwrap()
    };

    // Batch: must match in A, B, and C
    for id in &plan.batch {
        let da = find_dim(ia, a_dims, id);
        let db = find_dim(ib, b_dims, id);
        let dc = find_dim(ic, c_dims, id);
        if da != db || da != dc {
            return Err(EinsumError::DimensionMismatch {
                axis: format!("{:?}", id),
                dim_a: da,
                dim_b: db,
            });
        }
    }

    // Sum: must match in A and B
    for id in &plan.sum {
        let da = find_dim(ia, a_dims, id);
        let db = find_dim(ib, b_dims, id);
        if da != db {
            return Err(EinsumError::DimensionMismatch {
                axis: format!("{:?}", id),
                dim_a: da,
                dim_b: db,
            });
        }
    }

    // LO: must match in A and C
    for id in &plan.lo {
        let da = find_dim(ia, a_dims, id);
        let dc = find_dim(ic, c_dims, id);
        if da != dc {
            return Err(EinsumError::DimensionMismatch {
                axis: format!("{:?}", id),
                dim_a: da,
                dim_b: dc,
            });
        }
    }

    // RO: must match in B and C
    for id in &plan.ro {
        let db = find_dim(ib, b_dims, id);
        let dc = find_dim(ic, c_dims, id);
        if db != dc {
            return Err(EinsumError::DimensionMismatch {
                axis: format!("{:?}", id),
                dim_a: db,
                dim_b: dc,
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use strided_view::StridedArray;

    #[test]
    fn test_matmul_ij_jk_ik() {
        // C_ik = A_ij * B_jk
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[5.0, 6.0], [7.0, 8.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f64>::row_major(&[2, 2]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['i', 'k'],
            &['i', 'j'],
            &['j', 'k'],
            1.0,
            0.0,
        )
        .unwrap();

        assert_eq!(c.get(&[0, 0]), 19.0);
        assert_eq!(c.get(&[0, 1]), 22.0);
        assert_eq!(c.get(&[1, 0]), 43.0);
        assert_eq!(c.get(&[1, 1]), 50.0);
    }

    #[test]
    fn test_matmul_rect() {
        // A: 2x3, B: 3x4, C: 2x4
        let a =
            StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        let b =
            StridedArray::<f64>::from_fn_row_major(&[3, 4], |idx| (idx[0] * 4 + idx[1] + 1) as f64);
        let mut c = StridedArray::<f64>::row_major(&[2, 4]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['i', 'k'],
            &['i', 'j'],
            &['j', 'k'],
            1.0,
            0.0,
        )
        .unwrap();

        // A = [[1,2,3],[4,5,6]], B = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
        assert_eq!(c.get(&[0, 0]), 38.0);
        assert_eq!(c.get(&[1, 3]), 128.0);
    }

    #[test]
    fn test_batched_matmul() {
        // C_bik = A_bij * B_bjk
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2, 3], |idx| {
            (idx[0] * 6 + idx[1] * 3 + idx[2] + 1) as f64
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 3, 2], |idx| {
            (idx[0] * 6 + idx[1] * 2 + idx[2] + 1) as f64
        });
        let mut c = StridedArray::<f64>::row_major(&[2, 2, 2]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['b', 'i', 'k'],
            &['b', 'i', 'j'],
            &['b', 'j', 'k'],
            1.0,
            0.0,
        )
        .unwrap();

        // Batch 0: A0=[[1,2,3],[4,5,6]], B0=[[1,2],[3,4],[5,6]]
        // C0[0,0] = 1*1+2*3+3*5 = 22
        assert_eq!(c.get(&[0, 0, 0]), 22.0);
    }

    #[test]
    fn test_batched_matmul_col_major_output() {
        // C_bik = A_bij * B_bjk with col-major output (same layout as opteinsum)
        let a_data = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = StridedArray::<f64>::from_parts(a_data, &[2, 2, 2], &[4, 2, 1], 0).unwrap();
        let b = StridedArray::<f64>::from_parts(b_data, &[2, 2, 2], &[4, 2, 1], 0).unwrap();
        let mut c = StridedArray::<f64>::col_major(&[2, 2, 2]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['b', 'i', 'k'],
            &['b', 'i', 'j'],
            &['b', 'j', 'k'],
            1.0,
            0.0,
        )
        .unwrap();

        // batch 0: I * [[1,2],[3,4]] = [[1,2],[3,4]]
        assert_eq!(c.get(&[0, 0, 0]), 1.0);
        assert_eq!(c.get(&[0, 0, 1]), 2.0);
        assert_eq!(c.get(&[0, 1, 0]), 3.0);
        assert_eq!(c.get(&[0, 1, 1]), 4.0);
        // batch 1: 2I * [[5,6],[7,8]] = [[10,12],[14,16]]
        assert_eq!(c.get(&[1, 0, 0]), 10.0);
        assert_eq!(c.get(&[1, 1, 1]), 16.0);
    }

    #[test]
    fn test_outer_product() {
        // C_ij = A_i * B_j
        let a = StridedArray::<f64>::from_fn_row_major(&[3], |idx| (idx[0] + 1) as f64);
        let b = StridedArray::<f64>::from_fn_row_major(&[4], |idx| (idx[0] + 1) as f64);
        let mut c = StridedArray::<f64>::row_major(&[3, 4]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['i', 'j'],
            &['i'],
            &['j'],
            1.0,
            0.0,
        )
        .unwrap();

        assert_eq!(c.get(&[0, 0]), 1.0);
        assert_eq!(c.get(&[2, 3]), 12.0);
    }

    #[test]
    fn test_dot_product() {
        // C = A_i * B_i (scalar output)
        let a = StridedArray::<f64>::from_fn_row_major(&[3], |idx| (idx[0] + 1) as f64);
        let b = StridedArray::<f64>::from_fn_row_major(&[3], |idx| (idx[0] + 1) as f64);
        let mut c = StridedArray::<f64>::row_major(&[]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &[] as &[char],
            &['i'],
            &['i'],
            1.0,
            0.0,
        )
        .unwrap();

        // 1*1 + 2*2 + 3*3 = 14
        assert_eq!(c.get(&[]), 14.0);
    }

    #[test]
    fn test_alpha_beta() {
        // C = 2*A*B + 3*C_old
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 0.0], [0.0, 1.0]][idx[0]][idx[1]] // identity
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[10.0, 20.0], [30.0, 40.0]][idx[0]][idx[1]]
        });

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['i', 'k'],
            &['i', 'j'],
            &['j', 'k'],
            2.0,
            3.0,
        )
        .unwrap();

        // C = 2*I*B + 3*C_old
        assert_eq!(c.get(&[0, 0]), 32.0); // 2*1 + 3*10
        assert_eq!(c.get(&[1, 1]), 128.0); // 2*4 + 3*40
    }

    #[test]
    fn test_transposed_output() {
        // C_ki = A_ij * B_jk (output transposed)
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[5.0, 6.0], [7.0, 8.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f64>::row_major(&[2, 2]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['k', 'i'], // C indexed as (k, i) instead of (i, k)
            &['i', 'j'],
            &['j', 'k'],
            1.0,
            0.0,
        )
        .unwrap();

        // Normal matmul result: C_ik = [[19,22],[43,50]]
        // But C is indexed as (k, i), so C[k, i] = (A*B)[i, k]
        assert_eq!(c.get(&[0, 0]), 19.0); // C[k=0, i=0]
        assert_eq!(c.get(&[0, 1]), 43.0); // C[k=0, i=1]
        assert_eq!(c.get(&[1, 0]), 22.0); // C[k=1, i=0]
        assert_eq!(c.get(&[1, 1]), 50.0); // C[k=1, i=1]
    }

    #[test]
    fn test_left_trace() {
        // C_k = sum_j (sum_i A_ij) * B_jk
        // left_trace=[i], sum=[j], ro=[k]
        let a =
            StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        // A = [[1,2,3],[4,5,6]]
        // sum over i: [5, 7, 9]
        let b =
            StridedArray::<f64>::from_fn_row_major(&[3, 2], |idx| (idx[0] * 2 + idx[1] + 1) as f64);
        // B = [[1,2],[3,4],[5,6]]
        let mut c = StridedArray::<f64>::row_major(&[2]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['k'],
            &['i', 'j'],
            &['j', 'k'],
            1.0,
            0.0,
        )
        .unwrap();

        // C_k = sum_j [5,7,9][j] * B[j,k]
        // C[0] = 5*1 + 7*3 + 9*5 = 5 + 21 + 45 = 71
        // C[1] = 5*2 + 7*4 + 9*6 = 10 + 28 + 54 = 92
        assert_eq!(c.get(&[0]), 71.0);
        assert_eq!(c.get(&[1]), 92.0);
    }

    #[test]
    fn test_u32_labels() {
        // Same as matmul but with u32 labels
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[5.0, 6.0], [7.0, 8.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f64>::row_major(&[2, 2]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &[0u32, 2],
            &[0u32, 1],
            &[1u32, 2],
            1.0,
            0.0,
        )
        .unwrap();

        assert_eq!(c.get(&[0, 0]), 19.0);
        assert_eq!(c.get(&[1, 1]), 50.0);
    }

    #[test]
    fn test_complex_matmul() {
        use num_complex::Complex64;
        let i = Complex64::i();

        // A = [[1+i, 2], [3, 4-i]]
        let a_vals = [
            [1.0 + i, Complex64::new(2.0, 0.0)],
            [Complex64::new(3.0, 0.0), 4.0 - i],
        ];
        let a = StridedArray::<Complex64>::from_fn_row_major(&[2, 2], |idx| a_vals[idx[0]][idx[1]]);

        // B = [[1, i], [0, 1]]
        let b_vals = [
            [Complex64::new(1.0, 0.0), i],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        let b = StridedArray::<Complex64>::from_fn_row_major(&[2, 2], |idx| b_vals[idx[0]][idx[1]]);

        let mut c = StridedArray::<Complex64>::row_major(&[2, 2]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['i', 'k'],
            &['i', 'j'],
            &['j', 'k'],
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
        .unwrap();

        // C = A * B
        // C[0,0] = (1+i)*1 + 2*0 = 1+i
        // C[0,1] = (1+i)*i + 2*1 = i+i²+2 = i-1+2 = 1+i
        // C[1,0] = 3*1 + (4-i)*0 = 3
        // C[1,1] = 3*i + (4-i)*1 = 3i+4-i = 4+2i
        assert_eq!(c.get(&[0, 0]), 1.0 + i);
        assert_eq!(c.get(&[0, 1]), 1.0 + i);
        assert_eq!(c.get(&[1, 0]), Complex64::new(3.0, 0.0));
        assert_eq!(c.get(&[1, 1]), 4.0 + 2.0 * i);
    }

    #[test]
    fn test_complex_matmul_with_conj() {
        use num_complex::Complex64;
        let i = Complex64::i();

        // A = [[1+i, 2i], [3, 4-i]]
        let a_vals = [[1.0 + i, 2.0 * i], [Complex64::new(3.0, 0.0), 4.0 - i]];
        let a = StridedArray::<Complex64>::from_fn_row_major(&[2, 2], |idx| a_vals[idx[0]][idx[1]]);

        // B = identity
        let b = StridedArray::<Complex64>::from_fn_row_major(&[2, 2], |idx| {
            if idx[0] == idx[1] {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });

        let mut c = StridedArray::<Complex64>::row_major(&[2, 2]);

        // C = conj(A) * B = conj(A)
        let a_conj = a.view().conj();
        einsum2_into(
            c.view_mut(),
            &a_conj,
            &b.view(),
            &['i', 'k'],
            &['i', 'j'],
            &['j', 'k'],
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
        .unwrap();

        // conj(A) = [[1-i, -2i], [3, 4+i]]
        assert_eq!(c.get(&[0, 0]), 1.0 - i);
        assert_eq!(c.get(&[0, 1]), -2.0 * i);
        assert_eq!(c.get(&[1, 0]), Complex64::new(3.0, 0.0));
        assert_eq!(c.get(&[1, 1]), 4.0 + i);
    }

    #[test]
    fn test_complex_matmul_with_conj_both() {
        use num_complex::Complex64;
        let i = Complex64::i();

        // A = [[1+i, 0], [0, 2-i]]
        let a_vals = [
            [1.0 + i, Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), 2.0 - i],
        ];
        let a = StridedArray::<Complex64>::from_fn_row_major(&[2, 2], |idx| a_vals[idx[0]][idx[1]]);

        // B = [[1, i], [0, 1+i]]
        let b_vals = [
            [Complex64::new(1.0, 0.0), i],
            [Complex64::new(0.0, 0.0), 1.0 + i],
        ];
        let b = StridedArray::<Complex64>::from_fn_row_major(&[2, 2], |idx| b_vals[idx[0]][idx[1]]);

        let mut c = StridedArray::<Complex64>::row_major(&[2, 2]);

        // C = conj(A) * conj(B)
        let a_conj = a.view().conj();
        let b_conj = b.view().conj();
        einsum2_into(
            c.view_mut(),
            &a_conj,
            &b_conj,
            &['i', 'k'],
            &['i', 'j'],
            &['j', 'k'],
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
        .unwrap();

        // conj(A) = [[1-i, 0], [0, 2+i]]
        // conj(B) = [[1, -i], [0, 1-i]]
        // C = conj(A) * conj(B)
        // C[0,0] = (1-i)*1 + 0*0 = 1-i
        // C[0,1] = (1-i)*(-i) + 0*(1-i) = -i+i² = -i-1 = -(1+i)
        // C[1,0] = 0*1 + (2+i)*0 = 0
        // C[1,1] = 0*(-i) + (2+i)*(1-i) = 2-2i+i-i² = 2-i+1 = 3-i
        assert_eq!(c.get(&[0, 0]), 1.0 - i);
        assert_eq!(c.get(&[0, 1]), -(1.0 + i));
        assert_eq!(c.get(&[1, 0]), Complex64::new(0.0, 0.0));
        assert_eq!(c.get(&[1, 1]), 3.0 - i);
    }

    #[test]
    fn test_elementwise_hadamard() {
        // C_ijk = A_ijk * B_ijk — all batch, no contraction
        let a = StridedArray::<f64>::from_fn_row_major(&[3, 4, 5], |idx| {
            (idx[0] * 20 + idx[1] * 5 + idx[2] + 1) as f64
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[3, 4, 5], |idx| {
            (idx[0] * 20 + idx[1] * 5 + idx[2] + 1) as f64 * 0.1
        });
        let mut c = StridedArray::<f64>::row_major(&[3, 4, 5]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['i', 'j', 'k'],
            &['i', 'j', 'k'],
            &['i', 'j', 'k'],
            1.0,
            0.0,
        )
        .unwrap();

        // Spot check: C[0,0,0] = 1 * 0.1 = 0.1
        assert!((c.get(&[0, 0, 0]) - 0.1).abs() < 1e-12);
        // C[2,3,4] = 60 * 6.0 = 360
        assert!((c.get(&[2, 3, 4]) - 360.0).abs() < 1e-10);
    }

    #[test]
    fn test_elementwise_hadamard_with_alpha() {
        let a =
            StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        let b =
            StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        let mut c = StridedArray::<f64>::row_major(&[2, 3]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['i', 'j'],
            &['i', 'j'],
            &['i', 'j'],
            2.0,
            0.0,
        )
        .unwrap();

        // C[0,0] = 2.0 * 1 * 1 = 2.0
        assert_eq!(c.get(&[0, 0]), 2.0);
        // C[1,2] = 2.0 * 6 * 6 = 72.0
        assert_eq!(c.get(&[1, 2]), 72.0);
    }

    #[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
    #[test]
    fn test_einsum2_owned_matmul() {
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[5.0, 6.0], [7.0, 8.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f64>::row_major(&[2, 2]);

        einsum2_into_owned(
            c.view_mut(),
            a,
            b,
            &['i', 'k'],
            &['i', 'j'],
            &['j', 'k'],
            1.0,
            0.0,
            false,
            false,
        )
        .unwrap();

        assert_eq!(c.get(&[0, 0]), 19.0);
        assert_eq!(c.get(&[0, 1]), 22.0);
        assert_eq!(c.get(&[1, 0]), 43.0);
        assert_eq!(c.get(&[1, 1]), 50.0);
    }

    #[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
    #[test]
    fn test_einsum2_owned_batched() {
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2, 3], |idx| {
            (idx[0] * 6 + idx[1] * 3 + idx[2] + 1) as f64
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 3, 2], |idx| {
            (idx[0] * 6 + idx[1] * 2 + idx[2] + 1) as f64
        });
        let mut c = StridedArray::<f64>::row_major(&[2, 2, 2]);

        einsum2_into_owned(
            c.view_mut(),
            a,
            b,
            &['b', 'i', 'k'],
            &['b', 'i', 'j'],
            &['b', 'j', 'k'],
            1.0,
            0.0,
            false,
            false,
        )
        .unwrap();

        // Batch 0: A0=[[1,2,3],[4,5,6]], B0=[[1,2],[3,4],[5,6]]
        // C0[0,0] = 1*1+2*3+3*5 = 22
        assert_eq!(c.get(&[0, 0, 0]), 22.0);
    }

    #[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
    #[test]
    fn test_einsum2_owned_alpha_beta() {
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 0.0], [0.0, 1.0]][idx[0]][idx[1]]
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[10.0, 20.0], [30.0, 40.0]][idx[0]][idx[1]]
        });

        einsum2_into_owned(
            c.view_mut(),
            a,
            b,
            &['i', 'k'],
            &['i', 'j'],
            &['j', 'k'],
            2.0,
            3.0,
            false,
            false,
        )
        .unwrap();

        // C = 2*I*B + 3*C_old
        assert_eq!(c.get(&[0, 0]), 32.0); // 2*1 + 3*10
        assert_eq!(c.get(&[1, 1]), 128.0); // 2*4 + 3*40
    }

    #[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
    #[test]
    fn test_einsum2_owned_elementwise() {
        // All batch, no contraction -- element-wise fast path
        let a =
            StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        let b =
            StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        let mut c = StridedArray::<f64>::row_major(&[2, 3]);

        einsum2_into_owned(
            c.view_mut(),
            a,
            b,
            &['i', 'j'],
            &['i', 'j'],
            &['i', 'j'],
            2.0,
            0.0,
            false,
            false,
        )
        .unwrap();

        assert_eq!(c.get(&[0, 0]), 2.0); // 2 * 1 * 1
        assert_eq!(c.get(&[1, 2]), 72.0); // 2 * 6 * 6
    }

    #[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
    #[test]
    fn test_einsum2_owned_left_trace() {
        // C_k = sum_j (sum_i A_ij) * B_jk
        // left_trace=[i], sum=[j], ro=[k]
        let a =
            StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        // A = [[1,2,3],[4,5,6]]
        // sum over i: [5, 7, 9]
        let b =
            StridedArray::<f64>::from_fn_row_major(&[3, 2], |idx| (idx[0] * 2 + idx[1] + 1) as f64);
        // B = [[1,2],[3,4],[5,6]]
        let mut c = StridedArray::<f64>::row_major(&[2]);

        einsum2_into_owned(
            c.view_mut(),
            a,
            b,
            &['k'],
            &['i', 'j'],
            &['j', 'k'],
            1.0,
            0.0,
            false,
            false,
        )
        .unwrap();

        // C[0] = 5*1 + 7*3 + 9*5 = 5 + 21 + 45 = 71
        // C[1] = 5*2 + 7*4 + 9*6 = 10 + 28 + 54 = 92
        assert_eq!(c.get(&[0]), 71.0);
        assert_eq!(c.get(&[1]), 92.0);
    }

    #[test]
    fn test_einsum2_naive_matmul() {
        // einsum2_naive_into with identity maps = regular matmul
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[5.0, 6.0], [7.0, 8.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f64>::row_major(&[2, 2]);

        einsum2_naive_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['i', 'k'],
            &['i', 'j'],
            &['j', 'k'],
            1.0,
            0.0,
            |x| x,
            |x| x,
        )
        .unwrap();

        assert_eq!(c.get(&[0, 0]), 19.0);
        assert_eq!(c.get(&[0, 1]), 22.0);
        assert_eq!(c.get(&[1, 0]), 43.0);
        assert_eq!(c.get(&[1, 1]), 50.0);
    }

    #[test]
    fn test_einsum2_naive_elementwise() {
        // Element-wise (all batch) with identity maps
        let a =
            StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        let b =
            StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        let mut c = StridedArray::<f64>::row_major(&[2, 3]);

        einsum2_naive_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['i', 'j'],
            &['i', 'j'],
            &['i', 'j'],
            2.0,
            0.0,
            |x| x,
            |x| x,
        )
        .unwrap();

        assert_eq!(c.get(&[0, 0]), 2.0); // 2 * 1 * 1
        assert_eq!(c.get(&[1, 2]), 72.0); // 2 * 6 * 6
    }

    #[test]
    fn test_einsum2_naive_custom_type() {
        // Custom scalar type that does NOT implement ElementOpApply.
        // This demonstrates that einsum2_naive_into works with custom types.
        use num_traits::{One, Zero};

        #[derive(Debug, Clone, Copy, PartialEq)]
        struct MyVal(f64);

        impl Default for MyVal {
            fn default() -> Self {
                MyVal(0.0)
            }
        }

        impl std::ops::Add for MyVal {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                MyVal(self.0 + rhs.0)
            }
        }

        impl std::ops::Mul for MyVal {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self {
                MyVal(self.0 * rhs.0)
            }
        }

        impl Zero for MyVal {
            fn zero() -> Self {
                MyVal(0.0)
            }
            fn is_zero(&self) -> bool {
                self.0 == 0.0
            }
        }

        impl One for MyVal {
            fn one() -> Self {
                MyVal(1.0)
            }
        }

        // C_ik = A_ij * B_jk (matrix multiply with custom type)
        let a = StridedArray::from_parts(
            vec![MyVal(1.0), MyVal(2.0), MyVal(3.0), MyVal(4.0)],
            &[2, 2],
            &[2, 1],
            0,
        )
        .unwrap();
        let b = StridedArray::from_parts(
            vec![MyVal(5.0), MyVal(6.0), MyVal(7.0), MyVal(8.0)],
            &[2, 2],
            &[2, 1],
            0,
        )
        .unwrap();
        let mut c = StridedArray::<MyVal>::col_major(&[2, 2]);

        einsum2_naive_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['i', 'k'],
            &['i', 'j'],
            &['j', 'k'],
            MyVal(1.0),
            MyVal(0.0),
            |x| x,
            |x| x,
        )
        .unwrap();

        // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //   = [[19, 22], [43, 50]]
        assert_eq!(c.get(&[0, 0]), MyVal(19.0));
        assert_eq!(c.get(&[0, 1]), MyVal(22.0));
        assert_eq!(c.get(&[1, 0]), MyVal(43.0));
        assert_eq!(c.get(&[1, 1]), MyVal(50.0));
    }

    #[test]
    fn test_einsum2_naive_left_trace() {
        // C_k = sum_j (sum_i A_ij) * B_jk with map_a = identity
        let a =
            StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        let b =
            StridedArray::<f64>::from_fn_row_major(&[3, 2], |idx| (idx[0] * 2 + idx[1] + 1) as f64);
        let mut c = StridedArray::<f64>::row_major(&[2]);

        einsum2_naive_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['k'],
            &['i', 'j'],
            &['j', 'k'],
            1.0,
            0.0,
            |x| x,
            |x| x,
        )
        .unwrap();

        assert_eq!(c.get(&[0]), 71.0);
        assert_eq!(c.get(&[1]), 92.0);
    }

    /// Test backend that delegates to naive GEMM loops.
    struct TestNaiveBackend;

    impl BackendConfig for TestNaiveBackend {
        const MATERIALIZES_CONJ: bool = false;
        const REQUIRES_UNIT_STRIDE: bool = false;
    }

    impl BgemmBackend<f64> for TestNaiveBackend {
        fn bgemm_contiguous_into(
            c: &mut contiguous::ContiguousOperandMut<f64>,
            a: &contiguous::ContiguousOperand<f64>,
            b: &contiguous::ContiguousOperand<f64>,
            batch_dims: &[usize],
            m: usize,
            n: usize,
            k: usize,
            alpha: f64,
            beta: f64,
        ) -> strided_view::Result<()> {
            // Delegate to the contiguous naive GEMM loop
            let a_ptr = a.ptr();
            let b_ptr = b.ptr();
            let c_ptr = c.ptr();
            let a_rs = a.row_stride();
            let a_cs = a.col_stride();
            let b_rs = b.row_stride();
            let b_cs = b.col_stride();
            let c_rs = c.row_stride();
            let c_cs = c.col_stride();

            let mut batch_idx = crate::util::MultiIndex::new(batch_dims);
            while batch_idx.next().is_some() {
                let a_base = batch_idx.offset(a.batch_strides());
                let b_base = batch_idx.offset(b.batch_strides());
                let c_base = batch_idx.offset(c.batch_strides());

                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0.0f64;
                        for l in 0..k {
                            let a_val = unsafe {
                                *a_ptr.offset(a_base + i as isize * a_rs + l as isize * a_cs)
                            };
                            let b_val = unsafe {
                                *b_ptr.offset(b_base + l as isize * b_rs + j as isize * b_cs)
                            };
                            acc += a_val * b_val;
                        }
                        unsafe {
                            let c_elem =
                                c_ptr.offset(c_base + i as isize * c_rs + j as isize * c_cs);
                            if beta == 0.0 {
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

    #[test]
    fn test_einsum2_with_backend_matmul() {
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[5.0, 6.0], [7.0, 8.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f64>::row_major(&[2, 2]);

        einsum2_with_backend_into::<_, TestNaiveBackend, _>(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['i', 'k'],
            &['i', 'j'],
            &['j', 'k'],
            1.0,
            0.0,
        )
        .unwrap();

        assert_eq!(c.get(&[0, 0]), 19.0);
        assert_eq!(c.get(&[0, 1]), 22.0);
        assert_eq!(c.get(&[1, 0]), 43.0);
        assert_eq!(c.get(&[1, 1]), 50.0);
    }

    #[test]
    fn test_einsum2_with_backend_elementwise() {
        let a =
            StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        let b =
            StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        let mut c = StridedArray::<f64>::row_major(&[2, 3]);

        einsum2_with_backend_into::<_, TestNaiveBackend, _>(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['i', 'j'],
            &['i', 'j'],
            &['i', 'j'],
            2.0,
            0.0,
        )
        .unwrap();

        assert_eq!(c.get(&[0, 0]), 2.0); // 2 * 1 * 1
        assert_eq!(c.get(&[1, 2]), 72.0); // 2 * 6 * 6
    }

    #[test]
    fn test_einsum2_with_backend_custom_type() {
        use num_traits::{One, Zero};

        #[derive(Debug, Clone, Copy, PartialEq)]
        struct Tropical(f64);

        impl Default for Tropical {
            fn default() -> Self {
                Tropical(0.0)
            }
        }

        impl std::ops::Add for Tropical {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                Tropical(self.0 + rhs.0)
            }
        }

        impl std::ops::Mul for Tropical {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self {
                Tropical(self.0 * rhs.0)
            }
        }

        impl Zero for Tropical {
            fn zero() -> Self {
                Tropical(0.0)
            }
            fn is_zero(&self) -> bool {
                self.0 == 0.0
            }
        }

        impl One for Tropical {
            fn one() -> Self {
                Tropical(1.0)
            }
        }

        struct TropicalBackend;

        impl BackendConfig for TropicalBackend {
            const MATERIALIZES_CONJ: bool = false;
            const REQUIRES_UNIT_STRIDE: bool = false;
        }

        impl BgemmBackend<Tropical> for TropicalBackend {
            fn bgemm_contiguous_into(
                c: &mut contiguous::ContiguousOperandMut<Tropical>,
                a: &contiguous::ContiguousOperand<Tropical>,
                b: &contiguous::ContiguousOperand<Tropical>,
                batch_dims: &[usize],
                m: usize,
                n: usize,
                k: usize,
                alpha: Tropical,
                beta: Tropical,
            ) -> strided_view::Result<()> {
                // Simple naive GEMM for Tropical type
                let a_ptr = a.ptr();
                let b_ptr = b.ptr();
                let c_ptr = c.ptr();
                let a_rs = a.row_stride();
                let a_cs = a.col_stride();
                let b_rs = b.row_stride();
                let b_cs = b.col_stride();
                let c_rs = c.row_stride();
                let c_cs = c.col_stride();

                let mut batch_idx = crate::util::MultiIndex::new(batch_dims);
                while batch_idx.next().is_some() {
                    let a_base = batch_idx.offset(a.batch_strides());
                    let b_base = batch_idx.offset(b.batch_strides());
                    let c_base = batch_idx.offset(c.batch_strides());

                    for i in 0..m {
                        for j in 0..n {
                            let mut acc = Tropical::zero();
                            for l in 0..k {
                                let a_val = unsafe {
                                    *a_ptr.offset(a_base + i as isize * a_rs + l as isize * a_cs)
                                };
                                let b_val = unsafe {
                                    *b_ptr.offset(b_base + l as isize * b_rs + j as isize * b_cs)
                                };
                                acc = acc + a_val * b_val;
                            }
                            unsafe {
                                let c_elem =
                                    c_ptr.offset(c_base + i as isize * c_rs + j as isize * c_cs);
                                if beta == Tropical::zero() {
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

        let a = StridedArray::from_parts(
            vec![Tropical(1.0), Tropical(2.0), Tropical(3.0), Tropical(4.0)],
            &[2, 2],
            &[2, 1],
            0,
        )
        .unwrap();
        let b = StridedArray::from_parts(
            vec![Tropical(5.0), Tropical(6.0), Tropical(7.0), Tropical(8.0)],
            &[2, 2],
            &[2, 1],
            0,
        )
        .unwrap();
        let mut c = StridedArray::<Tropical>::col_major(&[2, 2]);

        einsum2_with_backend_into::<_, TropicalBackend, _>(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['i', 'k'],
            &['i', 'j'],
            &['j', 'k'],
            Tropical(1.0),
            Tropical(0.0),
        )
        .unwrap();

        // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //   = [[19, 22], [43, 50]]
        assert_eq!(c.get(&[0, 0]), Tropical(19.0));
        assert_eq!(c.get(&[0, 1]), Tropical(22.0));
        assert_eq!(c.get(&[1, 0]), Tropical(43.0));
        assert_eq!(c.get(&[1, 1]), Tropical(50.0));
    }
}
