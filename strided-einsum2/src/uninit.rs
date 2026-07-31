//! Overwrite-only contraction entry points.
//!
//! This module deliberately does not reuse the initialized `beta` path.  The
//! destination is borrowed as `MaybeUninit<T>` until the last logical element
//! has been written, so a provider failure cannot expose a partially
//! initialized slice as `T`.

use std::collections::HashSet;
use std::mem::MaybeUninit;

use strided_kernel::ExecContext;
use strided_view::{ElementOp, RawStridedMut, RawStridedRef, StridedView};

use crate::{AxisId, Einsum2Plan, EinsumError, Result, ScalarBase};

/// Naive overwrite kernel used by the private backend contract and as the
/// no-provider implementation. It never reads C.
#[cfg(not(any(feature = "blas", feature = "blas-inject")))]
pub(crate) fn bgemm_contiguous_naive<T>(
    c: &mut crate::contiguous::UninitContiguousOperand<'_, '_, T>,
    a: &crate::contiguous::ContiguousOperand<T>,
    b: &crate::contiguous::ContiguousOperand<T>,
    batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    _ctx: &ExecContext,
) -> strided_view::Result<()>
where
    T: ScalarBase + strided_view::ElementOpApply,
{
    let mut batch = crate::util::MultiIndex::new(batch_dims);
    while batch.next().is_some() {
        let a_base = batch.offset(a.batch_strides());
        let b_base = batch.offset(b.batch_strides());
        let c_base = batch.offset(c.batch_strides());
        for i in 0..m {
            for j in 0..n {
                let mut acc = T::zero();
                for l in 0..k {
                    let mut av = unsafe {
                        *a.ptr().offset(
                            a_base + i as isize * a.row_stride() + l as isize * a.col_stride(),
                        )
                    };
                    let mut bv = unsafe {
                        *b.ptr().offset(
                            b_base + l as isize * b.row_stride() + j as isize * b.col_stride(),
                        )
                    };
                    if a.conj() {
                        av = strided_view::Conj::apply(av);
                    }
                    if b.conj() {
                        bv = strided_view::Conj::apply(bv);
                    }
                    acc = acc + av * bv;
                }
                let offset = c_base + i as isize * c.row_stride() + j as isize * c.col_stride();
                unsafe {
                    c.ptr().offset(offset).write(MaybeUninit::new(alpha * acc));
                }
            }
        }
    }
    Ok(())
}

#[cfg(any(feature = "blas", feature = "blas-inject"))]
fn zero_raw_uninit<T: ScalarBase>(dest: &mut RawStridedMut<'_, MaybeUninit<T>>) -> Result<()> {
    fn visit<T: ScalarBase>(
        dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
        dims: &[usize],
        strides: &[isize],
        axis: usize,
        offset: isize,
    ) -> Result<()> {
        if axis == dims.len() {
            let relative = offset
                .checked_sub(dest.offset())
                .ok_or(strided_view::StridedError::OffsetOverflow)?;
            unsafe {
                dest.as_mut_ptr()
                    .offset(relative)
                    .write(MaybeUninit::new(T::zero()));
            }
            return Ok(());
        }
        for i in 0..dims[axis] {
            let next = checked_offset(offset, i, strides[axis])?;
            visit(dest, dims, strides, axis + 1, next)?;
        }
        Ok(())
    }
    visit(dest, dest.dims(), dest.strides(), 0, dest.offset())
}

#[cfg(any(feature = "blas", feature = "blas-inject"))]
fn bgemm_raw_backend<T, B>(
    mut dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
    a: &RawStridedRef<'_, T>,
    b: &RawStridedRef<'_, T>,
    _n_batch: usize,
    n_lo: usize,
    n_ro: usize,
    n_sum: usize,
    alpha: T,
    ctx: &ExecContext,
) -> Result<()>
where
    T: ScalarBase + strided_view::ElementOpApply,
    B: crate::backend::Backend<T> + crate::backend::OverwriteBackend<T>,
{
    let (groups, m, k, n) = preflight_raw_bgemm(&mut dest, a, b, _n_batch, n_lo, n_ro, n_sum)?;
    if dest.dims().iter().any(|&d| d == 0) {
        return Ok(());
    }
    let sum_dims = &a.dims()[n_lo..groups.a_sum_end];
    let batch_dims = &a.dims()[groups.a_sum_end..];
    if sum_dims.iter().any(|&d| d == 0) {
        zero_raw_uninit(&mut dest)?;
        return Ok(());
    }
    let a_op = crate::contiguous::prepare_input_raw(
        a,
        n_lo,
        n_sum,
        false,
        B::REQUIRES_UNIT_STRIDE,
        true,
        None,
    )?;
    let b_op = crate::contiguous::prepare_input_raw(
        b,
        n_sum,
        n_ro,
        false,
        B::REQUIRES_UNIT_STRIDE,
        true,
        None,
    )?;
    let mut c_op = crate::contiguous::prepare_output_raw_uninit(
        &mut dest,
        n_lo,
        n_ro,
        B::REQUIRES_UNIT_STRIDE,
    )?;
    B::bgemm_contiguous_overwrite(&mut c_op, &a_op, &b_op, batch_dims, m, n, k, alpha, ctx)?;
    c_op.finalize()?;
    Ok(())
}

fn checked_offset(offset: isize, index: usize, stride: isize) -> Result<isize> {
    let term = (index as isize)
        .checked_mul(stride)
        .ok_or(strided_view::StridedError::OffsetOverflow)?;
    offset
        .checked_add(term)
        .ok_or(strided_view::StridedError::OffsetOverflow)
        .map_err(Into::into)
}

fn visit_offsets(
    dims: &[usize],
    strides: &[isize],
    axis: usize,
    offset: isize,
    seen: &mut HashSet<isize>,
) -> Result<()> {
    if axis == dims.len() {
        if !seen.insert(offset) {
            return Err(strided_view::StridedError::NonInjectiveOutputLayout.into());
        }
        return Ok(());
    }
    for index in 0..dims[axis] {
        visit_offsets(
            dims,
            strides,
            axis + 1,
            checked_offset(offset, index, strides[axis])?,
            seen,
        )?;
    }
    Ok(())
}

fn validate_output<T>(dest: &mut RawStridedMut<'_, MaybeUninit<T>>) -> Result<()> {
    let mut seen = HashSet::new();
    visit_offsets(dest.dims(), dest.strides(), 0, dest.offset(), &mut seen)
}

fn ranges_overlap<T, U>(a_ptr: *const T, a_len: usize, b_ptr: *const U, b_len: usize) -> bool {
    let a_start = a_ptr as usize;
    let b_start = b_ptr as usize;
    let a_bytes = a_len.saturating_mul(std::mem::size_of::<T>());
    let b_bytes = b_len.saturating_mul(std::mem::size_of::<U>());
    let a_end = a_start.saturating_add(a_bytes);
    let b_end = b_start.saturating_add(b_bytes);
    a_start < b_end && b_start < a_end
}

fn validate_no_overlap<T, OpA, OpB>(
    dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
    a: &StridedView<'_, T, OpA>,
    b: &StridedView<'_, T, OpB>,
) -> Result<()>
where
    T: Copy,
    OpA: ElementOp<T>,
    OpB: ElementOp<T>,
{
    let d = dest.data_mut();
    if ranges_overlap(d.as_ptr(), d.len(), a.data().as_ptr(), a.data().len())
        || ranges_overlap(d.as_ptr(), d.len(), b.data().as_ptr(), b.data().len())
    {
        return Err(strided_view::StridedError::OverlappingInputOutput { input: 0 }.into());
    }
    Ok(())
}

/// Complete raw GEMM preflight, before labels, temporaries, or provider work.
fn preflight_raw_bgemm<T: Copy>(
    dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
    a: &RawStridedRef<'_, T>,
    b: &RawStridedRef<'_, T>,
    n_batch: usize,
    n_lo: usize,
    n_ro: usize,
    n_sum: usize,
) -> Result<(crate::raw_bgemm::BgemmGroupLayout, usize, usize, usize)> {
    let groups = crate::raw_bgemm::checked_bgemm_group_layout(n_batch, n_lo, n_ro, n_sum)?;
    crate::raw_bgemm::validate_bgemm_shapes(dest, a, b, n_batch, n_lo, n_ro, n_sum)?;
    let av: StridedView<'_, T> =
        unsafe { StridedView::new_unchecked(a.data(), a.dims(), a.strides(), a.offset()) };
    let bv: StridedView<'_, T> =
        unsafe { StridedView::new_unchecked(b.data(), b.dims(), b.strides(), b.offset()) };
    validate_output(dest)?;
    validate_no_overlap(dest, &av, &bv)?;
    let m = a.dims()[..n_lo]
        .iter()
        .try_fold(1usize, |v, &d| v.checked_mul(d))
        .ok_or(strided_view::StridedError::OffsetOverflow)?
        .max(1);
    let k = a.dims()[n_lo..groups.a_sum_end]
        .iter()
        .try_fold(1usize, |v, &d| v.checked_mul(d))
        .ok_or(strided_view::StridedError::OffsetOverflow)?
        .max(1);
    let n = b.dims()[n_sum..groups.b_ro_end]
        .iter()
        .try_fold(1usize, |v, &d| v.checked_mul(d))
        .ok_or(strided_view::StridedError::OffsetOverflow)?
        .max(1);
    #[cfg(any(feature = "blas", feature = "blas-inject"))]
    for value in [m, k, n] {
        i32::try_from(value).map_err(|_| strided_view::StridedError::OffsetOverflow)?;
    }
    Ok((groups, m, k, n))
}

fn validate_labels<T, OpA, OpB, ID>(
    plan: &Einsum2Plan<ID>,
    dest: &RawStridedMut<'_, MaybeUninit<T>>,
    a: &StridedView<'_, T, OpA>,
    b: &StridedView<'_, T, OpB>,
    ic: &[ID],
    ia: &[ID],
    ib: &[ID],
) -> Result<()>
where
    T: Copy,
    OpA: ElementOp<T>,
    OpB: ElementOp<T>,
    ID: AxisId,
{
    if ia.len() != a.dims().len() || ib.len() != b.dims().len() || ic.len() != dest.dims().len() {
        return Err(EinsumError::OutputShapeMismatch {
            expected: vec![ic.len()],
            got: vec![dest.dims().len()],
        });
    }
    let dim = |labels: &[ID], dims: &[usize], id: &ID| {
        labels.iter().position(|x| x == id).map(|i| dims[i])
    };
    for (axis, id) in ic.iter().enumerate() {
        let expected = dim(ia, a.dims(), id).or_else(|| dim(ib, b.dims(), id));
        if expected != Some(dest.dims()[axis]) {
            return Err(EinsumError::OutputShapeMismatch {
                expected: ic
                    .iter()
                    .map(|x| {
                        dim(ia, a.dims(), x)
                            .or_else(|| dim(ib, b.dims(), x))
                            .unwrap_or(0)
                    })
                    .collect(),
                got: dest.dims().to_vec(),
            });
        }
    }
    for id in plan.batch.iter().chain(plan.sum.iter()) {
        let da = dim(ia, a.dims(), id).ok_or_else(|| {
            EinsumError::InvalidDotGeneralConfig(format!(
                "planned axis {:?} is absent from lhs",
                id
            ))
        })?;
        let db = dim(ib, b.dims(), id).ok_or_else(|| {
            EinsumError::InvalidDotGeneralConfig(format!(
                "planned axis {:?} is absent from rhs",
                id
            ))
        })?;
        if da != db {
            return Err(EinsumError::DimensionMismatch {
                axis: format!("{:?}", id),
                dim_a: da,
                dim_b: db,
            });
        }
    }
    Ok(())
}

#[cfg(all(
    not(any(feature = "blas", feature = "blas-inject")),
    not(feature = "faer")
))]
fn visit_sum<T, OpA, OpB, ID>(
    sum_ids: &[ID],
    axis: usize,
    a_idx: &mut [usize],
    b_idx: &mut [usize],
    ia: &[ID],
    ib: &[ID],
    a: &StridedView<'_, T, OpA>,
    b: &StridedView<'_, T, OpB>,
    acc: &mut T,
) where
    T: ScalarBase,
    OpA: ElementOp<T>,
    OpB: ElementOp<T>,
    ID: AxisId,
{
    if axis == sum_ids.len() {
        *acc = *acc + a.get(a_idx) * b.get(b_idx);
        return;
    }
    let id = &sum_ids[axis];
    let ai = ia.iter().position(|x| x == id);
    let bi = ib.iter().position(|x| x == id);
    let dim = ai
        .map(|i| a.dims()[i])
        .or_else(|| bi.map(|i| b.dims()[i]))
        .unwrap_or(0);
    for i in 0..dim {
        if let Some(ai) = ai {
            a_idx[ai] = i;
        }
        if let Some(bi) = bi {
            b_idx[bi] = i;
        }
        visit_sum(sum_ids, axis + 1, a_idx, b_idx, ia, ib, a, b, acc);
    }
}

#[cfg(all(
    not(any(feature = "blas", feature = "blas-inject")),
    not(feature = "faer")
))]
fn visit_output<T, OpA, OpB, ID>(
    axis: usize,
    out_idx: &mut [usize],
    dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
    a_idx: &mut [usize],
    b_idx: &mut [usize],
    ic: &[ID],
    ia: &[ID],
    ib: &[ID],
    reduction_ids: &[ID],
    a: &StridedView<'_, T, OpA>,
    b: &StridedView<'_, T, OpB>,
    alpha: T,
) -> Result<()>
where
    T: ScalarBase,
    OpA: ElementOp<T>,
    OpB: ElementOp<T>,
    ID: AxisId,
{
    if axis == out_idx.len() {
        for (pos, id) in ic.iter().enumerate() {
            if let Some(ai) = ia.iter().position(|x| x == id) {
                a_idx[ai] = out_idx[pos];
            }
            if let Some(bi) = ib.iter().position(|x| x == id) {
                b_idx[bi] = out_idx[pos];
            }
        }
        let mut value = T::zero();
        visit_sum(reduction_ids, 0, a_idx, b_idx, ia, ib, a, b, &mut value);
        let mut offset = dest.offset();
        for (&idx, &stride) in out_idx.iter().zip(dest.strides()) {
            offset = checked_offset(offset, idx, stride)?;
        }
        let relative = offset
            .checked_sub(dest.offset())
            .ok_or(strided_view::StridedError::OffsetOverflow)?;
        unsafe {
            dest.as_mut_ptr()
                .offset(relative)
                .write(MaybeUninit::new(alpha * value))
        };
        return Ok(());
    }
    for i in 0..dest.dims()[axis] {
        out_idx[axis] = i;
        visit_output(
            axis + 1,
            out_idx,
            dest,
            a_idx,
            b_idx,
            ic,
            ia,
            ib,
            reduction_ids,
            a,
            b,
            alpha,
        )?;
    }
    Ok(())
}

/// Compute an einsum into a genuinely uninitialized destination.
#[allow(clippy::too_many_arguments)]
#[cfg(not(any(feature = "blas", feature = "blas-inject")))]
pub fn einsum2_into_uninit<T, OpA, OpB, ID>(
    dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
    a: &StridedView<'_, T, OpA>,
    b: &StridedView<'_, T, OpB>,
    ic: &[ID],
    ia: &[ID],
    ib: &[ID],
    alpha: T,
    _ctx: &ExecContext,
) -> Result<()>
where
    T: ScalarBase,
    OpA: ElementOp<T>,
    OpB: ElementOp<T>,
    ID: AxisId,
{
    let plan = Einsum2Plan::new(ia, ib, ic)?;
    validate_labels(&plan, dest, a, b, ic, ia, ib)?;
    validate_output(dest)?;
    validate_no_overlap(dest, a, b)?;
    #[cfg(feature = "faer")]
    {
        let _ = alpha;
        return Err(EinsumError::Unsupported(
            "Faer does not yet expose a MaybeUninit-safe overwrite GEMM API; see strided-rs#195"
                .to_owned(),
        ));
    }
    #[cfg(not(feature = "faer"))]
    {
        if dest.dims().iter().any(|&d| d == 0) {
            return Ok(());
        }
        let mut out_idx = vec![0; dest.dims().len()];
        let mut a_idx = vec![0; a.dims().len()];
        let mut b_idx = vec![0; b.dims().len()];
        let mut reduction_ids = plan.sum.clone();
        for id in ia {
            if !ic.contains(id) && !reduction_ids.contains(id) {
                reduction_ids.push(id.clone());
            }
        }
        for id in ib {
            if !ic.contains(id) && !reduction_ids.contains(id) {
                reduction_ids.push(id.clone());
            }
        }
        visit_output(
            0,
            &mut out_idx,
            dest,
            &mut a_idx,
            &mut b_idx,
            ic,
            ia,
            ib,
            &reduction_ids,
            a,
            b,
            alpha,
        )?;
        Ok(())
    }
}

/// BLAS-backed overwrite path. All public validation happens before the
/// canonical descriptors are prepared or a temporary is allocated.
#[allow(clippy::too_many_arguments)]
#[cfg(any(feature = "blas", feature = "blas-inject"))]
pub fn einsum2_into_uninit<T, OpA, OpB, ID>(
    dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
    a: &StridedView<'_, T, OpA>,
    b: &StridedView<'_, T, OpB>,
    ic: &[ID],
    ia: &[ID],
    ib: &[ID],
    alpha: T,
    ctx: &ExecContext,
) -> Result<()>
where
    T: crate::Scalar,
    OpA: ElementOp<T> + 'static,
    OpB: ElementOp<T> + 'static,
    ID: AxisId,
{
    let plan = Einsum2Plan::new(ia, ib, ic)?;
    validate_labels(&plan, dest, a, b, ic, ia, ib)?;
    validate_output(dest)?;
    validate_no_overlap(dest, a, b)?;
    if dest.dims().iter().any(|&d| d == 0) {
        return Ok(());
    }

    let left_trace = crate::trace::find_trace_indices(ia, ib, ic);
    let (a_buf, conj_a) = if !left_trace.is_empty() {
        (
            Some(crate::trace::reduce_trace_axes(a, &left_trace)?),
            false,
        )
    } else {
        (None, crate::op_is_conj::<OpA>())
    };
    let a_view: StridedView<'_, T> = match a_buf.as_ref() {
        Some(buf) => buf.view(),
        None => StridedView::new(a.data(), a.dims(), a.strides(), a.offset())?,
    };
    let right_trace = crate::trace::find_trace_indices(ib, ia, ic);
    let (b_buf, conj_b) = if !right_trace.is_empty() {
        (
            Some(crate::trace::reduce_trace_axes(b, &right_trace)?),
            false,
        )
    } else {
        (None, crate::op_is_conj::<OpB>())
    };
    let b_view: StridedView<'_, T> = match b_buf.as_ref() {
        Some(buf) => buf.view(),
        None => StridedView::new(b.data(), b.dims(), b.strides(), b.offset())?,
    };
    let a_perm = a_view.permute(&plan.left_perm)?;
    let b_perm = b_view.permute(&plan.right_perm)?;
    let c_dims: Vec<usize> = plan
        .c_to_internal_perm
        .iter()
        .map(|&axis| dest.dims()[axis])
        .collect();
    let c_strides: Vec<isize> = plan
        .c_to_internal_perm
        .iter()
        .map(|&axis| dest.strides()[axis])
        .collect();
    let dest_offset = dest.offset();
    let mut c_perm = RawStridedMut::new(dest.data_mut(), &c_dims, &c_strides, dest_offset)?;
    let a_raw = RawStridedRef::new(
        a_perm.data(),
        a_perm.dims(),
        a_perm.strides(),
        a_perm.offset(),
    )?;
    let b_raw = RawStridedRef::new(
        b_perm.data(),
        b_perm.dims(),
        b_perm.strides(),
        b_perm.offset(),
    )?;
    let materialize = crate::make_conj_fn::<T>();
    // BLAS has no conjugation flag. Materialize conjugation before preparing
    // the raw backend operands, while retaining the same preflight contract.
    if conj_a || conj_b {
        let av = if conj_a {
            let mut mapped =
                unsafe { strided_view::StridedArray::<T>::col_major_uninit(a_perm.dims()) };
            strided_kernel::map_into(&mut mapped.view_mut(), &a_perm, materialize.unwrap())?;
            mapped
        } else {
            strided_view::StridedArray::from_parts(
                a_perm.data().to_vec(),
                a_perm.dims(),
                a_perm.strides(),
                a_perm.offset(),
            )?
        };
        let bv = if conj_b {
            let mut mapped =
                unsafe { strided_view::StridedArray::<T>::col_major_uninit(b_perm.dims()) };
            strided_kernel::map_into(&mut mapped.view_mut(), &b_perm, materialize.unwrap())?;
            mapped
        } else {
            strided_view::StridedArray::from_parts(
                b_perm.data().to_vec(),
                b_perm.dims(),
                b_perm.strides(),
                b_perm.offset(),
            )?
        };
        let ar = RawStridedRef::new(av.data(), av.dims(), av.strides(), av.view().offset())?;
        let br = RawStridedRef::new(bv.data(), bv.dims(), bv.strides(), bv.view().offset())?;
        return bgemm_raw_backend::<T, crate::backend::ActiveBackend>(
            &mut c_perm,
            &ar,
            &br,
            plan.batch.len(),
            plan.lo.len(),
            plan.ro.len(),
            plan.sum.len(),
            alpha,
            ctx,
        );
    }
    bgemm_raw_backend::<T, crate::backend::ActiveBackend>(
        &mut c_perm,
        &a_raw,
        &b_raw,
        plan.batch.len(),
        plan.lo.len(),
        plan.ro.len(),
        plan.sum.len(),
        alpha,
        ctx,
    )
}

/// Owned-input variant of [`einsum2_into_uninit`].
#[allow(clippy::too_many_arguments)]
#[cfg(not(any(feature = "blas", feature = "blas-inject")))]
pub fn einsum2_into_owned_uninit<T, ID>(
    dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
    a: strided_view::StridedArray<T>,
    b: strided_view::StridedArray<T>,
    ic: &[ID],
    ia: &[ID],
    ib: &[ID],
    alpha: T,
    ctx: &ExecContext,
) -> Result<()>
where
    T: ScalarBase + strided_view::ElementOpApply,
    ID: AxisId,
{
    einsum2_into_uninit(dest, &a.view(), &b.view(), ic, ia, ib, alpha, ctx)
}

/// Owned-input variant for BLAS-backed overwrite execution.
#[allow(clippy::too_many_arguments)]
#[cfg(any(feature = "blas", feature = "blas-inject"))]
pub fn einsum2_into_owned_uninit<T, ID>(
    dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
    a: strided_view::StridedArray<T>,
    b: strided_view::StridedArray<T>,
    ic: &[ID],
    ia: &[ID],
    ib: &[ID],
    alpha: T,
    ctx: &ExecContext,
) -> Result<()>
where
    T: crate::Scalar,
    ID: AxisId,
{
    einsum2_into_uninit(dest, &a.view(), &b.view(), ic, ia, ib, alpha, ctx)
}

/// Canonical raw overwrite-only GEMM entry point.
#[allow(clippy::too_many_arguments)]
pub fn bgemm_raw_strided_into_uninit<T>(
    dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
    a: &RawStridedRef<'_, T>,
    b: &RawStridedRef<'_, T>,
    n_batch: usize,
    n_lo: usize,
    n_ro: usize,
    n_sum: usize,
    alpha: T,
    ctx: &ExecContext,
) -> Result<()>
where
    T: crate::Scalar,
{
    // This must precede label construction and all backend-specific
    // materialization/allocation. It also validates shape agreement, output
    // injectivity, conservative aliasing, checked products, and BLAS sizes.
    let (groups, _, _, _) = preflight_raw_bgemm(dest, a, b, n_batch, n_lo, n_ro, n_sum)?;
    let mut labels = Vec::with_capacity(groups.label_len);
    labels.extend((0..groups.c_rank).map(|x| x));
    #[cfg(not(any(feature = "blas", feature = "blas-inject")))]
    let ic = labels[..groups.c_rank].to_vec();
    #[cfg(not(any(feature = "blas", feature = "blas-inject")))]
    let sum_start = groups.c_rank;
    #[cfg(not(any(feature = "blas", feature = "blas-inject")))]
    let ia = (0..n_lo)
        .chain(sum_start..groups.label_len)
        .chain(groups.c_ro_end..groups.c_rank)
        .collect::<Vec<_>>();
    #[cfg(not(any(feature = "blas", feature = "blas-inject")))]
    let ib = (sum_start..groups.label_len)
        .chain(n_lo..groups.c_ro_end)
        .chain(groups.c_ro_end..groups.c_rank)
        .collect::<Vec<_>>();
    #[cfg(not(any(feature = "blas", feature = "blas-inject")))]
    let av: StridedView<'_, T> =
        unsafe { StridedView::new_unchecked(a.data(), a.dims(), a.strides(), a.offset()) };
    #[cfg(not(any(feature = "blas", feature = "blas-inject")))]
    let bv: StridedView<'_, T> =
        unsafe { StridedView::new_unchecked(b.data(), b.dims(), b.strides(), b.offset()) };
    #[cfg(any(feature = "blas", feature = "blas-inject"))]
    {
        return bgemm_raw_backend::<T, crate::backend::ActiveBackend>(
            dest, a, b, n_batch, n_lo, n_ro, n_sum, alpha, ctx,
        );
    }
    #[cfg(not(any(feature = "blas", feature = "blas-inject")))]
    einsum2_into_uninit(dest, &av, &bv, &ic, &ia, &ib, alpha, ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use strided_view::StridedArray;

    #[cfg(not(feature = "faer"))]
    #[test]
    fn matrix_product_writes_uninitialized_destination() {
        let a = StridedArray::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        let b = StridedArray::from_fn_row_major(&[3, 2], |idx| (idx[0] * 2 + idx[1] + 1) as f64);
        let mut storage = vec![MaybeUninit::<f64>::uninit(); 4];
        let dims = [2, 2];
        let strides = [2, 1];
        let mut c = RawStridedMut::new(&mut storage, &dims, &strides, 0).unwrap();
        einsum2_into_uninit(
            &mut c,
            &a.view(),
            &b.view(),
            &['i', 'k'],
            &['i', 'j'],
            &['j', 'k'],
            1.0,
            &ExecContext::serial(),
        )
        .unwrap();
        let values: Vec<f64> = storage
            .into_iter()
            .map(|x| unsafe { x.assume_init() })
            .collect();
        assert_eq!(values, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn rejects_noninjective_destination_before_writing() {
        let a = StridedArray::from_fn_col_major(&[2], |_| 1.0f64);
        let b = StridedArray::from_fn_col_major(&[2], |_| 2.0f64);
        let mut storage = vec![MaybeUninit::<f64>::uninit(); 1];
        let dims = [2];
        let strides = [0];
        let mut c = RawStridedMut::new(&mut storage, &dims, &strides, 0).unwrap();
        let err = einsum2_into_uninit(
            &mut c,
            &a.view(),
            &b.view(),
            &['i'],
            &['i'],
            &['i'],
            1.0,
            &ExecContext::serial(),
        )
        .unwrap_err();
        assert!(matches!(err, crate::EinsumError::Strided(_)));
    }

    #[cfg(feature = "faer")]
    #[test]
    fn faer_uninit_gemm_reports_typed_unsupported_error() {
        let a = StridedArray::from_fn_col_major(&[1, 1], |_| 1.0f64);
        let b = StridedArray::from_fn_col_major(&[1, 1], |_| 1.0f64);
        let mut storage = vec![MaybeUninit::<f64>::uninit()];
        let dims = [1usize, 1];
        let strides = [1isize, 1];
        let mut c = RawStridedMut::new(&mut storage, &dims, &strides, 0).unwrap();
        let err = einsum2_into_uninit(
            &mut c,
            &a.view(),
            &b.view(),
            &['i', 'k'],
            &['i', 'j'],
            &['j', 'k'],
            1.0,
            &ExecContext::serial(),
        )
        .unwrap_err();
        assert!(matches!(err, crate::EinsumError::Unsupported(_)));
    }

    #[cfg(not(feature = "faer"))]
    #[test]
    fn noncontiguous_output_is_written_back_after_overwrite() {
        let a = StridedArray::from_fn_col_major(&[2, 2, 2], |idx| {
            (1 + idx[0] + 2 * idx[1] + 4 * idx[2]) as f64
        });
        let b = StridedArray::from_fn_col_major(&[2, 2], |idx| (1 + idx[0] + 2 * idx[1]) as f64);
        let mut storage = vec![MaybeUninit::<f64>::uninit(); 8];
        let dims = [2usize, 2, 2];
        let strides = [1isize, 4, 2];
        let mut c = RawStridedMut::new(&mut storage, &dims, &strides, 0).unwrap();
        einsum2_into_uninit(
            &mut c,
            &a.view(),
            &b.view(),
            &['i', 'j', 'k'],
            &['i', 'j', 'l'],
            &['l', 'k'],
            1.0,
            &ExecContext::serial(),
        )
        .unwrap();
        let values: Vec<f64> = storage
            .into_iter()
            .map(|x| unsafe { x.assume_init() })
            .collect();
        assert_eq!(values, vec![11.0, 14.0, 23.0, 30.0, 17.0, 20.0, 37.0, 44.0]);
    }

    #[test]
    fn raw_uninit_gemm_rejects_wrapping_group_partition_without_panicking() {
        let a = StridedArray::from_fn_col_major(&[1], |_| 1.0f64);
        let b = StridedArray::from_fn_col_major(&[1, 1], |_| 1.0f64);
        let mut storage = vec![MaybeUninit::<f64>::uninit()];
        let c_dims: [usize; 0] = [];
        let c_strides: [isize; 0] = [];
        let mut c = RawStridedMut::new(&mut storage, &c_dims, &c_strides, 0).unwrap();
        let a_raw = RawStridedRef::new(a.data(), a.dims(), a.strides(), a.view().offset()).unwrap();
        let b_raw = RawStridedRef::new(b.data(), b.dims(), b.strides(), b.view().offset()).unwrap();

        let result = catch_unwind(AssertUnwindSafe(|| {
            bgemm_raw_strided_into_uninit(
                &mut c,
                &a_raw,
                &b_raw,
                1,
                usize::MAX,
                0,
                1,
                1.0,
                &ExecContext::serial(),
            )
        }));

        assert!(result.is_ok(), "invalid group partition must not panic");
        assert!(result.unwrap().is_err());
    }
}
