//! Allocation-free copy/axpy over borrowed raw strided layouts.
//!
//! [`StridedView`]/[`StridedViewMut`] own their metadata (`Arc<[usize]>` /
//! `Arc<[isize]>`) and the map/zip kernels build a traversal plan per call;
//! for small replay copies that fixed cost dominates. These entry points take
//! [`RawStridedRef`]/[`RawStridedMut`] (borrowed metadata), fuse the stride
//! pair into a stack-allocated loop nest, and run plain loops - no heap
//! allocation on any call path with rank at most [`RAW_FUSED_RANK_LIMIT`].
//! Higher ranks fall back to the view-based kernels.

use crate::ops_view::{axpy, copy_scale};
use crate::{ElementOpApply, RawStridedMut, RawStridedRef, Result};
use core::ops::{Add, Mul};

use crate::maybe_sync::MaybeSendSync;

/// Maximum rank fused on the stack before falling back to the view kernels.
pub const RAW_FUSED_RANK_LIMIT: usize = 8;

/// Stack-allocated fused stride pair (dims ordered by destination stride,
/// adjacent contiguous axes merged). Built once and replayed by both the
/// per-call raw kernels and the prepared [`crate::CopyPlan`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct FusedPairLayout {
    pub(crate) rank: usize,
    pub(crate) dims: [usize; RAW_FUSED_RANK_LIMIT],
    pub(crate) dst_strides: [isize; RAW_FUSED_RANK_LIMIT],
    pub(crate) src_strides: [isize; RAW_FUSED_RANK_LIMIT],
}

pub(crate) fn fuse_pair_layout(
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
) -> Option<FusedPairLayout> {
    if dims.len() > RAW_FUSED_RANK_LIMIT {
        return None;
    }
    let mut layout = FusedPairLayout {
        rank: 0,
        dims: [1; RAW_FUSED_RANK_LIMIT],
        dst_strides: [0; RAW_FUSED_RANK_LIMIT],
        src_strides: [0; RAW_FUSED_RANK_LIMIT],
    };
    for axis in 0..dims.len() {
        if dims[axis] == 1 {
            continue;
        }
        if dims[axis] == 0 {
            return Some(FusedPairLayout {
                rank: 1,
                dims: [0; RAW_FUSED_RANK_LIMIT],
                dst_strides: [0; RAW_FUSED_RANK_LIMIT],
                src_strides: [0; RAW_FUSED_RANK_LIMIT],
            });
        }
        let mut position = layout.rank;
        while position > 0 && layout.dst_strides[position - 1] > dst_strides[axis] {
            layout.dims[position] = layout.dims[position - 1];
            layout.dst_strides[position] = layout.dst_strides[position - 1];
            layout.src_strides[position] = layout.src_strides[position - 1];
            position -= 1;
        }
        layout.dims[position] = dims[axis];
        layout.dst_strides[position] = dst_strides[axis];
        layout.src_strides[position] = src_strides[axis];
        layout.rank += 1;
    }
    if layout.rank == 0 {
        layout.rank = 1;
        layout.dims[0] = 1;
    }
    let mut fused = 0usize;
    for axis in 1..layout.rank {
        // A merged extent that overflows (possible for a stride-0 broadcast
        // whose element count exceeds usize) leaves the axes unfused.
        let merged = isize::try_from(layout.dims[fused])
            .ok()
            .filter(|&extent| {
                layout.dst_strides[fused].checked_mul(extent) == Some(layout.dst_strides[axis])
                    && layout.src_strides[fused].checked_mul(extent)
                        == Some(layout.src_strides[axis])
            })
            .and_then(|_| layout.dims[fused].checked_mul(layout.dims[axis]));
        if let Some(merged) = merged {
            layout.dims[fused] = merged;
        } else {
            fused += 1;
            layout.dims[fused] = layout.dims[axis];
            layout.dst_strides[fused] = layout.dst_strides[axis];
            layout.src_strides[fused] = layout.src_strides[axis];
        }
    }
    layout.rank = fused + 1;
    Some(layout)
}

/// Number of logical elements covered by a fused layout.
///
/// Every constructor of a [`FusedPairLayout`] starts from dims whose product
/// was checked by [`crate::kernel::total_len`], and fusion only merges axes
/// with a checked product, so the product cannot overflow.
#[inline]
pub(crate) fn fused_total(layout: &FusedPairLayout) -> usize {
    layout.dims[..layout.rank].iter().product()
}

/// Serial replay of a fused pair layout over borrowed raw views.
///
/// The destination layout need not be injective: runs are replayed in order,
/// so accumulating callers such as [`axpy_raw`] keep their sequential
/// semantics. Parallel replay is only offered by [`crate::CopyPlan`], whose
/// compile step proves destination injectivity.
pub(crate) fn apply_fused_pair<D, S, Apply, Op>(
    dst: &mut RawStridedMut<'_, D>,
    src: &RawStridedRef<'_, S>,
    layout: &FusedPairLayout,
    apply: Apply,
    op: Op,
) where
    D: Copy,
    S: Copy,
    Apply: Fn(&mut D, S),
    Op: Fn(S) -> S,
{
    let total = fused_total(layout);
    if total == 0 {
        return;
    }
    let src_ptr = src.data().as_ptr();
    let src_base = src.offset();
    let dst_base = dst.offset();
    let dst_ptr = dst.data_mut().as_mut_ptr();
    // SAFETY: `RawStridedRef`/`RawStridedMut` guarantee (by `new`, or by the
    // caller of `new_unchecked`) that every offset reachable from their offset
    // through their dims/strides lies inside their data. Callers pass a layout
    // fused from exactly those dims/strides, so every logical index in
    // `0..total` maps to in-bounds source and destination slots. The shared
    // and exclusive borrows cannot overlap, and this is the only writer.
    unsafe {
        apply_fused_range(
            dst_ptr, dst_base, src_ptr, src_base, layout, 0, total, &apply, &op,
        );
    }
}

/// Replay logical indices `start..start + len` of a fused pair layout, in
/// column-major order of the fused axes (axis 0 fastest).
///
/// The start coordinate is decoded once; runs along axis 0 then advance the
/// outer coordinates incrementally, so no per-element range checks or
/// coordinate rebuilds remain in the hot loop.
///
/// # Safety
///
/// - `start + len` must not exceed [`fused_total`] of `layout`.
/// - For every logical index `i` in the range, `dst_base` plus the fused
///   destination offset of `i` must be an in-bounds, writable slot of the
///   allocation behind `dst_ptr`, and likewise `src_base` plus the source
///   offset must be an in-bounds, readable slot behind `src_ptr`.
/// - No other thread may access the destination slots of this range while it
///   runs, and the source slots must not be written concurrently. Destination
///   and source slots must not overlap.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn apply_fused_range<D, S, Apply, Op>(
    dst_ptr: *mut D,
    dst_base: isize,
    src_ptr: *const S,
    src_base: isize,
    layout: &FusedPairLayout,
    start: usize,
    len: usize,
    apply: &Apply,
    op: &Op,
) where
    D: Copy,
    S: Copy,
    Apply: Fn(&mut D, S),
    Op: Fn(S) -> S,
{
    if len == 0 {
        return;
    }
    let rank = layout.rank;
    let inner_len = layout.dims[0];
    let inner_dst = layout.dst_strides[0];
    let inner_src = layout.src_strides[0];

    // Decode the start coordinate once per range. `dst_offset`/`src_offset`
    // track the offset of the current run start (outer coordinates plus the
    // inner coordinate `inner_start`); every value is a reachable offset.
    let mut index = [0usize; RAW_FUSED_RANK_LIMIT];
    let mut rest = start;
    let mut dst_outer = dst_base;
    let mut src_outer = src_base;
    for axis in 0..rank {
        let dim = layout.dims[axis];
        index[axis] = rest % dim;
        rest /= dim;
        if axis > 0 {
            dst_outer += index[axis] as isize * layout.dst_strides[axis];
            src_outer += index[axis] as isize * layout.src_strides[axis];
        }
    }
    let mut inner_start = index[0];
    let mut remaining = len;
    loop {
        let run = (inner_len - inner_start).min(remaining);
        let dst_run = dst_outer + inner_start as isize * inner_dst;
        let src_run = src_outer + inner_start as isize * inner_src;
        // SAFETY: the run covers logical indices inside the caller's range, so
        // by this function's contract every slot touched below is in-bounds,
        // exclusively owned by this call (destination) and unaliased.
        unsafe {
            apply_run(
                dst_ptr, dst_run, inner_dst, src_ptr, src_run, inner_src, run, apply, op,
            )
        };
        remaining -= run;
        if remaining == 0 {
            return;
        }
        inner_start = 0;
        // Advance an axis only when another position along it follows, and
        // rewind it from its last position, so every intermediate base is a
        // reachable offset of a validated layout (issue #243 follow-up): a
        // layout ending within one stride of `isize::MAX` must not overflow.
        // `remaining > 0` guarantees a following run exists, so the loop
        // always finds an axis to advance before running out of rank.
        let mut axis = 1;
        while axis < rank {
            if index[axis] + 1 < layout.dims[axis] {
                index[axis] += 1;
                dst_outer += layout.dst_strides[axis];
                src_outer += layout.src_strides[axis];
                break;
            }
            let last = (layout.dims[axis] - 1) as isize;
            dst_outer -= last * layout.dst_strides[axis];
            src_outer -= last * layout.src_strides[axis];
            index[axis] = 0;
            axis += 1;
        }
    }
}

/// One run along the fused inner axis.
///
/// # Safety
///
/// Every `dst_offset + k * dst_stride` and `src_offset + k * src_stride` for
/// `k < len` must be an in-bounds slot as described in [`apply_fused_range`],
/// with exclusive access to the destination slots.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn apply_run<D, S, Apply, Op>(
    dst_ptr: *mut D,
    dst_offset: isize,
    dst_stride: isize,
    src_ptr: *const S,
    src_offset: isize,
    src_stride: isize,
    len: usize,
    apply: &Apply,
    op: &Op,
) where
    D: Copy,
    S: Copy,
    Apply: Fn(&mut D, S),
    Op: Fn(S) -> S,
{
    if dst_stride == 1 {
        // SAFETY: a unit destination stride makes the run `len` consecutive
        // in-bounds slots starting at `dst_offset`, exclusively owned here.
        let dst_run = unsafe { core::slice::from_raw_parts_mut(dst_ptr.offset(dst_offset), len) };
        match src_stride {
            1 => {
                // SAFETY: `len` consecutive in-bounds source slots.
                let src_run =
                    unsafe { core::slice::from_raw_parts(src_ptr.offset(src_offset), len) };
                for (dst, &value) in dst_run.iter_mut().zip(src_run) {
                    apply(dst, op(value));
                }
            }
            -1 => {
                // SAFETY: the run reads `src_offset - (len - 1) ..= src_offset`,
                // all in-bounds; the lowest one is the run's last source slot.
                let src_run = unsafe {
                    core::slice::from_raw_parts(
                        src_ptr.offset(src_offset - (len as isize - 1)),
                        len,
                    )
                };
                for (dst, &value) in dst_run.iter_mut().zip(src_run.iter().rev()) {
                    apply(dst, op(value));
                }
            }
            _ => {
                let src_start = unsafe { src_ptr.offset(src_offset) };
                for (position, dst) in dst_run.iter_mut().enumerate() {
                    // SAFETY: `position < len`, so this is a run source slot.
                    let value = unsafe { *src_start.offset(position as isize * src_stride) };
                    apply(dst, op(value));
                }
            }
        }
        return;
    }
    // SAFETY: the run start is an in-bounds slot of each allocation.
    let dst_start = unsafe { dst_ptr.offset(dst_offset) };
    let src_start = unsafe { src_ptr.offset(src_offset) };
    for position in 0..len as isize {
        // SAFETY: `position < len`, so both are slots of this run.
        unsafe {
            let value = *src_start.offset(position * src_stride);
            apply(&mut *dst_start.offset(position * dst_stride), op(value));
        }
    }
}

fn ensure_same_dims(dst: &[usize], src: &[usize]) -> Result<()> {
    if dst != src {
        return Err(crate::StridedError::ShapeMismatch(
            dst.to_vec(),
            src.to_vec(),
        ));
    }
    // Reject an element count beyond usize (a huge stride-0 broadcast)
    // instead of replaying it.
    crate::kernel::total_len(dst)?;
    Ok(())
}

/// `dest = scale * src` over borrowed raw strided layouts.
pub fn copy_scale_raw<T>(
    dest: &mut RawStridedMut<'_, T>,
    src: &RawStridedRef<'_, T>,
    scale: T,
) -> Result<()>
where
    T: Copy + Mul<T, Output = T> + ElementOpApply + MaybeSendSync,
{
    ensure_same_dims(dest.dims(), src.dims())?;
    match fuse_pair_layout(dest.dims(), dest.strides(), src.strides()) {
        Some(layout) => {
            apply_fused_pair(
                dest,
                src,
                &layout,
                |dst, value| *dst = value,
                |value: T| scale * value,
            );
            Ok(())
        }
        None => copy_scale(&mut dest.as_view_mut(), &src.as_view(), scale),
    }
}

/// `dest = scale * conj(src)` over borrowed raw strided layouts.
pub fn copy_scale_conj_raw<T>(
    dest: &mut RawStridedMut<'_, T>,
    src: &RawStridedRef<'_, T>,
    scale: T,
) -> Result<()>
where
    T: Copy + Mul<T, Output = T> + ElementOpApply + MaybeSendSync,
{
    ensure_same_dims(dest.dims(), src.dims())?;
    match fuse_pair_layout(dest.dims(), dest.strides(), src.strides()) {
        Some(layout) => {
            apply_fused_pair(
                dest,
                src,
                &layout,
                |dst, value| *dst = value,
                |value: T| scale * value.conj(),
            );
            Ok(())
        }
        None => copy_scale(&mut dest.as_view_mut(), &src.as_view().conj(), scale),
    }
}

/// `dest = alpha * src + dest` over borrowed raw strided layouts.
pub fn axpy_raw<T>(
    dest: &mut RawStridedMut<'_, T>,
    src: &RawStridedRef<'_, T>,
    alpha: T,
) -> Result<()>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + ElementOpApply + MaybeSendSync,
{
    ensure_same_dims(dest.dims(), src.dims())?;
    match fuse_pair_layout(dest.dims(), dest.strides(), src.strides()) {
        Some(layout) => {
            apply_fused_pair(
                dest,
                src,
                &layout,
                |dst, value| *dst = *dst + value,
                |value: T| alpha * value,
            );
            Ok(())
        }
        None => axpy(&mut dest.as_view_mut(), &src.as_view(), alpha),
    }
}

/// `dest = alpha * conj(src) + dest` over borrowed raw strided layouts.
pub fn axpy_conj_raw<T>(
    dest: &mut RawStridedMut<'_, T>,
    src: &RawStridedRef<'_, T>,
    alpha: T,
) -> Result<()>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + ElementOpApply + MaybeSendSync,
{
    ensure_same_dims(dest.dims(), src.dims())?;
    match fuse_pair_layout(dest.dims(), dest.strides(), src.strides()) {
        Some(layout) => {
            apply_fused_pair(
                dest,
                src,
                &layout,
                |dst, value| *dst = *dst + value,
                |value: T| alpha * value.conj(),
            );
            Ok(())
        }
        None => axpy(&mut dest.as_view_mut(), &src.as_view().conj(), alpha),
    }
}

#[cfg(test)]
#[path = "raw_ops/tests/tests.rs"]
mod tests;
