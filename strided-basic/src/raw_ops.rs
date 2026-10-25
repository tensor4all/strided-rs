//! Allocation-free copy/axpy over borrowed raw strided layouts.
//!
//! The map/zip kernels build a traversal plan per call; for small replay
//! copies that fixed cost dominates. These entry points take
//! [`RawStridedRef`]/[`RawStridedMut`], fuse the stride pair into a loop nest
//! (inline storage for fused ranks at most [`RAW_FUSED_RANK_LIMIT`], one
//! rank-bounded spill above that), and run plain loops - no view plan, no
//! cache, and no per-element allocation on any call path. Higher *original*
//! ranks are normalized and fused before that choice is made, so they no
//! longer leave the fused replay.

use crate::{ElementOpApply, RawStridedMut, RawStridedRef, Result};
use core::ops::{Add, Mul};

use crate::maybe_sync::MaybeSendSync;

/// Rank up to which a fused loop nest is stored inline.
///
/// The stateless [`copy_scale_raw`] family and [`crate::CopyPlan`] fuse any rank:
/// layouts whose *fused* rank is at most this limit replay from that inline
/// storage, and higher fused ranks spill to storage bounded by their rank.
///
/// Replay is allocation-free for an original rank at most this limit. A higher
/// *original* rank needs one transient scratch buffer even when the fused layout
/// still fits inline, because fusion happens before the representation is
/// chosen; the replay itself is still allocation-free in that case.
pub const RAW_FUSED_RANK_LIMIT: usize = 8;

/// One fused axis, used while sorting and merging axes and when a layout has to
/// spill beyond [`RAW_FUSED_RANK_LIMIT`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct FusedAxis {
    pub(crate) dim: usize,
    pub(crate) dst: isize,
    pub(crate) src: isize,
}

const EMPTY_AXIS: FusedAxis = FusedAxis {
    dim: 1,
    dst: 0,
    src: 0,
};

/// Stack-allocated fused stride pair (axes ordered by destination stride,
/// adjacent contiguous axes merged). Built once and replayed by both the
/// per-call raw kernels and the prepared [`crate::CopyPlan`].
///
/// This is the representation every call site replays through, so it keeps the
/// exact shape, size and `Copy`-ness it had before high-rank fusion existed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FusedPairLayout {
    pub(crate) rank: usize,
    pub(crate) dims: [usize; RAW_FUSED_RANK_LIMIT],
    dst_strides: [isize; RAW_FUSED_RANK_LIMIT],
    src_strides: [isize; RAW_FUSED_RANK_LIMIT],
}

impl FusedPairLayout {
    /// An empty rank-zero layout (one element at the base offset).
    ///
    /// Used as the neutral value when an inline `TinyVec<[CopyPlan; N]>` is
    /// default-initialized; it is never executed as a real plan.
    fn empty() -> Self {
        Self {
            rank: 0,
            dims: [1; RAW_FUSED_RANK_LIMIT],
            dst_strides: [0; RAW_FUSED_RANK_LIMIT],
            src_strides: [0; RAW_FUSED_RANK_LIMIT],
        }
    }

    fn from_axes(axes: &[FusedAxis]) -> Self {
        debug_assert!(axes.len() <= RAW_FUSED_RANK_LIMIT);
        let mut layout = Self {
            rank: axes.len(),
            dims: [1; RAW_FUSED_RANK_LIMIT],
            dst_strides: [0; RAW_FUSED_RANK_LIMIT],
            src_strides: [0; RAW_FUSED_RANK_LIMIT],
        };
        for (slot, axis) in layout.dims[..axes.len()].iter_mut().zip(axes) {
            *slot = axis.dim;
        }
        for (slot, axis) in layout.dst_strides[..axes.len()].iter_mut().zip(axes) {
            *slot = axis.dst;
        }
        for (slot, axis) in layout.src_strides[..axes.len()].iter_mut().zip(axes) {
            *slot = axis.src;
        }
        layout
    }
}

/// A fused pair traversal of any rank.
///
/// Fused ranks at most [`RAW_FUSED_RANK_LIMIT`] use the inline layout; longer
/// layouts keep their axes in one `Vec`. That storage is sized by the original
/// rank (the fusion scratch is reused, and `truncate` keeps its capacity), never
/// by the tensor's element count.
#[derive(Clone, Debug)]
pub(crate) enum PairTraversal {
    Inline(FusedPairLayout),
    Spilled(Vec<FusedAxis>),
}

impl Default for FusedPairLayout {
    fn default() -> Self {
        Self::empty()
    }
}

impl Default for PairTraversal {
    fn default() -> Self {
        Self::empty()
    }
}

impl PairTraversal {
    /// The empty inline layout, used as a neutral default.
    #[inline]
    fn empty() -> Self {
        Self::Inline(FusedPairLayout::empty())
    }

    /// The inline layout, or `None` when the traversal spilled.
    #[inline]
    pub(crate) fn inline(&self) -> Option<&FusedPairLayout> {
        match self {
            Self::Inline(layout) => Some(layout),
            Self::Spilled(_) => None,
        }
    }

    /// Number of fused axes.
    #[cfg(test)]
    pub(crate) fn rank(&self) -> usize {
        match self {
            Self::Inline(layout) => layout.rank,
            Self::Spilled(axes) => axes.len(),
        }
    }
}

/// Sort `dims`/`dst_strides`/`src_strides` by destination stride, drop size-1
/// axes, and merge axes that are contiguous in both layouts.
///
/// Returns the fused rank written to `scratch[..rank]`. A zero extent collapses
/// to a single zero-length axis, which every replay treats as an empty range.
fn fuse_axes(
    scratch: &mut [FusedAxis],
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
) -> usize {
    debug_assert!(scratch.len() >= dims.len());
    let mut rank = 0usize;
    for axis in 0..dims.len() {
        if dims[axis] == 1 {
            continue;
        }
        if dims[axis] == 0 {
            scratch[0] = FusedAxis {
                dim: 0,
                dst: 0,
                src: 0,
            };
            return 1;
        }
        let entry = FusedAxis {
            dim: dims[axis],
            dst: dst_strides[axis],
            src: src_strides[axis],
        };
        let mut position = rank;
        while position > 0 && scratch[position - 1].dst > entry.dst {
            scratch[position] = scratch[position - 1];
            position -= 1;
        }
        scratch[position] = entry;
        rank += 1;
    }
    if rank == 0 {
        scratch[0] = EMPTY_AXIS;
        return 1;
    }
    let mut fused = 0usize;
    for axis in 1..rank {
        // A merged extent that overflows (possible for a stride-0 broadcast
        // whose element count exceeds usize) leaves the axes unfused.
        let merged = isize::try_from(scratch[fused].dim)
            .ok()
            .filter(|&extent| {
                scratch[fused].dst.checked_mul(extent) == Some(scratch[axis].dst)
                    && scratch[fused].src.checked_mul(extent) == Some(scratch[axis].src)
            })
            .and_then(|_| scratch[fused].dim.checked_mul(scratch[axis].dim));
        if let Some(merged) = merged {
            scratch[fused].dim = merged;
        } else {
            fused += 1;
            scratch[fused] = scratch[axis];
        }
    }
    fused + 1
}

/// Fuse a raw stride pair into a traversal of any rank.
///
/// Inline when the fused rank is at most [`RAW_FUSED_RANK_LIMIT`], otherwise one
/// rank-bounded spill. An original rank above the limit uses a transient scratch
/// buffer even when the fused layout still fits inline, so that fusion happens
/// before the representation is chosen.
pub(crate) fn fuse_pair_traversal(
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
) -> PairTraversal {
    if dims.len() <= RAW_FUSED_RANK_LIMIT {
        let mut scratch = [EMPTY_AXIS; RAW_FUSED_RANK_LIMIT];
        let rank = fuse_axes(&mut scratch, dims, dst_strides, src_strides);
        debug_assert!(rank <= RAW_FUSED_RANK_LIMIT);
        return PairTraversal::Inline(FusedPairLayout::from_axes(&scratch[..rank]));
    }
    let mut scratch = vec![EMPTY_AXIS; dims.len()];
    let rank = fuse_axes(&mut scratch, dims, dst_strides, src_strides);
    scratch.truncate(rank);
    if rank <= RAW_FUSED_RANK_LIMIT {
        PairTraversal::Inline(FusedPairLayout::from_axes(&scratch))
    } else {
        PairTraversal::Spilled(scratch)
    }
}

/// Fuse a raw stride pair into the inline representation.
///
/// Returns `None` when the fused rank exceeds [`RAW_FUSED_RANK_LIMIT`]. Used by
/// the window plans, whose non-fused replay is compiled alongside the plan.
pub(crate) fn fuse_pair_layout(
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
) -> Option<FusedPairLayout> {
    match fuse_pair_traversal(dims, dst_strides, src_strides) {
        PairTraversal::Inline(layout) => Some(layout),
        PairTraversal::Spilled(_) => None,
    }
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

/// Number of logical elements covered by a spilled axis list.
#[inline]
fn spilled_total(axes: &[FusedAxis]) -> usize {
    axes.iter().map(|axis| axis.dim).product()
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

/// Serial replay of a spilled fused axis list over borrowed raw views.
///
/// Kept out of line and separate from [`apply_fused_pair`] so that the inline
/// entry points keep the code shape they had before high-rank fusion existed:
/// fused ranks above [`RAW_FUSED_RANK_LIMIT`] are the rare case, and inlining
/// this second carry loop into them measurably slowed the inline path (extra
/// register pressure in a loop that runs once per short run).
#[inline(never)]
pub(crate) fn apply_spilled_pair<D, S, Apply, Op>(
    dst: &mut RawStridedMut<'_, D>,
    src: &RawStridedRef<'_, S>,
    axes: &[FusedAxis],
    apply: Apply,
    op: Op,
) where
    D: Copy,
    S: Copy,
    Apply: Fn(&mut D, S),
    Op: Fn(S) -> S,
{
    let total = spilled_total(axes);
    if total == 0 {
        return;
    }
    let src_ptr = src.data().as_ptr();
    let src_base = src.offset();
    let dst_base = dst.offset();
    let dst_ptr = dst.data_mut().as_mut_ptr();
    // SAFETY: same contract as `apply_fused_pair`, for the axis list fused from
    // the same dims/strides.
    unsafe {
        apply_spilled_range(
            dst_ptr, dst_base, src_ptr, src_base, axes, 0, total, &apply, &op,
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

/// Replay logical indices `start..start + len` of a spilled fused axis list, in
/// column-major order of the axes (axis 0 fastest).
///
/// The carry loop is the one [`apply_fused_range`] uses, over interleaved axes
/// with its carry state in one rank-bounded `Vec`: no allocation scales with
/// the tensor size and no per-element coordinate decode happens.
///
/// # Safety
///
/// The [`apply_fused_range`] contract holds for `axes`.
///
/// Kept out of line: inlining this second carry loop into the inline replay
/// measurably slows that one down.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) unsafe fn apply_spilled_range<D, S, Apply, Op>(
    dst_ptr: *mut D,
    dst_base: isize,
    src_ptr: *const S,
    src_base: isize,
    axes: &[FusedAxis],
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
    let rank = axes.len();
    let inner_len = axes[0].dim;
    let inner_dst = axes[0].dst;
    let inner_src = axes[0].src;

    let mut index = vec![0usize; rank];
    let mut rest = start;
    let mut dst_outer = dst_base;
    let mut src_outer = src_base;
    for axis in 0..rank {
        let dim = axes[axis].dim;
        index[axis] = rest % dim;
        rest /= dim;
        if axis > 0 {
            dst_outer += index[axis] as isize * axes[axis].dst;
            src_outer += index[axis] as isize * axes[axis].src;
        }
    }
    let mut inner_start = index[0];
    let mut remaining = len;
    loop {
        let run = (inner_len - inner_start).min(remaining);
        let dst_run = dst_outer + inner_start as isize * inner_dst;
        let src_run = src_outer + inner_start as isize * inner_src;
        // SAFETY: the run covers logical indices inside the caller's range, so
        // by the caller's contract every slot touched below is in-bounds,
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
        // Same incremental carry as the inline replay: offsets only ever hold
        // reachable positions, and `remaining > 0` guarantees an axis advances.
        let mut axis = 1;
        while axis < rank {
            if index[axis] + 1 < axes[axis].dim {
                index[axis] += 1;
                dst_outer += axes[axis].dst;
                src_outer += axes[axis].src;
                break;
            }
            let last = (axes[axis].dim - 1) as isize;
            dst_outer -= last * axes[axis].dst;
            src_outer -= last * axes[axis].src;
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
    match fuse_pair_traversal(dest.dims(), dest.strides(), src.strides()) {
        PairTraversal::Inline(layout) => apply_fused_pair(
            dest,
            src,
            &layout,
            |dst, value| *dst = value,
            |value: T| scale * value,
        ),
        PairTraversal::Spilled(axes) => apply_spilled_pair(
            dest,
            src,
            &axes,
            |dst, value| *dst = value,
            |value: T| scale * value,
        ),
    }
    Ok(())
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
    match fuse_pair_traversal(dest.dims(), dest.strides(), src.strides()) {
        PairTraversal::Inline(layout) => apply_fused_pair(
            dest,
            src,
            &layout,
            |dst, value| *dst = value,
            |value: T| scale * value.conj(),
        ),
        PairTraversal::Spilled(axes) => apply_spilled_pair(
            dest,
            src,
            &axes,
            |dst, value| *dst = value,
            |value: T| scale * value.conj(),
        ),
    }
    Ok(())
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
    match fuse_pair_traversal(dest.dims(), dest.strides(), src.strides()) {
        PairTraversal::Inline(layout) => apply_fused_pair(
            dest,
            src,
            &layout,
            |dst, value| *dst = *dst + value,
            |value: T| alpha * value,
        ),
        PairTraversal::Spilled(axes) => apply_spilled_pair(
            dest,
            src,
            &axes,
            |dst, value| *dst = *dst + value,
            |value: T| alpha * value,
        ),
    }
    Ok(())
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
    match fuse_pair_traversal(dest.dims(), dest.strides(), src.strides()) {
        PairTraversal::Inline(layout) => apply_fused_pair(
            dest,
            src,
            &layout,
            |dst, value| *dst = *dst + value,
            |value: T| alpha * value.conj(),
        ),
        PairTraversal::Spilled(axes) => apply_spilled_pair(
            dest,
            src,
            &axes,
            |dst, value| *dst = *dst + value,
            |value: T| alpha * value.conj(),
        ),
    }
    Ok(())
}

#[cfg(test)]
#[path = "raw_ops/tests/tests.rs"]
mod tests;
