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

use crate::cost::KernelClass;
use crate::{ElementOpApply, Exec, RawStridedMut, RawStridedRef, Result};
use core::mem::MaybeUninit;
use core::num::NonZeroUsize;
use core::ops::{Add, Mul};

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
    pub(crate) dst_strides: [isize; RAW_FUSED_RANK_LIMIT],
    pub(crate) src_strides: [isize; RAW_FUSED_RANK_LIMIT],
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

    /// The cache tiles of this traversal, or `None` when the column-major walk
    /// is kept.
    ///
    /// The one decision entry point for both storage shapes, so a plan and a
    /// replay entry ask the same question whatever the fused rank.
    pub(crate) fn tiles<D, S>(&self) -> Option<Tiles> {
        match self {
            Self::Inline(layout) => pair_tiles::<D, S>(layout).map(Tiles::Inline),
            Self::Spilled(axes) => spilled_pair_tiles::<D, S>(axes),
        }
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

impl PairTraversal {
    /// The given axes as a traversal, in their given order: nothing is
    /// sorted, merged or dropped. Rank zero becomes the one-element
    /// `EMPTY_AXIS` layout that [`fuse_axes`] also produces.
    ///
    /// The three slices must have equal length, and the extents a checked,
    /// nonzero product (the replay totals multiply them unchecked).
    pub(crate) fn given(dims: &[usize], dst_strides: &[isize], src_strides: &[isize]) -> Self {
        debug_assert!(dims.len() == dst_strides.len() && dims.len() == src_strides.len());
        let rank = dims.len();
        if rank > RAW_FUSED_RANK_LIMIT {
            return Self::Spilled(
                dims.iter()
                    .zip(dst_strides)
                    .zip(src_strides)
                    .map(|((&dim, &dst), &src)| FusedAxis { dim, dst, src })
                    .collect(),
            );
        }
        let mut layout = FusedPairLayout::empty();
        layout.rank = rank.max(1);
        layout.dims[..rank].copy_from_slice(dims);
        layout.dst_strides[..rank].copy_from_slice(dst_strides);
        layout.src_strides[..rank].copy_from_slice(src_strides);
        Self::Inline(layout)
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

/// The cache tiles chosen for a traversal whose tiled walk visits the slots in
/// a different order from the column-major walk.
///
/// Built by [`PairTraversal::tiles`]: [`Tiles::Inline`] for a compressed rank
/// at most [`RAW_FUSED_RANK_LIMIT`], [`Tiles::Spilled`] above it.
#[derive(Clone, Debug)]
pub(crate) enum Tiles {
    Inline(PairTiles),
    Spilled(SpilledTiles),
}

impl Tiles {
    /// The tiles of the inline walk, or `None` for a spilled traversal (whose
    /// compressed layout, when it fits inline, is replayed through that same
    /// walk).
    pub(crate) fn inline(&self) -> Option<&PairTiles> {
        match self {
            Self::Inline(tiles) => Some(tiles),
            Self::Spilled(_) => None,
        }
    }
}

/// The cache tiles of a spilled axis list whose compressed rank exceeds
/// [`RAW_FUSED_RANK_LIMIT`].
///
/// `dims`/`dst`/`src` are the compressed axes (extent-1 axes dropped, order
/// kept) and address exactly the slots of the axis list they were built from,
/// in the same column-major order. They are rank-sized slices rather than one
/// array, because the tile walk and the dimension splitter both take slices.
#[derive(Clone, Debug)]
pub(crate) struct SpilledTiles {
    pub(crate) dims: Vec<usize>,
    pub(crate) dst: Vec<isize>,
    pub(crate) src: Vec<isize>,
    pub(crate) blocks: Vec<usize>,
}

/// The cache tiles of a fused pair layout whose tiled walk visits the slots in
/// a different order from the column-major walk.
///
/// Built by [`pair_tiles`]; `layout` addresses exactly the slots of the layout
/// it was built from, in the same column-major order, without its extent-1
/// axes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PairTiles {
    pub(crate) layout: FusedPairLayout,
    pub(crate) blocks: [usize; RAW_FUSED_RANK_LIMIT],
}

/// Cache tiles for replaying `layout` with destination element `D` and source
/// element `S`, or `None` when the column-major walk is kept.
///
/// The extent-1 axes are dropped first: they address offset 0 whatever their
/// stride, so the compressed layout addresses the same slots in the same
/// order, and an unvalidated stride on such an axis never reaches the block
/// arithmetic. Fewer than two remaining axes form one run, whose order no
/// block vector can change. The blocks come from the shared model
/// (`block::compute_block_sizes`), which sizes each operand by its own element
/// size exactly as the view path does. Tiles are returned only when they
/// change the visit order (the first axis whose block is smaller than its
/// extent is followed by an axis with a block above 1) and when that change can
/// reuse a cache line (see [`slabs_share_no_line`]).
///
/// The extents of `layout` must have a product that fits `usize` (every
/// constructor of a [`FusedPairLayout`] checks it).
pub(crate) fn pair_tiles<D, S>(layout: &FusedPairLayout) -> Option<PairTiles> {
    if fused_total(layout) <= 1 {
        return None;
    }
    let moving = layout.dims[..layout.rank]
        .iter()
        .filter(|&&dim| dim != 1)
        .count();
    if moving < 2 {
        return None;
    }
    let compressed;
    let layout = if moving == layout.rank {
        layout
    } else {
        compressed = without_unit_axes(layout);
        &compressed
    };
    let rank = layout.rank;
    let model = tile_blocks::<D, S>(
        &layout.dims[..rank],
        &layout.dst_strides[..rank],
        &layout.src_strides[..rank],
    )?;
    let mut blocks = [1; RAW_FUSED_RANK_LIMIT];
    blocks[..rank].copy_from_slice(&model);
    Some(PairTiles {
        layout: *layout,
        blocks,
    })
}

/// The block vector of the shared cache-block model for already-compressed
/// axes (`dims`/`dst`/`src` of one length, every extent at least 2), or `None`
/// when the column-major walk is kept.
///
/// The single source of the tiling decision for both storage shapes: the
/// inline [`FusedPairLayout`] and the spilled [`SpilledTiles`].
fn tile_blocks<D, S>(
    dims: &[usize],
    dst: &[isize],
    src: &[isize],
) -> Option<crate::inline::InlineVec<usize>> {
    // `None`: one of the model's exits keeps the full dimensions, which
    // cannot change the visit order.
    let model = crate::block::reduced_block_sizes(
        dims,
        &[dst, src],
        &[core::mem::size_of::<D>(), core::mem::size_of::<S>()],
    )?;
    let first_cut = (0..dims.len()).find(|&axis| model[axis] < dims[axis])?;
    if model[first_cut + 1..].iter().all(|&block| block == 1) {
        return None;
    }
    // The axes before `first_cut` are walked whole inside every tile, so each
    // operand is judged with its own element size, as the model now does.
    if first_cut > 0
        && slabs_share_no_line(dims, dst, first_cut, core::mem::size_of::<D>())
        && slabs_share_no_line(dims, src, first_cut, core::mem::size_of::<S>())
    {
        return None;
    }
    Some(model)
}

/// Cache tiles for replaying a spilled axis list, or `None` when the
/// column-major walk is kept.
///
/// The invariant: an axis list is tiled exactly when the same layout given
/// inline would be. The extent-1 axes therefore go first, in their given
/// order — they address offset 0 whatever their stride, so dropping them keeps
/// the slot order and the decision of the layout they padded. A compressed
/// rank at most [`RAW_FUSED_RANK_LIMIT`] is decided and replayed through
/// [`PairTiles`] itself ([#425]).
///
/// [#425]: https://github.com/tensor4all/strided-rs/issues/425
pub(crate) fn spilled_pair_tiles<D, S>(axes: &[FusedAxis]) -> Option<Tiles> {
    if spilled_total(axes) <= 1 {
        return None;
    }
    let mut compressed = FusedPairLayout::empty();
    let mut rank = 0;
    for axis in axes.iter().filter(|axis| axis.dim != 1) {
        if rank < RAW_FUSED_RANK_LIMIT {
            compressed.dims[rank] = axis.dim;
            compressed.dst_strides[rank] = axis.dst;
            compressed.src_strides[rank] = axis.src;
        }
        rank += 1;
    }
    if rank <= RAW_FUSED_RANK_LIMIT {
        // Fewer than two moving axes are one run, which `pair_tiles` declines.
        compressed.rank = rank;
        return pair_tiles::<D, S>(&compressed).map(Tiles::Inline);
    }
    // The exits the model decides before its metadata are answered from the axes,
    // before any slice is materialized: a call it declines allocates nothing
    // ([#425]).
    if crate::block::exits_before_metadata(
        axes.iter()
            .filter(|axis| axis.dim != 1)
            .map(|axis| (axis.dim, axis.dst, axis.src)),
        core::mem::size_of::<D>(),
        core::mem::size_of::<S>(),
    ) {
        return None;
    }
    let mut dims = Vec::with_capacity(rank);
    let mut dst = Vec::with_capacity(rank);
    let mut src = Vec::with_capacity(rank);
    for axis in axes.iter().filter(|axis| axis.dim != 1) {
        dims.push(axis.dim);
        dst.push(axis.dst);
        src.push(axis.src);
    }
    let blocks = tile_blocks::<D, S>(&dims, &dst, &src)?;
    Some(Tiles::Spilled(SpilledTiles {
        dims,
        dst,
        src,
        blocks: blocks.into_vec(),
    }))
}

/// `layout` without its extent-1 axes, in the given order.
fn without_unit_axes(layout: &FusedPairLayout) -> FusedPairLayout {
    let mut compressed = FusedPairLayout::empty();
    let mut rank = 0;
    for axis in 0..layout.rank {
        if layout.dims[axis] == 1 {
            continue;
        }
        compressed.dims[rank] = layout.dims[axis];
        compressed.dst_strides[rank] = layout.dst_strides[axis];
        compressed.src_strides[rank] = layout.src_strides[axis];
        rank += 1;
    }
    compressed.rank = rank;
    compressed
}

/// Whether one operand's *slabs*, its sub-boxes over the leading `prefix`
/// axes at each coordinate of the other axes, provably share no cache line,
/// so that no visit order of the other axes can reuse a line across slabs.
///
/// Both walks visit every slab contiguously in the same internal order, so a
/// tiled walk can save traffic only on a line touched by two slabs. With the
/// byte span `w` of a slab's bounding interval and the tail byte strides all
/// multiples of the cache line, and the tail strides nested (each, in
/// ascending order, at least `w` plus the spans of the smaller ones), the
/// bounding intervals of distinct slabs are disjoint and congruent modulo the
/// line. The helper sees no base address and assumes the lowest address of
/// each bounding interval is line-aligned. An aligned coordinate-zero pointer
/// is not enough when a prefix stride is negative. Under that assumption every
/// slab has lines of its own. Otherwise address-adjacent slabs can share one
/// boundary line, so declining forgoes at most one reuse per adjacent pair,
/// `S - 1` for `S` slabs. That is a large share of the touched lines for
/// sparse or line-sized slabs. A zero-sized operand moves no line. Arithmetic
/// is checked, and an unrepresentable quantity answers `false`, which keeps
/// the tiles.
///
/// Strided.jl's `_computeblocks` has no such exit: it tiles these layouts
/// because the kept prefix's region exceeds its budget, although the tile
/// only reorders slabs that share nothing.
fn slabs_share_no_line(dims: &[usize], strides: &[isize], prefix: usize, elem_size: usize) -> bool {
    if elem_size == 0 {
        return true;
    }
    let line = crate::CACHE_LINE_SIZE as u128;
    let elem = elem_size as u128;
    // `|s| * elem < 2^127`; only the products with extents and the sums can
    // overflow.
    let byte = |axis: usize| strides[axis].unsigned_abs() as u128 * elem;
    let reach = |axis: usize| ((dims[axis] - 1) as u128).checked_mul(byte(axis));
    let Some(width) = (0..prefix).try_fold(elem, |width, axis| width.checked_add(reach(axis)?))
    else {
        return false;
    };
    if width % line != 0 {
        return false;
    }
    // Inline up to `RAW_FUSED_RANK_LIMIT`, heap above it: the spilled route
    // reaches this with more axes (issue #425), and the inline route must stay
    // allocation-free.
    let count = dims.len() - prefix;
    let mut inline = [(0u128, 0usize); RAW_FUSED_RANK_LIMIT];
    let mut spilled = Vec::new();
    let tail: &mut [(u128, usize)] = if count <= RAW_FUSED_RANK_LIMIT {
        let tail = &mut inline[..count];
        for (slot, axis) in tail.iter_mut().zip(prefix..dims.len()) {
            *slot = (byte(axis), axis);
        }
        tail
    } else {
        spilled.extend((prefix..dims.len()).map(|axis| (byte(axis), axis)));
        &mut spilled
    };
    tail.sort_unstable();
    let mut span = width;
    for &(stride, axis) in tail.iter() {
        if stride % line != 0 || stride < span {
            return false;
        }
        let Some(next) = reach(axis).and_then(|reach| span.checked_add(reach)) else {
            return false;
        };
        span = next;
    }
    true
}

/// Replay every logical index of `tile_box` tile by tile: the tiles of extents
/// `blocks`, anchored at the box origin and clipped to the box, are visited in
/// column-major order of the tile grid, and each tile is walked in
/// column-major order by [`apply_fused_range`].
///
/// Kept out of line so that the untiled replay keeps its code shape.
///
/// # Safety
///
/// The [`apply_fused_range`] contract for the whole of `tile_box`, and every
/// `blocks[axis]` for `axis < tile_box.rank` is at least 1.
#[inline(never)]
pub(crate) unsafe fn apply_tiled_box<D, S, Apply>(
    dst_ptr: *mut D,
    dst_base: isize,
    src_ptr: *const S,
    src_base: isize,
    tile_box: &FusedPairLayout,
    blocks: &[usize; RAW_FUSED_RANK_LIMIT],
    apply: &Apply,
) where
    D: Copy,
    S: Copy,
    Apply: Fn(&mut D, S),
{
    let rank = tile_box.rank;
    let mut tile = *tile_box;
    for (dim, &block) in tile.dims[..rank].iter_mut().zip(blocks) {
        *dim = block.min(*dim);
    }
    // `origin[axis]` is the first box coordinate of the current tile along
    // `axis`; `dst_tile`/`src_tile` are the offsets of the tile origin.
    let mut origin = [0usize; RAW_FUSED_RANK_LIMIT];
    let mut dst_tile = dst_base;
    let mut src_tile = src_base;
    loop {
        #[cfg(test)]
        crate::threading::record_tile_box();
        // SAFETY: the tile is a sub-box of `tile_box` at a reachable origin, so
        // the caller's contract covers every logical index of the tile.
        unsafe {
            apply_fused_range(
                dst_ptr,
                dst_tile,
                src_ptr,
                src_tile,
                &tile,
                0,
                fused_total(&tile),
                apply,
            );
        }
        // Column-major odometer over the tile grid. An axis advances only
        // when another tile follows along it, so `blocks[axis]` is at most
        // its last box coordinate, and it rewinds by its current origin:
        // every intermediate offset is a reachable offset of the box.
        let mut axis = 0;
        loop {
            if axis == rank {
                return;
            }
            let next = origin[axis] + blocks[axis];
            if next < tile_box.dims[axis] {
                origin[axis] = next;
                dst_tile += blocks[axis] as isize * tile_box.dst_strides[axis];
                src_tile += blocks[axis] as isize * tile_box.src_strides[axis];
                tile.dims[axis] = blocks[axis].min(tile_box.dims[axis] - next);
                break;
            }
            dst_tile -= origin[axis] as isize * tile_box.dst_strides[axis];
            src_tile -= origin[axis] as isize * tile_box.src_strides[axis];
            origin[axis] = 0;
            tile.dims[axis] = blocks[axis].min(tile_box.dims[axis]);
            axis += 1;
        }
    }
}

/// Replay every logical index of `box_dims`, a sub-box of `tiles`, tile by
/// tile: the tiles of extents `tiles.blocks`, anchored at the box origin and
/// clipped to it, are visited in column-major order of the tile grid, and each
/// tile is walked in column-major order by [`apply_spilled_range`].
///
/// The spilled counterpart of [`apply_tiled_box`]: same tile odometer, same
/// per-element walk as the untiled replay, only the axis storage differs. Kept
/// out of line so that the untiled spilled replay keeps its code shape.
///
/// # Safety
///
/// The [`apply_spilled_range`] contract for the whole of `box_dims`, which
/// must be `tiles.dims` or a sub-box of it anchored at `dst_base`/`src_base`,
/// and every `tiles.blocks[axis]` at least 1.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) unsafe fn apply_spilled_tiled_box<D, S, Apply>(
    dst_ptr: *mut D,
    dst_base: isize,
    src_ptr: *const S,
    src_base: isize,
    tiles: &SpilledTiles,
    box_dims: &[usize],
    apply: &Apply,
) where
    D: Copy,
    S: Copy,
    Apply: Fn(&mut D, S),
{
    let rank = box_dims.len();
    // One rank-sized axis list for the current tile and one cursor for the
    // whole walk, however many tiles it visits.
    let mut tile: Vec<FusedAxis> = (0..rank)
        .map(|axis| FusedAxis {
            dim: 1,
            dst: tiles.dst[axis],
            src: tiles.src[axis],
        })
        .collect();
    let mut index = vec![0usize; rank];
    let mut origin = vec![0usize; rank];
    let mut dst_tile = dst_base;
    let mut src_tile = src_base;
    loop {
        #[cfg(test)]
        crate::threading::record_tile_box();
        let mut count = 1usize;
        for axis in 0..rank {
            let dim = tiles.blocks[axis].min(box_dims[axis] - origin[axis]);
            tile[axis].dim = dim;
            count *= dim;
        }
        // SAFETY: the tile is a sub-box of `box_dims` at a reachable origin, so
        // the caller's contract covers every logical index of the tile.
        unsafe {
            apply_spilled_range_with(
                dst_ptr, dst_tile, src_ptr, src_tile, &tile, 0, count, &mut index, apply,
            );
        }
        // The column-major odometer of [`apply_tiled_box`], over slices.
        let mut axis = 0;
        loop {
            if axis == rank {
                return;
            }
            let next = origin[axis] + tiles.blocks[axis];
            if next < box_dims[axis] {
                origin[axis] = next;
                dst_tile += tiles.blocks[axis] as isize * tiles.dst[axis];
                src_tile += tiles.blocks[axis] as isize * tiles.src[axis];
                break;
            }
            dst_tile -= origin[axis] as isize * tiles.dst[axis];
            src_tile -= origin[axis] as isize * tiles.src[axis];
            origin[axis] = 0;
            axis += 1;
        }
    }
}

/// Serial replay of a fused pair layout over borrowed raw views.
///
/// The destination layout need not be injective: runs are replayed in order,
/// so accumulating callers such as [`axpy_raw`] keep their sequential
/// semantics. Parallel replay goes through `copy_plan::replay_fused`, which
/// the plans and the `replay_*_raw` entries call only with a destination
/// proven injective or a serial token.
pub(crate) fn apply_fused_pair<D, S, Apply>(
    dst: &mut RawStridedMut<'_, D>,
    src: &RawStridedRef<'_, S>,
    layout: &FusedPairLayout,
    apply: Apply,
) where
    D: Copy,
    S: Copy,
    Apply: Fn(&mut D, S),
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
            dst_ptr, dst_base, src_ptr, src_base, layout, 0, total, &apply,
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
pub(crate) fn apply_spilled_pair<D, S, Apply>(
    dst: &mut RawStridedMut<'_, D>,
    src: &RawStridedRef<'_, S>,
    axes: &[FusedAxis],
    apply: Apply,
) where
    D: Copy,
    S: Copy,
    Apply: Fn(&mut D, S),
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
        apply_spilled_range(dst_ptr, dst_base, src_ptr, src_base, axes, 0, total, &apply);
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
// Range leaf: both bases, the layout, the range and the closure, called once per worker
// range.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn apply_fused_range<D, S, Apply>(
    dst_ptr: *mut D,
    dst_base: isize,
    src_ptr: *const S,
    src_base: isize,
    layout: &FusedPairLayout,
    start: usize,
    len: usize,
    apply: &Apply,
) where
    D: Copy,
    S: Copy,
    Apply: Fn(&mut D, S),
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
    // The axis indexes `index`, `layout.dims` and both stride tables at once.
    #[allow(clippy::needless_range_loop)]
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
                dst_ptr, dst_run, inner_dst, src_ptr, src_run, inner_src, run, apply,
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
// Same shape as `apply_fused_range` for the spilled layout, per worker range.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) unsafe fn apply_spilled_range<D, S, Apply>(
    dst_ptr: *mut D,
    dst_base: isize,
    src_ptr: *const S,
    src_base: isize,
    axes: &[FusedAxis],
    start: usize,
    len: usize,
    apply: &Apply,
) where
    D: Copy,
    S: Copy,
    Apply: Fn(&mut D, S),
{
    if len == 0 {
        return;
    }
    let mut index = vec![0usize; axes.len()];
    // SAFETY: forwarded caller contract; `index` is the rank-sized cursor.
    unsafe {
        apply_spilled_range_with(
            dst_ptr, dst_base, src_ptr, src_base, axes, start, len, &mut index, apply,
        )
    }
}

/// [`apply_spilled_range`] with the caller's rank-sized coordinate buffer, so
/// that the tile walk keeps one cursor for all its tiles.
///
/// # Safety
///
/// The [`apply_spilled_range`] contract, and `index.len()` at least
/// `axes.len()`.
#[allow(clippy::too_many_arguments)]
unsafe fn apply_spilled_range_with<D, S, Apply>(
    dst_ptr: *mut D,
    dst_base: isize,
    src_ptr: *const S,
    src_base: isize,
    axes: &[FusedAxis],
    start: usize,
    len: usize,
    index: &mut [usize],
    apply: &Apply,
) where
    D: Copy,
    S: Copy,
    Apply: Fn(&mut D, S),
{
    if len == 0 {
        return;
    }
    let rank = axes.len();
    debug_assert!(index.len() >= rank);
    let inner_len = axes[0].dim;
    let inner_dst = axes[0].dst;
    let inner_src = axes[0].src;

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
                dst_ptr, dst_run, inner_dst, src_ptr, src_run, inner_src, run, apply,
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
// Run leaf: both base/offset/stride triples, the length and the closure, per run.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn apply_run<D, S, Apply>(
    dst_ptr: *mut D,
    dst_offset: isize,
    dst_stride: isize,
    src_ptr: *const S,
    src_offset: isize,
    src_stride: isize,
    len: usize,
    apply: &Apply,
) where
    D: Copy,
    S: Copy,
    Apply: Fn(&mut D, S),
{
    if dst_stride == 1 {
        // SAFETY: a unit destination stride makes the run `len` consecutive
        // in-bounds slots starting at `dst_offset`, exclusively owned here.
        let dst_run = unsafe { core::slice::from_raw_parts_mut(dst_ptr.offset(dst_offset), len) };
        if src_stride == 1 {
            // A `copy_nonoverlapping` per run is not used for plain copies:
            // it measured slower than this element loop for short runs
            // (one `memcpy` call each) and for long cache-resident copies.
            // SAFETY: `len` consecutive in-bounds source slots, which the
            // exclusive destination run cannot overlap.
            let src_run = unsafe { core::slice::from_raw_parts(src_ptr.offset(src_offset), len) };
            for (dst, &value) in dst_run.iter_mut().zip(src_run) {
                apply(dst, value);
            }
            return;
        }
        if src_stride == -1 {
            // INVARIANT: the range walks never pass an empty run, and both
            // strides are nonzero, so the validated reach `len - 1` fits
            // `isize`; casting `len` itself would wrap for a run of
            // `isize::MAX + 1` elements. (A `saturating_sub` here measurably
            // slowed short unit-stride runs by changing the code layout of
            // the whole inlined walk.)
            let last = (len - 1) as isize;
            // SAFETY: the run reads `src_offset - (len - 1) ..= src_offset`,
            // all in-bounds; the lowest one is the run's last source slot.
            let src_run =
                unsafe { core::slice::from_raw_parts(src_ptr.offset(src_offset - last), len) };
            for (dst, &value) in dst_run.iter_mut().zip(src_run.iter().rev()) {
                apply(dst, value);
            }
            return;
        }
        // SAFETY: the run start is an in-bounds source slot.
        let src_start = unsafe { src_ptr.offset(src_offset) };
        for (position, dst) in dst_run.iter_mut().enumerate() {
            // SAFETY: `position < len`, so this is a run source slot.
            let value = unsafe { *src_start.offset(position as isize * src_stride) };
            apply(dst, value);
        }
        return;
    }
    // SAFETY: the run start is an in-bounds slot of each allocation.
    let dst_start = unsafe { dst_ptr.offset(dst_offset) };
    let src_start = unsafe { src_ptr.offset(src_offset) };
    // The bound stays `usize`: a checked broadcast run may hold more than
    // `isize::MAX` elements, and `len as isize` would wrap to an empty loop.
    for position in 0..len {
        // INVARIANT: a nonzero stride bounds the run by its validated reach,
        // so `position` fits `isize` exactly; a longer run is possible only
        // along a zero stride, where the wrapped cast is multiplied by zero.
        let position = position as isize;
        // SAFETY: `position < len`, so both are slots of this run.
        unsafe {
            let value = *src_start.offset(position * src_stride);
            apply(&mut *dst_start.offset(position * dst_stride), value);
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
///
/// # Errors
///
/// Returns [`crate::StridedError::ShapeMismatch`] when `dest` and `src` do
/// not describe the same extents, or [`crate::StridedError::OffsetOverflow`]
/// when the element count does not fit `usize`. Nothing is written on error.
pub fn copy_scale_raw<T>(
    dest: &mut RawStridedMut<'_, T>,
    src: &RawStridedRef<'_, T>,
    scale: T,
) -> Result<()>
where
    T: Copy + Mul<T, Output = T> + ElementOpApply + Send + Sync,
{
    ensure_same_dims(dest.dims(), src.dims())?;
    match fuse_pair_traversal(dest.dims(), dest.strides(), src.strides()) {
        PairTraversal::Inline(layout) => {
            apply_fused_pair(dest, src, &layout, |dst: &mut T, value: T| {
                *dst = scale * value
            })
        }
        PairTraversal::Spilled(axes) => {
            apply_spilled_pair(dest, src, &axes, |dst: &mut T, value: T| {
                *dst = scale * value
            })
        }
    }
    Ok(())
}

/// `dest = scale * conj(src)` over borrowed raw strided layouts.
///
/// # Errors
///
/// Returns [`crate::StridedError::ShapeMismatch`] when `dest` and `src` do
/// not describe the same extents, or [`crate::StridedError::OffsetOverflow`]
/// when the element count does not fit `usize`. Nothing is written on error.
pub fn copy_scale_conj_raw<T>(
    dest: &mut RawStridedMut<'_, T>,
    src: &RawStridedRef<'_, T>,
    scale: T,
) -> Result<()>
where
    T: Copy + Mul<T, Output = T> + ElementOpApply + Send + Sync,
{
    ensure_same_dims(dest.dims(), src.dims())?;
    match fuse_pair_traversal(dest.dims(), dest.strides(), src.strides()) {
        PairTraversal::Inline(layout) => {
            apply_fused_pair(dest, src, &layout, |dst: &mut T, value: T| {
                *dst = scale * value.conj()
            })
        }
        PairTraversal::Spilled(axes) => {
            apply_spilled_pair(dest, src, &axes, |dst: &mut T, value: T| {
                *dst = scale * value.conj()
            })
        }
    }
    Ok(())
}

/// `dest = alpha * src + dest` over borrowed raw strided layouts.
///
/// # Errors
///
/// Returns [`crate::StridedError::ShapeMismatch`] when `dest` and `src` do
/// not describe the same extents, or [`crate::StridedError::OffsetOverflow`]
/// when the element count does not fit `usize`. Nothing is written on error.
pub fn axpy_raw<T>(
    dest: &mut RawStridedMut<'_, T>,
    src: &RawStridedRef<'_, T>,
    alpha: T,
) -> Result<()>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + ElementOpApply + Send + Sync,
{
    ensure_same_dims(dest.dims(), src.dims())?;
    match fuse_pair_traversal(dest.dims(), dest.strides(), src.strides()) {
        PairTraversal::Inline(layout) => {
            apply_fused_pair(dest, src, &layout, |dst: &mut T, value: T| {
                *dst = *dst + alpha * value
            })
        }
        PairTraversal::Spilled(axes) => {
            apply_spilled_pair(dest, src, &axes, |dst: &mut T, value: T| {
                *dst = *dst + alpha * value
            })
        }
    }
    Ok(())
}

/// `dest = alpha * conj(src) + dest` over borrowed raw strided layouts.
///
/// # Errors
///
/// Returns [`crate::StridedError::ShapeMismatch`] when `dest` and `src` do
/// not describe the same extents, or [`crate::StridedError::OffsetOverflow`]
/// when the element count does not fit `usize`. Nothing is written on error.
pub fn axpy_conj_raw<T>(
    dest: &mut RawStridedMut<'_, T>,
    src: &RawStridedRef<'_, T>,
    alpha: T,
) -> Result<()>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + ElementOpApply + Send + Sync,
{
    ensure_same_dims(dest.dims(), src.dims())?;
    match fuse_pair_traversal(dest.dims(), dest.strides(), src.strides()) {
        PairTraversal::Inline(layout) => {
            apply_fused_pair(dest, src, &layout, |dst: &mut T, value: T| {
                *dst = *dst + alpha * value.conj()
            })
        }
        PairTraversal::Spilled(axes) => {
            apply_spilled_pair(dest, src, &axes, |dst: &mut T, value: T| {
                *dst = *dst + alpha * value.conj()
            })
        }
    }
    Ok(())
}

/// Checked replay of the given axes of two views: the shared body of the
/// `replay_*_raw` entries.
///
/// The element count is checked before anything is built. The unfused
/// traversal keeps every given extent, and the replay totals (`fused_total`,
/// `replay_spilled_raw`) take an unchecked product, so the zero-first count
/// of `kernel::total_len` has to keep an empty layout whose nonzero prefix
/// overflows, such as `[usize::MAX, 2, 0]`, away from them.
fn replay_given<D, S, Apply>(
    dest: &mut RawStridedMut<'_, D>,
    src: &RawStridedRef<'_, S>,
    apply: Apply,
    exec: Exec<'_>,
) -> Result<()>
where
    D: Copy + Send + Sync,
    S: Copy + Send + Sync,
    Apply: Fn(&mut D, S) + Sync,
{
    ensure_same_dims(dest.dims(), src.dims())?;
    let total = crate::kernel::total_len(dest.dims())?;
    if total == 0 {
        return Ok(());
    }
    let traversal = PairTraversal::given(dest.dims(), dest.strides(), src.strides());
    let injective =
        || crate::layout_check::is_injective_layout_without_alloc(dest.dims(), dest.strides());
    let mut tiles = traversal.tiles::<D, S>();
    let mut proven = None;
    if tiles.is_some() {
        let holds = injective();
        proven = Some(holds);
        if !holds {
            tiles = None;
        }
    }
    let exec = if tiles.is_some() {
        // The tiled tree gates every node on the caller's token itself; the
        // flat width rule would cap it at the width of one flat dispatch.
        exec
    } else {
        let exec = crate::cost::for_len(exec, KernelClass::Copy, total);
        if !exec.is_serial() && !proven.unwrap_or_else(injective) {
            // A destination not proven injective replays as one sequential
            // column-major walk, which is also its defined semantics; the
            // budget-one token cannot be widened again by the replay's own
            // width rule.
            exec.limit(NonZeroUsize::MIN)
        } else {
            exec
        }
    };
    // SAFETY: `traversal` is built from exactly the views' dims and strides
    // (`ensure_same_dims` proved the dims agree), so the views' bounds proofs
    // cover every replayed offset; `total_len` checked the nonzero product;
    // `tiles` is `PairTraversal::tiles` of `traversal`, and tiles or a
    // non-serial `exec` are left only for a destination proven injective;
    // the exclusive and shared borrows cannot overlap.
    unsafe { crate::copy_plan::replay_fused(dest, src, &traversal, tiles.as_ref(), apply, exec) };
    Ok(())
}

/// `dest = src`, replaying the axes of the two views exactly as given.
///
/// The views carry already-fused dims: no axis is reordered, dropped or
/// merged, and no plan is compiled. Axis 0 is the inner run of the
/// column-major walk; when the shared cache-block model finds tiles that change
/// the visit order and can reuse a cache line, the walk goes tile by tile
/// instead, each tile column-major over the given axes. Bounds were validated when the views were built
/// ([`RawStridedRef::new`], [`RawStridedMut::new`]). On a successful call the
/// replay itself (validation, traversal and dispatch) allocates nothing for a
/// rank at most [`RAW_FUSED_RANK_LIMIT`]; above it, the axis list, the tile
/// decision's slices and the cursors are stored in rank-sized buffers. An error
/// may allocate its payload.
///
/// The width comes from the repository rule on `exec`. A parallel or tiled
/// replay runs only on a destination proven injective without allocating; any
/// other destination is replayed serially in column-major order of the given
/// axes.
///
/// # Errors
///
/// [`StridedError::ShapeMismatch`](crate::StridedError::ShapeMismatch) when the
/// two views have different dims, and
/// [`StridedError::OffsetOverflow`](crate::StridedError::OffsetOverflow) when the
/// element count overflows `usize`; both before any write.
///
/// # Examples
///
/// Copy the outer band `1..3` of a stored `2 x 3` block into a transposed
/// destination:
///
/// ```rust
/// use strided_basic::{replay_copy_raw, Exec, RawStridedMut, RawStridedRef};
///
/// let stored = [0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0]; // column-major 2 x 3
/// let band = [2usize, 2];
/// let src = RawStridedRef::new(&stored, &band, &[1, 2], 2).unwrap();
/// let mut out = [0.0f64; 4];
/// let mut dst = RawStridedMut::new(&mut out, &band, &[2, 1], 0).unwrap();
/// replay_copy_raw(&mut dst, &src, Exec::serial()).unwrap();
/// assert_eq!(out, [2.0, 4.0, 3.0, 5.0]);
/// ```
pub fn replay_copy_raw<T>(
    dest: &mut RawStridedMut<'_, T>,
    src: &RawStridedRef<'_, T>,
    exec: Exec<'_>,
) -> Result<()>
where
    T: Copy + Send + Sync,
{
    replay_given(dest, src, |dst: &mut T, value: T| *dst = value, exec)
}

/// [`replay_copy_raw`] into a potentially uninitialized destination.
///
/// On success every reachable destination element is initialized and no
/// other slot is written. The destination is never read.
///
/// # Errors
///
/// As [`replay_copy_raw`].
///
/// # Examples
///
/// ```rust
/// use core::mem::MaybeUninit;
/// use strided_basic::{replay_copy_raw_uninit, Exec, RawStridedMut, RawStridedRef};
///
/// let src = [1.0f64, 2.0, 3.0];
/// let mut storage = [MaybeUninit::<f64>::uninit(); 3];
/// let mut dst = RawStridedMut::new(&mut storage, &[3], &[1], 0).unwrap();
/// let source = RawStridedRef::new(&src, &[3], &[-1], 2).unwrap();
/// replay_copy_raw_uninit(&mut dst, &source, Exec::serial()).unwrap();
/// drop(dst);
/// // SAFETY: every reachable destination slot was initialized above.
/// let values: Vec<f64> = storage.iter().map(|v| unsafe { v.assume_init() }).collect();
/// assert_eq!(values, [3.0, 2.0, 1.0]);
/// ```
pub fn replay_copy_raw_uninit<T>(
    dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
    src: &RawStridedRef<'_, T>,
    exec: Exec<'_>,
) -> Result<()>
where
    T: Copy + Send + Sync,
{
    replay_given(
        dest,
        src,
        |dst: &mut MaybeUninit<T>, value: T| {
            dst.write(value);
        },
        exec,
    )
}

/// `f(&mut dest[i], src[i])` for every element, replaying the axes of the two
/// views exactly as given.
///
/// Traversal, width and errors are those of [`replay_copy_raw`], and so is the
/// allocation guarantee, which covers the replay only: allocations made by `f`
/// are its own. `f` runs once per logical element with exactly its arithmetic,
/// so an accumulate such as `|d, s| *d += s` is exact. The order in which
/// elements are visited is unspecified, and on a parallel replay `f` runs
/// concurrently on disjoint slots: the result is defined only for an `f` whose
/// effect depends on the supplied slot values alone. On a destination that is
/// not proven injective, the elements are visited serially in column-major
/// order of the given axes.
///
/// # Errors
///
/// As [`replay_copy_raw`].
///
/// # Examples
///
/// ```rust
/// use strided_basic::{replay_map_raw, Exec, RawStridedMut, RawStridedRef};
///
/// let src = [1.0f64, 2.0, 3.0, 4.0];
/// let mut acc = [10.0f64, 20.0, 30.0, 40.0];
/// let source = RawStridedRef::new(&src, &[2, 2], &[2, 1], 0).unwrap();
/// let mut dst = RawStridedMut::new(&mut acc, &[2, 2], &[1, 2], 0).unwrap();
/// replay_map_raw(&mut dst, &source, |d, s| *d += s, Exec::serial()).unwrap();
/// assert_eq!(acc, [11.0, 23.0, 32.0, 44.0]);
/// ```
pub fn replay_map_raw<D, S, F>(
    dest: &mut RawStridedMut<'_, D>,
    src: &RawStridedRef<'_, S>,
    f: F,
    exec: Exec<'_>,
) -> Result<()>
where
    D: Copy + Send + Sync,
    S: Copy + Send + Sync,
    F: Fn(&mut D, S) + Send + Sync,
{
    replay_given(dest, src, f, exec)
}

/// `dest[i] = f(src[i])` into a potentially uninitialized destination.
///
/// `f` cannot read the destination. On success every reachable destination
/// element is initialized and no other slot is written. The visit-order,
/// callback and allocation contract is that of [`replay_map_raw`].
///
/// # Errors
///
/// As [`replay_copy_raw`].
///
/// # Examples
///
/// ```rust
/// use core::mem::MaybeUninit;
/// use strided_basic::{replay_map_raw_uninit, Exec, RawStridedMut, RawStridedRef};
///
/// let src = [1i32, 2, 3];
/// let mut storage = [MaybeUninit::<f64>::uninit(); 3];
/// let mut dst = RawStridedMut::new(&mut storage, &[3], &[1], 0).unwrap();
/// let source = RawStridedRef::new(&src, &[3], &[1], 0).unwrap();
/// replay_map_raw_uninit(&mut dst, &source, |s| f64::from(s) * 0.5, Exec::serial()).unwrap();
/// drop(dst);
/// // SAFETY: every reachable destination slot was initialized above.
/// let values: Vec<f64> = storage.iter().map(|v| unsafe { v.assume_init() }).collect();
/// assert_eq!(values, [0.5, 1.0, 1.5]);
/// ```
pub fn replay_map_raw_uninit<D, S, F>(
    dest: &mut RawStridedMut<'_, MaybeUninit<D>>,
    src: &RawStridedRef<'_, S>,
    f: F,
    exec: Exec<'_>,
) -> Result<()>
where
    D: Copy + Send + Sync,
    S: Copy + Send + Sync,
    F: Fn(S) -> D + Send + Sync,
{
    replay_given(
        dest,
        src,
        move |dst: &mut MaybeUninit<D>, value: S| {
            dst.write(f(value));
        },
        exec,
    )
}

#[cfg(test)]
#[path = "raw_ops/tests/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "raw_ops/tests/replay_tests.rs"]
mod replay_tests;

#[cfg(test)]
#[path = "raw_ops/tests/tiled_tests.rs"]
mod tiled_tests;
