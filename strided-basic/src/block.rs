//! Block size computation ported from Strided.jl
//!
//! This module computes optimal block sizes for cache-efficient iteration.
//! It uses Julia's `totalmemoryregion` for accurate memory region estimation
//! and `_computeblocks` for iterative block size reduction.
//!
//! # Element sizes
//!
//! Every operand carries its own element size, as Strided.jl forms byte
//! strides per array (`sizeof.(eltype.(arrays))`, `src/mapreduce.jl`). A
//! zero-size operand (a ZST, or an operand whose type has no storage) has a
//! zero byte stride on every axis, so every one of its stride orders is `1`.
//! It therefore contributes one cache line to the region and never shrinks a
//! block by itself, but it is *not* neutral: its constant order `1` is the
//! global minimum at every recursion level, so another operand's larger order
//! no longer matches it and the keep-first recursion stops. Byte-stride
//! strict increase, and with it the exit that keeps every axis, therefore
//! requires every element size to be positive. When every operand has size
//! zero, all orders are `1` everywhere and the model keeps the full
//! dimensions.
//!
//! # Arithmetic contract
//!
//! Byte strides, memory regions and halving scores are evaluated in `u128`
//! from exact byte magnitudes `|s| · elem_size[k]`; a memory region is first
//! attempted in checked `u64`, which gives the same exact value when no step
//! overflows and otherwise defers to `u128`. For extents `d ≥ 1` whose
//! product fits in `usize` (callers check it with `kernel::total_len`), every
//! intermediate is exact. The result is then the block vector of the
//! mathematical-integer model, for any element size and any stride, provided
//! the core's stride-order table (operands × rank entries) has a representable
//! length and its storage is obtained.
//!
//! - **Checked-length fallback.** If that length overflows `usize`, the full
//!   dimensions are returned. They are a valid block vector (`1 ≤ b_i ≤ d_i`)
//!   but not, in general, the model's result.
//! - **Outside the shape-product precondition.** The region arithmetic
//!   saturates: both budget comparisons keep their unbounded outcome and
//!   nothing panics.

use crate::fuse::compute_costs_into;
use crate::inline::{InlineVec, OPERAND_CAP};
use crate::{BLOCK_MEMORY_SIZE, CACHE_LINE_SIZE};
use strided_view::auxiliary::index_order_into;

/// Operand-major scratch of the core's stride orders: inline up to
/// [`OPERAND_CAP`] operands of `INLINE_RANK` axes, spilling above.
type Scratch<T> = tinyvec::TinyVec<[T; OPERAND_CAP * strided_view::INLINE_RANK]>;

/// The core's per-operand metadata. Byte strides are formed on the fly from
/// the element strides (`|s| * elem_size[k]`, exact in `u128`) rather than
/// stored; the stride orders sit in one flat table where row `k` holds
/// operand `k` at a pitch of the original rank on every recursion level, so
/// the suffix at `offset` is `[k * pitch + offset .. (k + 1) * pitch]`.
struct Rows<'a> {
    pitch: usize,
    strides: &'a [&'a [isize]],
    elem_sizes: &'a [usize],
    orders: &'a [usize],
}

impl Rows<'_> {
    fn orders(&self, operand: usize, offset: usize) -> &[usize] {
        &self.orders[operand * self.pitch + offset..(operand + 1) * self.pitch]
    }
}

/// Compute block sizes for tiled iteration.
///
/// This implementation follows Julia's `_computeblocks` algorithm on axes
/// that are already in iteration order:
/// 1. If the memory region fits in the block budget, use full dimensions
/// 2. Otherwise compute stride orders and costs, and reduce blocks by
///    cost-weighted halving and decrementing
///
/// The full dimensions are returned whenever [`reduced_block_sizes`] finds
/// that one of the model's sufficient exits applies.
///
/// # Arguments
/// * `dims` - The dimensions, in iteration order, each at least 1
/// * `strides_list` - Slice of element stride arrays, one per array, in the
///   same order as `dims`
/// * `elem_sizes` - Element size in bytes of each operand, in the same order
///
/// # Returns
/// Block sizes in iteration order
pub(crate) fn compute_block_sizes(
    dims: &[usize],
    strides_list: &[&[isize]],
    elem_sizes: &[usize],
) -> InlineVec<usize> {
    debug_assert_eq!(strides_list.len(), elem_sizes.len());
    let Some(blocks) = reduced_block_sizes(dims, strides_list, elem_sizes) else {
        return InlineVec::from(dims);
    };
    // The same exit the plan and replay walk takes (#422): the answer is the
    // full dimensions, which is how this caller asks for the flat walk.
    if declines_tiles(dims, strides_list, elem_sizes, &blocks) {
        return InlineVec::from(dims);
    }
    blocks
}

/// [`compute_block_sizes`] for a kernel that writes operand `dest`, with the
/// tiles fitted to the L1 sets by [`fit_l1_sets`] (#287).
///
/// The fit runs only on a block vector the model reduced and did not decline:
/// every exit that keeps the full dimensions keeps them here too.
pub(crate) fn destination_block_sizes(
    dims: &[usize],
    strides_list: &[&[isize]],
    elem_sizes: &[usize],
    dest: usize,
) -> InlineVec<usize> {
    let Some(mut blocks) = reduced_block_sizes(dims, strides_list, elem_sizes) else {
        return InlineVec::from(dims);
    };
    if declines_tiles(dims, strides_list, elem_sizes, &blocks) {
        return InlineVec::from(dims);
    }
    fit_l1_sets(dims, strides_list, elem_sizes, dest, &mut blocks);
    blocks
}

/// Shrink `blocks` until the lines a tile keeps live for reuse fit the L1
/// sets they fall in (#287).
///
/// Strided.jl's model budgets a tile by its total footprint only. A cache is
/// set-associative, though: one set holds at most [`L1_WAYS`](crate::L1_WAYS)
/// lines, and lines whose addresses differ by a multiple of the way size
/// `w = BLOCK_MEMORY_SIZE / L1_WAYS` share a set.
///
/// **Which lines are live.** In the tile walk axis 0 is the inner run, then
/// axes 1, 2, ... nest outward. For operand `k`, let `c_k` be its first axis
/// whose byte stride is below a line (its reuse axis). The lines touched along
/// the axes before `c_k` are revisited on the next step of axis `c_k` (the
/// next element of each line), so they must all stay resident across one
/// sweep of those axes. An operand with `c_k = 0` streams its lines and has
/// nothing live; one with no such axis never revisits a line.
///
/// **How many share a set.** Along an axis whose byte stride is a multiple
/// of the way size `w`, every step lands in the same set, so `b` steps put `b`
/// lines in one set. The operand's per-set load is the product of the extents
/// of such axes before `c_k`, and the loads of all operands add up. A
/// power-of-two leading dimension is the case: a 2048-element `f64` row stride
/// is 16 KiB, and the model's `[40, 32]` tile puts 40 lines in one 8-way set
/// ([#287]).
///
/// Only full aliasing counts. An axis whose stride is not a multiple of `w`
/// spreads its lines over `w / gcd(s, w) >= 2` sets, and on the #287 probe a
/// stride with two sets (2 KiB, `f64` 16384×256) showed no penalty at the
/// model's tile while the shorter runs of a fit cost the walk 1.4x; such axes
/// are left alone (see the worklog).
///
/// **The budget.** The load may not exceed
/// [`L1_LIVE_LINES_PER_SET`](crate::L1_LIVE_LINES_PER_SET), the ways less a
/// quarter of headroom for the pseudo-LRU replacement and the stride
/// prefetcher (see the constant).
///
/// **The fit.** While the load exceeds the budget, the outermost shrinkable
/// axis (the inner run stays long for the operands that stream it) is cut to
/// the largest extent that meets the budget, or, if that axis alone cannot, to
/// 1, and the next axis is tried. Blocks only shrink, so the model's footprint
/// bound still holds (upper bound), and only an axis on which some live
/// operand fully aliases is cut at all (no other axis moves).
///
/// **Which walks.** Only two-operand walks (one destination, one source:
/// `copy_into`, unary maps, the raw pair replay) are fitted; a walk with more
/// operands keeps the model's tiles. The #287 gate measured the fit on both:
/// every two-operand cell whose tiles changed ran faster (1.13x to 3.3x),
/// while three-operand `mul_into` regressed in 16 cells, up to 2.1x slower
/// (permuted B and rectangular transposes in `f64`), because the walk's
/// per-run cost on the shortened runs outweighs the set fit there. The
/// measurement is in the worklog.
///
/// **Numerical results.** Tiles change only the visit order. A caller either
/// writes each destination slot once from a pointwise expression (the map,
/// zip, fused and update entries validate an injective destination), or
/// accumulates into it (`add_acc` and the other `*_acc` entries accept a
/// destination with zero-stride axes). The order in which one output element
/// accumulates its contributions depends only on the blocks of the
/// destination's zero-stride axes, and those are never cut, so every output
/// is bit-identical to the model's tiles. A fold into one accumulator (`dot`,
/// the reductions) does not call this, because there every block orders the
/// fold.
///
/// [#287]: https://github.com/tensor4all/strided-rs/issues/287
pub(crate) fn fit_l1_sets(
    dims: &[usize],
    strides_list: &[&[isize]],
    elem_sizes: &[usize],
    dest: usize,
    blocks: &mut [usize],
) {
    debug_assert_eq!(strides_list.len(), elem_sizes.len());
    debug_assert_eq!(dims.len(), blocks.len());
    #[cfg(feature = "bench-hooks")]
    let budget = crate::bench_hooks::L1_SET_BUDGET.load(core::sync::atomic::Ordering::Relaxed);
    #[cfg(not(feature = "bench-hooks"))]
    let budget = crate::L1_LIVE_LINES_PER_SET;
    // Two-operand walks only: see "Which walks" above.
    if budget == 0 || strides_list.len() != 2 {
        return;
    }
    let sets = L1Sets::new(strides_list, elem_sizes);
    if sets.load(blocks, None) <= budget as u128 {
        return;
    }
    let frozen = |axis: usize| strides_list.get(dest).is_some_and(|s| s[axis] == 0);
    // Outermost first; each axis is visited once, and is left either meeting
    // the budget or contributing a factor of 1.
    for axis in (0..dims.len()).rev() {
        if frozen(axis) {
            continue;
        }
        let Some(floor) = sets.floor(axis, blocks[axis]) else {
            continue;
        };
        let current = blocks[axis];
        if sets.load(blocks, Some((axis, floor))) > budget as u128 {
            blocks[axis] = floor;
            continue;
        }
        // The load is nondecreasing in the extent: the largest extent in
        // `[floor, current)` within the budget.
        let (mut lo, mut hi) = (floor, current - 1);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if sets.load(blocks, Some((axis, mid))) <= budget as u128 {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        blocks[axis] = lo;
        return;
    }
}

/// The per-set line load of [`fit_l1_sets`]: each operand's reuse axis and the
/// byte strides before it.
struct L1Sets<'a> {
    strides_list: &'a [&'a [isize]],
    elem_sizes: &'a [usize],
}

impl<'a> L1Sets<'a> {
    const LINE: u128 = CACHE_LINE_SIZE as u128;
    const WAY: u128 = (BLOCK_MEMORY_SIZE / crate::L1_WAYS) as u128;

    fn new(strides_list: &'a [&'a [isize]], elem_sizes: &'a [usize]) -> Self {
        Self {
            strides_list,
            elem_sizes,
        }
    }

    /// Operand `k`'s reuse axis `c_k` (its first axis with a byte stride below
    /// a line), or `None` when it keeps no line live: it has no such axis, it
    /// is zero-sized, or `c_k = 0`.
    fn reuse_axis(&self, k: usize) -> Option<usize> {
        let elem = self.elem_sizes[k] as u128;
        if elem == 0 {
            return None;
        }
        let strides = self.strides_list[k];
        let c = strides
            .iter()
            .position(|&s| s.unsigned_abs() as u128 * elem < Self::LINE)?;
        (c > 0).then_some(c)
    }

    /// Whether every step of operand `k` along `axis` lands in the same set:
    /// its byte stride is a nonzero multiple of the way size.
    fn aliases(&self, k: usize, axis: usize) -> bool {
        let s = self.strides_list[k][axis].unsigned_abs() as u128 * self.elem_sizes[k] as u128;
        s != 0 && s.is_multiple_of(Self::WAY)
    }

    /// The summed per-set load of every operand's live lines, with `axis`
    /// optionally replaced by another extent. Saturates.
    fn load(&self, blocks: &[usize], with: Option<(usize, usize)>) -> u128 {
        let extent = |axis: usize| match with {
            Some((a, b)) if a == axis => b,
            _ => blocks[axis],
        };
        (0..self.strides_list.len())
            .filter_map(|k| {
                let c = self.reuse_axis(k)?;
                Some(
                    (0..c)
                        .filter(|&axis| self.aliases(k, axis))
                        .fold(1u128, |load, axis| {
                            load.saturating_mul(extent(axis) as u128)
                        }),
                )
            })
            .fold(0u128, u128::saturating_add)
    }

    /// `Some(1)` when cutting `axis` can lower the load (some operand keeps
    /// it live and fully aliases on it, and `block > 1`), else `None`.
    fn floor(&self, axis: usize, block: usize) -> Option<usize> {
        let counted = (0..self.strides_list.len())
            .any(|k| self.reuse_axis(k).is_some_and(|c| axis < c) && self.aliases(k, axis));
        (counted && block > 1).then_some(1)
    }
}

/// The model's block vector, or `None` when one of its sufficient exits keeps
/// the full dimensions before any metadata is built: rank zero, a memory
/// region within the budget, or operands whose byte strides are all nonzero
/// and strictly increasing in magnitude (each operand's stride order is
/// `1..=r`, so every axis is kept). The checked stride-order table length also
/// answers `None` (full dimensions) when it overflows. `Some` may still hold
/// the full dimensions when the core keeps every axis.
pub(crate) fn reduced_block_sizes(
    dims: &[usize],
    strides_list: &[&[isize]],
    elem_sizes: &[usize],
) -> Option<InlineVec<usize>> {
    debug_assert_eq!(strides_list.len(), elem_sizes.len());
    if model_exits(dims, strides_list, elem_sizes) {
        return None;
    }
    // A table length beyond `usize` cannot be stored; full dimensions are a
    // valid block vector, though not the model's result in general.
    let len = strides_list.len().checked_mul(dims.len())?;
    let mut orders: Scratch<usize> = Scratch::with_capacity(len);
    orders.resize(len, 0);
    let mut costs: InlineVec<isize> = InlineVec::with_capacity(dims.len());
    costs.resize(dims.len(), 0);
    let mut blocks = InlineVec::from(dims);
    reduced_block_sizes_into(
        dims,
        strides_list,
        elem_sizes,
        &mut blocks,
        &mut orders,
        &mut costs,
    );
    Some(blocks)
}

/// The model's sufficient exits that keep the full dimensions before any
/// metadata is built: where [`reduced_block_sizes`] returns `None`. A caller
/// asks this before it sets up the scratch of [`reduced_block_sizes_into`].
pub(crate) fn model_exits(dims: &[usize], strides_list: &[&[isize]], elem_sizes: &[usize]) -> bool {
    dims.is_empty()
        || memory_region(dims, strides_list, 0, elem_sizes) <= BLOCK_MEMORY_SIZE as u128
        || strides_list
            .iter()
            .zip(elem_sizes)
            .all(|(strides, &elem_size)| elem_size > 0 && strictly_increasing(strides))
}

/// [`reduced_block_sizes`] into caller storage, for a layout [`model_exits`]
/// does not answer: `blocks` and `costs` hold one slot per axis and `orders`
/// one per operand and axis, and `blocks` holds the model's blocks on return.
/// A caller that holds bounded storage for its rank runs the model without
/// allocating.
pub(crate) fn reduced_block_sizes_into(
    dims: &[usize],
    strides_list: &[&[isize]],
    elem_sizes: &[usize],
    blocks: &mut [usize],
    orders: &mut [usize],
    costs: &mut [isize],
) {
    debug_assert_eq!(strides_list.len(), elem_sizes.len());
    debug_assert_eq!(blocks.len(), dims.len());
    debug_assert_eq!(costs.len(), dims.len());
    debug_assert_eq!(orders.len(), strides_list.len() * dims.len());
    debug_assert!(!model_exits(dims, strides_list, elem_sizes));
    blocks.copy_from_slice(dims);
    let pitch = dims.len();
    for ((strides, &elem_size), row) in strides_list
        .iter()
        .zip(elem_sizes)
        .zip(orders.chunks_exact_mut(pitch.max(1)))
    {
        if elem_size == 0 {
            // Every byte stride is 0, so every axis has stride order 1.
            row.fill(1);
        } else {
            // With `elem_size > 0` the byte strides are an exact positive
            // multiple of the element strides, so both have the same order.
            // INVARIANT: `row` has one slot per axis, so the lengths agree.
            let filled = index_order_into(strides, row);
            debug_assert!(filled.is_ok());
        }
    }
    compute_costs_into(strides_list, costs);
    let rows = Rows {
        pitch,
        strides: strides_list,
        elem_sizes,
        orders,
    };
    compute_blocks_at(blocks, 0, costs, &rows, BLOCK_MEMORY_SIZE);
}

/// Whether every stride is nonzero and strictly larger in magnitude than the
/// previous one.
fn strictly_increasing(strides: &[isize]) -> bool {
    strides.first().is_some_and(|&s| s != 0)
        && strides
            .windows(2)
            .all(|w| w[0].unsigned_abs() < w[1].unsigned_abs())
}

/// The tests' byte strides as element strides of a 1-byte element, which
/// produce exactly those byte magnitudes.
#[cfg(test)]
fn byte_rows(byte_strides: &[&[u128]]) -> Vec<Vec<isize>> {
    byte_strides
        .iter()
        .map(|s| {
            s.iter()
                .map(|&b| isize::try_from(b).expect("test byte strides fit isize"))
                .collect()
        })
        .collect()
}

/// [`compute_blocks_at`] over per-operand slices, for the unit tests.
#[cfg(test)]
fn compute_blocks(
    dims: &[usize],
    costs: &[isize],
    byte_strides: &[&[u128]],
    stride_orders: &[&[usize]],
    block_size: usize,
) -> InlineVec<usize> {
    let mut blocks = InlineVec::from(dims);
    if dims.is_empty() {
        return blocks;
    }
    let strides = byte_rows(byte_strides);
    let strides: Vec<&[isize]> = strides.iter().map(|s| s.as_slice()).collect();
    // The byte strides are already formed, so every element size is 1.
    let elem_sizes = vec![1usize; strides.len()];
    let flat_orders: Vec<usize> = stride_orders
        .iter()
        .flat_map(|s| s.iter().copied())
        .collect();
    let rows = Rows {
        pitch: dims.len(),
        strides: &strides,
        elem_sizes: &elem_sizes,
        orders: &flat_orders,
    };
    compute_blocks_at(&mut blocks, 0, costs, &rows, block_size);
    blocks
}

/// Recursive core of Julia's `_computeblocks` on the axes `offset..` of
/// `blocks`, which hold the full dimensions there on entry and the blocks on
/// return. The leading `offset` axes are already fixed.
///
/// # Julia equivalent
/// ```julia
/// function _computeblocks(dims, costs, bytestrides, strideorders, blocksize)
///     if totalmemoryregion(dims, bytestrides) <= blocksize
///         return dims
///     end
///     # ... reduction logic
/// end
/// ```
fn compute_blocks_at(
    blocks: &mut [usize],
    offset: usize,
    costs: &[isize],
    rows: &Rows<'_>,
    block_size: usize,
) {
    if offset == blocks.len() {
        return;
    }
    let block_size_bytes = block_size as u128;
    let region = |blocks: &[usize]| memory_region(blocks, rows.strides, offset, rows.elem_sizes);

    // If everything fits in cache, use full dims
    if region(&blocks[offset..]) <= block_size_bytes {
        return;
    }

    // Check if first dimension is smallest stride for all arrays
    let operands = rows.strides.len();
    let min_order = (0..operands)
        .filter_map(|k| rows.orders(k, offset).iter().min().copied())
        .min()
        .unwrap_or(1);
    if (0..operands).all(|k| rows.orders(k, offset).first() == Some(&min_order)) {
        // First dimension has smallest stride in all arrays
        // Keep first dimension, recurse on rest
        compute_blocks_at(blocks, offset + 1, costs, rows, block_size);
        return;
    }

    // Check if minimum stride is larger than block size
    let min_stride = rows
        .strides
        .iter()
        .zip(rows.elem_sizes)
        .filter_map(|(s, &elem_size)| {
            s[offset..]
                .iter()
                .map(|&s| s.unsigned_abs() as u128 * elem_size as u128)
                .min()
        })
        .min()
        .unwrap_or(0);
    let tail = &mut blocks[offset..];
    if min_stride > block_size_bytes {
        tail.fill(1);
        return;
    }

    // Iteratively reduce blocks
    let costs = &costs[offset..];

    // Phase 1: Halve until within 2x of target
    while region(tail) >= 2 * block_size_bytes {
        let Some(i) = last_argmax_weighted(tail, costs) else {
            break;
        };
        tail[i] = tail[i].div_ceil(2);
    }

    // Phase 2: Decrement until within target
    while region(tail) > block_size_bytes {
        let Some(i) = last_argmax_weighted(tail, costs) else {
            break;
        };
        tail[i] -= 1;
    }
}

/// Unsigned arithmetic for [`region_in`]: `u64` checks every step and gives
/// up on overflow, `u128` saturates. The `u64` result is exact when returned;
/// `u128` is exact under the shape-product precondition and saturates
/// otherwise.
trait RegionInt: Copy + Ord {
    const ZERO: Self;
    const ONE: Self;
    fn from_usize(value: usize) -> Option<Self>;
    fn add(self, other: Self) -> Option<Self>;
    fn mul(self, other: Self) -> Option<Self>;
    fn div(self, other: Self) -> Self;
}

impl RegionInt for u64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    fn from_usize(value: usize) -> Option<Self> {
        u64::try_from(value).ok()
    }
    fn add(self, other: Self) -> Option<Self> {
        self.checked_add(other)
    }
    fn mul(self, other: Self) -> Option<Self> {
        self.checked_mul(other)
    }
    fn div(self, other: Self) -> Self {
        self / other
    }
}

impl RegionInt for u128 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    fn from_usize(value: usize) -> Option<Self> {
        Some(value as u128)
    }
    fn add(self, other: Self) -> Option<Self> {
        Some(self.saturating_add(other))
    }
    fn mul(self, other: Self) -> Option<Self> {
        Some(self.saturating_mul(other))
    }
    fn div(self, other: Self) -> Self {
        self / other
    }
}

/// Compute total memory region for cache considerations.
///
/// This estimates the memory footprint considering cache line effects.
/// Strides smaller than cache line contribute to contiguous region,
/// larger strides multiply the number of cache line blocks.
///
/// # Julia equivalent
/// ```julia
/// function totalmemoryregion(dims, bytestrides)
///     memoryregion = 0
///     for i in 1:length(bytestrides)
///         strides = bytestrides[i]
///         numcontigeouscachelines = 0
///         numcachelineblocks = 1
///         for (d, s) in zip(dims, strides)
///             if s < _cachelinelength
///                 numcontigeouscachelines += (d - 1) * s
///             else
///                 numcachelineblocks *= d
///             end
///         end
///         numcontigeouscachelines = div(numcontigeouscachelines, _cachelinelength) + 1
///         memoryregion += _cachelinelength * numcontigeouscachelines * numcachelineblocks
///     end
///     return memoryregion
/// end
/// ```
///
/// Evaluated over `dims` against each operand's strides from `offset`, with
/// byte magnitudes `|s| · elem_sizes[k]`. The checked `u64` evaluation is
/// exact when it completes; otherwise the saturating `u128` one runs, which is
/// exact whenever the extents' product fits in `usize`. Both are [`region_in`].
fn memory_region(
    dims: &[usize],
    strides_list: &[&[isize]],
    offset: usize,
    elem_sizes: &[usize],
) -> u128 {
    match region_in::<u64>(dims, strides_list, offset, elem_sizes) {
        Some(region) => u128::from(region),
        None => region_in::<u128>(dims, strides_list, offset, elem_sizes).unwrap_or(u128::MAX),
    }
}

/// The region of [`memory_region`] in the arithmetic of `T`; `None` when a
/// `u64` step overflows (never for `u128`).
fn region_in<T: RegionInt>(
    dims: &[usize],
    strides_list: &[&[isize]],
    offset: usize,
    elem_sizes: &[usize],
) -> Option<T> {
    let mut memory_region = T::ZERO;
    for (strides, &elem_size) in strides_list.iter().zip(elem_sizes) {
        let region = region_of::<T>(
            dims.iter().copied().zip(strides[offset..].iter().copied()),
            elem_size,
        )?;
        memory_region = memory_region.add(region)?;
    }
    Some(memory_region)
}

/// One term of [`region_in`]: the region of one operand over its
/// `(extent, stride)` axes in iteration order.
fn region_of<T: RegionInt>(
    axes: impl Iterator<Item = (usize, isize)>,
    elem_size: usize,
) -> Option<T> {
    let cache_line = T::from_usize(CACHE_LINE_SIZE)?;
    let elem_size = T::from_usize(elem_size)?;
    let mut num_contiguous_bytes = T::ZERO;
    let mut num_cache_line_blocks = T::ONE;
    for (d, s) in axes {
        let s = T::from_usize(s.unsigned_abs())?.mul(elem_size)?;
        if s < cache_line {
            // Small stride: contributes to contiguous region. The product
            // is below 2^70, so only the sum can saturate.
            num_contiguous_bytes =
                num_contiguous_bytes.add(T::from_usize(d.saturating_sub(1))?.mul(s)?)?;
        } else {
            // Large stride: multiplies cache line blocks
            num_cache_line_blocks = num_cache_line_blocks.mul(T::from_usize(d)?)?;
        }
    }
    // Convert to cache lines
    let contiguous_lines = num_contiguous_bytes.div(cache_line).add(T::ONE)?;
    cache_line.mul(contiguous_lines)?.mul(num_cache_line_blocks)
}

/// [`memory_region`] over one operand given as `(extent, stride)` axes.
fn axes_region<I: Iterator<Item = (usize, isize)> + Clone>(axes: I, elem_size: usize) -> u128 {
    region_of::<u64>(axes.clone(), elem_size)
        .map(u128::from)
        .or_else(|| region_of::<u128>(axes, elem_size))
        .unwrap_or(u128::MAX)
}

/// Whether an operand strides are nonzero and strictly increasing in magnitude.
fn axes_increasing<I: Iterator<Item = isize>>(mut axes: I) -> bool {
    let Some(first) = axes.next() else {
        return true;
    };
    if first == 0 {
        return false;
    }
    let mut prev = first.unsigned_abs();
    axes.all(|s| {
        let m = s.unsigned_abs();
        let ok = m > prev;
        prev = m;
        ok
    })
}

/// Whether [`reduced_block_sizes`] answers `None` from the axes alone, before
/// it builds any metadata: a memory region within [`BLOCK_MEMORY_SIZE`], or
/// both operands' byte strides nonzero and strictly increasing in magnitude
/// (which needs both element sizes positive).
///
/// A caller that would have to materialize `dims` and stride slices for the
/// model asks this first, over the axes it already holds, so that an exit
/// costs no allocation. The answer is what the model itself decides.
pub(crate) fn exits_before_metadata<I>(axes: I, dst_elem_size: usize, src_elem_size: usize) -> bool
where
    I: Iterator<Item = (usize, isize, isize)> + Clone,
{
    let total =
        axes_region(axes.clone().map(|(d, dst, _)| (d, dst)), dst_elem_size).saturating_add(
            axes_region(axes.clone().map(|(d, _, src)| (d, src)), src_elem_size),
        );
    if total <= BLOCK_MEMORY_SIZE as u128 {
        return true;
    }
    dst_elem_size > 0
        && src_elem_size > 0
        && axes_increasing(axes.clone().map(|(_, dst, _)| dst))
        && axes_increasing(axes.map(|(_, _, src)| src))
}

/// [`memory_region`] over per-operand byte strides, for the unit tests.
#[cfg(test)]
fn total_memory_region(dims: &[usize], byte_strides: &[&[u128]]) -> u128 {
    let strides = byte_rows(byte_strides);
    let strides: Vec<&[isize]> = strides.iter().map(|s| s.as_slice()).collect();
    memory_region(dims, &strides, 0, &vec![1usize; strides.len()])
}

/// Find the last index with maximum (value - 1) * cost.
///
/// Julia: `_lastargmax((blocks .- 1) .* costs)`, over blocks above 1 only.
/// Costs are nonnegative (`compute_costs` gives at least 1); the score is
/// exact in `u128`.
fn last_argmax_weighted(blocks: &[usize], costs: &[isize]) -> Option<usize> {
    if blocks.is_empty() {
        return None;
    }

    let mut max_score = 0u128;
    let mut max_idx = None;

    for (i, (&b, &c)) in blocks.iter().zip(costs.iter()).enumerate() {
        if b <= 1 {
            continue;
        }
        let score = (b - 1) as u128 * c as u128;
        if score >= max_score {
            max_score = score;
            max_idx = Some(i);
        }
    }

    max_idx
}

/// Whether `blocks` reorder slabs that no visit order can make share a line,
/// so the caller keeps the flat walk instead of tiling.
///
/// The axes before the first reduced one are walked whole inside every tile,
/// so those tiles are exactly the slabs [`slabs_share_no_line`] judges, and
/// every operand has to have lines of its own for tiling to buy anything.
/// `false` when no axis is reduced (nothing to decline) and when the first
/// axis is (with an empty prefix a slab is one element, which the helper's own
/// width test answers `false` for anyway).
pub(crate) fn declines_tiles(
    dims: &[usize],
    strides_list: &[&[isize]],
    elem_sizes: &[usize],
    blocks: &[usize],
) -> bool {
    debug_assert_eq!(strides_list.len(), elem_sizes.len());
    #[cfg(feature = "bench-hooks")]
    if !crate::bench_hooks::DECLINE_TILES.load(core::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    let Some(first_cut) = (0..dims.len()).find(|&axis| blocks[axis] < dims[axis]) else {
        return false;
    };
    first_cut > 0
        && strides_list
            .iter()
            .zip(elem_sizes)
            .all(|(strides, &elem_size)| slabs_share_no_line(dims, strides, first_cut, elem_size))
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
/// only reorders slabs that share nothing. The view path is a port of that
/// function, and [`declines_tiles`] applies the exit to its answer too
/// ([#428]).
///
/// [#428]: https://github.com/tensor4all/strided-rs/issues/428
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
    // Storage by tail length: inline up to `RAW_FUSED_RANK_LIMIT`, a bounded
    // out-of-line array up to `MAX_MOVING_AXES` (the most a replay passes,
    // since it passes moving axes only), heap above it for callers that pass
    // uncompressed axes.
    let count = dims.len() - prefix;
    let fill = |tail: &mut [(u128, usize)]| {
        for (slot, axis) in tail.iter_mut().zip(prefix..dims.len()) {
            *slot = (byte(axis), axis);
        }
    };
    let separated = |tail: &mut [(u128, usize)]| tail_separated(tail, width, line, &reach);
    if count <= crate::raw_ops::RAW_FUSED_RANK_LIMIT {
        let mut inline = [(0u128, 0usize); crate::raw_ops::RAW_FUSED_RANK_LIMIT];
        let tail = &mut inline[..count];
        fill(tail);
        separated(tail)
    } else if count <= crate::raw_ops::MAX_MOVING_AXES {
        wide_tail_separated(count, &fill, &separated)
    } else {
        let mut tail = vec![(0u128, 0usize); count];
        fill(&mut tail);
        separated(&mut tail)
    }
}

/// The bounded branch of [`slabs_share_no_line`], out of line so that the
/// inline tail's frame keeps its size.
#[inline(never)]
fn wide_tail_separated(
    count: usize,
    fill: &impl Fn(&mut [(u128, usize)]),
    separated: &impl Fn(&mut [(u128, usize)]) -> bool,
) -> bool {
    let mut wide = [(0u128, 0usize); crate::raw_ops::MAX_MOVING_AXES];
    // INVARIANT: the caller dispatches here only for
    // `count <= MAX_MOVING_AXES`.
    let tail = &mut wide[..count];
    fill(tail);
    separated(tail)
}

/// The decision of [`slabs_share_no_line`] over its tail of
/// `(byte stride, axis)` entries, sorted here.
fn tail_separated(
    tail: &mut [(u128, usize)],
    width: u128,
    line: u128,
    reach: &impl Fn(usize) -> Option<u128>,
) -> bool {
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

#[cfg(test)]
#[path = "block/tests/tests.rs"]
mod tests;
