//! Per-op reduction kernels for [`ErasedReducePlan`](super::ErasedReducePlan).
//!
//! The runtime [`ReduceOp`](super::ReduceOp) is matched once per plan
//! execution and selects one zero-sized kernel type. Every tensor-sized loop
//! below is generic over that kernel type, so the op never appears inside a
//! loop body and each loop is compiled once per `(dtype, op)` pair.
//!
//! Serial execution and every parallel worker range call the same range
//! functions ([`reduce_full_range`] and [`reduce_axes_range`]), so a
//! parallel chunk runs exactly the lane or SIMD kernel the serial path runs.

use super::{
    checked_reduce_reset, compress_reduce_outer_axes, ErasedReduceScalar, ReduceInnerAxis,
    ReduceInnerCursor, ReduceOuterAxis, ReduceOuterCursor,
};
use crate::{Result, StridedError};
use core::ops::Range;

/// Independent accumulators used by the non SIMD lane kernels.
pub(super) const REDUCE_LANES: usize = 8;

/// Minimum length of a unit-stride leading reduced run that switches an axis
/// reduction from the sequential fold to the contiguous run kernel.
pub(super) const AXIS_RUN_KERNEL_MIN_LEN: usize = 2 * REDUCE_LANES;

/// Minimum extent of a unit-stride leading kept axis that selects the
/// output block kernel.
pub(super) const AXIS_OUTPUT_BLOCK_MIN_LEN: usize = 16;

/// Maximum number of adjacent outputs the output block kernel accumulates at
/// once. The accumulator block stays in L1 (8 KiB for `Complex64`) while
/// each reduced step streams one contiguous source run of the same length.
pub(super) const AXIS_OUTPUT_BLOCK: usize = 512;

/// One reduction operation, fixed at compile time.
pub(super) trait ReduceKernel<T: ErasedReduceScalar>: 'static {
    fn identity() -> T;
    fn map(value: T) -> T;
    fn combine(lhs: T, rhs: T) -> T;

    /// Reduces a contiguous run.
    ///
    /// Full reductions, parallel full reduction chunks and qualifying axis
    /// runs all call this one function.
    #[inline(always)]
    fn contiguous(values: &[T]) -> T {
        lanes_contiguous::<T, Self>(values)
    }
}

pub(super) struct SumKernel;
pub(super) struct ProductKernel;
pub(super) struct SumSquaresKernel;
pub(super) struct MaxKernel;
pub(super) struct MinKernel;

impl<T: ErasedReduceScalar> ReduceKernel<T> for SumKernel {
    #[inline(always)]
    fn identity() -> T {
        T::zero()
    }
    #[inline(always)]
    fn map(value: T) -> T {
        value
    }
    #[inline(always)]
    fn combine(lhs: T, rhs: T) -> T {
        T::reduce_sum(lhs, rhs)
    }
    #[inline(always)]
    fn contiguous(values: &[T]) -> T {
        T::try_simd_sum(values).unwrap_or_else(|| lanes_contiguous::<T, Self>(values))
    }
}

impl<T: ErasedReduceScalar> ReduceKernel<T> for ProductKernel {
    #[inline(always)]
    fn identity() -> T {
        T::one()
    }
    #[inline(always)]
    fn map(value: T) -> T {
        value
    }
    #[inline(always)]
    fn combine(lhs: T, rhs: T) -> T {
        T::reduce_product(lhs, rhs)
    }
    #[inline(always)]
    fn contiguous(values: &[T]) -> T {
        T::try_simd_product(values).unwrap_or_else(|| lanes_contiguous::<T, Self>(values))
    }
}

impl<T: ErasedReduceScalar> ReduceKernel<T> for SumSquaresKernel {
    #[inline(always)]
    fn identity() -> T {
        T::zero()
    }
    #[inline(always)]
    fn map(value: T) -> T {
        T::reduce_product(value, value)
    }
    #[inline(always)]
    fn combine(lhs: T, rhs: T) -> T {
        T::reduce_sum(lhs, rhs)
    }
    #[inline(always)]
    fn contiguous(values: &[T]) -> T {
        T::try_simd_sum_squares(values).unwrap_or_else(|| lanes_contiguous::<T, Self>(values))
    }
}

impl<T: ErasedReduceScalar> ReduceKernel<T> for MaxKernel {
    #[inline(always)]
    fn identity() -> T {
        T::max_identity()
    }
    #[inline(always)]
    fn map(value: T) -> T {
        value
    }
    #[inline(always)]
    fn combine(lhs: T, rhs: T) -> T {
        T::reduce_max(lhs, rhs)
    }
}

impl<T: ErasedReduceScalar> ReduceKernel<T> for MinKernel {
    #[inline(always)]
    fn identity() -> T {
        T::min_identity()
    }
    #[inline(always)]
    fn map(value: T) -> T {
        value
    }
    #[inline(always)]
    fn combine(lhs: T, rhs: T) -> T {
        T::reduce_min(lhs, rhs)
    }
}

/// Multi-accumulator kernel over a contiguous run.
///
/// Lane `i` folds elements `i, i + 8, ...` in order, the remainder is folded
/// into the first lanes, and the lanes are folded left to right.
#[inline(always)]
fn lanes_contiguous<T, K>(values: &[T]) -> T
where
    T: ErasedReduceScalar,
    K: ReduceKernel<T> + ?Sized,
{
    crate::simd::dispatch_if_large(values.len(), || {
        let mut lanes = [K::identity(); REDUCE_LANES];
        let mut chunks = values.chunks_exact(REDUCE_LANES);
        for chunk in chunks.by_ref() {
            for (lane, &value) in lanes.iter_mut().zip(chunk) {
                *lane = K::combine(*lane, K::map(value));
            }
        }
        for (lane, &value) in lanes.iter_mut().zip(chunks.remainder()) {
            *lane = K::combine(*lane, K::map(value));
        }
        lanes.into_iter().fold(K::identity(), K::combine)
    })
}

/// Multi-accumulator kernel over a strided run, with the lane assignment of
/// [`lanes_contiguous`].
///
/// # Safety
/// `ptr.offset(i * stride)` must be a readable element for every `i < len`.
#[inline(always)]
unsafe fn lanes_strided<T, K>(ptr: *const T, stride: isize, len: usize) -> T
where
    T: ErasedReduceScalar,
    K: ReduceKernel<T>,
{
    let mut lanes = [K::identity(); REDUCE_LANES];
    let mut cursor = ptr;
    for _ in 0..len / REDUCE_LANES {
        let mut lane_ptr = cursor;
        for lane in &mut lanes {
            // SAFETY: the caller proves every element of the run is readable.
            *lane = K::combine(*lane, K::map(unsafe { *lane_ptr }));
            lane_ptr = lane_ptr.wrapping_offset(stride);
        }
        cursor = lane_ptr;
    }
    for lane in lanes.iter_mut().take(len % REDUCE_LANES) {
        // SAFETY: the caller proves every element of the run is readable.
        *lane = K::combine(*lane, K::map(unsafe { *cursor }));
        cursor = cursor.wrapping_offset(stride);
    }
    lanes.into_iter().fold(K::identity(), K::combine)
}

/// Reduces one run of `len` elements starting at `ptr`.
///
/// # Safety
/// `ptr.offset(i * stride)` must be a readable element for every `i < len`.
#[inline(always)]
unsafe fn reduce_run<T, K>(ptr: *const T, stride: isize, len: usize) -> T
where
    T: ErasedReduceScalar,
    K: ReduceKernel<T>,
{
    if stride == 1 {
        // SAFETY: a unit-stride readable run is a valid shared slice.
        K::contiguous(unsafe { core::slice::from_raw_parts(ptr, len) })
    } else {
        // SAFETY: forwarded caller contract.
        unsafe { lanes_strided::<T, K>(ptr, stride, len) }
    }
}

/// Precomputed traversal of a full reduction.
///
/// Source axes of extent one are dropped, the rest are ordered by increasing
/// absolute stride (stride zero last) and fused where they are contiguous.
/// The first fused axis is the run axis; the others step between runs.
#[derive(Clone, Debug)]
pub(super) struct FullTraversal {
    total: usize,
    run_len: usize,
    run_stride: isize,
    run_axes: Vec<ReduceOuterAxis>,
}

impl FullTraversal {
    pub(super) fn compile(dims: &[usize], strides: &[isize]) -> Result<Self> {
        if dims.len() != strides.len() {
            return Err(StridedError::StrideLengthMismatch);
        }
        let total = dims
            .iter()
            .try_fold(1usize, |total, &dim| total.checked_mul(dim))
            .ok_or(StridedError::OffsetOverflow)?;
        if total == 0 {
            return Ok(Self {
                total,
                run_len: 0,
                run_stride: 1,
                run_axes: Vec::new(),
            });
        }
        let mut axes: Vec<(usize, isize)> = dims
            .iter()
            .copied()
            .zip(strides.iter().copied())
            .filter(|&(extent, _)| extent != 1)
            .collect();
        axes.sort_by_key(|&(_, stride)| {
            if stride == 0 {
                usize::MAX
            } else {
                stride.unsigned_abs()
            }
        });
        let mut fused = compress_reduce_outer_axes(
            axes.into_iter()
                .map(|(extent, stride)| {
                    Ok(ReduceOuterAxis {
                        extent,
                        source_step: stride,
                        source_reset: checked_reduce_reset(extent, stride)?,
                        dest_step: 0,
                        dest_reset: 0,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        )?;
        if fused.is_empty() {
            return Ok(Self {
                total,
                run_len: 1,
                run_stride: 1,
                run_axes: fused,
            });
        }
        let run = fused.remove(0);
        Ok(Self {
            total,
            run_len: run.extent,
            run_stride: run.source_step,
            run_axes: fused,
        })
    }

    #[inline]
    pub(super) fn total(&self) -> usize {
        self.total
    }
}

/// Reduces traversal positions `range` of a full reduction.
///
/// # Safety
/// `source.offset(source_base + o)` must be readable for every offset `o`
/// reachable by `traversal` (proven by descriptor validation plus exact plan
/// layout equality), and `range` must be a non-empty subrange of
/// `0..traversal.total()`.
pub(super) unsafe fn reduce_full_range<T, K>(
    source: *const T,
    source_base: isize,
    traversal: &FullTraversal,
    range: Range<usize>,
) -> Result<T>
where
    T: ErasedReduceScalar,
    K: ReduceKernel<T>,
{
    debug_assert!(!range.is_empty() && range.end <= traversal.total);
    let run_len = traversal.run_len;
    let run_stride = traversal.run_stride;
    let mut col = range.start % run_len;
    // Decode once per worker range, then advance incrementally.
    let mut runs =
        ReduceOuterCursor::decode(range.start / run_len, source_base, 0, &traversal.run_axes)?;
    let mut remaining = range.len();
    let mut acc = None;
    loop {
        let len = (run_len - col).min(remaining);
        // INVARIANT: `col + len <= run_len` and the run start is a reachable
        // source offset, so every element of this run is readable.
        let start = runs.source_offset + col as isize * run_stride;
        // SAFETY: see the invariant above and the function contract.
        let partial = unsafe { reduce_run::<T, K>(source.offset(start), run_stride, len) };
        acc = Some(match acc {
            Some(acc) => K::combine(acc, partial),
            None => partial,
        });
        remaining -= len;
        if remaining == 0 {
            break;
        }
        col = 0;
        runs.advance();
    }
    Ok(acc.unwrap_or_else(K::identity))
}

/// Strategy for one axis reduction layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AxesStrategy {
    /// The leading reduced run has unit source stride: reduce each run with
    /// [`ReduceKernel::contiguous`] and fold the runs in order.
    ContiguousRuns,
    /// The leading kept axis has unit source stride: accumulate a block of
    /// adjacent outputs from one contiguous source run per reduced element
    /// (an axpy style column sweep), each output in sequential reduction
    /// order.
    OutputBlocks,
    /// Sequential fold in caller-supplied reduced-axis order.
    Sequential,
}

fn axes_strategy(outer_axes: &[ReduceOuterAxis], inner_axes: &[ReduceInnerAxis]) -> AxesStrategy {
    if inner_axes
        .first()
        .is_some_and(|axis| axis.source_step == 1 && axis.extent >= AXIS_RUN_KERNEL_MIN_LEN)
    {
        AxesStrategy::ContiguousRuns
    } else if outer_axes
        .first()
        .is_some_and(|axis| axis.source_step == 1 && axis.extent >= AXIS_OUTPUT_BLOCK_MIN_LEN)
    {
        AxesStrategy::OutputBlocks
    } else {
        AxesStrategy::Sequential
    }
}

/// Source and destination description of one axis reduction execution.
pub(super) struct AxesRange<'a, T> {
    pub(super) source: *const T,
    pub(super) source_base: isize,
    pub(super) dest: *mut T,
    pub(super) dest_base: isize,
    pub(super) outer_axes: &'a [ReduceOuterAxis],
    pub(super) inner_axes: &'a [ReduceInnerAxis],
    pub(super) reduce_total: usize,
}

/// Reduces outputs `range` of an axis reduction.
///
/// Every output is computed independently from its own reduced elements, so
/// the result for an output does not depend on how outputs are split across
/// worker ranges.
///
/// # Safety
/// `parts.source` and `parts.dest` must point to `T` storage whose reachable
/// offsets from the bases were validated against the plan layouts, the
/// destination must not overlap the source, `parts.reduce_total` must be
/// nonzero and `range` must be a subrange of the output domain.
pub(super) unsafe fn reduce_axes_range<T, K>(
    parts: AxesRange<'_, T>,
    range: Range<usize>,
) -> Result<()>
where
    T: ErasedReduceScalar,
    K: ReduceKernel<T>,
{
    if range.is_empty() {
        return Ok(());
    }
    let source = parts.source;
    let dest = parts.dest;
    let reduce_total = parts.reduce_total;
    debug_assert!(reduce_total != 0);
    // Decode once per worker range, then advance incrementally.
    let mut outer = ReduceOuterCursor::decode(
        range.start,
        parts.source_base,
        parts.dest_base,
        parts.outer_axes,
    )?;
    // INVARIANT: (1) compile_axes checked signed source/destination spans
    // and every cursor step/reset, including -(extent-1)*stride; (2) raw
    // descriptors validated every reachable offset; (3) execute checked exact
    // plan-layout equality before dispatch. Every pointer formed below is a
    // reachable cursor offset under that chain.
    match axes_strategy(parts.outer_axes, parts.inner_axes) {
        AxesStrategy::ContiguousRuns => {
            let run_len = parts.inner_axes[0].extent;
            let run_count = reduce_total / run_len;
            let mut runs = ReduceInnerCursor::new(0, &parts.inner_axes[1..]);
            for output in range.clone() {
                runs.reset(outer.source_offset);
                // SAFETY: each run is `run_len` reachable unit-stride elements.
                let mut acc = K::contiguous(unsafe {
                    core::slice::from_raw_parts(source.offset(runs.source_offset), run_len)
                });
                for _ in 1..run_count {
                    runs.advance();
                    // SAFETY: as above, for the next reachable run.
                    let partial = K::contiguous(unsafe {
                        core::slice::from_raw_parts(source.offset(runs.source_offset), run_len)
                    });
                    acc = K::combine(acc, partial);
                }
                // SAFETY: the destination cursor offset is reachable.
                unsafe { dest.offset(outer.dest_offset).write(acc) };
                if output + 1 < range.end {
                    outer.advance();
                }
            }
        }
        AxesStrategy::OutputBlocks => {
            let block_axis = parts.outer_axes[0];
            let mut inner = ReduceInnerCursor::new(0, parts.inner_axes);
            let mut acc = [K::identity(); AXIS_OUTPUT_BLOCK];
            let mut output = range.start;
            while output < range.end {
                // A block stays inside one run of the leading kept axis and
                // inside the worker range.
                let block_room = block_axis.extent - outer.leading_coord();
                let len = block_room.min(range.end - output).min(AXIS_OUTPUT_BLOCK);
                let acc = &mut acc[..len];
                acc.fill(K::identity());
                inner.reset(outer.source_offset);
                for value_index in 0..reduce_total {
                    // SAFETY: the `len` adjacent outputs of this block read
                    // `len` reachable unit-stride source elements along the
                    // leading kept axis at every reduced position.
                    let values = unsafe {
                        core::slice::from_raw_parts(source.offset(inner.source_offset), len)
                    };
                    for (slot, &value) in acc.iter_mut().zip(values) {
                        *slot = K::combine(*slot, K::map(value));
                    }
                    if value_index + 1 < reduce_total {
                        inner.advance();
                    }
                }
                if block_axis.dest_step == 1 {
                    // SAFETY: the block outputs are `len` reachable unit-stride
                    // destination elements, disjoint from the source.
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            acc.as_ptr(),
                            dest.offset(outer.dest_offset),
                            len,
                        )
                    };
                } else {
                    for (index, &value) in acc.iter().enumerate() {
                        let offset = outer.dest_offset + index as isize * block_axis.dest_step;
                        // SAFETY: the block outputs are reachable destination offsets.
                        unsafe { dest.offset(offset).write(value) };
                    }
                }
                for _ in 0..len {
                    output += 1;
                    if output < range.end {
                        outer.advance();
                    }
                }
            }
        }
        AxesStrategy::Sequential => {
            let mut inner = ReduceInnerCursor::new(0, parts.inner_axes);
            for output in range.clone() {
                inner.reset(outer.source_offset);
                // SAFETY: forwarded layout invariant.
                let acc = unsafe { sequential_fold::<T, K>(source, &mut inner, reduce_total) };
                // SAFETY: the destination cursor offset is reachable.
                unsafe { dest.offset(outer.dest_offset).write(acc) };
                if output + 1 < range.end {
                    outer.advance();
                }
            }
        }
    }
    Ok(())
}

/// Folds `count` elements visited by `inner` in cursor order.
///
/// # Safety
/// Every cursor offset visited must be a readable element of `source`.
#[inline(always)]
unsafe fn sequential_fold<T, K>(
    source: *const T,
    inner: &mut ReduceInnerCursor<'_>,
    count: usize,
) -> T
where
    T: ErasedReduceScalar,
    K: ReduceKernel<T>,
{
    let mut acc = K::identity();
    for value_index in 0..count {
        // SAFETY: forwarded caller contract.
        let value = unsafe { *source.offset(inner.source_offset) };
        acc = K::combine(acc, K::map(value));
        if value_index + 1 < count {
            inner.advance();
        }
    }
    acc
}
