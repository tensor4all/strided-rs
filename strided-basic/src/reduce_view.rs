//! Reduce operations on dynamic-rank strided views.

#[cfg(feature = "parallel")]
use crate::kernel::same_contiguous_layout;
use crate::kernel::{
    build_plan_fused, for_each_inner_block_preordered, sequential_contiguous_layout, total_len,
};
use crate::maybe_sync::{MaybeSendSync, MaybeSync};
use crate::simd;
use crate::view::{col_major_strides, StridedArray, StridedView};
use crate::{Result, StridedError};
use std::ops::Range;
use strided_view::ElementOp;

#[cfg(feature = "parallel")]
use crate::fuse::compute_costs;
#[cfg(feature = "parallel")]
use crate::threading::{
    for_each_inner_block_with_offsets, mapreduce_threaded, SendPtr, MINTHREADLENGTH,
};

/// Number of independent accumulators of [`fold_contiguous`].
const FOLD_LANES: usize = 16;

/// Minimum outputs along a unit-stride kept axis for the column sweep of
/// [`reduce_axis`].
const AXIS_SWEEP_MIN_LEN: usize = 16;

/// Byte budget of one output block of the column sweep of [`reduce_axis`].
const AXIS_SWEEP_BLOCK_BYTES: usize = 64 * 1024;

/// Folds a contiguous run with [`FOLD_LANES`] independent accumulators.
///
/// Lane `i` starts from element `i` and folds elements `i + FOLD_LANES`,
/// `i + 2 * FOLD_LANES`, ... in order; the remainder is folded into the first
/// lanes. `init` is combined exactly once, before lane 0, and the lanes are
/// then combined left to right. Independent lanes let the compiler keep
/// several vector accumulators in flight instead of one serial dependency
/// chain, which is what makes a closure-based sum or max approach the speed
/// of a hand written lane loop.
#[inline(always)]
fn fold_contiguous<T, Op, M, R, U>(src: &[T], init: U, map_fn: &M, reduce_fn: &R) -> U
where
    T: Copy,
    Op: ElementOp<T>,
    M: Fn(T) -> U,
    R: Fn(U, U) -> U,
    U: Clone,
{
    if src.len() < 2 * FOLD_LANES {
        let mut acc = init;
        for &value in src {
            acc = reduce_fn(acc, map_fn(Op::apply(value)));
        }
        return acc;
    }
    let (head, rest) = src.split_at(FOLD_LANES);
    let mut lanes: [U; FOLD_LANES] = core::array::from_fn(|lane| map_fn(Op::apply(head[lane])));
    let mut chunks = rest.chunks_exact(FOLD_LANES);
    for chunk in chunks.by_ref() {
        for (lane, &value) in lanes.iter_mut().zip(chunk) {
            *lane = reduce_fn(lane.clone(), map_fn(Op::apply(value)));
        }
    }
    for (lane, &value) in lanes.iter_mut().zip(chunks.remainder()) {
        *lane = reduce_fn(lane.clone(), map_fn(Op::apply(value)));
    }
    lanes.into_iter().fold(init, reduce_fn)
}

/// Folds `len` elements starting at `ptr` with step `stride`.
///
/// # Safety
/// `ptr.offset(i * stride)` must be a readable element for every `i < len`.
#[inline(always)]
unsafe fn fold_run<T, Op, M, R, U>(
    ptr: *const T,
    stride: isize,
    len: usize,
    init: U,
    map_fn: &M,
    reduce_fn: &R,
) -> U
where
    T: Copy,
    Op: ElementOp<T>,
    M: Fn(T) -> U,
    R: Fn(U, U) -> U,
    U: Clone,
{
    if stride == 1 {
        // SAFETY: the caller proves `len` readable unit-stride elements.
        let run = unsafe { std::slice::from_raw_parts(ptr, len) };
        return fold_contiguous::<T, Op, M, R, U>(run, init, map_fn, reduce_fn);
    }
    let mut acc = init;
    let mut cursor = ptr;
    for index in 0..len {
        // SAFETY: the caller proves every element of the run is readable.
        acc = reduce_fn(acc, map_fn(Op::apply(unsafe { *cursor })));
        if index + 1 < len {
            cursor = cursor.wrapping_offset(stride);
        }
    }
    acc
}

/// Full reduction with map function: `reduce(init, op, map.(src))`.
///
/// # Evaluation order
///
/// `reduce_fn` must be associative and commutative, and `init` must be an
/// identity of `reduce_fn` whenever the reduction runs on several threads.
/// A contiguous source is folded with sixteen independent lanes (see the
/// crate source for the exact lane assignment), non-contiguous sources are
/// traversed in a cache-friendly loop order, and parallel execution folds
/// chunks independently and combines them in chunk order. Floating point
/// results can therefore differ from a strict left to right fold in the last
/// bits; for a fixed shape, layout and thread count the result is repeatable.
///
/// Because `map_fn` and `reduce_fn` are opaque closures this entry cannot
/// select the dtype-specific SIMD kernels of [`crate::ErasedReducePlan`];
/// callers that reduce with a known operation (sum, product, max, min) on a
/// supported dtype can use that plan for the fastest path.
pub fn reduce<T: Copy + MaybeSendSync, Op: ElementOp<T>, M, R, U>(
    src: &StridedView<T, Op>,
    map_fn: M,
    reduce_fn: R,
    init: U,
) -> Result<U>
where
    M: Fn(T) -> U + MaybeSync,
    R: Fn(U, U) -> U + MaybeSync,
    U: Clone + MaybeSendSync,
{
    reduce_impl(src, map_fn, reduce_fn, init)
}

fn reduce_impl<T: Copy + MaybeSendSync, Op: ElementOp<T>, M, R, U>(
    src: &StridedView<T, Op>,
    map_fn: M,
    reduce_fn: R,
    init: U,
) -> Result<U>
where
    M: Fn(T) -> U + MaybeSync,
    R: Fn(U, U) -> U + MaybeSync,
    U: Clone + MaybeSendSync,
{
    let src_ptr = src.ptr();
    let src_dims = src.dims();
    let src_strides = src.strides();

    let contiguous = sequential_contiguous_layout(src_dims, &[src_strides])?;
    if contiguous.is_some() {
        let len = total_len(src_dims)?;
        let src = unsafe { std::slice::from_raw_parts(src_ptr, len) };
        return Ok(simd::dispatch_if_large(len, || {
            fold_contiguous::<T, Op, M, R, U>(src, init, &map_fn, &reduce_fn)
        }));
    }

    // Parallel contiguous fast path: split into scheduler chunks with slice-based iteration.
    // This enables LLVM auto-vectorization on each chunk, unlike the general threaded path
    // which uses scalar pointer-offset loops.
    #[cfg(feature = "parallel")]
    {
        let total = total_len(src_dims)?;
        let nthreads = crate::execution_policy::rayon_threads();
        if total > MINTHREADLENGTH
            && nthreads > 1
            && same_contiguous_layout(src_dims, &[src_strides]).is_some()
        {
            let src_slice = unsafe { std::slice::from_raw_parts(src_ptr, total) };
            let result = crate::threading::parallel_map_reduce(
                0..total,
                nthreads,
                &|range| {
                    simd::dispatch_if_large(range.len(), || {
                        fold_contiguous::<T, Op, M, R, U>(
                            &src_slice[range],
                            init.clone(),
                            &map_fn,
                            &reduce_fn,
                        )
                    })
                },
                &|a, b| reduce_fn(a, b),
            );
            return Ok(result);
        }
    }

    let strides_list: [&[isize]; 1] = [src_strides];

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(src_dims, &strides_list, None, std::mem::size_of::<T>());

    #[cfg(feature = "parallel")]
    {
        let total = total_len(&fused_dims)?;
        let nthreads = crate::execution_policy::rayon_threads();
        if total > MINTHREADLENGTH && nthreads > 1 {
            // False sharing avoidance: space output slots by cache line size
            let spacing = (64 / std::mem::size_of::<U>()).max(1);
            let mut threadedout = vec![init.clone(); spacing * nthreads];
            let threadedout_ptr = SendPtr(threadedout.as_mut_ptr());
            let src_send = SendPtr(src_ptr as *mut T);

            let costs = compute_costs(&ordered_strides);

            // For complete reduction, strides_list has 2 entries:
            // [0] = threadedout (stride 0 everywhere — broadcasting), [1] = src
            // The spacing/taskindex mechanism addresses output slots.
            let ndim = fused_dims.len();
            let mut threaded_strides = Vec::with_capacity(ordered_strides.len() + 1);
            threaded_strides.push(vec![0isize; ndim]); // threadedout: stride 0 (broadcast)
            for s in &ordered_strides {
                threaded_strides.push(s.clone());
            }
            let initial_offsets = vec![0isize; threaded_strides.len()];

            // Mask costs for threadedout stride=0 dims (all dims, since it's fully broadcast)
            // This means: do NOT split on dims where output stride is 0 — but for complete
            // reduction, ALL output strides are 0, so costs would all be masked to 0.
            // Julia handles this with the spacing mechanism: each task writes to its own slot.
            // We keep costs unmasked so splitting still works.

            mapreduce_threaded(
                &fused_dims,
                &plan.block,
                &threaded_strides,
                &initial_offsets,
                &costs,
                nthreads,
                spacing as isize,
                1,
                &|dims, blocks, strides_list, offsets| {
                    // offsets[0] = spacing * (taskindex - 1) for threadedout
                    // offsets[1] = offset into src
                    let out_offset = offsets[0] as usize;
                    let src_offsets = &offsets[1..];

                    for_each_inner_block_with_offsets(
                        dims,
                        blocks,
                        &strides_list[1..],
                        src_offsets,
                        |offsets, len, strides| {
                            let mut ptr = unsafe { src_send.as_const().offset(offsets[0]) };
                            let stride = strides[0];
                            let slot = unsafe { &mut *threadedout_ptr.as_ptr().add(out_offset) };
                            for _ in 0..len {
                                let val = Op::apply(unsafe { *ptr });
                                let mapped = map_fn(val);
                                *slot = reduce_fn(slot.clone(), mapped);
                                unsafe {
                                    ptr = ptr.offset(stride);
                                }
                            }
                            Ok(())
                        },
                    )
                },
            )?;

            // Merge thread-local results
            let mut result = init;
            for i in 0..nthreads {
                result = reduce_fn(result, threadedout[i * spacing].clone());
            }
            return Ok(result);
        }
    }

    let mut acc = init;
    let initial_offsets = vec![0isize; ordered_strides.len()];
    for_each_inner_block_preordered(
        &fused_dims,
        &plan.block,
        &ordered_strides,
        &initial_offsets,
        |offsets, len, strides| {
            let mut ptr = unsafe { src_ptr.offset(offsets[0]) };
            let stride = strides[0];
            for _ in 0..len {
                let val = Op::apply(unsafe { *ptr });
                let mapped = map_fn(val);
                acc = reduce_fn(acc.clone(), mapped);
                unsafe {
                    ptr = ptr.offset(stride);
                }
            }
            Ok(())
        },
    )?;

    Ok(acc)
}

/// Reduce along a single axis, returning a new StridedArray.
///
/// Every output folds `init` once and then its reduced elements. When the
/// reduced axis has unit stride those elements are folded with the lane
/// scheme of [`reduce`], so `reduce_fn` must be associative and commutative.
/// When the leading kept axis has unit stride, a block of adjacent outputs
/// is accumulated one contiguous source column at a time, which keeps the
/// left to right order per output. With the `parallel` feature, outputs are
/// split across threads once the number of source elements read exceeds the
/// threading threshold; each output is computed by one thread, so the result
/// does not depend on the thread count.
pub fn reduce_axis<T: Copy + MaybeSendSync, Op: ElementOp<T>, M, R, U>(
    src: &StridedView<T, Op>,
    axis: usize,
    map_fn: M,
    reduce_fn: R,
    init: U,
) -> Result<StridedArray<U>>
where
    M: Fn(T) -> U + MaybeSync,
    R: Fn(U, U) -> U + MaybeSync,
    U: Clone + MaybeSendSync,
{
    let rank = src.ndim();
    if axis >= rank {
        return Err(StridedError::InvalidAxis { axis, rank });
    }

    let src_dims = src.dims();
    let src_strides = src.strides();
    let src_ptr = src.ptr();
    // Reject an element count beyond usize (a huge stride-0 broadcast) up
    // front instead of allocating and looping over it.
    total_len(src_dims)?;

    let axis_len = src_dims[axis];
    let axis_stride = src_strides[axis];

    let kept: Vec<(usize, isize)> = src_dims
        .iter()
        .zip(src_strides)
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, (&d, &s))| (d, s))
        .collect();
    let out_dims: Vec<usize> = kept.iter().map(|&(d, _)| d).collect();

    if out_dims.is_empty() {
        // SAFETY: the view proves `axis_len` readable elements along `axis`.
        let acc = unsafe {
            fold_run::<T, Op, M, R, U>(src_ptr, axis_stride, axis_len, init, &map_fn, &reduce_fn)
        };
        let strides = col_major_strides(&[1]);
        return StridedArray::from_parts(vec![acc], &[1], &strides, 0);
    }

    let total_out = total_len(&out_dims)?;
    let out_strides = col_major_strides(&out_dims);
    let mut out =
        StridedArray::from_parts(vec![init.clone(); total_out], &out_dims, &out_strides, 0)?;
    if total_out == 0 || axis_len == 0 {
        return Ok(out);
    }
    let out_ptr = out.view_mut().as_mut_ptr();

    let parts = AxisParts {
        kept: &kept,
        axis_len,
        axis_stride,
        map_fn: &map_fn,
        reduce_fn: &reduce_fn,
        init: &init,
    };

    #[cfg(feature = "parallel")]
    {
        let work = total_out.saturating_mul(axis_len);
        let nthreads = crate::threading::parallel_threads_for_len(work).min(total_out);
        if nthreads > 1 {
            let src_send = SendPtr(src_ptr as *mut T);
            let out_send = SendPtr(out_ptr);
            let parts = &parts;
            crate::threading::parallel_for_each(0..total_out, nthreads, &|range| {
                // SAFETY: worker ranges are disjoint output ranges of the
                // freshly allocated column-major output, and the view proves
                // every source offset reachable from its kept coordinates.
                unsafe {
                    reduce_axis_range::<T, Op, M, R, U>(
                        src_send.as_const(),
                        out_send.as_ptr(),
                        range,
                        parts,
                    )
                }
            });
            return Ok(out);
        }
    }

    // SAFETY: as for the parallel ranges, with one range covering all outputs.
    unsafe { reduce_axis_range::<T, Op, M, R, U>(src_ptr, out_ptr, 0..total_out, &parts) };
    Ok(out)
}

/// Shared inputs of [`reduce_axis_range`].
struct AxisParts<'a, M, R, U> {
    /// `(extent, source stride)` of every kept axis, in output order.
    kept: &'a [(usize, isize)],
    axis_len: usize,
    axis_stride: isize,
    map_fn: &'a M,
    reduce_fn: &'a R,
    init: &'a U,
}

/// Computes outputs `range` (column-major output indices) of [`reduce_axis`].
///
/// # Safety
/// `out` must be the column-major output of `parts.kept` extents, `range` a
/// subrange of it that no other thread writes, and every source offset
/// reachable from a kept coordinate plus `k * parts.axis_stride` for
/// `k < parts.axis_len` must be a readable element of `src`.
unsafe fn reduce_axis_range<T, Op, M, R, U>(
    src: *const T,
    out: *mut U,
    range: Range<usize>,
    parts: &AxisParts<'_, M, R, U>,
) where
    T: Copy,
    Op: ElementOp<T>,
    M: Fn(T) -> U,
    R: Fn(U, U) -> U,
    U: Clone,
{
    let kept = parts.kept;
    let (lead_extent, lead_stride) = kept[0];
    let sweep = lead_stride == 1 && lead_extent >= AXIS_SWEEP_MIN_LEN && parts.axis_len > 1;
    let block = (AXIS_SWEEP_BLOCK_BYTES / std::mem::size_of::<U>().max(1)).max(AXIS_SWEEP_MIN_LEN);

    // Decode the first output once, then advance incrementally.
    let mut coords = vec![0usize; kept.len()];
    let mut rest = range.start;
    let mut src_off = 0isize;
    for (coord, &(extent, stride)) in coords.iter_mut().zip(kept) {
        *coord = rest % extent;
        rest /= extent;
        src_off += *coord as isize * stride;
    }

    let mut output = range.start;
    while output < range.end {
        let len = if sweep {
            (lead_extent - coords[0]).min(range.end - output).min(block)
        } else {
            1
        };
        if sweep {
            // SAFETY: the `len` outputs are inside the caller's range and
            // their unit-stride source columns are readable.
            unsafe { sweep_block::<T, Op, M, R, U>(src, src_off, out.add(output), len, parts) };
        } else {
            // SAFETY: one reachable output and its reduced run.
            unsafe {
                let acc = fold_run::<T, Op, M, R, U>(
                    src.offset(src_off),
                    parts.axis_stride,
                    parts.axis_len,
                    parts.init.clone(),
                    parts.map_fn,
                    parts.reduce_fn,
                );
                *out.add(output) = acc;
            }
        }
        output += len;
        if output >= range.end {
            break;
        }
        // Advance the kept coordinates by `len` along the leading axis; a
        // block never crosses the end of the leading axis.
        coords[0] += len;
        src_off += len as isize * lead_stride;
        let mut axis = 0;
        while coords[axis] == kept[axis].0 {
            src_off -= kept[axis].0 as isize * kept[axis].1;
            coords[axis] = 0;
            axis += 1;
            coords[axis] += 1;
            src_off += kept[axis].1;
        }
    }
}

/// Accumulates `len` adjacent outputs, one contiguous source column per
/// reduced index, keeping the left to right order per output.
///
/// # Safety
/// `out` must be `len` writable initialized outputs and `src + src_off +
/// k * parts.axis_stride` must start `len` readable elements for every
/// `k < parts.axis_len`.
#[inline(always)]
unsafe fn sweep_block<T, Op, M, R, U>(
    src: *const T,
    src_off: isize,
    out: *mut U,
    len: usize,
    parts: &AxisParts<'_, M, R, U>,
) where
    T: Copy,
    Op: ElementOp<T>,
    M: Fn(T) -> U,
    R: Fn(U, U) -> U,
    U: Clone,
{
    let map_fn = parts.map_fn;
    let reduce_fn = parts.reduce_fn;
    // SAFETY: forwarded caller contract.
    let out = unsafe { std::slice::from_raw_parts_mut(out, len) };
    let column = |k: usize| {
        // SAFETY: forwarded caller contract for reduced index `k`.
        unsafe {
            std::slice::from_raw_parts(src.offset(src_off + k as isize * parts.axis_stride), len)
        }
    };
    let first = column(0);
    for (slot, &value) in out.iter_mut().zip(first) {
        *slot = reduce_fn(parts.init.clone(), map_fn(Op::apply(value)));
    }
    let mut k = 1;
    while k + 4 <= parts.axis_len {
        let (c0, c1, c2, c3) = (column(k), column(k + 1), column(k + 2), column(k + 3));
        for (index, slot) in out.iter_mut().enumerate() {
            let mut acc = reduce_fn(slot.clone(), map_fn(Op::apply(c0[index])));
            acc = reduce_fn(acc, map_fn(Op::apply(c1[index])));
            acc = reduce_fn(acc, map_fn(Op::apply(c2[index])));
            acc = reduce_fn(acc, map_fn(Op::apply(c3[index])));
            *slot = acc;
        }
        k += 4;
    }
    while k < parts.axis_len {
        let values = column(k);
        for (slot, &value) in out.iter_mut().zip(values) {
            *slot = reduce_fn(slot.clone(), map_fn(Op::apply(value)));
        }
        k += 1;
    }
}
