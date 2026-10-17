//! Reduce operations on dynamic-rank strided views.

#[cfg(feature = "parallel")]
use crate::kernel::same_contiguous_layout;
use crate::kernel::{
    build_plan_fused, for_each_inner_block_preordered, sequential_contiguous_layout, total_len,
};
use crate::maybe_sync::{MaybeSendSync, MaybeSync};
use crate::simd;
use crate::view::{col_major_strides, StridedArray, StridedView, StridedViewMut};
use crate::{Result, StridedError};
use std::ops::Range;
use strided_view::ElementOp;

#[cfg(feature = "parallel")]
use crate::fuse::compute_costs;
use crate::inline::InlineVec;
#[cfg(feature = "parallel")]
use crate::threading::{
    for_each_inner_block_with_offsets, mapreduce_threaded_fold, SendPtr, MINTHREADLENGTH,
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
            let src_send = SendPtr(src_ptr as *mut T);
            let costs = compute_costs(&ordered_strides);
            let initial_offsets: crate::inline::OperandVec<isize> =
                ordered_strides.iter().map(|_| 0isize).collect();
            let map_fn_ref = &map_fn;
            let reduce_fn_ref = &reduce_fn;
            let init_ref = &init;
            // The reduction tree combines each split's partial result, so no
            // worker-count-sized output-slot vector is needed (#306 scope).
            let result = mapreduce_threaded_fold(
                &fused_dims,
                &plan.block,
                &ordered_strides,
                &initial_offsets,
                &costs,
                nthreads,
                &|dims, blocks, strides_list, offsets| {
                    let mut acc = init_ref.clone();
                    for_each_inner_block_with_offsets(
                        dims,
                        blocks,
                        strides_list,
                        offsets,
                        |offsets, len, strides| {
                            let mut ptr = unsafe { src_send.as_const().offset(offsets[0]) };
                            let stride = strides[0];
                            for _ in 0..len {
                                let val = Op::apply(unsafe { *ptr });
                                let mapped = map_fn_ref(val);
                                acc = reduce_fn_ref(acc.clone(), mapped);
                                unsafe {
                                    ptr = ptr.offset(stride);
                                }
                            }
                            Ok(())
                        },
                    )?;
                    Ok(acc)
                },
                &|a, b| reduce_fn_ref(a, b),
            )?;
            return Ok(result);
        }
    }

    let mut acc = init;
    let initial_offsets: crate::inline::OperandVec<isize> =
        ordered_strides.iter().map(|_| 0isize).collect();
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

/// Reduce along a single axis, returning a new `StridedArray`.
///
/// This allocates a column-major output and delegates to
/// [`reduce_axis_into`].
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
    // Reject an element count beyond `usize` before allocating the output.
    total_len(src.dims())?;
    let out_dims: InlineVec<usize> = src
        .dims()
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &d)| d)
        .collect();
    // A full reduction keeps the historical rank-1 shape-[1] output; the
    // `_into` form accepts any single-element destination for that case.
    let (out_dims, out_strides) = if out_dims.is_empty() {
        let mut dims = InlineVec::new();
        dims.push(1usize);
        let strides = col_major_strides(&dims);
        (dims, strides)
    } else {
        let strides = col_major_strides(&out_dims);
        (out_dims, strides)
    };
    let total_out = total_len(&out_dims)?;
    let mut out =
        StridedArray::<U>::from_parts(vec![init.clone(); total_out], &out_dims, &out_strides, 0)?;
    {
        let mut dest = out.view_mut();
        reduce_axis_into(&mut dest, src, axis, map_fn, reduce_fn, init)?;
    }
    Ok(out)
}

/// Reduce along a single axis into caller-owned storage.
///
/// `dest` must have the source's shape with `axis` removed, or any
/// single-element destination for a full reduction, and an injective layout.
/// Every destination element is written: an empty reduced axis fills each
/// output with `init`.
///
/// # Errors
///
/// Returns [`StridedError::InvalidAxis`] for `axis >= src.ndim()`,
/// [`StridedError::RankMismatch`] / [`StridedError::ShapeMismatch`] for a
/// destination shape that is not the reduced source shape,
/// [`StridedError::NonInjectiveOutputLayout`] for an overlapping destination,
/// and [`StridedError::OffsetOverflow`] for unrepresentable sizes.
///
/// # Examples
///
/// ```
/// use strided_basic::{reduce_axis_into, StridedArray};
/// # let src = StridedArray::<f64>::from_fn_col_major(&[2, 3], |i| (i[0] + 2 * i[1]) as f64);
/// let mut out = StridedArray::<f64>::col_major(&[3]);
/// reduce_axis_into(&mut out.view_mut(), &src.view(), 0, |x| x, |a, b| a + b, 0.0).unwrap();
/// assert_eq!(out.get(&[0]), 1.0);
/// ```
// ponytail: the array-returning wrapper regressed against the pre-0.6
// `reduce_axis` (contiguous 1T, shared EPYC host load ~5-6, 2026-10-17):
// 4x4x4 191 -> 322 ns, 32x32x32 7150 -> 8423 ns. The allocation-free planner
// and the `_into` split are the only structural changes; the inline cursor and
// `kept` metadata are candidates. Re-profile before B4.
pub fn reduce_axis_into<T: Copy + MaybeSendSync, Op: ElementOp<T>, M, R, U>(
    dest: &mut StridedViewMut<U>,
    src: &StridedView<T, Op>,
    axis: usize,
    map_fn: M,
    reduce_fn: R,
    init: U,
) -> Result<()>
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
    // front instead of looping over it.
    total_len(src_dims)?;

    let expected: InlineVec<usize> = src_dims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &d)| d)
        .collect();
    let dest_len = total_len(dest.dims())?;
    if expected.is_empty() {
        if dest_len != 1 {
            return Err(StridedError::ShapeMismatch(vec![dest_len], vec![1]));
        }
    } else {
        if dest.ndim() != expected.len() {
            return Err(StridedError::RankMismatch(dest.ndim(), expected.len()));
        }
        if dest.dims() != expected.as_slice() {
            return Err(StridedError::ShapeMismatch(
                dest.dims().to_vec(),
                expected.to_vec(),
            ));
        }
    }
    if !crate::layout_check::is_injective_layout(dest.dims(), dest.strides()) {
        return Err(StridedError::NonInjectiveOutputLayout);
    }

    let axis_len = src_dims[axis];
    let axis_stride = src_strides[axis];

    // Full reduction into a single destination element.
    if expected.is_empty() {
        // SAFETY: the view proves `axis_len` readable elements along `axis`.
        let acc = unsafe {
            fold_run::<T, Op, M, R, U>(src_ptr, axis_stride, axis_len, init, &map_fn, &reduce_fn)
        };
        unsafe { *dest.as_mut_ptr() = acc };
        return Ok(());
    }

    if dest_len == 0 {
        return Ok(());
    }

    let mut kept: InlineVec<(usize, isize)> = InlineVec::new();
    let mut dest_strides: InlineVec<isize> = InlineVec::new();
    {
        let mut d = 0usize;
        for i in 0..rank {
            if i == axis {
                continue;
            }
            kept.push((src_dims[i], src_strides[i]));
            dest_strides.push(dest.strides()[d]);
            d += 1;
        }
    }
    // A column-major destination is the common case (the array-returning
    // wrapper always allocates one). Its destination offset equals the linear
    // output index, so the per-output destination bookkeeping can be skipped.
    let dest_linear = {
        let mut prefix = 1usize;
        kept.iter()
            .zip(dest_strides.iter())
            .all(|(&(dim, _), &dest_stride)| {
                let linear = dest_stride == prefix as isize;
                prefix = prefix.saturating_mul(dim);
                linear
            })
    };
    let out_ptr = dest.as_mut_ptr();
    let parts = AxisParts {
        kept: &kept,
        dest_strides: &dest_strides,
        axis_len,
        axis_stride,
        dest_linear,
        map_fn: &map_fn,
        reduce_fn: &reduce_fn,
        init: &init,
    };

    #[cfg(feature = "parallel")]
    {
        let work = dest_len.saturating_mul(axis_len);
        let nthreads = crate::threading::parallel_threads_for_len(work).min(dest_len);
        if nthreads > 1 {
            let src_send = SendPtr(src_ptr as *mut T);
            let out_send = SendPtr(out_ptr);
            let parts = &parts;
            crate::threading::parallel_for_each(0..dest_len, nthreads, &|range| {
                // SAFETY: worker ranges are disjoint, the destination is
                // injective, and the view proves every source offset reachable
                // from its kept coordinates.
                unsafe {
                    reduce_axis_range::<T, Op, M, R, U>(
                        src_send.as_const(),
                        out_send.as_ptr(),
                        range,
                        parts,
                    )
                }
            });
            return Ok(());
        }
    }

    // SAFETY: as for the parallel ranges, with one range covering all outputs.
    unsafe { reduce_axis_range::<T, Op, M, R, U>(src_ptr, out_ptr, 0..dest_len, &parts) };
    Ok(())
}

/// Shared inputs of [`reduce_axis_range`].
struct AxisParts<'a, M, R, U> {
    /// `(extent, source stride)` of every kept axis, in output order.
    kept: &'a [(usize, isize)],
    /// Destination stride of every kept axis, in the same order.
    dest_strides: &'a [isize],
    axis_len: usize,
    axis_stride: isize,
    /// Whether the destination is the column-major output, so its offset is the
    /// linear output index and no destination cursor is needed.
    dest_linear: bool,
    map_fn: &'a M,
    reduce_fn: &'a R,
    init: &'a U,
}

/// Computes outputs `range` (column-major logical output indices) of
/// [`reduce_axis_into`].
///
/// # Safety
/// `out` must be a destination of `parts.kept` extents with injective layout,
/// `range` a subrange of its logical output indices that no other thread
/// writes, and every source offset reachable from a kept coordinate plus
/// `k * parts.axis_stride` for `k < parts.axis_len` must be a readable element
/// of `src`.
/// Computes outputs `range` of [`reduce_axis_into`], dispatching on the
/// destination layout so the common column-major case runs the same cursor as
/// the array-returning wrapper used before this was split (#306 performance).
///
/// # Safety
/// As [`reduce_axis_range_linear`] / [`reduce_axis_range_strided`].
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
    if parts.dest_linear {
        // SAFETY: forwarded caller contract; a linear destination is a valid
        // destination for the kept extents.
        unsafe { reduce_axis_range_linear::<T, Op, M, R, U>(src, out, range, parts) }
    } else {
        // SAFETY: forwarded caller contract.
        unsafe { reduce_axis_range_strided::<T, Op, M, R, U>(src, out, range, parts) }
    }
}

/// Column-major destination cursor: the destination offset is the linear output
/// index, so no destination stride bookkeeping is needed.
///
/// # Safety
/// `out` must be a column-major destination of `parts.kept` extents, `range` a
/// subrange of its logical output indices that no other thread writes, and
/// every source offset reachable from a kept coordinate plus
/// `k * parts.axis_stride` for `k < parts.axis_len` must be a readable element
/// of `src`.
unsafe fn reduce_axis_range_linear<T, Op, M, R, U>(
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

    // Decode the first output once, then advance incrementally. `coords` is
    // inline at rank <= INLINE_RANK.
    let mut coords: InlineVec<usize> = (0..kept.len()).map(|_| 0usize).collect();
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
            // SAFETY: the `len` outputs are inside the caller's range, their
            // unit-stride destination elements are contiguous, and their
            // unit-stride source columns are readable.
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

/// General strided destination cursor, tracking a destination offset next to
/// the source offset.
///
/// # Safety
/// `out` must be a destination of `parts.kept` extents with injective layout,
/// `range` a subrange of its logical output indices that no other thread
/// writes, and every source offset reachable from a kept coordinate plus
/// `k * parts.axis_stride` for `k < parts.axis_len` must be a readable element
/// of `src`.
unsafe fn reduce_axis_range_strided<T, Op, M, R, U>(
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
    let dest_strides = parts.dest_strides;
    let (lead_extent, lead_stride) = kept[0];
    let lead_dest_stride = dest_strides[0];
    let sweep = lead_stride == 1
        && lead_dest_stride == 1
        && lead_extent >= AXIS_SWEEP_MIN_LEN
        && parts.axis_len > 1;
    let block = (AXIS_SWEEP_BLOCK_BYTES / std::mem::size_of::<U>().max(1)).max(AXIS_SWEEP_MIN_LEN);

    let mut coords: InlineVec<usize> = (0..kept.len()).map(|_| 0usize).collect();
    let mut rest = range.start;
    let mut src_off = 0isize;
    let mut dest_off = 0isize;
    for (coord, (&(extent, stride), &dest_stride)) in
        coords.iter_mut().zip(kept.iter().zip(dest_strides))
    {
        *coord = rest % extent;
        rest /= extent;
        src_off += *coord as isize * stride;
        dest_off += *coord as isize * dest_stride;
    }

    let mut output = range.start;
    while output < range.end {
        let len = if sweep {
            (lead_extent - coords[0]).min(range.end - output).min(block)
        } else {
            1
        };
        if sweep {
            // SAFETY: as for the linear cursor; `lead_dest_stride == 1` makes
            // the `len` destinations contiguous.
            unsafe {
                sweep_block::<T, Op, M, R, U>(src, src_off, out.offset(dest_off), len, parts)
            };
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
                *out.offset(dest_off) = acc;
            }
        }
        output += len;
        if output >= range.end {
            break;
        }
        coords[0] += len;
        src_off += len as isize * lead_stride;
        dest_off += len as isize * lead_dest_stride;
        let mut axis = 0;
        while coords[axis] == kept[axis].0 {
            src_off -= kept[axis].0 as isize * kept[axis].1;
            dest_off -= kept[axis].0 as isize * dest_strides[axis];
            coords[axis] = 0;
            axis += 1;
            coords[axis] += 1;
            src_off += kept[axis].1;
            dest_off += dest_strides[axis];
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
