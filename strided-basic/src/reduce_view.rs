//! Reduce operations on dynamic-rank strided views.

use crate::kernel::{
    build_plan_fused, ensure_same_shape, for_each_inner_block_preordered, same_contiguous_layout,
    sequential_contiguous_layout, total_len,
};
use crate::simd;
use crate::view::{col_major_strides, StridedArray, StridedView, StridedViewMut};
use crate::{Exec, Result, StridedError};
use core::num::NonZeroUsize;
use std::ops::Range;
use strided_view::ElementOp;

use crate::cost::KernelClass;
use crate::fuse::compute_costs;
use crate::inline::InlineVec;
use crate::threading::{
    for_each_inner_block_with_offsets, is_parallel, mapreduce_threaded_fold, SendPtr,
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
/// The result is independent of the pool when the call runs with a serial
/// token ([`Exec::serial`]) or a budget of one: only serial branches run, and
/// their fold order is fixed by the source layout. With a larger budget the
/// fold is split over the budget, so the result is repeatable for a fixed
/// shape, layout and budget but can differ between budgets.
///
/// In neither case is this a strict left to right fold: a contiguous source is
/// folded with sixteen independent lanes (see the crate source for the exact
/// lane assignment), non-contiguous sources are traversed in a cache-friendly
/// loop order, and parallel execution folds per-task chunks independently
/// and combines them in a tree. Floating point results can therefore differ
/// from a sequential left to right sum in the last bits, and no bitwise
/// equality across machines or target features is guaranteed.
///
/// Because of that, `reduce_fn` must be associative and commutative, and
/// `init` must be an identity of `reduce_fn` whenever the reduction can run on
/// several threads.
///
/// # Allocation
///
/// At rank ≤ [`INLINE_RANK`](strided_view::INLINE_RANK) the view metadata and
/// the traversal plan are stored inline, so a call performs no heap
/// allocation. Above that bound the planner may spill to the heap.
///
/// Because `map_fn` and `reduce_fn` are opaque closures this entry cannot
/// select the dtype-specific SIMD kernels of [`crate::ErasedReducePlan`];
/// callers that reduce with a known operation (sum, product, max, min) on a
/// supported dtype can use that plan for the fastest path.
pub fn reduce<T: Copy + Send + Sync, Op: ElementOp<T>, M, R, U>(
    src: &StridedView<T, Op>,
    map_fn: M,
    reduce_fn: R,
    init: U,
    exec: Exec<'_>,
) -> Result<U>
where
    M: Fn(T) -> U + Sync,
    R: Fn(U, U) -> U + Sync,
    U: Clone + Send + Sync,
{
    reduce_impl(src, map_fn, reduce_fn, init, exec)
}

fn reduce_impl<T: Copy + Send + Sync, Op: ElementOp<T>, M, R, U>(
    src: &StridedView<T, Op>,
    map_fn: M,
    reduce_fn: R,
    init: U,
    exec: Exec<'_>,
) -> Result<U>
where
    M: Fn(T) -> U + Sync,
    R: Fn(U, U) -> U + Sync,
    U: Clone + Send + Sync,
{
    let src_ptr = src.ptr();
    let src_dims = src.dims();
    let src_strides = src.strides();

    // A zero-element reduction is exactly `init`. Return before planning: an
    // empty view is allowed to carry an arbitrarily large stride, which the
    // block-size arithmetic of the planner cannot represent.
    if total_len(src_dims)? == 0 {
        return Ok(init);
    }

    let contiguous =
        sequential_contiguous_layout(KernelClass::Reduce, src_dims, &[src_strides], &exec)?;
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
    let total = total_len(src_dims)?;
    if is_parallel(KernelClass::Reduce, total, &exec)
        && same_contiguous_layout(src_dims, &[src_strides]).is_some()
    {
        let src_slice = unsafe { std::slice::from_raw_parts(src_ptr, total) };
        let result = crate::threading::parallel_map_reduce(
            0..total,
            crate::cost::for_len_tree(exec, KernelClass::Reduce, total),
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

    let strides_list: [&[isize]; 1] = [src_strides];

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(src_dims, &strides_list, None, std::mem::size_of::<T>());

    let total = total_len(&fused_dims)?;
    if is_parallel(KernelClass::Reduce, total, &exec) {
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
            KernelClass::Reduce,
            exec,
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

/// Folds a contiguous run of two operands with [`FOLD_LANES`] independent
/// accumulators.
///
/// Same lane scheme as [`fold_contiguous`]: lane `i` starts from the pair at
/// index `i` and folds the pairs at `i + FOLD_LANES`, `i + 2 * FOLD_LANES`,
/// ... in order, the remainder goes to the first lanes, and `init` is combined
/// once before the lanes are combined left to right.
///
/// The scheduling below deliberately mirrors [`fold_contiguous`] instead of
/// sharing a generic "map index to value" lane helper: routing the existing
/// single-operand fold through such an accessor was measured at 0.34 ns/element
/// against 0.11 ns/element on a contiguous `f64` `reduce` (1 MiB, release,
/// `--release -j 4`), because the `chunks_exact` + `zip` form is what lets the
/// compiler drop the bound checks and vectorize the 16 lanes. Keep the two
/// bodies in step by hand; both must use the lane assignment documented here.
#[inline(always)]
fn fold_contiguous2<A, B, OpA, OpB, M, R, U>(
    a: &[A],
    b: &[B],
    init: U,
    map_fn: &M,
    reduce_fn: &R,
) -> U
where
    A: Copy,
    B: Copy,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
    M: Fn(A, B) -> U,
    R: Fn(U, U) -> U,
    U: Clone,
{
    debug_assert_eq!(a.len(), b.len());
    if a.len() < 2 * FOLD_LANES {
        let mut acc = init;
        for (&value_a, &value_b) in a.iter().zip(b) {
            acc = reduce_fn(acc, map_fn(OpA::apply(value_a), OpB::apply(value_b)));
        }
        return acc;
    }
    let (head_a, rest_a) = a.split_at(FOLD_LANES);
    let (head_b, rest_b) = b.split_at(FOLD_LANES);
    let mut lanes: [U; FOLD_LANES] =
        core::array::from_fn(|lane| map_fn(OpA::apply(head_a[lane]), OpB::apply(head_b[lane])));
    let mut chunks_a = rest_a.chunks_exact(FOLD_LANES);
    let mut chunks_b = rest_b.chunks_exact(FOLD_LANES);
    for (chunk_a, chunk_b) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        for ((lane, &value_a), &value_b) in lanes.iter_mut().zip(chunk_a).zip(chunk_b) {
            *lane = reduce_fn(
                lane.clone(),
                map_fn(OpA::apply(value_a), OpB::apply(value_b)),
            );
        }
    }
    for ((lane, &value_a), &value_b) in lanes
        .iter_mut()
        .zip(chunks_a.remainder())
        .zip(chunks_b.remainder())
    {
        *lane = reduce_fn(
            lane.clone(),
            map_fn(OpA::apply(value_a), OpB::apply(value_b)),
        );
    }
    lanes.into_iter().fold(init, reduce_fn)
}

/// Folds `len` operand pairs starting at `a_ptr`/`b_ptr` with the given
/// strides.
///
/// # Safety
/// `a_ptr.offset(i * a_stride)` and `b_ptr.offset(i * b_stride)` must be
/// readable elements for every `i < len`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn fold_run2<A, B, OpA, OpB, M, R, U>(
    a_ptr: *const A,
    a_stride: isize,
    b_ptr: *const B,
    b_stride: isize,
    len: usize,
    init: &U,
    map_fn: &M,
    reduce_fn: &R,
) -> U
where
    A: Copy,
    B: Copy,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
    M: Fn(A, B) -> U,
    R: Fn(U, U) -> U,
    U: Clone,
{
    if a_stride == 1 && b_stride == 1 {
        // SAFETY: the caller proves `len` readable unit-stride elements of
        // both operands.
        let a = unsafe { std::slice::from_raw_parts(a_ptr, len) };
        let b = unsafe { std::slice::from_raw_parts(b_ptr, len) };
        return fold_contiguous2::<A, B, OpA, OpB, M, R, U>(a, b, init.clone(), map_fn, reduce_fn);
    }
    let mut acc = init.clone();
    let mut cursor_a = a_ptr;
    let mut cursor_b = b_ptr;
    for index in 0..len {
        // SAFETY: the caller proves every pair of the run is readable.
        let mapped = map_fn(
            OpA::apply(unsafe { *cursor_a }),
            OpB::apply(unsafe { *cursor_b }),
        );
        acc = reduce_fn(acc, mapped);
        if index + 1 < len {
            cursor_a = cursor_a.wrapping_offset(a_stride);
            cursor_b = cursor_b.wrapping_offset(b_stride);
        }
    }
    acc
}

/// Two-input mapped reduction: `reduce(init, op, map.(a[i], b[i]))`.
///
/// This is the two-operand sibling of [`reduce`]. It exists so that the
/// accumulator type `U` is chosen by the caller instead of being tied to the
/// operand types by `Mul`, which is what makes a widening accumulation
/// possible — for example an `f32` dot accumulated in `f64`:
///
/// ```
/// use strided_basic::{zip_reduce, Exec, StridedArray};
///
/// let a = StridedArray::<f32>::from_fn_col_major(&[2, 3], |i| (1 + i[0] + 2 * i[1]) as f32);
/// let b = StridedArray::<f32>::from_fn_col_major(&[2, 3], |i| (1 + i[0]) as f32 * 0.5);
/// let dot: f64 = zip_reduce(
///     &a.view(),
///     &b.view(),
///     |x: f32, y: f32| x as f64 * y as f64,
///     |x, y| x + y,
///     0.0,
///     Exec::serial(),
/// )
/// .unwrap();
/// assert_eq!(dot, 16.5);
/// ```
///
/// Both operands must have the same shape. Each operand's [`ElementOp`] is
/// applied before `map_fn`, so conjugation and transposition flags are honoured
/// without materializing a conjugated copy. Permuted, gapped, negative and
/// stride-0 (broadcast) operands are supported: the traversal is the fused
/// plan of [`reduce`], extended to two stride lists.
///
/// # Evaluation order
///
/// The result is independent of the pool when the call runs with a serial
/// token ([`Exec::serial`]) or a budget of one: only serial branches run, and
/// their fold order is fixed by the operand layouts. With a larger budget the
/// fold is split over the budget, so the result is repeatable for a fixed
/// shape, layout and budget but can differ between budgets.
///
/// In neither case is this a strict left to right fold: a contiguous operand
/// pair is folded with the same sixteen independent lanes as [`reduce`], the
/// general plan is traversed in a cache-friendly loop order, and parallel
/// execution folds per-worker chunks independently and combines them in a
/// tree. Floating point results can therefore differ from a sequential left to
/// right sum in the last bits, and no bitwise equality across machines or
/// target features is guaranteed.
///
/// Because of that, `reduce_fn` must be associative and commutative, and
/// `init` must be an identity of `reduce_fn` whenever the reduction can run on
/// several threads.
///
/// # Allocation
///
/// At rank ≤ [`INLINE_RANK`](strided_view::INLINE_RANK) the view metadata and
/// the two-operand traversal plan are stored inline, so a call performs no heap
/// allocation. Above that bound the planner may spill to the heap.
///
/// # Errors
///
/// Returns [`StridedError::RankMismatch`] / [`StridedError::ShapeMismatch`] when
/// the operand shapes differ, and [`StridedError::OffsetOverflow`] for
/// unrepresentable element counts.
pub fn zip_reduce<A, B, OpA, OpB, M, R, U>(
    a: &StridedView<A, OpA>,
    b: &StridedView<B, OpB>,
    map_fn: M,
    reduce_fn: R,
    init: U,
    exec: Exec<'_>,
) -> Result<U>
where
    A: Copy + Send + Sync,
    B: Copy + Send + Sync,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
    M: Fn(A, B) -> U + Sync,
    R: Fn(U, U) -> U + Sync,
    U: Clone + Send + Sync,
{
    ensure_same_shape(a.dims(), b.dims())?;

    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let dims = a.dims();
    let a_strides = a.strides();
    let b_strides = b.strides();

    // A zero-element reduction is exactly `init`. Return before planning: an
    // empty view is allowed to carry an arbitrarily large stride, which the
    // block-size arithmetic of the planner cannot represent.
    let total = total_len(dims)?;
    if total == 0 {
        return Ok(init);
    }

    if sequential_contiguous_layout(KernelClass::Reduce, dims, &[a_strides, b_strides], &exec)?
        .is_some()
    {
        return Ok(simd::dispatch_if_large(total, || {
            // SAFETY: both operands prove `total` readable elements.
            let sa = unsafe { std::slice::from_raw_parts(a_ptr, total) };
            let sb = unsafe { std::slice::from_raw_parts(b_ptr, total) };
            fold_contiguous2::<A, B, OpA, OpB, M, R, U>(sa, sb, init, &map_fn, &reduce_fn)
        }));
    }

    // Parallel contiguous fast path: split into scheduler chunks so each chunk
    // folds two contiguous slices, matching the serial contiguous kernel.
    if is_parallel(KernelClass::Reduce, total, &exec)
        && same_contiguous_layout(dims, &[a_strides, b_strides]).is_some()
    {
        // SAFETY: both operands prove `total` readable elements.
        let sa = unsafe { std::slice::from_raw_parts(a_ptr, total) };
        let sb = unsafe { std::slice::from_raw_parts(b_ptr, total) };
        let result = crate::threading::parallel_map_reduce(
            0..total,
            crate::cost::for_len_tree(exec, KernelClass::Reduce, total),
            &|range| {
                let len = range.len();
                simd::dispatch_if_large(len, || {
                    fold_contiguous2::<A, B, OpA, OpB, M, R, U>(
                        &sa[range.clone()],
                        &sb[range],
                        init.clone(),
                        &map_fn,
                        &reduce_fn,
                    )
                })
            },
            &|left, right| reduce_fn(left, right),
        );
        return Ok(result);
    }

    let strides_list: [&[isize]; 2] = [a_strides, b_strides];
    let elem_size = std::mem::size_of::<A>().max(std::mem::size_of::<B>());
    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(dims, &strides_list, None, elem_size);

    let total = total_len(&fused_dims)?;
    if is_parallel(KernelClass::Reduce, total, &exec) {
        let a_send = SendPtr(a_ptr as *mut A);
        let b_send = SendPtr(b_ptr as *mut B);
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
            KernelClass::Reduce,
            exec,
            &|dims, blocks, strides_list, offsets| {
                let mut acc = init_ref.clone();
                for_each_inner_block_with_offsets(
                    dims,
                    blocks,
                    strides_list,
                    offsets,
                    |offsets, len, strides| {
                        // SAFETY: the plan proves every pair of this block
                        // is a readable element of both operands.
                        acc = unsafe {
                            fold_run2::<A, B, OpA, OpB, M, R, U>(
                                a_send.as_const().offset(offsets[0]),
                                strides[0],
                                b_send.as_const().offset(offsets[1]),
                                strides[1],
                                len,
                                &acc,
                                map_fn_ref,
                                reduce_fn_ref,
                            )
                        };
                        Ok(())
                    },
                )?;
                Ok(acc)
            },
            &|left, right| reduce_fn_ref(left, right),
        )?;
        return Ok(result);
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
            // SAFETY: the plan proves every pair of this block is a readable
            // element of both operands.
            acc = unsafe {
                fold_run2::<A, B, OpA, OpB, M, R, U>(
                    a_ptr.offset(offsets[0]),
                    strides[0],
                    b_ptr.offset(offsets[1]),
                    strides[1],
                    len,
                    &acc,
                    &map_fn,
                    &reduce_fn,
                )
            };
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
/// left to right order per output. Outputs are split across the budget of
/// `exec` once the number of source elements read exceeds the threading
/// threshold; each output is computed by one task, so the result does not
/// depend on the budget.
pub fn reduce_axis<T: Copy + Send + Sync, Op: ElementOp<T>, M, R, U>(
    src: &StridedView<T, Op>,
    axis: usize,
    map_fn: M,
    reduce_fn: R,
    init: U,
    exec: Exec<'_>,
) -> Result<StridedArray<U>>
where
    M: Fn(T) -> U + Sync,
    R: Fn(U, U) -> U + Sync,
    U: Clone + Send + Sync,
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
        // The fresh column-major destination needs no validation.
        reduce_axis_into_impl(&mut dest, src, axis, map_fn, reduce_fn, init, exec)?;
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
/// use strided_basic::{reduce_axis_into, Exec, StridedArray};
/// # let src = StridedArray::<f64>::from_fn_col_major(&[2, 3], |i| (i[0] + 2 * i[1]) as f64);
/// let mut out = StridedArray::<f64>::col_major(&[3]);
/// reduce_axis_into(&mut out.view_mut(), &src.view(), 0, |x| x, |a, b| a + b, 0.0, Exec::serial()).unwrap();
/// assert_eq!(out.get(&[0]), 1.0);
/// ```
///
/// Validation runs here; `reduce_axis_into_impl` is the shared kernel. The
/// array-returning wrapper owns a fresh column-major destination and calls the
/// kernel directly, so the common path does not re-validate.
pub fn reduce_axis_into<T: Copy + Send + Sync, Op: ElementOp<T>, M, R, U>(
    dest: &mut StridedViewMut<U>,
    src: &StridedView<T, Op>,
    axis: usize,
    map_fn: M,
    reduce_fn: R,
    init: U,
    exec: Exec<'_>,
) -> Result<()>
where
    M: Fn(T) -> U + Sync,
    R: Fn(U, U) -> U + Sync,
    U: Clone + Send + Sync,
{
    let rank = src.ndim();
    if axis >= rank {
        return Err(StridedError::InvalidAxis { axis, rank });
    }
    let src_dims = src.dims();
    // Reject an element count beyond usize (a huge stride-0 broadcast) up
    // front instead of looping over it.
    total_len(src_dims)?;
    let kept_len = rank - 1;
    let dest_len = total_len(dest.dims())?;
    if kept_len == 0 {
        if dest_len != 1 {
            return Err(StridedError::ShapeMismatch(vec![dest_len], vec![1]));
        }
    } else {
        if dest.ndim() != kept_len {
            return Err(StridedError::RankMismatch(dest.ndim(), kept_len));
        }
        let dest_dims = dest.dims();
        let mut d = 0usize;
        for (i, &src_dim) in src_dims.iter().enumerate() {
            if i == axis {
                continue;
            }
            if src_dim != dest_dims[d] {
                let expected: Vec<usize> = src_dims
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != axis)
                    .map(|(_, &d)| d)
                    .collect();
                return Err(StridedError::ShapeMismatch(dest_dims.to_vec(), expected));
            }
            d += 1;
        }
        if !crate::layout_check::is_injective_layout(dest.dims(), dest.strides()) {
            return Err(StridedError::NonInjectiveOutputLayout);
        }
    }
    reduce_axis_into_impl(dest, src, axis, map_fn, reduce_fn, init, exec)
}

/// Shared kernel of [`reduce_axis_into`] and [`reduce_axis`].
///
/// Assumes the destination has the reduced source shape (or one element for a
/// full reduction) and an injective layout; the public entry validates that,
/// while the array-returning wrapper owns a freshly allocated column-major
/// destination and can skip the re-validation.
fn reduce_axis_into_impl<T: Copy + Send + Sync, Op: ElementOp<T>, M, R, U>(
    dest: &mut StridedViewMut<U>,
    src: &StridedView<T, Op>,
    axis: usize,
    map_fn: M,
    reduce_fn: R,
    init: U,
    exec: Exec<'_>,
) -> Result<()>
where
    M: Fn(T) -> U + Sync,
    R: Fn(U, U) -> U + Sync,
    U: Clone + Send + Sync,
{
    let rank = src.ndim();
    let src_dims = src.dims();
    let src_strides = src.strides();
    let src_ptr = src.ptr();
    let dest_len = total_len(dest.dims())?;
    let axis_len = src_dims[axis];
    let axis_stride = src_strides[axis];

    if axis_len == 0 {
        // No source elements to fold: every output is `init`. Fill the
        // destination directly rather than traversing the source, because an
        // empty source view carries a dangling base pointer and `offset` on it
        // is UB even when the fold reads zero elements.
        fill_outputs_with(dest, &init);
        return Ok(());
    }

    if rank == 1 {
        // Full reduction into a single destination element.
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

    // Build the kept-axis metadata in one pass. A column-major destination is
    // the common case (the array-returning wrapper always allocates one); its
    // offset equals the linear output index, so its stride list is only
    // materialized for a strided destination.
    let dest_stride_view = dest.strides();
    let mut kept: InlineVec<(usize, isize)> = InlineVec::new();
    let mut dest_linear = true;
    let mut prefix = 1usize;
    for (i, (&dim, &src_stride)) in src_dims.iter().zip(src_strides).enumerate() {
        if i == axis {
            continue;
        }
        let d = kept.len();
        kept.push((dim, src_stride));
        if dest_linear && dest_stride_view[d] != prefix as isize {
            dest_linear = false;
        }
        prefix = prefix.saturating_mul(dim);
    }
    let dest_strides: InlineVec<isize> = if dest_linear {
        InlineVec::new()
    } else {
        dest_stride_view.iter().copied().collect()
    };
    let out_ptr = dest.as_mut_ptr();
    let parts = AxisParts {
        kept: &kept,
        dest_strides: &dest_strides,
        axis_len,
        axis_stride,
        map_fn: &map_fn,
        reduce_fn: &reduce_fn,
        init: &init,
    };

    let work = dest_len.saturating_mul(axis_len);
    let exec = crate::cost::for_len(exec, KernelClass::Reduce, work)
        .limit(NonZeroUsize::new(dest_len).unwrap_or(NonZeroUsize::MIN));
    if !exec.is_serial() {
        let src_send = SendPtr(src_ptr as *mut T);
        let out_send = SendPtr(out_ptr);
        let parts = &parts;
        if dest_linear {
            crate::threading::parallel_for_each(0..dest_len, exec, &|range| {
                // SAFETY: worker ranges are disjoint, the destination is
                // injective, and the view proves every source offset
                // reachable from its kept coordinates.
                unsafe {
                    reduce_axis_range_linear::<T, Op, M, R, U>(
                        src_send.as_const(),
                        out_send.as_ptr(),
                        range,
                        parts,
                    )
                }
            });
        } else {
            crate::threading::parallel_for_each(0..dest_len, exec, &|range| {
                // SAFETY: as for the linear path.
                unsafe {
                    reduce_axis_range_strided::<T, Op, M, R, U>(
                        src_send.as_const(),
                        out_send.as_ptr(),
                        range,
                        parts,
                    )
                }
            });
        }
        return Ok(());
    }

    // SAFETY: as for the parallel ranges, with one range covering all outputs.
    if dest_linear {
        unsafe {
            reduce_axis_range_linear::<T, Op, M, R, U>(src_ptr, out_ptr, 0..dest_len, &parts)
        };
    } else {
        unsafe {
            reduce_axis_range_strided::<T, Op, M, R, U>(src_ptr, out_ptr, 0..dest_len, &parts)
        };
    }
    Ok(())
}

/// Writes `value` into every logical destination element.
///
/// Used when the reduced axis is empty: the fold would read zero source
/// elements, so the destination is exactly `init` everywhere and no source
/// pointer may be formed. The odometer keeps the fill allocation-free within
/// [`strided_view::INLINE_RANK`] and spills above it.
fn fill_outputs_with<U: Clone>(dest: &mut StridedViewMut<U>, value: &U) {
    let dims = dest.dims();
    if dims.iter().any(|&dim| dim == 0) {
        return;
    }
    let strides = dest.strides();
    let rank = dims.len();
    let total: usize = dims.iter().product();
    let base = dest.as_mut_ptr();
    if rank == 0 {
        // SAFETY: a rank-0 destination is one writable element.
        unsafe { *base = value.clone() };
        return;
    }
    let mut coords_inline = [0usize; strided_view::INLINE_RANK];
    let mut coords_heap: Vec<usize> = Vec::new();
    let coords: &mut [usize] = if rank <= strided_view::INLINE_RANK {
        &mut coords_inline[..rank]
    } else {
        coords_heap.resize(rank, 0);
        &mut coords_heap[..]
    };
    let mut offset = 0isize;
    for _ in 0..total {
        // SAFETY: the destination is an injective view validated by the
        // caller, and the cursor visits each logical element exactly once, so
        // every written offset is in bounds.
        unsafe { *base.offset(offset) = value.clone() };
        for axis in 0..rank {
            coords[axis] += 1;
            offset += strides[axis];
            if coords[axis] < dims[axis] {
                break;
            }
            coords[axis] = 0;
            offset -= dims[axis] as isize * strides[axis];
        }
    }
}

/// Shared inputs of [`reduce_axis_range_linear`] and
/// [`reduce_axis_range_strided`].
struct AxisParts<'a, M, R, U> {
    /// `(extent, source stride)` of every kept axis, in output order.
    kept: &'a [(usize, isize)],
    /// Destination stride of every kept axis, in the same order.
    dest_strides: &'a [isize],
    axis_len: usize,
    axis_stride: isize,
    map_fn: &'a M,
    reduce_fn: &'a R,
    init: &'a U,
}

/// Column-major destination cursor: the destination offset is the linear output
/// index, so no destination stride bookkeeping is needed.
///
/// Kept out of line so the common path stays the same single-cursor loop as the
/// pre-split kernel.
///
/// # Safety
/// `out` must be a column-major destination of `parts.kept` extents, `range` a
/// subrange of its logical output indices that no other thread writes, and
/// every source offset reachable from a kept coordinate plus
/// `k * parts.axis_stride` for `k < parts.axis_len` must be a readable element
/// of `src`.
#[inline(never)]
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

    // Decode the first output once, then advance incrementally. A fixed stack
    // array with a spill covers every inline rank without the per-access enum
    // match a `TinyVec` cursor would add to the hot loop, and allocates only
    // above `INLINE_RANK` (#306 P1).
    let coords_len = kept.len();
    let mut coords_inline = [0usize; strided_view::INLINE_RANK];
    let mut coords_heap: Vec<usize> = Vec::new();
    let coords: &mut [usize] = if coords_len <= strided_view::INLINE_RANK {
        &mut coords_inline[..coords_len]
    } else {
        coords_heap.resize(coords_len, 0);
        &mut coords_heap[..]
    };
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
#[inline(never)]
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

    // Same fixed-array-plus-spill cursor as `reduce_axis_range_linear`: the
    // `TinyVec` enum match is not paid per output access in the hot loop and
    // only ranks above `INLINE_RANK` allocate (#306 P1).
    let coords_len = kept.len();
    let mut coords_inline = [0usize; strided_view::INLINE_RANK];
    let mut coords_heap: Vec<usize> = Vec::new();
    let coords: &mut [usize] = if coords_len <= strided_view::INLINE_RANK {
        &mut coords_inline[..coords_len]
    } else {
        coords_heap.resize(coords_len, 0);
        &mut coords_heap[..]
    };
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
