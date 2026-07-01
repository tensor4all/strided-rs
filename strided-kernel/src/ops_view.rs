//! High-level operations on dynamic-rank strided views.

use crate::kernel::{
    build_plan_fused, ensure_same_shape, for_each_inner_block_preordered, same_contiguous_layout,
    sequential_contiguous_layout, total_len,
};
use crate::map_view::{map_into, zip_map2_into};
use crate::maybe_sync::{MaybeSendSync, MaybeSync};
use crate::reduce_view::reduce;
use crate::simd;
use crate::view::{StridedView, StridedViewMut};
use crate::{Result, StridedError};
use num_traits::{One, Zero};
use std::ops::{Add, Mul};
use strided_view::{ElementOp, ElementOpApply};

#[cfg(feature = "parallel")]
use crate::fuse::compute_costs;
#[cfg(feature = "parallel")]
use crate::threading::{
    for_each_inner_block_with_offsets, mapreduce_threaded, SendPtr, MINTHREADLENGTH,
};
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

// ============================================================================
// Stride-specialized inner loop helpers for ops_view
//
// When all inner strides are 1 (contiguous in the innermost dimension),
// we use slice-based iteration so LLVM can auto-vectorize effectively.
// This mirrors the inner_loop_map* helpers in map_view.rs.
// ============================================================================

/// Inner loop for add: `dst[i] += Op::apply(src[i])`.
#[inline(always)]
unsafe fn inner_loop_add<D: Copy + Add<S, Output = D>, S: Copy, Op: ElementOp<S>>(
    dp: *mut D,
    ds: isize,
    sp: *const S,
    ss: isize,
    len: usize,
) {
    if ds == 1 && ss == 1 {
        let dst = std::slice::from_raw_parts_mut(dp, len);
        let src = std::slice::from_raw_parts(sp, len);
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                dst[i] = dst[i] + Op::apply(src[i]);
            }
        });
    } else {
        let mut dp = dp;
        let mut sp = sp;
        for _ in 0..len {
            *dp = *dp + Op::apply(*sp);
            dp = dp.offset(ds);
            sp = sp.offset(ss);
        }
    }
}

/// Inner loop for mul: `dst[i] *= Op::apply(src[i])`.
#[inline(always)]
unsafe fn inner_loop_mul<D: Copy + Mul<S, Output = D>, S: Copy, Op: ElementOp<S>>(
    dp: *mut D,
    ds: isize,
    sp: *const S,
    ss: isize,
    len: usize,
) {
    if ds == 1 && ss == 1 {
        let dst = std::slice::from_raw_parts_mut(dp, len);
        let src = std::slice::from_raw_parts(sp, len);
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                dst[i] = dst[i] * Op::apply(src[i]);
            }
        });
    } else {
        let mut dp = dp;
        let mut sp = sp;
        for _ in 0..len {
            *dp = *dp * Op::apply(*sp);
            dp = dp.offset(ds);
            sp = sp.offset(ss);
        }
    }
}

/// Inner loop for axpy: `dst[i] = alpha * Op::apply(src[i]) + dst[i]`.
#[inline(always)]
unsafe fn inner_loop_axpy<
    D: Copy + Add<D, Output = D>,
    S: Copy,
    A: Copy + Mul<S, Output = D>,
    Op: ElementOp<S>,
>(
    dp: *mut D,
    ds: isize,
    sp: *const S,
    ss: isize,
    len: usize,
    alpha: A,
) {
    if ds == 1 && ss == 1 {
        let dst = std::slice::from_raw_parts_mut(dp, len);
        let src = std::slice::from_raw_parts(sp, len);
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                dst[i] = alpha * Op::apply(src[i]) + dst[i];
            }
        });
    } else {
        let mut dp = dp;
        let mut sp = sp;
        for _ in 0..len {
            *dp = alpha * Op::apply(*sp) + *dp;
            dp = dp.offset(ds);
            sp = sp.offset(ss);
        }
    }
}

/// Inner loop for fma: `dst[i] += OpA::apply(a[i]) * OpB::apply(b[i])`.
#[inline(always)]
unsafe fn inner_loop_fma<
    D: Copy + Add<D, Output = D>,
    A: Copy + Mul<B, Output = D>,
    B: Copy,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
>(
    dp: *mut D,
    ds: isize,
    ap: *const A,
    a_s: isize,
    bp: *const B,
    b_s: isize,
    len: usize,
) {
    if ds == 1 && a_s == 1 && b_s == 1 {
        let dst = std::slice::from_raw_parts_mut(dp, len);
        let sa = std::slice::from_raw_parts(ap, len);
        let sb = std::slice::from_raw_parts(bp, len);
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                dst[i] = dst[i] + OpA::apply(sa[i]) * OpB::apply(sb[i]);
            }
        });
    } else {
        let mut dp = dp;
        let mut ap = ap;
        let mut bp = bp;
        for _ in 0..len {
            *dp = *dp + OpA::apply(*ap) * OpB::apply(*bp);
            dp = dp.offset(ds);
            ap = ap.offset(a_s);
            bp = bp.offset(b_s);
        }
    }
}

/// Inner loop for dot: `acc += OpA::apply(a[i]) * OpB::apply(b[i])`.
#[inline(always)]
unsafe fn inner_loop_dot<
    A: Copy + Mul<B, Output = R>,
    B: Copy,
    R: Copy + Add<R, Output = R>,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
>(
    ap: *const A,
    a_s: isize,
    bp: *const B,
    b_s: isize,
    len: usize,
    mut acc: R,
) -> R {
    if a_s == 1 && b_s == 1 {
        let sa = std::slice::from_raw_parts(ap, len);
        let sb = std::slice::from_raw_parts(bp, len);
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                acc = acc + OpA::apply(sa[i]) * OpB::apply(sb[i]);
            }
        });
    } else {
        let mut ap = ap;
        let mut bp = bp;
        for _ in 0..len {
            acc = acc + OpA::apply(*ap) * OpB::apply(*bp);
            ap = ap.offset(a_s);
            bp = bp.offset(b_s);
        }
    }
    acc
}

/// Copy elements from source to destination: `dest[i] = src[i]`.
pub fn copy_into<T: Copy + MaybeSendSync, Op: ElementOp<T>>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T, Op>,
) -> Result<()> {
    ensure_same_shape(dest.dims(), src.dims())?;

    let dst_ptr = dest.as_mut_ptr();
    let src_ptr = src.ptr();
    let dst_dims = dest.dims();
    let dst_strides = dest.strides();
    let src_strides = src.strides();

    if sequential_contiguous_layout(dst_dims, &[dst_strides, src_strides]).is_some() {
        let len = total_len(dst_dims);
        if Op::IS_IDENTITY {
            debug_assert!(
                {
                    let nbytes = len
                        .checked_mul(std::mem::size_of::<T>())
                        .expect("copy size must not overflow");
                    let dst_start = dst_ptr as usize;
                    let src_start = src_ptr as usize;
                    let dst_end = dst_start.saturating_add(nbytes);
                    let src_end = src_start.saturating_add(nbytes);
                    dst_end <= src_start || src_end <= dst_start
                },
                "overlapping src/dest is not supported"
            );
            unsafe { std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, len) };
        } else {
            let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, len) };
            let src = unsafe { std::slice::from_raw_parts(src_ptr, len) };
            simd::dispatch_if_large(len, || {
                for i in 0..len {
                    dst[i] = Op::apply(src[i]);
                }
            });
        }
        return Ok(());
    }

    map_into(dest, src, |x| x)
}

/// Copy elements from `src` to `dst`, optimized for col-major destination.
///
/// Delegates to `strided_perm::copy_into_col_major` for the actual work.
pub fn copy_into_col_major<T: Copy + MaybeSendSync>(
    dst: &mut StridedViewMut<T>,
    src: &StridedView<T>,
) -> Result<()> {
    strided_perm::copy_into_col_major(dst, src)
}

/// Element-wise addition: `dest[i] += src[i]`.
///
/// Source may have a different element type from destination.
pub fn add<
    D: Copy + Add<S, Output = D> + MaybeSendSync,
    S: Copy + MaybeSendSync,
    Op: ElementOp<S>,
>(
    dest: &mut StridedViewMut<D>,
    src: &StridedView<S, Op>,
) -> Result<()> {
    ensure_same_shape(dest.dims(), src.dims())?;

    let dst_ptr = dest.as_mut_ptr();
    let src_ptr = src.ptr();
    let dst_dims = dest.dims();
    let dst_strides = dest.strides();
    let src_strides = src.strides();

    if sequential_contiguous_layout(dst_dims, &[dst_strides, src_strides]).is_some() {
        let len = total_len(dst_dims);
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, len) };
        let src = unsafe { std::slice::from_raw_parts(src_ptr, len) };
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                dst[i] = dst[i] + Op::apply(src[i]);
            }
        });
        return Ok(());
    }

    let strides_list: [&[isize]; 2] = [dst_strides, src_strides];
    let elem_size = std::mem::size_of::<D>().max(std::mem::size_of::<S>());

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(dst_dims, &strides_list, Some(0), elem_size);

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        if total > MINTHREADLENGTH && rayon::current_num_threads() > 1 {
            let dst_send = SendPtr(dst_ptr);
            let src_send = SendPtr(src_ptr as *mut S);

            let costs = compute_costs(&ordered_strides);
            let initial_offsets = vec![0isize; strides_list.len()];
            let nthreads = rayon::current_num_threads();

            return mapreduce_threaded(
                &fused_dims,
                &plan.block,
                &ordered_strides,
                &initial_offsets,
                &costs,
                nthreads,
                0,
                1,
                &|dims, blocks, strides_list, offsets| {
                    for_each_inner_block_with_offsets(
                        dims,
                        blocks,
                        strides_list,
                        offsets,
                        |offsets, len, strides| {
                            unsafe {
                                inner_loop_add::<D, S, Op>(
                                    dst_send.as_ptr().offset(offsets[0]),
                                    strides[0],
                                    src_send.as_const().offset(offsets[1]),
                                    strides[1],
                                    len,
                                )
                            };
                            Ok(())
                        },
                    )
                },
            );
        }
    }

    let initial_offsets = vec![0isize; ordered_strides.len()];
    for_each_inner_block_preordered(
        &fused_dims,
        &plan.block,
        &ordered_strides,
        &initial_offsets,
        |offsets, len, strides| {
            unsafe {
                inner_loop_add::<D, S, Op>(
                    dst_ptr.offset(offsets[0]),
                    strides[0],
                    src_ptr.offset(offsets[1]),
                    strides[1],
                    len,
                )
            };
            Ok(())
        },
    )
}

/// Element-wise multiplication: `dest[i] *= src[i]`.
///
/// Source may have a different element type from destination.
pub fn mul<
    D: Copy + Mul<S, Output = D> + MaybeSendSync,
    S: Copy + MaybeSendSync,
    Op: ElementOp<S>,
>(
    dest: &mut StridedViewMut<D>,
    src: &StridedView<S, Op>,
) -> Result<()> {
    ensure_same_shape(dest.dims(), src.dims())?;

    let dst_ptr = dest.as_mut_ptr();
    let src_ptr = src.ptr();
    let dst_dims = dest.dims();
    let dst_strides = dest.strides();
    let src_strides = src.strides();

    if sequential_contiguous_layout(dst_dims, &[dst_strides, src_strides]).is_some() {
        let len = total_len(dst_dims);
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, len) };
        let src = unsafe { std::slice::from_raw_parts(src_ptr, len) };
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                dst[i] = dst[i] * Op::apply(src[i]);
            }
        });
        return Ok(());
    }

    let strides_list: [&[isize]; 2] = [dst_strides, src_strides];
    let elem_size = std::mem::size_of::<D>().max(std::mem::size_of::<S>());

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(dst_dims, &strides_list, Some(0), elem_size);

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        if total > MINTHREADLENGTH && rayon::current_num_threads() > 1 {
            let dst_send = SendPtr(dst_ptr);
            let src_send = SendPtr(src_ptr as *mut S);

            let costs = compute_costs(&ordered_strides);
            let initial_offsets = vec![0isize; strides_list.len()];
            let nthreads = rayon::current_num_threads();

            return mapreduce_threaded(
                &fused_dims,
                &plan.block,
                &ordered_strides,
                &initial_offsets,
                &costs,
                nthreads,
                0,
                1,
                &|dims, blocks, strides_list, offsets| {
                    for_each_inner_block_with_offsets(
                        dims,
                        blocks,
                        strides_list,
                        offsets,
                        |offsets, len, strides| {
                            unsafe {
                                inner_loop_mul::<D, S, Op>(
                                    dst_send.as_ptr().offset(offsets[0]),
                                    strides[0],
                                    src_send.as_const().offset(offsets[1]),
                                    strides[1],
                                    len,
                                )
                            };
                            Ok(())
                        },
                    )
                },
            );
        }
    }

    let initial_offsets = vec![0isize; ordered_strides.len()];
    for_each_inner_block_preordered(
        &fused_dims,
        &plan.block,
        &ordered_strides,
        &initial_offsets,
        |offsets, len, strides| {
            unsafe {
                inner_loop_mul::<D, S, Op>(
                    dst_ptr.offset(offsets[0]),
                    strides[0],
                    src_ptr.offset(offsets[1]),
                    strides[1],
                    len,
                )
            };
            Ok(())
        },
    )
}

/// AXPY: `dest[i] = alpha * src[i] + dest[i]`.
///
/// Alpha, source, and destination may have different element types.
pub fn axpy<D, S, A, Op>(
    dest: &mut StridedViewMut<D>,
    src: &StridedView<S, Op>,
    alpha: A,
) -> Result<()>
where
    A: Copy + Mul<S, Output = D> + MaybeSync,
    D: Copy + Add<D, Output = D> + MaybeSendSync,
    S: Copy + MaybeSendSync,
    Op: ElementOp<S>,
{
    ensure_same_shape(dest.dims(), src.dims())?;

    let dst_ptr = dest.as_mut_ptr();
    let src_ptr = src.ptr();
    let dst_dims = dest.dims();
    let dst_strides = dest.strides();
    let src_strides = src.strides();

    if sequential_contiguous_layout(dst_dims, &[dst_strides, src_strides]).is_some() {
        let len = total_len(dst_dims);
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, len) };
        let src = unsafe { std::slice::from_raw_parts(src_ptr, len) };
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                dst[i] = alpha * Op::apply(src[i]) + dst[i];
            }
        });
        return Ok(());
    }

    let strides_list: [&[isize]; 2] = [dst_strides, src_strides];
    let elem_size = std::mem::size_of::<D>().max(std::mem::size_of::<S>());

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(dst_dims, &strides_list, Some(0), elem_size);

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        if total > MINTHREADLENGTH && rayon::current_num_threads() > 1 {
            let dst_send = SendPtr(dst_ptr);
            let src_send = SendPtr(src_ptr as *mut S);

            let costs = compute_costs(&ordered_strides);
            let initial_offsets = vec![0isize; strides_list.len()];
            let nthreads = rayon::current_num_threads();

            return mapreduce_threaded(
                &fused_dims,
                &plan.block,
                &ordered_strides,
                &initial_offsets,
                &costs,
                nthreads,
                0,
                1,
                &|dims, blocks, strides_list, offsets| {
                    for_each_inner_block_with_offsets(
                        dims,
                        blocks,
                        strides_list,
                        offsets,
                        |offsets, len, strides| {
                            unsafe {
                                inner_loop_axpy::<D, S, A, Op>(
                                    dst_send.as_ptr().offset(offsets[0]),
                                    strides[0],
                                    src_send.as_const().offset(offsets[1]),
                                    strides[1],
                                    len,
                                    alpha,
                                )
                            };
                            Ok(())
                        },
                    )
                },
            );
        }
    }

    let initial_offsets = vec![0isize; ordered_strides.len()];
    for_each_inner_block_preordered(
        &fused_dims,
        &plan.block,
        &ordered_strides,
        &initial_offsets,
        |offsets, len, strides| {
            unsafe {
                inner_loop_axpy::<D, S, A, Op>(
                    dst_ptr.offset(offsets[0]),
                    strides[0],
                    src_ptr.offset(offsets[1]),
                    strides[1],
                    len,
                    alpha,
                )
            };
            Ok(())
        },
    )
}

/// Fused multiply-add: `dest[i] += OpA::apply(a[i]) * OpB::apply(b[i])`.
///
/// Operands may have different element types. Element operations are applied lazily.
pub fn fma<D, A, B, OpA, OpB>(
    dest: &mut StridedViewMut<D>,
    a: &StridedView<A, OpA>,
    b: &StridedView<B, OpB>,
) -> Result<()>
where
    A: Copy + Mul<B, Output = D> + MaybeSendSync,
    B: Copy + MaybeSendSync,
    D: Copy + Add<D, Output = D> + MaybeSendSync,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
{
    ensure_same_shape(dest.dims(), a.dims())?;
    ensure_same_shape(dest.dims(), b.dims())?;

    let dst_ptr = dest.as_mut_ptr();
    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let dst_dims = dest.dims();
    let dst_strides = dest.strides();
    let a_strides = a.strides();
    let b_strides = b.strides();

    if sequential_contiguous_layout(dst_dims, &[dst_strides, a_strides, b_strides]).is_some() {
        let len = total_len(dst_dims);
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, len) };
        let sa = unsafe { std::slice::from_raw_parts(a_ptr, len) };
        let sb = unsafe { std::slice::from_raw_parts(b_ptr, len) };
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                dst[i] = dst[i] + OpA::apply(sa[i]) * OpB::apply(sb[i]);
            }
        });
        return Ok(());
    }

    let strides_list: [&[isize]; 3] = [dst_strides, a_strides, b_strides];
    let elem_size = std::mem::size_of::<D>()
        .max(std::mem::size_of::<A>())
        .max(std::mem::size_of::<B>());

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(dst_dims, &strides_list, Some(0), elem_size);

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        if total > MINTHREADLENGTH && rayon::current_num_threads() > 1 {
            let dst_send = SendPtr(dst_ptr);
            let a_send = SendPtr(a_ptr as *mut A);
            let b_send = SendPtr(b_ptr as *mut B);

            let costs = compute_costs(&ordered_strides);
            let initial_offsets = vec![0isize; strides_list.len()];
            let nthreads = rayon::current_num_threads();

            return mapreduce_threaded(
                &fused_dims,
                &plan.block,
                &ordered_strides,
                &initial_offsets,
                &costs,
                nthreads,
                0,
                1,
                &|dims, blocks, strides_list, offsets| {
                    for_each_inner_block_with_offsets(
                        dims,
                        blocks,
                        strides_list,
                        offsets,
                        |offsets, len, strides| {
                            unsafe {
                                inner_loop_fma::<D, A, B, OpA, OpB>(
                                    dst_send.as_ptr().offset(offsets[0]),
                                    strides[0],
                                    a_send.as_const().offset(offsets[1]),
                                    strides[1],
                                    b_send.as_const().offset(offsets[2]),
                                    strides[2],
                                    len,
                                )
                            };
                            Ok(())
                        },
                    )
                },
            );
        }
    }

    let initial_offsets = vec![0isize; ordered_strides.len()];
    for_each_inner_block_preordered(
        &fused_dims,
        &plan.block,
        &ordered_strides,
        &initial_offsets,
        |offsets, len, strides| {
            unsafe {
                inner_loop_fma::<D, A, B, OpA, OpB>(
                    dst_ptr.offset(offsets[0]),
                    strides[0],
                    a_ptr.offset(offsets[1]),
                    strides[1],
                    b_ptr.offset(offsets[2]),
                    strides[2],
                    len,
                )
            };
            Ok(())
        },
    )
}

#[cfg(feature = "parallel")]
fn parallel_simd_sum<T: Copy + Zero + Add<Output = T> + simd::MaybeSimdOps + Send + Sync>(
    src: &[T],
) -> Option<T> {
    use rayon::prelude::*;
    // Check that T has SIMD support
    if T::try_simd_sum(&[]).is_none() {
        return None;
    }
    let nthreads = rayon::current_num_threads();
    let chunk_size = (src.len() + nthreads - 1) / nthreads;
    let result = src
        .par_chunks(chunk_size)
        .map(|chunk| T::try_simd_sum(chunk).unwrap())
        .reduce(|| T::zero(), |a, b| a + b);
    Some(result)
}

/// Sum all elements: `sum(src)`.
pub fn sum<
    T: Copy + Zero + Add<Output = T> + MaybeSendSync + simd::MaybeSimdOps,
    Op: ElementOp<T>,
>(
    src: &StridedView<T, Op>,
) -> Result<T> {
    // SIMD fast path: contiguous Identity view with SIMD support
    if Op::IS_IDENTITY {
        if same_contiguous_layout(src.dims(), &[src.strides()]).is_some() {
            let len = total_len(src.dims());
            let src_slice = unsafe { std::slice::from_raw_parts(src.ptr(), len) };

            #[cfg(feature = "parallel")]
            if len > MINTHREADLENGTH {
                if let Some(result) = parallel_simd_sum(src_slice) {
                    return Ok(result);
                }
            }

            if let Some(result) = T::try_simd_sum(src_slice) {
                return Ok(result);
            }
        }
    }
    reduce(src, |x| x, |a, b| a + b, T::zero())
}

/// Dot product: `sum(OpA::apply(a[i]) * OpB::apply(b[i]))`.
///
/// Operands may have different element types. Result type `R` must be `A * B`.
/// SIMD fast path fires only when `A == B == R` (same type) and both Identity ops.
pub fn dot<A, B, R, OpA, OpB>(a: &StridedView<A, OpA>, b: &StridedView<B, OpB>) -> Result<R>
where
    A: Copy + Mul<B, Output = R> + MaybeSendSync + 'static,
    B: Copy + MaybeSendSync + 'static,
    R: Copy + Zero + Add<Output = R> + MaybeSendSync + simd::MaybeSimdOps + 'static,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
{
    ensure_same_shape(a.dims(), b.dims())?;

    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let a_strides = a.strides();
    let b_strides = b.strides();
    let a_dims = a.dims();

    if same_contiguous_layout(a_dims, &[a_strides, b_strides]).is_some() {
        let len = total_len(a_dims);

        // SIMD fast path: both contiguous, both Identity ops, same type
        if OpA::IS_IDENTITY
            && OpB::IS_IDENTITY
            && std::any::TypeId::of::<A>() == std::any::TypeId::of::<R>()
            && std::any::TypeId::of::<B>() == std::any::TypeId::of::<R>()
        {
            let sa = unsafe { std::slice::from_raw_parts(a_ptr as *const R, len) };
            let sb = unsafe { std::slice::from_raw_parts(b_ptr as *const R, len) };
            if let Some(result) = R::try_simd_dot(sa, sb) {
                return Ok(result);
            }
        }

        // Generic contiguous fast path
        let sa = unsafe { std::slice::from_raw_parts(a_ptr, len) };
        let sb = unsafe { std::slice::from_raw_parts(b_ptr, len) };
        let mut acc = R::zero();
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                acc = acc + OpA::apply(sa[i]) * OpB::apply(sb[i]);
            }
        });
        return Ok(acc);
    }

    let strides_list: [&[isize]; 2] = [a_strides, b_strides];
    let elem_size = std::mem::size_of::<A>()
        .max(std::mem::size_of::<B>())
        .max(std::mem::size_of::<R>());

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(a_dims, &strides_list, None, elem_size);

    let mut acc = R::zero();
    let initial_offsets = vec![0isize; ordered_strides.len()];
    for_each_inner_block_preordered(
        &fused_dims,
        &plan.block,
        &ordered_strides,
        &initial_offsets,
        |offsets, len, strides| {
            acc = unsafe {
                inner_loop_dot::<A, B, R, OpA, OpB>(
                    a_ptr.offset(offsets[0]),
                    strides[0],
                    b_ptr.offset(offsets[1]),
                    strides[1],
                    len,
                    acc,
                )
            };
            Ok(())
        },
    )?;

    Ok(acc)
}

/// Symmetrize a square matrix: `dest = (src + src^T) / 2`.
pub fn symmetrize_into<T>(dest: &mut StridedViewMut<T>, src: &StridedView<T>) -> Result<()>
where
    T: Copy
        + Add<Output = T>
        + Mul<Output = T>
        + num_traits::FromPrimitive
        + std::ops::Div<Output = T>
        + MaybeSendSync,
{
    if src.ndim() != 2 {
        return Err(StridedError::RankMismatch(src.ndim(), 2));
    }
    let rows = src.dims()[0];
    let cols = src.dims()[1];
    if rows != cols {
        return Err(StridedError::NonSquare { rows, cols });
    }

    let src_t = src.permute(&[1, 0])?;
    let half = T::from_f64(0.5).ok_or(StridedError::ScalarConversion)?;

    zip_map2_into(dest, src, &src_t, |a, b| (a + b) * half)
}

/// Conjugate-symmetrize a square matrix: `dest = (src + conj(src^T)) / 2`.
pub fn symmetrize_conj_into<T>(dest: &mut StridedViewMut<T>, src: &StridedView<T>) -> Result<()>
where
    T: Copy
        + ElementOpApply
        + Add<Output = T>
        + Mul<Output = T>
        + num_traits::FromPrimitive
        + std::ops::Div<Output = T>
        + MaybeSendSync,
{
    if src.ndim() != 2 {
        return Err(StridedError::RankMismatch(src.ndim(), 2));
    }
    let rows = src.dims()[0];
    let cols = src.dims()[1];
    if rows != cols {
        return Err(StridedError::NonSquare { rows, cols });
    }

    // adjoint = conj + transpose
    let src_adj = src.adjoint_2d()?;
    let half = T::from_f64(0.5).ok_or(StridedError::ScalarConversion)?;

    zip_map2_into(dest, src, &src_adj, |a, b| (a + b) * half)
}

/// Copy with scaling: `dest[i] = scale * src[i]`.
///
/// Scale, source, and destination may have different element types.
pub fn copy_scale<D, S, A, Op>(
    dest: &mut StridedViewMut<D>,
    src: &StridedView<S, Op>,
    scale: A,
) -> Result<()>
where
    A: Copy + Mul<S, Output = D> + MaybeSync,
    D: Copy + MaybeSendSync,
    S: Copy + MaybeSendSync,
    Op: ElementOp<S>,
{
    map_into(dest, src, |x| scale * x)
}

/// Copy with complex conjugation: `dest[i] = conj(src[i])`.
pub fn copy_conj<T: Copy + ElementOpApply + MaybeSendSync>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T>,
) -> Result<()> {
    let src_conj = src.conj();
    copy_into(dest, &src_conj)
}

#[inline]
fn element_transpose_is_identity<T: 'static>() -> bool {
    use std::any::TypeId;

    macro_rules! matches_type {
        ($($ty:ty),* $(,)?) => {{
            let id = TypeId::of::<T>();
            false $(|| id == TypeId::of::<$ty>())*
        }};
    }

    matches_type!(
        f32,
        f64,
        i8,
        i16,
        i32,
        i64,
        i128,
        isize,
        u8,
        u16,
        u32,
        u64,
        u128,
        usize,
        num_complex::Complex32,
        num_complex::Complex64,
    )
}

#[inline]
fn element_zero_is_all_bits_zero<T: 'static>() -> bool {
    use std::any::TypeId;

    macro_rules! matches_type {
        ($($ty:ty),* $(,)?) => {{
            let id = TypeId::of::<T>();
            false $(|| id == TypeId::of::<$ty>())*
        }};
    }

    matches_type!(
        f32,
        f64,
        i8,
        i16,
        i32,
        i64,
        i128,
        isize,
        u8,
        u16,
        u32,
        u64,
        u128,
        usize,
        num_complex::Complex32,
        num_complex::Complex64,
    )
}

#[inline]
unsafe fn fill_2d<T: Copy + MaybeSendSync>(
    dst: *mut T,
    dim0: usize,
    dim1: usize,
    dst_stride0: isize,
    dst_stride1: isize,
    value: T,
) {
    #[cfg(feature = "parallel")]
    {
        let total = dim0.saturating_mul(dim1);
        if total > MINTHREADLENGTH && rayon::current_num_threads() > 1 {
            let dst_send = SendPtr(dst);
            if dst_stride0.unsigned_abs() <= dst_stride1.unsigned_abs() {
                (0..dim1).into_par_iter().for_each(|j| {
                    let dst = dst_send.as_ptr();
                    unsafe {
                        let base = j as isize * dst_stride1;
                        for i in 0..dim0 {
                            *dst.offset(base + i as isize * dst_stride0) = value;
                        }
                    }
                });
            } else {
                (0..dim0).into_par_iter().for_each(|i| {
                    let dst = dst_send.as_ptr();
                    unsafe {
                        let base = i as isize * dst_stride0;
                        for j in 0..dim1 {
                            *dst.offset(base + j as isize * dst_stride1) = value;
                        }
                    }
                });
            }
            return;
        }
    }

    if dst_stride0.unsigned_abs() <= dst_stride1.unsigned_abs() {
        for j in 0..dim1 {
            let base = j as isize * dst_stride1;
            for i in 0..dim0 {
                *dst.offset(base + i as isize * dst_stride0) = value;
            }
        }
    } else {
        for i in 0..dim0 {
            let base = i as isize * dst_stride0;
            for j in 0..dim1 {
                *dst.offset(base + j as isize * dst_stride1) = value;
            }
        }
    }
}

#[inline]
unsafe fn fill_contiguous<T>(dst: *mut T, len: usize, value: T)
where
    T: Copy + Zero + PartialEq + MaybeSendSync + 'static,
{
    if element_zero_is_all_bits_zero::<T>() && value == T::zero() {
        std::ptr::write_bytes(dst, 0, len);
        return;
    }

    let dst = std::slice::from_raw_parts_mut(dst, len);
    dst.fill(value);
}

#[inline(always)]
unsafe fn transpose_scale_4x4_f64(
    dst: *mut f64,
    dst_stride0: isize,
    dst_stride1: isize,
    src: *const f64,
    src_stride0: isize,
    src_stride1: isize,
    i: usize,
    j: usize,
    scale: f64,
) {
    let src_base = src.offset(i as isize * src_stride0 + j as isize * src_stride1);

    let s00 = *src_base;
    let s10 = *src_base.offset(src_stride0);
    let s20 = *src_base.offset(2 * src_stride0);
    let s30 = *src_base.offset(3 * src_stride0);

    let src_col1 = src_base.offset(src_stride1);
    let s01 = *src_col1;
    let s11 = *src_col1.offset(src_stride0);
    let s21 = *src_col1.offset(2 * src_stride0);
    let s31 = *src_col1.offset(3 * src_stride0);

    let src_col2 = src_base.offset(2 * src_stride1);
    let s02 = *src_col2;
    let s12 = *src_col2.offset(src_stride0);
    let s22 = *src_col2.offset(2 * src_stride0);
    let s32 = *src_col2.offset(3 * src_stride0);

    let src_col3 = src_base.offset(3 * src_stride1);
    let s03 = *src_col3;
    let s13 = *src_col3.offset(src_stride0);
    let s23 = *src_col3.offset(2 * src_stride0);
    let s33 = *src_col3.offset(3 * src_stride0);

    let dst_row0 = dst.offset(j as isize * dst_stride0 + i as isize * dst_stride1);
    *dst_row0 = scale * s00;
    *dst_row0.offset(dst_stride0) = scale * s01;
    *dst_row0.offset(2 * dst_stride0) = scale * s02;
    *dst_row0.offset(3 * dst_stride0) = scale * s03;

    let dst_row1 = dst_row0.offset(dst_stride1);
    *dst_row1 = scale * s10;
    *dst_row1.offset(dst_stride0) = scale * s11;
    *dst_row1.offset(2 * dst_stride0) = scale * s12;
    *dst_row1.offset(3 * dst_stride0) = scale * s13;

    let dst_row2 = dst_row0.offset(2 * dst_stride1);
    *dst_row2 = scale * s20;
    *dst_row2.offset(dst_stride0) = scale * s21;
    *dst_row2.offset(2 * dst_stride0) = scale * s22;
    *dst_row2.offset(3 * dst_stride0) = scale * s23;

    let dst_row3 = dst_row0.offset(3 * dst_stride1);
    *dst_row3 = scale * s30;
    *dst_row3.offset(dst_stride0) = scale * s31;
    *dst_row3.offset(2 * dst_stride0) = scale * s32;
    *dst_row3.offset(3 * dst_stride0) = scale * s33;
}

#[inline]
unsafe fn copy_transpose_scale_2d_f64_tiled_raw(
    dst: *mut f64,
    dst_stride0: isize,
    dst_stride1: isize,
    src: *const f64,
    src_stride0: isize,
    src_stride1: isize,
    src_rows: usize,
    src_cols: usize,
    scale: f64,
) {
    const TILE: usize = 4;
    let row_full = src_rows / TILE * TILE;
    let col_full = src_cols / TILE * TILE;

    #[cfg(feature = "parallel")]
    {
        let total = src_rows.saturating_mul(src_cols);
        if total > MINTHREADLENGTH && rayon::current_num_threads() > 1 {
            let dst_send = SendPtr(dst);
            let src_send = SendPtr(src as *mut f64);
            let row_tiles = row_full / TILE;
            (0..row_tiles).into_par_iter().for_each(|tile_i| {
                let i = tile_i * TILE;
                let dst = dst_send.as_ptr();
                let src = src_send.as_const();
                unsafe {
                    let mut j = 0;
                    while j < col_full {
                        transpose_scale_4x4_f64(
                            dst,
                            dst_stride0,
                            dst_stride1,
                            src,
                            src_stride0,
                            src_stride1,
                            i,
                            j,
                            scale,
                        );
                        j += TILE;
                    }
                    for j in col_full..src_cols {
                        for ii in i..i + TILE {
                            *dst.offset(j as isize * dst_stride0 + ii as isize * dst_stride1) =
                                scale
                                    * *src.offset(
                                        ii as isize * src_stride0 + j as isize * src_stride1,
                                    );
                        }
                    }
                }
            });
            for i in row_full..src_rows {
                for j in 0..src_cols {
                    *dst.offset(j as isize * dst_stride0 + i as isize * dst_stride1) =
                        scale * *src.offset(i as isize * src_stride0 + j as isize * src_stride1);
                }
            }
            return;
        }
    }

    let mut i = 0;
    while i < row_full {
        let mut j = 0;
        while j < col_full {
            transpose_scale_4x4_f64(
                dst,
                dst_stride0,
                dst_stride1,
                src,
                src_stride0,
                src_stride1,
                i,
                j,
                scale,
            );
            j += TILE;
        }
        for j in col_full..src_cols {
            for ii in i..i + TILE {
                *dst.offset(j as isize * dst_stride0 + ii as isize * dst_stride1) =
                    scale * *src.offset(ii as isize * src_stride0 + j as isize * src_stride1);
            }
        }
        i += TILE;
    }
    for i in row_full..src_rows {
        for j in 0..src_cols {
            *dst.offset(j as isize * dst_stride0 + i as isize * dst_stride1) =
                scale * *src.offset(i as isize * src_stride0 + j as isize * src_stride1);
        }
    }
}

#[inline]
#[cfg(test)]
unsafe fn try_copy_transpose_scale_2d_f64_tiled(
    dest: &mut StridedViewMut<f64>,
    src: &StridedView<f64>,
    scale: f64,
) -> bool {
    if src.ndim() != 2 || dest.ndim() != 2 {
        return false;
    }
    let src_dims = src.dims();
    if dest.dims() != [src_dims[1], src_dims[0]] {
        return false;
    }
    if src.strides()[0] != 1 || dest.strides()[0] != 1 {
        return false;
    }

    copy_transpose_scale_2d_f64_tiled_raw(
        dest.as_mut_ptr(),
        dest.strides()[0],
        dest.strides()[1],
        src.ptr(),
        src.strides()[0],
        src.strides()[1],
        src_dims[0],
        src_dims[1],
        scale,
    );
    true
}

#[inline]
unsafe fn try_copy_transpose_scale_2d_f64_tiled_typed<T>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T>,
    scale: T,
) -> bool
where
    T: Copy + 'static,
{
    if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f64>() {
        return false;
    }

    let scale = *(&scale as *const T).cast::<f64>();
    if src.ndim() != 2 || dest.ndim() != 2 {
        return false;
    }
    let src_dims = src.dims();
    if dest.dims() != [src_dims[1], src_dims[0]] {
        return false;
    }
    if src.strides()[0] != 1 || dest.strides()[0] != 1 {
        return false;
    }

    copy_transpose_scale_2d_f64_tiled_raw(
        dest.as_mut_ptr().cast::<f64>(),
        dest.strides()[0],
        dest.strides()[1],
        src.ptr().cast::<f64>(),
        src.strides()[0],
        src.strides()[1],
        src_dims[0],
        src_dims[1],
        scale,
    );
    true
}

#[inline(always)]
unsafe fn transpose_scale_4x4_identity<T>(
    dst: *mut T,
    dst_stride0: isize,
    dst_stride1: isize,
    src: *const T,
    src_stride0: isize,
    src_stride1: isize,
    i: usize,
    j: usize,
    scale: T,
) where
    T: Copy + Mul<Output = T>,
{
    let src_base = src.offset(i as isize * src_stride0 + j as isize * src_stride1);

    let s00 = *src_base;
    let s10 = *src_base.offset(src_stride0);
    let s20 = *src_base.offset(2 * src_stride0);
    let s30 = *src_base.offset(3 * src_stride0);

    let src_col1 = src_base.offset(src_stride1);
    let s01 = *src_col1;
    let s11 = *src_col1.offset(src_stride0);
    let s21 = *src_col1.offset(2 * src_stride0);
    let s31 = *src_col1.offset(3 * src_stride0);

    let src_col2 = src_base.offset(2 * src_stride1);
    let s02 = *src_col2;
    let s12 = *src_col2.offset(src_stride0);
    let s22 = *src_col2.offset(2 * src_stride0);
    let s32 = *src_col2.offset(3 * src_stride0);

    let src_col3 = src_base.offset(3 * src_stride1);
    let s03 = *src_col3;
    let s13 = *src_col3.offset(src_stride0);
    let s23 = *src_col3.offset(2 * src_stride0);
    let s33 = *src_col3.offset(3 * src_stride0);

    let dst_row0 = dst.offset(j as isize * dst_stride0 + i as isize * dst_stride1);
    *dst_row0 = scale * s00;
    *dst_row0.offset(dst_stride0) = scale * s01;
    *dst_row0.offset(2 * dst_stride0) = scale * s02;
    *dst_row0.offset(3 * dst_stride0) = scale * s03;

    let dst_row1 = dst_row0.offset(dst_stride1);
    *dst_row1 = scale * s10;
    *dst_row1.offset(dst_stride0) = scale * s11;
    *dst_row1.offset(2 * dst_stride0) = scale * s12;
    *dst_row1.offset(3 * dst_stride0) = scale * s13;

    let dst_row2 = dst_row0.offset(2 * dst_stride1);
    *dst_row2 = scale * s20;
    *dst_row2.offset(dst_stride0) = scale * s21;
    *dst_row2.offset(2 * dst_stride0) = scale * s22;
    *dst_row2.offset(3 * dst_stride0) = scale * s23;

    let dst_row3 = dst_row0.offset(3 * dst_stride1);
    *dst_row3 = scale * s30;
    *dst_row3.offset(dst_stride0) = scale * s31;
    *dst_row3.offset(2 * dst_stride0) = scale * s32;
    *dst_row3.offset(3 * dst_stride0) = scale * s33;
}

#[inline]
unsafe fn copy_transpose_scale_2d_identity_tiled_raw<T>(
    dst: *mut T,
    dst_stride0: isize,
    dst_stride1: isize,
    src: *const T,
    src_stride0: isize,
    src_stride1: isize,
    src_rows: usize,
    src_cols: usize,
    scale: T,
) where
    T: Copy + Mul<Output = T> + MaybeSendSync,
{
    const TILE: usize = 4;
    let row_full = src_rows / TILE * TILE;
    let col_full = src_cols / TILE * TILE;

    #[cfg(feature = "parallel")]
    {
        let total = src_rows.saturating_mul(src_cols);
        if total > MINTHREADLENGTH && rayon::current_num_threads() > 1 {
            let dst_send = SendPtr(dst);
            let src_send = SendPtr(src as *mut T);
            let row_tiles = row_full / TILE;
            (0..row_tiles).into_par_iter().for_each(|tile_i| {
                let i = tile_i * TILE;
                let dst = dst_send.as_ptr();
                let src = src_send.as_const();
                unsafe {
                    let mut j = 0;
                    while j < col_full {
                        transpose_scale_4x4_identity(
                            dst,
                            dst_stride0,
                            dst_stride1,
                            src,
                            src_stride0,
                            src_stride1,
                            i,
                            j,
                            scale,
                        );
                        j += TILE;
                    }
                    for j in col_full..src_cols {
                        for ii in i..i + TILE {
                            *dst.offset(j as isize * dst_stride0 + ii as isize * dst_stride1) =
                                scale
                                    * *src.offset(
                                        ii as isize * src_stride0 + j as isize * src_stride1,
                                    );
                        }
                    }
                }
            });
            for i in row_full..src_rows {
                for j in 0..src_cols {
                    *dst.offset(j as isize * dst_stride0 + i as isize * dst_stride1) =
                        scale * *src.offset(i as isize * src_stride0 + j as isize * src_stride1);
                }
            }
            return;
        }
    }

    let mut i = 0;
    while i < row_full {
        let mut j = 0;
        while j < col_full {
            transpose_scale_4x4_identity(
                dst,
                dst_stride0,
                dst_stride1,
                src,
                src_stride0,
                src_stride1,
                i,
                j,
                scale,
            );
            j += TILE;
        }
        for j in col_full..src_cols {
            for ii in i..i + TILE {
                *dst.offset(j as isize * dst_stride0 + ii as isize * dst_stride1) =
                    scale * *src.offset(ii as isize * src_stride0 + j as isize * src_stride1);
            }
        }
        i += TILE;
    }
    for i in row_full..src_rows {
        for j in 0..src_cols {
            *dst.offset(j as isize * dst_stride0 + i as isize * dst_stride1) =
                scale * *src.offset(i as isize * src_stride0 + j as isize * src_stride1);
        }
    }
}

#[inline]
unsafe fn try_copy_transpose_scale_2d_identity_tiled<T>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T>,
    scale: T,
) -> bool
where
    T: Copy + Mul<Output = T> + MaybeSendSync,
{
    if src.ndim() != 2 || dest.ndim() != 2 {
        return false;
    }
    let src_dims = src.dims();
    if dest.dims() != [src_dims[1], src_dims[0]] {
        return false;
    }
    if src.strides()[0] != 1 || dest.strides()[0] != 1 {
        return false;
    }

    copy_transpose_scale_2d_identity_tiled_raw(
        dest.as_mut_ptr(),
        dest.strides()[0],
        dest.strides()[1],
        src.ptr(),
        src.strides()[0],
        src.strides()[1],
        src_dims[0],
        src_dims[1],
        scale,
    );
    true
}

#[inline]
unsafe fn copy_transpose_scale_2d_loop<T>(
    dst: *mut T,
    dst_stride0: isize,
    dst_stride1: isize,
    src: *const T,
    src_stride0: isize,
    src_stride1: isize,
    src_rows: usize,
    src_cols: usize,
    scale: T,
) where
    T: Copy + ElementOpApply + Mul<Output = T> + MaybeSendSync,
{
    #[cfg(feature = "parallel")]
    {
        let total = src_rows.saturating_mul(src_cols);
        if total > MINTHREADLENGTH && rayon::current_num_threads() > 1 {
            let dst_send = SendPtr(dst);
            let src_send = SendPtr(src as *mut T);
            if dst_stride0.unsigned_abs() <= dst_stride1.unsigned_abs() {
                (0..src_rows).into_par_iter().for_each(|i| {
                    let dst = dst_send.as_ptr();
                    let src = src_send.as_const();
                    unsafe {
                        for j in 0..src_cols {
                            let value = (*src
                                .offset(i as isize * src_stride0 + j as isize * src_stride1))
                            .transpose();
                            *dst.offset(j as isize * dst_stride0 + i as isize * dst_stride1) =
                                scale * value;
                        }
                    }
                });
            } else {
                (0..src_cols).into_par_iter().for_each(|j| {
                    let dst = dst_send.as_ptr();
                    let src = src_send.as_const();
                    unsafe {
                        for i in 0..src_rows {
                            let value = (*src
                                .offset(i as isize * src_stride0 + j as isize * src_stride1))
                            .transpose();
                            *dst.offset(j as isize * dst_stride0 + i as isize * dst_stride1) =
                                scale * value;
                        }
                    }
                });
            }
            return;
        }
    }

    if dst_stride0.unsigned_abs() <= dst_stride1.unsigned_abs() {
        for i in 0..src_rows {
            for j in 0..src_cols {
                let value =
                    (*src.offset(i as isize * src_stride0 + j as isize * src_stride1)).transpose();
                *dst.offset(j as isize * dst_stride0 + i as isize * dst_stride1) = scale * value;
            }
        }
    } else {
        for j in 0..src_cols {
            for i in 0..src_rows {
                let value =
                    (*src.offset(i as isize * src_stride0 + j as isize * src_stride1)).transpose();
                *dst.offset(j as isize * dst_stride0 + i as isize * dst_stride1) = scale * value;
            }
        }
    }
}

/// Copy with transpose and scaling: `dest[j,i] = scale * src[i,j]`.
pub fn copy_transpose_scale_into<T>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T>,
    scale: T,
) -> Result<()>
where
    T: Copy + ElementOpApply + Mul<Output = T> + Zero + One + PartialEq + MaybeSendSync + 'static,
{
    if src.ndim() != 2 || dest.ndim() != 2 {
        return Err(StridedError::RankMismatch(src.ndim(), 2));
    }

    let src_dims = src.dims();
    let expected_dims = [src_dims[1], src_dims[0]];
    ensure_same_shape(dest.dims(), &expected_dims)?;

    if scale == T::zero() {
        unsafe {
            if same_contiguous_layout(dest.dims(), &[dest.strides()]).is_some() {
                fill_contiguous(dest.as_mut_ptr(), total_len(dest.dims()), T::zero());
            } else {
                fill_2d(
                    dest.as_mut_ptr(),
                    dest.dims()[0],
                    dest.dims()[1],
                    dest.strides()[0],
                    dest.strides()[1],
                    T::zero(),
                );
            }
        }
        return Ok(());
    }

    let transpose_is_identity = element_transpose_is_identity::<T>();

    unsafe {
        if transpose_is_identity && try_copy_transpose_scale_2d_f64_tiled_typed(dest, src, scale) {
            return Ok(());
        }
        if transpose_is_identity && try_copy_transpose_scale_2d_identity_tiled(dest, src, scale) {
            return Ok(());
        }
    }

    if scale == T::one() && transpose_is_identity {
        let src_t = src.permute(&[1, 0])?;
        #[cfg(feature = "parallel")]
        {
            if total_len(dest.dims()) > MINTHREADLENGTH && rayon::current_num_threads() > 1 {
                return strided_perm::copy_into_par(dest, &src_t);
            }
        }
        return strided_perm::copy_into(dest, &src_t);
    }

    unsafe {
        copy_transpose_scale_2d_loop(
            dest.as_mut_ptr(),
            dest.strides()[0],
            dest.strides()[1],
            src.ptr(),
            src.strides()[0],
            src.strides()[1],
            src_dims[0],
            src_dims[1],
            scale,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tiled_tests {
    use super::*;
    use crate::view::StridedArray;

    #[test]
    fn test_f64_tiled_transpose_scale_handles_remainders() {
        let rows = 7;
        let cols = 9;
        let a = StridedArray::<f64>::from_fn_col_major(&[rows, cols], |idx| {
            (idx[0] * 100 + idx[1]) as f64
        });
        let mut out = StridedArray::<f64>::col_major(&[cols, rows]);

        let used_tiled = {
            let src = a.view();
            let mut dst = out.view_mut();
            unsafe { try_copy_transpose_scale_2d_f64_tiled(&mut dst, &src, 3.0) }
        };

        assert!(used_tiled);
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(out.get(&[j, i]), 3.0 * a.get(&[i, j]));
            }
        }
    }

    #[test]
    fn test_identity_tiled_transpose_scale_handles_integer_remainders() {
        let rows = 6;
        let cols = 5;
        let a = StridedArray::<u64>::from_fn_col_major(&[rows, cols], |idx| {
            (idx[0] * 100 + idx[1]) as u64
        });
        let mut out = StridedArray::<u64>::col_major(&[cols, rows]);

        let used_tiled = {
            let src = a.view();
            let mut dst = out.view_mut();
            unsafe { try_copy_transpose_scale_2d_identity_tiled(&mut dst, &src, 2) }
        };

        assert!(used_tiled);
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(out.get(&[j, i]), 2 * a.get(&[i, j]));
            }
        }
    }

    #[test]
    fn test_zero_scale_fills_non_contiguous_destination() {
        let rows = 3;
        let cols = 4;
        let a = StridedArray::<u64>::from_fn_col_major(&[rows, cols], |idx| {
            (idx[0] * 10 + idx[1] + 1) as u64
        });
        let mut out_base = StridedArray::<u64>::from_fn_col_major(&[rows, cols], |_| 99);
        let mut out_t = out_base.view_mut().permute(&[1, 0]).unwrap();

        copy_transpose_scale_into(&mut out_t, &a.view(), 0).unwrap();

        for i in 0..cols {
            for j in 0..rows {
                assert_eq!(out_t.get(&[i, j]), 0);
            }
        }
    }
}
