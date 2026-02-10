//! High-level operations on dynamic-rank strided views.

use crate::kernel::{
    build_plan_fused, ensure_same_shape, for_each_inner_block_preordered, same_contiguous_layout,
    sequential_contiguous_layout, total_len,
};
use crate::map_view::{map_into, zip_map2_into};
use crate::maybe_sync::MaybeSendSync;
use crate::reduce_view::reduce;
use crate::simd;
use crate::view::{StridedView, StridedViewMut};
use crate::{Result, StridedError};
use num_traits::Zero;
use std::any::TypeId;
use std::ops::{Add, Mul};
use strided_view::{ElementOp, ElementOpApply};

#[cfg(feature = "parallel")]
use crate::fuse::compute_costs;
#[cfg(feature = "parallel")]
use crate::threading::{
    for_each_inner_block_with_offsets, mapreduce_threaded, SendPtr, MINTHREADLENGTH,
};

// ============================================================================
// Stride-specialized inner loop helpers for ops_view
//
// When all inner strides are 1 (contiguous in the innermost dimension),
// we use slice-based iteration so LLVM can auto-vectorize effectively.
// This mirrors the inner_loop_map* helpers in map_view.rs.
// ============================================================================

/// Inner loop for add: `dst[i] += Op::apply(src[i])`.
#[inline(always)]
unsafe fn inner_loop_add<T: Copy + ElementOpApply + Add<Output = T>, Op: ElementOp>(
    dp: *mut T,
    ds: isize,
    sp: *const T,
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
unsafe fn inner_loop_mul<T: Copy + ElementOpApply + Mul<Output = T>, Op: ElementOp>(
    dp: *mut T,
    ds: isize,
    sp: *const T,
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
    T: Copy + ElementOpApply + Mul<Output = T> + Add<Output = T>,
    Op: ElementOp,
>(
    dp: *mut T,
    ds: isize,
    sp: *const T,
    ss: isize,
    len: usize,
    alpha: T,
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

/// Inner loop for fma: `dst[i] += a[i] * b[i]`.
#[inline(always)]
unsafe fn inner_loop_fma<T: Copy + Mul<Output = T> + Add<Output = T>>(
    dp: *mut T,
    ds: isize,
    ap: *const T,
    a_s: isize,
    bp: *const T,
    b_s: isize,
    len: usize,
) {
    if ds == 1 && a_s == 1 && b_s == 1 {
        let dst = std::slice::from_raw_parts_mut(dp, len);
        let sa = std::slice::from_raw_parts(ap, len);
        let sb = std::slice::from_raw_parts(bp, len);
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                dst[i] = dst[i] + sa[i] * sb[i];
            }
        });
    } else {
        let mut dp = dp;
        let mut ap = ap;
        let mut bp = bp;
        for _ in 0..len {
            *dp = *dp + *ap * *bp;
            dp = dp.offset(ds);
            ap = ap.offset(a_s);
            bp = bp.offset(b_s);
        }
    }
}

/// Inner loop for dot: `acc += OpA::apply(a[i]) * OpB::apply(b[i])`.
#[inline(always)]
unsafe fn inner_loop_dot<
    T: Copy + ElementOpApply + Mul<Output = T> + Add<Output = T>,
    OpA: ElementOp,
    OpB: ElementOp,
>(
    ap: *const T,
    a_s: isize,
    bp: *const T,
    b_s: isize,
    len: usize,
    mut acc: T,
) -> T {
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
pub fn copy_into<T: Copy + ElementOpApply + MaybeSendSync, Op: ElementOp>(
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

/// Element-wise addition: `dest[i] += src[i]`.
pub fn add<T: Copy + ElementOpApply + Add<Output = T> + MaybeSendSync, Op: ElementOp>(
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

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(dst_dims, &strides_list, Some(0), std::mem::size_of::<T>());

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        if total > MINTHREADLENGTH {
            let dst_send = SendPtr(dst_ptr);
            let src_send = SendPtr(src_ptr as *mut T);

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
                                inner_loop_add::<T, Op>(
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
                inner_loop_add::<T, Op>(
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
pub fn mul<T: Copy + ElementOpApply + Mul<Output = T> + MaybeSendSync, Op: ElementOp>(
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

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(dst_dims, &strides_list, Some(0), std::mem::size_of::<T>());

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        if total > MINTHREADLENGTH {
            let dst_send = SendPtr(dst_ptr);
            let src_send = SendPtr(src_ptr as *mut T);

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
                                inner_loop_mul::<T, Op>(
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
                inner_loop_mul::<T, Op>(
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
pub fn axpy<
    T: Copy + ElementOpApply + Mul<Output = T> + Add<Output = T> + MaybeSendSync,
    Op: ElementOp,
>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T, Op>,
    alpha: T,
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
                dst[i] = alpha * Op::apply(src[i]) + dst[i];
            }
        });
        return Ok(());
    }

    let strides_list: [&[isize]; 2] = [dst_strides, src_strides];

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(dst_dims, &strides_list, Some(0), std::mem::size_of::<T>());

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        if total > MINTHREADLENGTH {
            let dst_send = SendPtr(dst_ptr);
            let src_send = SendPtr(src_ptr as *mut T);

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
                                inner_loop_axpy::<T, Op>(
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
                inner_loop_axpy::<T, Op>(
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

/// Fused multiply-add: `dest[i] += a[i] * b[i]`.
pub fn fma<T: Copy + ElementOpApply + Mul<Output = T> + Add<Output = T> + MaybeSendSync>(
    dest: &mut StridedViewMut<T>,
    a: &StridedView<T>,
    b: &StridedView<T>,
) -> Result<()> {
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
                dst[i] = dst[i] + sa[i] * sb[i];
            }
        });
        return Ok(());
    }

    let strides_list: [&[isize]; 3] = [dst_strides, a_strides, b_strides];

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(dst_dims, &strides_list, Some(0), std::mem::size_of::<T>());

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        if total > MINTHREADLENGTH {
            let dst_send = SendPtr(dst_ptr);
            let a_send = SendPtr(a_ptr as *mut T);
            let b_send = SendPtr(b_ptr as *mut T);

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
                                inner_loop_fma::<T>(
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
                inner_loop_fma::<T>(
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

/// Sum all elements: `sum(src)`.
pub fn sum<
    T: Copy + ElementOpApply + Zero + Add<Output = T> + MaybeSendSync + 'static,
    Op: ElementOp,
>(
    src: &StridedView<T, Op>,
) -> Result<T> {
    #[cfg(feature = "simd")]
    {
        let len = total_len(src.dims());
        if same_contiguous_layout(src.dims(), &[src.strides()]).is_some() {
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let src_slice = unsafe { std::slice::from_raw_parts(src.ptr() as *const f32, len) };
                #[cfg(feature = "parallel")]
                if len > MINTHREADLENGTH {
                    use rayon::prelude::*;
                    let nthreads = rayon::current_num_threads();
                    let chunk_size = (len + nthreads - 1) / nthreads;
                    let out: f32 = src_slice
                        .par_chunks(chunk_size)
                        .map(|chunk| simd::sum_f32(chunk))
                        .sum();
                    return Ok(unsafe { std::mem::transmute_copy::<f32, T>(&out) });
                }
                let out = simd::sum_f32(src_slice);
                return Ok(unsafe { std::mem::transmute_copy::<f32, T>(&out) });
            }
            if TypeId::of::<T>() == TypeId::of::<f64>() {
                let src_slice = unsafe { std::slice::from_raw_parts(src.ptr() as *const f64, len) };
                #[cfg(feature = "parallel")]
                if len > MINTHREADLENGTH {
                    use rayon::prelude::*;
                    let nthreads = rayon::current_num_threads();
                    let chunk_size = (len + nthreads - 1) / nthreads;
                    let out: f64 = src_slice
                        .par_chunks(chunk_size)
                        .map(|chunk| simd::sum_f64(chunk))
                        .sum();
                    return Ok(unsafe { std::mem::transmute_copy::<f64, T>(&out) });
                }
                let out = simd::sum_f64(src_slice);
                return Ok(unsafe { std::mem::transmute_copy::<f64, T>(&out) });
            }
        }
    }
    reduce(src, |x| x, |a, b| a + b, T::zero())
}

/// Dot product: `sum(a[i] * b[i])`.
pub fn dot<
    T: Copy + ElementOpApply + Zero + Mul<Output = T> + Add<Output = T> + MaybeSendSync + 'static,
    OpA: ElementOp,
    OpB: ElementOp,
>(
    a: &StridedView<T, OpA>,
    b: &StridedView<T, OpB>,
) -> Result<T> {
    ensure_same_shape(a.dims(), b.dims())?;

    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let a_strides = a.strides();
    let b_strides = b.strides();
    let a_dims = a.dims();

    if same_contiguous_layout(a_dims, &[a_strides, b_strides]).is_some() {
        let len = total_len(a_dims);
        let sa = unsafe { std::slice::from_raw_parts(a_ptr, len) };
        let sb = unsafe { std::slice::from_raw_parts(b_ptr, len) };
        #[cfg(feature = "simd")]
        {
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let sa = unsafe { std::slice::from_raw_parts(a_ptr as *const f32, len) };
                let sb = unsafe { std::slice::from_raw_parts(b_ptr as *const f32, len) };
                let out = simd::dot_f32(sa, sb);
                return Ok(unsafe { std::mem::transmute_copy::<f32, T>(&out) });
            }
            if TypeId::of::<T>() == TypeId::of::<f64>() {
                let sa = unsafe { std::slice::from_raw_parts(a_ptr as *const f64, len) };
                let sb = unsafe { std::slice::from_raw_parts(b_ptr as *const f64, len) };
                let out = simd::dot_f64(sa, sb);
                return Ok(unsafe { std::mem::transmute_copy::<f64, T>(&out) });
            }
        }
        let mut acc = T::zero();
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                acc = acc + OpA::apply(sa[i]) * OpB::apply(sb[i]);
            }
        });
        return Ok(acc);
    }

    let strides_list: [&[isize]; 2] = [a_strides, b_strides];

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(a_dims, &strides_list, None, std::mem::size_of::<T>());

    let mut acc = T::zero();
    let initial_offsets = vec![0isize; ordered_strides.len()];
    for_each_inner_block_preordered(
        &fused_dims,
        &plan.block,
        &ordered_strides,
        &initial_offsets,
        |offsets, len, strides| {
            acc = unsafe {
                inner_loop_dot::<T, OpA, OpB>(
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
pub fn copy_scale<T: Copy + ElementOpApply + Mul<Output = T> + MaybeSendSync, Op: ElementOp>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T, Op>,
    scale: T,
) -> Result<()> {
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

/// Copy with transpose and scaling: `dest[j,i] = scale * src[i,j]`.
pub fn copy_transpose_scale_into<T>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T>,
    scale: T,
) -> Result<()>
where
    T: Copy + ElementOpApply + Mul<Output = T> + MaybeSendSync,
{
    if src.ndim() != 2 || dest.ndim() != 2 {
        return Err(StridedError::RankMismatch(src.ndim(), 2));
    }
    let src_t = src.transpose_2d()?;
    map_into(dest, &src_t, |x| scale * x)
}
