//! High-level operations on dynamic-rank strided views.

use crate::kernel::{
    build_plan_fused, build_plan_fused_small, ensure_same_shape, for_each_inner_block_preordered,
    same_contiguous_layout, sequential_contiguous_layout, total_len, SMALL_TENSOR_THRESHOLD,
};
use crate::map_view::{copy_map_into, map_into, zip_map2_into};
use crate::reduce_view::reduce;
use crate::simd;
use crate::view::{StridedView, StridedViewMut};
use crate::{Exec, Result, StridedError};
use num_traits::{One, Zero};
use std::mem::MaybeUninit;
use std::ops::{Add, Mul};
use strided_view::{ElementOp, ElementOpApply};

use crate::cost::KernelClass;
use crate::fuse::compute_costs;
use crate::threading::{
    for_each_inner_block_with_offsets, is_parallel, mapreduce_threaded, SendPtr,
};
// ============================================================================
// Stride-specialized inner loop helpers for ops_view
//
// When all inner strides are 1 (contiguous in the innermost dimension),
// we use slice-based iteration so LLVM can auto-vectorize effectively.
// This mirrors the inner_loop_map* helpers in map_view.rs.
// ============================================================================

// INVARIANT: validated layouts keep all dereferences in bounds. Wrapping
// advances permit an unused final cursor outside a reversed/gapped allocation.
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
            dp = dp.wrapping_offset(ds);
            sp = sp.wrapping_offset(ss);
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
            dp = dp.wrapping_offset(ds);
            sp = sp.wrapping_offset(ss);
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
            dp = dp.wrapping_offset(ds);
            sp = sp.wrapping_offset(ss);
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
            dp = dp.wrapping_offset(ds);
            ap = ap.wrapping_offset(a_s);
            bp = bp.wrapping_offset(b_s);
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
            ap = ap.wrapping_offset(a_s);
            bp = bp.wrapping_offset(b_s);
        }
    }
    acc
}

/// Copy elements from source to destination: `dest[i] = src[i]`.
pub fn copy_into<T: Copy + Send + Sync, Op: ElementOp<T>>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T, Op>,
    exec: Exec<'_>,
) -> Result<()> {
    ensure_same_shape(dest.dims(), src.dims())?;

    let dst_ptr = dest.as_mut_ptr();
    let src_ptr = src.ptr();
    let dst_dims = dest.dims();
    let dst_strides = dest.strides();
    let src_strides = src.strides();

    if sequential_contiguous_layout(
        KernelClass::Copy,
        dst_dims,
        &[dst_strides, src_strides],
        &exec,
    )?
    .is_some()
    {
        let len = total_len(dst_dims)?;
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

    copy_map_into(dest, src, |x| x, exec)
}

/// Copy initialized values into a potentially uninitialized destination.
///
/// On success every logical destination element is initialized; holes are
/// untouched. Accepts identity views; use `map_into` for element operations
/// such as conjugation. Equal contiguous f32/f64 layouts and every other type
/// keep the element map; a non-contiguous f32/f64 copy uses the permutation
/// engine, split within the budget of `exec`.
///
/// # Errors
/// Returns a shape/rank error for mismatched views, `NonInjectiveOutputLayout`
/// for overlapping destination elements, or `OffsetOverflow` for size overflow.
///
/// # Examples
/// ```
/// use std::mem::MaybeUninit;
/// use strided_basic::{copy_into_uninit, Exec, StridedView, StridedViewMut};
/// let src = [1.0_f64, 2.0];
/// let mut dst = [MaybeUninit::uninit(); 2];
/// copy_into_uninit(
///     &mut StridedViewMut::new(&mut dst, &[2], &[1], 0).unwrap(),
///     &StridedView::new(&src, &[2], &[1], 0).unwrap(),
///     Exec::serial(),
/// ).unwrap();
/// // SAFETY: the successful copy initialized both destination elements.
/// assert_eq!(unsafe { dst[1].assume_init() }, 2.0);
/// ```
pub fn copy_into_uninit<T: Copy + Send + Sync + 'static>(
    dest: &mut StridedViewMut<MaybeUninit<T>>,
    src: &StridedView<T>,
    exec: Exec<'_>,
) -> Result<()> {
    ensure_same_shape(dest.dims(), src.dims())?;
    if !crate::layout_check::is_injective_layout(dest.dims(), dest.strides()) {
        return Err(StridedError::NonInjectiveOutputLayout);
    }
    if src.is_empty() {
        return Ok(());
    }
    let _len = src
        .dims()
        .iter()
        .try_fold(1usize, |n, &dim| n.checked_mul(dim))
        .ok_or(StridedError::OffsetOverflow)?;
    // The permutation engine's 4/8-byte paths reinterpret storage as native
    // floats. Restrict that path to those exact types: arbitrary Copy types can
    // contain uninitialized padding or have weaker alignment (e.g. Complex32).
    let native_float = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        || std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();
    // Equal contiguous layouts go to the element map, whose contiguous path
    // parallelizes a large copy; the permutation engine has only a serial
    // contiguous shortcut. Non-contiguous native-float copies use the
    // permutation engine, serial or parallel.
    if !native_float
        || same_contiguous_layout(dest.dims(), &[dest.strides(), src.strides()]).is_some()
    {
        return copy_map_into(dest, src, MaybeUninit::new, exec);
    }
    // INVARIANT: the permutation engine selects its float paths by the
    // source element type and only stores into the destination, so an
    // uninitialized destination with an initialized source is valid.
    crate::threading::copy_permuted_uninit(dest, src, exec)
}

/// Copy elements from `src` to `dst`, optimized for col-major destination.
///
/// Delegates to the centralized permutation-copy dispatch for the actual work.
pub fn copy_into_col_major<T: Copy + Send + Sync + 'static>(
    dst: &mut StridedViewMut<T>,
    src: &StridedView<T>,
) -> Result<()> {
    strided_perm::copy_into_col_major(dst, src, Exec::serial())
}

/// Element-wise addition: `dest[i] += src[i]`.
///
/// Source may have a different element type from destination.
pub fn add<D: Copy + Add<S, Output = D> + Send + Sync, S: Copy + Send + Sync, Op: ElementOp<S>>(
    dest: &mut StridedViewMut<D>,
    src: &StridedView<S, Op>,
    exec: Exec<'_>,
) -> Result<()> {
    ensure_same_shape(dest.dims(), src.dims())?;

    let dst_ptr = dest.as_mut_ptr();
    let src_ptr = src.ptr();
    let dst_dims = dest.dims();
    let dst_strides = dest.strides();
    let src_strides = src.strides();

    if sequential_contiguous_layout(
        KernelClass::Map,
        dst_dims,
        &[dst_strides, src_strides],
        &exec,
    )?
    .is_some()
    {
        let len = total_len(dst_dims)?;
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

    let total = total_len(dst_dims)?;
    let (fused_dims, ordered_strides, plan) = if total <= SMALL_TENSOR_THRESHOLD {
        build_plan_fused_small(dst_dims, &strides_list)
    } else {
        build_plan_fused(dst_dims, &strides_list, Some(0), elem_size)
    };

    let total = total_len(&fused_dims)?;
    if is_parallel(KernelClass::Map, total, &exec) {
        let dst_send = SendPtr(dst_ptr);
        let src_send = SendPtr(src_ptr as *mut S);

        let costs = compute_costs(&ordered_strides);
        let initial_offsets: crate::inline::OperandVec<isize> =
            strides_list.iter().map(|_| 0isize).collect();
        return mapreduce_threaded(
            &fused_dims,
            &plan.block,
            &ordered_strides,
            &initial_offsets,
            &costs,
            KernelClass::Map,
            exec,
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

    let initial_offsets: crate::inline::OperandVec<isize> =
        ordered_strides.iter().map(|_| 0isize).collect();
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
pub fn mul<D: Copy + Mul<S, Output = D> + Send + Sync, S: Copy + Send + Sync, Op: ElementOp<S>>(
    dest: &mut StridedViewMut<D>,
    src: &StridedView<S, Op>,
    exec: Exec<'_>,
) -> Result<()> {
    ensure_same_shape(dest.dims(), src.dims())?;

    let dst_ptr = dest.as_mut_ptr();
    let src_ptr = src.ptr();
    let dst_dims = dest.dims();
    let dst_strides = dest.strides();
    let src_strides = src.strides();

    if sequential_contiguous_layout(
        KernelClass::Map,
        dst_dims,
        &[dst_strides, src_strides],
        &exec,
    )?
    .is_some()
    {
        let len = total_len(dst_dims)?;
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

    let total = total_len(dst_dims)?;
    let (fused_dims, ordered_strides, plan) = if total <= SMALL_TENSOR_THRESHOLD {
        build_plan_fused_small(dst_dims, &strides_list)
    } else {
        build_plan_fused(dst_dims, &strides_list, Some(0), elem_size)
    };

    let total = total_len(&fused_dims)?;
    if is_parallel(KernelClass::Map, total, &exec) {
        let dst_send = SendPtr(dst_ptr);
        let src_send = SendPtr(src_ptr as *mut S);

        let costs = compute_costs(&ordered_strides);
        let initial_offsets: crate::inline::OperandVec<isize> =
            strides_list.iter().map(|_| 0isize).collect();
        return mapreduce_threaded(
            &fused_dims,
            &plan.block,
            &ordered_strides,
            &initial_offsets,
            &costs,
            KernelClass::Map,
            exec,
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

    let initial_offsets: crate::inline::OperandVec<isize> =
        ordered_strides.iter().map(|_| 0isize).collect();
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
    exec: Exec<'_>,
) -> Result<()>
where
    A: Copy + Mul<S, Output = D> + Sync,
    D: Copy + Add<D, Output = D> + Send + Sync,
    S: Copy + Send + Sync,
    Op: ElementOp<S>,
{
    ensure_same_shape(dest.dims(), src.dims())?;

    let dst_ptr = dest.as_mut_ptr();
    let src_ptr = src.ptr();
    let dst_dims = dest.dims();
    let dst_strides = dest.strides();
    let src_strides = src.strides();

    if sequential_contiguous_layout(
        KernelClass::Map,
        dst_dims,
        &[dst_strides, src_strides],
        &exec,
    )?
    .is_some()
    {
        let len = total_len(dst_dims)?;
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

    let total = total_len(dst_dims)?;
    let (fused_dims, ordered_strides, plan) = if total <= SMALL_TENSOR_THRESHOLD {
        build_plan_fused_small(dst_dims, &strides_list)
    } else {
        build_plan_fused(dst_dims, &strides_list, Some(0), elem_size)
    };

    let total = total_len(&fused_dims)?;
    if is_parallel(KernelClass::Map, total, &exec) {
        let dst_send = SendPtr(dst_ptr);
        let src_send = SendPtr(src_ptr as *mut S);

        let costs = compute_costs(&ordered_strides);
        let initial_offsets: crate::inline::OperandVec<isize> =
            strides_list.iter().map(|_| 0isize).collect();
        return mapreduce_threaded(
            &fused_dims,
            &plan.block,
            &ordered_strides,
            &initial_offsets,
            &costs,
            KernelClass::Map,
            exec,
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

    let initial_offsets: crate::inline::OperandVec<isize> =
        ordered_strides.iter().map(|_| 0isize).collect();
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
    exec: Exec<'_>,
) -> Result<()>
where
    A: Copy + Mul<B, Output = D> + Send + Sync,
    B: Copy + Send + Sync,
    D: Copy + Add<D, Output = D> + Send + Sync,
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

    if sequential_contiguous_layout(
        KernelClass::Map,
        dst_dims,
        &[dst_strides, a_strides, b_strides],
        &exec,
    )?
    .is_some()
    {
        let len = total_len(dst_dims)?;
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

    let total = total_len(dst_dims)?;
    let (fused_dims, ordered_strides, plan) = if total <= SMALL_TENSOR_THRESHOLD {
        build_plan_fused_small(dst_dims, &strides_list)
    } else {
        build_plan_fused(dst_dims, &strides_list, Some(0), elem_size)
    };

    let total = total_len(&fused_dims)?;
    if is_parallel(KernelClass::Map, total, &exec) {
        let dst_send = SendPtr(dst_ptr);
        let a_send = SendPtr(a_ptr as *mut A);
        let b_send = SendPtr(b_ptr as *mut B);

        let costs = compute_costs(&ordered_strides);
        let initial_offsets: crate::inline::OperandVec<isize> =
            strides_list.iter().map(|_| 0isize).collect();
        return mapreduce_threaded(
            &fused_dims,
            &plan.block,
            &ordered_strides,
            &initial_offsets,
            &costs,
            KernelClass::Map,
            exec,
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

    let initial_offsets: crate::inline::OperandVec<isize> =
        ordered_strides.iter().map(|_| 0isize).collect();
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

/// Sum all elements: `sum(src)`.
///
/// # Evaluation order
///
/// The result is independent of the pool when the call runs with a serial
/// token ([`Exec::serial`]) or a budget of one. With a larger budget a
/// contiguous identity operand above the threading threshold is summed per
/// task and the partial sums are combined in a tree, so the result is
/// repeatable for a fixed budget but can differ between budgets.
///
/// Neither path is a strict left to right sum: a contiguous run is folded with
/// independent SIMD or scalar lanes (see [`reduce`] for the lane scheme of the
/// generic path). The last bits can therefore differ from a naive loop, and no
/// bitwise equality across machines or target features is guaranteed.
///
/// For a deterministic accumulation of `f32` data in `f64`, use
/// [`zip_reduce`](crate::zip_reduce) with an `f64` map and a `0.0f64` init.
///
/// # Allocation
///
/// At rank ≤ [`INLINE_RANK`](strided_view::INLINE_RANK) the view metadata and
/// the traversal plan are stored inline, so a call performs no heap allocation.
pub fn sum<
    T: Copy + Zero + Add<Output = T> + Send + Sync + simd::MaybeSimdOps,
    Op: ElementOp<T>,
>(
    src: &StridedView<T, Op>,
    exec: Exec<'_>,
) -> Result<T> {
    // A zero-element sum is the additive identity. Return before any planning
    // so an empty view with an enormous stride never reaches block sizing.
    if total_len(src.dims())? == 0 {
        return Ok(T::zero());
    }
    // SIMD fast path: contiguous Identity view with SIMD support
    if Op::IS_IDENTITY {
        if same_contiguous_layout(src.dims(), &[src.strides()]).is_some() {
            let len = total_len(src.dims())?;
            let src_slice = unsafe { std::slice::from_raw_parts(src.ptr(), len) };

            // The partial sums of the halves are combined in a tree; a type
            // without SIMD support takes the generic reduction below.
            if T::try_simd_sum(&[]).is_some() && is_parallel(KernelClass::Reduce, len, &exec) {
                return Ok(crate::threading::parallel_map_reduce(
                    0..len,
                    crate::cost::for_len_tree(exec, KernelClass::Reduce, len),
                    &|range| T::try_simd_sum(&src_slice[range]).unwrap(),
                    &|left, right| left + right,
                ));
            }

            if let Some(result) = T::try_simd_sum(src_slice) {
                return Ok(result);
            }
        }
    }
    reduce(src, |x| x, |a, b| a + b, T::zero(), exec)
}

/// Dot product: `sum(OpA::apply(a[i]) * OpB::apply(b[i]))`.
///
/// Operands may have different element types. Result type `R` must be `A * B`.
/// SIMD fast path fires only when `A == B == R` (same type) and both Identity ops.
///
/// A widening dot — `f32` operands with an `f64` accumulator — is not
/// expressible here because `R` is pinned to `A * B`; use
/// [`zip_reduce`](crate::zip_reduce) with a
/// `|x, y| x as f64 * y as f64` map instead.
///
/// # Evaluation order
///
/// This entry takes no parallel branch and no token, so its result does not
/// depend on any pool: it is computed on the calling thread. The homogeneous contiguous fast path folds with the SIMD
/// lane scheme of the element type; every other layout folds a single
/// accumulator along the traversal order of the fused plan, which is left to
/// right over each unit-stride run. Neither is guaranteed to be bitwise equal
/// across machines or target features.
///
/// # Allocation
///
/// At rank ≤ [`INLINE_RANK`](strided_view::INLINE_RANK) the view metadata and
/// the traversal plan are stored inline, so a call performs no heap allocation.
pub fn dot<A, B, R, OpA, OpB>(a: &StridedView<A, OpA>, b: &StridedView<B, OpB>) -> Result<R>
where
    A: Copy + Mul<B, Output = R> + Send + Sync + 'static,
    B: Copy + Send + Sync + 'static,
    R: Copy + Zero + Add<Output = R> + Send + Sync + simd::MaybeSimdOps + 'static,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
{
    ensure_same_shape(a.dims(), b.dims())?;

    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let a_strides = a.strides();
    let b_strides = b.strides();
    let a_dims = a.dims();
    let len = total_len(a_dims)?;

    // A zero-element dot product is the additive identity. Return before any
    // planning so an empty view with an enormous stride never reaches block
    // sizing.
    if len == 0 {
        return Ok(R::zero());
    }

    if same_contiguous_layout(a_dims, &[a_strides, b_strides]).is_some() {
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

    let total = total_len(a_dims)?;
    let (fused_dims, ordered_strides, plan) = if total <= SMALL_TENSOR_THRESHOLD {
        build_plan_fused_small(a_dims, &strides_list)
    } else {
        build_plan_fused(a_dims, &strides_list, None, elem_size)
    };

    let mut acc = R::zero();
    let initial_offsets: crate::inline::OperandVec<isize> =
        ordered_strides.iter().map(|_| 0isize).collect();
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
pub fn symmetrize_into<T>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T>,
    exec: Exec<'_>,
) -> Result<()>
where
    T: Copy
        + Add<Output = T>
        + Mul<Output = T>
        + num_traits::FromPrimitive
        + std::ops::Div<Output = T>
        + Send
        + Sync,
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

    zip_map2_into(dest, src, &src_t, |a, b| (a + b) * half, exec)
}

/// Conjugate-symmetrize a square matrix: `dest = (src + conj(src^T)) / 2`.
pub fn symmetrize_conj_into<T>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T>,
    exec: Exec<'_>,
) -> Result<()>
where
    T: Copy
        + ElementOpApply
        + Add<Output = T>
        + Mul<Output = T>
        + num_traits::FromPrimitive
        + std::ops::Div<Output = T>
        + Send
        + Sync,
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

    zip_map2_into(dest, src, &src_adj, |a, b| (a + b) * half, exec)
}

/// Copy with scaling: `dest[i] = scale * src[i]`.
///
/// Scale, source, and destination may have different element types.
pub fn copy_scale<D, S, A, Op>(
    dest: &mut StridedViewMut<D>,
    src: &StridedView<S, Op>,
    scale: A,
    exec: Exec<'_>,
) -> Result<()>
where
    A: Copy + Mul<S, Output = D> + Sync,
    D: Copy + Send + Sync,
    S: Copy + Send + Sync,
    Op: ElementOp<S>,
{
    map_into(dest, src, |x| scale * x, exec)
}

/// Copy with complex conjugation: `dest[i] = conj(src[i])`.
pub fn copy_conj<T: Copy + ElementOpApply + Send + Sync>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T>,
    exec: Exec<'_>,
) -> Result<()> {
    let src_conj = src.conj();
    copy_into(dest, &src_conj, exec)
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
unsafe fn fill_2d<T: Copy + Send + Sync>(
    dst: *mut T,
    dim0: usize,
    dim1: usize,
    dst_stride0: isize,
    dst_stride1: isize,
    value: T,
    exec: Exec<'_>,
) {
    let total = dim0.saturating_mul(dim1);
    let exec = crate::cost::for_len(exec, KernelClass::Copy, total);
    if !exec.is_serial() {
        let dst_send = SendPtr(dst);
        if dst_stride0.unsigned_abs() <= dst_stride1.unsigned_abs() {
            crate::threading::parallel_for_each(0..dim1, exec, &|columns| {
                for j in columns {
                    let dst = dst_send.as_ptr();
                    unsafe {
                        let base = j as isize * dst_stride1;
                        for i in 0..dim0 {
                            *dst.offset(base + i as isize * dst_stride0) = value;
                        }
                    }
                }
            });
        } else {
            crate::threading::parallel_for_each(0..dim0, exec, &|rows| {
                for i in rows {
                    let dst = dst_send.as_ptr();
                    unsafe {
                        let base = i as isize * dst_stride0;
                        for j in 0..dim1 {
                            *dst.offset(base + j as isize * dst_stride1) = value;
                        }
                    }
                }
            });
        }
        return;
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
    T: Copy + Zero + PartialEq + Send + Sync + 'static,
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
#[allow(clippy::too_many_arguments)]
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
    exec: Exec<'_>,
) {
    const TILE: usize = 4;
    let row_full = src_rows / TILE * TILE;
    let col_full = src_cols / TILE * TILE;

    let total = src_rows.saturating_mul(src_cols);
    let exec = crate::cost::for_len(exec, KernelClass::Map, total);
    if !exec.is_serial() {
        let dst_send = SendPtr(dst);
        let src_send = SendPtr(src as *mut f64);
        let row_tiles = row_full / TILE;
        crate::threading::parallel_for_each(0..row_tiles, exec, &|tiles| {
            for tile_i in tiles {
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
    exec: Exec<'_>,
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
        exec,
    );
    true
}

#[inline]
unsafe fn try_copy_transpose_scale_2d_f64_tiled_typed<T>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T>,
    scale: T,
    exec: Exec<'_>,
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
        exec,
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
#[allow(clippy::too_many_arguments)]
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
    exec: Exec<'_>,
) where
    T: Copy + Mul<Output = T> + Send + Sync,
{
    const TILE: usize = 4;
    let row_full = src_rows / TILE * TILE;
    let col_full = src_cols / TILE * TILE;

    let total = src_rows.saturating_mul(src_cols);
    let exec = crate::cost::for_len(exec, KernelClass::Map, total);
    if !exec.is_serial() {
        let dst_send = SendPtr(dst);
        let src_send = SendPtr(src as *mut T);
        let row_tiles = row_full / TILE;
        crate::threading::parallel_for_each(0..row_tiles, exec, &|tiles| {
            for tile_i in tiles {
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
    exec: Exec<'_>,
) -> bool
where
    T: Copy + Mul<Output = T> + Send + Sync,
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
        exec,
    );
    true
}

#[inline]
#[allow(clippy::too_many_arguments)]
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
    exec: Exec<'_>,
) where
    T: Copy + ElementOpApply + Mul<Output = T> + Send + Sync,
{
    let total = src_rows.saturating_mul(src_cols);
    let exec = crate::cost::for_len(exec, KernelClass::Map, total);
    if !exec.is_serial() {
        let dst_send = SendPtr(dst);
        let src_send = SendPtr(src as *mut T);
        if dst_stride0.unsigned_abs() <= dst_stride1.unsigned_abs() {
            crate::threading::parallel_for_each(0..src_rows, exec, &|rows| {
                for i in rows {
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
                }
            });
        } else {
            crate::threading::parallel_for_each(0..src_cols, exec, &|columns| {
                for j in columns {
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
                }
            });
        }
        return;
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
    exec: Exec<'_>,
) -> Result<()>
where
    T: Copy + ElementOpApply + Mul<Output = T> + Zero + One + PartialEq + Send + Sync + 'static,
{
    let mut exec = exec;
    if src.ndim() != 2 || dest.ndim() != 2 {
        return Err(StridedError::RankMismatch(src.ndim(), 2));
    }

    let src_dims = src.dims();
    let expected_dims = [src_dims[1], src_dims[0]];
    ensure_same_shape(dest.dims(), &expected_dims)?;
    // Reject an element count beyond usize (a huge stride-0 broadcast) up
    // front instead of looping over it.
    total_len(src_dims)?;

    if scale == T::zero() {
        unsafe {
            if same_contiguous_layout(dest.dims(), &[dest.strides()]).is_some() {
                fill_contiguous(dest.as_mut_ptr(), total_len(dest.dims())?, T::zero());
            } else {
                fill_2d(
                    dest.as_mut_ptr(),
                    dest.dims()[0],
                    dest.dims()[1],
                    dest.strides()[0],
                    dest.strides()[1],
                    T::zero(),
                    exec,
                );
            }
        }
        return Ok(());
    }

    let transpose_is_identity = element_transpose_is_identity::<T>();

    unsafe {
        if transpose_is_identity
            && try_copy_transpose_scale_2d_f64_tiled_typed(dest, src, scale, exec.reborrow())
        {
            return Ok(());
        }
        if transpose_is_identity
            && try_copy_transpose_scale_2d_identity_tiled(dest, src, scale, exec.reborrow())
        {
            return Ok(());
        }
    }

    if scale == T::one() && transpose_is_identity {
        let src_t = src.permute(&[1, 0])?;
        return crate::threading::copy_permuted(dest, &src_t, exec);
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
            exec,
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "ops_view/tests/tiled_tests.rs"]
mod tiled_tests;
