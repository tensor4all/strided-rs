//! Map operations on dynamic-rank strided views.
//!
//! These are the canonical view-based map functions, equivalent to Julia's `Base.map!`.

use crate::kernel::{
    build_plan_fused, ensure_same_shape, for_each_inner_block_preordered,
    sequential_contiguous_layout, total_len,
};
use crate::maybe_sync::{MaybeSendSync, MaybeSync};
use crate::simd;
use crate::view::{StridedView, StridedViewMut};
use crate::Result;
use strided_view::{ElementOp, ElementOpApply};

#[cfg(feature = "parallel")]
use crate::threading::{
    compute_costs, for_each_inner_block_with_offsets, mapreduce_threaded, MINTHREADLENGTH,
};

// ============================================================================
// Stride-specialized inner loop helpers
//
// When all inner strides are 1 (contiguous in the innermost dimension),
// we use slice-based iteration so LLVM can auto-vectorize effectively.
// This is the Rust equivalent of Julia's @simd on the innermost loop.
// ============================================================================

/// Unary inner loop: `dest[i] = f(Op::apply(src[i]))` for `len` elements.
#[inline(always)]
unsafe fn inner_loop_map1<T: Copy + ElementOpApply, Op: ElementOp>(
    dp: *mut T,
    ds: isize,
    sp: *const T,
    ss: isize,
    len: usize,
    f: &impl Fn(T) -> T,
) {
    if ds == 1 && ss == 1 {
        let src = std::slice::from_raw_parts(sp, len);
        let dst = std::slice::from_raw_parts_mut(dp, len);
        simd::dispatch_if_large(len, || {
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d = f(Op::apply(*s));
            }
        });
    } else {
        let mut dp = dp;
        let mut sp = sp;
        for _ in 0..len {
            *dp = f(Op::apply(*sp));
            dp = dp.offset(ds);
            sp = sp.offset(ss);
        }
    }
}

/// Binary inner loop: `dest[i] = f(OpA::apply(a[i]), OpB::apply(b[i]))`.
#[inline(always)]
unsafe fn inner_loop_map2<T: Copy + ElementOpApply, OpA: ElementOp, OpB: ElementOp>(
    dp: *mut T,
    ds: isize,
    ap: *const T,
    a_s: isize,
    bp: *const T,
    b_s: isize,
    len: usize,
    f: &impl Fn(T, T) -> T,
) {
    if ds == 1 && a_s == 1 && b_s == 1 {
        let src_a = std::slice::from_raw_parts(ap, len);
        let src_b = std::slice::from_raw_parts(bp, len);
        let dst = std::slice::from_raw_parts_mut(dp, len);
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                dst[i] = f(OpA::apply(src_a[i]), OpB::apply(src_b[i]));
            }
        });
    } else {
        let mut dp = dp;
        let mut ap = ap;
        let mut bp = bp;
        for _ in 0..len {
            *dp = f(OpA::apply(*ap), OpB::apply(*bp));
            dp = dp.offset(ds);
            ap = ap.offset(a_s);
            bp = bp.offset(b_s);
        }
    }
}

/// Ternary inner loop: `dest[i] = f(a[i], b[i], c[i])`.
#[inline(always)]
unsafe fn inner_loop_map3<
    T: Copy + ElementOpApply,
    OpA: ElementOp,
    OpB: ElementOp,
    OpC: ElementOp,
>(
    dp: *mut T,
    ds: isize,
    ap: *const T,
    a_s: isize,
    bp: *const T,
    b_s: isize,
    cp: *const T,
    c_s: isize,
    len: usize,
    f: &impl Fn(T, T, T) -> T,
) {
    if ds == 1 && a_s == 1 && b_s == 1 && c_s == 1 {
        let src_a = std::slice::from_raw_parts(ap, len);
        let src_b = std::slice::from_raw_parts(bp, len);
        let src_c = std::slice::from_raw_parts(cp, len);
        let dst = std::slice::from_raw_parts_mut(dp, len);
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                dst[i] = f(
                    OpA::apply(src_a[i]),
                    OpB::apply(src_b[i]),
                    OpC::apply(src_c[i]),
                );
            }
        });
    } else {
        let mut dp = dp;
        let mut ap = ap;
        let mut bp = bp;
        let mut cp = cp;
        for _ in 0..len {
            *dp = f(OpA::apply(*ap), OpB::apply(*bp), OpC::apply(*cp));
            dp = dp.offset(ds);
            ap = ap.offset(a_s);
            bp = bp.offset(b_s);
            cp = cp.offset(c_s);
        }
    }
}

/// Quaternary inner loop: `dest[i] = f(a[i], b[i], c[i], e[i])`.
#[inline(always)]
unsafe fn inner_loop_map4<
    T: Copy + ElementOpApply,
    OpA: ElementOp,
    OpB: ElementOp,
    OpC: ElementOp,
    OpE: ElementOp,
>(
    dp: *mut T,
    ds: isize,
    ap: *const T,
    a_s: isize,
    bp: *const T,
    b_s: isize,
    cp: *const T,
    c_s: isize,
    ep: *const T,
    e_s: isize,
    len: usize,
    f: &impl Fn(T, T, T, T) -> T,
) {
    if ds == 1 && a_s == 1 && b_s == 1 && c_s == 1 && e_s == 1 {
        let src_a = std::slice::from_raw_parts(ap, len);
        let src_b = std::slice::from_raw_parts(bp, len);
        let src_c = std::slice::from_raw_parts(cp, len);
        let src_e = std::slice::from_raw_parts(ep, len);
        let dst = std::slice::from_raw_parts_mut(dp, len);
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                dst[i] = f(
                    OpA::apply(src_a[i]),
                    OpB::apply(src_b[i]),
                    OpC::apply(src_c[i]),
                    OpE::apply(src_e[i]),
                );
            }
        });
    } else {
        let mut dp = dp;
        let mut ap = ap;
        let mut bp = bp;
        let mut cp = cp;
        let mut ep = ep;
        for _ in 0..len {
            *dp = f(
                OpA::apply(*ap),
                OpB::apply(*bp),
                OpC::apply(*cp),
                OpE::apply(*ep),
            );
            dp = dp.offset(ds);
            ap = ap.offset(a_s);
            bp = bp.offset(b_s);
            cp = cp.offset(c_s);
            ep = ep.offset(e_s);
        }
    }
}

/// Apply a function element-wise from source to destination.
///
/// The element operation `Op` is applied lazily when reading from `src`.
pub fn map_into<T: Copy + ElementOpApply + MaybeSendSync, Op: ElementOp>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T, Op>,
    f: impl Fn(T) -> T + MaybeSync,
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
                dst[i] = f(Op::apply(src[i]));
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
            use crate::threading::SendPtr;
            let dst_send = SendPtr(dst_ptr);
            let src_send = SendPtr(src_ptr as *mut T);

            let costs = compute_costs(&ordered_strides, fused_dims.len());
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
                            let dp = unsafe { dst_send.as_ptr().offset(offsets[0]) };
                            let sp = unsafe { src_send.as_const().offset(offsets[1]) };
                            unsafe {
                                inner_loop_map1::<T, Op>(dp, strides[0], sp, strides[1], len, &f)
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
            let dp = unsafe { dst_ptr.offset(offsets[0]) };
            let sp = unsafe { src_ptr.offset(offsets[1]) };
            unsafe { inner_loop_map1::<T, Op>(dp, strides[0], sp, strides[1], len, &f) };
            Ok(())
        },
    )
}

/// Binary element-wise operation: `dest[i] = f(a[i], b[i])`.
pub fn zip_map2_into<T: Copy + ElementOpApply + MaybeSendSync, OpA: ElementOp, OpB: ElementOp>(
    dest: &mut StridedViewMut<T>,
    a: &StridedView<T, OpA>,
    b: &StridedView<T, OpB>,
    f: impl Fn(T, T) -> T + MaybeSync,
) -> Result<()> {
    ensure_same_shape(dest.dims(), a.dims())?;
    ensure_same_shape(dest.dims(), b.dims())?;

    let dst_ptr = dest.as_mut_ptr();
    let dst_dims = dest.dims();
    let dst_strides = dest.strides();
    let a_ptr = a.ptr();
    let b_ptr = b.ptr();

    let a_strides = a.strides();
    let b_strides = b.strides();

    if sequential_contiguous_layout(dst_dims, &[dst_strides, a_strides, b_strides]).is_some() {
        let len = total_len(dst_dims);
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, len) };
        let sa = unsafe { std::slice::from_raw_parts(a_ptr, len) };
        let sb = unsafe { std::slice::from_raw_parts(b_ptr, len) };
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                dst[i] = f(OpA::apply(sa[i]), OpB::apply(sb[i]));
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
            use crate::threading::SendPtr;
            let dst_send = SendPtr(dst_ptr);
            let a_send = SendPtr(a_ptr as *mut T);
            let b_send = SendPtr(b_ptr as *mut T);

            let costs = compute_costs(&ordered_strides, fused_dims.len());
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
                            let dp = unsafe { dst_send.as_ptr().offset(offsets[0]) };
                            let ap = unsafe { a_send.as_const().offset(offsets[1]) };
                            let bp = unsafe { b_send.as_const().offset(offsets[2]) };
                            unsafe {
                                inner_loop_map2::<T, OpA, OpB>(
                                    dp, strides[0], ap, strides[1], bp, strides[2], len, &f,
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
            let dp = unsafe { dst_ptr.offset(offsets[0]) };
            let ap = unsafe { a_ptr.offset(offsets[1]) };
            let bp = unsafe { b_ptr.offset(offsets[2]) };
            unsafe {
                inner_loop_map2::<T, OpA, OpB>(
                    dp, strides[0], ap, strides[1], bp, strides[2], len, &f,
                )
            };
            Ok(())
        },
    )
}

/// Ternary element-wise operation: `dest[i] = f(a[i], b[i], c[i])`.
pub fn zip_map3_into<
    T: Copy + ElementOpApply + MaybeSendSync,
    OpA: ElementOp,
    OpB: ElementOp,
    OpC: ElementOp,
>(
    dest: &mut StridedViewMut<T>,
    a: &StridedView<T, OpA>,
    b: &StridedView<T, OpB>,
    c: &StridedView<T, OpC>,
    f: impl Fn(T, T, T) -> T + MaybeSync,
) -> Result<()> {
    ensure_same_shape(dest.dims(), a.dims())?;
    ensure_same_shape(dest.dims(), b.dims())?;
    ensure_same_shape(dest.dims(), c.dims())?;

    let dst_ptr = dest.as_mut_ptr();
    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let c_ptr = c.ptr();

    let dst_dims = dest.dims();
    let dst_strides = dest.strides();

    if sequential_contiguous_layout(
        dst_dims,
        &[dst_strides, a.strides(), b.strides(), c.strides()],
    )
    .is_some()
    {
        let len = total_len(dst_dims);
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, len) };
        let sa = unsafe { std::slice::from_raw_parts(a_ptr, len) };
        let sb = unsafe { std::slice::from_raw_parts(b_ptr, len) };
        let sc = unsafe { std::slice::from_raw_parts(c_ptr, len) };
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                dst[i] = f(OpA::apply(sa[i]), OpB::apply(sb[i]), OpC::apply(sc[i]));
            }
        });
        return Ok(());
    }

    let strides_list: [&[isize]; 4] = [dst_strides, a.strides(), b.strides(), c.strides()];

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(dst_dims, &strides_list, Some(0), std::mem::size_of::<T>());

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        if total > MINTHREADLENGTH {
            use crate::threading::SendPtr;
            let dst_send = SendPtr(dst_ptr);
            let a_send = SendPtr(a_ptr as *mut T);
            let b_send = SendPtr(b_ptr as *mut T);
            let c_send = SendPtr(c_ptr as *mut T);

            let costs = compute_costs(&ordered_strides, fused_dims.len());
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
                            let dp = unsafe { dst_send.as_ptr().offset(offsets[0]) };
                            let ap = unsafe { a_send.as_const().offset(offsets[1]) };
                            let bp = unsafe { b_send.as_const().offset(offsets[2]) };
                            let cp = unsafe { c_send.as_const().offset(offsets[3]) };
                            unsafe {
                                inner_loop_map3::<T, OpA, OpB, OpC>(
                                    dp, strides[0], ap, strides[1], bp, strides[2], cp, strides[3],
                                    len, &f,
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
            let dp = unsafe { dst_ptr.offset(offsets[0]) };
            let ap = unsafe { a_ptr.offset(offsets[1]) };
            let bp = unsafe { b_ptr.offset(offsets[2]) };
            let cp = unsafe { c_ptr.offset(offsets[3]) };
            unsafe {
                inner_loop_map3::<T, OpA, OpB, OpC>(
                    dp, strides[0], ap, strides[1], bp, strides[2], cp, strides[3], len, &f,
                )
            };
            Ok(())
        },
    )
}

/// Quaternary element-wise operation: `dest[i] = f(a[i], b[i], c[i], e[i])`.
pub fn zip_map4_into<
    T: Copy + ElementOpApply + MaybeSendSync,
    OpA: ElementOp,
    OpB: ElementOp,
    OpC: ElementOp,
    OpE: ElementOp,
>(
    dest: &mut StridedViewMut<T>,
    a: &StridedView<T, OpA>,
    b: &StridedView<T, OpB>,
    c: &StridedView<T, OpC>,
    e: &StridedView<T, OpE>,
    f: impl Fn(T, T, T, T) -> T + MaybeSync,
) -> Result<()> {
    ensure_same_shape(dest.dims(), a.dims())?;
    ensure_same_shape(dest.dims(), b.dims())?;
    ensure_same_shape(dest.dims(), c.dims())?;
    ensure_same_shape(dest.dims(), e.dims())?;

    let dst_ptr = dest.as_mut_ptr();
    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let c_ptr = c.ptr();
    let e_ptr = e.ptr();

    let dst_dims = dest.dims();
    let dst_strides = dest.strides();

    if sequential_contiguous_layout(
        dst_dims,
        &[
            dst_strides,
            a.strides(),
            b.strides(),
            c.strides(),
            e.strides(),
        ],
    )
    .is_some()
    {
        let len = total_len(dst_dims);
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, len) };
        let sa = unsafe { std::slice::from_raw_parts(a_ptr, len) };
        let sb = unsafe { std::slice::from_raw_parts(b_ptr, len) };
        let sc = unsafe { std::slice::from_raw_parts(c_ptr, len) };
        let se = unsafe { std::slice::from_raw_parts(e_ptr, len) };
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                dst[i] = f(
                    OpA::apply(sa[i]),
                    OpB::apply(sb[i]),
                    OpC::apply(sc[i]),
                    OpE::apply(se[i]),
                );
            }
        });
        return Ok(());
    }

    let strides_list: [&[isize]; 5] = [
        dst_strides,
        a.strides(),
        b.strides(),
        c.strides(),
        e.strides(),
    ];

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(dst_dims, &strides_list, Some(0), std::mem::size_of::<T>());

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        if total > MINTHREADLENGTH {
            use crate::threading::SendPtr;
            let dst_send = SendPtr(dst_ptr);
            let a_send = SendPtr(a_ptr as *mut T);
            let b_send = SendPtr(b_ptr as *mut T);
            let c_send = SendPtr(c_ptr as *mut T);
            let e_send = SendPtr(e_ptr as *mut T);

            let costs = compute_costs(&ordered_strides, fused_dims.len());
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
                            let dp = unsafe { dst_send.as_ptr().offset(offsets[0]) };
                            let ap = unsafe { a_send.as_const().offset(offsets[1]) };
                            let bp = unsafe { b_send.as_const().offset(offsets[2]) };
                            let cp = unsafe { c_send.as_const().offset(offsets[3]) };
                            let ep = unsafe { e_send.as_const().offset(offsets[4]) };
                            unsafe {
                                inner_loop_map4::<T, OpA, OpB, OpC, OpE>(
                                    dp, strides[0], ap, strides[1], bp, strides[2], cp, strides[3],
                                    ep, strides[4], len, &f,
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
            let dp = unsafe { dst_ptr.offset(offsets[0]) };
            let ap = unsafe { a_ptr.offset(offsets[1]) };
            let bp = unsafe { b_ptr.offset(offsets[2]) };
            let cp = unsafe { c_ptr.offset(offsets[3]) };
            let ep = unsafe { e_ptr.offset(offsets[4]) };
            unsafe {
                inner_loop_map4::<T, OpA, OpB, OpC, OpE>(
                    dp, strides[0], ap, strides[1], bp, strides[2], cp, strides[3], ep, strides[4],
                    len, &f,
                )
            };
            Ok(())
        },
    )
}
