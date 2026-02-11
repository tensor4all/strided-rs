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
use strided_view::ElementOp;

#[cfg(feature = "parallel")]
use crate::fuse::compute_costs;
#[cfg(feature = "parallel")]
use crate::threading::{for_each_inner_block_with_offsets, mapreduce_threaded, MINTHREADLENGTH};

// ============================================================================
// Stride-specialized inner loop helpers
//
// When all inner strides are 1 (contiguous in the innermost dimension),
// we use slice-based iteration so LLVM can auto-vectorize effectively.
// This is the Rust equivalent of Julia's @simd on the innermost loop.
// ============================================================================

/// Unary inner loop: `dest[i] = f(Op::apply(src[i]))` for `len` elements.
#[inline(always)]
unsafe fn inner_loop_map1<D: Copy, A: Copy, Op: ElementOp<A>>(
    dp: *mut D,
    ds: isize,
    sp: *const A,
    ss: isize,
    len: usize,
    f: &impl Fn(A) -> D,
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
unsafe fn inner_loop_map2<D: Copy, A: Copy, B: Copy, OpA: ElementOp<A>, OpB: ElementOp<B>>(
    dp: *mut D,
    ds: isize,
    ap: *const A,
    a_s: isize,
    bp: *const B,
    b_s: isize,
    len: usize,
    f: &impl Fn(A, B) -> D,
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
    D: Copy,
    A: Copy,
    B: Copy,
    C: Copy,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
    OpC: ElementOp<C>,
>(
    dp: *mut D,
    ds: isize,
    ap: *const A,
    a_s: isize,
    bp: *const B,
    b_s: isize,
    cp: *const C,
    c_s: isize,
    len: usize,
    f: &impl Fn(A, B, C) -> D,
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
    D: Copy,
    A: Copy,
    B: Copy,
    C: Copy,
    E: Copy,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
    OpC: ElementOp<C>,
    OpE: ElementOp<E>,
>(
    dp: *mut D,
    ds: isize,
    ap: *const A,
    a_s: isize,
    bp: *const B,
    b_s: isize,
    cp: *const C,
    c_s: isize,
    ep: *const E,
    e_s: isize,
    len: usize,
    f: &impl Fn(A, B, C, E) -> D,
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
/// Source and destination may have different element types.
pub fn map_into<D: Copy + MaybeSendSync, A: Copy + MaybeSendSync, Op: ElementOp<A>>(
    dest: &mut StridedViewMut<D>,
    src: &StridedView<A, Op>,
    f: impl Fn(A) -> D + MaybeSync,
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
    let elem_size = std::mem::size_of::<D>().max(std::mem::size_of::<A>());

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(dst_dims, &strides_list, Some(0), elem_size);

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        if total > MINTHREADLENGTH {
            use crate::threading::SendPtr;
            let dst_send = SendPtr(dst_ptr);
            let src_send = SendPtr(src_ptr as *mut A);

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
                            let dp = unsafe { dst_send.as_ptr().offset(offsets[0]) };
                            let sp = unsafe { src_send.as_const().offset(offsets[1]) };
                            unsafe {
                                inner_loop_map1::<D, A, Op>(dp, strides[0], sp, strides[1], len, &f)
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
            unsafe { inner_loop_map1::<D, A, Op>(dp, strides[0], sp, strides[1], len, &f) };
            Ok(())
        },
    )
}

/// Binary element-wise operation: `dest[i] = f(a[i], b[i])`.
///
/// Source operands `a` and `b` may have different element types from each other
/// and from `dest`. The closure `f` handles per-element type conversion.
pub fn zip_map2_into<
    D: Copy + MaybeSendSync,
    A: Copy + MaybeSendSync,
    B: Copy + MaybeSendSync,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
>(
    dest: &mut StridedViewMut<D>,
    a: &StridedView<A, OpA>,
    b: &StridedView<B, OpB>,
    f: impl Fn(A, B) -> D + MaybeSync,
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
    let elem_size = std::mem::size_of::<D>()
        .max(std::mem::size_of::<A>())
        .max(std::mem::size_of::<B>());

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(dst_dims, &strides_list, Some(0), elem_size);

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        if total > MINTHREADLENGTH {
            use crate::threading::SendPtr;
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
                            let dp = unsafe { dst_send.as_ptr().offset(offsets[0]) };
                            let ap = unsafe { a_send.as_const().offset(offsets[1]) };
                            let bp = unsafe { b_send.as_const().offset(offsets[2]) };
                            unsafe {
                                inner_loop_map2::<D, A, B, OpA, OpB>(
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
                inner_loop_map2::<D, A, B, OpA, OpB>(
                    dp, strides[0], ap, strides[1], bp, strides[2], len, &f,
                )
            };
            Ok(())
        },
    )
}

/// Ternary element-wise operation: `dest[i] = f(a[i], b[i], c[i])`.
pub fn zip_map3_into<
    D: Copy + MaybeSendSync,
    A: Copy + MaybeSendSync,
    B: Copy + MaybeSendSync,
    C: Copy + MaybeSendSync,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
    OpC: ElementOp<C>,
>(
    dest: &mut StridedViewMut<D>,
    a: &StridedView<A, OpA>,
    b: &StridedView<B, OpB>,
    c: &StridedView<C, OpC>,
    f: impl Fn(A, B, C) -> D + MaybeSync,
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
    let elem_size = std::mem::size_of::<D>()
        .max(std::mem::size_of::<A>())
        .max(std::mem::size_of::<B>())
        .max(std::mem::size_of::<C>());

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(dst_dims, &strides_list, Some(0), elem_size);

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        if total > MINTHREADLENGTH {
            use crate::threading::SendPtr;
            let dst_send = SendPtr(dst_ptr);
            let a_send = SendPtr(a_ptr as *mut A);
            let b_send = SendPtr(b_ptr as *mut B);
            let c_send = SendPtr(c_ptr as *mut C);

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
                            let dp = unsafe { dst_send.as_ptr().offset(offsets[0]) };
                            let ap = unsafe { a_send.as_const().offset(offsets[1]) };
                            let bp = unsafe { b_send.as_const().offset(offsets[2]) };
                            let cp = unsafe { c_send.as_const().offset(offsets[3]) };
                            unsafe {
                                inner_loop_map3::<D, A, B, C, OpA, OpB, OpC>(
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
                inner_loop_map3::<D, A, B, C, OpA, OpB, OpC>(
                    dp, strides[0], ap, strides[1], bp, strides[2], cp, strides[3], len, &f,
                )
            };
            Ok(())
        },
    )
}

/// Quaternary element-wise operation: `dest[i] = f(a[i], b[i], c[i], e[i])`.
pub fn zip_map4_into<
    D: Copy + MaybeSendSync,
    A: Copy + MaybeSendSync,
    B: Copy + MaybeSendSync,
    C: Copy + MaybeSendSync,
    E: Copy + MaybeSendSync,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
    OpC: ElementOp<C>,
    OpE: ElementOp<E>,
>(
    dest: &mut StridedViewMut<D>,
    a: &StridedView<A, OpA>,
    b: &StridedView<B, OpB>,
    c: &StridedView<C, OpC>,
    e: &StridedView<E, OpE>,
    f: impl Fn(A, B, C, E) -> D + MaybeSync,
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
    let elem_size = std::mem::size_of::<D>()
        .max(std::mem::size_of::<A>())
        .max(std::mem::size_of::<B>())
        .max(std::mem::size_of::<C>())
        .max(std::mem::size_of::<E>());

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(dst_dims, &strides_list, Some(0), elem_size);

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        if total > MINTHREADLENGTH {
            use crate::threading::SendPtr;
            let dst_send = SendPtr(dst_ptr);
            let a_send = SendPtr(a_ptr as *mut A);
            let b_send = SendPtr(b_ptr as *mut B);
            let c_send = SendPtr(c_ptr as *mut C);
            let e_send = SendPtr(e_ptr as *mut E);

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
                            let dp = unsafe { dst_send.as_ptr().offset(offsets[0]) };
                            let ap = unsafe { a_send.as_const().offset(offsets[1]) };
                            let bp = unsafe { b_send.as_const().offset(offsets[2]) };
                            let cp = unsafe { c_send.as_const().offset(offsets[3]) };
                            let ep = unsafe { e_send.as_const().offset(offsets[4]) };
                            unsafe {
                                inner_loop_map4::<D, A, B, C, E, OpA, OpB, OpC, OpE>(
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
                inner_loop_map4::<D, A, B, C, E, OpA, OpB, OpC, OpE>(
                    dp, strides[0], ap, strides[1], bp, strides[2], cp, strides[3], ep, strides[4],
                    len, &f,
                )
            };
            Ok(())
        },
    )
}
