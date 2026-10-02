//! In-place update operations: the destination is also an input.
//!
//! `map_into` and `zip_map*_into` write `dest[i] = f(src...)` and never read
//! the old destination. These functions compute
//!
//! ```text
//! dest[i] = f(OpD(dest[i]), OpA(a[i]), OpB(b[i]))
//! ```
//!
//! reading the previous `dest[i]` through the destination view itself, so no
//! second view over the destination's storage is ever formed (two live views
//! of one allocation, one mutable, would be aliasing). Every element of an
//! injective destination is read and written exactly once, by one call of `f`.
//!
//! The element operation of each input, and of the destination read, is a type
//! parameter (`Identity`, `Conj`, ...), so a conjugating update is a
//! monomorphized loop with no per-element flag. Traversal, blocking, contiguous
//! inner loops and threading are those of the other map kernels (threading
//! follows the active [`ExecContext`](crate::ExecContext)).

use crate::kernel::{
    build_plan_fused, build_plan_fused_small, ensure_same_shape, for_each_inner_block_preordered,
    total_len, SMALL_TENSOR_THRESHOLD,
};
use crate::layout_check::is_injective_layout;
use crate::maybe_sync::{MaybeSendSync, MaybeSync};
use crate::simd;
use crate::view::{StridedView, StridedViewMut};
use crate::{Result, StridedError};
use strided_view::ElementOp;

#[cfg(feature = "parallel")]
use crate::fuse::compute_costs;
#[cfg(feature = "parallel")]
use crate::threading::{for_each_inner_block_with_offsets, mapreduce_threaded, MINTHREADLENGTH};

/// A raw pointer that may cross threads: dereferenced only at the validated,
/// disjoint (for the destination) offsets the traversal generates.
struct Raw<T>(*mut T);
impl<T> Clone for Raw<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Raw<T> {}
// SAFETY: see the type documentation; every user upholds it.
unsafe impl<T> Send for Raw<T> {}
unsafe impl<T> Sync for Raw<T> {}

fn validate_destination(dims: &[usize], strides: &[isize]) -> Result<()> {
    if is_injective_layout(dims, strides) {
        Ok(())
    } else {
        Err(StridedError::NonInjectiveOutputLayout)
    }
}

/// Walk `dims` with the strided traversal and call `body(offsets, len,
/// strides)` per inner block; the destination is `strides_list[0]`.
///
/// The destination layout must already be validated injective, which makes
/// the blocks' destination regions disjoint when they run on different
/// threads.
fn run_update<F>(dims: &[usize], strides_list: &[&[isize]], elem_size: usize, body: F) -> Result<()>
where
    F: Fn(&[isize], usize, &[isize]) + MaybeSync,
{
    let total = total_len(dims)?;
    if total == 0 {
        return Ok(());
    }
    // One element (rank 0, or every extent one): a single length-1 block.
    if dims.iter().all(|&d| d == 1) {
        let zeros = vec![0isize; strides_list.len()];
        body(&zeros, 1, &zeros);
        return Ok(());
    }
    // Small tensor fast path: skip compute_order and compute_block_sizes
    let (fused_dims, ordered_strides, plan) = if total <= SMALL_TENSOR_THRESHOLD {
        build_plan_fused_small(dims, strides_list)
    } else {
        build_plan_fused(dims, strides_list, Some(0), elem_size)
    };

    #[cfg(feature = "parallel")]
    {
        let total = total_len(&fused_dims)?;
        let nthreads = crate::execution_policy::rayon_threads();
        if total > MINTHREADLENGTH && nthreads > 1 {
            let costs = compute_costs(&ordered_strides);
            let initial_offsets = vec![0isize; strides_list.len()];
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
                            body(offsets, len, strides);
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
            body(offsets, len, strides);
            Ok(())
        },
    )
}

/// Unary inner loop: `d[i] = f(OpD(d[i]))`.
///
/// # Safety
/// `dp` addresses `len` live, exclusively owned elements at stride `ds`.
#[inline(always)]
unsafe fn inner_loop_update1<D: Copy, OpD: ElementOp<D>>(
    dp: *mut D,
    ds: isize,
    len: usize,
    f: &impl Fn(D) -> D,
) {
    if ds == 1 {
        let dst = std::slice::from_raw_parts_mut(dp, len);
        simd::dispatch_if_large(len, || {
            for d in dst.iter_mut() {
                *d = f(OpD::apply(*d));
            }
        });
    } else {
        let mut dp = dp;
        for _ in 0..len {
            *dp = f(OpD::apply(*dp));
            dp = dp.offset(ds);
        }
    }
}

/// Binary inner loop: `d[i] = f(OpD(d[i]), OpA(a[i]))`.
///
/// # Safety
/// As [`inner_loop_update1`]; `ap` addresses `len` live elements that do not
/// overlap the destination.
#[inline(always)]
unsafe fn inner_loop_update2<D: Copy, A: Copy, OpD: ElementOp<D>, OpA: ElementOp<A>>(
    dp: *mut D,
    ds: isize,
    ap: *const A,
    a_s: isize,
    len: usize,
    f: &impl Fn(D, A) -> D,
) {
    if ds == 1 && a_s == 1 {
        let src_a = std::slice::from_raw_parts(ap, len);
        let dst = std::slice::from_raw_parts_mut(dp, len);
        simd::dispatch_if_large(len, || {
            for (d, &a) in dst.iter_mut().zip(src_a) {
                *d = f(OpD::apply(*d), OpA::apply(a));
            }
        });
    } else {
        let (mut dp, mut ap) = (dp, ap);
        for _ in 0..len {
            *dp = f(OpD::apply(*dp), OpA::apply(*ap));
            dp = dp.offset(ds);
            ap = ap.offset(a_s);
        }
    }
}

/// Ternary inner loop: `d[i] = f(OpD(d[i]), OpA(a[i]), OpB(b[i]))`.
///
/// # Safety
/// As [`inner_loop_update2`], with `bp` likewise.
#[inline(always)]
#[allow(clippy::too_many_arguments)] // INVARIANT: three strided operands, length and body.
unsafe fn inner_loop_update3<
    D: Copy,
    A: Copy,
    B: Copy,
    OpD: ElementOp<D>,
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
    f: &impl Fn(D, A, B) -> D,
) {
    if ds == 1 && a_s == 1 && b_s == 1 {
        let src_a = std::slice::from_raw_parts(ap, len);
        let src_b = std::slice::from_raw_parts(bp, len);
        let dst = std::slice::from_raw_parts_mut(dp, len);
        simd::dispatch_if_large(len, || {
            for ((d, &a), &b) in dst.iter_mut().zip(src_a).zip(src_b) {
                *d = f(OpD::apply(*d), OpA::apply(a), OpB::apply(b));
            }
        });
    } else {
        let (mut dp, mut ap, mut bp) = (dp, ap, bp);
        for _ in 0..len {
            *dp = f(OpD::apply(*dp), OpA::apply(*ap), OpB::apply(*bp));
            dp = dp.offset(ds);
            ap = ap.offset(a_s);
            bp = bp.offset(b_s);
        }
    }
}

/// Update in place: `dest[i] = f(OpD(dest[i]))`.
///
/// `OpD` is applied lazily to the old value before `f` sees it.
///
/// # Errors
///
/// [`StridedError::NonInjectiveOutputLayout`] when `dest` maps two indices to
/// one element (checked before any write).
///
/// # Examples
///
/// ```
/// use strided_basic::{map_update_into, Identity, StridedArray};
/// let mut d = StridedArray::<f64>::from_fn_col_major(&[2, 2], |i| (i[0] + 2 * i[1]) as f64);
/// map_update_into::<_, Identity>(&mut d.view_mut(), |x| 2.0 * x + 1.0).unwrap();
/// assert_eq!(d.get(&[1, 1]), 7.0);
/// ```
pub fn map_update_into<D, OpD>(
    dest: &mut StridedViewMut<D>,
    f: impl Fn(D) -> D + MaybeSync,
) -> Result<()>
where
    D: Copy + MaybeSendSync,
    OpD: ElementOp<D>,
{
    validate_destination(dest.dims(), dest.strides())?;
    let dp = Raw(dest.as_mut_ptr());
    run_update(
        dest.dims(),
        &[dest.strides()],
        std::mem::size_of::<D>(),
        |offsets, len, strides| {
            // SAFETY: the destination is injective and in bounds; blocks are disjoint.
            unsafe {
                inner_loop_update1::<D, OpD>(dp.0.offset(offsets[0]), strides[0], len, &f);
            }
        },
    )
}

/// Update in place from one input: `dest[i] = f(OpD(dest[i]), OpA(a[i]))`.
///
/// # Errors
///
/// [`StridedError::ShapeMismatch`] for unequal shapes and
/// [`StridedError::NonInjectiveOutputLayout`], both before any write.
///
/// # Examples
///
/// ```
/// use strided_basic::{zip_update2_into, Identity, StridedArray};
/// let a = StridedArray::<f64>::from_fn_col_major(&[3], |i| i[0] as f64);
/// let mut d = StridedArray::<f64>::from_fn_col_major(&[3], |_| 10.0);
/// // d = 2 * a + 0.5 * d
/// zip_update2_into::<_, _, Identity, Identity>(&mut d.view_mut(), &a.view(), |d, a| {
///     2.0 * a + 0.5 * d
/// })
/// .unwrap();
/// assert_eq!(d.get(&[2]), 9.0);
/// ```
pub fn zip_update2_into<D, A, OpD, OpA>(
    dest: &mut StridedViewMut<D>,
    a: &StridedView<A, OpA>,
    f: impl Fn(D, A) -> D + MaybeSync,
) -> Result<()>
where
    D: Copy + MaybeSendSync,
    A: Copy + MaybeSendSync,
    OpD: ElementOp<D>,
    OpA: ElementOp<A>,
{
    ensure_same_shape(dest.dims(), a.dims())?;
    validate_destination(dest.dims(), dest.strides())?;
    let dp = Raw(dest.as_mut_ptr());
    let ap = Raw(a.ptr() as *mut A);
    run_update(
        dest.dims(),
        &[dest.strides(), a.strides()],
        std::mem::size_of::<D>().max(std::mem::size_of::<A>()),
        |offsets, len, strides| {
            // SAFETY: bounds are the views'; `a` is a distinct borrow from `dest`.
            unsafe {
                inner_loop_update2::<D, A, OpD, OpA>(
                    dp.0.offset(offsets[0]),
                    strides[0],
                    ap.0.offset(offsets[1]).cast_const(),
                    strides[1],
                    len,
                    &f,
                );
            }
        },
    )
}

/// Update in place from two inputs:
/// `dest[i] = f(OpD(dest[i]), OpA(a[i]), OpB(b[i]))`.
///
/// # Errors
///
/// As [`zip_update2_into`].
///
/// # Examples
///
/// ```
/// use strided_basic::{zip_update3_into, Conj, Identity, StridedArray};
/// use num_complex::Complex64;
/// let one = Complex64::new(1.0, 1.0);
/// let a = StridedArray::<Complex64>::from_fn_col_major(&[2], |_| one);
/// let b = StridedArray::<Complex64>::from_fn_col_major(&[2], |_| one);
/// let mut d = StridedArray::<Complex64>::from_fn_col_major(&[2], |_| one);
/// // d = conj(d) + a * b
/// zip_update3_into::<_, _, _, Conj, Identity, Identity>(
///     &mut d.view_mut(), &a.view(), &b.view(), |d, a, b| d + a * b,
/// )
/// .unwrap();
/// assert_eq!(d.get(&[0]), Complex64::new(1.0, 1.0)); // (1 - i) + 2i
/// ```
pub fn zip_update3_into<D, A, B, OpD, OpA, OpB>(
    dest: &mut StridedViewMut<D>,
    a: &StridedView<A, OpA>,
    b: &StridedView<B, OpB>,
    f: impl Fn(D, A, B) -> D + MaybeSync,
) -> Result<()>
where
    D: Copy + MaybeSendSync,
    A: Copy + MaybeSendSync,
    B: Copy + MaybeSendSync,
    OpD: ElementOp<D>,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
{
    ensure_same_shape(dest.dims(), a.dims())?;
    ensure_same_shape(dest.dims(), b.dims())?;
    validate_destination(dest.dims(), dest.strides())?;
    let dp = Raw(dest.as_mut_ptr());
    let ap = Raw(a.ptr() as *mut A);
    let bp = Raw(b.ptr() as *mut B);
    run_update(
        dest.dims(),
        &[dest.strides(), a.strides(), b.strides()],
        std::mem::size_of::<D>()
            .max(std::mem::size_of::<A>())
            .max(std::mem::size_of::<B>()),
        |offsets, len, strides| {
            // SAFETY: bounds are the views'; `a`, `b` are distinct borrows from `dest`.
            unsafe {
                inner_loop_update3::<D, A, B, OpD, OpA, OpB>(
                    dp.0.offset(offsets[0]),
                    strides[0],
                    ap.0.offset(offsets[1]).cast_const(),
                    strides[1],
                    bp.0.offset(offsets[2]).cast_const(),
                    strides[2],
                    len,
                    &f,
                );
            }
        },
    )
}

#[cfg(test)]
#[path = "update_view/tests/tests.rs"]
mod tests;
