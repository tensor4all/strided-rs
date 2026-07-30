//! Map operations on dynamic-rank strided views.
//!
//! These are the canonical view-based map functions, equivalent to Julia's `Base.map!`.

use crate::kernel::{
    build_plan_fused, build_plan_fused_small, ensure_same_shape, for_each_inner_block_preordered,
    sequential_contiguous_layout, total_len, SMALL_TENSOR_THRESHOLD,
};
use crate::maybe_sync::{MaybeSendSync, MaybeSync};
use crate::simd;
use crate::view::{StridedView, StridedViewMut};
use crate::{Result, StridedError};
use std::ops::Mul;
use strided_view::ElementOp;

#[cfg(feature = "parallel")]
use crate::fuse::compute_costs;
#[cfg(feature = "parallel")]
use crate::threading::{for_each_inner_block_with_offsets, mapreduce_threaded, MINTHREADLENGTH};
#[cfg(feature = "parallel")]
use smallvec::SmallVec;

#[cfg(feature = "parallel")]
type AxisVec<T> = SmallVec<[T; 8]>;
#[cfg(not(feature = "parallel"))]
type AxisVec<T> = Vec<T>;

const CONTIGUOUS_RANGE_MIN_LEN: usize = 1 << 15;

#[inline]
fn validate_destination_layout(dims: &[usize], strides: &[isize]) -> Result<()> {
    if crate::fused::is_injective_layout(dims, strides) {
        Ok(())
    } else {
        Err(StridedError::NonInjectiveOutputLayout)
    }
}

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
    } else if ds == 1 && a_s == 1 && b_s == 0 {
        let src_a = std::slice::from_raw_parts(ap, len);
        let b = OpB::apply(*bp);
        let dst = std::slice::from_raw_parts_mut(dp, len);
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                dst[i] = f(OpA::apply(src_a[i]), b);
            }
        });
    } else if ds == 1 && a_s == 0 && b_s == 1 {
        let a = OpA::apply(*ap);
        let src_b = std::slice::from_raw_parts(bp, len);
        let dst = std::slice::from_raw_parts_mut(dp, len);
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                dst[i] = f(a, OpB::apply(src_b[i]));
            }
        });
    } else if ds == 1 && a_s == 0 && b_s == 0 {
        let a = OpA::apply(*ap);
        let b = OpB::apply(*bp);
        let dst = std::slice::from_raw_parts_mut(dp, len);
        simd::dispatch_if_large(len, || {
            for d in dst.iter_mut() {
                *d = f(a, b);
            }
        });
    } else if ds == 1 && b_s == 0 {
        let b = OpB::apply(*bp);
        let dst = std::slice::from_raw_parts_mut(dp, len);
        let mut ap = ap;
        simd::dispatch_if_large(len, || {
            for d in dst.iter_mut() {
                *d = f(OpA::apply(*ap), b);
                ap = ap.offset(a_s);
            }
        });
    } else if ds == 1 && a_s == 0 {
        let a = OpA::apply(*ap);
        let dst = std::slice::from_raw_parts_mut(dp, len);
        let mut bp = bp;
        simd::dispatch_if_large(len, || {
            for d in dst.iter_mut() {
                *d = f(a, OpB::apply(*bp));
                bp = bp.offset(b_s);
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

/// Binary multiplication inner loop for identity element ops.
#[inline(always)]
unsafe fn inner_loop_mul2<
    D: Copy + 'static,
    A: Copy + Mul<B, Output = D> + 'static,
    B: Copy + 'static,
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
        let src_a = std::slice::from_raw_parts(ap, len);
        let src_b = std::slice::from_raw_parts(bp, len);
        let dst = std::slice::from_raw_parts_mut(dp, len);
        if len >= 64 && simd::try_mul_contiguous(dst, src_a, src_b) {
            return;
        }
        for i in 0..len {
            dst[i] = src_a[i] * src_b[i];
        }
    } else if ds == 1 && a_s == 1 && b_s == 0 {
        let src_a = std::slice::from_raw_parts(ap, len);
        let b = *bp;
        let dst = std::slice::from_raw_parts_mut(dp, len);
        for i in 0..len {
            dst[i] = src_a[i] * b;
        }
    } else if ds == 1 && a_s == 0 && b_s == 1 {
        let a = *ap;
        let src_b = std::slice::from_raw_parts(bp, len);
        let dst = std::slice::from_raw_parts_mut(dp, len);
        for i in 0..len {
            dst[i] = a * src_b[i];
        }
    } else if ds == 1 && a_s == 0 && b_s == 0 {
        let a = *ap;
        let b = *bp;
        let dst = std::slice::from_raw_parts_mut(dp, len);
        for d in dst.iter_mut() {
            *d = a * b;
        }
    } else if ds == 1 && b_s == 0 {
        let b = *bp;
        let dst = std::slice::from_raw_parts_mut(dp, len);
        let mut ap = ap;
        for d in dst.iter_mut() {
            *d = *ap * b;
            ap = ap.offset(a_s);
        }
    } else if ds == 1 && a_s == 0 {
        let a = *ap;
        let dst = std::slice::from_raw_parts_mut(dp, len);
        let mut bp = bp;
        for d in dst.iter_mut() {
            *d = a * *bp;
            bp = bp.offset(b_s);
        }
    } else {
        let mut dp = dp;
        let mut ap = ap;
        let mut bp = bp;
        for _ in 0..len {
            *dp = *ap * *bp;
            dp = dp.offset(ds);
            ap = ap.offset(a_s);
            bp = bp.offset(b_s);
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ContiguousMulRangePlan {
    axis_order: AxisVec<usize>,
    inner_len: usize,
    inner_axis_count: usize,
    row_len: usize,
    outer_axis_start: usize,
    fast_axis: usize,
    a_fast_stride: isize,
    b_fast_stride: isize,
    a_row_stride: isize,
    b_row_stride: isize,
}

#[cfg(feature = "parallel")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TransposedScalarTileKind {
    RhsScalar,
    LhsScalar,
}

#[cfg(feature = "parallel")]
fn transposed_scalar_tile_kind(plan: &ContiguousMulRangePlan) -> Option<TransposedScalarTileKind> {
    let row_len = isize::try_from(plan.row_len).ok()?;
    if plan.a_fast_stride == row_len
        && plan.a_row_stride == 1
        && plan.b_fast_stride == 0
        && plan.b_row_stride == 0
    {
        return Some(TransposedScalarTileKind::RhsScalar);
    }

    if plan.b_fast_stride == row_len
        && plan.b_row_stride == 1
        && plan.a_fast_stride == 0
        && plan.a_row_stride == 0
    {
        return Some(TransposedScalarTileKind::LhsScalar);
    }

    None
}

fn compact_axis_order(dims: &[usize], strides: &[isize]) -> Option<AxisVec<usize>> {
    if dims.len() != strides.len() {
        return None;
    }

    let mut active = AxisVec::<usize>::new();
    let mut inactive = AxisVec::<usize>::new();
    for (axis, (&dim, &stride)) in dims.iter().zip(strides.iter()).enumerate() {
        if stride < 0 {
            return None;
        }
        if dim > 1 {
            active.push(axis);
        } else {
            inactive.push(axis);
        }
    }

    active.sort_by(|&lhs, &rhs| strides[lhs].cmp(&strides[rhs]).then_with(|| lhs.cmp(&rhs)));

    let mut expected = 1isize;
    for &axis in &active {
        if strides[axis] != expected {
            return None;
        }
        expected = expected.saturating_mul(dims[axis] as isize);
    }

    active.extend(inactive);
    Some(active)
}

fn can_fuse_contiguous_range_axis(dim: usize, prev_stride: isize, next_stride: isize) -> bool {
    dim <= 1 || (prev_stride == 0 && next_stride == 0) || next_stride == prev_stride * dim as isize
}

fn contiguous_mul_range_plan(
    dims: &[usize],
    dst_strides: &[isize],
    a_strides: &[isize],
    b_strides: &[isize],
) -> Option<ContiguousMulRangePlan> {
    let axis_order = compact_axis_order(dims, dst_strides)?;
    if dims.is_empty() {
        return Some(ContiguousMulRangePlan {
            axis_order,
            inner_len: 1,
            inner_axis_count: 0,
            row_len: 1,
            outer_axis_start: 0,
            fast_axis: 0,
            a_fast_stride: 0,
            b_fast_stride: 0,
            a_row_stride: 0,
            b_row_stride: 0,
        });
    }

    let first_pos = axis_order
        .iter()
        .position(|&axis| dims[axis] > 1)
        .unwrap_or(0);
    let first_axis = axis_order[first_pos];
    let mut inner_len = dims[first_axis].max(1);
    let mut inner_axis_count = first_pos + 1;
    let mut prev_axis = first_axis;

    for &axis in axis_order.iter().skip(first_pos + 1) {
        if can_fuse_contiguous_range_axis(
            dims[prev_axis],
            dst_strides[prev_axis],
            dst_strides[axis],
        ) && can_fuse_contiguous_range_axis(
            dims[prev_axis],
            a_strides[prev_axis],
            a_strides[axis],
        ) && can_fuse_contiguous_range_axis(
            dims[prev_axis],
            b_strides[prev_axis],
            b_strides[axis],
        ) {
            inner_len = inner_len.checked_mul(dims[axis].max(1))?;
            inner_axis_count += 1;
            prev_axis = axis;
        } else {
            break;
        }
    }

    let row = axis_order
        .iter()
        .enumerate()
        .skip(inner_axis_count)
        .find(|&(_, &axis)| dims[axis] > 1);
    let (row_len, outer_axis_start, a_row_stride, b_row_stride) =
        if let Some((row_pos, &row_axis)) = row {
            (
                dims[row_axis],
                row_pos + 1,
                a_strides[row_axis],
                b_strides[row_axis],
            )
        } else {
            (1, axis_order.len(), 0, 0)
        };

    Some(ContiguousMulRangePlan {
        axis_order,
        inner_len,
        inner_axis_count,
        row_len,
        outer_axis_start,
        fast_axis: first_axis,
        a_fast_stride: a_strides[first_axis],
        b_fast_stride: b_strides[first_axis],
        a_row_stride,
        b_row_stride,
    })
}

struct ContiguousMulOuterCursor<'a> {
    dims: &'a [usize],
    a_strides: &'a [isize],
    b_strides: &'a [isize],
    axes: AxisVec<usize>,
    coords: AxisVec<usize>,
    a_offset: isize,
    b_offset: isize,
}

impl<'a> ContiguousMulOuterCursor<'a> {
    fn new(
        dims: &'a [usize],
        a_strides: &'a [isize],
        b_strides: &'a [isize],
        plan: &ContiguousMulRangePlan,
        outer_group: usize,
    ) -> Self {
        let axes: AxisVec<usize> = plan
            .axis_order
            .iter()
            .skip(plan.outer_axis_start)
            .copied()
            .collect();
        let mut coords = AxisVec::<usize>::with_capacity(axes.len());
        let mut rem = outer_group;
        let mut a_offset = 0isize;
        let mut b_offset = 0isize;

        for &axis in &axes {
            let dim = dims[axis].max(1);
            let coord = rem % dim;
            rem /= dim;
            coords.push(coord);
            a_offset += coord as isize * a_strides[axis];
            b_offset += coord as isize * b_strides[axis];
        }

        Self {
            dims,
            a_strides,
            b_strides,
            axes,
            coords,
            a_offset,
            b_offset,
        }
    }

    fn advance(&mut self) {
        for (i, &axis) in self.axes.iter().enumerate() {
            let dim = self.dims[axis].max(1);
            if dim <= 1 {
                continue;
            }

            self.coords[i] += 1;
            self.a_offset += self.a_strides[axis];
            self.b_offset += self.b_strides[axis];

            if self.coords[i] < dim {
                break;
            }

            self.coords[i] = 0;
            self.a_offset -= dim as isize * self.a_strides[axis];
            self.b_offset -= dim as isize * self.b_strides[axis];
        }
    }
}

#[inline(always)]
unsafe fn run_contiguous_mul_row_block<
    D: Copy + 'static,
    A: Copy + Mul<B, Output = D> + 'static,
    B: Copy + 'static,
>(
    dst_ptr: *mut D,
    a_ptr: *const A,
    b_ptr: *const B,
    plan: &ContiguousMulRangePlan,
    base_index: usize,
    total: usize,
    base_a_offset: isize,
    base_b_offset: isize,
) {
    let inner_len = plan.inner_len.max(1);
    let row_len = plan.row_len.max(1);
    #[cfg(feature = "parallel")]
    let block_len = inner_len.saturating_mul(row_len);

    #[cfg(feature = "parallel")]
    if total.saturating_sub(base_index) >= block_len {
        match transposed_scalar_tile_kind(plan) {
            Some(TransposedScalarTileKind::RhsScalar) => {
                if simd::try_mul_transposed_scalar_rhs_2d::<D, A, B>(
                    dst_ptr.add(base_index),
                    a_ptr.offset(base_a_offset),
                    b_ptr.offset(base_b_offset),
                    inner_len,
                    row_len,
                    plan.a_fast_stride,
                    plan.a_row_stride,
                ) {
                    return;
                }
            }
            Some(TransposedScalarTileKind::LhsScalar) => {
                if simd::try_mul_transposed_scalar_lhs_2d::<D, A, B>(
                    dst_ptr.add(base_index),
                    a_ptr.offset(base_a_offset),
                    b_ptr.offset(base_b_offset),
                    inner_len,
                    row_len,
                    plan.b_fast_stride,
                    plan.b_row_stride,
                ) {
                    return;
                }
            }
            None => {}
        }
    }

    let mut index = base_index;
    let mut a_offset = base_a_offset;
    let mut b_offset = base_b_offset;

    for _ in 0..row_len {
        if index >= total {
            break;
        }
        let len = inner_len.min(total - index);
        inner_loop_mul2::<D, A, B>(
            dst_ptr.add(index),
            1,
            a_ptr.offset(a_offset),
            plan.a_fast_stride,
            b_ptr.offset(b_offset),
            plan.b_fast_stride,
            len,
        );
        index += inner_len;
        a_offset += plan.a_row_stride;
        b_offset += plan.b_row_stride;
    }
}

#[cfg(feature = "parallel")]
fn strided_offset_for_contiguous_linear_index(
    dims: &[usize],
    strides: &[isize],
    axis_order: &[usize],
    mut index: usize,
) -> isize {
    let mut offset = 0isize;
    for &axis in axis_order {
        let dim = dims[axis];
        if dim == 0 {
            return 0;
        }
        let coord = index % dim;
        index /= dim;
        offset += coord as isize * strides[axis];
    }
    offset
}

fn try_contiguous_range_mul<
    D: Copy + MaybeSendSync + 'static,
    A: Copy + MaybeSendSync + Mul<B, Output = D> + 'static,
    B: Copy + MaybeSendSync + 'static,
>(
    dst_ptr: *mut D,
    dims: &[usize],
    dst_strides: &[isize],
    a_ptr: *const A,
    a_strides: &[isize],
    b_ptr: *const B,
    b_strides: &[isize],
) -> bool {
    let total = total_len(dims);
    if total == 0 {
        return true;
    }
    if total <= CONTIGUOUS_RANGE_MIN_LEN {
        return false;
    }

    let Some(plan) = contiguous_mul_range_plan(dims, dst_strides, a_strides, b_strides) else {
        return false;
    };

    let inner_len = plan.inner_len.max(1);
    let row_len = plan.row_len.max(1);
    let block_len = inner_len.saturating_mul(row_len).max(1);
    let outer_groups = total.div_ceil(block_len);

    #[cfg(feature = "parallel")]
    {
        let nthreads = crate::execution_policy::rayon_threads();
        if nthreads > 1 {
            use crate::threading::{parallel_for_each, SendPtr};

            let dst = SendPtr(dst_ptr);
            let a = SendPtr(a_ptr as *mut A);
            let b = SendPtr(b_ptr as *mut B);

            if outer_groups < nthreads {
                let chunk_len = total.div_ceil(nthreads);
                let nchunks = total.div_ceil(chunk_len);

                parallel_for_each(0..nchunks, nthreads, &|chunks| {
                    for chunk in chunks {
                        let start = chunk * chunk_len;
                        let end = (start + chunk_len).min(total);
                        let mut index = start;

                        while index < end {
                            let in_inner = index % inner_len;
                            let len = (inner_len - in_inner).min(end - index);
                            let a_offset = strided_offset_for_contiguous_linear_index(
                                dims,
                                a_strides,
                                &plan.axis_order,
                                index,
                            );
                            let b_offset = strided_offset_for_contiguous_linear_index(
                                dims,
                                b_strides,
                                &plan.axis_order,
                                index,
                            );

                            unsafe {
                                inner_loop_mul2::<D, A, B>(
                                    dst.as_ptr().add(index),
                                    1,
                                    a.as_const().offset(a_offset),
                                    plan.a_fast_stride,
                                    b.as_const().offset(b_offset),
                                    plan.b_fast_stride,
                                    len,
                                );
                            }
                            index += len;
                        }
                    }
                });

                return true;
            }

            let groups_per_chunk = outer_groups.div_ceil(nthreads);
            let nchunks = outer_groups.div_ceil(groups_per_chunk);

            parallel_for_each(0..nchunks, nthreads, &|chunks| {
                for chunk in chunks {
                    let group_start = chunk * groups_per_chunk;
                    let group_end = (group_start + groups_per_chunk).min(outer_groups);
                    let mut cursor = ContiguousMulOuterCursor::new(
                        dims,
                        a_strides,
                        b_strides,
                        &plan,
                        group_start,
                    );

                    for group in group_start..group_end {
                        let index = group * block_len;
                        unsafe {
                            run_contiguous_mul_row_block::<D, A, B>(
                                dst.as_ptr(),
                                a.as_const(),
                                b.as_const(),
                                &plan,
                                index,
                                total,
                                cursor.a_offset,
                                cursor.b_offset,
                            );
                        }
                        cursor.advance();
                    }
                }
            });

            true
        } else {
            run_contiguous_range_mul_single_thread(
                dst_ptr,
                dims,
                a_ptr,
                a_strides,
                b_ptr,
                b_strides,
                &plan,
                total,
                block_len,
                outer_groups,
            )
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        run_contiguous_range_mul_single_thread(
            dst_ptr,
            dims,
            a_ptr,
            a_strides,
            b_ptr,
            b_strides,
            &plan,
            total,
            block_len,
            outer_groups,
        )
    }
}

fn run_contiguous_range_mul_single_thread<
    D: Copy + MaybeSendSync + 'static,
    A: Copy + MaybeSendSync + Mul<B, Output = D> + 'static,
    B: Copy + MaybeSendSync + 'static,
>(
    dst_ptr: *mut D,
    dims: &[usize],
    a_ptr: *const A,
    a_strides: &[isize],
    b_ptr: *const B,
    b_strides: &[isize],
    plan: &ContiguousMulRangePlan,
    total: usize,
    block_len: usize,
    outer_groups: usize,
) -> bool {
    let mut cursor = ContiguousMulOuterCursor::new(dims, a_strides, b_strides, plan, 0);
    for group in 0..outer_groups {
        let index = group * block_len;
        unsafe {
            run_contiguous_mul_row_block::<D, A, B>(
                dst_ptr,
                a_ptr,
                b_ptr,
                plan,
                index,
                total,
                cursor.a_offset,
                cursor.b_offset,
            );
        }
        cursor.advance();
    }
    true
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
    map_parts_into::<D, A, Op>(
        dest.as_mut_ptr(),
        dest.dims(),
        dest.strides(),
        src.ptr(),
        src.dims(),
        src.strides(),
        f,
    )
}

pub(crate) fn map_raw_into<D: Copy + MaybeSendSync, A: Copy + MaybeSendSync, Op: ElementOp<A>>(
    dest: &mut crate::RawStridedMut<'_, D>,
    src: &crate::RawStridedRef<'_, A>,
    f: impl Fn(A) -> D + MaybeSync,
) -> Result<()> {
    map_parts_into::<D, A, Op>(
        dest.as_mut_ptr(),
        dest.dims(),
        dest.strides(),
        src.ptr(),
        src.dims(),
        src.strides(),
        f,
    )
}

#[allow(clippy::too_many_arguments)]
fn map_parts_into<D: Copy + MaybeSendSync, A: Copy + MaybeSendSync, Op: ElementOp<A>>(
    dst_ptr: *mut D,
    dst_dims: &[usize],
    dst_strides: &[isize],
    src_ptr: *const A,
    src_dims: &[usize],
    src_strides: &[isize],
    f: impl Fn(A) -> D + MaybeSync,
) -> Result<()> {
    ensure_same_shape(dst_dims, src_dims)?;
    validate_destination_layout(dst_dims, dst_strides)?;

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
    let total = total_len(dst_dims);

    // Small tensor fast path: skip compute_order and compute_block_sizes
    let (fused_dims, ordered_strides, plan) = if total <= SMALL_TENSOR_THRESHOLD {
        build_plan_fused_small(dst_dims, &strides_list)
    } else {
        build_plan_fused(dst_dims, &strides_list, Some(0), elem_size)
    };

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        let nthreads = crate::execution_policy::rayon_threads();
        if total > MINTHREADLENGTH && nthreads > 1 {
            use crate::threading::SendPtr;
            let dst_send = SendPtr(dst_ptr);
            let src_send = SendPtr(src_ptr as *mut A);

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
    zip_map2_parts_into::<D, A, B, OpA, OpB>(
        dest.as_mut_ptr(),
        dest.dims(),
        dest.strides(),
        a.ptr(),
        a.dims(),
        a.strides(),
        b.ptr(),
        b.dims(),
        b.strides(),
        f,
    )
}

/// Runtime comparison selected once before entering the element loop.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompareOp {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Compare two ordered views elementwise into a Boolean destination.
///
/// Unlike embedding a runtime operation match inside a [`zip_map2_into`]
/// closure, this entry point selects a fixed comparison before traversal.
///
/// # Errors
///
/// Returns [`StridedError::DimensionMismatch`] when the source and destination
/// shapes differ, or [`StridedError::NonInjectiveOutputLayout`] when distinct
/// logical destination elements may overlap.
pub fn compare_into<T, OpA, OpB>(
    dest: &mut StridedViewMut<bool>,
    a: &StridedView<T, OpA>,
    b: &StridedView<T, OpB>,
    op: CompareOp,
) -> Result<()>
where
    T: Copy + MaybeSendSync + PartialOrd,
    OpA: ElementOp<T>,
    OpB: ElementOp<T>,
{
    match op {
        CompareOp::Eq => zip_map2_into(dest, a, b, |lhs, rhs| lhs == rhs),
        CompareOp::Lt => zip_map2_into(dest, a, b, |lhs, rhs| lhs < rhs),
        CompareOp::Le => zip_map2_into(dest, a, b, |lhs, rhs| lhs <= rhs),
        CompareOp::Gt => zip_map2_into(dest, a, b, |lhs, rhs| lhs > rhs),
        CompareOp::Ge => zip_map2_into(dest, a, b, |lhs, rhs| lhs >= rhs),
    }
}

pub(crate) fn zip_map2_raw_into<
    D: Copy + MaybeSendSync,
    A: Copy + MaybeSendSync,
    B: Copy + MaybeSendSync,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
>(
    dest: &mut crate::RawStridedMut<'_, D>,
    a: &crate::RawStridedRef<'_, A>,
    b: &crate::RawStridedRef<'_, B>,
    f: impl Fn(A, B) -> D + MaybeSync,
) -> Result<()> {
    zip_map2_parts_into::<D, A, B, OpA, OpB>(
        dest.as_mut_ptr(),
        dest.dims(),
        dest.strides(),
        a.ptr(),
        a.dims(),
        a.strides(),
        b.ptr(),
        b.dims(),
        b.strides(),
        f,
    )
}

#[allow(clippy::too_many_arguments)]
fn zip_map2_parts_into<
    D: Copy + MaybeSendSync,
    A: Copy + MaybeSendSync,
    B: Copy + MaybeSendSync,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
>(
    dst_ptr: *mut D,
    dst_dims: &[usize],
    dst_strides: &[isize],
    a_ptr: *const A,
    a_dims: &[usize],
    a_strides: &[isize],
    b_ptr: *const B,
    b_dims: &[usize],
    b_strides: &[isize],
    f: impl Fn(A, B) -> D + MaybeSync,
) -> Result<()> {
    ensure_same_shape(dst_dims, a_dims)?;
    ensure_same_shape(dst_dims, b_dims)?;
    validate_destination_layout(dst_dims, dst_strides)?;

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
    let total = total_len(dst_dims);

    // Small tensor fast path: skip compute_order and compute_block_sizes
    let (fused_dims, ordered_strides, plan) = if total <= SMALL_TENSOR_THRESHOLD {
        build_plan_fused_small(dst_dims, &strides_list)
    } else {
        build_plan_fused(dst_dims, &strides_list, Some(0), elem_size)
    };

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        let nthreads = crate::execution_policy::rayon_threads();
        if total > MINTHREADLENGTH && nthreads > 1 {
            use crate::threading::SendPtr;
            let dst_send = SendPtr(dst_ptr);
            let a_send = SendPtr(a_ptr as *mut A);
            let b_send = SendPtr(b_ptr as *mut B);

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

fn mul_identity_into_raw<
    D: Copy + MaybeSendSync + 'static,
    A: Copy + MaybeSendSync + Mul<B, Output = D> + 'static,
    B: Copy + MaybeSendSync + 'static,
>(
    dest: &mut StridedViewMut<D>,
    a_ptr: *const A,
    a_strides: &[isize],
    b_ptr: *const B,
    b_strides: &[isize],
) -> Result<()> {
    let dst_ptr = dest.as_mut_ptr();
    let dst_dims = dest.dims();
    let dst_strides = dest.strides();
    debug_assert_eq!(dst_dims.len(), a_strides.len());
    debug_assert_eq!(dst_dims.len(), b_strides.len());

    if sequential_contiguous_layout(dst_dims, &[dst_strides, a_strides, b_strides]).is_some() {
        let len = total_len(dst_dims);
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, len) };
        let sa = unsafe { std::slice::from_raw_parts(a_ptr, len) };
        let sb = unsafe { std::slice::from_raw_parts(b_ptr, len) };
        if simd::try_mul_contiguous(dst, sa, sb) {
            return Ok(());
        }
        for i in 0..len {
            dst[i] = sa[i] * sb[i];
        }
        return Ok(());
    }

    let strides_list: [&[isize]; 3] = [dst_strides, a_strides, b_strides];
    let elem_size = std::mem::size_of::<D>()
        .max(std::mem::size_of::<A>())
        .max(std::mem::size_of::<B>());
    let total = total_len(dst_dims);

    if try_contiguous_range_mul(
        dst_ptr,
        dst_dims,
        dst_strides,
        a_ptr,
        a_strides,
        b_ptr,
        b_strides,
    ) {
        return Ok(());
    }

    let (fused_dims, ordered_strides, plan) = if total <= SMALL_TENSOR_THRESHOLD {
        build_plan_fused_small(dst_dims, &strides_list)
    } else {
        build_plan_fused(dst_dims, &strides_list, Some(0), elem_size)
    };

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        let nthreads = crate::execution_policy::rayon_threads();
        if total > MINTHREADLENGTH && nthreads > 1 {
            use crate::threading::SendPtr;
            let dst_send = SendPtr(dst_ptr);
            let a_send = SendPtr(a_ptr as *mut A);
            let b_send = SendPtr(b_ptr as *mut B);

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
                            let dp = unsafe { dst_send.as_ptr().offset(offsets[0]) };
                            let ap = unsafe { a_send.as_const().offset(offsets[1]) };
                            let bp = unsafe { b_send.as_const().offset(offsets[2]) };
                            unsafe {
                                inner_loop_mul2::<D, A, B>(
                                    dp, strides[0], ap, strides[1], bp, strides[2], len,
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
                inner_loop_mul2::<D, A, B>(dp, strides[0], ap, strides[1], bp, strides[2], len)
            };
            Ok(())
        },
    )
}

fn mul_identity_into<
    D: Copy + MaybeSendSync + 'static,
    A: Copy + Mul<B, Output = D> + MaybeSendSync + 'static,
    B: Copy + MaybeSendSync + 'static,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
>(
    dest: &mut StridedViewMut<D>,
    a: &StridedView<A, OpA>,
    b: &StridedView<B, OpB>,
) -> Result<()> {
    ensure_same_shape(dest.dims(), a.dims())?;
    ensure_same_shape(dest.dims(), b.dims())?;
    mul_identity_into_raw(dest, a.ptr(), a.strides(), b.ptr(), b.strides())
}

/// Element-wise multiplication: `dest[i] = a[i] * b[i]`.
///
/// All views must have the same shape. Broadcast operands should be represented
/// as stride-0 views before calling this function.
pub fn mul_into<
    D: Copy + MaybeSendSync + 'static,
    A: Copy + Mul<B, Output = D> + MaybeSendSync + 'static,
    B: Copy + MaybeSendSync + 'static,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
>(
    dest: &mut StridedViewMut<D>,
    a: &StridedView<A, OpA>,
    b: &StridedView<B, OpB>,
) -> Result<()> {
    if OpA::IS_IDENTITY && OpB::IS_IDENTITY {
        return mul_identity_into(dest, a, b);
    }

    zip_map2_into(dest, a, b, |x, y| x * y)
}

fn broadcast_strides_for_axes(
    source_dims: &[usize],
    source_strides: &[isize],
    target_dims: &[usize],
    axes: &[usize],
) -> Result<AxisVec<isize>> {
    if source_dims.len() != axes.len() {
        return Err(StridedError::RankMismatch(source_dims.len(), axes.len()));
    }
    debug_assert_eq!(source_dims.len(), source_strides.len());

    let mut seen = AxisVec::<bool>::new();
    seen.resize(target_dims.len(), false);
    let mut strides = AxisVec::<isize>::new();
    strides.resize(target_dims.len(), 0);
    for (src_axis, &dst_axis) in axes.iter().enumerate() {
        if dst_axis >= target_dims.len() {
            return Err(StridedError::InvalidAxis {
                axis: dst_axis,
                rank: target_dims.len(),
            });
        }
        if seen[dst_axis] {
            return Err(StridedError::InvalidAxis {
                axis: dst_axis,
                rank: target_dims.len(),
            });
        }
        seen[dst_axis] = true;

        let source_dim = source_dims[src_axis];
        let target_dim = target_dims[dst_axis];
        if source_dim != target_dim && source_dim != 1 {
            return Err(StridedError::ShapeMismatch(
                source_dims.to_vec(),
                target_dims.to_vec(),
            ));
        }
        if source_dim == target_dim {
            strides[dst_axis] = source_strides[src_axis];
        }
    }

    Ok(strides)
}

fn broadcast_view_with_strides<'a, T, Op: ElementOp<T>>(
    view: &StridedView<'a, T, Op>,
    target_dims: &[usize],
    strides: &[isize],
) -> StridedView<'a, T, Op> {
    unsafe { StridedView::new_unchecked(view.data(), target_dims, strides, view.offset()) }
}

/// Broadcasted element-wise multiplication: `dest[i] = a[i] * b[i]`.
///
/// `a_axes` and `b_axes` map each source axis to an axis of `dest`. Output axes
/// not referenced by a source operand are treated as stride-0 broadcast axes.
pub fn broadcast_mul_into<
    D: Copy + MaybeSendSync + 'static,
    A: Copy + Mul<B, Output = D> + MaybeSendSync + 'static,
    B: Copy + MaybeSendSync + 'static,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
>(
    dest: &mut StridedViewMut<D>,
    a: &StridedView<A, OpA>,
    a_axes: &[usize],
    b: &StridedView<B, OpB>,
    b_axes: &[usize],
) -> Result<()> {
    let a_strides = broadcast_strides_for_axes(a.dims(), a.strides(), dest.dims(), a_axes)?;
    let b_strides = broadcast_strides_for_axes(b.dims(), b.strides(), dest.dims(), b_axes)?;

    if OpA::IS_IDENTITY && OpB::IS_IDENTITY {
        return mul_identity_into_raw(dest, a.ptr(), &a_strides, b.ptr(), &b_strides);
    }

    let a = broadcast_view_with_strides(a, dest.dims(), &a_strides);
    let b = broadcast_view_with_strides(b, dest.dims(), &b_strides);
    mul_into(dest, &a, &b)
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
    validate_destination_layout(dest.dims(), dest.strides())?;

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
    let total = total_len(dst_dims);

    // Small tensor fast path: skip compute_order and compute_block_sizes
    let (fused_dims, ordered_strides, plan) = if total <= SMALL_TENSOR_THRESHOLD {
        build_plan_fused_small(dst_dims, &strides_list)
    } else {
        build_plan_fused(dst_dims, &strides_list, Some(0), elem_size)
    };

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        let nthreads = crate::execution_policy::rayon_threads();
        if total > MINTHREADLENGTH && nthreads > 1 {
            use crate::threading::SendPtr;
            let dst_send = SendPtr(dst_ptr);
            let a_send = SendPtr(a_ptr as *mut A);
            let b_send = SendPtr(b_ptr as *mut B);
            let c_send = SendPtr(c_ptr as *mut C);

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
    validate_destination_layout(dest.dims(), dest.strides())?;

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
    let total = total_len(dst_dims);

    // Small tensor fast path: skip compute_order and compute_block_sizes
    let (fused_dims, ordered_strides, plan) = if total <= SMALL_TENSOR_THRESHOLD {
        build_plan_fused_small(dst_dims, &strides_list)
    } else {
        build_plan_fused(dst_dims, &strides_list, Some(0), elem_size)
    };

    #[cfg(feature = "parallel")]
    {
        let total: usize = fused_dims.iter().product();
        let nthreads = crate::execution_policy::rayon_threads();
        if total > MINTHREADLENGTH && nthreads > 1 {
            use crate::threading::SendPtr;
            let dst_send = SendPtr(dst_ptr);
            let a_send = SendPtr(a_ptr as *mut A);
            let b_send = SendPtr(b_ptr as *mut B);
            let c_send = SendPtr(c_ptr as *mut C);
            let e_send = SendPtr(e_ptr as *mut E);

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

#[cfg(test)]
mod scalar_branch_tests {
    use super::*;
    use crate::StridedArray;
    use strided_view::Identity;

    #[test]
    fn test_inner_loop_map2_stride_specializations() {
        let a = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0];
        let b = [17.0, 19.0, 23.0, 29.0, 31.0, 37.0];

        let mut out = [0.0; 3];
        unsafe {
            inner_loop_map2::<f64, f64, f64, Identity, Identity>(
                out.as_mut_ptr(),
                1,
                a.as_ptr(),
                1,
                b.as_ptr(),
                1,
                3,
                &|x, y| x + y,
            );
        }
        assert_eq!(out, [19.0, 22.0, 28.0]);

        let mut out = [0.0; 3];
        unsafe {
            inner_loop_map2::<f64, f64, f64, Identity, Identity>(
                out.as_mut_ptr(),
                1,
                a.as_ptr(),
                1,
                b.as_ptr(),
                0,
                3,
                &|x, y| x * y,
            );
        }
        assert_eq!(out, [34.0, 51.0, 85.0]);

        let mut out = [0.0; 3];
        unsafe {
            inner_loop_map2::<f64, f64, f64, Identity, Identity>(
                out.as_mut_ptr(),
                1,
                a.as_ptr(),
                0,
                b.as_ptr(),
                1,
                3,
                &|x, y| x * y,
            );
        }
        assert_eq!(out, [34.0, 38.0, 46.0]);

        let mut out = [0.0; 3];
        unsafe {
            inner_loop_map2::<f64, f64, f64, Identity, Identity>(
                out.as_mut_ptr(),
                1,
                a.as_ptr(),
                0,
                b.as_ptr(),
                0,
                3,
                &|x, y| x + y,
            );
        }
        assert_eq!(out, [19.0, 19.0, 19.0]);

        let mut out = [0.0; 3];
        unsafe {
            inner_loop_map2::<f64, f64, f64, Identity, Identity>(
                out.as_mut_ptr(),
                1,
                a.as_ptr(),
                2,
                b.as_ptr(),
                0,
                3,
                &|x, y| x + y,
            );
        }
        assert_eq!(out, [19.0, 22.0, 28.0]);

        let mut out = [0.0; 3];
        unsafe {
            inner_loop_map2::<f64, f64, f64, Identity, Identity>(
                out.as_mut_ptr(),
                1,
                a.as_ptr(),
                0,
                b.as_ptr(),
                2,
                3,
                &|x, y| x + y,
            );
        }
        assert_eq!(out, [19.0, 25.0, 33.0]);
    }

    #[test]
    fn test_inner_loop_mul2_stride_specializations() {
        let a = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0];
        let b = [17.0, 19.0, 23.0, 29.0, 31.0, 37.0];

        let mut out = [0.0; 3];
        unsafe {
            inner_loop_mul2::<f64, f64, f64>(out.as_mut_ptr(), 1, a.as_ptr(), 1, b.as_ptr(), 1, 3);
        }
        assert_eq!(out, [34.0, 57.0, 115.0]);

        let mut out = [0.0; 3];
        unsafe {
            inner_loop_mul2::<f64, f64, f64>(out.as_mut_ptr(), 1, a.as_ptr(), 0, b.as_ptr(), 1, 3);
        }
        assert_eq!(out, [34.0, 38.0, 46.0]);

        let mut out = [0.0; 3];
        unsafe {
            inner_loop_mul2::<f64, f64, f64>(out.as_mut_ptr(), 1, a.as_ptr(), 0, b.as_ptr(), 0, 3);
        }
        assert_eq!(out, [34.0, 34.0, 34.0]);

        let mut out = [0.0; 3];
        unsafe {
            inner_loop_mul2::<f64, f64, f64>(out.as_mut_ptr(), 1, a.as_ptr(), 0, b.as_ptr(), 2, 3);
        }
        assert_eq!(out, [34.0, 46.0, 62.0]);
    }

    #[test]
    fn test_broadcast_mul_into_error_branches_and_non_identity_ops() {
        let lhs = StridedArray::<f64>::row_major(&[2, 3]);
        let rhs = StridedArray::<f64>::row_major(&[2, 3]);
        let mut out = StridedArray::<f64>::row_major(&[2, 3]);

        let err = broadcast_mul_into(&mut out.view_mut(), &lhs.view(), &[0], &rhs.view(), &[0, 1])
            .unwrap_err();
        assert!(matches!(err, StridedError::RankMismatch(2, 1)));

        let err = broadcast_mul_into(
            &mut out.view_mut(),
            &lhs.view(),
            &[0, 3],
            &rhs.view(),
            &[0, 1],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            StridedError::InvalidAxis { axis: 3, rank: 2 }
        ));

        let err = broadcast_mul_into(
            &mut out.view_mut(),
            &lhs.view(),
            &[0, 0],
            &rhs.view(),
            &[0, 1],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            StridedError::InvalidAxis { axis: 0, rank: 2 }
        ));

        let rhs_bad = StridedArray::<f64>::row_major(&[2, 4]);
        let err = broadcast_mul_into(
            &mut out.view_mut(),
            &lhs.view(),
            &[0, 1],
            &rhs_bad.view(),
            &[0, 1],
        )
        .unwrap_err();
        assert!(matches!(err, StridedError::ShapeMismatch(_, _)));

        let lhs_conj = lhs.view().conj();
        broadcast_mul_into(
            &mut out.view_mut(),
            &lhs_conj,
            &[0, 1],
            &rhs.view(),
            &[0, 1],
        )
        .unwrap();
    }

    #[test]
    fn contiguous_mul_range_plan_available_without_parallel_feature() {
        let dims = [3usize; 16];
        let dst = [
            1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441, 1594323, 4782969,
            14348907,
        ];
        let lhs = [1isize, 3, 9, 27, 81, 243, 729, 2187, 0, 0, 0, 0, 0, 0, 0, 0];
        let rhs = [0isize, 0, 0, 0, 0, 0, 0, 0, 1, 3, 9, 27, 81, 243, 729, 2187];

        let plan = contiguous_mul_range_plan(&dims, &dst, &lhs, &rhs).unwrap();

        assert_eq!(plan.inner_len, 6561);
        assert_eq!(plan.row_len, 3);
        assert_eq!(plan.fast_axis, 0);
    }

    #[test]
    fn contiguous_range_mul_single_thread_computes_large_broadcast_mul() {
        let dims = [3usize; 10];
        let dst = [1isize, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683];
        let lhs = [1isize, 3, 9, 27, 81, 0, 0, 0, 0, 0];
        let rhs = [0isize, 0, 0, 0, 0, 1, 3, 9, 27, 81];
        let plan = contiguous_mul_range_plan(&dims, &dst, &lhs, &rhs).unwrap();
        let total = total_len(&dims);
        let block_len = plan.inner_len.max(1).saturating_mul(plan.row_len.max(1));
        let outer_groups = total.div_ceil(block_len);

        let a = vec![2.0; 243];
        let b = vec![3.0; 243];
        let mut out = vec![0.0; total];

        assert!(run_contiguous_range_mul_single_thread::<f64, f64, f64>(
            out.as_mut_ptr(),
            &dims,
            a.as_ptr(),
            &lhs,
            b.as_ptr(),
            &rhs,
            &plan,
            total,
            block_len,
            outer_groups,
        ));
        assert!(out.iter().all(|&x| x == 6.0));
    }
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;

    fn compact_strides_for_axis_order<const N: usize>(
        dims: [usize; N],
        axis_order: [usize; N],
    ) -> [isize; N] {
        let mut strides = [0isize; N];
        let mut stride = 1isize;
        for &axis in &axis_order {
            strides[axis] = stride;
            stride *= dims[axis] as isize;
        }
        strides
    }

    #[test]
    fn test_contiguous_mul_range_plan_pure_outer() {
        let dims = [7usize, 11];
        let dst = [1isize, 7];
        let lhs = [1isize, 0];
        let rhs = [0isize, 1];

        let plan = contiguous_mul_range_plan(&dims, &dst, &lhs, &rhs).unwrap();

        assert_eq!(plan.inner_len, 7);
        assert_eq!(plan.row_len, 11);
        assert_eq!(plan.fast_axis, 0);
        assert_eq!(plan.a_fast_stride, 1);
        assert_eq!(plan.b_fast_stride, 0);
        assert_eq!(plan.a_row_stride, 0);
        assert_eq!(plan.b_row_stride, 1);
    }

    #[test]
    fn test_compact_axis_order_accepts_all_rank4_axis_permutations() {
        fn visit(dims: [usize; 4], axes: &mut [usize; 4], pos: usize, count: &mut usize) {
            if pos == axes.len() {
                let dst = compact_strides_for_axis_order(dims, *axes);
                let axis_order = compact_axis_order(&dims, &dst).unwrap();
                assert_eq!(&axis_order[..], &axes[..]);
                *count += 1;
                return;
            }

            for i in pos..axes.len() {
                axes.swap(pos, i);
                visit(dims, axes, pos + 1, count);
                axes.swap(pos, i);
            }
        }

        let dims = [2usize, 3, 5, 7];
        let mut axes = [0usize, 1, 2, 3];
        let mut count = 0usize;
        visit(dims, &mut axes, 0, &mut count);

        assert_eq!(count, 24);
    }

    #[test]
    fn test_compact_axis_order_rejects_strided_layout_with_holes() {
        let dims = [2usize, 3, 5];
        let strides = [1isize, 4, 2];

        assert_eq!(compact_axis_order(&dims, &strides), None);
    }

    #[test]
    fn test_contiguous_mul_range_plan_uses_permuted_compact_output_for_unrelated_shape() {
        let dims = [2usize, 3, 5, 7, 11];
        let dst = compact_strides_for_axis_order(dims, [2usize, 0, 4, 1, 3]);
        let lhs = [5isize, 0, 1, 0, 10];
        let rhs = [0isize, 1, 0, 3, 0];

        let plan = contiguous_mul_range_plan(&dims, &dst, &lhs, &rhs).unwrap();

        assert_eq!(&plan.axis_order[..], &[2, 0, 4, 1, 3]);
        assert_eq!(plan.inner_len, 110);
        assert_eq!(plan.row_len, 3);
        assert_eq!(plan.fast_axis, 2);
        assert_eq!(plan.a_fast_stride, 1);
        assert_eq!(plan.b_fast_stride, 0);
        assert_eq!(plan.a_row_stride, 0);
        assert_eq!(plan.b_row_stride, 1);
        assert_eq!(transposed_scalar_tile_kind(&plan), None);
    }

    #[test]
    fn test_contiguous_mul_range_plan_compact_batched_outer() {
        let dims = [3usize, 5, 7, 11];
        let dst = [1isize, 3, 15, 105];
        let lhs = [1isize, 3, 0, 15];
        let rhs = [0isize, 0, 1, 7];

        let plan = contiguous_mul_range_plan(&dims, &dst, &lhs, &rhs).unwrap();

        assert_eq!(plan.inner_len, 15);
        assert_eq!(plan.row_len, 7);
        assert_eq!(plan.fast_axis, 0);
        assert_eq!(plan.a_fast_stride, 1);
        assert_eq!(plan.b_fast_stride, 0);
        assert_eq!(plan.a_row_stride, 0);
        assert_eq!(plan.b_row_stride, 1);
    }

    #[test]
    fn test_contiguous_mul_range_plan_noncompact_batched_outer() {
        let dims = [5usize, 5, 7, 11];
        let dst = [1isize, 5, 25, 175];
        let lhs = [5isize, 1, 0, 25];
        let rhs = [0isize, 0, 1, 7];

        let plan = contiguous_mul_range_plan(&dims, &dst, &lhs, &rhs).unwrap();

        assert_eq!(plan.inner_len, 5);
        assert_eq!(plan.row_len, 5);
        assert_eq!(plan.fast_axis, 0);
        assert_eq!(plan.a_fast_stride, 5);
        assert_eq!(plan.b_fast_stride, 0);
        assert_eq!(plan.a_row_stride, 1);
        assert_eq!(plan.b_row_stride, 0);
    }

    #[test]
    fn test_contiguous_mul_range_plan_noncompact_row_major_output() {
        let dims = [5usize, 5, 7, 11];
        let dst = [5isize, 1, 25, 175];
        let lhs = [5isize, 1, 0, 25];
        let rhs = [0isize, 0, 1, 7];

        let plan = contiguous_mul_range_plan(&dims, &dst, &lhs, &rhs).unwrap();

        assert_eq!(plan.inner_len, 25);
        assert_eq!(plan.row_len, 7);
        assert_eq!(plan.fast_axis, 1);
        assert_eq!(plan.a_fast_stride, 1);
        assert_eq!(plan.b_fast_stride, 0);
        assert_eq!(plan.a_row_stride, 0);
        assert_eq!(plan.b_row_stride, 1);
        assert_eq!(transposed_scalar_tile_kind(&plan), None);
    }

    #[test]
    fn test_broadcast_strides_for_axes_batched_outer() {
        let target_dims = [3usize, 5, 7, 11];
        let lhs_dims = [3usize, 5, 11];
        let lhs_strides = [3isize, 1, 15];
        let rhs_dims = [7usize, 11];
        let rhs_strides = [1isize, 7];

        let lhs =
            broadcast_strides_for_axes(&lhs_dims, &lhs_strides, &target_dims, &[0, 1, 3]).unwrap();
        let rhs =
            broadcast_strides_for_axes(&rhs_dims, &rhs_strides, &target_dims, &[2, 3]).unwrap();

        assert_eq!(&lhs[..], &[3, 1, 0, 15]);
        assert_eq!(&rhs[..], &[0, 0, 1, 7]);
    }

    #[test]
    fn test_broadcast_strides_for_axes_uses_zero_stride_for_size_one_source_dim() {
        let target_dims = [8usize, 4];
        let source_dims = [1usize, 4];
        let source_strides = [1isize, 1];

        let strides =
            broadcast_strides_for_axes(&source_dims, &source_strides, &target_dims, &[0, 1])
                .unwrap();

        assert_eq!(&strides[..], &[0, 1]);
    }

    #[test]
    fn test_transposed_scalar_tile_kind_detects_noncompact_rhs_scalar() {
        let dims = [5usize, 5, 7, 11];
        let dst = [1isize, 5, 25, 175];
        let lhs = [5isize, 1, 0, 25];
        let rhs = [0isize, 0, 1, 7];

        let plan = contiguous_mul_range_plan(&dims, &dst, &lhs, &rhs).unwrap();

        assert_eq!(
            transposed_scalar_tile_kind(&plan),
            Some(TransposedScalarTileKind::RhsScalar)
        );
    }

    #[test]
    fn test_contiguous_mul_outer_cursor_matches_linear_offsets() {
        let dims = [16usize, 16, 64, 64];
        let dst = [1isize, 16, 256, 16_384];
        let lhs = [16isize, 1, 0, 256];
        let rhs = [0isize, 0, 1, 64];
        let plan = contiguous_mul_range_plan(&dims, &dst, &lhs, &rhs).unwrap();
        let mut cursor = ContiguousMulOuterCursor::new(&dims, &lhs, &rhs, &plan, 13);
        let block_len = plan.inner_len * plan.row_len;

        for group in 13..80 {
            let index = group * block_len;
            assert_eq!(
                cursor.a_offset,
                strided_offset_for_contiguous_linear_index(&dims, &lhs, &plan.axis_order, index)
            );
            assert_eq!(
                cursor.b_offset,
                strided_offset_for_contiguous_linear_index(&dims, &rhs, &plan.axis_order, index)
            );
            cursor.advance();
        }
    }
}
