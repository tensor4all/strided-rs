//! High-level operations on dynamic-rank strided views.

use crate::element_op::{ElementOp, ElementOpApply};
use crate::kernel::{
    build_plan, ensure_same_shape, for_each_inner_block, is_contiguous, total_len,
};
use crate::map_view::{map_into, zip_map2_into};
use crate::reduce_view::reduce;
use crate::strided_view::{StridedView, StridedViewMut};
use crate::{Result, StridedError};
use num_traits::Zero;
use std::ops::{Add, Mul};

/// Copy elements from source to destination: `dest[i] = src[i]`.
pub fn copy_into<T: Copy + ElementOpApply, Op: ElementOp>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T, Op>,
) -> Result<()> {
    ensure_same_shape(dest.dims(), src.dims())?;

    let dst_ptr = dest.as_mut_ptr();
    let src_ptr = src.ptr();
    let dst_dims = dest.dims();
    let dst_strides = dest.strides();
    let src_strides = src.strides();

    if is_contiguous(dst_dims, dst_strides) && is_contiguous(src.dims(), src_strides) {
        let len = total_len(dst_dims);
        let mut dp = dst_ptr;
        let mut sp = src_ptr;
        for _ in 0..len {
            unsafe {
                *dp = Op::apply(*sp);
                dp = dp.add(1);
                sp = sp.add(1);
            }
        }
        return Ok(());
    }

    map_into(dest, src, |x| x)
}

/// Element-wise addition: `dest[i] += src[i]`.
pub fn add<T: Copy + ElementOpApply + Add<Output = T>, Op: ElementOp>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T, Op>,
) -> Result<()> {
    ensure_same_shape(dest.dims(), src.dims())?;

    let dst_ptr = dest.as_mut_ptr();
    let src_ptr = src.ptr();
    let dst_dims = dest.dims();
    let dst_strides = dest.strides();
    let src_strides = src.strides();

    if is_contiguous(dst_dims, dst_strides) && is_contiguous(src.dims(), src_strides) {
        let len = total_len(dst_dims);
        let mut dp = dst_ptr;
        let mut sp = src_ptr;
        for _ in 0..len {
            unsafe {
                *dp = *dp + Op::apply(*sp);
                dp = dp.add(1);
                sp = sp.add(1);
            }
        }
        return Ok(());
    }

    let dst_strides_v = dst_strides.to_vec();
    let src_strides_v = src_strides.to_vec();
    let dst_dims_v = dst_dims.to_vec();
    let strides_list: [&[isize]; 2] = [&dst_strides_v, &src_strides_v];

    let plan = build_plan(
        &dst_dims_v,
        &strides_list,
        Some(0),
        std::mem::size_of::<T>(),
    );

    for_each_inner_block(
        &dst_dims_v,
        &plan,
        &strides_list,
        |offsets, len, strides| {
            let mut dp = unsafe { dst_ptr.offset(offsets[0]) };
            let mut sp = unsafe { src_ptr.offset(offsets[1]) };
            let ds = strides[0];
            let ss = strides[1];
            for _ in 0..len {
                unsafe {
                    *dp = *dp + Op::apply(*sp);
                    dp = dp.offset(ds);
                    sp = sp.offset(ss);
                }
            }
            Ok(())
        },
    )
}

/// Element-wise multiplication: `dest[i] *= src[i]`.
pub fn mul<T: Copy + ElementOpApply + Mul<Output = T>, Op: ElementOp>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T, Op>,
) -> Result<()> {
    ensure_same_shape(dest.dims(), src.dims())?;

    let dst_ptr = dest.as_mut_ptr();
    let src_ptr = src.ptr();
    let dst_dims = dest.dims();
    let dst_strides = dest.strides();
    let src_strides = src.strides();

    if is_contiguous(dst_dims, dst_strides) && is_contiguous(src.dims(), src_strides) {
        let len = total_len(dst_dims);
        let mut dp = dst_ptr;
        let mut sp = src_ptr;
        for _ in 0..len {
            unsafe {
                *dp = *dp * Op::apply(*sp);
                dp = dp.add(1);
                sp = sp.add(1);
            }
        }
        return Ok(());
    }

    let dst_strides_v = dst_strides.to_vec();
    let src_strides_v = src_strides.to_vec();
    let dst_dims_v = dst_dims.to_vec();
    let strides_list: [&[isize]; 2] = [&dst_strides_v, &src_strides_v];

    let plan = build_plan(
        &dst_dims_v,
        &strides_list,
        Some(0),
        std::mem::size_of::<T>(),
    );

    for_each_inner_block(
        &dst_dims_v,
        &plan,
        &strides_list,
        |offsets, len, strides| {
            let mut dp = unsafe { dst_ptr.offset(offsets[0]) };
            let mut sp = unsafe { src_ptr.offset(offsets[1]) };
            let ds = strides[0];
            let ss = strides[1];
            for _ in 0..len {
                unsafe {
                    *dp = *dp * Op::apply(*sp);
                    dp = dp.offset(ds);
                    sp = sp.offset(ss);
                }
            }
            Ok(())
        },
    )
}

/// AXPY: `dest[i] = alpha * src[i] + dest[i]`.
pub fn axpy<T: Copy + ElementOpApply + Mul<Output = T> + Add<Output = T>, Op: ElementOp>(
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

    if is_contiguous(dst_dims, dst_strides) && is_contiguous(src.dims(), src_strides) {
        let len = total_len(dst_dims);
        let mut dp = dst_ptr;
        let mut sp = src_ptr;
        for _ in 0..len {
            unsafe {
                *dp = alpha * Op::apply(*sp) + *dp;
                dp = dp.add(1);
                sp = sp.add(1);
            }
        }
        return Ok(());
    }

    let dst_strides_v = dst_strides.to_vec();
    let src_strides_v = src_strides.to_vec();
    let dst_dims_v = dst_dims.to_vec();
    let strides_list: [&[isize]; 2] = [&dst_strides_v, &src_strides_v];

    let plan = build_plan(
        &dst_dims_v,
        &strides_list,
        Some(0),
        std::mem::size_of::<T>(),
    );

    for_each_inner_block(
        &dst_dims_v,
        &plan,
        &strides_list,
        |offsets, len, strides| {
            let mut dp = unsafe { dst_ptr.offset(offsets[0]) };
            let mut sp = unsafe { src_ptr.offset(offsets[1]) };
            let ds = strides[0];
            let ss = strides[1];
            for _ in 0..len {
                unsafe {
                    *dp = alpha * Op::apply(*sp) + *dp;
                    dp = dp.offset(ds);
                    sp = sp.offset(ss);
                }
            }
            Ok(())
        },
    )
}

/// Fused multiply-add: `dest[i] += a[i] * b[i]`.
pub fn fma<T: Copy + ElementOpApply + Mul<Output = T> + Add<Output = T>>(
    dest: &mut StridedViewMut<T>,
    a: &StridedView<T>,
    b: &StridedView<T>,
) -> Result<()> {
    ensure_same_shape(dest.dims(), a.dims())?;
    ensure_same_shape(dest.dims(), b.dims())?;

    let dst_ptr = dest.as_mut_ptr();
    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let dst_dims = dest.dims().to_vec();
    let dst_strides = dest.strides().to_vec();
    let a_strides = a.strides().to_vec();
    let b_strides = b.strides().to_vec();

    if is_contiguous(&dst_dims, &dst_strides)
        && is_contiguous(a.dims(), &a_strides)
        && is_contiguous(b.dims(), &b_strides)
    {
        let len = total_len(&dst_dims);
        let mut dp = dst_ptr;
        let mut ap = a_ptr;
        let mut bp = b_ptr;
        for _ in 0..len {
            unsafe {
                *dp = *dp + *ap * *bp;
                dp = dp.add(1);
                ap = ap.add(1);
                bp = bp.add(1);
            }
        }
        return Ok(());
    }

    let strides_list: [&[isize]; 3] = [&dst_strides, &a_strides, &b_strides];
    let plan = build_plan(&dst_dims, &strides_list, Some(0), std::mem::size_of::<T>());

    for_each_inner_block(&dst_dims, &plan, &strides_list, |offsets, len, strides| {
        let mut dp = unsafe { dst_ptr.offset(offsets[0]) };
        let mut ap = unsafe { a_ptr.offset(offsets[1]) };
        let mut bp = unsafe { b_ptr.offset(offsets[2]) };
        let ds = strides[0];
        let a_s = strides[1];
        let b_s = strides[2];
        for _ in 0..len {
            unsafe {
                *dp = *dp + *ap * *bp;
                dp = dp.offset(ds);
                ap = ap.offset(a_s);
                bp = bp.offset(b_s);
            }
        }
        Ok(())
    })
}

/// Sum all elements: `sum(src)`.
pub fn sum<T: Copy + ElementOpApply + Zero + Add<Output = T>, Op: ElementOp>(
    src: &StridedView<T, Op>,
) -> Result<T> {
    reduce(src, |x| x, |a, b| a + b, T::zero())
}

/// Dot product: `sum(a[i] * b[i])`.
pub fn dot<
    T: Copy + ElementOpApply + Zero + Mul<Output = T> + Add<Output = T>,
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

    if is_contiguous(a_dims, a_strides) && is_contiguous(b.dims(), b_strides) {
        let len = total_len(a_dims);
        let mut ap = a_ptr;
        let mut bp = b_ptr;
        let mut acc = T::zero();
        for _ in 0..len {
            unsafe {
                acc = acc + OpA::apply(*ap) * OpB::apply(*bp);
                ap = ap.add(1);
                bp = bp.add(1);
            }
        }
        return Ok(acc);
    }

    let a_strides_v = a_strides.to_vec();
    let b_strides_v = b_strides.to_vec();
    let a_dims_v = a_dims.to_vec();
    let strides_list: [&[isize]; 2] = [&a_strides_v, &b_strides_v];

    let plan = build_plan(&a_dims_v, &strides_list, None, std::mem::size_of::<T>());

    let mut acc = Some(T::zero());
    for_each_inner_block(&a_dims_v, &plan, &strides_list, |offsets, len, strides| {
        let mut ap = unsafe { a_ptr.offset(offsets[0]) };
        let mut bp = unsafe { b_ptr.offset(offsets[1]) };
        let a_s = strides[0];
        let b_s = strides[1];
        let mut local = acc.take().ok_or(StridedError::OffsetOverflow)?;
        for _ in 0..len {
            unsafe {
                local = local + OpA::apply(*ap) * OpB::apply(*bp);
                ap = ap.offset(a_s);
                bp = bp.offset(b_s);
            }
        }
        acc = Some(local);
        Ok(())
    })?;

    acc.ok_or(StridedError::OffsetOverflow)
}

/// Symmetrize a square matrix: `dest = (src + src^T) / 2`.
pub fn symmetrize_into<T>(dest: &mut StridedViewMut<T>, src: &StridedView<T>) -> Result<()>
where
    T: Copy
        + ElementOpApply
        + Add<Output = T>
        + Mul<Output = T>
        + num_traits::FromPrimitive
        + std::ops::Div<Output = T>,
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
        + std::ops::Div<Output = T>,
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
pub fn copy_scale<T: Copy + ElementOpApply + Mul<Output = T>, Op: ElementOp>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T, Op>,
    scale: T,
) -> Result<()> {
    map_into(dest, src, |x| scale * x)
}

/// Copy with complex conjugation: `dest[i] = conj(src[i])`.
pub fn copy_conj<T: Copy + ElementOpApply>(
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
    T: Copy + ElementOpApply + Mul<Output = T>,
{
    if src.ndim() != 2 || dest.ndim() != 2 {
        return Err(StridedError::RankMismatch(src.ndim(), 2));
    }
    let src_t = src.transpose_2d()?;
    map_into(dest, &src_t, |x| scale * x)
}
