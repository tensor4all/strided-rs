//! Map operations on dynamic-rank strided views.
//!
//! These are the canonical view-based map functions, equivalent to Julia's `Base.map!`.

use crate::element_op::{ElementOp, ElementOpApply};
use crate::kernel::{
    build_plan_fused, ensure_same_shape, for_each_inner_block, is_contiguous, total_len,
};
use crate::strided_view::{StridedView, StridedViewMut};
use crate::Result;
use std::cmp::min;

/// Apply a function element-wise from source to destination.
///
/// The element operation `Op` is applied lazily when reading from `src`.
pub fn map_into<T: Copy + ElementOpApply, Op: ElementOp>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T, Op>,
    f: impl Fn(T) -> T,
) -> Result<()> {
    ensure_same_shape(dest.dims(), src.dims())?;

    let dst_ptr = dest.as_mut_ptr();
    let src_ptr = src.ptr();
    let dst_dims = dest.dims();
    let src_dims = src.dims();
    let dst_strides = dest.strides();
    let src_strides = src.strides();

    if is_contiguous(dst_dims, dst_strides) && is_contiguous(src_dims, src_strides) {
        let len = total_len(dst_dims);
        let mut dp = dst_ptr;
        let mut sp = src_ptr;
        for _ in 0..len {
            let val = Op::apply(unsafe { *sp });
            let out = f(val);
            unsafe {
                *dp = out;
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

    let (fused_dims, plan) = build_plan_fused(
        &dst_dims_v,
        &strides_list,
        Some(0),
        std::mem::size_of::<T>(),
    );

    for_each_inner_block(
        &fused_dims,
        &plan,
        &strides_list,
        |offsets, len, strides| {
            let mut dp = unsafe { dst_ptr.offset(offsets[0]) };
            let mut sp = unsafe { src_ptr.offset(offsets[1]) };
            let ds = strides[0];
            let ss = strides[1];
            for _ in 0..len {
                let val = Op::apply(unsafe { *sp });
                let out = f(val);
                unsafe {
                    *dp = out;
                    dp = dp.offset(ds);
                    sp = sp.offset(ss);
                }
            }
            Ok(())
        },
    )
}

/// Binary element-wise operation: `dest[i] = f(a[i], b[i])`.
pub fn zip_map2_into<T: Copy + ElementOpApply, OpA: ElementOp, OpB: ElementOp>(
    dest: &mut StridedViewMut<T>,
    a: &StridedView<T, OpA>,
    b: &StridedView<T, OpB>,
    f: impl Fn(T, T) -> T,
) -> Result<()> {
    ensure_same_shape(dest.dims(), a.dims())?;
    ensure_same_shape(dest.dims(), b.dims())?;

    // Clone metadata up front to avoid borrow conflicts with dest
    let dst_dims_v = dest.dims().to_vec();
    let dst_strides_v = dest.strides().to_vec();
    let a_strides_v = a.strides().to_vec();
    let b_strides_v = b.strides().to_vec();

    let dst_ptr = dest.as_mut_ptr();
    let a_ptr = a.ptr();
    let b_ptr = b.ptr();

    let a_dims_v = a.dims().to_vec();
    let b_dims_v = b.dims().to_vec();

    if is_contiguous(&dst_dims_v, &dst_strides_v)
        && is_contiguous(&a_dims_v, &a_strides_v)
        && is_contiguous(&b_dims_v, &b_strides_v)
    {
        let len = total_len(&dst_dims_v);
        let mut dp = dst_ptr;
        let mut ap = a_ptr;
        let mut bp = b_ptr;
        for _ in 0..len {
            let va = OpA::apply(unsafe { *ap });
            let vb = OpB::apply(unsafe { *bp });
            let out = f(va, vb);
            unsafe {
                *dp = out;
                dp = dp.add(1);
                ap = ap.add(1);
                bp = bp.add(1);
            }
        }
        return Ok(());
    }

    if zip_map2_2d_fast::<T, OpA, OpB>(dest, a, b, &f)? {
        return Ok(());
    }
    let strides_list: [&[isize]; 3] = [&dst_strides_v, &a_strides_v, &b_strides_v];

    let (fused_dims, plan) = build_plan_fused(
        &dst_dims_v,
        &strides_list,
        Some(0),
        std::mem::size_of::<T>(),
    );

    for_each_inner_block(
        &fused_dims,
        &plan,
        &strides_list,
        |offsets, len, strides| {
            let mut dp = unsafe { dst_ptr.offset(offsets[0]) };
            let mut ap = unsafe { a_ptr.offset(offsets[1]) };
            let mut bp = unsafe { b_ptr.offset(offsets[2]) };
            let ds = strides[0];
            let a_s = strides[1];
            let b_s = strides[2];
            for _ in 0..len {
                let va = OpA::apply(unsafe { *ap });
                let vb = OpB::apply(unsafe { *bp });
                let out = f(va, vb);
                unsafe {
                    *dp = out;
                    dp = dp.offset(ds);
                    ap = ap.offset(a_s);
                    bp = bp.offset(b_s);
                }
            }
            Ok(())
        },
    )
}

/// Ternary element-wise operation: `dest[i] = f(a[i], b[i], c[i])`.
pub fn zip_map3_into<T: Copy + ElementOpApply, OpA: ElementOp, OpB: ElementOp, OpC: ElementOp>(
    dest: &mut StridedViewMut<T>,
    a: &StridedView<T, OpA>,
    b: &StridedView<T, OpB>,
    c: &StridedView<T, OpC>,
    f: impl Fn(T, T, T) -> T,
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

    if is_contiguous(dst_dims, dst_strides)
        && is_contiguous(a.dims(), a.strides())
        && is_contiguous(b.dims(), b.strides())
        && is_contiguous(c.dims(), c.strides())
    {
        let len = total_len(dst_dims);
        let mut dp = dst_ptr;
        let mut ap = a_ptr;
        let mut bp = b_ptr;
        let mut cp = c_ptr;
        for _ in 0..len {
            let va = OpA::apply(unsafe { *ap });
            let vb = OpB::apply(unsafe { *bp });
            let vc = OpC::apply(unsafe { *cp });
            let out = f(va, vb, vc);
            unsafe {
                *dp = out;
                dp = dp.add(1);
                ap = ap.add(1);
                bp = bp.add(1);
                cp = cp.add(1);
            }
        }
        return Ok(());
    }

    let dst_strides_v = dst_strides.to_vec();
    let a_strides_v = a.strides().to_vec();
    let b_strides_v = b.strides().to_vec();
    let c_strides_v = c.strides().to_vec();
    let dst_dims_v = dst_dims.to_vec();
    let strides_list: [&[isize]; 4] = [&dst_strides_v, &a_strides_v, &b_strides_v, &c_strides_v];

    let (fused_dims, plan) = build_plan_fused(
        &dst_dims_v,
        &strides_list,
        Some(0),
        std::mem::size_of::<T>(),
    );

    for_each_inner_block(
        &fused_dims,
        &plan,
        &strides_list,
        |offsets, len, strides| {
            let mut dp = unsafe { dst_ptr.offset(offsets[0]) };
            let mut ap = unsafe { a_ptr.offset(offsets[1]) };
            let mut bp = unsafe { b_ptr.offset(offsets[2]) };
            let mut cp = unsafe { c_ptr.offset(offsets[3]) };
            let ds = strides[0];
            let a_s = strides[1];
            let b_s = strides[2];
            let c_s = strides[3];
            for _ in 0..len {
                let va = OpA::apply(unsafe { *ap });
                let vb = OpB::apply(unsafe { *bp });
                let vc = OpC::apply(unsafe { *cp });
                let out = f(va, vb, vc);
                unsafe {
                    *dp = out;
                    dp = dp.offset(ds);
                    ap = ap.offset(a_s);
                    bp = bp.offset(b_s);
                    cp = cp.offset(c_s);
                }
            }
            Ok(())
        },
    )
}

/// Quaternary element-wise operation: `dest[i] = f(a[i], b[i], c[i], e[i])`.
pub fn zip_map4_into<
    T: Copy + ElementOpApply,
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
    f: impl Fn(T, T, T, T) -> T,
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

    if is_contiguous(dst_dims, dst_strides)
        && is_contiguous(a.dims(), a.strides())
        && is_contiguous(b.dims(), b.strides())
        && is_contiguous(c.dims(), c.strides())
        && is_contiguous(e.dims(), e.strides())
    {
        let len = total_len(dst_dims);
        let mut dp = dst_ptr;
        let mut ap = a_ptr;
        let mut bp = b_ptr;
        let mut cp = c_ptr;
        let mut ep = e_ptr;
        for _ in 0..len {
            let va = OpA::apply(unsafe { *ap });
            let vb = OpB::apply(unsafe { *bp });
            let vc = OpC::apply(unsafe { *cp });
            let ve = OpE::apply(unsafe { *ep });
            let out = f(va, vb, vc, ve);
            unsafe {
                *dp = out;
                dp = dp.add(1);
                ap = ap.add(1);
                bp = bp.add(1);
                cp = cp.add(1);
                ep = ep.add(1);
            }
        }
        return Ok(());
    }

    let dst_strides_v = dst_strides.to_vec();
    let a_strides_v = a.strides().to_vec();
    let b_strides_v = b.strides().to_vec();
    let c_strides_v = c.strides().to_vec();
    let e_strides_v = e.strides().to_vec();
    let dst_dims_v = dst_dims.to_vec();
    let strides_list: [&[isize]; 5] = [
        &dst_strides_v,
        &a_strides_v,
        &b_strides_v,
        &c_strides_v,
        &e_strides_v,
    ];

    let (fused_dims, plan) = build_plan_fused(
        &dst_dims_v,
        &strides_list,
        Some(0),
        std::mem::size_of::<T>(),
    );

    for_each_inner_block(
        &fused_dims,
        &plan,
        &strides_list,
        |offsets, len, strides| {
            let mut dp = unsafe { dst_ptr.offset(offsets[0]) };
            let mut ap = unsafe { a_ptr.offset(offsets[1]) };
            let mut bp = unsafe { b_ptr.offset(offsets[2]) };
            let mut cp = unsafe { c_ptr.offset(offsets[3]) };
            let mut ep = unsafe { e_ptr.offset(offsets[4]) };
            let ds = strides[0];
            let a_s = strides[1];
            let b_s = strides[2];
            let c_s = strides[3];
            let e_s = strides[4];
            for _ in 0..len {
                let va = OpA::apply(unsafe { *ap });
                let vb = OpB::apply(unsafe { *bp });
                let vc = OpC::apply(unsafe { *cp });
                let ve = OpE::apply(unsafe { *ep });
                let out = f(va, vb, vc, ve);
                unsafe {
                    *dp = out;
                    dp = dp.offset(ds);
                    ap = ap.offset(a_s);
                    bp = bp.offset(b_s);
                    cp = cp.offset(c_s);
                    ep = ep.offset(e_s);
                }
            }
            Ok(())
        },
    )
}

// ============================================================================
// 2D fast paths
// ============================================================================

fn zip_map2_2d_tiled_transpose<T: Copy + ElementOpApply, OpA: ElementOp, OpB: ElementOp>(
    dest: &mut StridedViewMut<T>,
    a: &StridedView<T, OpA>,
    b: &StridedView<T, OpB>,
    f: &impl Fn(T, T) -> T,
) -> Result<bool> {
    let n = dest.dims()[0];
    if n != dest.dims()[1] {
        return Ok(false);
    }

    let elem_size = std::mem::size_of::<T>();
    const BLOCK_MEMORY_SIZE: usize = 1 << 15;
    let tile_size = if elem_size > 0 {
        let max_elems = BLOCK_MEMORY_SIZE / elem_size / 2;
        (max_elems as f64).sqrt() as usize
    } else {
        32
    }
    .clamp(16, 64);

    let d_ptr = dest.as_mut_ptr();
    let a_ptr = a.ptr();
    let b_ptr = b.ptr();

    let d_row_stride = dest.strides()[0];
    let a_row_stride = a.strides()[0];
    let b_col_stride = b.strides()[1];

    for i_tile in (0..n).step_by(tile_size) {
        let i_end = min(i_tile + tile_size, n);
        for j_tile in (0..n).step_by(tile_size) {
            let j_end = min(j_tile + tile_size, n);

            for i in i_tile..i_end {
                let d_row_ptr = unsafe { d_ptr.offset((i as isize) * d_row_stride) };
                let a_row_ptr = unsafe { a_ptr.offset((i as isize) * a_row_stride) };

                for j in j_tile..j_end {
                    let va = OpA::apply(unsafe { *a_row_ptr.offset(j as isize) });
                    let vb = OpB::apply(unsafe {
                        *b_ptr.offset((i as isize) + (j as isize) * b_col_stride)
                    });
                    let result = f(va, vb);
                    unsafe {
                        *d_row_ptr.offset(j as isize) = result;
                    }
                }
            }
        }
    }

    Ok(true)
}

fn zip_map2_2d_fast<T: Copy + ElementOpApply, OpA: ElementOp, OpB: ElementOp>(
    dest: &mut StridedViewMut<T>,
    a: &StridedView<T, OpA>,
    b: &StridedView<T, OpB>,
    f: &impl Fn(T, T) -> T,
) -> Result<bool> {
    if dest.ndim() != 2 {
        return Ok(false);
    }

    let rows = dest.dims()[0];
    let cols = dest.dims()[1];
    if rows == 0 || cols == 0 {
        return Ok(true);
    }

    let d_row = dest.strides()[0];
    let d_col = dest.strides()[1];
    let a_row = a.strides()[0];
    let a_col = a.strides()[1];
    let b_row = b.strides()[0];
    let b_col = b.strides()[1];

    // Detect transpose pattern
    if rows == cols && rows >= 32 {
        let a_is_rowmajor = a_col.unsigned_abs() == 1 && a_row.unsigned_abs() == cols;
        let a_is_colmajor = a_row.unsigned_abs() == 1 && a_col.unsigned_abs() == rows;
        let b_is_rowmajor = b_col.unsigned_abs() == 1 && b_row.unsigned_abs() == cols;
        let b_is_colmajor = b_row.unsigned_abs() == 1 && b_col.unsigned_abs() == rows;
        let d_is_rowmajor = d_col.unsigned_abs() == 1 && d_row.unsigned_abs() == cols;

        if d_is_rowmajor && a_is_rowmajor && b_is_colmajor {
            if let Ok(true) = zip_map2_2d_tiled_transpose(dest, a, b, f) {
                return Ok(true);
            }
        }
        if d_is_rowmajor && a_is_colmajor && b_is_rowmajor {
            if let Ok(true) = zip_map2_2d_tiled_transpose(dest, b, a, &|va, vb| f(vb, va)) {
                return Ok(true);
            }
        }
    }

    let d_col_contig = d_col.unsigned_abs() == 1;
    let d_row_contig = d_row.unsigned_abs() == 1;

    let prefer_col = if d_col_contig {
        true
    } else if d_row_contig {
        false
    } else {
        let score_col = 2 * d_col.unsigned_abs() + a_col.unsigned_abs() + b_col.unsigned_abs();
        let score_row = 2 * d_row.unsigned_abs() + a_row.unsigned_abs() + b_row.unsigned_abs();
        score_col <= score_row
    };

    let dst_ptr = dest.as_mut_ptr();
    let a_ptr = a.ptr();
    let b_ptr = b.ptr();

    if prefer_col {
        for i in 0..rows {
            let d_base = offset2d(i, 0, d_row, d_col)?;
            let a_base = offset2d(i, 0, a_row, a_col)?;
            let b_base = offset2d(i, 0, b_row, b_col)?;
            let mut dp = unsafe { dst_ptr.offset(d_base) };
            let mut ap = unsafe { a_ptr.offset(a_base) };
            let mut bp = unsafe { b_ptr.offset(b_base) };
            for _ in 0..cols {
                let va = OpA::apply(unsafe { *ap });
                let vb = OpB::apply(unsafe { *bp });
                let out = f(va, vb);
                unsafe {
                    *dp = out;
                    dp = dp.offset(d_col);
                    ap = ap.offset(a_col);
                    bp = bp.offset(b_col);
                }
            }
        }
        return Ok(true);
    }

    for j in 0..cols {
        let d_base = offset2d(0, j, d_row, d_col)?;
        let a_base = offset2d(0, j, a_row, a_col)?;
        let b_base = offset2d(0, j, b_row, b_col)?;
        let mut dp = unsafe { dst_ptr.offset(d_base) };
        let mut ap = unsafe { a_ptr.offset(a_base) };
        let mut bp = unsafe { b_ptr.offset(b_base) };
        for _ in 0..rows {
            let va = OpA::apply(unsafe { *ap });
            let vb = OpB::apply(unsafe { *bp });
            let out = f(va, vb);
            unsafe {
                *dp = out;
                dp = dp.offset(d_row);
                ap = ap.offset(a_row);
                bp = bp.offset(b_row);
            }
        }
    }
    Ok(true)
}

fn offset2d(i: usize, j: usize, row_stride: isize, col_stride: isize) -> Result<isize> {
    let i = isize::try_from(i).map_err(|_| crate::StridedError::OffsetOverflow)?;
    let j = isize::try_from(j).map_err(|_| crate::StridedError::OffsetOverflow)?;
    let i_off = row_stride
        .checked_mul(i)
        .ok_or(crate::StridedError::OffsetOverflow)?;
    let j_off = col_stride
        .checked_mul(j)
        .ok_or(crate::StridedError::OffsetOverflow)?;
    i_off
        .checked_add(j_off)
        .ok_or(crate::StridedError::OffsetOverflow)
}
