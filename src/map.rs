use crate::fuse::fuse_dims;
use crate::kernel::{
    build_plan, ensure_same_shape, for_each_inner_block, is_contiguous, total_len, StridedView,
    StridedViewMut,
};
use crate::{Result, StridedError};
use mdarray::{Layout, Shape, Slice};

/// Apply dimension fusion to simplify iteration.
///
/// Fuses contiguous dimensions across all arrays, reducing the number of loop levels.
/// Returns the fused dimensions (strides remain unchanged).
#[inline]
fn apply_fusion(dims: &[usize], strides_list: &[&[isize]]) -> Vec<usize> {
    if dims.len() <= 1 {
        return dims.to_vec();
    }
    fuse_dims(dims, strides_list)
}

pub fn map_into<T, SD, SS, LD, LS, F>(
    dest: &mut Slice<T, SD, LD>,
    src: &Slice<T, SS, LS>,
    f: F,
) -> Result<()>
where
    SD: Shape,
    SS: Shape,
    LD: Layout,
    LS: Layout,
    F: Fn(&T) -> T,
{
    let dst_view = StridedViewMut::from_slice(dest)?;
    let src_view = StridedView::from_slice(src)?;
    ensure_same_shape(&dst_view.dims, &src_view.dims)?;

    if is_contiguous(&dst_view.dims, &dst_view.strides)
        && is_contiguous(&src_view.dims, &src_view.strides)
    {
        let len = total_len(&dst_view.dims);
        let mut dst_ptr = dst_view.ptr;
        let mut src_ptr = src_view.ptr;
        for _ in 0..len {
            let out = f(unsafe { &*src_ptr });
            unsafe {
                *dst_ptr = out;
                dst_ptr = dst_ptr.add(1);
                src_ptr = src_ptr.add(1);
            }
        }
        return Ok(());
    }

    let strides_list = [&dst_view.strides[..], &src_view.strides[..]];

    // Apply dimension fusion to reduce loop levels
    let fused_dims = apply_fusion(&dst_view.dims, &strides_list);

    let plan = build_plan(
        &fused_dims,
        &strides_list,
        Some(0),
        std::mem::size_of::<T>(),
    );

    for_each_inner_block(
        &fused_dims,
        &plan,
        &strides_list,
        |offsets, len, strides| {
            let mut dst_ptr = unsafe { dst_view.ptr.offset(offsets[0]) };
            let mut src_ptr = unsafe { src_view.ptr.offset(offsets[1]) };
            let dst_stride = strides[0];
            let src_stride = strides[1];
            for _ in 0..len {
                let out = f(unsafe { &*src_ptr });
                unsafe {
                    *dst_ptr = out;
                    dst_ptr = dst_ptr.offset(dst_stride);
                    src_ptr = src_ptr.offset(src_stride);
                }
            }
            Ok(())
        },
    )
}

pub fn zip_map2_into<T, SD, SA, SB, LD, LA, LB, F>(
    dest: &mut Slice<T, SD, LD>,
    a: &Slice<T, SA, LA>,
    b: &Slice<T, SB, LB>,
    f: F,
) -> Result<()>
where
    SD: Shape,
    SA: Shape,
    SB: Shape,
    LD: Layout,
    LA: Layout,
    LB: Layout,
    F: Fn(&T, &T) -> T,
{
    let dst_view = StridedViewMut::from_slice(dest)?;
    let a_view = StridedView::from_slice(a)?;
    let b_view = StridedView::from_slice(b)?;
    ensure_same_shape(&dst_view.dims, &a_view.dims)?;
    ensure_same_shape(&dst_view.dims, &b_view.dims)?;

    if is_contiguous(&dst_view.dims, &dst_view.strides)
        && is_contiguous(&a_view.dims, &a_view.strides)
        && is_contiguous(&b_view.dims, &b_view.strides)
    {
        let len = total_len(&dst_view.dims);
        let mut dst_ptr = dst_view.ptr;
        let mut a_ptr = a_view.ptr;
        let mut b_ptr = b_view.ptr;
        for _ in 0..len {
            let out = f(unsafe { &*a_ptr }, unsafe { &*b_ptr });
            unsafe {
                *dst_ptr = out;
                dst_ptr = dst_ptr.add(1);
                a_ptr = a_ptr.add(1);
                b_ptr = b_ptr.add(1);
            }
        }
        return Ok(());
    }

    if zip_map2_2d_fast(&dst_view, &a_view, &b_view, &f)? {
        return Ok(());
    }

    let strides_list = [
        &dst_view.strides[..],
        &a_view.strides[..],
        &b_view.strides[..],
    ];

    // Apply dimension fusion to reduce loop levels
    let fused_dims = apply_fusion(&dst_view.dims, &strides_list);

    let plan = build_plan(
        &fused_dims,
        &strides_list,
        Some(0),
        std::mem::size_of::<T>(),
    );

    for_each_inner_block(
        &fused_dims,
        &plan,
        &strides_list,
        |offsets, len, strides| {
            let mut dst_ptr = unsafe { dst_view.ptr.offset(offsets[0]) };
            let mut a_ptr = unsafe { a_view.ptr.offset(offsets[1]) };
            let mut b_ptr = unsafe { b_view.ptr.offset(offsets[2]) };
            let dst_stride = strides[0];
            let a_stride = strides[1];
            let b_stride = strides[2];
            for _ in 0..len {
                let out = f(unsafe { &*a_ptr }, unsafe { &*b_ptr });
                unsafe {
                    *dst_ptr = out;
                    dst_ptr = dst_ptr.offset(dst_stride);
                    a_ptr = a_ptr.offset(a_stride);
                    b_ptr = b_ptr.offset(b_stride);
                }
            }
            Ok(())
        },
    )
}

fn zip_map2_2d_fast<T, F>(
    dst_view: &StridedViewMut<T>,
    a_view: &StridedView<T>,
    b_view: &StridedView<T>,
    f: &F,
) -> Result<bool>
where
    F: Fn(&T, &T) -> T,
{
    if dst_view.dims.len() != 2 {
        return Ok(false);
    }

    let rows = dst_view.dims[0];
    let cols = dst_view.dims[1];
    if rows == 0 || cols == 0 {
        return Ok(true);
    }

    let d_row = dst_view.strides[0];
    let d_col = dst_view.strides[1];
    let a_row = a_view.strides[0];
    let a_col = a_view.strides[1];
    let b_row = b_view.strides[0];
    let b_col = b_view.strides[1];

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

    if prefer_col {
        for i in 0..rows {
            let d_base = offset2d(i, 0, d_row, d_col)?;
            let a_base = offset2d(i, 0, a_row, a_col)?;
            let b_base = offset2d(i, 0, b_row, b_col)?;
            let mut d_ptr = unsafe { dst_view.ptr.offset(d_base) };
            let mut a_ptr = unsafe { a_view.ptr.offset(a_base) };
            let mut b_ptr = unsafe { b_view.ptr.offset(b_base) };
            for _ in 0..cols {
                let out = f(unsafe { &*a_ptr }, unsafe { &*b_ptr });
                unsafe {
                    *d_ptr = out;
                    d_ptr = d_ptr.offset(d_col);
                    a_ptr = a_ptr.offset(a_col);
                    b_ptr = b_ptr.offset(b_col);
                }
            }
        }
        return Ok(true);
    }

    for j in 0..cols {
        let d_base = offset2d(0, j, d_row, d_col)?;
        let a_base = offset2d(0, j, a_row, a_col)?;
        let b_base = offset2d(0, j, b_row, b_col)?;
        let mut d_ptr = unsafe { dst_view.ptr.offset(d_base) };
        let mut a_ptr = unsafe { a_view.ptr.offset(a_base) };
        let mut b_ptr = unsafe { b_view.ptr.offset(b_base) };
        for _ in 0..rows {
            let out = f(unsafe { &*a_ptr }, unsafe { &*b_ptr });
            unsafe {
                *d_ptr = out;
                d_ptr = d_ptr.offset(d_row);
                a_ptr = a_ptr.offset(a_row);
                b_ptr = b_ptr.offset(b_row);
            }
        }
    }
    Ok(true)
}

fn offset2d(i: usize, j: usize, row_stride: isize, col_stride: isize) -> Result<isize> {
    let i = isize::try_from(i).map_err(|_| StridedError::OffsetOverflow)?;
    let j = isize::try_from(j).map_err(|_| StridedError::OffsetOverflow)?;
    let i_off = row_stride
        .checked_mul(i)
        .ok_or(StridedError::OffsetOverflow)?;
    let j_off = col_stride
        .checked_mul(j)
        .ok_or(StridedError::OffsetOverflow)?;
    i_off.checked_add(j_off).ok_or(StridedError::OffsetOverflow)
}

pub fn zip_map3_into<T, SD, SA, SB, SC, LD, LA, LB, LC, F>(
    dest: &mut Slice<T, SD, LD>,
    a: &Slice<T, SA, LA>,
    b: &Slice<T, SB, LB>,
    c: &Slice<T, SC, LC>,
    f: F,
) -> Result<()>
where
    SD: Shape,
    SA: Shape,
    SB: Shape,
    SC: Shape,
    LD: Layout,
    LA: Layout,
    LB: Layout,
    LC: Layout,
    F: Fn(&T, &T, &T) -> T,
{
    let dst_view = StridedViewMut::from_slice(dest)?;
    let a_view = StridedView::from_slice(a)?;
    let b_view = StridedView::from_slice(b)?;
    let c_view = StridedView::from_slice(c)?;
    ensure_same_shape(&dst_view.dims, &a_view.dims)?;
    ensure_same_shape(&dst_view.dims, &b_view.dims)?;
    ensure_same_shape(&dst_view.dims, &c_view.dims)?;

    if is_contiguous(&dst_view.dims, &dst_view.strides)
        && is_contiguous(&a_view.dims, &a_view.strides)
        && is_contiguous(&b_view.dims, &b_view.strides)
        && is_contiguous(&c_view.dims, &c_view.strides)
    {
        let len = total_len(&dst_view.dims);
        let mut dst_ptr = dst_view.ptr;
        let mut a_ptr = a_view.ptr;
        let mut b_ptr = b_view.ptr;
        let mut c_ptr = c_view.ptr;
        for _ in 0..len {
            let out = f(unsafe { &*a_ptr }, unsafe { &*b_ptr }, unsafe { &*c_ptr });
            unsafe {
                *dst_ptr = out;
                dst_ptr = dst_ptr.add(1);
                a_ptr = a_ptr.add(1);
                b_ptr = b_ptr.add(1);
                c_ptr = c_ptr.add(1);
            }
        }
        return Ok(());
    }

    let strides_list = [
        &dst_view.strides[..],
        &a_view.strides[..],
        &b_view.strides[..],
        &c_view.strides[..],
    ];

    // Apply dimension fusion to reduce loop levels
    let fused_dims = apply_fusion(&dst_view.dims, &strides_list);

    let plan = build_plan(
        &fused_dims,
        &strides_list,
        Some(0),
        std::mem::size_of::<T>(),
    );

    for_each_inner_block(
        &fused_dims,
        &plan,
        &strides_list,
        |offsets, len, strides| {
            let mut dst_ptr = unsafe { dst_view.ptr.offset(offsets[0]) };
            let mut a_ptr = unsafe { a_view.ptr.offset(offsets[1]) };
            let mut b_ptr = unsafe { b_view.ptr.offset(offsets[2]) };
            let mut c_ptr = unsafe { c_view.ptr.offset(offsets[3]) };
            let dst_stride = strides[0];
            let a_stride = strides[1];
            let b_stride = strides[2];
            let c_stride = strides[3];
            for _ in 0..len {
                let out = f(unsafe { &*a_ptr }, unsafe { &*b_ptr }, unsafe { &*c_ptr });
                unsafe {
                    *dst_ptr = out;
                    dst_ptr = dst_ptr.offset(dst_stride);
                    a_ptr = a_ptr.offset(a_stride);
                    b_ptr = b_ptr.offset(b_stride);
                    c_ptr = c_ptr.offset(c_stride);
                }
            }
            Ok(())
        },
    )
}

pub fn zip_map4_into<T, SD, SA, SB, SC, SE, LD, LA, LB, LC, LE, F>(
    dest: &mut Slice<T, SD, LD>,
    a: &Slice<T, SA, LA>,
    b: &Slice<T, SB, LB>,
    c: &Slice<T, SC, LC>,
    e: &Slice<T, SE, LE>,
    f: F,
) -> Result<()>
where
    SD: Shape,
    SA: Shape,
    SB: Shape,
    SC: Shape,
    SE: Shape,
    LD: Layout,
    LA: Layout,
    LB: Layout,
    LC: Layout,
    LE: Layout,
    F: Fn(&T, &T, &T, &T) -> T,
{
    let dst_view = StridedViewMut::from_slice(dest)?;
    let a_view = StridedView::from_slice(a)?;
    let b_view = StridedView::from_slice(b)?;
    let c_view = StridedView::from_slice(c)?;
    let e_view = StridedView::from_slice(e)?;
    ensure_same_shape(&dst_view.dims, &a_view.dims)?;
    ensure_same_shape(&dst_view.dims, &b_view.dims)?;
    ensure_same_shape(&dst_view.dims, &c_view.dims)?;
    ensure_same_shape(&dst_view.dims, &e_view.dims)?;

    if is_contiguous(&dst_view.dims, &dst_view.strides)
        && is_contiguous(&a_view.dims, &a_view.strides)
        && is_contiguous(&b_view.dims, &b_view.strides)
        && is_contiguous(&c_view.dims, &c_view.strides)
        && is_contiguous(&e_view.dims, &e_view.strides)
    {
        let len = total_len(&dst_view.dims);
        let mut dst_ptr = dst_view.ptr;
        let mut a_ptr = a_view.ptr;
        let mut b_ptr = b_view.ptr;
        let mut c_ptr = c_view.ptr;
        let mut e_ptr = e_view.ptr;
        for _ in 0..len {
            let out = f(
                unsafe { &*a_ptr },
                unsafe { &*b_ptr },
                unsafe { &*c_ptr },
                unsafe { &*e_ptr },
            );
            unsafe {
                *dst_ptr = out;
                dst_ptr = dst_ptr.add(1);
                a_ptr = a_ptr.add(1);
                b_ptr = b_ptr.add(1);
                c_ptr = c_ptr.add(1);
                e_ptr = e_ptr.add(1);
            }
        }
        return Ok(());
    }

    if dst_view.dims.len() == 4 && is_contiguous(&dst_view.dims, &dst_view.strides) {
        zip_map4_4d_contig_dst(&dst_view, &a_view, &b_view, &c_view, &e_view, &f)?;
        return Ok(());
    }

    let strides_list = [
        &dst_view.strides[..],
        &a_view.strides[..],
        &b_view.strides[..],
        &c_view.strides[..],
        &e_view.strides[..],
    ];

    // Apply dimension fusion to reduce loop levels
    let fused_dims = apply_fusion(&dst_view.dims, &strides_list);

    let plan = build_plan(
        &fused_dims,
        &strides_list,
        Some(0),
        std::mem::size_of::<T>(),
    );

    for_each_inner_block(
        &fused_dims,
        &plan,
        &strides_list,
        |offsets, len, strides| {
            let mut dst_ptr = unsafe { dst_view.ptr.offset(offsets[0]) };
            let mut a_ptr = unsafe { a_view.ptr.offset(offsets[1]) };
            let mut b_ptr = unsafe { b_view.ptr.offset(offsets[2]) };
            let mut c_ptr = unsafe { c_view.ptr.offset(offsets[3]) };
            let mut e_ptr = unsafe { e_view.ptr.offset(offsets[4]) };
            let dst_stride = strides[0];
            let a_stride = strides[1];
            let b_stride = strides[2];
            let c_stride = strides[3];
            let e_stride = strides[4];
            for _ in 0..len {
                let out = f(
                    unsafe { &*a_ptr },
                    unsafe { &*b_ptr },
                    unsafe { &*c_ptr },
                    unsafe { &*e_ptr },
                );
                unsafe {
                    *dst_ptr = out;
                    dst_ptr = dst_ptr.offset(dst_stride);
                    a_ptr = a_ptr.offset(a_stride);
                    b_ptr = b_ptr.offset(b_stride);
                    c_ptr = c_ptr.offset(c_stride);
                    e_ptr = e_ptr.offset(e_stride);
                }
            }
            Ok(())
        },
    )
}

fn zip_map4_4d_contig_dst<T, F>(
    dst_view: &StridedViewMut<T>,
    a_view: &StridedView<T>,
    b_view: &StridedView<T>,
    c_view: &StridedView<T>,
    e_view: &StridedView<T>,
    f: &F,
) -> Result<()>
where
    F: Fn(&T, &T, &T, &T) -> T,
{
    let d0 = dst_view.dims[0];
    let d1 = dst_view.dims[1];
    let d2 = dst_view.dims[2];
    let d3 = dst_view.dims[3];
    if d0 == 0 || d1 == 0 || d2 == 0 || d3 == 0 {
        return Ok(());
    }

    let ds0 = dst_view.strides[0];
    let ds1 = dst_view.strides[1];
    let ds2 = dst_view.strides[2];
    let ds3 = dst_view.strides[3];
    let as0 = a_view.strides[0];
    let as1 = a_view.strides[1];
    let as2 = a_view.strides[2];
    let as3 = a_view.strides[3];
    let bs0 = b_view.strides[0];
    let bs1 = b_view.strides[1];
    let bs2 = b_view.strides[2];
    let bs3 = b_view.strides[3];
    let cs0 = c_view.strides[0];
    let cs1 = c_view.strides[1];
    let cs2 = c_view.strides[2];
    let cs3 = c_view.strides[3];
    let es0 = e_view.strides[0];
    let es1 = e_view.strides[1];
    let es2 = e_view.strides[2];
    let es3 = e_view.strides[3];

    unsafe {
        let mut dst_i0 = dst_view.ptr;
        let mut a_i0 = a_view.ptr;
        let mut b_i0 = b_view.ptr;
        let mut c_i0 = c_view.ptr;
        let mut e_i0 = e_view.ptr;
        for _ in 0..d0 {
            let mut dst_i1 = dst_i0;
            let mut a_i1 = a_i0;
            let mut b_i1 = b_i0;
            let mut c_i1 = c_i0;
            let mut e_i1 = e_i0;
            for _ in 0..d1 {
                let mut dst_i2 = dst_i1;
                let mut a_i2 = a_i1;
                let mut b_i2 = b_i1;
                let mut c_i2 = c_i1;
                let mut e_i2 = e_i1;
                for _ in 0..d2 {
                    let mut dst_i3 = dst_i2;
                    let mut a_i3 = a_i2;
                    let mut b_i3 = b_i2;
                    let mut c_i3 = c_i2;
                    let mut e_i3 = e_i2;
                    for _ in 0..d3 {
                        let out = f(&*a_i3, &*b_i3, &*c_i3, &*e_i3);
                        *dst_i3 = out;
                        dst_i3 = dst_i3.offset(ds3);
                        a_i3 = a_i3.offset(as3);
                        b_i3 = b_i3.offset(bs3);
                        c_i3 = c_i3.offset(cs3);
                        e_i3 = e_i3.offset(es3);
                    }
                    dst_i2 = dst_i2.offset(ds2);
                    a_i2 = a_i2.offset(as2);
                    b_i2 = b_i2.offset(bs2);
                    c_i2 = c_i2.offset(cs2);
                    e_i2 = e_i2.offset(es2);
                }
                dst_i1 = dst_i1.offset(ds1);
                a_i1 = a_i1.offset(as1);
                b_i1 = b_i1.offset(bs1);
                c_i1 = c_i1.offset(cs1);
                e_i1 = e_i1.offset(es1);
            }
            dst_i0 = dst_i0.offset(ds0);
            a_i0 = a_i0.offset(as0);
            b_i0 = b_i0.offset(bs0);
            c_i0 = c_i0.offset(cs0);
            e_i0 = e_i0.offset(es0);
        }
    }

    Ok(())
}
