use crate::kernel::{
    build_plan, ensure_same_shape, for_each_inner_block, is_contiguous, total_len, StridedView,
    StridedViewMut,
};
use crate::{Result, StridedError};
use mdarray::{Layout, Shape, Slice};

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
    let plan = build_plan(
        &dst_view.dims,
        &strides_list,
        Some(0),
        std::mem::size_of::<T>(),
    );

    for_each_inner_block(
        &dst_view.dims,
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
    let plan = build_plan(
        &dst_view.dims,
        &strides_list,
        Some(0),
        std::mem::size_of::<T>(),
    );

    for_each_inner_block(
        &dst_view.dims,
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
    let plan = build_plan(
        &dst_view.dims,
        &strides_list,
        Some(0),
        std::mem::size_of::<T>(),
    );

    for_each_inner_block(
        &dst_view.dims,
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
