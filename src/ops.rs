use crate::kernel::{
    build_plan, ensure_same_shape, for_each_inner_block, is_contiguous, total_len, validate_layout,
    StridedView, StridedViewMut,
};
use crate::map::{map_into, zip_map2_into, zip_map3_into};
use crate::reduce::reduce;
use crate::{Result, StridedError};
use mdarray::{Layout, Shape, Slice};
use num_complex::ComplexFloat;
use num_traits::FromPrimitive;
use num_traits::Zero;
use std::mem::MaybeUninit;
use std::ops::{Add, Mul};
const TRANSPOSE_TILE: usize = 16;

pub fn copy_into<T, SD, SS, LD, LS>(
    dest: &mut Slice<T, SD, LD>,
    src: &Slice<T, SS, LS>,
) -> Result<()>
where
    T: Clone,
    SD: Shape,
    SS: Shape,
    LD: Layout,
    LS: Layout,
{
    let dst_view = StridedViewMut::from_slice(dest)?;
    let src_view = StridedView::from_slice(src)?;
    ensure_same_shape(&dst_view.dims, &src_view.dims)?;

    if is_contiguous(&dst_view.dims, &dst_view.strides)
        && is_contiguous(&src_view.dims, &src_view.strides)
    {
        let len = total_len(&dst_view.dims);
        unsafe {
            let dst_slice = std::slice::from_raw_parts_mut(dst_view.ptr, len);
            let src_slice = std::slice::from_raw_parts(src_view.ptr, len);
            dst_slice.clone_from_slice(src_slice);
        }
        return Ok(());
    }

    if copy_2d_contig_write(&dst_view, &src_view)? {
        return Ok(());
    }

    map_into(dest, src, |x| x.clone())
}

/// Copy `src` into an uninitialized destination buffer.
///
/// # Safety
/// - `dest_ptr` must be valid for writes to all elements described by
///   `dest_dims` and `dest_strides`.
/// - The destination memory must be uninitialized or otherwise safe to overwrite
///   without dropping existing values.
/// - `dest_ptr` must not overlap with `src` memory.
pub unsafe fn copy_into_uninit<T, SS, LS>(
    dest_ptr: *mut MaybeUninit<T>,
    dest_dims: &[usize],
    dest_strides: &[isize],
    src: &Slice<T, SS, LS>,
) -> Result<()>
where
    T: Clone,
    SS: Shape,
    LS: Layout,
{
    validate_layout(dest_dims, dest_strides)?;
    let src_view = StridedView::from_slice(src)?;
    ensure_same_shape(dest_dims, &src_view.dims)?;

    if is_contiguous(dest_dims, dest_strides)
        && is_contiguous(&src_view.dims, &src_view.strides)
    {
        let len = total_len(dest_dims);
        let mut dst_ptr = dest_ptr;
        let mut src_ptr = src_view.ptr;
        for _ in 0..len {
            let val = unsafe { &*src_ptr }.clone();
            unsafe {
                (*dst_ptr).write(val);
                dst_ptr = dst_ptr.add(1);
                src_ptr = src_ptr.add(1);
            }
        }
        return Ok(());
    }

    let strides_list = [&dest_strides[..], &src_view.strides[..]];
    let plan = build_plan(dest_dims, &strides_list, Some(0), std::mem::size_of::<T>());
    for_each_inner_block(
        dest_dims,
        &plan,
        &strides_list,
        |offsets, len, strides| {
            let mut dst_ptr = unsafe { dest_ptr.offset(offsets[0]) };
            let mut src_ptr = unsafe { src_view.ptr.offset(offsets[1]) };
            let dst_stride = strides[0];
            let src_stride = strides[1];
            for _ in 0..len {
                let val = unsafe { &*src_ptr }.clone();
                unsafe {
                    (*dst_ptr).write(val);
                    dst_ptr = dst_ptr.offset(dst_stride);
                    src_ptr = src_ptr.offset(src_stride);
                }
            }
            Ok(())
        },
    )
}

pub fn copy_conj<T, SD, SS, LD, LS>(
    dest: &mut Slice<T, SD, LD>,
    src: &Slice<T, SS, LS>,
) -> Result<()>
where
    T: ComplexFloat,
    SD: Shape,
    SS: Shape,
    LD: Layout,
    LS: Layout,
{
    map_into(dest, src, |x| x.conj())
}

pub fn copy_scale<T, SD, SS, LD, LS>(
    dest: &mut Slice<T, SD, LD>,
    src: &Slice<T, SS, LS>,
    alpha: T,
) -> Result<()>
where
    T: Clone + Mul<Output = T>,
    SD: Shape,
    SS: Shape,
    LD: Layout,
    LS: Layout,
{
    map_into(dest, src, |x| alpha.clone() * x.clone())
}

pub fn add<T, SD, SA, SB, LD, LA, LB>(
    dest: &mut Slice<T, SD, LD>,
    a: &Slice<T, SA, LA>,
    b: &Slice<T, SB, LB>,
) -> Result<()>
where
    T: Clone + Add<Output = T>,
    SD: Shape,
    SA: Shape,
    SB: Shape,
    LD: Layout,
    LA: Layout,
    LB: Layout,
{
    zip_map2_into(dest, a, b, |x, y| x.clone() + y.clone())
}

pub fn mul<T, SD, SA, SB, LD, LA, LB>(
    dest: &mut Slice<T, SD, LD>,
    a: &Slice<T, SA, LA>,
    b: &Slice<T, SB, LB>,
) -> Result<()>
where
    T: Clone + Mul<Output = T>,
    SD: Shape,
    SA: Shape,
    SB: Shape,
    LD: Layout,
    LA: Layout,
    LB: Layout,
{
    zip_map2_into(dest, a, b, |x, y| x.clone() * y.clone())
}

pub fn axpy<T, SD, SA, SB, LD, LA, LB>(
    dest: &mut Slice<T, SD, LD>,
    a: &Slice<T, SA, LA>,
    b: &Slice<T, SB, LB>,
    alpha: T,
) -> Result<()>
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
    SD: Shape,
    SA: Shape,
    SB: Shape,
    LD: Layout,
    LA: Layout,
    LB: Layout,
{
    zip_map2_into(dest, a, b, |x, y| alpha.clone() * x.clone() + y.clone())
}

pub fn fma<T, SD, SA, SB, SC, LD, LA, LB, LC>(
    dest: &mut Slice<T, SD, LD>,
    a: &Slice<T, SA, LA>,
    b: &Slice<T, SB, LB>,
    c: &Slice<T, SC, LC>,
) -> Result<()>
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
    SD: Shape,
    SA: Shape,
    SB: Shape,
    SC: Shape,
    LD: Layout,
    LA: Layout,
    LB: Layout,
    LC: Layout,
{
    zip_map3_into(dest, a, b, c, |x, y, z| x.clone() * y.clone() + z.clone())
}

pub fn sum<T, S, L>(src: &Slice<T, S, L>) -> Result<T>
where
    T: Clone + Zero + Add<Output = T>,
    S: Shape,
    L: Layout,
{
    reduce(src, |x| x.clone(), |a, b| a + b, T::zero())
}

pub fn dot<T, SA, SB, LA, LB>(a: &Slice<T, SA, LA>, b: &Slice<T, SB, LB>) -> Result<T>
where
    T: Clone + Zero + Add<Output = T> + Mul<Output = T>,
    SA: Shape,
    SB: Shape,
    LA: Layout,
    LB: Layout,
{
    let a_view = StridedView::from_slice(a)?;
    let b_view = StridedView::from_slice(b)?;
    ensure_same_shape(&a_view.dims, &b_view.dims)?;

    if is_contiguous(&a_view.dims, &a_view.strides) && is_contiguous(&b_view.dims, &b_view.strides)
    {
        let len = total_len(&a_view.dims);
        let mut a_ptr = a_view.ptr;
        let mut b_ptr = b_view.ptr;
        let mut acc = T::zero();
        for _ in 0..len {
            acc = acc + unsafe { &*a_ptr }.clone() * unsafe { &*b_ptr }.clone();
            unsafe {
                a_ptr = a_ptr.add(1);
                b_ptr = b_ptr.add(1);
            }
        }
        return Ok(acc);
    }

    let strides_list = [&a_view.strides[..], &b_view.strides[..]];
    let plan = build_plan(&a_view.dims, &strides_list, None, std::mem::size_of::<T>());

    let mut acc = Some(T::zero());
    for_each_inner_block(
        &a_view.dims,
        &plan,
        &strides_list,
        |offsets, len, strides| {
            let mut a_ptr = unsafe { a_view.ptr.offset(offsets[0]) };
            let mut b_ptr = unsafe { b_view.ptr.offset(offsets[1]) };
            let a_stride = strides[0];
            let b_stride = strides[1];
            for _ in 0..len {
                let current = acc.take().ok_or(crate::StridedError::OffsetOverflow)?;
                let next = current + unsafe { &*a_ptr }.clone() * unsafe { &*b_ptr }.clone();
                acc = Some(next);
                unsafe {
                    a_ptr = a_ptr.offset(a_stride);
                    b_ptr = b_ptr.offset(b_stride);
                }
            }
            Ok(())
        },
    )?;

    acc.ok_or(StridedError::OffsetOverflow)
}

pub fn symmetrize_into<T, SD, SS, LD, LS>(
    dest: &mut Slice<T, SD, LD>,
    src: &Slice<T, SS, LS>,
) -> Result<()>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + FromPrimitive,
    SD: Shape,
    SS: Shape,
    LD: Layout,
    LS: Layout,
{
    let dst_view = StridedViewMut::from_slice(dest)?;
    let src_view = StridedView::from_slice(src)?;
    ensure_same_shape(&dst_view.dims, &src_view.dims)?;

    if src_view.dims.len() != 2 {
        return Err(StridedError::RankMismatch(src_view.dims.len(), 2));
    }
    let n = src_view.dims[0];
    let m = src_view.dims[1];
    if n != m {
        return Err(StridedError::NonSquare { rows: n, cols: m });
    }

    let half = T::from_f64(0.5).ok_or(StridedError::ScalarConversion)?;
    let tile = sym_tile_size::<T>();
    let s_row = src_view.strides[0];
    let s_col = src_view.strides[1];
    let d_row = dst_view.strides[0];
    let d_col = dst_view.strides[1];

    for i0 in (0..n).step_by(tile) {
        let i_max = (i0 + tile).min(n);
        for j0 in (i0..n).step_by(tile) {
            let j_max = (j0 + tile).min(n);

            if i0 == j0 {
                for i in i0..i_max {
                    let start_j = i.max(j0);
                    let mut src_ij = offset2d(i, start_j, s_row, s_col)?;
                    let mut dst_ij = offset2d(i, start_j, d_row, d_col)?;
                    for j in start_j..j_max {
                        let aij = unsafe { &*src_view.ptr.offset(src_ij) }.clone();
                        if i == j {
                            unsafe {
                                *dst_view.ptr.offset(dst_ij) = aij;
                            }
                        } else {
                            let src_ji = offset2d(j, i, s_row, s_col)?;
                            let dst_ji = offset2d(j, i, d_row, d_col)?;
                            let aji = unsafe { &*src_view.ptr.offset(src_ji) }.clone();
                            let out = (aij + aji) * half.clone();
                            unsafe {
                                *dst_view.ptr.offset(dst_ij) = out.clone();
                                *dst_view.ptr.offset(dst_ji) = out;
                            }
                        }
                        src_ij = checked_add(src_ij, s_col)?;
                        dst_ij = checked_add(dst_ij, d_col)?;
                    }
                }
            } else {
                for i in i0..i_max {
                    let mut src_ij = offset2d(i, j0, s_row, s_col)?;
                    let mut dst_ij = offset2d(i, j0, d_row, d_col)?;
                    let mut src_ji = offset2d(j0, i, s_row, s_col)?;
                    let mut dst_ji = offset2d(j0, i, d_row, d_col)?;
                    for _ in j0..j_max {
                        let aij = unsafe { &*src_view.ptr.offset(src_ij) }.clone();
                        let aji = unsafe { &*src_view.ptr.offset(src_ji) }.clone();
                        let out = (aij + aji) * half.clone();
                        unsafe {
                            *dst_view.ptr.offset(dst_ij) = out.clone();
                            *dst_view.ptr.offset(dst_ji) = out;
                        }
                        src_ij = checked_add(src_ij, s_col)?;
                        dst_ij = checked_add(dst_ij, d_col)?;
                        src_ji = checked_add(src_ji, s_row)?;
                        dst_ji = checked_add(dst_ji, d_row)?;
                    }
                }
            }
        }
    }

    Ok(())
}

pub fn copy_transpose_scale_into<T, SD, SS, LD, LS>(
    dest: &mut Slice<T, SD, LD>,
    src: &Slice<T, SS, LS>,
    alpha: T,
) -> Result<()>
where
    T: Clone + Mul<Output = T>,
    SD: Shape,
    SS: Shape,
    LD: Layout,
    LS: Layout,
{
    let tile = transpose_tile_size::<T>().min(TRANSPOSE_TILE).max(1);
    copy_transpose_scale_into_tiled(dest, src, alpha, tile)
}

pub fn copy_transpose_scale_into_tiled<T, SD, SS, LD, LS>(
    dest: &mut Slice<T, SD, LD>,
    src: &Slice<T, SS, LS>,
    alpha: T,
    tile: usize,
) -> Result<()>
where
    T: Clone + Mul<Output = T>,
    SD: Shape,
    SS: Shape,
    LD: Layout,
    LS: Layout,
{
    let dst_view = StridedViewMut::from_slice(dest)?;
    let src_view = StridedView::from_slice(src)?;

    if src_view.dims.len() != 2 {
        return Err(StridedError::RankMismatch(src_view.dims.len(), 2));
    }
    if dst_view.dims.len() != 2 {
        return Err(StridedError::RankMismatch(dst_view.dims.len(), 2));
    }

    let rows = src_view.dims[0];
    let cols = src_view.dims[1];
    let expected = [cols, rows];
    if dst_view.dims[0] != expected[0] || dst_view.dims[1] != expected[1] {
        return Err(StridedError::ShapeMismatch(
            dst_view.dims.clone(),
            expected.to_vec(),
        ));
    }

    let tile = tile.min(TRANSPOSE_TILE).max(1);
    let s_row = src_view.strides[0];
    let s_col = src_view.strides[1];
    let d_row = dst_view.strides[0];
    let d_col = dst_view.strides[1];
    let prefer_write_contig =
        d_col.unsigned_abs() == 1 && (s_col.unsigned_abs() != 1 || s_row.unsigned_abs() <= d_row.unsigned_abs());
    let prefer_read_contig = s_col.unsigned_abs() == 1;

    if prefer_write_contig {
        for i0 in (0..rows).step_by(tile) {
            let i_max = (i0 + tile).min(rows);
            for j0 in (0..cols).step_by(tile) {
                let j_max = (j0 + tile).min(cols);
                for j in j0..j_max {
                    let src_ij = offset2d(i0, j, s_row, s_col)?;
                    let dst_ji = offset2d(j, i0, d_row, d_col)?;
                    let mut src_ptr = unsafe { src_view.ptr.offset(src_ij) };
                    let mut dst_ptr = unsafe { dst_view.ptr.offset(dst_ji) };
                    for _ in i0..i_max {
                        let aij = unsafe { &*src_ptr }.clone();
                        let out = alpha.clone() * aij;
                        unsafe {
                            *dst_ptr = out;
                            src_ptr = src_ptr.offset(s_row);
                            dst_ptr = dst_ptr.offset(d_col);
                        }
                    }
                }
            }
        }
    } else if prefer_read_contig {
        for i0 in (0..rows).step_by(tile) {
            let i_max = (i0 + tile).min(rows);
            for j0 in (0..cols).step_by(tile) {
                let j_max = (j0 + tile).min(cols);
                for i in i0..i_max {
                    let src_ij = offset2d(i, j0, s_row, s_col)?;
                    let dst_ji = offset2d(j0, i, d_row, d_col)?;
                    let mut src_ptr = unsafe { src_view.ptr.offset(src_ij) };
                    let mut dst_ptr = unsafe { dst_view.ptr.offset(dst_ji) };
                    for _ in j0..j_max {
                        let aij = unsafe { &*src_ptr }.clone();
                        let out = alpha.clone() * aij;
                        unsafe {
                            *dst_ptr = out;
                            src_ptr = src_ptr.offset(s_col);
                            dst_ptr = dst_ptr.offset(d_row);
                        }
                    }
                }
            }
        }
    } else {
        for i0 in (0..rows).step_by(tile) {
            let i_max = (i0 + tile).min(rows);
            for j0 in (0..cols).step_by(tile) {
                let j_max = (j0 + tile).min(cols);
                for j in j0..j_max {
                    let mut src_ij = offset2d(i0, j, s_row, s_col)?;
                    let mut dst_ji = offset2d(j, i0, d_row, d_col)?;
                    for _ in i0..i_max {
                        let aij = unsafe { &*src_view.ptr.offset(src_ij) }.clone();
                        let out = alpha.clone() * aij;
                        unsafe {
                            *dst_view.ptr.offset(dst_ji) = out;
                        }
                        src_ij = checked_add(src_ij, s_row)?;
                        dst_ji = checked_add(dst_ji, d_col)?;
                    }
                }
            }
        }
    }

    Ok(())
}

fn sym_tile_size<T>() -> usize {
    let bytes = std::mem::size_of::<T>().max(1);
    let per_array = crate::BLOCK_MEMORY_SIZE / 4;
    let tile_elems = (per_array / bytes).max(1);
    let tile = (tile_elems as f64).sqrt() as usize;
    tile.max(1)
}

fn transpose_tile_size<T>() -> usize {
    let bytes = std::mem::size_of::<T>().max(1);
    let per_array = crate::BLOCK_MEMORY_SIZE / 2;
    let tile_elems = (per_array / bytes).max(1);
    let tile = (tile_elems as f64).sqrt() as usize;
    tile.max(1).min(TRANSPOSE_TILE)
}

fn copy_2d_contig_write<T>(
    dst_view: &StridedViewMut<T>,
    src_view: &StridedView<T>,
) -> Result<bool>
where
    T: Clone,
{
    if dst_view.dims.len() != 2 {
        return Ok(false);
    }

    let rows = dst_view.dims[0];
    let cols = dst_view.dims[1];
    let s_row = src_view.strides[0];
    let s_col = src_view.strides[1];
    let d_row = dst_view.strides[0];
    let d_col = dst_view.strides[1];

    if d_col.unsigned_abs() == 1 {
        for i in 0..rows {
            let src_ij = offset2d(i, 0, s_row, s_col)?;
            let dst_ij = offset2d(i, 0, d_row, d_col)?;
            let mut src_ptr = unsafe { src_view.ptr.offset(src_ij) };
            let mut dst_ptr = unsafe { dst_view.ptr.offset(dst_ij) };
            for _ in 0..cols {
                let val = unsafe { &*src_ptr }.clone();
                unsafe {
                    *dst_ptr = val;
                    src_ptr = src_ptr.offset(s_col);
                    dst_ptr = dst_ptr.offset(d_col);
                }
            }
        }
        return Ok(true);
    }

    if d_row.unsigned_abs() == 1 {
        for j in 0..cols {
            let src_ij = offset2d(0, j, s_row, s_col)?;
            let dst_ij = offset2d(0, j, d_row, d_col)?;
            let mut src_ptr = unsafe { src_view.ptr.offset(src_ij) };
            let mut dst_ptr = unsafe { dst_view.ptr.offset(dst_ij) };
            for _ in 0..rows {
                let val = unsafe { &*src_ptr }.clone();
                unsafe {
                    *dst_ptr = val;
                    src_ptr = src_ptr.offset(s_row);
                    dst_ptr = dst_ptr.offset(d_row);
                }
            }
        }
        return Ok(true);
    }

    Ok(false)
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

fn checked_add(a: isize, b: isize) -> Result<isize> {
    a.checked_add(b).ok_or(StridedError::OffsetOverflow)
}
