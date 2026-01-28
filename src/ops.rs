use crate::fuse::fuse_dims;
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

/// Apply dimension fusion to simplify iteration.
#[inline]
fn apply_fusion(dims: &[usize], strides_list: &[&[isize]]) -> Vec<usize> {
    if dims.len() <= 1 {
        return dims.to_vec();
    }
    fuse_dims(dims, strides_list)
}

pub fn copy_into<T, SD, SS, LD, LS>(
    dest: &mut Slice<T, SD, LD>,
    src: &Slice<T, SS, LS>,
) -> Result<()>
where
    T: Clone + 'static,
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

    if dst_view.dims.len() == 4 && is_contiguous(&dst_view.dims, &dst_view.strides) {
        copy_4d_contig_dst(&dst_view, &src_view)?;
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

    if is_contiguous(dest_dims, dest_strides) && is_contiguous(&src_view.dims, &src_view.strides) {
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

    let strides_list = [dest_strides, &src_view.strides[..]];

    // Apply dimension fusion to reduce loop levels
    let fused_dims = apply_fusion(dest_dims, &strides_list);

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

    // Apply dimension fusion to reduce loop levels
    let fused_dims = apply_fusion(&a_view.dims, &strides_list);

    let plan = build_plan(&fused_dims, &strides_list, None, std::mem::size_of::<T>());

    let mut acc = Some(T::zero());
    for_each_inner_block(
        &fused_dims,
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
    if is_contiguous(&dst_view.dims, &dst_view.strides)
        && is_contiguous(&src_view.dims, &src_view.strides)
    {
        return symmetrize_into_contig_row_major(&dst_view, &src_view, &half);
    }
    if is_contiguous_col_major_2d(&dst_view.dims, &dst_view.strides)
        && is_contiguous_col_major_2d(&src_view.dims, &src_view.strides)
    {
        return symmetrize_into_contig_col_major(&dst_view, &src_view, &half);
    }
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

    let tile = transpose_tile_size::<T>().clamp(1, TRANSPOSE_TILE);
    let s_row = src_view.strides[0];
    let s_col = src_view.strides[1];
    let d_row = dst_view.strides[0];
    let d_col = dst_view.strides[1];
    let prefer_write_contig = d_col.unsigned_abs() == 1
        && (s_col.unsigned_abs() != 1 || s_row.unsigned_abs() <= d_row.unsigned_abs());
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

/// Compute `B = (A + Aᴴ) / 2` into `dest`.
///
/// - Requires a square 2D input.
/// - Uses conjugate symmetry; diagonal elements become the real part of `A`.
pub fn symmetrize_conj_into<T, SD, SS, LD, LS>(
    dest: &mut Slice<T, SD, LD>,
    src: &Slice<T, SS, LS>,
) -> Result<()>
where
    T: ComplexFloat + Add<Output = T> + Mul<Output = T> + FromPrimitive,
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
    if is_contiguous(&dst_view.dims, &dst_view.strides)
        && is_contiguous(&src_view.dims, &src_view.strides)
    {
        return symmetrize_conj_into_contig_row_major(&dst_view, &src_view, &half);
    }
    if is_contiguous_col_major_2d(&dst_view.dims, &dst_view.strides)
        && is_contiguous_col_major_2d(&src_view.dims, &src_view.strides)
    {
        return symmetrize_conj_into_contig_col_major(&dst_view, &src_view, &half);
    }

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
                        let aij = *unsafe { &*src_view.ptr.offset(src_ij) };
                        if i == j {
                            let out = (aij + aij.conj()) * half;
                            unsafe {
                                *dst_view.ptr.offset(dst_ij) = out;
                            }
                        } else {
                            let src_ji = offset2d(j, i, s_row, s_col)?;
                            let dst_ji = offset2d(j, i, d_row, d_col)?;
                            let aji = *unsafe { &*src_view.ptr.offset(src_ji) };
                            let out = (aij + aji.conj()) * half;
                            unsafe {
                                *dst_view.ptr.offset(dst_ij) = out;
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
                        let aij = *unsafe { &*src_view.ptr.offset(src_ij) };
                        let aji = *unsafe { &*src_view.ptr.offset(src_ji) };
                        let out = (aij + aji.conj()) * half;
                        unsafe {
                            *dst_view.ptr.offset(dst_ij) = out;
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

fn symmetrize_into_contig_row_major<T>(
    dest: &StridedViewMut<T>,
    src: &StridedView<T>,
    half: &T,
) -> Result<()>
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
{
    let n = src.dims[0];
    let total = n.checked_mul(n).ok_or(StridedError::OffsetOverflow)?;
    if total == 0 {
        return Ok(());
    }

    let src_ptr = src.ptr;
    let dst_ptr = dest.ptr;
    for i in 0..n {
        let row_base = i.checked_mul(n).ok_or(StridedError::OffsetOverflow)?;
        let mut idx_ij = row_base
            .checked_add(i)
            .ok_or(StridedError::OffsetOverflow)?;
        let mut idx_ji = idx_ij;
        for j in i..n {
            let aij = unsafe { &*src_ptr.add(idx_ij) }.clone();
            if i == j {
                unsafe {
                    *dst_ptr.add(idx_ij) = aij;
                }
            } else {
                let aji = unsafe { &*src_ptr.add(idx_ji) }.clone();
                let out = (aij + aji) * half.clone();
                unsafe {
                    *dst_ptr.add(idx_ij) = out.clone();
                    *dst_ptr.add(idx_ji) = out;
                }
            }
            idx_ij = idx_ij.checked_add(1).ok_or(StridedError::OffsetOverflow)?;
            idx_ji = idx_ji.checked_add(n).ok_or(StridedError::OffsetOverflow)?;
        }
    }

    Ok(())
}

fn symmetrize_into_contig_col_major<T>(
    dest: &StridedViewMut<T>,
    src: &StridedView<T>,
    half: &T,
) -> Result<()>
where
    T: Clone + Add<Output = T> + Mul<Output = T>,
{
    let n = src.dims[0];
    let total = n.checked_mul(n).ok_or(StridedError::OffsetOverflow)?;
    if total == 0 {
        return Ok(());
    }

    let src_ptr = src.ptr;
    let dst_ptr = dest.ptr;
    for i in 0..n {
        let diag = i
            .checked_mul(n)
            .and_then(|v| v.checked_add(i))
            .ok_or(StridedError::OffsetOverflow)?;
        let mut idx_ij = diag;
        let mut idx_ji = diag;
        for _j in i..n {
            let aij = unsafe { &*src_ptr.add(idx_ij) }.clone();
            if idx_ij == idx_ji {
                unsafe {
                    *dst_ptr.add(idx_ij) = aij;
                }
            } else {
                let aji = unsafe { &*src_ptr.add(idx_ji) }.clone();
                let out = (aij + aji) * half.clone();
                unsafe {
                    *dst_ptr.add(idx_ij) = out.clone();
                    *dst_ptr.add(idx_ji) = out;
                }
            }
            idx_ij = idx_ij.checked_add(n).ok_or(StridedError::OffsetOverflow)?;
            idx_ji = idx_ji.checked_add(1).ok_or(StridedError::OffsetOverflow)?;
        }
    }

    Ok(())
}

fn symmetrize_conj_into_contig_row_major<T>(
    dest: &StridedViewMut<T>,
    src: &StridedView<T>,
    half: &T,
) -> Result<()>
where
    T: ComplexFloat + Add<Output = T> + Mul<Output = T>,
{
    let n = src.dims[0];
    let total = n.checked_mul(n).ok_or(StridedError::OffsetOverflow)?;
    if total == 0 {
        return Ok(());
    }

    let src_ptr = src.ptr;
    let dst_ptr = dest.ptr;
    for i in 0..n {
        let row_base = i.checked_mul(n).ok_or(StridedError::OffsetOverflow)?;
        let mut idx_ij = row_base
            .checked_add(i)
            .ok_or(StridedError::OffsetOverflow)?;
        let mut idx_ji = idx_ij;
        for _ in i..n {
            let aij = *unsafe { &*src_ptr.add(idx_ij) };
            if idx_ij == idx_ji {
                let out = (aij + aij.conj()) * *half;
                unsafe {
                    *dst_ptr.add(idx_ij) = out;
                }
            } else {
                let aji = *unsafe { &*src_ptr.add(idx_ji) };
                let out = (aij + aji.conj()) * *half;
                unsafe {
                    *dst_ptr.add(idx_ij) = out;
                    *dst_ptr.add(idx_ji) = out;
                }
            }
            idx_ij = idx_ij.checked_add(1).ok_or(StridedError::OffsetOverflow)?;
            idx_ji = idx_ji.checked_add(n).ok_or(StridedError::OffsetOverflow)?;
        }
    }

    Ok(())
}

fn symmetrize_conj_into_contig_col_major<T>(
    dest: &StridedViewMut<T>,
    src: &StridedView<T>,
    half: &T,
) -> Result<()>
where
    T: ComplexFloat + Add<Output = T> + Mul<Output = T>,
{
    let n = src.dims[0];
    let total = n.checked_mul(n).ok_or(StridedError::OffsetOverflow)?;
    if total == 0 {
        return Ok(());
    }

    let src_ptr = src.ptr;
    let dst_ptr = dest.ptr;
    for i in 0..n {
        let diag = i
            .checked_mul(n)
            .and_then(|v| v.checked_add(i))
            .ok_or(StridedError::OffsetOverflow)?;
        let mut idx_ij = diag;
        let mut idx_ji = diag;
        for _ in i..n {
            let aij = *unsafe { &*src_ptr.add(idx_ij) };
            if idx_ij == idx_ji {
                let out = (aij + aij.conj()) * *half;
                unsafe {
                    *dst_ptr.add(idx_ij) = out;
                }
            } else {
                let aji = *unsafe { &*src_ptr.add(idx_ji) };
                let out = (aij + aji.conj()) * *half;
                unsafe {
                    *dst_ptr.add(idx_ij) = out;
                    *dst_ptr.add(idx_ji) = out;
                }
            }
            idx_ij = idx_ij.checked_add(n).ok_or(StridedError::OffsetOverflow)?;
            idx_ji = idx_ji.checked_add(1).ok_or(StridedError::OffsetOverflow)?;
        }
    }

    Ok(())
}

fn is_contiguous_col_major_2d(dims: &[usize], strides: &[isize]) -> bool {
    if dims.len() != 2 || strides.len() != 2 {
        return false;
    }
    let rows = dims[0] as isize;
    if rows <= 0 {
        return true;
    }
    strides[0] == 1 && strides[1] == rows
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
    tile.clamp(1, TRANSPOSE_TILE)
}

/// Optimized scale-transpose using 4x4 micro-kernel.
///
/// This version minimizes overhead by:
/// - Using unchecked pointer arithmetic in inner loops (bounds validated upfront)
/// - Processing 4x4 blocks to maximize register utilization
/// - Specializing for row-major contiguous output (most common case)
pub fn copy_transpose_scale_into_fast<T>(
    dest: &mut Slice<T, impl Shape, impl Layout>,
    src: &Slice<T, impl Shape, impl Layout>,
    alpha: T,
) -> Result<()>
where
    T: Copy + Mul<Output = T>,
{
    let dst_view = StridedViewMut::from_slice(dest)?;
    let src_view = StridedView::from_slice(src)?;

    if src_view.dims.len() != 2 {
        return Err(StridedError::RankMismatch(src_view.dims.len(), 2));
    }
    if dst_view.dims.len() != 2 {
        return Err(StridedError::RankMismatch(dst_view.dims.len(), 2));
    }

    let rows = src_view.dims[0]; // src rows = dst cols
    let cols = src_view.dims[1]; // src cols = dst rows
    let expected = [cols, rows];
    if dst_view.dims[0] != expected[0] || dst_view.dims[1] != expected[1] {
        return Err(StridedError::ShapeMismatch(
            dst_view.dims.clone(),
            expected.to_vec(),
        ));
    }

    let s_row = src_view.strides[0];
    let s_col = src_view.strides[1];
    let d_row = dst_view.strides[0];
    let d_col = dst_view.strides[1];

    // Fast path: output is row-major contiguous (d_col == 1)
    // This is the common case: dest[j, i] = alpha * src[i, j]
    // Inner loop writes sequentially to dest rows
    if d_col == 1 {
        const TILE: usize = 4;
        let src_ptr = src_view.ptr;
        let dst_ptr = dst_view.ptr;

        // Process 4x4 tiles
        let rows_full = rows - rows % TILE;
        let cols_full = cols - cols % TILE;

        for j0 in (0..cols_full).step_by(TILE) {
            for i0 in (0..rows_full).step_by(TILE) {
                // 4x4 micro-kernel: load 4 rows of 4 elements each, transpose and store
                unsafe {
                    // Compute base offsets (only once per tile)
                    let src_base = src_ptr.offset(i0 as isize * s_row + j0 as isize * s_col);
                    let dst_base = dst_ptr.offset(j0 as isize * d_row + i0 as isize);

                    // Load 4x4 block from source (row by row)
                    // src[i0+k, j0+l] for k,l in 0..4
                    for l in 0..TILE {
                        let src_col = src_base.offset(l as isize * s_col);
                        let dst_row = dst_base.offset(l as isize * d_row);
                        for k in 0..TILE {
                            let val = *src_col.offset(k as isize * s_row);
                            *dst_row.add(k) = alpha * val;
                        }
                    }
                }
            }

            // Handle remaining rows (i0 >= rows_full)
            for i in rows_full..rows {
                unsafe {
                    let src_base = src_ptr.offset(i as isize * s_row + j0 as isize * s_col);
                    let dst_base = dst_ptr.offset(j0 as isize * d_row + i as isize);
                    for l in 0..TILE {
                        let val = *src_base.offset(l as isize * s_col);
                        *dst_base.offset(l as isize * d_row) = alpha * val;
                    }
                }
            }
        }

        // Handle remaining columns (j0 >= cols_full)
        for j in cols_full..cols {
            unsafe {
                let src_col = src_ptr.offset(j as isize * s_col);
                let dst_row = dst_ptr.offset(j as isize * d_row);
                for i in 0..rows {
                    let val = *src_col.offset(i as isize * s_row);
                    *dst_row.add(i) = alpha * val;
                }
            }
        }

        return Ok(());
    }

    // General case: use simple tiled loop
    const TILE: usize = 8;
    let src_ptr = src_view.ptr;
    let dst_ptr = dst_view.ptr;

    for i0 in (0..rows).step_by(TILE) {
        let i_end = (i0 + TILE).min(rows);
        for j0 in (0..cols).step_by(TILE) {
            let j_end = (j0 + TILE).min(cols);
            for i in i0..i_end {
                for j in j0..j_end {
                    unsafe {
                        let src_off = i as isize * s_row + j as isize * s_col;
                        let dst_off = j as isize * d_row + i as isize * d_col;
                        let val = *src_ptr.offset(src_off);
                        *dst_ptr.offset(dst_off) = alpha * val;
                    }
                }
            }
        }
    }

    Ok(())
}

fn copy_2d_contig_write<T>(dst_view: &StridedViewMut<T>, src_view: &StridedView<T>) -> Result<bool>
where
    T: Clone + 'static,
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

    // Specialized fast path: true transpose copy for square matrices.
    // This matches the common benchmark pattern: dst is row-major contiguous,
    // src is the transpose view of a row-major contiguous matrix (i.e. src is column-major).
    // For T: Copy we can use a stack tile buffer to turn strided loads/stores into
    // mostly contiguous accesses (Julia/Strided.jl style).
    if rows == cols
        && rows >= 64
        && d_col == 1
        && d_row.unsigned_abs() == cols
        && s_row.unsigned_abs() == 1
        && s_col.unsigned_abs() == rows
        && copy_2d_transpose_microtile(dst_view, src_view)?
    {
        return Ok(true);
    }

    // Check if this is a transpose-like pattern that benefits from tiling
    // Transpose pattern: reading with large stride, writing sequentially
    // Use tiling for arrays larger than 32x32 to improve cache performance
    // Smaller threshold helps with 100x100 arrays
    let is_transpose_pattern =
        (d_col == 1 && s_col.unsigned_abs() > 1) || (d_row == 1 && s_row.unsigned_abs() > 1);
    let is_large_enough = rows >= 32 && cols >= 32;

    if is_transpose_pattern && is_large_enough {
        // Prefer a POD (memcpy-based) fast path for common numeric element types.
        // Use a concrete-type check here so f32/f64 take the POD fast path immediately.
        // (We'll later replace this with a static `T: Pod` bound once Pod wrappers/derives
        // are added for other types.)
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()
            || std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            if copy_2d_transpose_pod(dst_view, src_view)? {
                return Ok(true);
            }
        }
        return copy_2d_tiled(dst_view, src_view);
    }

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

/// Tiled transpose copy for square matrices when:
/// - dst is row-major contiguous (d_col == 1)
/// - src is column-major contiguous (s_row == 1)
///
/// Returns:
/// - `Some(true)` if the optimized path ran
/// - `Some(false)` if the shape/layout doesn't match the fast path
/// - `None` if `T` is not `Copy` (caller should fall back)
#[inline]
fn copy_2d_transpose_microtile<T>(
    dst_view: &StridedViewMut<T>,
    src_view: &StridedView<T>,
) -> Result<bool>
where
    T: Clone,
{
    if dst_view.dims.len() != 2 || src_view.dims.len() != 2 {
        return Ok(false);
    }
    let n = dst_view.dims[0];
    if n == 0 || n != dst_view.dims[1] || n != src_view.dims[0] || n != src_view.dims[1] {
        return Ok(false);
    }

    let d_row = dst_view.strides[0];
    let d_col = dst_view.strides[1];
    let s_row = src_view.strides[0];
    let s_col = src_view.strides[1];

    if d_col != 1 || d_row.unsigned_abs() != n {
        return Ok(false);
    }
    if s_row.unsigned_abs() != 1 || s_col.unsigned_abs() != n {
        return Ok(false);
    }

    // Use Julia's block computation to pick outer tile sizes.
    use crate::block::compute_block_sizes;

    let order = [0usize, 1usize];
    let elem_size = std::mem::size_of::<T>();
    let strides_list: [&[isize]; 2] = [&dst_view.strides[..], &src_view.strides[..]];
    let blocks = compute_block_sizes(&[n, n], &order, &strides_list, elem_size);
    let outer_i = blocks.get(0).copied().unwrap_or(n).max(1);
    let outer_j = blocks.get(1).copied().unwrap_or(n).max(1);

    const MICRO: usize = 4; // micro-kernel size (register block)

    let src_ptr = src_view.ptr;
    let dst_ptr = dst_view.ptr;

    for i0 in (0..n).step_by(outer_i) {
        let i_end = (i0 + outer_i).min(n);
        for j0 in (0..n).step_by(outer_j) {
            let j_end = (j0 + outer_j).min(n);

            for ii in (i0..i_end).step_by(MICRO) {
                for jj in (j0..j_end).step_by(MICRO) {
                    // compute block dims
                    let i_len = (ii + MICRO).min(i_end) - ii;
                    let j_len = (jj + MICRO).min(j_end) - jj;

                    unsafe {
                        // load micro-tile into registers (scalars)
                        let mut buf: [std::mem::MaybeUninit<T>; MICRO * MICRO] =
                            std::mem::MaybeUninit::uninit().assume_init();
                        for j in 0..j_len {
                            let s_col_ptr = src_ptr.offset((ii as isize) * s_row + ((jj + j) as isize) * s_col);
                            for i in 0..i_len {
                                let v = (&*s_col_ptr.offset(i as isize * s_row)).clone();
                                buf[i * MICRO + j].write(v);
                            }
                        }

                        // store transposed
                        for j in 0..j_len {
                            let d_row_ptr = dst_ptr.offset(((jj + j) as isize) * d_row + (ii as isize) * d_col);
                            for i in 0..i_len {
                                let v = buf[i * MICRO + j].assume_init_read();
                                *d_row_ptr.add(i) = v;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(true)
}

fn copy_4d_contig_dst<T>(dst_view: &StridedViewMut<T>, src_view: &StridedView<T>) -> Result<()>
where
    T: Clone,
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
    let ss0 = src_view.strides[0];
    let ss1 = src_view.strides[1];
    let ss2 = src_view.strides[2];
    let ss3 = src_view.strides[3];

    unsafe {
        let mut dst_i0 = dst_view.ptr;
        let mut src_i0 = src_view.ptr;
        for _ in 0..d0 {
            let mut dst_i1 = dst_i0;
            let mut src_i1 = src_i0;
            for _ in 0..d1 {
                let mut dst_i2 = dst_i1;
                let mut src_i2 = src_i1;
                for _ in 0..d2 {
                    let mut dst_i3 = dst_i2;
                    let mut src_i3 = src_i2;
                    for _ in 0..d3 {
                        let val = (&*src_i3).clone();
                        *dst_i3 = val;
                        dst_i3 = dst_i3.offset(ds3);
                        src_i3 = src_i3.offset(ss3);
                    }
                    dst_i2 = dst_i2.offset(ds2);
                    src_i2 = src_i2.offset(ss2);
                }
                dst_i1 = dst_i1.offset(ds1);
                src_i1 = src_i1.offset(ss1);
            }
            dst_i0 = dst_i0.offset(ds0);
            src_i0 = src_i0.offset(ss0);
        }
    }

    Ok(())
}

/// Tiled 2D copy for better cache performance on transpose-like patterns.
///
/// Uses 8x8 blocking to keep data in L1 cache during transpose operations.
fn copy_2d_tiled<T>(dst_view: &StridedViewMut<T>, src_view: &StridedView<T>) -> Result<bool>
where
    T: Clone,
{
    const TILE: usize = 8;

    let rows = dst_view.dims[0];
    let cols = dst_view.dims[1];
    let s_row = src_view.strides[0];
    let s_col = src_view.strides[1];
    let d_row = dst_view.strides[0];
    let d_col = dst_view.strides[1];

    let src_ptr = src_view.ptr;
    let dst_ptr = dst_view.ptr;

    // Process full tiles
    let rows_full = rows - rows % TILE;
    let cols_full = cols - cols % TILE;

    for i0 in (0..rows_full).step_by(TILE) {
        for j0 in (0..cols_full).step_by(TILE) {
            // Process 8x8 tile
            unsafe {
                let src_base = src_ptr.offset(i0 as isize * s_row + j0 as isize * s_col);
                let dst_base = dst_ptr.offset(i0 as isize * d_row + j0 as isize * d_col);

                for i in 0..TILE {
                    let src_row = src_base.offset(i as isize * s_row);
                    let dst_row = dst_base.offset(i as isize * d_row);
                    for j in 0..TILE {
                        let val = (*src_row.offset(j as isize * s_col)).clone();
                        *dst_row.offset(j as isize * d_col) = val;
                    }
                }
            }
        }

        // Handle remaining columns
        for j in cols_full..cols {
            unsafe {
                let src_col = src_ptr.offset(j as isize * s_col);
                let dst_col = dst_ptr.offset(j as isize * d_col);
                for i in i0..i0 + TILE {
                    let val = (*src_col.offset(i as isize * s_row)).clone();
                    *dst_col.offset(i as isize * d_row) = val;
                }
            }
        }
    }

    // Handle remaining rows
    for i in rows_full..rows {
        unsafe {
            let src_row = src_ptr.offset(i as isize * s_row);
            let dst_row = dst_ptr.offset(i as isize * d_row);
            for j in 0..cols {
                let val = (*src_row.offset(j as isize * s_col)).clone();
                *dst_row.offset(j as isize * d_col) = val;
            }
        }
    }

    Ok(true)
}

/// POD-optimized transpose copy: use memcpy-like element moves instead of `clone()`.
/// Called only when the element type has no drop glue (i.e. safe to bitwise-move).
fn copy_2d_transpose_pod<T>(dst_view: &StridedViewMut<T>, src_view: &StridedView<T>) -> Result<bool>
where
    T: Clone,
{
    if dst_view.dims.len() != 2 || src_view.dims.len() != 2 {
        return Ok(false);
    }
    let n = dst_view.dims[0];
    if n == 0 || n != dst_view.dims[1] || n != src_view.dims[0] || n != src_view.dims[1] {
        return Ok(false);
    }

    let d_row = dst_view.strides[0];
    let d_col = dst_view.strides[1];
    let s_row = src_view.strides[0];
    let s_col = src_view.strides[1];

    // Canonical layouts only: dst row-major contiguous, src column-major contiguous.
    if d_col != 1 || d_row.unsigned_abs() != n {
        return Ok(false);
    }
    if s_row.unsigned_abs() != 1 || s_col.unsigned_abs() != n {
        return Ok(false);
    }

    use crate::block::compute_block_sizes;
    let order = [0usize, 1usize];
    let elem_size = std::mem::size_of::<T>();
    let strides_list: [&[isize]; 2] = [&dst_view.strides[..], &src_view.strides[..]];
    let blocks = compute_block_sizes(&[n, n], &order, &strides_list, elem_size);
    let outer_i = blocks.get(0).copied().unwrap_or(n).max(1);
    let outer_j = blocks.get(1).copied().unwrap_or(n).max(1);

    const MICRO: usize = 8; // larger micro-kernel for POD types

    let src_ptr_u8 = src_view.ptr as *const u8;
    let dst_ptr_u8 = dst_view.ptr as *mut u8;

    for i0 in (0..n).step_by(outer_i) {
        let i_end = (i0 + outer_i).min(n);
        for j0 in (0..n).step_by(outer_j) {
            let j_end = (j0 + outer_j).min(n);

            for ii in (i0..i_end).step_by(MICRO) {
                for jj in (j0..j_end).step_by(MICRO) {
                    let i_len = (ii + MICRO).min(i_end) - ii;
                    let j_len = (jj + MICRO).min(j_end) - jj;

                    for j in 0..j_len {
                        // src column pointer in bytes
                        let src_col_base = unsafe {
                            src_ptr_u8.offset((ii as isize * s_row + (jj + j) as isize * s_col) * elem_size as isize)
                        };
                        // dst row pointer in bytes
                        let dst_row_base = unsafe {
                            dst_ptr_u8.offset(((jj + j) as isize * d_row + ii as isize * d_col) * elem_size as isize)
                        };

                        // If element size matches f64 or f32, copy entire inner column block
                        // as elements rather than byte-by-byte; this reduces per-element
                        // overhead and improves codegen for common numeric types.
                        if elem_size == 8 {
                            // f64 path
                            let src_first = unsafe { src_col_base } as *const f64;
                            let dst_first = unsafe { dst_row_base } as *mut f64;
                            unsafe { std::ptr::copy_nonoverlapping(src_first, dst_first, i_len) };
                        } else if elem_size == 4 {
                            // f32 path
                            let src_first = unsafe { src_col_base } as *const f32;
                            let dst_first = unsafe { dst_row_base } as *mut f32;
                            unsafe { std::ptr::copy_nonoverlapping(src_first, dst_first, i_len) };
                        } else {
                            for i in 0..i_len {
                                let src_elem = unsafe { src_col_base.offset(i as isize * s_row * elem_size as isize) };
                                let dst_elem = unsafe { dst_row_base.add(i * elem_size) };
                                unsafe { std::ptr::copy_nonoverlapping(src_elem, dst_elem, elem_size) };
                            }
                        }
                    }
                }
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

fn checked_add(a: isize, b: isize) -> Result<isize> {
    a.checked_add(b).ok_or(StridedError::OffsetOverflow)
}
