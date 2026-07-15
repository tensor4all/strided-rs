//! Allocation-free copy/axpy over borrowed raw strided layouts.
//!
//! [`StridedView`]/[`StridedViewMut`] own their metadata (`Arc<[usize]>` /
//! `Arc<[isize]>`) and the map/zip kernels build a traversal plan per call;
//! for small replay copies that fixed cost dominates. These entry points take
//! [`RawStridedRef`]/[`RawStridedMut`] (borrowed metadata), fuse the stride
//! pair into a stack-allocated loop nest, and run plain loops - no heap
//! allocation on any call path with rank at most [`RAW_FUSED_RANK_LIMIT`].
//! Higher ranks fall back to the view-based kernels.

use crate::ops_view::{axpy, copy_scale};
use crate::{ElementOpApply, RawStridedMut, RawStridedRef, Result};
use core::ops::{Add, Mul};

use crate::maybe_sync::MaybeSendSync;

/// Maximum rank fused on the stack before falling back to the view kernels.
pub const RAW_FUSED_RANK_LIMIT: usize = 8;

/// Stack-allocated fused stride pair (dims ordered by destination stride,
/// adjacent contiguous axes merged). Built once and replayed by both the
/// per-call raw kernels and the prepared [`crate::CopyPlan`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct FusedPairLayout {
    pub(crate) rank: usize,
    pub(crate) dims: [usize; RAW_FUSED_RANK_LIMIT],
    pub(crate) dst_strides: [isize; RAW_FUSED_RANK_LIMIT],
    pub(crate) src_strides: [isize; RAW_FUSED_RANK_LIMIT],
}

pub(crate) fn fuse_pair_layout(
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
) -> Option<FusedPairLayout> {
    if dims.len() > RAW_FUSED_RANK_LIMIT {
        return None;
    }
    let mut layout = FusedPairLayout {
        rank: 0,
        dims: [1; RAW_FUSED_RANK_LIMIT],
        dst_strides: [0; RAW_FUSED_RANK_LIMIT],
        src_strides: [0; RAW_FUSED_RANK_LIMIT],
    };
    for axis in 0..dims.len() {
        if dims[axis] == 1 {
            continue;
        }
        if dims[axis] == 0 {
            return Some(FusedPairLayout {
                rank: 1,
                dims: [0; RAW_FUSED_RANK_LIMIT],
                dst_strides: [0; RAW_FUSED_RANK_LIMIT],
                src_strides: [0; RAW_FUSED_RANK_LIMIT],
            });
        }
        let mut position = layout.rank;
        while position > 0 && layout.dst_strides[position - 1] > dst_strides[axis] {
            layout.dims[position] = layout.dims[position - 1];
            layout.dst_strides[position] = layout.dst_strides[position - 1];
            layout.src_strides[position] = layout.src_strides[position - 1];
            position -= 1;
        }
        layout.dims[position] = dims[axis];
        layout.dst_strides[position] = dst_strides[axis];
        layout.src_strides[position] = src_strides[axis];
        layout.rank += 1;
    }
    if layout.rank == 0 {
        layout.rank = 1;
        layout.dims[0] = 1;
    }
    let mut fused = 0usize;
    for axis in 1..layout.rank {
        let extent = layout.dims[fused] as isize;
        if layout.dst_strides[fused] * extent == layout.dst_strides[axis]
            && layout.src_strides[fused] * extent == layout.src_strides[axis]
        {
            layout.dims[fused] *= layout.dims[axis];
        } else {
            fused += 1;
            layout.dims[fused] = layout.dims[axis];
            layout.dst_strides[fused] = layout.dst_strides[axis];
            layout.src_strides[fused] = layout.src_strides[axis];
        }
    }
    layout.rank = fused + 1;
    Some(layout)
}

pub(crate) fn apply_fused_pair<D, S, Apply, Op>(
    dst: &mut RawStridedMut<'_, D>,
    src: &RawStridedRef<'_, S>,
    layout: &FusedPairLayout,
    apply: Apply,
    op: Op,
) where
    D: Copy,
    S: Copy,
    Apply: Fn(&mut D, S),
    Op: Fn(S) -> S,
{
    if layout.dims[..layout.rank].iter().any(|&dim| dim == 0) {
        return;
    }
    let inner_len = layout.dims[0];
    let inner_dst = layout.dst_strides[0];
    let inner_src = layout.src_strides[0];
    let src_data = src.data();
    let src_offset = src.offset();
    let dst_offset = dst.offset();
    let dst_data = dst.data_mut();
    let mut index = [0usize; RAW_FUSED_RANK_LIMIT];
    let mut dst_base = dst_offset;
    let mut src_base = src_offset;
    loop {
        if inner_dst == 1 && inner_src == 1 {
            let dst_start = dst_base as usize;
            let src_start = src_base as usize;
            let dst_run = &mut dst_data[dst_start..dst_start + inner_len];
            let src_run = &src_data[src_start..src_start + inner_len];
            for position in 0..inner_len {
                apply(&mut dst_run[position], op(src_run[position]));
            }
        } else {
            for position in 0..inner_len {
                let dst_position = (dst_base + position as isize * inner_dst) as usize;
                let src_position = (src_base + position as isize * inner_src) as usize;
                apply(&mut dst_data[dst_position], op(src_data[src_position]));
            }
        }
        let mut axis = 1;
        loop {
            if axis >= layout.rank {
                return;
            }
            index[axis] += 1;
            dst_base += layout.dst_strides[axis];
            src_base += layout.src_strides[axis];
            if index[axis] < layout.dims[axis] {
                break;
            }
            dst_base -= layout.dims[axis] as isize * layout.dst_strides[axis];
            src_base -= layout.dims[axis] as isize * layout.src_strides[axis];
            index[axis] = 0;
            axis += 1;
        }
    }
}

fn ensure_same_dims(dst: &[usize], src: &[usize]) -> Result<()> {
    if dst != src {
        return Err(crate::StridedError::ShapeMismatch(
            dst.to_vec(),
            src.to_vec(),
        ));
    }
    Ok(())
}

/// `dest = scale * src` over borrowed raw strided layouts.
pub fn copy_scale_raw<T>(
    dest: &mut RawStridedMut<'_, T>,
    src: &RawStridedRef<'_, T>,
    scale: T,
) -> Result<()>
where
    T: Copy + Mul<T, Output = T> + ElementOpApply + MaybeSendSync,
{
    ensure_same_dims(dest.dims(), src.dims())?;
    match fuse_pair_layout(dest.dims(), dest.strides(), src.strides()) {
        Some(layout) => {
            apply_fused_pair(
                dest,
                src,
                &layout,
                |dst, value| *dst = value,
                |value: T| scale * value,
            );
            Ok(())
        }
        None => copy_scale(&mut dest.as_view_mut(), &src.as_view(), scale),
    }
}

/// `dest = scale * conj(src)` over borrowed raw strided layouts.
pub fn copy_scale_conj_raw<T>(
    dest: &mut RawStridedMut<'_, T>,
    src: &RawStridedRef<'_, T>,
    scale: T,
) -> Result<()>
where
    T: Copy + Mul<T, Output = T> + ElementOpApply + MaybeSendSync,
{
    ensure_same_dims(dest.dims(), src.dims())?;
    match fuse_pair_layout(dest.dims(), dest.strides(), src.strides()) {
        Some(layout) => {
            apply_fused_pair(
                dest,
                src,
                &layout,
                |dst, value| *dst = value,
                |value: T| scale * value.conj(),
            );
            Ok(())
        }
        None => copy_scale(&mut dest.as_view_mut(), &src.as_view().conj(), scale),
    }
}

/// `dest = alpha * src + dest` over borrowed raw strided layouts.
pub fn axpy_raw<T>(
    dest: &mut RawStridedMut<'_, T>,
    src: &RawStridedRef<'_, T>,
    alpha: T,
) -> Result<()>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + ElementOpApply + MaybeSendSync,
{
    ensure_same_dims(dest.dims(), src.dims())?;
    match fuse_pair_layout(dest.dims(), dest.strides(), src.strides()) {
        Some(layout) => {
            apply_fused_pair(
                dest,
                src,
                &layout,
                |dst, value| *dst = *dst + value,
                |value: T| alpha * value,
            );
            Ok(())
        }
        None => axpy(&mut dest.as_view_mut(), &src.as_view(), alpha),
    }
}

/// `dest = alpha * conj(src) + dest` over borrowed raw strided layouts.
pub fn axpy_conj_raw<T>(
    dest: &mut RawStridedMut<'_, T>,
    src: &RawStridedRef<'_, T>,
    alpha: T,
) -> Result<()>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + ElementOpApply + MaybeSendSync,
{
    ensure_same_dims(dest.dims(), src.dims())?;
    match fuse_pair_layout(dest.dims(), dest.strides(), src.strides()) {
        Some(layout) => {
            apply_fused_pair(
                dest,
                src,
                &layout,
                |dst, value| *dst = *dst + value,
                |value: T| alpha * value.conj(),
            );
            Ok(())
        }
        None => axpy(&mut dest.as_view_mut(), &src.as_view().conj(), alpha),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StridedView, StridedViewMut};

    fn reference_copy_scale(
        dst: &mut [f64],
        src: &[f64],
        dims: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        scale: f64,
    ) {
        let mut dest_view = StridedViewMut::new(dst, dims, dst_strides, 0).unwrap();
        let src_view: StridedView<'_, f64> = StridedView::new(src, dims, src_strides, 0).unwrap();
        copy_scale(&mut dest_view, &src_view, scale).unwrap();
    }

    #[test]
    fn raw_copy_scale_matches_view_kernel() {
        let dims = [2usize, 3, 2];
        let src_strides = [1isize, 2, 6];
        let dst_strides = [6isize, 2, 1];
        let src: Vec<f64> = (0..12).map(|value| value as f64 - 3.0).collect();
        let mut expected = vec![0.0; 12];
        reference_copy_scale(&mut expected, &src, &dims, &dst_strides, &src_strides, 1.5);

        let mut actual = vec![0.0; 12];
        let mut dest = RawStridedMut::new(&mut actual, &dims, &dst_strides, 0).unwrap();
        let source = RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap();
        copy_scale_raw(&mut dest, &source, 1.5).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn raw_axpy_accumulates() {
        let dims = [4usize];
        let strides = [1isize];
        let src = [1.0f64, 2.0, 3.0, 4.0];
        let mut dst = [10.0f64, 20.0, 30.0, 40.0];
        let mut dest = RawStridedMut::new(&mut dst, &dims, &strides, 0).unwrap();
        let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
        axpy_raw(&mut dest, &source, 2.0).unwrap();

        assert_eq!(dst, [12.0, 24.0, 36.0, 48.0]);
    }

    #[test]
    fn raw_copy_scale_conjugates_complex_sources() {
        use num_complex::Complex64;
        let dims = [2usize];
        let strides = [1isize];
        let src = [Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)];
        let mut dst = [Complex64::new(0.0, 0.0); 2];
        let mut dest = RawStridedMut::new(&mut dst, &dims, &strides, 0).unwrap();
        let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
        copy_scale_conj_raw(&mut dest, &source, Complex64::new(2.0, 0.0)).unwrap();

        assert_eq!(dst[0], Complex64::new(2.0, -4.0));
        assert_eq!(dst[1], Complex64::new(-6.0, -8.0));
    }
}
