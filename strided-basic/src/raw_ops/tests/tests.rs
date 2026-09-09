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
