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

/// Naive per-element reference: walk logical indices column-major.
fn reference_strided_copy(
    dst: &mut [i64],
    src: &[i64],
    dims: &[usize],
    dst_strides: &[isize],
    dst_offset: isize,
    src_strides: &[isize],
    src_offset: isize,
) {
    let total: usize = dims.iter().product();
    for linear in 0..total {
        let mut rest = linear;
        let mut d = dst_offset;
        let mut s = src_offset;
        for axis in 0..dims.len() {
            let coord = (rest % dims[axis]) as isize;
            rest /= dims[axis];
            d += coord * dst_strides[axis];
            s += coord * src_strides[axis];
        }
        dst[d as usize] = src[s as usize];
    }
}

/// Every split of the logical range into three consecutive pieces must replay
/// exactly the serial result: this covers worker ranges that start and end in
/// the middle of an inner run and of outer axes.
#[test]
fn fused_range_replay_matches_reference_for_every_split() {
    // (dims, dst_strides, dst_offset, src_strides, src_offset, dst_len, src_len)
    type Case = (
        Vec<usize>,
        Vec<isize>,
        isize,
        Vec<isize>,
        isize,
        usize,
        usize,
    );
    let cases: Vec<Case> = vec![
        // contiguous rank 1
        (vec![7], vec![1], 0, vec![1], 0, 7, 7),
        // reversed source (stride -1)
        (vec![7], vec![1], 0, vec![-1], 6, 7, 7),
        // stride-2 source (slice with step)
        (vec![5, 3], vec![1, 5], 0, vec![2, 12], 1, 15, 40),
        // strided destination (transpose)
        (vec![3, 4], vec![4, 1], 0, vec![1, 3], 0, 12, 12),
        // negative outer source stride, reversed middle axis
        (vec![3, 2, 3], vec![1, 3, 6], 2, vec![1, -3, 6], 3, 20, 18),
        // negative strides on every axis
        (
            vec![2, 3, 2],
            vec![1, 2, 6],
            0,
            vec![-1, -2, -6],
            11,
            12,
            12,
        ),
        // stride-0 broadcast source
        (vec![4, 3], vec![1, 4], 0, vec![0, 1], 0, 12, 3),
    ];
    for (dims, dst_strides, dst_offset, src_strides, src_offset, dst_len, src_len) in cases {
        let src: Vec<i64> = (0..src_len as i64).map(|v| v * 10 + 1).collect();
        let mut expected = vec![-1i64; dst_len];
        reference_strided_copy(
            &mut expected,
            &src,
            &dims,
            &dst_strides,
            dst_offset,
            &src_strides,
            src_offset,
        );
        let layout = fuse_pair_layout(&dims, &dst_strides, &src_strides).unwrap();
        let total = fused_total(&layout);
        assert_eq!(total, dims.iter().product::<usize>());

        let mut serial = vec![-1i64; dst_len];
        {
            let mut dest =
                RawStridedMut::new(&mut serial, &dims, &dst_strides, dst_offset).unwrap();
            let source = RawStridedRef::new(&src, &dims, &src_strides, src_offset).unwrap();
            apply_fused_pair(&mut dest, &source, &layout, |d, v| *d = v, |v| v);
        }
        assert_eq!(serial, expected, "serial replay for dims {dims:?}");

        for first in 0..=total {
            for second in first..=total {
                let mut actual = vec![-1i64; dst_len];
                for (start, end) in [(0, first), (first, second), (second, total)] {
                    // SAFETY: the offsets come from validated raw views of the
                    // same layout (checked by `RawStridedMut::new` above).
                    unsafe {
                        apply_fused_range(
                            actual.as_mut_ptr(),
                            dst_offset,
                            src.as_ptr(),
                            src_offset,
                            &layout,
                            start,
                            end - start,
                            &|d: &mut i64, v| *d = v,
                            &|v| v,
                        );
                    }
                }
                assert_eq!(actual, expected, "dims {dims:?} split at {first}, {second}");
            }
        }
    }
}
