//! Element counts whose product overflows `usize` must be reported as typed
//! errors (or handled as empty) instead of panicking in debug builds.
//!
//! `StridedView` accepts a stride-0 broadcast of any extent, and any extents
//! at all when one axis is zero, so `dims.iter().product()` can overflow
//! before it reaches the zero (follow-up of issue #263).

use std::mem::MaybeUninit;
use strided_basic::*;

const H: usize = 1 << 40;
static SRC: [f64; 4] = [1.0; 4];

fn src(dims: &'static [usize], strides: &'static [isize]) -> StridedView<'static, f64> {
    StridedView::new(&SRC, dims, strides, 0).unwrap()
}

#[test]
fn broadcast_reductions_with_overflowing_count_return_error() {
    let dims: &[usize] = &[H, H];
    let strides: &[isize] = &[0, 0];
    assert!(sum(&src(dims, strides)).is_err());
    assert!(dot::<f64, f64, f64, _, _>(&src(dims, strides), &src(dims, strides)).is_err());
    assert!(reduce(&src(dims, strides), |x| x, |a, b| a + b, 0.0).is_err());
    assert!(reduce_axis(&src(dims, strides), 0, |x| x, |a, b| a + b, 0.0).is_err());
}

#[test]
fn broadcast_updates_with_overflowing_count_return_error() {
    let dims: &[usize] = &[H, H];
    let strides: &[isize] = &[0, 0];
    let mut buf = [0.0f64; 4];
    let mut dst = StridedViewMut::new(&mut buf, dims, strides, 0).unwrap();
    assert!(copy_into(&mut dst, &src(dims, strides)).is_err());
    assert!(copy_conj(&mut dst, &src(dims, strides)).is_err());
    assert!(copy_scale(&mut dst, &src(dims, strides), 2.0).is_err());
    assert!(copy_into_col_major(&mut dst, &src(dims, strides)).is_err());
    assert!(copy_transpose_scale_into(&mut dst, &src(dims, strides), 2.0).is_err());
    assert!(map_into(&mut dst, &src(dims, strides), |x| x).is_err());
    assert!(zip_map2_into(
        &mut dst,
        &src(dims, strides),
        &src(dims, strides),
        |x, y| x + y
    )
    .is_err());
    assert!(add(&mut dst, &src(dims, strides)).is_err());
    assert!(mul(&mut dst, &src(dims, strides)).is_err());
    assert!(axpy(&mut dst, &src(dims, strides), 2.0).is_err());
    assert!(fma(&mut dst, &src(dims, strides), &src(dims, strides)).is_err());
}

#[test]
fn empty_layouts_with_overflowing_extents_are_noops() {
    // Zero trailing axis: nothing to visit, whatever the other extents are.
    let dims: &[usize] = &[H, H, 0];
    for (dst_strides, src_strides) in [
        (&[0isize, 0, 1][..], &[0isize, 0, 1][..]),
        (&[1, 1, 1][..], &[0, 0, 0][..]),
    ] {
        let dst_strides: &'static [isize] = Box::leak(dst_strides.to_vec().into_boxed_slice());
        let src_strides: &'static [isize] = Box::leak(src_strides.to_vec().into_boxed_slice());
        let s = || src(dims, src_strides);
        let mut buf = [0.0f64; 4];
        let mut dst = StridedViewMut::new(&mut buf, dims, dst_strides, 0).unwrap();
        copy_into(&mut dst, &s()).unwrap();
        copy_conj(&mut dst, &s()).unwrap();
        copy_scale(&mut dst, &s(), 2.0).unwrap();
        copy_into_col_major(&mut dst, &s()).unwrap();
        map_into(&mut dst, &s(), |x| x).unwrap();
        zip_map2_into(&mut dst, &s(), &s(), |x, y| x + y).unwrap();
        zip_map3_into(&mut dst, &s(), &s(), &s(), |x, y, z| x + y + z).unwrap();
        mul_into(&mut dst, &s(), &s()).unwrap();
        add(&mut dst, &s()).unwrap();
        mul(&mut dst, &s()).unwrap();
        axpy(&mut dst, &s(), 2.0).unwrap();
        fma(&mut dst, &s(), &s()).unwrap();
        assert_eq!(buf, [0.0; 4]);

        let mut ubuf = [MaybeUninit::<f64>::uninit(); 4];
        let mut udst = StridedViewMut::new(&mut ubuf, dims, dst_strides, 0).unwrap();
        copy_into_uninit(&mut udst, &s()).unwrap();

        assert_eq!(sum(&s()).unwrap(), 0.0);
        assert_eq!(dot::<f64, f64, f64, _, _>(&s(), &s()).unwrap(), 0.0);
        assert_eq!(reduce(&s(), |x| x, |a, b| a + b, 0.0).unwrap(), 0.0);
    }
}

#[test]
fn raw_ops_and_copy_plan_reject_overflowing_count() {
    let dims: &[usize] = &[H, H];
    let strides: &[isize] = &[0, 0];
    let mut buf = [0.0f64; 4];
    let mut dst = RawStridedMut::new(&mut buf[..], dims, strides, 0).unwrap();
    let s = RawStridedRef::new(&SRC[..], dims, strides, 0).unwrap();
    assert!(copy_scale_raw(&mut dst, &s, 2.0).is_err());
    assert!(copy_scale_conj_raw(&mut dst, &s, 2.0).is_err());
    assert!(axpy_raw(&mut dst, &s, 2.0).is_err());
    assert!(axpy_conj_raw(&mut dst, &s, 2.0).is_err());
    assert!(CopyPlan::compile(dims, strides, strides).is_err());

    let dims: &[usize] = &[H, H, 0];
    let plan = CopyPlan::compile(dims, &[1, 1, 1], &[0, 0, 0]).unwrap();
    let mut dst = RawStridedMut::new(&mut buf[..], dims, &[1, 1, 1], 0).unwrap();
    let s = RawStridedRef::new(&SRC[..], dims, &[0, 0, 0], 0).unwrap();
    plan.execute(&mut dst, &s).unwrap();
}
