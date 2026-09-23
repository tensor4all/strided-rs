//! Regression tests for issue #263: legal stride-0 broadcast layouts whose
//! fused extent overflows `usize` must return a typed error from the public
//! permutation copy entry points instead of panicking inside the planner.

use strided_perm::copy_into;
use strided_view::{StridedError, StridedView, StridedViewMut};

const HUGE: usize = 1 << 40;

#[test]
fn copy_into_reports_overflowing_broadcast_extent() {
    let src_data = [1.0f64];
    let mut dst_data = [0.0f64];
    let src = StridedView::new(&src_data, &[HUGE, HUGE], &[0, 0], 0).unwrap();
    let mut dst = StridedViewMut::new(&mut dst_data, &[HUGE, HUGE], &[0, 0], 0).unwrap();

    assert!(matches!(
        copy_into(&mut dst, &src),
        Err(StridedError::OffsetOverflow)
    ));
    assert_eq!(dst_data, [0.0], "destination must be untouched on error");
}

#[test]
fn copy_into_reports_extent_beyond_isize() {
    // A rank-1 stride-0 view accepts `dim - 1 <= isize::MAX`, so `1 << 63`
    // is constructible but cannot take part in isize stride arithmetic.
    let src_data = [1.0f64];
    let mut dst_data = [0.0f64];
    let src = StridedView::new(&src_data, &[1usize << 63], &[0], 0).unwrap();
    let mut dst = StridedViewMut::new(&mut dst_data, &[1usize << 63], &[0], 0).unwrap();

    assert!(matches!(
        copy_into(&mut dst, &src),
        Err(StridedError::OffsetOverflow)
    ));
}

#[test]
fn copy_into_broadcast_source_still_copies() {
    // Ordinary broadcast inputs keep working on the planned path.
    let src_data = [7.0f64, 8.0];
    let src = StridedView::new(&src_data, &[2, 3], &[1, 0], 0).unwrap();
    let mut out = [0.0f64; 6];
    copy_into(
        &mut StridedViewMut::new(&mut out, &[2, 3], &[3, 1], 0).unwrap(),
        &src,
    )
    .unwrap();
    assert_eq!(out, [7.0, 7.0, 7.0, 8.0, 8.0, 8.0]);
}

#[cfg(feature = "parallel")]
#[test]
fn copy_into_par_reports_overflowing_broadcast_extent() {
    let src_data = [1.0f64];
    let mut dst_data = [0.0f64];
    let src = StridedView::new(&src_data, &[HUGE, HUGE], &[0, 0], 0).unwrap();
    let mut dst = StridedViewMut::new(&mut dst_data, &[HUGE, HUGE], &[0, 0], 0).unwrap();

    assert!(matches!(
        strided_perm::copy_into_par(&mut dst, &src),
        Err(StridedError::OffsetOverflow)
    ));
    assert_eq!(dst_data, [0.0]);
}
