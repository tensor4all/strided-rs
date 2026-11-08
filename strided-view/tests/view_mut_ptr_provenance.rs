//! Regression tests for the pointer provenance of `StridedViewMut`.
//!
//! `data_mut()` reborrows the backing slice. Any pointer the view derived from
//! an earlier borrow of that slice loses write provenance at that moment, so a
//! view must not cache a pointer that `data_mut()` invalidates. These tests run
//! the access sequences that used to be UB under Stacked Borrows and Tree
//! Borrows; run them with
//! `cargo +nightly miri test -p strided-view --test view_mut_ptr_provenance`
//! (`MIRIFLAGS=-Zmiri-tree-borrows` for the Tree Borrows flavour).

use strided_view::StridedViewMut;

#[test]
fn get_set_after_data_mut() {
    let mut data = [1.0f64, 2.0, 3.0, 4.0];
    let mut view = StridedViewMut::new(&mut data, &[2, 2], &[1, 2], 0).unwrap();

    let _ = view.data_mut();

    view.set(&[1, 1], 5.0);
    assert_eq!(view.get(&[1, 1]), 5.0);
    assert_eq!(view.get(&[0, 1]), 3.0);
    assert_eq!(data[3], 5.0);
}

#[test]
fn read_only_after_data_mut() {
    let mut data = [1.0f64, 2.0, 3.0, 4.0];
    let mut view = StridedViewMut::new(&mut data, &[4], &[1], 0).unwrap();

    let _ = view.data_mut();

    assert_eq!(view.get(&[2]), 3.0);
}

#[test]
fn as_view_get_after_data_mut() {
    let mut data = [1.0f64, 2.0, 3.0, 4.0];
    let mut view = StridedViewMut::new(&mut data, &[2, 2], &[1, 2], 0).unwrap();

    let _ = view.data_mut();

    assert_eq!(view.as_view().get(&[1, 0]), 2.0);
}

#[test]
fn set_as_mut_ptr_get_interleaved() {
    let mut data = [0.0f64; 6];
    let mut view = StridedViewMut::new(&mut data, &[2, 3], &[1, 2], 0).unwrap();

    for i in 0..2 {
        for j in 0..3 {
            view.set(&[i, j], (i * 3 + j) as f64);
            let _ = view.data_mut();
            let _ = view.as_view().get(&[i, j]);
        }
    }

    // The raw pointer must still be writable after the slice reborrows.
    let ptr = view.as_mut_ptr();
    unsafe { *ptr = 42.0 };
    assert_eq!(view.get(&[0, 0]), 42.0);
}

#[test]
fn offset_view_round_trip_through_data_mut() {
    let mut data = [10.0f64, 11.0, 12.0, 13.0];
    let mut view = StridedViewMut::new(&mut data, &[2], &[1], 1).unwrap();

    assert_eq!(view.data().len(), 4);
    let seen = view.data_mut();
    seen[0] = 0.0;
    assert_eq!(seen, &[0.0, 11.0, 12.0, 13.0][..]);

    view.set(&[1], 21.0);
    assert_eq!(view.get(&[0]), 11.0);
    assert_eq!(view.get(&[1]), 21.0);
    assert_eq!(data, [0.0, 11.0, 21.0, 13.0]);
}

#[test]
fn issue_383_reported_repro() {
    let mut a = [0i32];
    let mut v = StridedViewMut::new(&mut a, &[1], &[1], 0).unwrap();
    v.data_mut()[0] = 1;
    v.set(&[0], 7);
    assert_eq!(v.get(&[0]), 7);
    assert_eq!(a, [7]);
}

/// A zero-extent shape does not shrink the backing slice: `data`/`data_mut`
/// report the caller's buffer, which is a different thing from the shape.
#[test]
fn zero_element_shape_keeps_the_backing_slice() {
    let mut a = [1i32, 2];
    let mut v = StridedViewMut::new(&mut a, &[0], &[1], 0).unwrap();

    assert_eq!(v.data(), &[1, 2]);
    v.data_mut()[0] = 3;
    assert_eq!(v.as_view().data(), &[3, 2]);
    assert_eq!(a, [3, 2]);
}
