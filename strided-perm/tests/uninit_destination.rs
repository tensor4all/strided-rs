//! `copy_into_uninit` writes the same elements as `copy_into`, on the native
//! float path (the destination type no longer hides it) and on the generic one.

use core::mem::MaybeUninit;
use strided_perm::{copy_into, copy_into_uninit};
use strided_view::{StridedView, StridedViewMut};

fn transpose_case<T: Copy + PartialEq + core::fmt::Debug + Send + Sync + 'static>(
    rows: usize,
    cols: usize,
    lift: impl Fn(usize) -> T + Copy,
) {
    let data: Vec<T> = (0..rows * cols).map(lift).collect();
    let src = StridedView::new(&data, &[rows, cols], &[1, rows as isize], 0)
        .unwrap()
        .permute(&[1, 0])
        .unwrap();
    let mut want = vec![lift(0); rows * cols];
    let mut want_view =
        StridedViewMut::new(&mut want, &[cols, rows], &[1, cols as isize], 0).unwrap();
    copy_into(&mut want_view, &src).unwrap();

    let mut got: Vec<MaybeUninit<T>> = vec![MaybeUninit::uninit(); rows * cols];
    let mut got_view =
        StridedViewMut::new(&mut got, &[cols, rows], &[1, cols as isize], 0).unwrap();
    copy_into_uninit(&mut got_view, &src).unwrap();
    // SAFETY: the copy wrote every destination element.
    let got: Vec<T> = got
        .into_iter()
        .map(|x| unsafe { x.assume_init() })
        .collect();
    assert_eq!(got, want);

    #[cfg(feature = "parallel")]
    {
        let mut par: Vec<MaybeUninit<T>> = vec![MaybeUninit::uninit(); rows * cols];
        let mut par_view =
            StridedViewMut::new(&mut par, &[cols, rows], &[1, cols as isize], 0).unwrap();
        strided_perm::copy_into_uninit_par(&mut par_view, &src).unwrap();
        // SAFETY: as above.
        let par: Vec<T> = par
            .into_iter()
            .map(|x| unsafe { x.assume_init() })
            .collect();
        assert_eq!(par, want);
    }
}

#[test]
fn f64_transpose_matches_the_initialized_copy() {
    // Odd extents exercise the micro-kernel remainders; the size clears the
    // parallel threshold.
    transpose_case(301, 257, |i| i as f64 * 0.5);
}

#[test]
fn f32_transpose_matches_the_initialized_copy() {
    transpose_case(203, 129, |i| i as f32);
}

#[test]
fn generic_element_transpose_matches_the_initialized_copy() {
    transpose_case(67, 41, |i| (i as u16, i as u8));
}

#[test]
fn empty_and_mismatched_shapes_behave_like_copy_into() {
    let data: Vec<f64> = Vec::new();
    let src = StridedView::new(&data, &[0, 3], &[1, 0], 0).unwrap();
    let mut got: Vec<MaybeUninit<f64>> = Vec::new();
    let mut view = StridedViewMut::new(&mut got, &[0, 3], &[1, 0], 0).unwrap();
    copy_into_uninit(&mut view, &src).unwrap();

    let data = [1.0f64, 2.0, 3.0, 4.0];
    let src = StridedView::new(&data, &[2, 2], &[1, 2], 0).unwrap();
    let mut got = [MaybeUninit::<f64>::uninit(); 4];
    let mut view = StridedViewMut::new(&mut got, &[4], &[1], 0).unwrap();
    assert!(copy_into_uninit(&mut view, &src).is_err());
}
