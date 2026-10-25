//! Regression for tensor4all/strided-rs#207: reducing an axis whose extent is
//! zero must not form a source pointer from a dangling base.
//!
//! The fold never reads an empty reduced axis, so every kept position is the
//! initial value, and no source pointer is derived.

use strided_basic::Exec;
use strided_basic::*;

#[test]
fn reduce_axis_with_an_empty_reduced_axis_returns_initial_values() {
    let data: [f64; 0] = [];
    let dims = [0usize, 3];
    let strides = [3isize, 1];
    let src: StridedView<'_, f64> = StridedView::new(&data, &dims, &strides, 0).unwrap();

    let out = reduce_axis(&src, 0, |x| x, |a, b| a + b, 7.0, Exec::serial()).unwrap();
    assert_eq!(out.dims(), &[3]);
    assert_eq!(out.into_data(), vec![7.0, 7.0, 7.0]);
}

#[test]
fn reduce_axis_with_an_empty_reduced_axis_removed_entirely_returns_the_init() {
    let data: [f64; 0] = [];
    let dims = [0usize];
    let strides = [1isize];
    let src: StridedView<'_, f64> = StridedView::new(&data, &dims, &strides, 0).unwrap();

    let out = reduce_axis(&src, 0, |x| x, |a, b| a + b, 7.0, Exec::serial()).unwrap();
    assert_eq!(out.dims(), &[1]);
    assert_eq!(out.into_data(), vec![7.0]);
}

#[test]
fn reduce_axis_over_a_zero_extent_kept_axis_returns_an_empty_output() {
    let data: [f64; 0] = [];
    let dims = [0usize, 3];
    let strides = [3isize, 1];
    let src: StridedView<'_, f64> = StridedView::new(&data, &dims, &strides, 0).unwrap();

    let out = reduce_axis(&src, 1, |x| x, |a, b| a + b, 7.0, Exec::serial()).unwrap();
    assert_eq!(out.dims(), &[0]);
    assert_eq!(out.into_data(), Vec::<f64>::new());
}
