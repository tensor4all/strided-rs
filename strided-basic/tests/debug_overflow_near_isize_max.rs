//! Traversal must never form an offset outside the layout's reachable range.
//!
//! With a zero-sized element type a valid layout may place its last element
//! within one stride of `isize::MAX`. An odometer that steps one stride past
//! each finished axis before rewinding overflows `isize` there, which panics
//! in debug builds (follow-up of issue #264).

use std::ops::{Add, Mul};
use strided_basic::*;

/// Zero-sized numeric element, so layouts can span close to `isize::MAX`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Z;

impl Add for Z {
    type Output = Z;
    fn add(self, _: Z) -> Z {
        Z
    }
}

impl Mul for Z {
    type Output = Z;
    fn mul(self, _: Z) -> Z {
        Z
    }
}

impl ElementOpApply for Z {}

struct Case {
    dims: Vec<usize>,
    dst_strides: Vec<isize>,
    src_strides: Vec<isize>,
    dst_offset: isize,
    src_offset: isize,
}

fn cases() -> Vec<Case> {
    let m = isize::MAX;
    let mut strides10: Vec<isize> = (0..10).map(|k| 1isize << k).collect();
    strides10[9] = m - 1023;
    let reversed10: Vec<isize> = strides10.iter().rev().copied().collect();
    vec![
        Case {
            dims: vec![2, 2],
            dst_strides: vec![1, m - 1],
            src_strides: vec![m - 1, 1],
            dst_offset: 0,
            src_offset: 0,
        },
        Case {
            dims: vec![2, 2],
            dst_strides: vec![-1, -(m - 1)],
            src_strides: vec![-(m - 1), -1],
            dst_offset: m,
            src_offset: m,
        },
        Case {
            dims: vec![2, 2, 2],
            dst_strides: vec![1, 2, m - 3],
            src_strides: vec![m - 3, 2, 1],
            dst_offset: 0,
            src_offset: 0,
        },
        Case {
            dims: vec![2; 10],
            dst_strides: strides10,
            src_strides: reversed10,
            dst_offset: 0,
            src_offset: 0,
        },
    ]
}

/// A `usize::MAX` long buffer of `Z` without cloning `usize::MAX` times,
/// which would take forever in a debug build.
fn huge_buffer() -> Vec<Z> {
    let mut v = Vec::with_capacity(usize::MAX);
    // SAFETY: `Z` is zero-sized with no invalid bit patterns, so a zero-sized
    // allocation already has capacity `usize::MAX` and every element is valid.
    unsafe { v.set_len(usize::MAX) };
    v
}

fn element_count(dims: &[usize]) -> u64 {
    dims.iter().product::<usize>() as u64
}

#[test]
fn typed_ops_on_near_max_layouts_do_not_overflow() {
    let src_data = huge_buffer();
    let mut dst_data = huge_buffer();
    for c in cases() {
        let src = || StridedView::new(&src_data, &c.dims, &c.src_strides, c.src_offset).unwrap();
        let mut dst =
            StridedViewMut::new(&mut dst_data, &c.dims, &c.dst_strides, c.dst_offset).unwrap();
        copy_into(&mut dst, &src()).unwrap();
        copy_into_col_major(&mut dst, &src()).unwrap();
        map_into(&mut dst, &src(), |x| x).unwrap();
        zip_map2_into(&mut dst, &src(), &src(), |x, y| x + y).unwrap();
        add(&mut dst, &src()).unwrap();
        axpy(&mut dst, &src(), Z).unwrap();
        let count = reduce(&src(), |_| 1u64, |a, b| a + b, 0u64).unwrap();
        assert_eq!(count, element_count(&c.dims));
        reduce_axis(&src(), 0, |_| 1u64, |a, b| a + b, 0u64).unwrap();
    }
}

#[test]
fn raw_ops_and_copy_plan_on_near_max_layouts_do_not_overflow() {
    let src_data = huge_buffer();
    let mut dst_data = huge_buffer();
    for c in cases() {
        let src = RawStridedRef::new(&src_data[..], &c.dims, &c.src_strides, c.src_offset).unwrap();
        let mut dst =
            RawStridedMut::new(&mut dst_data[..], &c.dims, &c.dst_strides, c.dst_offset).unwrap();
        copy_scale_raw(&mut dst, &src, Z).unwrap();
        axpy_raw(&mut dst, &src, Z).unwrap();
        CopyPlan::compile(&c.dims, &c.dst_strides, &c.src_strides)
            .unwrap()
            .execute(&mut dst, &src)
            .unwrap();
    }
}
