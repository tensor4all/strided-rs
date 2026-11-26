//! Issue #206: elementwise read-modify-write ops must not run on a destination
//! where two logical indices name one element. The result would depend on write
//! order, and under a threaded `Exec` on a data race. The contract is a
//! `NonInjectiveOutputLayout` error before the first write, identically for a
//! serial and a pooled token, every time.

mod common;

use common::Budget;
use strided_basic::{
    add_acc, axpy_acc, copy_transpose_scale_into, fma_acc, mul_acc, StridedError, StridedView,
    StridedViewMut,
};

const N: usize = 1 << 9;
const REPEATS: usize = 20;

/// Destination with a stride-0 axis of extent `N`: `N * N` indices, `N` elements.
fn aliased_dest(data: &mut [f64]) -> StridedViewMut<'_, f64> {
    StridedViewMut::new(data, &[N, N], &[1, 0], 0).unwrap()
}

fn check(op: impl Fn(&mut StridedViewMut<f64>, Budget) -> Result<(), StridedError>) {
    for budget in [Budget::Serial, Budget::Pooled(4)] {
        for _ in 0..REPEATS {
            let mut data = vec![1.0f64; N];
            let before = data.clone();
            let res = op(&mut aliased_dest(&mut data), budget);
            assert!(matches!(res, Err(StridedError::NonInjectiveOutputLayout)));
            assert_eq!(data, before, "a rejected call must not write");
        }
    }
}

#[test]
fn aliased_destination_is_rejected_by_every_rmw_op() {
    let src = vec![2.0f64; N * N];
    let sv = StridedView::<f64>::new(&src, &[N, N], &[1, N as isize], 0).unwrap();
    check(|d, b| add_acc(d, &sv, b.exec()));
    check(|d, b| mul_acc(d, &sv, b.exec()));
    check(|d, b| axpy_acc(d, &sv, 1.5, b.exec()));
    check(|d, b| fma_acc(d, &sv, &sv, b.exec()));
}

#[test]
fn aliased_destination_is_rejected_by_copy_transpose_scale() {
    let src = vec![2.0f64; N * N];
    let sv = StridedView::<f64>::new(&src, &[N, N], &[1, N as isize], 0).unwrap();
    for scale in [0.0, 1.0, 3.0] {
        check(|d, b| copy_transpose_scale_into(d, &sv, scale, b.exec()));
    }
}

#[test]
fn injective_destination_gives_the_same_answer_serial_and_threaded() {
    let src: Vec<f64> = (0..N * N).map(|i| (i % 13) as f64).collect();
    let sv = StridedView::<f64>::new(&src, &[N, N], &[1, N as isize], 0).unwrap();
    let run = |b: Budget| {
        let mut data = vec![1.0f64; N * N];
        let mut d = StridedViewMut::new(&mut data, &[N, N], &[N as isize, 1], 0).unwrap();
        axpy_acc(&mut d, &sv, 2.0, b.exec()).unwrap();
        data
    };
    let want = run(Budget::Serial);
    for _ in 0..REPEATS {
        assert_eq!(run(Budget::Pooled(4)), want);
        assert_eq!(run(Budget::Serial), want);
    }
}
