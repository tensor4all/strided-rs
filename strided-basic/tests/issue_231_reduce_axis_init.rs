//! Regression test for issue #231: the contiguous fast path of `reduce_axis`
//! overwrote the output with the first mapped element and dropped `init`.

use strided_basic::{reduce_axis, StridedArray};

fn expected(dims: &[usize], f: impl Fn(&[usize]) -> f64, init: f64) -> Vec<f64> {
    // Reduce over axis 0 of a rank-2 array; output is indexed by axis 1.
    (0..dims[1])
        .map(|j| (0..dims[0]).fold(init, |acc, i| acc + f(&[i, j])))
        .collect()
}

fn value(idx: &[usize]) -> f64 {
    (idx[0] * 10 + idx[1]) as f64 + 1.0
}

fn check(dims: &[usize], a: &StridedArray<f64>, init: f64) {
    let out = reduce_axis(&a.view(), 0, |x| x, |x, y| x + y, init).unwrap();
    let want = expected(dims, value, init);
    let got: Vec<f64> = (0..dims[1]).map(|j| out.get(&[j])).collect();
    assert_eq!(got, want, "dims={dims:?} init={init}");
}

#[test]
fn issue_231_fast_path_row_major_keeps_init() {
    // Row-major: the kept axis has stride 1, which selects the fast path.
    for dims in [[2usize, 3], [5, 17], [1, 4]] {
        let a = StridedArray::<f64>::from_fn_row_major(&dims, value);
        check(&dims, &a, 100.0);
        check(&dims, &a, 0.0);
    }
}

#[test]
fn issue_231_general_path_col_major_keeps_init() {
    // Column-major: the kept axis is strided, which selects the general path.
    for dims in [[2usize, 3], [5, 17], [1, 4]] {
        let a = StridedArray::<f64>::from_fn_col_major(&dims, value);
        check(&dims, &a, 100.0);
    }
}

#[test]
fn issue_231_fast_path_non_additive_init() {
    // A max reduction with an init larger than every element must return init.
    let dims = [3usize, 8];
    let a = StridedArray::<f64>::from_fn_row_major(&dims, value);
    let out = reduce_axis(&a.view(), 0, |x| x, f64::max, 1.0e9).unwrap();
    for j in 0..dims[1] {
        assert_eq!(out.get(&[j]), 1.0e9);
    }
}
