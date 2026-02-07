#![cfg(feature = "blas-inject")]

use strided_einsum2::einsum2_into;
use strided_view::StridedArray;

#[test]
fn test_blas_inject_works_without_manual_registration() {
    let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
        [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
    });
    let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
        [[5.0, 6.0], [7.0, 8.0]][idx[0]][idx[1]]
    });
    let mut c = StridedArray::<f64>::row_major(&[2, 2]);

    einsum2_into(
        c.view_mut(),
        &a.view(),
        &b.view(),
        &['i', 'k'],
        &['i', 'j'],
        &['j', 'k'],
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(c.get(&[0, 0]), 19.0);
    assert_eq!(c.get(&[0, 1]), 22.0);
    assert_eq!(c.get(&[1, 0]), 43.0);
    assert_eq!(c.get(&[1, 1]), 50.0);
}
