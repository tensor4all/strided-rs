use std::mem::MaybeUninit;

use strided_basic::{
    add, axpy, dot, fma, map_into, map_update_into, mul, mul_into, mul_into_uninit, zip_map2_into,
    zip_map3_into, zip_map4_into, zip_update2_into, zip_update3_into, Identity, StridedView,
    StridedViewMut,
};

fn values(output: &[MaybeUninit<f64>], indices: &[usize]) -> Vec<f64> {
    indices
        .iter()
        .map(|&index| {
            // SAFETY: the pointwise operation writes every logical destination element.
            unsafe { output[index].assume_init() }
        })
        .collect()
}

#[test]
fn pointwise_leaves_may_advance_past_the_owned_storage() {
    // Reversed unary input/output exercises negative inner strides.
    let source = [1.0, 2.0, 3.0];
    let mut output = [0.0; 3];
    let source = StridedView::<f64>::new(&source, &[3], &[-1], 2).unwrap();
    let mut output = StridedViewMut::new(&mut output, &[3], &[-1], 2).unwrap();
    map_into(&mut output, &source, |x| x + 10.0).unwrap();
    assert_eq!(output.data(), &[11.0, 12.0, 13.0]);

    // Positive gaps end exactly at the allocation boundary; the broadcast
    // operand is intentionally represented with stride zero.
    let a_data = [1.0, -1.0, 2.0, -1.0, 3.0];
    let b_data = [10.0];
    let mut binary_data = [0.0; 5];
    let a = StridedView::<f64>::new(&a_data, &[3], &[2], 0).unwrap();
    let b = StridedView::<f64>::new(&b_data, &[3], &[0], 0).unwrap();
    let mut binary = StridedViewMut::new(&mut binary_data, &[3], &[2], 0).unwrap();
    zip_map2_into(&mut binary, &a, &b, |x, y| x + y).unwrap();
    assert_eq!(binary_data, [11.0, 0.0, 12.0, 0.0, 13.0]);
    for (lhs, rhs) in [(&a, &b), (&b, &a)] {
        let mut contiguous = [0.0; 3];
        let mut view = StridedViewMut::new(&mut contiguous, &[3], &[1], 0).unwrap();
        zip_map2_into(&mut view, lhs, rhs, |x, y| x + y).unwrap();
        assert_eq!(contiguous, [11.0, 12.0, 13.0]);
        let mut fresh = [MaybeUninit::uninit(); 3];
        let mut view = StridedViewMut::new(&mut fresh, &[3], &[1], 0).unwrap();
        mul_into_uninit(&mut view, lhs, rhs).unwrap();
        assert_eq!(values(&fresh, &[0, 1, 2]), [10.0, 20.0, 30.0]);
    }

    // Ternary and initialized multiplication use reversed inputs/output.
    let one_data = [1.0, 2.0, 3.0];
    let two_data = [4.0, 5.0, 6.0];
    let three_data = [7.0, 8.0, 9.0];
    let one = StridedView::<f64>::new(&one_data, &[3], &[-1], 2).unwrap();
    let two = StridedView::<f64>::new(&two_data, &[3], &[-1], 2).unwrap();
    let three = StridedView::<f64>::new(&three_data, &[3], &[-1], 2).unwrap();
    let mut ternary_data = [0.0; 3];
    let mut ternary = StridedViewMut::new(&mut ternary_data, &[3], &[-1], 2).unwrap();
    zip_map3_into(&mut ternary, &one, &two, &three, |x, y, z| x + y + z).unwrap();
    assert_eq!(ternary_data, [12.0, 15.0, 18.0]);

    let mut product_data = [0.0; 5];
    let mut product = StridedViewMut::new(&mut product_data, &[3], &[2], 0).unwrap();
    mul_into(&mut product, &a, &b).unwrap();
    assert_eq!(product_data, [10.0, 0.0, 20.0, 0.0, 30.0]);

    let mut product_uninit = vec![MaybeUninit::uninit(); 5];
    let mut product_uninit_view = StridedViewMut::new(&mut product_uninit, &[3], &[2], 0).unwrap();
    mul_into_uninit(&mut product_uninit_view, &a, &b).unwrap();
    assert_eq!(values(&product_uninit, &[0, 2, 4]), [10.0, 20.0, 30.0]);

    // Update siblings use the same non-contiguous pointwise leaves.
    let mut add_data = [1.0, 2.0, 3.0];
    let mut add_view = StridedViewMut::new(&mut add_data, &[3], &[-1], 2).unwrap();
    add(&mut add_view, &one).unwrap();
    assert_eq!(add_data, [2.0, 4.0, 6.0]);

    let mut mul_data = [2.0, 3.0, 4.0];
    let mut mul_view = StridedViewMut::new(&mut mul_data, &[3], &[-1], 2).unwrap();
    mul(&mut mul_view, &two).unwrap();
    assert_eq!(mul_data, [8.0, 15.0, 24.0]);

    let mut axpy_data = [1.0, 1.0, 1.0];
    let mut axpy_view = StridedViewMut::new(&mut axpy_data, &[3], &[-1], 2).unwrap();
    axpy(&mut axpy_view, &three, 2.0).unwrap();
    assert_eq!(axpy_data, [15.0, 17.0, 19.0]);

    let mut fma_data = [1.0, 1.0, 1.0];
    let mut fma_view = StridedViewMut::new(&mut fma_data, &[3], &[-1], 2).unwrap();
    fma(&mut fma_view, &one, &two).unwrap();
    assert_eq!(fma_data, [5.0, 11.0, 19.0]);
    assert_eq!(dot(&one, &two).unwrap(), 32.0);

    let mut fourth_data = [0.0; 3];
    let mut fourth = StridedViewMut::new(&mut fourth_data, &[3], &[-1], 2).unwrap();
    zip_map4_into(&mut fourth, &one, &two, &three, &one, |w, x, y, z| {
        w + x + y + z
    })
    .unwrap();
    assert_eq!(fourth_data, [13.0, 17.0, 21.0]);

    let mut update_data = [1.0; 3];
    let mut update = StridedViewMut::new(&mut update_data, &[3], &[-1], 2).unwrap();
    map_update_into::<_, Identity>(&mut update, |old| old * 2.0).unwrap();
    zip_update2_into::<_, _, Identity, _>(&mut update, &one, |old, x| old + x).unwrap();
    zip_update3_into::<_, _, _, Identity, _, _>(&mut update, &one, &two, |old, x, y| old + x * y)
        .unwrap();
    assert_eq!(update_data, [7.0, 14.0, 23.0]);
}
