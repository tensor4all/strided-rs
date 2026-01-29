use approx::assert_relative_eq;
use strided_rs::{
    copy_into, copy_transpose_scale_into, dot, map_into, reduce, reduce_axis, symmetrize_into,
    zip_map2_into, zip_map4_into, StridedArray,
};

fn make_tensor(rows: usize, cols: usize) -> StridedArray<f64> {
    StridedArray::from_fn_row_major(&[rows, cols], |idx| (idx[0] * cols + idx[1]) as f64)
}

#[test]
fn test_map_into_transposed() {
    let a = make_tensor(8, 5);
    let a_view = a.view();
    let a_t = a_view.permute(&[1, 0]).unwrap();

    let mut out = StridedArray::<f64>::row_major(&[5, 8]);
    map_into(&mut out.view_mut(), &a_t, |x| x * 2.0).unwrap();

    for i in 0..5 {
        for j in 0..8 {
            let expected = a.get(&[j, i]) * 2.0;
            assert_relative_eq!(out.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_zip_map2_mixed_strides() {
    let a = make_tensor(6, 4);
    let b = make_tensor(6, 4);
    let a_view = a.view();
    let b_view = b.view();
    let a_t = a_view.permute(&[1, 0]).unwrap();
    let b_t = b_view.permute(&[1, 0]).unwrap();

    let mut out = StridedArray::<f64>::row_major(&[4, 6]);
    zip_map2_into(&mut out.view_mut(), &a_t, &b_t, |x, y| x + y).unwrap();

    for i in 0..4 {
        for j in 0..6 {
            let expected = a.get(&[j, i]) + b.get(&[j, i]);
            assert_relative_eq!(out.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_reduce_sum() {
    let a = make_tensor(10, 12);
    let result = reduce(&a.view(), |x| x, |a, b| a + b, 0.0).unwrap();
    let expected: f64 = a.iter().copied().sum();
    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

#[test]
fn test_reduce_axis_sum() {
    let a = StridedArray::<f64>::from_fn_row_major(&[4, 3, 2], |idx| {
        (idx[0] + 2 * idx[1] + 3 * idx[2]) as f64
    });

    let result = reduce_axis(&a.view(), 1, |x| x, |a, b| a + b, 0.0).unwrap();

    // Expected: sum along axis 1 (3 elements)
    // Result shape: [4, 2]
    assert_eq!(result.dims(), &[4, 2]);

    for i in 0..4 {
        for k in 0..2 {
            let mut expected = 0.0;
            for j in 0..3 {
                expected += (i + 2 * j + 3 * k) as f64;
            }
            assert_relative_eq!(result.get(&[i, k]), expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_dot() {
    let a = make_tensor(7, 3);
    let b = make_tensor(7, 3);
    let result = dot(&a.view(), &b.view()).unwrap();
    let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

#[test]
fn test_copy_into() {
    let a = make_tensor(4, 5);
    let mut out = StridedArray::<f64>::row_major(&[4, 5]);
    copy_into(&mut out.view_mut(), &a.view()).unwrap();
    for i in 0..4 {
        for j in 0..5 {
            assert_relative_eq!(out.get(&[i, j]), a.get(&[i, j]), epsilon = 1e-10);
        }
    }
}

#[test]
fn test_copy_transpose_scale_into() {
    let a = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 10 + idx[1]) as f64);
    let mut out = StridedArray::<f64>::row_major(&[3, 2]);

    copy_transpose_scale_into(&mut out.view_mut(), &a.view(), 3.0).unwrap();

    for i in 0..2 {
        for j in 0..3 {
            assert_relative_eq!(out.get(&[j, i]), 3.0 * a.get(&[i, j]), epsilon = 1e-10);
        }
    }
}

#[test]
fn test_symmetrize_into() {
    let n = 4;
    let a = StridedArray::<f64>::from_fn_row_major(&[n, n], |idx| (idx[0] * 10 + idx[1]) as f64);
    let mut out = StridedArray::<f64>::row_major(&[n, n]);

    symmetrize_into(&mut out.view_mut(), &a.view()).unwrap();

    for i in 0..n {
        for j in 0..n {
            let expected = (a.get(&[i, j]) + a.get(&[j, i])) * 0.5;
            assert_relative_eq!(out.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_zip_map4_into_contiguous() {
    let a = make_tensor(4, 5);
    let b = make_tensor(4, 5);
    let c = make_tensor(4, 5);
    let d = make_tensor(4, 5);
    let mut out = StridedArray::<f64>::row_major(&[4, 5]);

    zip_map4_into(
        &mut out.view_mut(),
        &a.view(),
        &b.view(),
        &c.view(),
        &d.view(),
        |a, b, c, d| a + b + c + d,
    )
    .unwrap();

    for i in 0..4 {
        for j in 0..5 {
            let expected = a.get(&[i, j]) + b.get(&[i, j]) + c.get(&[i, j]) + d.get(&[i, j]);
            assert_relative_eq!(out.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_zip_map4_into_permuted() {
    let size = 8usize;
    let a = StridedArray::<f64>::from_fn_row_major(&[size, size, size, size], |idx| {
        (idx[0] + 2 * idx[1] + 3 * idx[2] + 4 * idx[3]) as f64
    });

    let av = a.view();
    let p1 = av.permute(&[0, 1, 2, 3]).unwrap();
    let p2 = av.permute(&[1, 2, 3, 0]).unwrap();
    let p3 = av.permute(&[2, 3, 0, 1]).unwrap();
    let p4 = av.permute(&[3, 0, 1, 2]).unwrap();

    let mut out = StridedArray::<f64>::row_major(&[size, size, size, size]);

    zip_map4_into(&mut out.view_mut(), &p1, &p2, &p3, &p4, |a, b, c, d| {
        a + b + c + d
    })
    .unwrap();

    for i in 0..size {
        for j in 0..size {
            for k in 0..size {
                for l in 0..size {
                    let expected = p1.get(&[i, j, k, l])
                        + p2.get(&[i, j, k, l])
                        + p3.get(&[i, j, k, l])
                        + p4.get(&[i, j, k, l]);
                    assert_relative_eq!(out.get(&[i, j, k, l]), expected, epsilon = 1e-10);
                }
            }
        }
    }
}

#[test]
fn test_col_major_tensor() {
    let t = StridedArray::<f64>::from_fn_col_major(&[3, 4], |idx| (idx[0] * 10 + idx[1]) as f64);

    // Column-major: strides [1, 3]
    assert_eq!(t.strides(), &[1, 3]);
    assert_eq!(t.get(&[0, 0]), 0.0);
    assert_eq!(t.get(&[1, 0]), 10.0);
    assert_eq!(t.get(&[2, 3]), 23.0);

    // View operations
    let v = t.view();
    let vt = v.transpose_2d().unwrap();
    assert_eq!(vt.dims(), &[4, 3]);
    assert_eq!(vt.get(&[0, 0]), 0.0);
    assert_eq!(vt.get(&[0, 1]), 10.0);
    assert_eq!(vt.get(&[3, 2]), 23.0);
}

#[test]
fn test_strided_view_broadcast_and_copy() {
    let data = vec![1.0, 2.0, 3.0];
    let row = strided_rs::StridedView::<f64>::new(&data, &[1, 3], &[3, 1], 0).unwrap();
    let broad = row.broadcast(&[4, 3]).unwrap();

    let mut dest = StridedArray::<f64>::row_major(&[4, 3]);
    copy_into(&mut dest.view_mut(), &broad).unwrap();

    for i in 0..4 {
        assert_relative_eq!(dest.get(&[i, 0]), 1.0, epsilon = 1e-10);
        assert_relative_eq!(dest.get(&[i, 1]), 2.0, epsilon = 1e-10);
        assert_relative_eq!(dest.get(&[i, 2]), 3.0, epsilon = 1e-10);
    }
}
