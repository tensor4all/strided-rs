use approx::assert_relative_eq;
use num_complex::Complex64;
use strided_kernel::{
    add, axpy, copy_conj, copy_into, copy_scale, copy_transpose_scale_into, dot, fma, map_into,
    mul, reduce, reduce_axis, sum, symmetrize_conj_into, symmetrize_into, zip_map2_into,
    zip_map3_into, zip_map4_into, StridedArray, StridedError,
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
fn test_copy_into_mixed_layouts() {
    let a = StridedArray::<f64>::from_fn_col_major(&[4, 5], |idx| (idx[0] * 10 + idx[1]) as f64);
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
    let row = strided_kernel::StridedView::<f64>::new(&data, &[1, 3], &[3, 1], 0).unwrap();
    let broad = row.broadcast(&[4, 3]).unwrap();

    let mut dest = StridedArray::<f64>::row_major(&[4, 3]);
    copy_into(&mut dest.view_mut(), &broad).unwrap();

    for i in 0..4 {
        assert_relative_eq!(dest.get(&[i, 0]), 1.0, epsilon = 1e-10);
        assert_relative_eq!(dest.get(&[i, 1]), 2.0, epsilon = 1e-10);
        assert_relative_eq!(dest.get(&[i, 2]), 3.0, epsilon = 1e-10);
    }
}

// ============================================================================
// Large-array tests: exceed MINTHREADLENGTH (32768) to exercise parallel paths
// ============================================================================

/// Helper: create a large col-major 2D array (n*n > 32768 when n >= 182).
fn make_large(n: usize) -> StridedArray<f64> {
    StridedArray::from_fn_col_major(&[n, n], |idx| (idx[0] * n + idx[1]) as f64)
}

#[test]
fn test_large_map_into() {
    let n = 200;
    let a = make_large(n);
    let mut out = StridedArray::<f64>::col_major(&[n, n]);

    map_into(&mut out.view_mut(), &a.view(), |x| x * 3.0).unwrap();

    for i in (0..n).step_by(17) {
        for j in (0..n).step_by(19) {
            assert_relative_eq!(out.get(&[i, j]), a.get(&[i, j]) * 3.0, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_large_map_into_permuted() {
    let n = 200;
    let a = make_large(n);
    let a_t = a.view().permute(&[1, 0]).unwrap();
    let mut out = StridedArray::<f64>::col_major(&[n, n]);

    map_into(&mut out.view_mut(), &a_t, |x| x * 2.0).unwrap();

    for i in (0..n).step_by(17) {
        for j in (0..n).step_by(19) {
            assert_relative_eq!(out.get(&[i, j]), a.get(&[j, i]) * 2.0, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_large_zip_map2() {
    let n = 200;
    let a = make_large(n);
    let b = StridedArray::from_fn_col_major(&[n, n], |idx| (idx[0] + idx[1]) as f64);
    let mut out = StridedArray::<f64>::col_major(&[n, n]);

    zip_map2_into(&mut out.view_mut(), &a.view(), &b.view(), |x, y| x + y).unwrap();

    for i in (0..n).step_by(17) {
        for j in (0..n).step_by(19) {
            assert_relative_eq!(
                out.get(&[i, j]),
                a.get(&[i, j]) + b.get(&[i, j]),
                epsilon = 1e-10
            );
        }
    }
}

#[test]
fn test_large_zip_map4_permuted() {
    let n = 14; // 14^4 = 38416 > 32768
    let a = StridedArray::from_fn_col_major(&[n, n, n, n], |idx| {
        (idx[0] + 2 * idx[1] + 3 * idx[2] + 4 * idx[3]) as f64
    });
    let av = a.view();
    let p1 = av.permute(&[0, 1, 2, 3]).unwrap();
    let p2 = av.permute(&[1, 2, 3, 0]).unwrap();
    let p3 = av.permute(&[2, 3, 0, 1]).unwrap();
    let p4 = av.permute(&[3, 0, 1, 2]).unwrap();

    let mut out = StridedArray::<f64>::col_major(&[n, n, n, n]);
    zip_map4_into(&mut out.view_mut(), &p1, &p2, &p3, &p4, |a, b, c, d| {
        a + b + c + d
    })
    .unwrap();

    for i in (0..n).step_by(3) {
        for j in (0..n).step_by(3) {
            for k in (0..n).step_by(3) {
                for l in (0..n).step_by(3) {
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
fn test_large_reduce() {
    let n = 200;
    let a = make_large(n);
    let result = sum(&a.view()).unwrap();
    let expected: f64 = a.iter().copied().sum();
    assert_relative_eq!(result, expected, epsilon = 1e-6);
}

#[test]
fn test_large_add() {
    let n = 200;
    let b = StridedArray::from_fn_col_major(&[n, n], |idx| (idx[0] + idx[1]) as f64);
    let mut dest = StridedArray::from_fn_col_major(&[n, n], |idx| (idx[0] * 3 + idx[1]) as f64);
    let expected_base = dest.iter().cloned().collect::<Vec<_>>();

    add(&mut dest.view_mut(), &b.view()).unwrap();

    for i in (0..n).step_by(17) {
        for j in (0..n).step_by(19) {
            let flat = j * n + i; // col-major
            let expected = expected_base[flat] + b.get(&[i, j]);
            assert_relative_eq!(dest.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_large_mul() {
    let n = 200;
    let a = StridedArray::from_fn_col_major(&[n, n], |idx| (idx[0] + idx[1] + 1) as f64);
    let mut dest = StridedArray::from_fn_col_major(&[n, n], |idx| (idx[0] * 2 + idx[1] + 1) as f64);
    let expected_base = dest.iter().cloned().collect::<Vec<_>>();

    mul(&mut dest.view_mut(), &a.view()).unwrap();

    for i in (0..n).step_by(17) {
        for j in (0..n).step_by(19) {
            let flat = j * n + i;
            let expected = expected_base[flat] * a.get(&[i, j]);
            assert_relative_eq!(dest.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_large_axpy() {
    let n = 200;
    let x = make_large(n);
    let mut y = StridedArray::from_fn_col_major(&[n, n], |idx| (idx[0] + idx[1]) as f64);
    let y_orig = y.iter().cloned().collect::<Vec<_>>();
    let alpha = 2.5;

    axpy(&mut y.view_mut(), &x.view(), alpha).unwrap();

    for i in (0..n).step_by(17) {
        for j in (0..n).step_by(19) {
            let flat = j * n + i;
            let expected = alpha * x.get(&[i, j]) + y_orig[flat];
            assert_relative_eq!(y.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_large_fma() {
    let n = 200;
    let a = make_large(n);
    let b = StridedArray::from_fn_col_major(&[n, n], |idx| (idx[0] + idx[1] + 1) as f64);
    let mut dest = StridedArray::from_fn_col_major(&[n, n], |idx| (idx[0] * 3 + idx[1]) as f64);
    let dest_orig = dest.iter().cloned().collect::<Vec<_>>();

    fma(&mut dest.view_mut(), &a.view(), &b.view()).unwrap();

    for i in (0..n).step_by(17) {
        for j in (0..n).step_by(19) {
            let flat = j * n + i;
            let expected = dest_orig[flat] + a.get(&[i, j]) * b.get(&[i, j]);
            assert_relative_eq!(dest.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_large_dot() {
    let n = 200;
    let a = make_large(n);
    let b = StridedArray::from_fn_col_major(&[n, n], |idx| (idx[0] + idx[1] + 1) as f64);
    let result = dot(&a.view(), &b.view()).unwrap();
    let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    assert_relative_eq!(result, expected, epsilon = 1e-4);
}

#[test]
fn test_large_symmetrize() {
    let n = 200;
    let a = make_large(n);
    let mut out = StridedArray::<f64>::col_major(&[n, n]);

    symmetrize_into(&mut out.view_mut(), &a.view()).unwrap();

    for i in (0..n).step_by(17) {
        for j in (0..n).step_by(19) {
            let expected = (a.get(&[i, j]) + a.get(&[j, i])) * 0.5;
            assert_relative_eq!(out.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_large_copy_into_permuted() {
    let n = 200;
    let a = make_large(n);
    let a_t = a.view().permute(&[1, 0]).unwrap();
    let mut out = StridedArray::<f64>::col_major(&[n, n]);

    copy_into(&mut out.view_mut(), &a_t).unwrap();

    for i in (0..n).step_by(17) {
        for j in (0..n).step_by(19) {
            assert_relative_eq!(out.get(&[i, j]), a.get(&[j, i]), epsilon = 1e-10);
        }
    }
}

// ============================================================================
// Additional coverage tests for ops_view.rs
// ============================================================================

// 1. mul — non-contiguous (permuted) path
#[test]
fn test_mul_permuted() {
    let rows = 6;
    let cols = 5;
    let src =
        StridedArray::<f64>::from_fn_row_major(&[rows, cols], |idx| (idx[0] + idx[1] + 1) as f64);
    let mut dest = StridedArray::<f64>::from_fn_col_major(&[cols, rows], |idx| {
        (idx[0] * 2 + idx[1] + 1) as f64
    });
    // Snapshot dest values before mutation
    let mut dest_orig = vec![0.0; cols * rows];
    for i in 0..cols {
        for j in 0..rows {
            dest_orig[i * rows + j] = dest.get(&[i, j]);
        }
    }

    let src_t = src.view().permute(&[1, 0]).unwrap(); // shape [cols, rows]
    mul(&mut dest.view_mut(), &src_t).unwrap();

    for i in 0..cols {
        for j in 0..rows {
            let expected = dest_orig[i * rows + j] * src.get(&[j, i]);
            assert_relative_eq!(dest.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

// 2. axpy — non-contiguous (permuted) path
#[test]
fn test_axpy_permuted() {
    let rows = 6;
    let cols = 5;
    let x = StridedArray::<f64>::from_fn_row_major(&[rows, cols], |idx| {
        (idx[0] * cols + idx[1]) as f64
    });
    let mut y =
        StridedArray::<f64>::from_fn_col_major(&[cols, rows], |idx| (idx[0] + idx[1]) as f64);
    let mut y_orig = vec![0.0; cols * rows];
    for i in 0..cols {
        for j in 0..rows {
            y_orig[i * rows + j] = y.get(&[i, j]);
        }
    }
    let alpha = 3.5;

    let x_t = x.view().permute(&[1, 0]).unwrap(); // shape [cols, rows]
    axpy(&mut y.view_mut(), &x_t, alpha).unwrap();

    for i in 0..cols {
        for j in 0..rows {
            let expected = alpha * x.get(&[j, i]) + y_orig[i * rows + j];
            assert_relative_eq!(y.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

// 3. fma — non-contiguous (permuted) path
#[test]
fn test_fma_permuted() {
    let rows = 6;
    let cols = 5;
    let a = StridedArray::<f64>::from_fn_row_major(&[rows, cols], |idx| {
        (idx[0] * cols + idx[1]) as f64
    });
    let b =
        StridedArray::<f64>::from_fn_row_major(&[rows, cols], |idx| (idx[0] + idx[1] + 1) as f64);
    let mut dest =
        StridedArray::<f64>::from_fn_col_major(&[cols, rows], |idx| (idx[0] * 3 + idx[1]) as f64);
    let mut dest_orig = vec![0.0; cols * rows];
    for i in 0..cols {
        for j in 0..rows {
            dest_orig[i * rows + j] = dest.get(&[i, j]);
        }
    }

    let a_t = a.view().permute(&[1, 0]).unwrap();
    let b_t = b.view().permute(&[1, 0]).unwrap();
    fma(&mut dest.view_mut(), &a_t, &b_t).unwrap();

    for i in 0..cols {
        for j in 0..rows {
            let expected = dest_orig[i * rows + j] + a.get(&[j, i]) * b.get(&[j, i]);
            assert_relative_eq!(dest.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

// 4. dot — non-contiguous (permuted) path
#[test]
fn test_dot_permuted() {
    let rows = 7;
    let cols = 5;
    let a = StridedArray::<f64>::from_fn_row_major(&[rows, cols], |idx| {
        (idx[0] * cols + idx[1]) as f64
    });
    let b =
        StridedArray::<f64>::from_fn_row_major(&[rows, cols], |idx| (idx[0] + idx[1] + 1) as f64);

    let a_t = a.view().permute(&[1, 0]).unwrap(); // shape [cols, rows]
    let b_t = b.view().permute(&[1, 0]).unwrap();

    let result = dot(&a_t, &b_t).unwrap();

    // dot should give same result regardless of memory layout
    let mut expected = 0.0;
    for i in 0..cols {
        for j in 0..rows {
            expected += a.get(&[j, i]) * b.get(&[j, i]);
        }
    }
    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

// 5. sum — non-contiguous (permuted) path
#[test]
fn test_sum_permuted() {
    let rows = 8;
    let cols = 6;
    let a = StridedArray::<f64>::from_fn_row_major(&[rows, cols], |idx| {
        (idx[0] * cols + idx[1]) as f64
    });
    let a_t = a.view().permute(&[1, 0]).unwrap();

    let result = sum(&a_t).unwrap();
    let expected: f64 = a.iter().copied().sum();
    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

// 6. copy_into — with conj (element op) path using Complex64
#[test]
fn test_copy_into_conj_complex() {
    let a = StridedArray::<Complex64>::from_fn_row_major(&[3, 4], |idx| {
        Complex64::new((idx[0] * 4 + idx[1]) as f64, (idx[0] + idx[1]) as f64)
    });
    let mut out = StridedArray::<Complex64>::row_major(&[3, 4]);

    let a_conj = a.view().conj();
    copy_into(&mut out.view_mut(), &a_conj).unwrap();

    for i in 0..3 {
        for j in 0..4 {
            let expected = a.get(&[i, j]).conj();
            assert_relative_eq!(out.get(&[i, j]).re, expected.re, epsilon = 1e-10);
            assert_relative_eq!(out.get(&[i, j]).im, expected.im, epsilon = 1e-10);
        }
    }
}

// 7. copy_into — contiguous identity path (ptr::copy_nonoverlapping)
#[test]
fn test_copy_into_contiguous_identity() {
    // Both src and dest are row-major contiguous => hits ptr::copy_nonoverlapping fast path
    let a = StridedArray::<f64>::from_fn_row_major(&[10, 12], |idx| (idx[0] * 12 + idx[1]) as f64);
    let mut out = StridedArray::<f64>::row_major(&[10, 12]);

    copy_into(&mut out.view_mut(), &a.view()).unwrap();

    for i in 0..10 {
        for j in 0..12 {
            assert_relative_eq!(out.get(&[i, j]), a.get(&[i, j]), epsilon = 1e-10);
        }
    }
}

// 8. copy_scale function
#[test]
fn test_copy_scale() {
    let a = StridedArray::<f64>::from_fn_row_major(&[5, 6], |idx| (idx[0] * 6 + idx[1]) as f64);
    let mut out = StridedArray::<f64>::row_major(&[5, 6]);
    let scale = 2.5;

    copy_scale(&mut out.view_mut(), &a.view(), scale).unwrap();

    for i in 0..5 {
        for j in 0..6 {
            let expected = scale * a.get(&[i, j]);
            assert_relative_eq!(out.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

// 9. copy_conj function
#[test]
fn test_copy_conj_complex() {
    let a = StridedArray::<Complex64>::from_fn_row_major(&[4, 3], |idx| {
        Complex64::new((idx[0] * 3 + idx[1]) as f64, idx[0] as f64 - idx[1] as f64)
    });
    let mut out = StridedArray::<Complex64>::row_major(&[4, 3]);

    copy_conj(&mut out.view_mut(), &a.view()).unwrap();

    for i in 0..4 {
        for j in 0..3 {
            let expected = a.get(&[i, j]).conj();
            assert_relative_eq!(out.get(&[i, j]).re, expected.re, epsilon = 1e-10);
            assert_relative_eq!(out.get(&[i, j]).im, expected.im, epsilon = 1e-10);
        }
    }
}

// 10. symmetrize_conj_into function
#[test]
fn test_symmetrize_conj_into_complex() {
    let n = 4;
    let a = StridedArray::<Complex64>::from_fn_row_major(&[n, n], |idx| {
        Complex64::new((idx[0] * n + idx[1]) as f64, idx[0] as f64 - idx[1] as f64)
    });
    let mut out = StridedArray::<Complex64>::row_major(&[n, n]);

    symmetrize_conj_into(&mut out.view_mut(), &a.view()).unwrap();

    for i in 0..n {
        for j in 0..n {
            // (src + conj(src^T)) / 2 = (a[i,j] + conj(a[j,i])) / 2
            let expected = (a.get(&[i, j]) + a.get(&[j, i]).conj()) * 0.5;
            assert_relative_eq!(out.get(&[i, j]).re, expected.re, epsilon = 1e-10);
            assert_relative_eq!(out.get(&[i, j]).im, expected.im, epsilon = 1e-10);
        }
    }
}

// 11. symmetrize_into error cases
#[test]
fn test_symmetrize_into_error_non_2d() {
    let a = StridedArray::<f64>::from_fn_row_major(&[3, 3, 3], |idx| {
        (idx[0] * 9 + idx[1] * 3 + idx[2]) as f64
    });
    let mut out = StridedArray::<f64>::row_major(&[3, 3, 3]);

    let result = symmetrize_into(&mut out.view_mut(), &a.view());
    assert!(result.is_err());
    match result.unwrap_err() {
        StridedError::RankMismatch(ndim, 2) => assert_eq!(ndim, 3),
        e => panic!("expected RankMismatch, got: {:?}", e),
    }
}

#[test]
fn test_symmetrize_into_error_non_square() {
    let a = StridedArray::<f64>::from_fn_row_major(&[3, 5], |idx| (idx[0] * 5 + idx[1]) as f64);
    let mut out = StridedArray::<f64>::row_major(&[3, 5]);

    let result = symmetrize_into(&mut out.view_mut(), &a.view());
    assert!(result.is_err());
    match result.unwrap_err() {
        StridedError::NonSquare { rows, cols } => {
            assert_eq!(rows, 3);
            assert_eq!(cols, 5);
        }
        e => panic!("expected NonSquare, got: {:?}", e),
    }
}

// 12. copy_transpose_scale_into error case (non-2D)
#[test]
fn test_copy_transpose_scale_into_error_non_2d() {
    let a = StridedArray::<f64>::from_fn_row_major(&[2, 3, 4], |idx| {
        (idx[0] * 12 + idx[1] * 4 + idx[2]) as f64
    });
    let mut out = StridedArray::<f64>::row_major(&[2, 3, 4]);

    let result = copy_transpose_scale_into(&mut out.view_mut(), &a.view(), 2.0);
    assert!(result.is_err());
    match result.unwrap_err() {
        StridedError::RankMismatch(ndim, 2) => assert_eq!(ndim, 3),
        e => panic!("expected RankMismatch, got: {:?}", e),
    }
}

// 13. add — small contiguous path
#[test]
fn test_add_small_contiguous() {
    let src = StridedArray::<f64>::from_fn_row_major(&[3, 4], |idx| (idx[0] + idx[1]) as f64);
    let mut dest =
        StridedArray::<f64>::from_fn_row_major(&[3, 4], |idx| (idx[0] * 4 + idx[1]) as f64);
    let dest_orig: Vec<f64> = dest.iter().copied().collect();

    add(&mut dest.view_mut(), &src.view()).unwrap();

    for i in 0..3 {
        for j in 0..4 {
            let flat = i * 4 + j;
            let expected = dest_orig[flat] + src.get(&[i, j]);
            assert_relative_eq!(dest.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

// 14. mul — small contiguous path
#[test]
fn test_mul_small_contiguous() {
    let src = StridedArray::<f64>::from_fn_row_major(&[3, 4], |idx| (idx[0] + idx[1] + 1) as f64);
    let mut dest =
        StridedArray::<f64>::from_fn_row_major(&[3, 4], |idx| (idx[0] * 4 + idx[1] + 1) as f64);
    let dest_orig: Vec<f64> = dest.iter().copied().collect();

    mul(&mut dest.view_mut(), &src.view()).unwrap();

    for i in 0..3 {
        for j in 0..4 {
            let flat = i * 4 + j;
            let expected = dest_orig[flat] * src.get(&[i, j]);
            assert_relative_eq!(dest.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

// 15. copy_scale with permuted source (non-contiguous)
#[test]
fn test_copy_scale_permuted() {
    let a = StridedArray::<f64>::from_fn_row_major(&[5, 7], |idx| (idx[0] * 7 + idx[1]) as f64);
    let a_t = a.view().permute(&[1, 0]).unwrap(); // shape [7, 5]
    let mut out = StridedArray::<f64>::row_major(&[7, 5]);
    let scale = -1.5;

    copy_scale(&mut out.view_mut(), &a_t, scale).unwrap();

    for i in 0..7 {
        for j in 0..5 {
            let expected = scale * a.get(&[j, i]);
            assert_relative_eq!(out.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

// ============================================================================
// Mixed-layout tests: dest row-major, src col-major to force blocked inner loops
// These exercise inner_loop_add/mul/axpy/fma/dot (non-contiguous paths)
// ============================================================================

#[test]
fn test_add_mixed_layout() {
    // dest row-major [4,1], src col-major [1,4] => different contiguous orders => blocked path
    let m = 5;
    let n = 6;
    let src = StridedArray::<f64>::from_fn_col_major(&[m, n], |idx| (idx[0] + idx[1]) as f64);
    let mut dest =
        StridedArray::<f64>::from_fn_row_major(&[m, n], |idx| (idx[0] * n + idx[1]) as f64);
    let dest_orig: Vec<f64> = dest.iter().copied().collect();

    add(&mut dest.view_mut(), &src.view()).unwrap();

    for i in 0..m {
        for j in 0..n {
            // row-major flat index
            let flat = i * n + j;
            let expected = dest_orig[flat] + src.get(&[i, j]);
            assert_relative_eq!(dest.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_mul_mixed_layout() {
    let m = 5;
    let n = 6;
    let src = StridedArray::<f64>::from_fn_col_major(&[m, n], |idx| (idx[0] + idx[1] + 1) as f64);
    let mut dest =
        StridedArray::<f64>::from_fn_row_major(&[m, n], |idx| (idx[0] * n + idx[1] + 1) as f64);
    let dest_orig: Vec<f64> = dest.iter().copied().collect();

    mul(&mut dest.view_mut(), &src.view()).unwrap();

    for i in 0..m {
        for j in 0..n {
            let flat = i * n + j;
            let expected = dest_orig[flat] * src.get(&[i, j]);
            assert_relative_eq!(dest.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_axpy_mixed_layout() {
    let m = 5;
    let n = 6;
    let x = StridedArray::<f64>::from_fn_col_major(&[m, n], |idx| (idx[0] * n + idx[1]) as f64);
    let mut y = StridedArray::<f64>::from_fn_row_major(&[m, n], |idx| (idx[0] + idx[1]) as f64);
    let y_orig: Vec<f64> = y.iter().copied().collect();
    let alpha = 2.5;

    axpy(&mut y.view_mut(), &x.view(), alpha).unwrap();

    for i in 0..m {
        for j in 0..n {
            let flat = i * n + j;
            let expected = alpha * x.get(&[i, j]) + y_orig[flat];
            assert_relative_eq!(y.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_fma_mixed_layout() {
    let m = 5;
    let n = 6;
    let a = StridedArray::<f64>::from_fn_col_major(&[m, n], |idx| (idx[0] * n + idx[1]) as f64);
    let b = StridedArray::<f64>::from_fn_row_major(&[m, n], |idx| (idx[0] + idx[1] + 1) as f64);
    let mut dest =
        StridedArray::<f64>::from_fn_col_major(&[m, n], |idx| (idx[0] * 3 + idx[1]) as f64);
    let dest_orig: Vec<f64> = dest.iter().copied().collect();

    fma(&mut dest.view_mut(), &a.view(), &b.view()).unwrap();

    // dest is col-major, so iter() gives col-major order
    for i in 0..m {
        for j in 0..n {
            let flat = j * m + i; // col-major flat index
            let expected = dest_orig[flat] + a.get(&[i, j]) * b.get(&[i, j]);
            assert_relative_eq!(dest.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_dot_mixed_layout() {
    let m = 7;
    let n = 5;
    let a = StridedArray::<f64>::from_fn_row_major(&[m, n], |idx| (idx[0] * n + idx[1]) as f64);
    let b = StridedArray::<f64>::from_fn_col_major(&[m, n], |idx| (idx[0] + idx[1] + 1) as f64);

    let result = dot(&a.view(), &b.view()).unwrap();

    let mut expected = 0.0;
    for i in 0..m {
        for j in 0..n {
            expected += a.get(&[i, j]) * b.get(&[i, j]);
        }
    }
    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

#[test]
fn test_sum_col_major_small() {
    // col-major small array — exercises reduce non-SIMD path with non-row-major layout
    let a = StridedArray::<f64>::from_fn_col_major(&[3, 4], |idx| (idx[0] * 4 + idx[1]) as f64);
    let result = sum(&a.view()).unwrap();
    let expected: f64 = (0..3)
        .flat_map(|i| (0..4).map(move |j| (i * 4 + j) as f64))
        .sum();
    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

// f32 tests for SIMD sum/dot paths
#[test]
fn test_sum_f32_contiguous() {
    let a = StridedArray::<f32>::from_fn_row_major(&[100], |idx| (idx[0] + 1) as f32);
    let result = sum(&a.view()).unwrap();
    let expected: f32 = (1..=100).map(|x| x as f32).sum();
    assert_relative_eq!(result, expected, epsilon = 1e-3);
}

#[test]
fn test_dot_f32_contiguous() {
    let a = StridedArray::<f32>::from_fn_row_major(&[100], |idx| (idx[0] + 1) as f32);
    let b = StridedArray::<f32>::from_fn_row_major(&[100], |idx| (idx[0] * 2 + 1) as f32);
    let result = dot(&a.view(), &b.view()).unwrap();
    let expected: f32 = (0..100).map(|i| (i + 1) as f32 * (i * 2 + 1) as f32).sum();
    assert_relative_eq!(result, expected, epsilon = 1e-1);
}

// copy_into with conj on non-contiguous layout to force blocked path
#[test]
fn test_copy_into_conj_mixed_layout() {
    let a = StridedArray::<Complex64>::from_fn_col_major(&[3, 4], |idx| {
        Complex64::new((idx[0] * 4 + idx[1]) as f64, (idx[0] + idx[1]) as f64)
    });
    let mut out = StridedArray::<Complex64>::row_major(&[3, 4]);

    let a_conj = a.view().conj();
    copy_into(&mut out.view_mut(), &a_conj).unwrap();

    for i in 0..3 {
        for j in 0..4 {
            let expected = a.get(&[i, j]).conj();
            assert_relative_eq!(out.get(&[i, j]).re, expected.re, epsilon = 1e-10);
            assert_relative_eq!(out.get(&[i, j]).im, expected.im, epsilon = 1e-10);
        }
    }
}

// ============================================================================
// zip_map3_into coverage tests
// ============================================================================

// 16. zip_map3_into — contiguous data
#[test]
fn test_zip_map3_into_contiguous() {
    let a = StridedArray::<f64>::from_fn_row_major(&[4, 5], |idx| (idx[0] * 5 + idx[1]) as f64);
    let b = StridedArray::<f64>::from_fn_row_major(&[4, 5], |idx| (idx[0] + idx[1] + 1) as f64);
    let c = StridedArray::<f64>::from_fn_row_major(&[4, 5], |idx| (idx[0] * 2 + idx[1] * 3) as f64);
    let mut out = StridedArray::<f64>::row_major(&[4, 5]);

    zip_map3_into(
        &mut out.view_mut(),
        &a.view(),
        &b.view(),
        &c.view(),
        |x, y, z| x + y * z,
    )
    .unwrap();

    for i in 0..4 {
        for j in 0..5 {
            let expected = a.get(&[i, j]) + b.get(&[i, j]) * c.get(&[i, j]);
            assert_relative_eq!(out.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

// 17. zip_map3_into — non-contiguous (permuted) data
#[test]
fn test_zip_map3_into_permuted() {
    let rows = 6;
    let cols = 5;
    let a = StridedArray::<f64>::from_fn_row_major(&[rows, cols], |idx| {
        (idx[0] * cols + idx[1]) as f64
    });
    let b =
        StridedArray::<f64>::from_fn_row_major(&[rows, cols], |idx| (idx[0] + idx[1] + 1) as f64);
    let c = StridedArray::<f64>::from_fn_row_major(&[rows, cols], |idx| {
        (idx[0] * 2 + idx[1] * 3) as f64
    });

    let a_t = a.view().permute(&[1, 0]).unwrap(); // shape [cols, rows]
    let b_t = b.view().permute(&[1, 0]).unwrap();
    let c_t = c.view().permute(&[1, 0]).unwrap();

    let mut out = StridedArray::<f64>::row_major(&[cols, rows]);

    zip_map3_into(&mut out.view_mut(), &a_t, &b_t, &c_t, |x, y, z| x * y + z).unwrap();

    for i in 0..cols {
        for j in 0..rows {
            let expected = a.get(&[j, i]) * b.get(&[j, i]) + c.get(&[j, i]);
            assert_relative_eq!(out.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

// 18. zip_map3_into — mixed strides (some permuted, some not)
#[test]
fn test_zip_map3_into_mixed_strides() {
    let a = StridedArray::<f64>::from_fn_row_major(&[5, 4], |idx| (idx[0] * 4 + idx[1]) as f64);
    let b = StridedArray::<f64>::from_fn_col_major(&[5, 4], |idx| (idx[0] + idx[1]) as f64);
    let c = StridedArray::<f64>::from_fn_row_major(&[4, 5], |idx| (idx[0] * 5 + idx[1] + 1) as f64);
    let c_t = c.view().permute(&[1, 0]).unwrap(); // shape [5, 4]

    let mut out = StridedArray::<f64>::col_major(&[5, 4]);

    zip_map3_into(
        &mut out.view_mut(),
        &a.view(),
        &b.view(),
        &c_t,
        |x, y, z| x + y + z,
    )
    .unwrap();

    for i in 0..5 {
        for j in 0..4 {
            let expected = a.get(&[i, j]) + b.get(&[i, j]) + c.get(&[j, i]);
            assert_relative_eq!(out.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

// ============================================================================
// reduce_view.rs coverage tests
// ============================================================================

// 19. reduce — non-contiguous (mixed layout: force blocked path)
#[test]
fn test_reduce_mixed_layout() {
    let m = 8;
    let n = 6;
    // col-major ensures non-contiguous when blocking algorithm picks row-major order
    let a = StridedArray::<f64>::from_fn_col_major(&[m, n], |idx| (idx[0] * n + idx[1] + 1) as f64);
    let a_t = a.view().permute(&[1, 0]).unwrap(); // different stride order

    let result = reduce(&a_t, |x| x, |a, b| a + b, 0.0).unwrap();
    let expected: f64 = (0..m)
        .flat_map(|i| (0..n).map(move |j| (i * n + j + 1) as f64))
        .sum();
    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

// 20. reduce — product over non-contiguous data
#[test]
fn test_reduce_product_permuted() {
    // Use small values to avoid overflow
    let a = StridedArray::<f64>::from_fn_row_major(&[3, 4], |idx| {
        1.0 + 0.01 * (idx[0] * 4 + idx[1]) as f64
    });
    let a_t = a.view().permute(&[1, 0]).unwrap();

    let result = reduce(&a_t, |x| x, |a, b| a * b, 1.0).unwrap();
    let expected: f64 = a.iter().copied().product();
    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

// 21. reduce_axis — invalid axis (error case)
#[test]
fn test_reduce_axis_invalid_axis() {
    let a =
        StridedArray::<f64>::from_fn_row_major(&[3, 4, 2], |idx| (idx[0] + idx[1] + idx[2]) as f64);

    let result = reduce_axis(&a.view(), 3, |x| x, |a, b| a + b, 0.0);
    assert!(result.is_err());
    match result.unwrap_err() {
        StridedError::InvalidAxis { axis, rank } => {
            assert_eq!(axis, 3);
            assert_eq!(rank, 3);
        }
        e => panic!("expected InvalidAxis, got: {:?}", e),
    }
}

// 22. reduce_axis — 1D input (reduces to scalar wrapped in array)
#[test]
fn test_reduce_axis_1d_to_scalar() {
    let a = StridedArray::<f64>::from_fn_row_major(&[5], |idx| (idx[0] + 1) as f64);

    let result = reduce_axis(&a.view(), 0, |x| x, |a, b| a + b, 0.0).unwrap();

    // 1 + 2 + 3 + 4 + 5 = 15
    assert_eq!(result.dims(), &[1]);
    assert_relative_eq!(result.get(&[0]), 15.0, epsilon = 1e-10);
}

// 23. reduce_axis — with permuted source
#[test]
fn test_reduce_axis_permuted() {
    let a = StridedArray::<f64>::from_fn_row_major(&[3, 4], |idx| (idx[0] * 4 + idx[1] + 1) as f64);
    let a_t = a.view().permute(&[1, 0]).unwrap(); // shape [4, 3]

    // Reduce along axis 0 of the permuted view (axis 0 of transposed = axis 1 of original)
    let result = reduce_axis(&a_t, 0, |x| x, |a, b| a + b, 0.0).unwrap();
    assert_eq!(result.dims(), &[3]);

    for j in 0..3 {
        let mut expected = 0.0;
        for i in 0..4 {
            expected += a.get(&[j, i]); // a_t[i, j] = a[j, i]
        }
        assert_relative_eq!(result.get(&[j]), expected, epsilon = 1e-10);
    }
}

// ============================================================================
// Small array tests: exercise dispatch_if_large with len < 64
// ============================================================================

// 24. Small contiguous map_into (len = 6 < 64)
#[test]
fn test_small_contiguous_map_into() {
    let a = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1]) as f64);
    let mut out = StridedArray::<f64>::row_major(&[2, 3]);

    map_into(&mut out.view_mut(), &a.view(), |x| x * 5.0).unwrap();

    for i in 0..2 {
        for j in 0..3 {
            assert_relative_eq!(out.get(&[i, j]), a.get(&[i, j]) * 5.0, epsilon = 1e-10);
        }
    }
}

// 25. Small contiguous sum (len = 6 < 64)
#[test]
fn test_small_contiguous_sum() {
    let a = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1]) as f64);
    let result = sum(&a.view()).unwrap();
    // 0 + 1 + 2 + 3 + 4 + 5 = 15
    assert_relative_eq!(result, 15.0, epsilon = 1e-10);
}

// 26. Small contiguous dot (len = 6 < 64)
#[test]
fn test_small_contiguous_dot() {
    let a = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
    let b = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] + idx[1]) as f64);
    let result = dot(&a.view(), &b.view()).unwrap();
    let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

// 27. Small contiguous add (len = 6 < 64)
#[test]
fn test_small_contiguous_add() {
    let src = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] + idx[1]) as f64);
    let mut dest =
        StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1]) as f64);
    let dest_orig: Vec<f64> = dest.iter().copied().collect();

    add(&mut dest.view_mut(), &src.view()).unwrap();

    for i in 0..2 {
        for j in 0..3 {
            let flat = i * 3 + j;
            let expected = dest_orig[flat] + src.get(&[i, j]);
            assert_relative_eq!(dest.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

// 28. Small contiguous copy_into (len = 4 < 64)
#[test]
fn test_small_contiguous_copy_into() {
    let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| (idx[0] * 10 + idx[1]) as f64);
    let mut out = StridedArray::<f64>::row_major(&[2, 2]);

    copy_into(&mut out.view_mut(), &a.view()).unwrap();

    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(out.get(&[i, j]), a.get(&[i, j]), epsilon = 1e-10);
        }
    }
}

// ============================================================================
// reduce_view.rs coverage: non-contiguous blocked path for `reduce`
// ============================================================================

// 29. reduce on a 3D permuted view — forces non-contiguous blocked path
#[test]
fn test_reduce_3d_permuted_blocked() {
    // 3D array with permutation [2,0,1] — strides become non-contiguous
    let a = StridedArray::<f64>::from_fn_row_major(&[4, 5, 6], |idx| {
        (idx[0] * 30 + idx[1] * 6 + idx[2] + 1) as f64
    });
    let a_perm = a.view().permute(&[2, 0, 1]).unwrap(); // shape [6,4,5], non-contiguous

    let result = reduce(&a_perm, |x| x, |a, b| a + b, 0.0).unwrap();
    let expected: f64 = a.iter().copied().sum();
    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

// 30. reduce with map_fn and non-contiguous data
#[test]
fn test_reduce_sum_of_squares_permuted() {
    let a = StridedArray::<f64>::from_fn_row_major(&[5, 7], |idx| (idx[0] * 7 + idx[1] + 1) as f64);
    let a_t = a.view().permute(&[1, 0]).unwrap(); // shape [7,5], non-contiguous

    let result = reduce(&a_t, |x| x * x, |a, b| a + b, 0.0).unwrap();
    let expected: f64 = a.iter().copied().map(|x| x * x).sum();
    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

// 31. reduce max on non-contiguous data (non-additive reduce_fn)
#[test]
fn test_reduce_max_permuted() {
    let a = StridedArray::<f64>::from_fn_row_major(&[6, 8], |idx| (idx[0] * 8 + idx[1]) as f64);
    let a_t = a.view().permute(&[1, 0]).unwrap();

    let result = reduce(
        &a_t,
        |x| x,
        |a, b| if a > b { a } else { b },
        f64::NEG_INFINITY,
    )
    .unwrap();
    let expected: f64 = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

// 32. reduce on col-major 3D array — ensures blocked path with different stride order
#[test]
fn test_reduce_col_major_3d() {
    let a = StridedArray::<f64>::from_fn_col_major(&[3, 5, 4], |idx| {
        (idx[0] * 20 + idx[1] * 4 + idx[2] + 1) as f64
    });
    let a_perm = a.view().permute(&[1, 2, 0]).unwrap(); // shuffle strides

    let result = reduce(&a_perm, |x| x, |a, b| a + b, 0.0).unwrap();
    let mut expected = 0.0;
    for i in 0..3 {
        for j in 0..5 {
            for k in 0..4 {
                expected += a.get(&[i, j, k]);
            }
        }
    }
    assert_relative_eq!(result, expected, epsilon = 1e-10);
}

// ============================================================================
// reduce_view.rs coverage: general blocked iteration path for `reduce_axis`
// ============================================================================

// 33. reduce_axis on a 3D permuted array — forces blocked iteration (lines 200-254)
#[test]
fn test_reduce_axis_3d_permuted_blocked() {
    // Use permuted 3D array to ensure strides mismatch forces blocked path
    let a = StridedArray::<f64>::from_fn_row_major(&[3, 4, 5], |idx| {
        (idx[0] * 20 + idx[1] * 5 + idx[2] + 1) as f64
    });
    let a_perm = a.view().permute(&[2, 0, 1]).unwrap(); // shape [5, 3, 4]

    // Reduce along axis 1 of the permuted view (axis 0 of original)
    let result = reduce_axis(&a_perm, 1, |x| x, |a, b| a + b, 0.0).unwrap();
    assert_eq!(result.dims(), &[5, 4]); // shape after removing axis 1

    for i in 0..5 {
        for j in 0..4 {
            // a_perm[i, :, j] = a[:, j, i] — sum over axis 0 of original
            let mut expected = 0.0;
            for k in 0..3 {
                expected += a.get(&[k, j, i]);
            }
            assert_relative_eq!(result.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

// 34. reduce_axis on col-major with mixed strides — ensures blocked iteration
#[test]
fn test_reduce_axis_col_major_mixed() {
    let a = StridedArray::<f64>::from_fn_col_major(&[4, 3, 2], |idx| {
        (idx[0] * 6 + idx[1] * 2 + idx[2] + 1) as f64
    });

    // Reduce along axis 2 — the remaining dims [4,3] have col-major strides
    // while output will be col-major, but the source strides may differ
    let result = reduce_axis(&a.view(), 2, |x| x, |a, b| a + b, 0.0).unwrap();
    assert_eq!(result.dims(), &[4, 3]);

    for i in 0..4 {
        for j in 0..3 {
            let mut expected = 0.0;
            for k in 0..2 {
                expected += a.get(&[i, j, k]);
            }
            assert_relative_eq!(result.get(&[i, j]), expected, epsilon = 1e-10);
        }
    }
}

// 35. reduce_axis with map_fn on permuted data
#[test]
fn test_reduce_axis_map_fn_permuted() {
    let a = StridedArray::<f64>::from_fn_row_major(&[3, 4], |idx| (idx[0] * 4 + idx[1] + 1) as f64);
    let a_t = a.view().permute(&[1, 0]).unwrap(); // shape [4, 3]

    // Reduce axis 1 of transposed with map_fn = |x| x * x
    let result = reduce_axis(&a_t, 1, |x| x * x, |a, b| a + b, 0.0).unwrap();
    assert_eq!(result.dims(), &[4]);

    for i in 0..4 {
        let mut expected = 0.0;
        for j in 0..3 {
            let val = a.get(&[j, i]); // a_t[i, j] = a[j, i]
            expected += val * val;
        }
        assert_relative_eq!(result.get(&[i]), expected, epsilon = 1e-10);
    }
}

// 36. reduce_axis on 4D data — higher dimensional blocked path
#[test]
fn test_reduce_axis_4d_blocked() {
    let a = StridedArray::<f64>::from_fn_row_major(&[2, 3, 4, 5], |idx| {
        (idx[0] * 60 + idx[1] * 20 + idx[2] * 5 + idx[3] + 1) as f64
    });
    let a_perm = a.view().permute(&[3, 1, 0, 2]).unwrap(); // shape [5, 3, 2, 4]

    // Reduce along axis 2 of permuted view
    let result = reduce_axis(&a_perm, 2, |x| x, |a, b| a + b, 0.0).unwrap();
    assert_eq!(result.dims(), &[5, 3, 4]); // shape after removing axis 2

    for i in 0..5 {
        for j in 0..3 {
            for k in 0..4 {
                // a_perm[i, j, :, k] — sum over axis 2 of permuted
                // a_perm[i,j,l,k] = a[l, j, k, i]
                let mut expected = 0.0;
                for l in 0..2 {
                    expected += a.get(&[l, j, k, i]);
                }
                assert_relative_eq!(result.get(&[i, j, k]), expected, epsilon = 1e-10);
            }
        }
    }
}

// ============================================================================
// Mixed scalar type tests (Issue #5)
// ============================================================================

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

// 37. map_into: f64 → Complex64
#[test]
fn test_map_into_mixed_f64_to_c64() {
    let src = StridedArray::<f64>::from_fn_row_major(&[3, 4], |idx| (idx[0] * 4 + idx[1]) as f64);
    let mut dest = StridedArray::<Complex64>::row_major(&[3, 4]);

    map_into(&mut dest.view_mut(), &src.view(), |x| {
        Complex64::new(x, x * 2.0)
    })
    .unwrap();

    for i in 0..3 {
        for j in 0..4 {
            let v = (i * 4 + j) as f64;
            assert_eq!(dest.get(&[i, j]), c(v, v * 2.0));
        }
    }
}

// 38. zip_map2_into: (f64, Complex64) → Complex64
#[test]
fn test_zip_map2_into_mixed_f64_c64() {
    let a = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
    let b = StridedArray::<Complex64>::from_fn_row_major(&[2, 3], |idx| {
        c((idx[0] + 1) as f64, (idx[1] + 1) as f64)
    });
    let mut dest = StridedArray::<Complex64>::row_major(&[2, 3]);

    zip_map2_into(&mut dest.view_mut(), &a.view(), &b.view(), |x, y| {
        Complex64::new(x, 0.0) + y
    })
    .unwrap();

    for i in 0..2 {
        for j in 0..3 {
            let x = (i * 3 + j + 1) as f64;
            let y = c((i + 1) as f64, (j + 1) as f64);
            assert_eq!(dest.get(&[i, j]), Complex64::new(x, 0.0) + y);
        }
    }
}

// 39. add: Complex64 += f64
#[test]
fn test_add_mixed_c64_plus_f64() {
    let mut dest = StridedArray::<Complex64>::from_fn_row_major(&[3, 3], |idx| {
        c(idx[0] as f64, idx[1] as f64)
    });
    let src =
        StridedArray::<f64>::from_fn_row_major(&[3, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);

    let orig: Vec<Complex64> = dest.iter().copied().collect();
    add(&mut dest.view_mut(), &src.view()).unwrap();

    for i in 0..3 {
        for j in 0..3 {
            let flat = i * 3 + j;
            let expected = orig[flat] + (flat + 1) as f64;
            assert_eq!(dest.get(&[i, j]), expected);
        }
    }
}

// 40. mul: Complex64 *= f64
#[test]
fn test_mul_mixed_c64_times_f64() {
    let mut dest = StridedArray::<Complex64>::from_fn_row_major(&[2, 3], |idx| {
        c(idx[0] as f64, idx[1] as f64)
    });
    let src = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] + idx[1] + 1) as f64);

    let orig: Vec<Complex64> = dest.iter().copied().collect();
    mul(&mut dest.view_mut(), &src.view()).unwrap();

    for i in 0..2 {
        for j in 0..3 {
            let flat = i * 3 + j;
            let scale = (i + j + 1) as f64;
            assert_eq!(dest.get(&[i, j]), orig[flat] * scale);
        }
    }
}

// 41. axpy: Complex64 += f64 * f64 (alpha: f64, src: f64, dest: Complex64)
#[test]
fn test_axpy_mixed() {
    let mut dest = StridedArray::<Complex64>::from_fn_row_major(&[2, 4], |idx| {
        c((idx[0] * 4 + idx[1]) as f64, 1.0)
    });
    let src = StridedArray::<f64>::from_fn_row_major(&[2, 4], |idx| (idx[0] + idx[1] + 1) as f64);
    let alpha = 2.5_f64;

    let orig: Vec<Complex64> = dest.iter().copied().collect();
    // alpha * src[i] -> f64, then Complex64 + f64 -> Complex64
    // Note: alpha and src are both f64, so A*S -> f64, and D + D needs Complex64
    // Actually this won't work directly since alpha*src produces f64, not Complex64.
    // We need alpha: Complex64 to produce Complex64 output.
    // Let's use Complex64 alpha instead:
    let alpha_c = Complex64::new(alpha, 0.0);
    axpy(&mut dest.view_mut(), &src.view(), alpha_c).unwrap();

    for i in 0..2 {
        for j in 0..4 {
            let flat = i * 4 + j;
            let s = (i + j + 1) as f64;
            let expected = orig[flat] + alpha_c * s;
            assert_eq!(dest.get(&[i, j]), expected);
        }
    }
}

// 42. fma: Complex64 += f64 * Complex64
#[test]
fn test_fma_mixed_f64_c64() {
    let mut dest = StridedArray::<Complex64>::from_fn_row_major(&[3, 2], |_| c(0.0, 0.0));
    let a = StridedArray::<f64>::from_fn_row_major(&[3, 2], |idx| (idx[0] * 2 + idx[1] + 1) as f64);
    let b = StridedArray::<Complex64>::from_fn_row_major(&[3, 2], |idx| {
        c(idx[0] as f64, idx[1] as f64)
    });

    fma(&mut dest.view_mut(), &a.view(), &b.view()).unwrap();

    for i in 0..3 {
        for j in 0..2 {
            let av = (i * 2 + j + 1) as f64;
            let bv = c(i as f64, j as f64);
            // f64 * Complex64 -> Complex64
            assert_eq!(dest.get(&[i, j]), av * bv);
        }
    }
}

// 43. dot: f64 · Complex64 → Complex64
#[test]
fn test_dot_mixed_f64_c64() {
    let a = StridedArray::<f64>::from_fn_row_major(&[4], |idx| (idx[0] + 1) as f64);
    let b = StridedArray::<Complex64>::from_fn_row_major(&[4], |idx| c(idx[0] as f64, 1.0));

    let result: Complex64 = dot(&a.view(), &b.view()).unwrap();

    // Expected: sum(a[i] * b[i]) = 1*(0+i) + 2*(1+i) + 3*(2+i) + 4*(3+i)
    //         = (0+2+6+12) + (1+2+3+4)i = 20 + 10i
    assert_eq!(result, c(20.0, 10.0));
}

// 44. dot same-type still works (regression guard for SIMD path)
#[test]
fn test_dot_same_type_simd_regression() {
    let n = 1000;
    let a = StridedArray::<f64>::from_fn_row_major(&[n], |idx| (idx[0] + 1) as f64);
    let b = StridedArray::<f64>::from_fn_row_major(&[n], |idx| (idx[0] + 1) as f64);

    let result: f64 = dot(&a.view(), &b.view()).unwrap();

    // sum(i^2, i=1..1000) = n*(n+1)*(2n+1)/6
    let expected = (n * (n + 1) * (2 * n + 1)) as f64 / 6.0;
    assert_relative_eq!(result, expected, epsilon = 1e-6);
}

// 45. copy_scale with different scale type
#[test]
fn test_copy_scale_mixed() {
    let src =
        StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
    let mut dest = StridedArray::<Complex64>::row_major(&[2, 3]);
    let scale = c(0.0, 1.0); // multiply by i

    copy_scale(&mut dest.view_mut(), &src.view(), scale).unwrap();

    for i in 0..2 {
        for j in 0..3 {
            let v = (i * 3 + j + 1) as f64;
            // scale * v = i * v (purely imaginary)
            assert_eq!(dest.get(&[i, j]), c(0.0, v));
        }
    }
}

/// Custom Copy type that does NOT implement ElementOpApply.
/// Verifies that Identity views work with map_into, reduce, etc.
#[test]
fn test_custom_type_map_into_without_element_op_apply() {
    #[derive(Debug, Clone, Copy, PartialEq, Default)]
    struct Wrapper(f64);

    let src_data = vec![Wrapper(1.0), Wrapper(2.0), Wrapper(3.0), Wrapper(4.0)];
    let src = StridedArray::from_parts(src_data, &[2, 2], &[2, 1], 0).unwrap();
    let mut dest = StridedArray::<Wrapper>::col_major(&[2, 2]);

    // map_into with Identity view on custom type
    map_into(&mut dest.view_mut(), &src.view(), |Wrapper(x)| {
        Wrapper(x * 2.0)
    })
    .unwrap();

    assert_eq!(dest.get(&[0, 0]), Wrapper(2.0));
    assert_eq!(dest.get(&[0, 1]), Wrapper(4.0));
    assert_eq!(dest.get(&[1, 0]), Wrapper(6.0));
    assert_eq!(dest.get(&[1, 1]), Wrapper(8.0));
}
