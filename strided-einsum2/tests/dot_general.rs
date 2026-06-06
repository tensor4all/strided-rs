use strided_einsum2::{dot_general_into, DotGeneralConfig};
use strided_view::{col_major_strides, StridedArray};

fn col_major_array(data: Vec<f64>, shape: &[usize]) -> StridedArray<f64> {
    StridedArray::from_parts(data, shape, &col_major_strides(shape), 0).unwrap()
}

fn get_col_major(data: &[f64], shape: &[usize], idx: &[usize]) -> f64 {
    let mut stride = 1usize;
    let mut offset = 0usize;
    for (&i, &dim) in idx.iter().zip(shape) {
        offset += i * stride;
        stride *= dim;
    }
    data[offset]
}

fn expected_matmul_col_major(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for j in 0..n {
        for i in 0..m {
            let mut acc = 0.0;
            for p in 0..k {
                acc += get_col_major(a, a_shape, &[i, p]) * get_col_major(b, b_shape, &[p, j]);
            }
            out[i + m * j] = acc;
        }
    }
    out
}

#[test]
fn dot_general_matmul_matches_col_major_reference() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let a = col_major_array(a_data.clone(), &[2, 3]);
    let b = col_major_array(b_data.clone(), &[3, 2]);
    let mut c = StridedArray::<f64>::col_major(&[2, 2]);

    dot_general_into(
        c.view_mut(),
        &a.view(),
        &b.view(),
        &DotGeneralConfig {
            lhs_contracting_dims: &[1],
            rhs_contracting_dims: &[0],
            lhs_batch_dims: &[],
            rhs_batch_dims: &[],
        },
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(
        c.data(),
        expected_matmul_col_major(&a_data, &[2, 3], &b_data, &[3, 2], 2, 2, 3).as_slice()
    );
}

#[test]
fn dot_general_batched_matmul_uses_batch_trailing_output_shape() {
    let a = StridedArray::<f64>::from_fn_col_major(&[2, 3, 2], |idx| {
        (100 * idx[2] + 10 * idx[1] + idx[0] + 1) as f64
    });
    let b = StridedArray::<f64>::from_fn_col_major(&[3, 4, 2], |idx| {
        (100 * idx[2] + 10 * idx[1] + idx[0] + 1) as f64
    });
    let mut c = StridedArray::<f64>::col_major(&[2, 4, 2]);

    dot_general_into(
        c.view_mut(),
        &a.view(),
        &b.view(),
        &DotGeneralConfig {
            lhs_contracting_dims: &[1],
            rhs_contracting_dims: &[0],
            lhs_batch_dims: &[2],
            rhs_batch_dims: &[2],
        },
        1.0,
        0.0,
    )
    .unwrap();

    for batch in 0..2 {
        for j in 0..4 {
            for i in 0..2 {
                let mut expected = 0.0;
                for p in 0..3 {
                    expected += a.get(&[i, p, batch]) * b.get(&[p, j, batch]);
                }
                assert_eq!(c.get(&[i, j, batch]), expected);
            }
        }
    }
}

#[test]
fn dot_general_matches_tenferro_batched_matmul_colmajor_layout() {
    let batch = 2;
    let m = 3;
    let n = 4;
    let k = 5;
    let lhs_row_memory = StridedArray::<f64>::from_fn_col_major(&[k, m, batch], |idx| {
        (1000 * idx[2] + 100 * idx[1] + 10 * idx[0] + 1) as f64
    });
    let rhs_row_memory = StridedArray::<f64>::from_fn_col_major(&[n, k, batch], |idx| {
        (1000 * idx[2] + 100 * idx[1] + 10 * idx[0] + 7) as f64
    });
    let mut out = StridedArray::<f64>::col_major(&[n, m, batch]);

    dot_general_into(
        out.view_mut(),
        &rhs_row_memory.view(),
        &lhs_row_memory.view(),
        &DotGeneralConfig {
            lhs_contracting_dims: &[1],
            rhs_contracting_dims: &[0],
            lhs_batch_dims: &[2],
            rhs_batch_dims: &[2],
        },
        1.0,
        0.0,
    )
    .unwrap();

    for b in 0..batch {
        for i in 0..m {
            for out_colmajor_n in 0..n {
                let mut expected = 0.0;
                for p in 0..k {
                    expected += lhs_row_memory.get(&[p, i, b])
                        * rhs_row_memory.get(&[out_colmajor_n, p, b]);
                }
                assert_eq!(out.get(&[out_colmajor_n, i, b]), expected);
            }
        }
    }
}

#[test]
fn dot_general_accepts_transposed_input_view() {
    let a = StridedArray::<f64>::from_fn_col_major(&[3, 2], |idx| (idx[0] + 3 * idx[1] + 1) as f64);
    let a_t = a.view().permute(&[1, 0]).unwrap();
    let b = StridedArray::<f64>::from_fn_col_major(&[3, 2], |idx| (idx[0] + 3 * idx[1] + 7) as f64);
    let mut c = StridedArray::<f64>::col_major(&[2, 2]);

    dot_general_into(
        c.view_mut(),
        &a_t,
        &b.view(),
        &DotGeneralConfig {
            lhs_contracting_dims: &[1],
            rhs_contracting_dims: &[0],
            lhs_batch_dims: &[],
            rhs_batch_dims: &[],
        },
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(c.data(), &[50.0, 122.0, 68.0, 167.0]);
}

#[test]
fn dot_general_inner_product_returns_rank0_scalar() {
    let a = col_major_array(vec![1.0, 2.0, 3.0], &[3]);
    let b = col_major_array(vec![4.0, 5.0, 6.0], &[3]);
    let mut c = StridedArray::<f64>::col_major(&[]);

    dot_general_into(
        c.view_mut(),
        &a.view(),
        &b.view(),
        &DotGeneralConfig {
            lhs_contracting_dims: &[0],
            rhs_contracting_dims: &[0],
            lhs_batch_dims: &[],
            rhs_batch_dims: &[],
        },
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(c.data(), &[32.0]);
}

#[test]
fn dot_general_zero_contracting_dim_zero_fills_output() {
    let a = col_major_array(Vec::new(), &[2, 0]);
    let b = col_major_array(Vec::new(), &[0, 3]);
    let mut c = StridedArray::<f64>::from_fn_col_major(&[2, 3], |_| 5.0);

    dot_general_into(
        c.view_mut(),
        &a.view(),
        &b.view(),
        &DotGeneralConfig {
            lhs_contracting_dims: &[1],
            rhs_contracting_dims: &[0],
            lhs_batch_dims: &[],
            rhs_batch_dims: &[],
        },
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(c.data(), &[0.0; 6]);
}

#[test]
fn dot_general_rejects_wrong_output_shape() {
    let a = StridedArray::<f64>::col_major(&[2, 3]);
    let b = StridedArray::<f64>::col_major(&[3, 4]);
    let mut c = StridedArray::<f64>::col_major(&[4, 2]);

    let err = dot_general_into(
        c.view_mut(),
        &a.view(),
        &b.view(),
        &DotGeneralConfig {
            lhs_contracting_dims: &[1],
            rhs_contracting_dims: &[0],
            lhs_batch_dims: &[],
            rhs_batch_dims: &[],
        },
        1.0,
        0.0,
    )
    .unwrap_err();

    assert!(err.to_string().contains("output shape mismatch"));
}
