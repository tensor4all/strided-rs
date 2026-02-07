use approx::assert_abs_diff_eq;
use num_complex::Complex64;
use strided_opteinsum::{einsum, EinsumOperand};
use strided_view::{row_major_strides, StridedArray};

fn make_f64(dims: &[usize], data: Vec<f64>) -> EinsumOperand<'static> {
    let strides = row_major_strides(dims);
    StridedArray::from_parts(data, dims, &strides, 0)
        .unwrap()
        .into()
}

fn make_c64(dims: &[usize], data: Vec<Complex64>) -> EinsumOperand<'static> {
    let strides = row_major_strides(dims);
    StridedArray::from_parts(data, dims, &strides, 0)
        .unwrap()
        .into()
}

#[test]
fn test_matmul_e2e() {
    let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let result = einsum("ij,jk->ik", vec![a, b]).unwrap();
    match result {
        EinsumOperand::F64(data) => {
            let arr = data.as_array();
            assert_abs_diff_eq!(arr.get(&[0, 0]), 19.0);
            assert_abs_diff_eq!(arr.get(&[0, 1]), 22.0);
            assert_abs_diff_eq!(arr.get(&[1, 0]), 43.0);
            assert_abs_diff_eq!(arr.get(&[1, 1]), 50.0);
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_nested_chain_e2e() {
    let a = make_f64(&[2, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let b = make_f64(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let c = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let result = einsum("(ij,jk),kl->il", vec![a, b, c]).unwrap();
    match result {
        EinsumOperand::F64(data) => {
            let arr = data.as_array();
            assert_eq!(arr.dims(), &[2, 2]);
            // A*B = [[1*1+0+0, 1*2+0+0],[0+1*3+0, 0+1*4+0]] = [[1,2],[3,4]]
            // (A*B)*I = [[1,2],[3,4]]
            assert_abs_diff_eq!(arr.get(&[0, 0]), 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[0, 1]), 2.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[1, 0]), 3.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[1, 1]), 4.0, epsilon = 1e-10);
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_trace_e2e() {
    let a = make_f64(&[3, 3], vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    let result = einsum("ii->", vec![a]).unwrap();
    match result {
        EinsumOperand::F64(data) => {
            assert_abs_diff_eq!(data.as_array().data()[0], 6.0);
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_partial_trace_e2e() {
    // iij->j: sum over diagonal i
    let a = make_f64(&[2, 2, 3], (0..12).map(|x| x as f64).collect());
    // A[0,0,:] = [0,1,2], A[1,1,:] = [9,10,11]
    // result = [9, 11, 13]
    let result = einsum("iij->j", vec![a]).unwrap();
    match result {
        EinsumOperand::F64(data) => {
            let d = data.as_array().data().to_vec();
            let mut sorted = d.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_abs_diff_eq!(sorted[0], 9.0);
            assert_abs_diff_eq!(sorted[1], 11.0);
            assert_abs_diff_eq!(sorted[2], 13.0);
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_mixed_f64_c64_e2e() {
    let a = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]); // identity
    let b_data = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(3.0, 0.0),
    ];
    let b = make_c64(&[2, 2], b_data);
    let result = einsum("ij,jk->ik", vec![a, b]).unwrap();
    // I * B = B
    assert!(result.is_c64());
    match result {
        EinsumOperand::C64(data) => {
            let arr = data.as_array();
            assert_eq!(arr.dims(), &[2, 2]);
            assert_abs_diff_eq!(arr.get(&[0, 0]).re, 1.0);
            assert_abs_diff_eq!(arr.get(&[0, 0]).im, 1.0);
        }
        _ => panic!("expected C64"),
    }
}

#[test]
fn test_batch_matmul_e2e() {
    // bij,bjk->bik (batched matmul)
    let a = make_f64(
        &[2, 2, 2],
        vec![
            1.0, 0.0, 0.0, 1.0, // batch 0: identity
            2.0, 0.0, 0.0, 2.0, // batch 1: 2*identity
        ],
    );
    let b = make_f64(
        &[2, 2, 2],
        vec![
            1.0, 2.0, 3.0, 4.0, // batch 0
            5.0, 6.0, 7.0, 8.0, // batch 1
        ],
    );
    let result = einsum("bij,bjk->bik", vec![a, b]).unwrap();
    match result {
        EinsumOperand::F64(data) => {
            let arr = data.as_array();
            assert_eq!(arr.dims(), &[2, 2, 2]);
            // Batch 0: I * [[1,2],[3,4]] = [[1,2],[3,4]]
            assert_abs_diff_eq!(arr.get(&[0, 0, 0]), 1.0);
            assert_abs_diff_eq!(arr.get(&[0, 1, 1]), 4.0);
            // Batch 1: 2I * [[5,6],[7,8]] = [[10,12],[14,16]]
            assert_abs_diff_eq!(arr.get(&[1, 0, 0]), 10.0);
            assert_abs_diff_eq!(arr.get(&[1, 1, 1]), 16.0);
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_flat_three_tensor_e2e() {
    // ij,jk,kl->il (flat, omeco optimized)
    let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]); // identity
    let c = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    // A*I = A, A*C = [[19,22],[43,50]]
    let result = einsum("ij,jk,kl->il", vec![a, b, c]).unwrap();
    match result {
        EinsumOperand::F64(data) => {
            let arr = data.as_array();
            assert_abs_diff_eq!(arr.get(&[0, 0]), 19.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[1, 1]), 50.0, epsilon = 1e-10);
        }
        _ => panic!("expected F64"),
    }
}
