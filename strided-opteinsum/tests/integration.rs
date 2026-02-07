use approx::assert_abs_diff_eq;
use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};
use strided_opteinsum::{einsum, parse_einsum, EinsumNode, EinsumOperand};
use strided_view::{row_major_strides, StridedArray};

#[path = "support/loop_einsum.rs"]
mod loop_einsum;

use loop_einsum::{loop_einsum_complex, row_major_c64, row_major_f64, LoopTensor};

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

fn collect_leaf_ids(node: &EinsumNode, out: &mut Vec<Option<Vec<char>>>) {
    match node {
        EinsumNode::Leaf { ids, tensor_index } => {
            if *tensor_index >= out.len() {
                out.resize_with(*tensor_index + 1, || None);
            }
            out[*tensor_index] = Some(ids.clone());
        }
        EinsumNode::Contract { args } => {
            for arg in args {
                collect_leaf_ids(arg, out);
            }
        }
    }
}

fn leaf_ids_for_notation(notation: &str) -> Vec<Vec<char>> {
    let code = parse_einsum(notation).expect("notation should parse");
    let mut leaf_ids_opt = Vec::new();
    collect_leaf_ids(&code.root, &mut leaf_ids_opt);
    leaf_ids_opt
        .into_iter()
        .map(|x| x.expect("leaf index should be populated"))
        .collect()
}

fn to_complex_array(result: EinsumOperand<'static>) -> StridedArray<Complex64> {
    match result {
        EinsumOperand::F64(data) => {
            let arr = data.as_array();
            let cdata: Vec<Complex64> =
                arr.data().iter().map(|&x| Complex64::new(x, 0.0)).collect();
            StridedArray::from_parts(cdata, arr.dims(), arr.strides(), 0)
                .expect("f64->c64 conversion should preserve layout")
        }
        EinsumOperand::C64(data) => {
            let arr = data.as_array();
            StridedArray::from_parts(arr.data().to_vec(), arr.dims(), arr.strides(), 0)
                .expect("c64 clone should preserve layout")
        }
    }
}

fn for_each_index(dims: &[usize], mut f: impl FnMut(&[usize])) {
    fn rec(dims: &[usize], depth: usize, idx: &mut Vec<usize>, f: &mut dyn FnMut(&[usize])) {
        if depth == dims.len() {
            f(idx);
            return;
        }
        for i in 0..dims[depth] {
            idx.push(i);
            rec(dims, depth + 1, idx, f);
            idx.pop();
        }
    }
    let mut idx = Vec::new();
    rec(dims, 0, &mut idx, &mut f);
}

fn assert_complex_arrays_close(
    actual: &StridedArray<Complex64>,
    expected: &StridedArray<Complex64>,
    eps: f64,
    ctx: &str,
) {
    // Scalar representation can appear either as [] or [1] depending on allocation path.
    let scalar_like = |dims: &[usize]| dims.is_empty() || dims == [1];
    if scalar_like(actual.dims()) && scalar_like(expected.dims()) {
        let dr = (actual.data()[0].re - expected.data()[0].re).abs();
        let di = (actual.data()[0].im - expected.data()[0].im).abs();
        assert!(
            dr <= eps && di <= eps,
            "{ctx}: scalar mismatch, actual={:?}, expected={:?}, eps={eps}",
            actual.data()[0],
            expected.data()[0]
        );
        return;
    }
    assert!(
        actual.dims() == expected.dims()
            || (scalar_like(actual.dims()) && scalar_like(expected.dims())),
        "shape mismatch: actual={:?}, expected={:?}",
        actual.dims(),
        expected.dims()
    );
    for_each_index(actual.dims(), |idx| {
        let av = actual.get(idx);
        let ev = expected.get(idx);
        let dr = (av.re - ev.re).abs();
        let di = (av.im - ev.im).abs();
        assert!(
            dr <= eps && di <= eps,
            "{ctx}: mismatch at {:?}, actual={:?}, expected={:?}, eps={eps}",
            idx,
            av,
            ev
        );
    });
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

fn build_random_case(
    notation: &str,
    rng: &mut StdRng,
    mode: &str,
    allow_zero_dim: bool,
) -> (Vec<LoopTensor>, Vec<EinsumOperand<'static>>) {
    let leaf_ids = leaf_ids_for_notation(notation);

    let mut dim_map = std::collections::HashMap::<char, usize>::new();
    for ids in &leaf_ids {
        for &id in ids {
            dim_map.entry(id).or_insert_with(|| {
                if allow_zero_dim && rng.gen_bool(0.2) {
                    0
                } else {
                    rng.gen_range(1..=3)
                }
            });
        }
    }

    let mut loop_ops = Vec::with_capacity(leaf_ids.len());
    let mut eins_ops = Vec::with_capacity(leaf_ids.len());

    for ids in leaf_ids {
        let dims: Vec<usize> = ids.iter().map(|id| dim_map[id]).collect();
        let len = dims.iter().product();

        let is_complex = match mode {
            "f64" => false,
            "c64" => true,
            "mixed" => rng.gen_bool(0.5),
            _ => panic!("unknown mode"),
        };

        if is_complex {
            let data: Vec<Complex64> = (0..len)
                .map(|_| Complex64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)))
                .collect();
            loop_ops.push(row_major_c64(&dims, data.clone()));
            eins_ops.push(make_c64(&dims, data));
        } else {
            let data: Vec<f64> = (0..len).map(|_| rng.gen_range(-1.0..1.0)).collect();
            loop_ops.push(row_major_f64(&dims, data.clone()));
            eins_ops.push(make_f64(&dims, data));
        }
    }

    (loop_ops, eins_ops)
}

fn run_differential_case(
    notation: &str,
    rng: &mut StdRng,
    mode: &str,
    allow_zero_dim: bool,
    eps: f64,
) {
    let (loop_ops, eins_ops) = build_random_case(notation, rng, mode, allow_zero_dim);
    let expected =
        loop_einsum_complex(notation, &loop_ops).expect("reference evaluator should succeed");
    let actual =
        to_complex_array(einsum(notation, eins_ops).expect("strided-opteinsum should succeed"));
    let ctx = format!("notation={notation}, mode={mode}");
    assert_complex_arrays_close(&actual, &expected, eps, &ctx);
}

#[test]
fn test_differential_loop_einsum_f64() {
    let mut rng = StdRng::seed_from_u64(0x5EED_F64);
    let notations = ["ij,jk->ik", "ij,kj->ik", "i,j->ij", "i,i->", "bij,bjk->bik"];
    for _ in 0..30 {
        let notation = notations[rng.gen_range(0..notations.len())];
        run_differential_case(notation, &mut rng, "f64", false, 1e-10);
    }
}

#[test]
fn test_differential_loop_einsum_c64() {
    let mut rng = StdRng::seed_from_u64(0x5EED_C64);
    let notations = ["ij,jk->ik", "ij,kj->ik", "i,j->ij", "i,i->", "bij,bjk->bik"];
    for _ in 0..20 {
        let notation = notations[rng.gen_range(0..notations.len())];
        run_differential_case(notation, &mut rng, "c64", false, 1e-10);
    }
}
