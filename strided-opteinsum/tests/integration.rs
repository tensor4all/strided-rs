use approx::assert_abs_diff_eq;
use num_complex::Complex64;
use num_traits::Zero;
use rand::{rngs::StdRng, Rng, SeedableRng};
use strided_opteinsum::{einsum, einsum_into, parse_einsum, EinsumNode, EinsumOperand};
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

// ==========================================================================
// Tests for the top-level einsum() public API (lib.rs)
// ==========================================================================

#[test]
fn test_einsum_simple_matmul() {
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
fn test_einsum_single_tensor_permute() {
    let a = make_f64(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = einsum("ij->ji", vec![a]).unwrap();
    match result {
        EinsumOperand::F64(data) => {
            let arr = data.as_array();
            assert_eq!(arr.dims(), &[3, 2]);
            assert_abs_diff_eq!(arr.get(&[0, 0]), 1.0);
            assert_abs_diff_eq!(arr.get(&[0, 1]), 4.0);
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_einsum_scalar_output_trace() {
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
fn test_einsum_parse_error() {
    let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let err = einsum("ij,jk", vec![a]).unwrap_err();
    assert!(
        matches!(err, strided_opteinsum::EinsumError::ParseError(_)),
        "expected ParseError, got {:?}",
        err
    );
}

#[test]
fn test_einsum_operand_count_mismatch() {
    let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let err = einsum("ij,jk->ik", vec![a]).unwrap_err();
    assert!(matches!(
        err,
        strided_opteinsum::EinsumError::OperandCountMismatch {
            expected: 2,
            found: 1
        }
    ));
}

#[test]
fn test_einsum_nested_notation() {
    // Test that einsum parses nested notation correctly
    let a = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]); // identity
    let b = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let c = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let result = einsum("(ij,jk),kl->il", vec![a, b, c]).unwrap();
    match result {
        EinsumOperand::F64(data) => {
            let arr = data.as_array();
            // I*B = B, B*C = [[19,22],[43,50]]
            assert_abs_diff_eq!(arr.get(&[0, 0]), 19.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[1, 1]), 50.0, epsilon = 1e-10);
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_einsum_c64() {
    let c64 = |r, i| Complex64::new(r, i);
    let a = make_c64(
        &[2, 2],
        vec![c64(1.0, 1.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(1.0, 1.0)],
    );
    let b = make_c64(
        &[2, 2],
        vec![c64(2.0, 0.0), c64(3.0, 0.0), c64(4.0, 0.0), c64(5.0, 0.0)],
    );
    let result = einsum("ij,jk->ik", vec![a, b]).unwrap();
    assert!(result.is_c64());
    match result {
        EinsumOperand::C64(data) => {
            let arr = data.as_array();
            assert_eq!(arr.dims(), &[2, 2]);
            // (1+i)*[[2,3],[4,5]] = [[(1+i)*2, (1+i)*3],[(1+i)*4, (1+i)*5]]
            assert_abs_diff_eq!(arr.get(&[0, 0]).re, 2.0);
            assert_abs_diff_eq!(arr.get(&[0, 0]).im, 2.0);
        }
        _ => panic!("expected C64"),
    }
}

#[test]
fn test_einsum_mixed_f64_c64() {
    let a = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]); // identity, f64
    let c64 = |r, i| Complex64::new(r, i);
    let b = make_c64(
        &[2, 2],
        vec![c64(1.0, 2.0), c64(3.0, 4.0), c64(5.0, 6.0), c64(7.0, 8.0)],
    );
    let result = einsum("ij,jk->ik", vec![a, b]).unwrap();
    // I * B = B, result should be C64
    assert!(result.is_c64());
    match result {
        EinsumOperand::C64(data) => {
            let arr = data.as_array();
            assert_abs_diff_eq!(arr.get(&[0, 0]).re, 1.0);
            assert_abs_diff_eq!(arr.get(&[0, 0]).im, 2.0);
        }
        _ => panic!("expected C64"),
    }
}

// ==========================================================================
// Tests for the top-level einsum_into() public API (lib.rs)
// ==========================================================================

#[test]
fn test_einsum_into_f64_basic() {
    let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let mut c = StridedArray::<f64>::col_major(&[2, 2]);
    einsum_into("ij,jk->ik", vec![a, b], c.view_mut(), 1.0, 0.0).unwrap();
    assert_abs_diff_eq!(c.get(&[0, 0]), 19.0);
    assert_abs_diff_eq!(c.get(&[0, 1]), 22.0);
    assert_abs_diff_eq!(c.get(&[1, 0]), 43.0);
    assert_abs_diff_eq!(c.get(&[1, 1]), 50.0);
}

#[test]
fn test_einsum_into_f64_alpha_beta() {
    // result = 2 * (A*B) + 3 * C_old
    let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    // A*B = [[19,22],[43,50]]
    let mut c = StridedArray::<f64>::col_major(&[2, 2]);
    for v in c.data_mut().iter_mut() {
        *v = 1.0;
    }
    einsum_into("ij,jk->ik", vec![a, b], c.view_mut(), 2.0, 3.0).unwrap();
    // 2*19+3 = 41, 2*22+3 = 47, 2*43+3 = 89, 2*50+3 = 103
    assert_abs_diff_eq!(c.get(&[0, 0]), 41.0, epsilon = 1e-10);
    assert_abs_diff_eq!(c.get(&[0, 1]), 47.0, epsilon = 1e-10);
    assert_abs_diff_eq!(c.get(&[1, 0]), 89.0, epsilon = 1e-10);
    assert_abs_diff_eq!(c.get(&[1, 1]), 103.0, epsilon = 1e-10);
}

#[test]
fn test_einsum_into_c64_basic() {
    let c64 = |r| Complex64::new(r, 0.0);
    let a = make_c64(&[2, 2], vec![c64(1.0), c64(2.0), c64(3.0), c64(4.0)]);
    let b = make_c64(&[2, 2], vec![c64(5.0), c64(6.0), c64(7.0), c64(8.0)]);
    let mut out = StridedArray::<Complex64>::col_major(&[2, 2]);
    einsum_into(
        "ij,jk->ik",
        vec![a, b],
        out.view_mut(),
        c64(1.0),
        Complex64::zero(),
    )
    .unwrap();
    assert_abs_diff_eq!(out.get(&[0, 0]).re, 19.0);
    assert_abs_diff_eq!(out.get(&[1, 1]).re, 50.0);
}

#[test]
fn test_einsum_into_mixed_types_c64_output() {
    // f64 operand + c64 operand -> c64 output
    let a = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]); // identity
    let c64 = |r| Complex64::new(r, 0.0);
    let b = make_c64(&[2, 2], vec![c64(5.0), c64(6.0), c64(7.0), c64(8.0)]);
    let mut out = StridedArray::<Complex64>::col_major(&[2, 2]);
    einsum_into(
        "ij,jk->ik",
        vec![a, b],
        out.view_mut(),
        c64(1.0),
        Complex64::zero(),
    )
    .unwrap();
    // I * B = B
    assert_abs_diff_eq!(out.get(&[0, 0]).re, 5.0);
    assert_abs_diff_eq!(out.get(&[0, 1]).re, 6.0);
    assert_abs_diff_eq!(out.get(&[1, 0]).re, 7.0);
    assert_abs_diff_eq!(out.get(&[1, 1]).re, 8.0);
}

#[test]
fn test_einsum_into_parse_error() {
    let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut out = StridedArray::<f64>::col_major(&[2, 2]);
    let err = einsum_into("ij,jk", vec![a], out.view_mut(), 1.0, 0.0).unwrap_err();
    assert!(matches!(err, strided_opteinsum::EinsumError::ParseError(_)));
}

#[test]
fn test_einsum_into_type_mismatch_f64_output_c64_operand() {
    let c64 = |r| Complex64::new(r, 0.0);
    let a = make_c64(&[2, 2], vec![c64(1.0), c64(2.0), c64(3.0), c64(4.0)]);
    let b = make_c64(&[2, 2], vec![c64(5.0), c64(6.0), c64(7.0), c64(8.0)]);
    let mut out = StridedArray::<f64>::col_major(&[2, 2]);
    let err = einsum_into("ij,jk->ik", vec![a, b], out.view_mut(), 1.0, 0.0).unwrap_err();
    assert!(matches!(
        err,
        strided_opteinsum::EinsumError::TypeMismatch { .. }
    ));
}

#[test]
fn test_einsum_into_shape_mismatch() {
    let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let mut out = StridedArray::<f64>::col_major(&[3, 3]); // wrong shape
    let err = einsum_into("ij,jk->ik", vec![a, b], out.view_mut(), 1.0, 0.0).unwrap_err();
    assert!(matches!(
        err,
        strided_opteinsum::EinsumError::OutputShapeMismatch { .. }
    ));
}

#[test]
fn test_einsum_into_operand_count_mismatch() {
    let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut out = StridedArray::<f64>::col_major(&[2, 2]);
    let err = einsum_into("ij,jk->ik", vec![a], out.view_mut(), 1.0, 0.0).unwrap_err();
    assert!(matches!(
        err,
        strided_opteinsum::EinsumError::OperandCountMismatch {
            expected: 2,
            found: 1
        }
    ));
}

#[test]
fn test_einsum_into_single_tensor_identity() {
    // ij->ij is identity (no permute, no trace)
    let a = make_f64(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut out = StridedArray::<f64>::col_major(&[2, 3]);
    einsum_into("ij->ij", vec![a], out.view_mut(), 1.0, 0.0).unwrap();
    assert_abs_diff_eq!(out.get(&[0, 0]), 1.0);
    assert_abs_diff_eq!(out.get(&[0, 2]), 3.0);
    assert_abs_diff_eq!(out.get(&[1, 0]), 4.0);
    assert_abs_diff_eq!(out.get(&[1, 2]), 6.0);
}

#[test]
fn test_einsum_into_single_tensor_permute() {
    let a = make_f64(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut out = StridedArray::<f64>::col_major(&[3, 2]);
    einsum_into("ij->ji", vec![a], out.view_mut(), 1.0, 0.0).unwrap();
    assert_abs_diff_eq!(out.get(&[0, 0]), 1.0);
    assert_abs_diff_eq!(out.get(&[0, 1]), 4.0);
    assert_abs_diff_eq!(out.get(&[1, 0]), 2.0);
    assert_abs_diff_eq!(out.get(&[2, 1]), 6.0);
}

#[test]
fn test_einsum_into_single_tensor_trace() {
    let a = make_f64(&[3, 3], vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    let mut out = StridedArray::<f64>::col_major(&[]);
    einsum_into("ii->", vec![a], out.view_mut(), 1.0, 0.0).unwrap();
    assert_abs_diff_eq!(out.data()[0], 6.0);
}

#[test]
fn test_einsum_into_three_tensor_omeco() {
    let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]); // identity
    let c_op = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let mut out = StridedArray::<f64>::col_major(&[2, 2]);
    einsum_into("ij,jk,kl->il", vec![a, b, c_op], out.view_mut(), 1.0, 0.0).unwrap();
    // A*I = A, A*C = [[19,22],[43,50]]
    assert_abs_diff_eq!(out.get(&[0, 0]), 19.0, epsilon = 1e-10);
    assert_abs_diff_eq!(out.get(&[1, 1]), 50.0, epsilon = 1e-10);
}

#[test]
fn test_einsum_into_nested_notation() {
    let a = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]); // identity
    let b = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let c_op = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let mut out = StridedArray::<f64>::col_major(&[2, 2]);
    einsum_into("(ij,jk),kl->il", vec![a, b, c_op], out.view_mut(), 1.0, 0.0).unwrap();
    // I*B = B, B*C = [[19,22],[43,50]]
    assert_abs_diff_eq!(out.get(&[0, 0]), 19.0, epsilon = 1e-10);
    assert_abs_diff_eq!(out.get(&[1, 1]), 50.0, epsilon = 1e-10);
}

#[test]
fn test_einsum_into_dot_product() {
    let a = make_f64(&[3], vec![1.0, 2.0, 3.0]);
    let b = make_f64(&[3], vec![4.0, 5.0, 6.0]);
    let mut out = StridedArray::<f64>::col_major(&[]);
    einsum_into("i,i->", vec![a, b], out.view_mut(), 1.0, 0.0).unwrap();
    // 1*4 + 2*5 + 3*6 = 32
    assert_abs_diff_eq!(out.data()[0], 32.0);
}

#[test]
fn test_einsum_into_outer_product() {
    let a = make_f64(&[2], vec![3.0, 5.0]);
    let b = make_f64(&[3], vec![10.0, 20.0, 30.0]);
    let mut out = StridedArray::<f64>::col_major(&[2, 3]);
    einsum_into("i,j->ij", vec![a, b], out.view_mut(), 1.0, 0.0).unwrap();
    assert_abs_diff_eq!(out.get(&[0, 0]), 30.0);
    assert_abs_diff_eq!(out.get(&[0, 2]), 90.0);
    assert_abs_diff_eq!(out.get(&[1, 0]), 50.0);
    assert_abs_diff_eq!(out.get(&[1, 2]), 150.0);
}

#[test]
fn test_einsum_into_single_tensor_trace_alpha_beta() {
    // Test alpha/beta with single-tensor path through evaluate_into (Leaf branch)
    let a = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 2.0]);
    let mut out = StridedArray::<f64>::col_major(&[]);
    out.data_mut()[0] = 10.0;
    // out = alpha * trace(A) + beta * out = 2 * 3 + 5 * 10 = 56
    einsum_into("ii->", vec![a], out.view_mut(), 2.0, 5.0).unwrap();
    assert_abs_diff_eq!(out.data()[0], 56.0, epsilon = 1e-10);
}

// ==========================================================================
// Additional StridedData coverage via EinsumOperand
// ==========================================================================

#[test]
fn test_einsum_operand_from_view_f64_roundtrip() {
    let mut arr = StridedArray::<f64>::col_major(&[2, 3]);
    for (i, v) in arr.data_mut().iter_mut().enumerate() {
        *v = (i as f64) * 10.0;
    }
    let view = arr.view();
    let op = EinsumOperand::from_view(&view);
    assert!(op.is_f64());
    assert_eq!(op.dims(), &[2, 3]);

    // Use it in an einsum to prove it works through the pipeline
    let result = einsum("ij->ji", vec![op]).unwrap();
    assert!(result.is_f64());
    assert_eq!(result.dims(), &[3, 2]);
}

#[test]
fn test_einsum_operand_from_view_c64_roundtrip() {
    let mut arr = StridedArray::<Complex64>::col_major(&[2, 2]);
    arr.data_mut()[0] = Complex64::new(1.0, 2.0);
    arr.data_mut()[1] = Complex64::new(3.0, 4.0);
    arr.data_mut()[2] = Complex64::new(5.0, 6.0);
    arr.data_mut()[3] = Complex64::new(7.0, 8.0);
    let view = arr.view();
    let op = EinsumOperand::from_view(&view);
    assert!(op.is_c64());
    assert_eq!(op.dims(), &[2, 2]);

    let result = einsum("ij->ji", vec![op]).unwrap();
    assert!(result.is_c64());
    assert_eq!(result.dims(), &[2, 2]);
}

// ==========================================================================
// expr.rs coverage: Complex64 operand paths
// ==========================================================================

#[test]
fn test_eval_pair_c64_c64_matmul() {
    // Exercises eval_pair C64/C64 branch (lines 200-266 of expr.rs)
    let c64 = |r, i| Complex64::new(r, i);
    let a = make_c64(
        &[2, 2],
        vec![c64(1.0, 1.0), c64(2.0, 0.0), c64(0.0, 1.0), c64(3.0, -1.0)],
    );
    let b = make_c64(
        &[2, 2],
        vec![c64(1.0, 0.0), c64(0.0, 1.0), c64(-1.0, 0.0), c64(2.0, 0.0)],
    );
    let result = einsum("ij,jk->ik", vec![a, b]).unwrap();
    assert!(result.is_c64());
    // Verify correct complex multiplication
    // C[0,0] = (1+i)*1 + 2*(-1) = 1+i-2 = -1+i
    // C[0,1] = (1+i)*(0+i) + 2*2 = (0+i+i*i)+4 = (-1+i)+4 = 3+i  -- wait: i^2=-1
    // (1+i)*(i) = i + i^2 = i - 1 = -1+i
    // C[0,1] = -1+i + 4 = 3+i
    match result {
        EinsumOperand::C64(data) => {
            let arr = data.as_array();
            assert_abs_diff_eq!(arr.get(&[0, 0]).re, -1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[0, 0]).im, 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[0, 1]).re, 3.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[0, 1]).im, 1.0, epsilon = 1e-10);
        }
        _ => panic!("expected C64"),
    }
}

#[test]
fn test_eval_single_c64_trace() {
    // Exercises eval_single C64 branch (lines 397-400 of expr.rs)
    let c64 = |r, i| Complex64::new(r, i);
    let a = make_c64(
        &[3, 3],
        vec![
            c64(1.0, 1.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(2.0, -1.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(3.0, 2.0),
        ],
    );
    let result = einsum("ii->", vec![a]).unwrap();
    assert!(result.is_c64());
    match result {
        EinsumOperand::C64(data) => {
            let arr = data.as_array();
            // trace = (1+i) + (2-i) + (3+2i) = 6+2i
            assert_abs_diff_eq!(arr.data()[0].re, 6.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.data()[0].im, 2.0, epsilon = 1e-10);
        }
        _ => panic!("expected C64"),
    }
}

#[test]
fn test_eval_single_c64_permute() {
    // Exercises eval_single C64 branch with permutation
    let c64 = |r, i| Complex64::new(r, i);
    let a = make_c64(
        &[2, 3],
        vec![
            c64(1.0, 0.0),
            c64(2.0, 1.0),
            c64(3.0, -1.0),
            c64(4.0, 2.0),
            c64(5.0, 0.0),
            c64(6.0, 3.0),
        ],
    );
    let result = einsum("ij->ji", vec![a]).unwrap();
    assert!(result.is_c64());
    match result {
        EinsumOperand::C64(data) => {
            let arr = data.as_array();
            assert_eq!(arr.dims(), &[3, 2]);
            assert_abs_diff_eq!(arr.get(&[0, 0]).re, 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[0, 1]).re, 4.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[0, 1]).im, 2.0, epsilon = 1e-10);
        }
        _ => panic!("expected C64"),
    }
}

#[test]
fn test_eval_single_c64_sum_axis() {
    // Exercises eval_single C64 with axis reduction (ij->i)
    let c64 = |r, i| Complex64::new(r, i);
    let a = make_c64(
        &[2, 3],
        vec![
            c64(1.0, 1.0),
            c64(2.0, 2.0),
            c64(3.0, 3.0),
            c64(4.0, -1.0),
            c64(5.0, -2.0),
            c64(6.0, -3.0),
        ],
    );
    let result = einsum("ij->i", vec![a]).unwrap();
    assert!(result.is_c64());
    match result {
        EinsumOperand::C64(data) => {
            let arr = data.as_array();
            assert_eq!(arr.dims(), &[2]);
            // row 0: (1+i)+(2+2i)+(3+3i) = 6+6i
            assert_abs_diff_eq!(arr.data()[0].re, 6.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.data()[0].im, 6.0, epsilon = 1e-10);
            // row 1: (4-i)+(5-2i)+(6-3i) = 15-6i
            assert_abs_diff_eq!(arr.data()[1].re, 15.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.data()[1].im, -6.0, epsilon = 1e-10);
        }
        _ => panic!("expected C64"),
    }
}

#[test]
fn test_mixed_f64_c64_eval_pair_promotion() {
    // Exercises eval_pair mixed-type promotion branch (lines 267-272 of expr.rs)
    // Use identity * B so the result must equal B regardless of internal layout
    let a = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]); // identity
    let c64 = |r, i| Complex64::new(r, i);
    let b = make_c64(
        &[2, 2],
        vec![c64(1.0, 2.0), c64(3.0, 4.0), c64(5.0, 6.0), c64(7.0, 8.0)],
    );
    // I * B = B
    let result = einsum("ij,jk->ik", vec![a, b]).unwrap();
    assert!(result.is_c64());
    match result {
        EinsumOperand::C64(data) => {
            let arr = data.as_array();
            assert_abs_diff_eq!(arr.get(&[0, 0]).re, 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[0, 0]).im, 2.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[0, 1]).re, 3.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[0, 1]).im, 4.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[1, 0]).re, 5.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[1, 0]).im, 6.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[1, 1]).re, 7.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[1, 1]).im, 8.0, epsilon = 1e-10);
        }
        _ => panic!("expected C64"),
    }
}

// ==========================================================================
// expr.rs coverage: accumulate_into paths via evaluate_into
// ==========================================================================

#[test]
fn test_accumulate_into_copy_scale_path() {
    // alpha != 1, beta == 0 -> exercises copy_scale branch (line 353 of expr.rs)
    let a = make_f64(&[2, 2], vec![1.0, 0.0, 0.0, 2.0]);
    let mut out = StridedArray::<f64>::col_major(&[]);
    out.data_mut()[0] = 999.0; // should be overwritten
                               // out = 3 * trace(A) + 0 * out = 3 * 3 = 9
    einsum_into("ii->", vec![a], out.view_mut(), 3.0, 0.0).unwrap();
    assert_abs_diff_eq!(out.data()[0], 9.0, epsilon = 1e-10);
}

#[test]
fn test_accumulate_into_general_path() {
    // alpha != 1, beta != 0 -> exercises general path (lines 356-378 of expr.rs)
    let a = make_f64(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut out = StridedArray::<f64>::col_major(&[3, 2]);
    // Fill output with known values
    for (i, v) in out.data_mut().iter_mut().enumerate() {
        *v = (i + 1) as f64;
    }
    // out = 2 * transpose(A) + 3 * out_old
    einsum_into("ij->ji", vec![a], out.view_mut(), 2.0, 3.0).unwrap();
    // Verify: out[0,0] = 2*A[0,0] + 3*old[0,0]
    // A transposed: [[1,4],[2,5],[3,6]]
    // old (col-major [3,2]): data = [1,2,3,4,5,6] -> col-major: [0,0]=1, [1,0]=2, [2,0]=3, [0,1]=4, [1,1]=5, [2,1]=6
    // out[0,0] = 2*1 + 3*1 = 5
    assert_abs_diff_eq!(out.get(&[0, 0]), 5.0, epsilon = 1e-10);
    // out[1,0] = 2*2 + 3*2 = 10
    assert_abs_diff_eq!(out.get(&[1, 0]), 10.0, epsilon = 1e-10);
    // out[0,1] = 2*4 + 3*4 = 20
    assert_abs_diff_eq!(out.get(&[0, 1]), 20.0, epsilon = 1e-10);
}

#[test]
fn test_evaluate_into_single_child_contract_trace() {
    // Exercises evaluate_into Contract with 1 child general trace/reduction path
    // (lines 812-816 of expr.rs)
    // Use parenthesized notation to create a Contract node wrapping a single Leaf
    // that requires trace (not permutation-only)
    // "ii->" wrapped in a Contract: the inner Contract node reduces to scalar
    let a = make_f64(&[3, 3], vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    // Use parenthesized notation that creates a nested Contract around the trace
    let mut out = StridedArray::<f64>::col_major(&[]);
    einsum_into("ii->", vec![a], out.view_mut(), 1.0, 0.0).unwrap();
    assert_abs_diff_eq!(out.data()[0], 6.0, epsilon = 1e-10);
}

#[test]
fn test_evaluate_into_single_child_contract_sum_axis() {
    // Exercise evaluate_into Contract with 1 child that requires axis reduction
    let a = make_f64(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut out = StridedArray::<f64>::col_major(&[2]);
    // ij->i : sum over j
    einsum_into("ij->i", vec![a], out.view_mut(), 1.0, 0.0).unwrap();
    // row 0: 1+2+3 = 6, row 1: 4+5+6 = 15
    assert_abs_diff_eq!(out.data()[0], 6.0, epsilon = 1e-10);
    assert_abs_diff_eq!(out.data()[1], 15.0, epsilon = 1e-10);
}

#[test]
fn test_c64_three_tensor_omeco() {
    // Exercise the C64/C64 branch in eval_pair with 3+ tensor omeco path
    let c64 = |r, i| Complex64::new(r, i);
    let a = make_c64(
        &[2, 2],
        vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(1.0, 0.0)], // identity
    );
    let b = make_c64(
        &[2, 2],
        vec![c64(1.0, 1.0), c64(2.0, 0.0), c64(3.0, -1.0), c64(4.0, 0.0)],
    );
    let c_op = make_c64(
        &[2, 2],
        vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(1.0, 0.0)], // identity
    );
    // I * B * I = B
    let result = einsum("ij,jk,kl->il", vec![a, b, c_op]).unwrap();
    assert!(result.is_c64());
    match result {
        EinsumOperand::C64(data) => {
            let arr = data.as_array();
            assert_abs_diff_eq!(arr.get(&[0, 0]).re, 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[0, 0]).im, 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[1, 1]).re, 4.0, epsilon = 1e-10);
        }
        _ => panic!("expected C64"),
    }
}

#[test]
fn test_evaluate_into_c64_alpha_beta() {
    // Exercise accumulate_into with Complex64 alpha/beta
    let c64 = |r, i| Complex64::new(r, i);
    let a = make_c64(
        &[2, 2],
        vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(2.0, 0.0)],
    );
    let mut out = StridedArray::<Complex64>::col_major(&[]);
    out.data_mut()[0] = c64(10.0, 0.0);
    // out = (2+i) * trace(A) + (1+0i) * out_old
    // trace = 3+0i
    // result = (2+i)*(3) + (1)*(10) = (6+3i) + 10 = 16+3i
    einsum_into(
        "ii->",
        vec![a],
        out.view_mut(),
        c64(2.0, 1.0),
        c64(1.0, 0.0),
    )
    .unwrap();
    assert_abs_diff_eq!(out.data()[0].re, 16.0, epsilon = 1e-10);
    assert_abs_diff_eq!(out.data()[0].im, 3.0, epsilon = 1e-10);
}

// ==========================================================================
// single_tensor.rs coverage: additional paths
// ==========================================================================

#[test]
fn test_single_tensor_c64_trace() {
    // Exercise single_tensor_einsum with Complex64 (full trace)
    let c64 = |r, i| Complex64::new(r, i);
    let a = make_c64(
        &[2, 2],
        vec![c64(1.0, 1.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(3.0, -2.0)],
    );
    let result = einsum("ii->", vec![a]).unwrap();
    assert!(result.is_c64());
    match result {
        EinsumOperand::C64(data) => {
            let arr = data.as_array();
            // trace = (1+i) + (3-2i) = 4-i
            assert_abs_diff_eq!(arr.data()[0].re, 4.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.data()[0].im, -1.0, epsilon = 1e-10);
        }
        _ => panic!("expected C64"),
    }
}

#[test]
fn test_single_tensor_c64_partial_trace() {
    // Exercise single_tensor_einsum with Complex64 (partial trace iij->j)
    let c64 = |r, i| Complex64::new(r, i);
    let a = make_c64(
        &[2, 2, 3],
        vec![
            // A[0,0,:] = [1+i, 2+i, 3+i]
            c64(1.0, 1.0),
            c64(2.0, 1.0),
            c64(3.0, 1.0),
            // A[0,1,:] = [0, 0, 0]
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            // A[1,0,:] = [0, 0, 0]
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            // A[1,1,:] = [4-i, 5-i, 6-i]
            c64(4.0, -1.0),
            c64(5.0, -1.0),
            c64(6.0, -1.0),
        ],
    );
    let result = einsum("iij->j", vec![a]).unwrap();
    assert!(result.is_c64());
    match result {
        EinsumOperand::C64(data) => {
            let arr = data.as_array();
            assert_eq!(arr.len(), 3);
            // result[j] = A[0,0,j] + A[1,1,j]
            // j=0: (1+i)+(4-i) = 5
            // j=1: (2+i)+(5-i) = 7
            // j=2: (3+i)+(6-i) = 9
            let mut vals: Vec<(f64, f64)> = (0..3)
                .map(|j| (arr.data()[j].re, arr.data()[j].im))
                .collect();
            vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            assert_abs_diff_eq!(vals[0].0, 5.0, epsilon = 1e-10);
            assert_abs_diff_eq!(vals[0].1, 0.0, epsilon = 1e-10);
            assert_abs_diff_eq!(vals[1].0, 7.0, epsilon = 1e-10);
            assert_abs_diff_eq!(vals[2].0, 9.0, epsilon = 1e-10);
        }
        _ => panic!("expected C64"),
    }
}

#[test]
fn test_single_tensor_multi_pair_diagonal() {
    // Exercise the general fallback path in single_tensor.rs (lines 135-207)
    // with multiple repeated index pairs: iijj-> (double trace)
    // This has TWO pairs: (i,i) and (j,j), so has_triple triggers and falls through
    // to the general fallback path.
    let n = 3;
    let total = n * n * n * n;
    let data: Vec<f64> = (0..total).map(|x| x as f64).collect();
    let a = make_f64(&[n, n, n, n], data);
    // iijj -> : sum of A[i,i,j,j] for all i,j
    let result = einsum("iijj->", vec![a]).unwrap();
    match result {
        EinsumOperand::F64(data) => {
            let arr = data.as_array();
            let mut expected = 0.0;
            for i in 0..n {
                for j in 0..n {
                    // A[i,i,j,j] in row-major [n,n,n,n]:
                    // flat index = i*n^3 + i*n^2 + j*n + j = i*(n^3+n^2) + j*(n+1)
                    let flat = i * n * n * n + i * n * n + j * n + j;
                    expected += flat as f64;
                }
            }
            assert_abs_diff_eq!(arr.data()[0], expected, epsilon = 1e-10);
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_single_tensor_identity_copy() {
    // Exercise the identity case at line 279-291 of single_tensor.rs
    // where current_ids == output_ids, no diagonal, no reduction => copy src
    let a = make_f64(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = einsum("ij->ij", vec![a]).unwrap();
    match result {
        EinsumOperand::F64(data) => {
            let arr = data.as_array();
            assert_eq!(arr.dims(), &[2, 3]);
            assert_abs_diff_eq!(arr.get(&[0, 0]), 1.0);
            assert_abs_diff_eq!(arr.get(&[0, 1]), 2.0);
            assert_abs_diff_eq!(arr.get(&[0, 2]), 3.0);
            assert_abs_diff_eq!(arr.get(&[1, 0]), 4.0);
            assert_abs_diff_eq!(arr.get(&[1, 1]), 5.0);
            assert_abs_diff_eq!(arr.get(&[1, 2]), 6.0);
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_single_tensor_diagonal_and_reduce() {
    // Exercise the diagonal-then-reduce path (lines 180-200 of single_tensor.rs)
    // ijj->i: extract diagonal on axes 1,2 then the result is already in output order
    let data: Vec<f64> = (0..18).map(|x| x as f64).collect();
    let a = make_f64(&[2, 3, 3], data);
    // A[i,j,j] -> result[i] = sum_j A[i,j,j]
    let result = einsum("ijj->i", vec![a]).unwrap();
    match result {
        EinsumOperand::F64(rdata) => {
            let arr = rdata.as_array();
            assert_eq!(arr.dims(), &[2]);
            // A[0,0,0]=0, A[0,1,1]=4, A[0,2,2]=8 -> sum=12
            assert_abs_diff_eq!(arr.data()[0], 12.0, epsilon = 1e-10);
            // A[1,0,0]=9, A[1,1,1]=13, A[1,2,2]=17 -> sum=39
            assert_abs_diff_eq!(arr.data()[1], 39.0, epsilon = 1e-10);
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_single_tensor_diagonal_no_reduce() {
    // Exercise diagonal extraction without reduction (ijj->ij at lines 201-206)
    let data: Vec<f64> = (0..18).map(|x| x as f64).collect();
    let a = make_f64(&[2, 3, 3], data);
    // ijj->ij: just extract diagonal (no reduction)
    let result = einsum("ijj->ij", vec![a]).unwrap();
    match result {
        EinsumOperand::F64(rdata) => {
            let arr = rdata.as_array();
            assert_eq!(arr.dims(), &[2, 3]);
            // A[0,0,0]=0, A[0,1,1]=4, A[0,2,2]=8
            assert_abs_diff_eq!(arr.get(&[0, 0]), 0.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[0, 1]), 4.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[0, 2]), 8.0, epsilon = 1e-10);
            // A[1,0,0]=9, A[1,1,1]=13, A[1,2,2]=17
            assert_abs_diff_eq!(arr.get(&[1, 0]), 9.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[1, 1]), 13.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[1, 2]), 17.0, epsilon = 1e-10);
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_single_tensor_diagonal_and_reduce_with_permute() {
    // Exercise diagonal + reduce + permute path
    // iij->j uses the partial trace fast path; use jii->j to test with permutation needed
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let a = make_f64(&[3, 2, 2], data);
    // jii-> : sum over diagonal i of A[j,i,i], result is vector of j
    let result = einsum("jii->j", vec![a]).unwrap();
    match result {
        EinsumOperand::F64(rdata) => {
            let arr = rdata.as_array();
            assert_eq!(arr.dims(), &[3]);
            // A[0,0,0]=0, A[0,1,1]=3 -> 3
            // A[1,0,0]=4, A[1,1,1]=7 -> 11
            // A[2,0,0]=8, A[2,1,1]=11 -> 19
            let mut vals: Vec<f64> = arr.data().to_vec();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_abs_diff_eq!(vals[0], 3.0, epsilon = 1e-10);
            assert_abs_diff_eq!(vals[1], 11.0, epsilon = 1e-10);
            assert_abs_diff_eq!(vals[2], 19.0, epsilon = 1e-10);
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_single_tensor_reduce_without_diagonal() {
    // Exercise the reduction-only path (no diagonal, lines 222-234) in single_tensor.rs
    // ij->j: sum over first axis. No repeated indices.
    let a = make_f64(&[3, 4], (0..12).map(|x| x as f64).collect());
    let result = einsum("ij->j", vec![a]).unwrap();
    match result {
        EinsumOperand::F64(rdata) => {
            let arr = rdata.as_array();
            assert_eq!(arr.dims(), &[4]);
            // col j: sum over rows = sum(0+j, 4+j, 8+j) = 12+3j
            for j in 0..4 {
                let expected = (0 + j) as f64 + (4 + j) as f64 + (8 + j) as f64;
                assert_abs_diff_eq!(arr.data()[j], expected, epsilon = 1e-10);
            }
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_differential_loop_einsum_c64_single_tensor() {
    // Run differential testing with single-tensor C64 operations
    let mut rng = StdRng::seed_from_u64(0xC64_5171);
    let notations = ["ij->ji", "ii->", "ij->i", "ij->j"];
    for _ in 0..20 {
        let notation = notations[rng.gen_range(0..notations.len())];
        run_differential_case(notation, &mut rng, "c64", false, 1e-10);
    }
}

#[test]
fn test_mixed_c64_f64_dot_product() {
    // Exercises the mixed-type promotion code path (eval_pair lines 267-272)
    // using a dot product which is layout-agnostic (scalar result).
    let c64 = |r, i| Complex64::new(r, i);
    let a = make_c64(&[3], vec![c64(1.0, 1.0), c64(2.0, 0.0), c64(0.0, 3.0)]);
    let b = make_f64(&[3], vec![1.0, 2.0, 3.0]);
    // dot = (1+i)*1 + 2*2 + (0+3i)*3 = (1+i) + 4 + 9i = 5 + 10i
    let result = einsum("i,i->", vec![a, b]).unwrap();
    assert!(result.is_c64());
    match result {
        EinsumOperand::C64(data) => {
            let arr = data.as_array();
            assert_abs_diff_eq!(arr.data()[0].re, 5.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.data()[0].im, 10.0, epsilon = 1e-10);
        }
        _ => panic!("expected C64"),
    }
}

// ============================================================================
// View-based operand tests: exercise (Owned,View), (View,Owned), (View,View)
// branches in eval_pair (lines 161-195)
// ============================================================================

#[test]
fn test_eval_pair_view_view_f64() {
    // Both operands as View — covers (View, View) branch in eval_pair F64
    let a_arr = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
        [1.0, 2.0, 3.0, 4.0][idx[0] * 2 + idx[1]]
    });
    let b_arr = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
        [5.0, 6.0, 7.0, 8.0][idx[0] * 2 + idx[1]]
    });
    let a = EinsumOperand::from_view(&a_arr.view());
    let b = EinsumOperand::from_view(&b_arr.view());
    let result = einsum("ij,jk->ik", vec![a, b]).unwrap();
    match result {
        EinsumOperand::F64(data) => {
            let arr = data.as_array();
            assert_abs_diff_eq!(arr.get(&[0, 0]), 19.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[1, 1]), 50.0, epsilon = 1e-10);
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_eval_pair_owned_view_f64() {
    // Left Owned, Right View — covers (Owned, View) branch
    let a = make_f64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b_arr = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
        [5.0, 6.0, 7.0, 8.0][idx[0] * 2 + idx[1]]
    });
    let b = EinsumOperand::from_view(&b_arr.view());
    let result = einsum("ij,jk->ik", vec![a, b]).unwrap();
    match result {
        EinsumOperand::F64(data) => {
            let arr = data.as_array();
            assert_abs_diff_eq!(arr.get(&[0, 0]), 19.0, epsilon = 1e-10);
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_eval_pair_view_owned_f64() {
    // Left View, Right Owned — covers (View, Owned) branch
    let a_arr = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
        [1.0, 2.0, 3.0, 4.0][idx[0] * 2 + idx[1]]
    });
    let a = EinsumOperand::from_view(&a_arr.view());
    let b = make_f64(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let result = einsum("ij,jk->ik", vec![a, b]).unwrap();
    match result {
        EinsumOperand::F64(data) => {
            let arr = data.as_array();
            assert_abs_diff_eq!(arr.get(&[0, 0]), 19.0, epsilon = 1e-10);
        }
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_eval_pair_view_view_c64() {
    // Both C64 Views — covers (View, View) branch in eval_pair C64
    let c64 = |r, i| Complex64::new(r, i);
    let a_arr = StridedArray::<Complex64>::from_fn_row_major(&[2, 2], |idx| {
        [c64(1.0, 0.0), c64(2.0, 0.0), c64(3.0, 0.0), c64(4.0, 0.0)][idx[0] * 2 + idx[1]]
    });
    let b_arr = StridedArray::<Complex64>::from_fn_row_major(&[2, 2], |idx| {
        [c64(5.0, 0.0), c64(6.0, 0.0), c64(7.0, 0.0), c64(8.0, 0.0)][idx[0] * 2 + idx[1]]
    });
    let a = EinsumOperand::from_view(&a_arr.view());
    let b = EinsumOperand::from_view(&b_arr.view());
    let result = einsum("ij,jk->ik", vec![a, b]).unwrap();
    match result {
        EinsumOperand::C64(data) => {
            let arr = data.as_array();
            assert_abs_diff_eq!(arr.get(&[0, 0]).re, 19.0, epsilon = 1e-10);
            assert_abs_diff_eq!(arr.get(&[1, 1]).re, 50.0, epsilon = 1e-10);
        }
        _ => panic!("expected C64"),
    }
}

// ============================================================================
// evaluate_into with 3+ tensors (omeco path, lines 836-877)
// ============================================================================

#[test]
fn test_einsum_into_three_tensor_flat_omeco() {
    // Flat three-tensor: "ij,jk,kl->il" — exercises evaluate_into 3+ children path
    let a = make_f64(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = make_f64(
        &[3, 4],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    );
    let c = make_f64(&[4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let mut out = StridedArray::<f64>::row_major(&[2, 2]);
    einsum_into("ij,jk,kl->il", vec![a, b, c], out.view_mut(), 1.0, 0.0).unwrap();

    // A*B = [[1,2,3,0],[4,5,6,0]], (A*B)*C = [[1*1+2*3+3*5, 1*2+2*4+3*6],[4*1+5*3+6*5, 4*2+5*4+6*6]]
    //     = [[1+6+15, 2+8+18],[4+15+30, 8+20+36]] = [[22,28],[49,64]]
    assert_abs_diff_eq!(out.get(&[0, 0]), 22.0, epsilon = 1e-10);
    assert_abs_diff_eq!(out.get(&[0, 1]), 28.0, epsilon = 1e-10);
    assert_abs_diff_eq!(out.get(&[1, 0]), 49.0, epsilon = 1e-10);
    assert_abs_diff_eq!(out.get(&[1, 1]), 64.0, epsilon = 1e-10);
}

// ============================================================================
// evaluate_into with View operands — covers eval_pair_into View branches
// ============================================================================

#[test]
fn test_einsum_into_view_operands() {
    // View-based operands through einsum_into — covers eval_pair_into (View, View) branch
    let a_arr = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
        [1.0, 2.0, 3.0, 4.0][idx[0] * 2 + idx[1]]
    });
    let b_arr = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
        [5.0, 6.0, 7.0, 8.0][idx[0] * 2 + idx[1]]
    });
    let a = EinsumOperand::from_view(&a_arr.view());
    let b = EinsumOperand::from_view(&b_arr.view());
    let mut out = StridedArray::<f64>::row_major(&[2, 2]);
    einsum_into("ij,jk->ik", vec![a, b], out.view_mut(), 1.0, 0.0).unwrap();
    assert_abs_diff_eq!(out.get(&[0, 0]), 19.0, epsilon = 1e-10);
    assert_abs_diff_eq!(out.get(&[1, 1]), 50.0, epsilon = 1e-10);
}
