use strided_kernel::{
    fused_elementwise_into, FusedInst, FusedOp, FusedPlan, StridedArray, StridedError,
};

fn input(values: &[f64], dims: &[usize]) -> StridedArray<f64> {
    StridedArray::from_parts(values.to_vec(), dims, &[1], 0).unwrap()
}

#[test]
fn fused_rejects_input_count_mismatch() {
    let a = input(&[1.0, 2.0], &[2]);
    let mut out = StridedArray::<f64>::col_major(&[2]);
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![0],
        ops: vec![],
    };

    let err =
        fused_elementwise_into(&mut [out.view_mut()], &[a.view()], &plan).expect_err("must fail");

    assert!(matches!(err, StridedError::RankMismatch(1, 2)));
    assert_eq!(out.get(&[0]), 0.0);
    assert_eq!(out.get(&[1]), 0.0);
}

#[test]
fn fused_rejects_destination_count_mismatch() {
    let a = input(&[1.0, 2.0], &[2]);
    let mut out = StridedArray::<f64>::col_major(&[2]);
    let plan = FusedPlan {
        input_count: 1,
        outputs: vec![0, 0],
        ops: vec![],
    };

    let err =
        fused_elementwise_into(&mut [out.view_mut()], &[a.view()], &plan).expect_err("must fail");

    assert!(matches!(err, StridedError::RankMismatch(1, 2)));
    assert_eq!(out.get(&[0]), 0.0);
    assert_eq!(out.get(&[1]), 0.0);
}

#[test]
fn fused_rejects_bad_value_id_before_writing() {
    let a = input(&[1.0, 2.0], &[2]);
    let mut out = StridedArray::<f64>::col_major(&[2]);
    let plan = FusedPlan {
        input_count: 1,
        outputs: vec![1],
        ops: vec![FusedInst {
            op: FusedOp::Negate,
            inputs: vec![2],
        }],
    };

    let err =
        fused_elementwise_into(&mut [out.view_mut()], &[a.view()], &plan).expect_err("must fail");

    assert!(matches!(
        err,
        StridedError::InvalidAxis { axis: 2, rank: 1 }
    ));
    assert_eq!(out.get(&[0]), 0.0);
    assert_eq!(out.get(&[1]), 0.0);
}

#[test]
fn fused_rejects_wrong_op_arity_before_writing() {
    let a = input(&[1.0, 2.0], &[2]);
    let mut out = StridedArray::<f64>::col_major(&[2]);
    let plan = FusedPlan {
        input_count: 1,
        outputs: vec![1],
        ops: vec![FusedInst {
            op: FusedOp::Add,
            inputs: vec![0],
        }],
    };

    let err =
        fused_elementwise_into(&mut [out.view_mut()], &[a.view()], &plan).expect_err("must fail");

    assert!(matches!(err, StridedError::RankMismatch(1, 2)));
    assert_eq!(out.get(&[0]), 0.0);
    assert_eq!(out.get(&[1]), 0.0);
}

#[test]
fn fused_rejects_shape_mismatch_before_writing() {
    let a = input(&[1.0, 2.0], &[2]);
    let b = input(&[1.0, 2.0, 3.0], &[3]);
    let mut out = StridedArray::<f64>::col_major(&[2]);
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![2],
        ops: vec![FusedInst {
            op: FusedOp::Add,
            inputs: vec![0, 1],
        }],
    };

    let err = fused_elementwise_into(&mut [out.view_mut()], &[a.view(), b.view()], &plan)
        .expect_err("must fail");

    assert!(matches!(err, StridedError::ShapeMismatch(_, _)));
    assert_eq!(out.get(&[0]), 0.0);
    assert_eq!(out.get(&[1]), 0.0);
}
