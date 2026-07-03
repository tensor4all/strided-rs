use approx::assert_relative_eq;
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

#[test]
fn fused_interprets_reused_dag_value() {
    let a = input(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let b = input(&[10.0, 20.0, 30.0, 40.0], &[4]);
    let mut out = StridedArray::<f64>::col_major(&[4]);
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![3],
        ops: vec![
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![0, 1],
            },
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![2, 0],
            },
        ],
    };

    fused_elementwise_into(&mut [out.view_mut()], &[a.view(), b.view()], &plan).unwrap();

    for i in 0..4 {
        assert_relative_eq!(
            out.get(&[i]),
            (a.get(&[i]) + b.get(&[i])) * a.get(&[i]),
            epsilon = 1e-12
        );
    }
}

#[test]
fn fused_interprets_multiple_outputs() {
    let a = input(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let b = input(&[10.0, 20.0, 30.0, 40.0], &[4]);
    let mut sum = StridedArray::<f64>::col_major(&[4]);
    let mut product = StridedArray::<f64>::col_major(&[4]);
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![2, 3],
        ops: vec![
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![0, 1],
            },
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![0, 1],
            },
        ],
    };

    fused_elementwise_into(
        &mut [sum.view_mut(), product.view_mut()],
        &[a.view(), b.view()],
        &plan,
    )
    .unwrap();

    for i in 0..4 {
        assert_relative_eq!(sum.get(&[i]), a.get(&[i]) + b.get(&[i]), epsilon = 1e-12);
        assert_relative_eq!(
            product.get(&[i]),
            a.get(&[i]) * b.get(&[i]),
            epsilon = 1e-12
        );
    }
}

#[test]
fn fused_interprets_broadcast_stride_zero_inputs() {
    let a = input(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let b = input(&[0.5, 1.5, 2.5, 3.5], &[4]);
    let c = input(&[2.0], &[1]);
    let c_view = c.view();
    let c_broadcast = c_view.broadcast(&[4]).unwrap();
    let mut out = StridedArray::<f64>::col_major(&[4]);
    let plan = FusedPlan {
        input_count: 3,
        outputs: vec![5],
        ops: vec![
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![0, 1],
            },
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![3, 2],
            },
            FusedInst {
                op: FusedOp::Exp,
                inputs: vec![4],
            },
        ],
    };

    fused_elementwise_into(
        &mut [out.view_mut()],
        &[a.view(), b.view(), c_broadcast],
        &plan,
    )
    .unwrap();

    for i in 0..4 {
        assert_relative_eq!(
            out.get(&[i]),
            (a.get(&[i]) * b.get(&[i]) + 2.0).exp(),
            epsilon = 1e-12
        );
    }
}

#[test]
fn fused_interprets_noncontiguous_inputs_and_outputs() {
    let a_base =
        StridedArray::<f64>::from_fn_row_major(&[3, 2], |idx| (1 + idx[0] * 10 + idx[1]) as f64);
    let b_base =
        StridedArray::<f64>::from_fn_row_major(&[3, 2], |idx| (2 + idx[0] * 7 + idx[1]) as f64);
    let a = a_base.view().permute(&[1, 0]).unwrap();
    let b = b_base.view().permute(&[1, 0]).unwrap();
    let mut out_base = StridedArray::<f64>::row_major(&[3, 2]);
    let out = out_base.view_mut().permute(&[1, 0]).unwrap();
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![2],
        ops: vec![FusedInst {
            op: FusedOp::Add,
            inputs: vec![0, 1],
        }],
    };

    fused_elementwise_into(&mut [out], &[a.clone(), b.clone()], &plan).unwrap();

    let out = out_base.view().permute(&[1, 0]).unwrap();
    for i in 0..2 {
        for j in 0..3 {
            assert_relative_eq!(
                out.get(&[i, j]),
                a.get(&[i, j]) + b.get(&[i, j]),
                epsilon = 1e-12
            );
        }
    }
}
