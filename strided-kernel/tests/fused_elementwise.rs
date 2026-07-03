use approx::assert_relative_eq;
use num_complex::{Complex32, Complex64};
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

#[test]
fn fused_real_maximum_minimum_match_nan_contract() {
    let a = input(&[f64::NAN, 3.0, f64::NAN], &[3]);
    let b = input(&[3.0, f64::NAN, f64::NAN], &[3]);
    let mut max_out = StridedArray::<f64>::col_major(&[3]);
    let mut min_out = StridedArray::<f64>::col_major(&[3]);
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![2, 3],
        ops: vec![
            FusedInst {
                op: FusedOp::Maximum,
                inputs: vec![0, 1],
            },
            FusedInst {
                op: FusedOp::Minimum,
                inputs: vec![0, 1],
            },
        ],
    };

    fused_elementwise_into(
        &mut [max_out.view_mut(), min_out.view_mut()],
        &[a.view(), b.view()],
        &plan,
    )
    .unwrap();

    assert_eq!(max_out.get(&[0]), 3.0);
    assert_eq!(min_out.get(&[0]), 3.0);
    assert_eq!(max_out.get(&[1]), 3.0);
    assert_eq!(min_out.get(&[1]), 3.0);
    assert!(max_out.get(&[2]).is_nan());
    assert!(min_out.get(&[2]).is_nan());
}

#[test]
fn fused_complex_divide_matches_native_complex_division() {
    let a = StridedArray::<Complex64>::from_parts(
        vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
        &[2],
        &[1],
        0,
    )
    .unwrap();
    let b = StridedArray::<Complex64>::from_parts(
        vec![Complex64::new(0.25, -1.0), Complex64::new(2.0, 3.0)],
        &[2],
        &[1],
        0,
    )
    .unwrap();
    let mut out = StridedArray::<Complex64>::col_major(&[2]);
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![2],
        ops: vec![FusedInst {
            op: FusedOp::Divide,
            inputs: vec![0, 1],
        }],
    };

    fused_elementwise_into(&mut [out.view_mut()], &[a.view(), b.view()], &plan).unwrap();

    for i in 0..2 {
        let expected = a.get(&[i]) / b.get(&[i]);
        assert_relative_eq!(out.get(&[i]).re, expected.re, epsilon = 1e-12);
        assert_relative_eq!(out.get(&[i]).im, expected.im, epsilon = 1e-12);
    }
}

#[test]
fn fused_complex_abs_returns_norm_in_real_component() {
    let a = StridedArray::<Complex64>::from_parts(
        vec![Complex64::new(3.0, 4.0), Complex64::new(5.0, 12.0)],
        &[2],
        &[1],
        0,
    )
    .unwrap();
    let mut out = StridedArray::<Complex64>::col_major(&[2]);
    let plan = FusedPlan {
        input_count: 1,
        outputs: vec![1],
        ops: vec![FusedInst {
            op: FusedOp::Abs,
            inputs: vec![0],
        }],
    };

    fused_elementwise_into(&mut [out.view_mut()], &[a.view()], &plan).unwrap();

    assert_relative_eq!(out.get(&[0]).re, 5.0, epsilon = 1e-12);
    assert_relative_eq!(out.get(&[0]).im, 0.0, epsilon = 1e-12);
    assert_relative_eq!(out.get(&[1]).re, 13.0, epsilon = 1e-12);
    assert_relative_eq!(out.get(&[1]).im, 0.0, epsilon = 1e-12);
}

#[test]
fn fused_complex32_add_multiply_matches_native_complex_arithmetic() {
    let a = StridedArray::<Complex32>::from_parts(
        vec![Complex32::new(1.0, 2.0), Complex32::new(-3.0, 0.5)],
        &[2],
        &[1],
        0,
    )
    .unwrap();
    let b = StridedArray::<Complex32>::from_parts(
        vec![Complex32::new(0.25, -1.0), Complex32::new(2.0, 3.0)],
        &[2],
        &[1],
        0,
    )
    .unwrap();
    let mut out = StridedArray::<Complex32>::col_major(&[2]);
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

    for i in 0..2 {
        let expected = (a.get(&[i]) + b.get(&[i])) * a.get(&[i]);
        assert_relative_eq!(out.get(&[i]).re, expected.re, epsilon = 1e-5);
        assert_relative_eq!(out.get(&[i]).im, expected.im, epsilon = 1e-5);
    }
}

#[test]
fn fused_all_real_ops_have_basic_parity() {
    let x = input(&[1.25, 2.5], &[2]);
    let y = input(&[0.5, 1.5], &[2]);
    let lo = input(&[1.5, 1.5], &[2]);
    let hi = input(&[2.0, 2.0], &[2]);
    let mut outputs: Vec<StridedArray<f64>> = (0..11)
        .map(|_| StridedArray::<f64>::col_major(&[2]))
        .collect();
    let plan = FusedPlan {
        input_count: 4,
        outputs: (4..15).collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Divide,
                inputs: vec![0, 1],
            },
            FusedInst {
                op: FusedOp::Clamp,
                inputs: vec![0, 2, 3],
            },
            FusedInst {
                op: FusedOp::Log,
                inputs: vec![0],
            },
            FusedInst {
                op: FusedOp::Sin,
                inputs: vec![0],
            },
            FusedInst {
                op: FusedOp::Cos,
                inputs: vec![0],
            },
            FusedInst {
                op: FusedOp::Tanh,
                inputs: vec![0],
            },
            FusedInst {
                op: FusedOp::Sqrt,
                inputs: vec![0],
            },
            FusedInst {
                op: FusedOp::Rsqrt,
                inputs: vec![0],
            },
            FusedInst {
                op: FusedOp::Pow,
                inputs: vec![0, 1],
            },
            FusedInst {
                op: FusedOp::Expm1,
                inputs: vec![0],
            },
            FusedInst {
                op: FusedOp::Log1p,
                inputs: vec![0],
            },
        ],
    };

    {
        let mut views: Vec<_> = outputs.iter_mut().map(|out| out.view_mut()).collect();
        fused_elementwise_into(
            &mut views,
            &[x.view(), y.view(), lo.view(), hi.view()],
            &plan,
        )
        .unwrap();
    }

    for i in 0..2 {
        let x = x.get(&[i]);
        let y = y.get(&[i]);
        let expected = [
            x / y,
            x.clamp(1.5, 2.0),
            x.ln(),
            x.sin(),
            x.cos(),
            x.tanh(),
            x.sqrt(),
            1.0 / x.sqrt(),
            x.powf(y),
            x.exp_m1(),
            x.ln_1p(),
        ];
        for (output, &expected) in outputs.iter().zip(expected.iter()) {
            assert_relative_eq!(output.get(&[i]), expected, epsilon = 1e-12);
        }
    }
}

#[test]
fn fused_specialized_binary_plan_handles_broadcast_inputs() {
    let a = input(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let b = input(&[10.0], &[1]);
    let b_view = b.view();
    let b_broadcast = b_view.broadcast(&[4]).unwrap();
    let mut out = StridedArray::<f64>::col_major(&[4]);
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![2],
        ops: vec![FusedInst {
            op: FusedOp::Add,
            inputs: vec![0, 1],
        }],
    };

    fused_elementwise_into(&mut [out.view_mut()], &[a.view(), b_broadcast], &plan).unwrap();

    for i in 0..4 {
        assert_relative_eq!(out.get(&[i]), a.get(&[i]) + 10.0, epsilon = 1e-12);
    }
}
