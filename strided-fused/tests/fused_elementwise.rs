use approx::assert_relative_eq;
use num_complex::{Complex32, Complex64};
use strided_basic::{StridedArray, StridedError};
use strided_fused::{fused_elementwise_into, FusedInst, FusedOp, FusedPlan};

fn input(values: &[f64], dims: &[usize]) -> StridedArray<f64> {
    StridedArray::from_parts(values.to_vec(), dims, &[1], 0).unwrap()
}

#[test]
fn fused_rejects_input_count_mismatch() {
    let a = input(&[1.0, 2.0], &[2]);
    let mut out = StridedArray::<f64>::col_major(&[2]);
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![0].into_iter().collect(),
        ops: vec![].into_iter().collect(),
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
        outputs: vec![0, 0].into_iter().collect(),
        ops: vec![].into_iter().collect(),
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
        outputs: vec![1].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Negate,
            inputs: vec![2].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
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
        outputs: vec![1].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Add,
            inputs: vec![0].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
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
        outputs: vec![2].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Add,
            inputs: vec![0, 1].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    };

    let err = fused_elementwise_into(&mut [out.view_mut()], &[a.view(), b.view()], &plan)
        .expect_err("must fail");

    assert!(matches!(err, StridedError::ShapeMismatch(_, _)));
    assert_eq!(out.get(&[0]), 0.0);
    assert_eq!(out.get(&[1]), 0.0);
}

#[test]
fn fused_rejects_stride_zero_destination_before_writing() {
    let a = input(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let mut out = StridedArray::<f64>::from_parts(vec![0.0], &[4], &[0], 0).unwrap();
    let plan = FusedPlan {
        input_count: 1,
        outputs: vec![1].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Negate,
            inputs: vec![0].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    };

    let err =
        fused_elementwise_into(&mut [out.view_mut()], &[a.view()], &plan).expect_err("must fail");

    assert!(err.to_string().contains("output layout"));
    assert_eq!(out.data(), &[0.0]);
}

#[test]
fn fused_rejects_overlapping_destination_offsets_before_writing() {
    let a = StridedArray::<f64>::from_parts(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &[1, 2], 0).unwrap();
    let mut out =
        StridedArray::<f64>::from_parts(vec![0.0, 0.0, 0.0], &[2, 2], &[1, 1], 0).unwrap();
    let plan = FusedPlan {
        input_count: 1,
        outputs: vec![1].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Negate,
            inputs: vec![0].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    };

    let err =
        fused_elementwise_into(&mut [out.view_mut()], &[a.view()], &plan).expect_err("must fail");

    assert!(err.to_string().contains("output layout"));
    assert_eq!(out.data(), &[0.0, 0.0, 0.0]);
}

#[test]
fn fused_interprets_reused_dag_value() {
    let a = input(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let b = input(&[10.0, 20.0, 30.0, 40.0], &[4]);
    let mut out = StridedArray::<f64>::col_major(&[4]);
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![3].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![0, 1].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![2, 0].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
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
        outputs: vec![2, 3].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![0, 1].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![0, 1].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
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
        outputs: vec![5].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![0, 1].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![3, 2].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Exp,
                inputs: vec![4].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
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
        outputs: vec![2].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Add,
            inputs: vec![0, 1].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
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
    // Two outputs skip the static specialization, so this goes through the
    // chunked replay. Any NaN operand must stay NaN, and signed-zero ties
    // return the left operand, exactly like the ordinary binary kernel.
    let a = input(&[f64::NAN, 3.0, f64::NAN, 0.0, -0.0], &[5]);
    let b = input(&[3.0, f64::NAN, f64::NAN, -0.0, 0.0], &[5]);
    let mut max_out = StridedArray::<f64>::col_major(&[5]);
    let mut min_out = StridedArray::<f64>::col_major(&[5]);
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![2, 3].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Maximum,
                inputs: vec![0, 1].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Minimum,
                inputs: vec![0, 1].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
    };

    fused_elementwise_into(
        &mut [max_out.view_mut(), min_out.view_mut()],
        &[a.view(), b.view()],
        &plan,
    )
    .unwrap();

    for index in 0..3 {
        assert!(max_out.get(&[index]).is_nan(), "maximum {index}");
        assert!(min_out.get(&[index]).is_nan(), "minimum {index}");
    }
    assert_eq!(max_out.get(&[3]).to_bits(), 0.0f64.to_bits());
    assert_eq!(min_out.get(&[3]).to_bits(), 0.0f64.to_bits());
    assert_eq!(max_out.get(&[4]).to_bits(), (-0.0f64).to_bits());
    assert_eq!(min_out.get(&[4]).to_bits(), (-0.0f64).to_bits());
}

#[test]
fn fused_real_clamp_propagates_nan_bounds() {
    let x = input(&[1.0, 2.0, 3.0], &[3]);
    let lo = input(&[0.0, 4.0, f64::NAN], &[3]);
    let hi = input(&[3.0, 1.0, 2.0], &[3]);
    let mut out = StridedArray::<f64>::col_major(&[3]);
    let plan = FusedPlan {
        input_count: 3,
        outputs: vec![3].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Clamp,
            inputs: vec![0, 1, 2].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    };

    // A single-output plan takes the static ternary specialization.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fused_elementwise_into(
            &mut [out.view_mut()],
            &[x.view(), lo.view(), hi.view()],
            &plan,
        )
    }));

    assert!(result.is_ok(), "clamp should not panic for data values");
    result.unwrap().unwrap();
    assert_eq!(out.get(&[0]), 1.0);
    assert_eq!(out.get(&[1]), 1.0);
    assert!(
        out.get(&[2]).is_nan(),
        "a NaN bound must propagate through clamp"
    );
}

#[test]
fn fused_f32_static_maximum_minimum_and_clamp_propagate_nan() {
    let a = StridedArray::<f32>::from_parts(vec![f32::NAN, 3.0, -0.0], &[3], &[1], 0).unwrap();
    let b = StridedArray::<f32>::from_parts(vec![3.0, f32::NAN, 0.0], &[3], &[1], 0).unwrap();

    for op in [FusedOp::Maximum, FusedOp::Minimum] {
        let mut out = StridedArray::<f32>::col_major(&[3]);
        let plan = FusedPlan {
            input_count: 2,
            outputs: vec![2].into_iter().collect(),
            ops: vec![FusedInst {
                op,
                inputs: vec![0, 1].into_iter().collect(),
            }]
            .into_iter()
            .collect(),
        };
        fused_elementwise_into(&mut [out.view_mut()], &[a.view(), b.view()], &plan).unwrap();
        assert!(out.get(&[0]).is_nan(), "{op:?} lhs NaN");
        assert!(out.get(&[1]).is_nan(), "{op:?} rhs NaN");
        assert_eq!(out.get(&[2]).to_bits(), (-0.0f32).to_bits(), "{op:?}");
    }

    let x = StridedArray::<f32>::from_parts(vec![1.0, 2.0, 3.0], &[3], &[1], 0).unwrap();
    let lo = StridedArray::<f32>::from_parts(vec![0.0, 4.0, f32::NAN], &[3], &[1], 0).unwrap();
    let hi = StridedArray::<f32>::from_parts(vec![3.0, 1.0, 2.0], &[3], &[1], 0).unwrap();
    let mut out = StridedArray::<f32>::col_major(&[3]);
    let plan = FusedPlan {
        input_count: 3,
        outputs: vec![3].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Clamp,
            inputs: vec![0, 1, 2].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    };
    fused_elementwise_into(
        &mut [out.view_mut()],
        &[x.view(), lo.view(), hi.view()],
        &plan,
    )
    .unwrap();
    assert_eq!(out.get(&[0]), 1.0);
    assert_eq!(out.get(&[1]), 1.0);
    assert!(out.get(&[2]).is_nan(), "f32 clamp NaN bound");
}

#[test]
fn fused_clamp_signed_zero_ties_match_the_ordinary_kernel() {
    // `maximum(lo, x)` and `minimum(hi, ..)` keep the bound on a signed-zero
    // tie, so `clamp(-0.0, 0.0, 1.0)` is `+0.0` (`strided-kernel` test
    // `maximum_minimum_signed_zero_ties_return_lhs`).
    let x = input(&[-0.0], &[1]);
    let lo = input(&[0.0], &[1]);
    let hi = input(&[1.0], &[1]);
    let mut out = StridedArray::<f64>::col_major(&[1]);
    let plan = FusedPlan {
        input_count: 3,
        outputs: vec![3].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Clamp,
            inputs: vec![0, 1, 2].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    };
    fused_elementwise_into(
        &mut [out.view_mut()],
        &[x.view(), lo.view(), hi.view()],
        &plan,
    )
    .unwrap();
    assert_eq!(out.get(&[0]).to_bits(), 0.0f64.to_bits());
}

#[test]
fn fused_maximum_minimum_interpreter_propagates_nan() {
    // A stride-2 input is not chunk-replay eligible, so this forces the
    // per-element interpreter. Two outputs skip the static specialization.
    let a_data = [f64::NAN, 0.0, 3.0, 0.0, 5.0];
    let b_data = [3.0, 0.0, f64::NAN, 0.0, 5.0];
    let a = StridedArray::<f64>::from_parts(a_data.to_vec(), &[3], &[2], 0).unwrap();
    let b = StridedArray::<f64>::from_parts(b_data.to_vec(), &[3], &[2], 0).unwrap();
    let mut max_out = StridedArray::<f64>::col_major(&[3]);
    let mut min_out = StridedArray::<f64>::col_major(&[3]);
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![2, 3].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Maximum,
                inputs: vec![0, 1].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Minimum,
                inputs: vec![0, 1].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
    };
    fused_elementwise_into(
        &mut [max_out.view_mut(), min_out.view_mut()],
        &[a.view(), b.view()],
        &plan,
    )
    .unwrap();
    assert!(max_out.get(&[0]).is_nan());
    assert!(min_out.get(&[0]).is_nan());
    assert!(max_out.get(&[1]).is_nan());
    assert!(min_out.get(&[1]).is_nan());
    assert_eq!(max_out.get(&[2]), 5.0);
    assert_eq!(min_out.get(&[2]), 5.0);
}

#[test]
fn fused_maximum_minimum_chunked_replay_propagates_nan_across_chunks() {
    // Longer than `FUSED_CHUNK_ELEMS` (1024) so the replay runs several
    // chunks; two outputs keep the plan off the static specialization.
    const LEN: usize = 4096;
    let a: Vec<f64> = (0..LEN)
        .map(|index| {
            if index % 3 == 0 {
                f64::NAN
            } else {
                index as f64
            }
        })
        .collect();
    let b: Vec<f64> = (0..LEN)
        .map(|index| {
            if index % 3 == 1 {
                f64::NAN
            } else {
                index as f64
            }
        })
        .collect();
    let a = StridedArray::<f64>::from_parts(a, &[LEN], &[1], 0).unwrap();
    let b = StridedArray::<f64>::from_parts(b, &[LEN], &[1], 0).unwrap();
    let mut max_out = StridedArray::<f64>::col_major(&[LEN]);
    let mut min_out = StridedArray::<f64>::col_major(&[LEN]);
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![2, 3].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Maximum,
                inputs: vec![0, 1].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Minimum,
                inputs: vec![0, 1].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
    };
    fused_elementwise_into(
        &mut [max_out.view_mut(), min_out.view_mut()],
        &[a.view(), b.view()],
        &plan,
    )
    .unwrap();
    for index in 0..LEN {
        let expected_nan = index % 3 != 2;
        assert_eq!(max_out.get(&[index]).is_nan(), expected_nan, "max {index}");
        assert_eq!(min_out.get(&[index]).is_nan(), expected_nan, "min {index}");
    }
    // A finite element past the first chunk checks the second chunk ran.
    assert_eq!(max_out.get(&[LEN - 2]), (LEN - 2) as f64);
}

#[test]
fn fused_integer_add_uses_wrapping_arithmetic() {
    let a = StridedArray::<i32>::from_parts(vec![i32::MAX, i32::MIN, -3], &[3], &[1], 0).unwrap();
    let b = StridedArray::<i32>::from_parts(vec![1, -1, 2], &[3], &[1], 0).unwrap();
    let mut out = StridedArray::<i32>::col_major(&[3]);
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![2].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Add,
            inputs: vec![0, 1].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    };

    fused_elementwise_into(&mut [out.view_mut()], &[a.view(), b.view()], &plan).unwrap();

    assert_eq!(out.data(), &[i32::MIN, i32::MAX, -1]);
}

#[test]
fn fused_bool_arithmetic_is_rejected_before_writing() {
    let a = StridedArray::<bool>::from_parts(vec![true, false], &[2], &[1], 0).unwrap();
    let b = StridedArray::<bool>::from_parts(vec![false, true], &[2], &[1], 0).unwrap();
    let mut out = StridedArray::<bool>::from_parts(vec![true, true], &[2], &[1], 0).unwrap();
    let plan = FusedPlan {
        input_count: 2,
        outputs: vec![2].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Add,
            inputs: vec![0, 1].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    };

    let err = fused_elementwise_into(&mut [out.view_mut()], &[a.view(), b.view()], &plan)
        .expect_err("bool add must fail");

    assert!(matches!(
        err,
        StridedError::UnsupportedOp {
            op: "add",
            dtype: "bool"
        }
    ));
    assert_eq!(out.data(), &[true, true]);
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
        outputs: vec![2].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Divide,
            inputs: vec![0, 1].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
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
        outputs: vec![1].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Abs,
            inputs: vec![0].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
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
        outputs: vec![3].into_iter().collect(),
        ops: vec![
            FusedInst {
                op: FusedOp::Add,
                inputs: vec![0, 1].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![2, 0].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
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
                inputs: vec![0, 1].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Clamp,
                inputs: vec![0, 2, 3].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Log,
                inputs: vec![0].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Sin,
                inputs: vec![0].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Cos,
                inputs: vec![0].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Tanh,
                inputs: vec![0].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Sqrt,
                inputs: vec![0].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Rsqrt,
                inputs: vec![0].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Pow,
                inputs: vec![0, 1].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Expm1,
                inputs: vec![0].into_iter().collect(),
            },
            FusedInst {
                op: FusedOp::Log1p,
                inputs: vec![0].into_iter().collect(),
            },
        ]
        .into_iter()
        .collect(),
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
            x.max(1.5).min(2.0),
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
        outputs: vec![2].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Add,
            inputs: vec![0, 1].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    };

    fused_elementwise_into(&mut [out.view_mut()], &[a.view(), b_broadcast], &plan).unwrap();

    for i in 0..4 {
        assert_relative_eq!(out.get(&[i]), a.get(&[i]) + 10.0, epsilon = 1e-12);
    }
}
