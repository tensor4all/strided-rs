use core::mem::MaybeUninit;
use num_complex::{Complex32, Complex64};
use strided_basic::{
    ErasedRawStridedMut, ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut, Exec,
    KernelDType, StridedError,
};
use strided_fused::{ErasedFusedPlan, FusedInst, FusedOp, FusedPlan};

fn binary_plan(op: FusedOp) -> FusedPlan {
    FusedPlan {
        input_count: 2,
        outputs: vec![2].into_iter().collect(),
        ops: vec![FusedInst {
            op,
            inputs: vec![0, 1].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    }
}

fn unary_plan(op: FusedOp) -> FusedPlan {
    FusedPlan {
        input_count: 1,
        outputs: vec![1].into_iter().collect(),
        ops: vec![FusedInst {
            op,
            inputs: vec![0].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    }
}

#[test]
fn erased_fused_plan_executes_f64_binary_add_transposed_layout() {
    let dims = [2usize, 3];
    let src_strides = [3isize, 1];
    let dst_strides = [1isize, 2];
    let a = [0.0f64, 1.0, 2.0, 10.0, 11.0, 12.0];
    let b = [100.0f64, 101.0, 102.0, 110.0, 111.0, 112.0];
    let mut dst = [0.0f64; 6];

    let plan = ErasedFusedPlan::compile(KernelDType::F64, binary_plan(FusedOp::Add)).unwrap();
    let inputs = [
        ErasedRawStridedRef::from_slice(&a, &dims, &src_strides, 0).unwrap(),
        ErasedRawStridedRef::from_slice(&b, &dims, &src_strides, 0).unwrap(),
    ];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &dst_strides, 0).unwrap();

    plan.execute(&mut dest, &inputs, Exec::serial()).unwrap();

    assert_eq!(dst, [100.0, 120.0, 102.0, 122.0, 104.0, 124.0]);
}

#[test]
fn erased_fused_plan_executes_f32_ternary_clamp() {
    let dims = [3usize];
    let strides = [1isize];
    let x = [-2.0f32, 0.5, 4.0];
    let min = [-1.0f32, 0.0, 1.0];
    let max = [1.0f32, 1.0, 3.0];
    let mut dst = [0.0f32; 3];

    let plan = ErasedFusedPlan::compile(
        KernelDType::F32,
        FusedPlan {
            input_count: 3,
            outputs: vec![3].into_iter().collect(),
            ops: vec![FusedInst {
                op: FusedOp::Clamp,
                inputs: vec![0, 1, 2].into_iter().collect(),
            }]
            .into_iter()
            .collect(),
        },
    )
    .unwrap();
    let inputs = [
        ErasedRawStridedRef::from_slice(&x, &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::from_slice(&min, &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::from_slice(&max, &dims, &strides, 0).unwrap(),
    ];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &strides, 0).unwrap();

    plan.execute(&mut dest, &inputs, Exec::serial()).unwrap();

    assert_eq!(dst, [-1.0, 0.5, 3.0]);
}

#[test]
fn erased_fused_plan_executes_c32_unary_conjugate() {
    let dims = [2usize];
    let strides = [1isize];
    let input = [Complex32::new(1.0, -2.0), Complex32::new(-3.0, 4.0)];
    let mut dst = [Complex32::new(0.0, 0.0); 2];

    let plan = ErasedFusedPlan::compile(KernelDType::C32, unary_plan(FusedOp::Conj)).unwrap();
    let inputs = [ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap()];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &strides, 0).unwrap();

    plan.execute(&mut dest, &inputs, Exec::serial()).unwrap();

    assert_eq!(dst, [input[0].conj(), input[1].conj()]);
}

#[test]
fn erased_fused_plan_executes_c64_binary_multiply() {
    let dims = [2usize];
    let strides = [1isize];
    let a = [Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)];
    let b = [Complex64::new(4.0, -1.0), Complex64::new(0.5, 2.0)];
    let mut dst = [Complex64::new(0.0, 0.0); 2];

    let plan = ErasedFusedPlan::compile(KernelDType::C64, binary_plan(FusedOp::Multiply)).unwrap();
    let inputs = [
        ErasedRawStridedRef::from_slice(&a, &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::from_slice(&b, &dims, &strides, 0).unwrap(),
    ];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &strides, 0).unwrap();

    plan.execute(&mut dest, &inputs, Exec::serial()).unwrap();

    assert_eq!(dst, [a[0] * b[0], a[1] * b[1]]);
}

#[test]
fn erased_fused_plan_executes_f64_four_input_dag() {
    let dims = [2usize];
    let strides = [1isize];
    let a = [1.0f64, 2.0];
    let b = [10.0f64, 20.0];
    let c = [3.0f64, 4.0];
    let d = [5.0f64, 6.0];
    let mut dst = [0.0f64; 2];

    let plan = ErasedFusedPlan::compile(
        KernelDType::F64,
        FusedPlan {
            input_count: 4,
            outputs: vec![6].into_iter().collect(),
            ops: vec![
                FusedInst {
                    op: FusedOp::Add,
                    inputs: vec![0, 1].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Multiply,
                    inputs: vec![2, 3].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Add,
                    inputs: vec![4, 5].into_iter().collect(),
                },
            ]
            .into_iter()
            .collect(),
        },
    )
    .unwrap();
    let inputs = [
        ErasedRawStridedRef::from_slice(&a, &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::from_slice(&b, &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::from_slice(&c, &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::from_slice(&d, &dims, &strides, 0).unwrap(),
    ];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &strides, 0).unwrap();

    plan.execute(&mut dest, &inputs, Exec::serial()).unwrap();

    assert_eq!(dst, [26.0, 46.0]);
}

#[test]
fn erased_fused_plan_executes_i32_safe_integer_dag() {
    let dims = [3usize];
    let strides = [1isize];
    let x = [-3i32, 2, 5];
    let y = [4i32, -7, 1];
    let lo = [0i32, 0, 0];
    let hi = [10i32, 10, 10];
    let mut dst = [0i32; 3];

    let plan = ErasedFusedPlan::compile(
        KernelDType::I32,
        FusedPlan {
            input_count: 4,
            outputs: vec![10].into_iter().collect(),
            ops: vec![
                FusedInst {
                    op: FusedOp::Add,
                    inputs: vec![0, 1].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Multiply,
                    inputs: vec![4, 0].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Negate,
                    inputs: vec![5].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Abs,
                    inputs: vec![6].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Maximum,
                    inputs: vec![7, 2].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Minimum,
                    inputs: vec![8, 3].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Clamp,
                    inputs: vec![9, 2, 3].into_iter().collect(),
                },
            ]
            .into_iter()
            .collect(),
        },
    )
    .unwrap();
    let inputs = [
        ErasedRawStridedRef::from_slice(&x, &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::from_slice(&y, &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::from_slice(&lo, &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::from_slice(&hi, &dims, &strides, 0).unwrap(),
    ];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &strides, 0).unwrap();

    plan.execute(&mut dest, &inputs, Exec::serial()).unwrap();

    assert_eq!(dst, [3, 10, 10]);
}

#[test]
fn erased_fused_plan_executes_i64_binary_add_transposed_layout() {
    let dims = [2usize, 3];
    let src_strides = [3isize, 1];
    let dst_strides = [1isize, 2];
    let a = [0i64, 1, 2, 10, 11, 12];
    let b = [100i64, 101, 102, 110, 111, 112];
    let mut dst = [0i64; 6];

    let plan = ErasedFusedPlan::compile(KernelDType::I64, binary_plan(FusedOp::Add)).unwrap();
    let inputs = [
        ErasedRawStridedRef::from_slice(&a, &dims, &src_strides, 0).unwrap(),
        ErasedRawStridedRef::from_slice(&b, &dims, &src_strides, 0).unwrap(),
    ];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &dst_strides, 0).unwrap();

    plan.execute(&mut dest, &inputs, Exec::serial()).unwrap();

    assert_eq!(dst, [100, 120, 102, 122, 104, 124]);
}

#[test]
fn erased_fused_plan_executes_bool_identity_conj() {
    let dims = [3usize];
    let strides = [1isize];
    let input = [true, false, true];
    let mut dst = [false; 3];

    let plan = ErasedFusedPlan::compile(KernelDType::Bool, unary_plan(FusedOp::Conj)).unwrap();
    let inputs = [ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap()];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &strides, 0).unwrap();

    plan.execute(&mut dest, &inputs, Exec::serial()).unwrap();

    assert_eq!(dst, input);
}

#[test]
fn erased_fused_plan_rejects_dtype_mismatch_before_writing() {
    let dims = [2usize];
    let strides = [1isize];
    let a = [1.0f64, 2.0];
    let b = [3.0f64, 4.0];
    let mut dst = [0.0f32; 2];

    let plan = ErasedFusedPlan::compile(KernelDType::F64, binary_plan(FusedOp::Add)).unwrap();
    let inputs = [
        ErasedRawStridedRef::from_slice(&a, &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::from_slice(&b, &dims, &strides, 0).unwrap(),
    ];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &strides, 0).unwrap();

    let err = plan
        .execute(&mut dest, &inputs, Exec::serial())
        .unwrap_err();

    assert!(matches!(err, StridedError::DTypeMismatch { .. }));
    assert_eq!(dst, [0.0, 0.0]);
}

#[test]
fn erased_fused_plan_rejects_input_count_mismatch_before_writing() {
    let dims = [2usize];
    let strides = [1isize];
    let a = [1.0f64, 2.0];
    let mut dst = [9.0f64, 10.0];

    let plan = ErasedFusedPlan::compile(KernelDType::F64, binary_plan(FusedOp::Add)).unwrap();
    let inputs = [ErasedRawStridedRef::from_slice(&a, &dims, &strides, 0).unwrap()];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &strides, 0).unwrap();

    let err = plan
        .execute(&mut dest, &inputs, Exec::serial())
        .unwrap_err();

    assert!(matches!(err, StridedError::RankMismatch(1, 2)));
    assert_eq!(dst, [9.0, 10.0]);
}

#[test]
fn erased_fused_plan_rejects_input_dtype_mismatch_before_writing() {
    let dims = [2usize];
    let strides = [1isize];
    let a = [1.0f64, 2.0];
    let b = [3.0f32, 4.0];
    let mut dst = [9.0f64, 10.0];

    let plan = ErasedFusedPlan::compile(KernelDType::F64, binary_plan(FusedOp::Add)).unwrap();
    let inputs = [
        ErasedRawStridedRef::from_slice(&a, &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::from_slice(&b, &dims, &strides, 0).unwrap(),
    ];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &strides, 0).unwrap();

    let err = plan
        .execute(&mut dest, &inputs, Exec::serial())
        .unwrap_err();

    assert!(matches!(err, StridedError::DTypeMismatch { .. }));
    assert_eq!(dst, [9.0, 10.0]);
}

#[test]
fn erased_fused_plan_rejects_runtime_shape_mismatch_before_writing() {
    let dest_dims = [2usize];
    let input_dims = [3usize];
    let strides = [1isize];
    let a = [1.0f64, 2.0, 3.0];
    let b = [4.0f64, 5.0, 6.0];
    let mut dst = [9.0f64, 10.0];

    let plan = ErasedFusedPlan::compile(KernelDType::F64, binary_plan(FusedOp::Add)).unwrap();
    let inputs = [
        ErasedRawStridedRef::from_slice(&a, &input_dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::from_slice(&b, &input_dims, &strides, 0).unwrap(),
    ];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dest_dims, &strides, 0).unwrap();

    let err = plan
        .execute(&mut dest, &inputs, Exec::serial())
        .unwrap_err();

    assert!(matches!(err, StridedError::ShapeMismatch(_, _)));
    assert_eq!(dst, [9.0, 10.0]);
}

#[test]
fn erased_fused_plan_exposes_dtype_and_plan() {
    let plan = ErasedFusedPlan::compile(KernelDType::F64, binary_plan(FusedOp::Add)).unwrap();

    assert_eq!(plan.dtype(), KernelDType::F64);
    assert_eq!(plan.plan().input_count, 2);
    assert_eq!(plan.plan().outputs.as_slice(), &[2]);
}

#[test]
fn erased_fused_plan_rejects_unsupported_ops_by_dtype() {
    let unsupported_integer_op =
        ErasedFusedPlan::compile(KernelDType::I32, binary_plan(FusedOp::Divide)).unwrap_err();
    assert!(matches!(
        unsupported_integer_op,
        StridedError::UnsupportedOp {
            op: "divide",
            dtype: "i32"
        }
    ));

    let unsupported_bool_op =
        ErasedFusedPlan::compile(KernelDType::Bool, binary_plan(FusedOp::Add)).unwrap_err();
    assert!(matches!(
        unsupported_bool_op,
        StridedError::UnsupportedOp {
            op: "add",
            dtype: "bool"
        }
    ));
}

#[test]
fn erased_fused_plan_rejects_unsupported_arity() {
    let too_many_inputs = FusedPlan {
        input_count: 5,
        outputs: vec![0].into_iter().collect(),
        ops: vec![].into_iter().collect(),
    };
    let unsupported_arity =
        ErasedFusedPlan::compile(KernelDType::F64, too_many_inputs).unwrap_err();
    assert!(matches!(
        unsupported_arity,
        StridedError::UnsupportedArity { arity: 5, max: 4 }
    ));

    let zero_inputs = FusedPlan {
        input_count: 0,
        outputs: vec![0].into_iter().collect(),
        ops: vec![].into_iter().collect(),
    };
    let unsupported_arity = ErasedFusedPlan::compile(KernelDType::F64, zero_inputs).unwrap_err();
    assert!(matches!(
        unsupported_arity,
        StridedError::UnsupportedArity { arity: 0, max: 4 }
    ));
}

#[test]
fn erased_fused_plan_rejects_invalid_output_contracts() {
    // Repeated output ids are allowed: each name writes the same value.
    let repeated = FusedPlan {
        input_count: 2,
        outputs: vec![2, 2].into_iter().collect(),
        ops: vec![FusedInst {
            op: FusedOp::Add,
            inputs: vec![0, 1].into_iter().collect(),
        }]
        .into_iter()
        .collect(),
    };
    let compiled = ErasedFusedPlan::compile(KernelDType::F64, repeated).unwrap();
    assert_eq!(compiled.plan().outputs.as_slice(), &[2, 2]);

    let zero_outputs = FusedPlan {
        input_count: 2,
        outputs: vec![].into_iter().collect(),
        ops: vec![].into_iter().collect(),
    };
    let err = ErasedFusedPlan::compile(KernelDType::F64, zero_outputs).unwrap_err();
    assert!(matches!(err, StridedError::RankMismatch(0, 1)));

    let invalid_output = FusedPlan {
        input_count: 2,
        outputs: vec![3].into_iter().collect(),
        ops: vec![].into_iter().collect(),
    };
    let err = ErasedFusedPlan::compile(KernelDType::F64, invalid_output).unwrap_err();
    assert!(matches!(
        err,
        StridedError::InvalidAxis { axis: 3, rank: 2 }
    ));
}

#[test]
fn erased_fused_plan_executes_multiple_outputs_with_pass_through_and_intermediate() {
    let dims = [2usize];
    let unit = [1isize];
    let gapped = [2isize];
    let input = [2.0f64, -3.0];
    let plan = ErasedFusedPlan::compile(
        KernelDType::F64,
        FusedPlan {
            input_count: 1,
            // First output is the input pass-through, second is a value computed
            // from the intermediate, third is the intermediate itself.
            outputs: vec![0, 2, 1].into_iter().collect(),
            ops: vec![
                FusedInst {
                    op: FusedOp::Multiply,
                    inputs: vec![0, 0].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Negate,
                    inputs: vec![1].into_iter().collect(),
                },
            ]
            .into_iter()
            .collect(),
        },
    )
    .unwrap();
    let inputs = [ErasedRawStridedRef::from_slice(&input, &dims, &unit, 0).unwrap()];
    let mut pass_through = [0.0f64; 2];
    let mut squared_negated = [0.0f64; 3];
    let mut squared = [0.0f64; 2];
    {
        let mut dests = [
            ErasedRawStridedMut::from_slice_mut(&mut pass_through, &dims, &unit, 0).unwrap(),
            ErasedRawStridedMut::from_slice_mut(&mut squared_negated, &dims, &gapped, 0).unwrap(),
            ErasedRawStridedMut::from_slice_mut(&mut squared, &dims, &unit, 0).unwrap(),
        ];
        plan.execute_outputs(&mut dests, &inputs, Exec::serial())
            .unwrap();
    }
    assert_eq!(pass_through, [2.0, -3.0]);
    assert_eq!([squared_negated[0], squared_negated[2]], [-4.0, -9.0]);
    assert_eq!(squared, [4.0, 9.0]);
}

#[test]
fn erased_fused_plan_executes_multiple_uninit_outputs_for_f32() {
    let dims = [3usize];
    let unit = [1isize];
    let input = [1.0f32, -2.0, 3.0];
    let plan = ErasedFusedPlan::compile(
        KernelDType::F32,
        FusedPlan {
            input_count: 1,
            outputs: vec![1, 2, 0].into_iter().collect(),
            ops: vec![
                FusedInst {
                    op: FusedOp::Negate,
                    inputs: vec![0].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Abs,
                    inputs: vec![1].into_iter().collect(),
                },
            ]
            .into_iter()
            .collect(),
        },
    )
    .unwrap();
    let input_ref = ErasedRawStridedRef::from_slice(&input, &dims, &unit, 0).unwrap();
    let pointers = [ErasedRawStridedPtr::from_ref(&input_ref)];
    let mut first = [MaybeUninit::<f32>::uninit(); 3];
    let mut second = [MaybeUninit::<f32>::uninit(); 3];
    let mut third = [MaybeUninit::<f32>::uninit(); 3];
    {
        let mut dests = [
            ErasedRawStridedUninitMut::from_uninit_slice(&mut first, &dims, &unit, 0).unwrap(),
            ErasedRawStridedUninitMut::from_uninit_slice(&mut second, &dims, &unit, 0).unwrap(),
            ErasedRawStridedUninitMut::from_uninit_slice(&mut third, &dims, &unit, 0).unwrap(),
        ];
        plan.execute_outputs_uninit(&mut dests, &pointers, Exec::serial())
            .unwrap();
    }
    let init = |slots: &[MaybeUninit<f32>; 3]| -> Vec<f32> {
        // SAFETY: `execute_outputs_uninit` writes every reachable element.
        slots
            .iter()
            .map(|slot| unsafe { slot.assume_init() })
            .collect()
    };
    assert_eq!(init(&first), vec![-1.0, 2.0, -3.0]);
    assert_eq!(init(&second), vec![1.0, 2.0, 3.0]);
    assert_eq!(init(&third), vec![1.0, -2.0, 3.0]);
}

#[test]
fn erased_fused_plan_executes_multiple_outputs_for_integer_dtype() {
    let dims = [3usize];
    let unit = [1isize];
    let x = [3i32, -2, 5];
    let y = [1i32, 4, -1];
    let plan = ErasedFusedPlan::compile(
        KernelDType::I32,
        FusedPlan {
            input_count: 2,
            outputs: vec![2, 3].into_iter().collect(),
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
        },
    )
    .unwrap();
    let inputs = [
        ErasedRawStridedRef::from_slice(&x, &dims, &unit, 0).unwrap(),
        ErasedRawStridedRef::from_slice(&y, &dims, &unit, 0).unwrap(),
    ];
    let mut sums = [0i32; 3];
    let mut scaled = [0i32; 3];
    {
        let mut dests = [
            ErasedRawStridedMut::from_slice_mut(&mut sums, &dims, &unit, 0).unwrap(),
            ErasedRawStridedMut::from_slice_mut(&mut scaled, &dims, &unit, 0).unwrap(),
        ];
        plan.execute_outputs(&mut dests, &inputs, Exec::serial())
            .unwrap();
    }
    assert_eq!(sums, [4, 2, 4]);
    assert_eq!(scaled, [12, -4, 20]);
}

#[test]
fn erased_fused_plan_executes_rank_zero_multiple_outputs() {
    let dims: [usize; 0] = [];
    let strides: [isize; 0] = [];
    let input = [2.0f64];
    let plan = ErasedFusedPlan::compile(
        KernelDType::F64,
        FusedPlan {
            input_count: 1,
            outputs: vec![0, 1].into_iter().collect(),
            ops: vec![FusedInst {
                op: FusedOp::Negate,
                inputs: vec![0].into_iter().collect(),
            }]
            .into_iter()
            .collect(),
        },
    )
    .unwrap();
    let inputs = [ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap()];
    let mut pass_through = [0.0f64; 1];
    let mut negated = [0.0f64; 1];
    {
        let mut dests = [
            ErasedRawStridedMut::from_slice_mut(&mut pass_through, &dims, &strides, 0).unwrap(),
            ErasedRawStridedMut::from_slice_mut(&mut negated, &dims, &strides, 0).unwrap(),
        ];
        plan.execute_outputs(&mut dests, &inputs, Exec::serial())
            .unwrap();
    }
    assert_eq!(pass_through, [2.0]);
    assert_eq!(negated, [-2.0]);
}

#[test]
fn erased_fused_plan_executes_rank_zero_uninit_multiple_outputs() {
    let dims: [usize; 0] = [];
    let strides: [isize; 0] = [];
    let input = [2.0f64];
    let plan = ErasedFusedPlan::compile(
        KernelDType::F64,
        FusedPlan {
            input_count: 1,
            outputs: vec![0, 1, 2].into_iter().collect(),
            ops: vec![
                FusedInst {
                    op: FusedOp::Negate,
                    inputs: vec![0].into_iter().collect(),
                },
                FusedInst {
                    op: FusedOp::Negate,
                    inputs: vec![1].into_iter().collect(),
                },
            ]
            .into_iter()
            .collect(),
        },
    )
    .unwrap();
    let input_ref = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let pointers = [ErasedRawStridedPtr::from_ref(&input_ref)];
    let mut pass_through = [MaybeUninit::<f64>::uninit(); 1];
    let mut negated = [MaybeUninit::<f64>::uninit(); 1];
    let mut twice = [MaybeUninit::<f64>::uninit(); 1];
    {
        let mut dests = [
            ErasedRawStridedUninitMut::from_uninit_slice(&mut pass_through, &dims, &strides, 0)
                .unwrap(),
            ErasedRawStridedUninitMut::from_uninit_slice(&mut negated, &dims, &strides, 0).unwrap(),
            ErasedRawStridedUninitMut::from_uninit_slice(&mut twice, &dims, &strides, 0).unwrap(),
        ];
        plan.execute_outputs_uninit(&mut dests, &pointers, Exec::serial())
            .unwrap();
    }
    // SAFETY: `execute_outputs_uninit` writes every reachable element.
    assert_eq!(unsafe { pass_through[0].assume_init() }, 2.0);
    // SAFETY: as above.
    assert_eq!(unsafe { negated[0].assume_init() }, -2.0);
    // SAFETY: as above.
    assert_eq!(unsafe { twice[0].assume_init() }, 2.0);
}

#[test]
fn erased_fused_plan_writes_a_repeated_output_id_to_every_destination() {
    let dims = [3usize];
    let unit = [1isize];
    let input = [1.0f64, -2.0, 3.0];
    let plan = ErasedFusedPlan::compile(
        KernelDType::F64,
        FusedPlan {
            input_count: 1,
            outputs: vec![0, 0, 1].into_iter().collect(),
            ops: vec![FusedInst {
                op: FusedOp::Negate,
                inputs: vec![0].into_iter().collect(),
            }]
            .into_iter()
            .collect(),
        },
    )
    .unwrap();
    let inputs = [ErasedRawStridedRef::from_slice(&input, &dims, &unit, 0).unwrap()];
    let mut first = [0.0f64; 3];
    let mut second = [0.0f64; 3];
    let mut negated = [0.0f64; 3];
    {
        let mut dests = [
            ErasedRawStridedMut::from_slice_mut(&mut first, &dims, &unit, 0).unwrap(),
            ErasedRawStridedMut::from_slice_mut(&mut second, &dims, &unit, 0).unwrap(),
            ErasedRawStridedMut::from_slice_mut(&mut negated, &dims, &unit, 0).unwrap(),
        ];
        plan.execute_outputs(&mut dests, &inputs, Exec::serial())
            .unwrap();
    }
    assert_eq!(first, input);
    assert_eq!(second, input);
    assert_eq!(negated, [-1.0, 2.0, -3.0]);
}

#[test]
fn erased_fused_plan_rejects_destination_count_mismatch() {
    let dims = [2usize];
    let unit = [1isize];
    let input = [1.0f64, 2.0];
    let plan = ErasedFusedPlan::compile(
        KernelDType::F64,
        FusedPlan {
            input_count: 1,
            outputs: vec![0, 1].into_iter().collect(),
            ops: vec![FusedInst {
                op: FusedOp::Negate,
                inputs: vec![0].into_iter().collect(),
            }]
            .into_iter()
            .collect(),
        },
    )
    .unwrap();
    let inputs = [ErasedRawStridedRef::from_slice(&input, &dims, &unit, 0).unwrap()];
    let mut only = [9.0f64, 10.0];
    {
        let mut dests = [ErasedRawStridedMut::from_slice_mut(&mut only, &dims, &unit, 0).unwrap()];
        let err = plan
            .execute_outputs(&mut dests, &inputs, Exec::serial())
            .unwrap_err();
        assert!(matches!(err, StridedError::RankMismatch(1, 2)));
    }
    assert_eq!(only, [9.0, 10.0]);
}

#[test]
fn erased_fused_plan_rejects_input_output_overlap_for_uninit_outputs() {
    let dims = [2usize];
    let unit = [1isize];
    let plan = ErasedFusedPlan::compile(
        KernelDType::F64,
        FusedPlan {
            input_count: 1,
            outputs: vec![0, 1].into_iter().collect(),
            ops: vec![FusedInst {
                op: FusedOp::Negate,
                inputs: vec![0].into_iter().collect(),
            }]
            .into_iter()
            .collect(),
        },
    )
    .unwrap();
    let mut storage = [1.0f64, 2.0, 3.0, 4.0];
    let byte_len = core::mem::size_of_val(&storage);
    let base = std::ptr::NonNull::new(storage.as_mut_ptr().cast::<u8>()).unwrap();
    // SAFETY: both descriptors alias the same allocation on purpose; the overlap
    // check must reject before any reference is formed or any write happens.
    let input = unsafe {
        ErasedRawStridedPtr::from_raw_parts(KernelDType::F64, base, byte_len, &dims, &unit, 0)
    }
    .unwrap();
    let overlapping = unsafe {
        ErasedRawStridedUninitMut::from_raw_parts(KernelDType::F64, base, byte_len, &dims, &unit, 0)
    }
    .unwrap();
    // The first destination is disjoint and pre-filled. The overlap in the
    // second must be rejected before the first is written.
    let mut first_storage = [MaybeUninit::new(7.5f64); 2];
    let first =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut first_storage, &dims, &unit, 0).unwrap();
    let err = {
        let mut dests = [first, overlapping];
        plan.execute_outputs_uninit(&mut dests, &[input], Exec::serial())
            .unwrap_err()
    };
    assert!(matches!(
        err,
        StridedError::OverlappingInputOutput { input: 0 }
    ));
    for (index, slot) in first_storage.iter().enumerate() {
        // SAFETY: the sentinel slots were initialized before the call.
        assert_eq!(
            unsafe { slot.assume_init() },
            7.5,
            "first destination element {index} was written before validation failed"
        );
    }
    assert_eq!(storage, [1.0, 2.0, 3.0, 4.0]);
}
