use num_complex::{Complex32, Complex64};
use strided_kernel::{
    ErasedFusedPlan, ErasedRawStridedMut, ErasedRawStridedRef, ExecContext, FusedInst, FusedOp,
    FusedPlan, KernelDType, StridedError,
};

fn as_bytes<T>(data: &[T]) -> &[u8] {
    unsafe {
        core::slice::from_raw_parts(
            data.as_ptr().cast::<u8>(),
            data.len() * core::mem::size_of::<T>(),
        )
    }
}

fn as_bytes_mut<T>(data: &mut [T]) -> &mut [u8] {
    unsafe {
        core::slice::from_raw_parts_mut(
            data.as_mut_ptr().cast::<u8>(),
            data.len() * core::mem::size_of::<T>(),
        )
    }
}

fn binary_plan(op: FusedOp) -> FusedPlan {
    FusedPlan {
        input_count: 2,
        outputs: vec![2],
        ops: vec![FusedInst {
            op,
            inputs: vec![0, 1],
        }],
    }
}

fn unary_plan(op: FusedOp) -> FusedPlan {
    FusedPlan {
        input_count: 1,
        outputs: vec![1],
        ops: vec![FusedInst {
            op,
            inputs: vec![0],
        }],
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
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&a), &dims, &src_strides, 0).unwrap(),
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&b), &dims, &src_strides, 0).unwrap(),
    ];
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut dst),
        &dims,
        &dst_strides,
        0,
    )
    .unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &inputs)
        .unwrap();

    assert_eq!(dst, [100.0, 120.0, 102.0, 122.0, 104.0, 124.0]);
}

#[test]
fn erased_fused_plan_executes_f32_ternary_clamp_with_ambient_context() {
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
            outputs: vec![3],
            ops: vec![FusedInst {
                op: FusedOp::Clamp,
                inputs: vec![0, 1, 2],
            }],
        },
    )
    .unwrap();
    let inputs = [
        ErasedRawStridedRef::new(KernelDType::F32, as_bytes(&x), &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::new(KernelDType::F32, as_bytes(&min), &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::new(KernelDType::F32, as_bytes(&max), &dims, &strides, 0).unwrap(),
    ];
    let mut dest =
        ErasedRawStridedMut::new(KernelDType::F32, as_bytes_mut(&mut dst), &dims, &strides, 0)
            .unwrap();

    plan.execute(&ExecContext::ambient(), &mut dest, &inputs)
        .unwrap();

    assert_eq!(dst, [-1.0, 0.5, 3.0]);
}

#[test]
fn erased_fused_plan_executes_c32_unary_conjugate() {
    let dims = [2usize];
    let strides = [1isize];
    let input = [Complex32::new(1.0, -2.0), Complex32::new(-3.0, 4.0)];
    let mut dst = [Complex32::new(0.0, 0.0); 2];

    let plan = ErasedFusedPlan::compile(KernelDType::C32, unary_plan(FusedOp::Conj)).unwrap();
    let inputs = [
        ErasedRawStridedRef::new(KernelDType::C32, as_bytes(&input), &dims, &strides, 0).unwrap(),
    ];
    let mut dest =
        ErasedRawStridedMut::new(KernelDType::C32, as_bytes_mut(&mut dst), &dims, &strides, 0)
            .unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &inputs)
        .unwrap();

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
        ErasedRawStridedRef::new(KernelDType::C64, as_bytes(&a), &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::new(KernelDType::C64, as_bytes(&b), &dims, &strides, 0).unwrap(),
    ];
    let mut dest =
        ErasedRawStridedMut::new(KernelDType::C64, as_bytes_mut(&mut dst), &dims, &strides, 0)
            .unwrap();

    plan.execute(&ExecContext::max_threads(1).unwrap(), &mut dest, &inputs)
        .unwrap();

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
            outputs: vec![6],
            ops: vec![
                FusedInst {
                    op: FusedOp::Add,
                    inputs: vec![0, 1],
                },
                FusedInst {
                    op: FusedOp::Multiply,
                    inputs: vec![2, 3],
                },
                FusedInst {
                    op: FusedOp::Add,
                    inputs: vec![4, 5],
                },
            ],
        },
    )
    .unwrap();
    let inputs = [
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&a), &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&b), &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&c), &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&d), &dims, &strides, 0).unwrap(),
    ];
    let mut dest =
        ErasedRawStridedMut::new(KernelDType::F64, as_bytes_mut(&mut dst), &dims, &strides, 0)
            .unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &inputs)
        .unwrap();

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
            outputs: vec![10],
            ops: vec![
                FusedInst {
                    op: FusedOp::Add,
                    inputs: vec![0, 1],
                },
                FusedInst {
                    op: FusedOp::Multiply,
                    inputs: vec![4, 0],
                },
                FusedInst {
                    op: FusedOp::Negate,
                    inputs: vec![5],
                },
                FusedInst {
                    op: FusedOp::Abs,
                    inputs: vec![6],
                },
                FusedInst {
                    op: FusedOp::Maximum,
                    inputs: vec![7, 2],
                },
                FusedInst {
                    op: FusedOp::Minimum,
                    inputs: vec![8, 3],
                },
                FusedInst {
                    op: FusedOp::Clamp,
                    inputs: vec![9, 2, 3],
                },
            ],
        },
    )
    .unwrap();
    let inputs = [
        ErasedRawStridedRef::new(KernelDType::I32, as_bytes(&x), &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::new(KernelDType::I32, as_bytes(&y), &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::new(KernelDType::I32, as_bytes(&lo), &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::new(KernelDType::I32, as_bytes(&hi), &dims, &strides, 0).unwrap(),
    ];
    let mut dest =
        ErasedRawStridedMut::new(KernelDType::I32, as_bytes_mut(&mut dst), &dims, &strides, 0)
            .unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &inputs)
        .unwrap();

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
        ErasedRawStridedRef::new(KernelDType::I64, as_bytes(&a), &dims, &src_strides, 0).unwrap(),
        ErasedRawStridedRef::new(KernelDType::I64, as_bytes(&b), &dims, &src_strides, 0).unwrap(),
    ];
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::I64,
        as_bytes_mut(&mut dst),
        &dims,
        &dst_strides,
        0,
    )
    .unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &inputs)
        .unwrap();

    assert_eq!(dst, [100, 120, 102, 122, 104, 124]);
}

#[test]
fn erased_fused_plan_executes_bool_identity_conj() {
    let dims = [3usize];
    let strides = [1isize];
    let input = [true, false, true];
    let mut dst = [false; 3];

    let plan = ErasedFusedPlan::compile(KernelDType::Bool, unary_plan(FusedOp::Conj)).unwrap();
    let inputs =
        [
            ErasedRawStridedRef::new(KernelDType::Bool, as_bytes(&input), &dims, &strides, 0)
                .unwrap(),
        ];
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::Bool,
        as_bytes_mut(&mut dst),
        &dims,
        &strides,
        0,
    )
    .unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &inputs)
        .unwrap();

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
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&a), &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&b), &dims, &strides, 0).unwrap(),
    ];
    let mut dest =
        ErasedRawStridedMut::new(KernelDType::F32, as_bytes_mut(&mut dst), &dims, &strides, 0)
            .unwrap();

    let err = plan
        .execute(&ExecContext::serial(), &mut dest, &inputs)
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
    let inputs =
        [ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&a), &dims, &strides, 0).unwrap()];
    let mut dest =
        ErasedRawStridedMut::new(KernelDType::F64, as_bytes_mut(&mut dst), &dims, &strides, 0)
            .unwrap();

    let err = plan
        .execute(&ExecContext::serial(), &mut dest, &inputs)
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
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&a), &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::new(KernelDType::F32, as_bytes(&b), &dims, &strides, 0).unwrap(),
    ];
    let mut dest =
        ErasedRawStridedMut::new(KernelDType::F64, as_bytes_mut(&mut dst), &dims, &strides, 0)
            .unwrap();

    let err = plan
        .execute(&ExecContext::serial(), &mut dest, &inputs)
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
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&a), &input_dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&b), &input_dims, &strides, 0).unwrap(),
    ];
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut dst),
        &dest_dims,
        &strides,
        0,
    )
    .unwrap();

    let err = plan
        .execute(&ExecContext::serial(), &mut dest, &inputs)
        .unwrap_err();

    assert!(matches!(err, StridedError::ShapeMismatch(_, _)));
    assert_eq!(dst, [9.0, 10.0]);
}

#[test]
fn erased_fused_plan_exposes_dtype_and_plan() {
    let plan = ErasedFusedPlan::compile(KernelDType::F64, binary_plan(FusedOp::Add)).unwrap();

    assert_eq!(plan.dtype(), KernelDType::F64);
    assert_eq!(plan.plan().input_count, 2);
    assert_eq!(plan.plan().outputs, [2]);
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
        outputs: vec![0],
        ops: vec![],
    };
    let unsupported_arity =
        ErasedFusedPlan::compile(KernelDType::F64, too_many_inputs).unwrap_err();
    assert!(matches!(
        unsupported_arity,
        StridedError::UnsupportedArity { arity: 5, max: 4 }
    ));

    let zero_inputs = FusedPlan {
        input_count: 0,
        outputs: vec![0],
        ops: vec![],
    };
    let unsupported_arity = ErasedFusedPlan::compile(KernelDType::F64, zero_inputs).unwrap_err();
    assert!(matches!(
        unsupported_arity,
        StridedError::UnsupportedArity { arity: 0, max: 4 }
    ));
}

#[test]
fn erased_fused_plan_rejects_invalid_output_contracts() {
    let output_count_mismatch = FusedPlan {
        input_count: 2,
        outputs: vec![2, 2],
        ops: vec![FusedInst {
            op: FusedOp::Add,
            inputs: vec![0, 1],
        }],
    };
    let err = ErasedFusedPlan::compile(KernelDType::F64, output_count_mismatch).unwrap_err();
    assert!(matches!(err, StridedError::RankMismatch(2, 1)));

    let invalid_output = FusedPlan {
        input_count: 2,
        outputs: vec![3],
        ops: vec![],
    };
    let err = ErasedFusedPlan::compile(KernelDType::F64, invalid_output).unwrap_err();
    assert!(matches!(
        err,
        StridedError::InvalidAxis { axis: 3, rank: 2 }
    ));
}
