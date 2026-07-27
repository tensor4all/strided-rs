use num_complex::{Complex32, Complex64};
use strided_kernel::{
    ErasedRawStridedMut, ErasedRawStridedRef, ErasedReducePlan, ExecContext, KernelDType, ReduceOp,
    StridedError,
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

#[test]
fn erased_reduce_plan_executes_f64_sum_transposed_layout() {
    let dims = [2usize, 3];
    let src_strides = [1isize, 2];
    let input = [0.0f64, 10.0, 1.0, 11.0, 2.0, 12.0];
    let mut output = [0.0f64];

    let plan =
        ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Sum, &dims, &src_strides).unwrap();
    let source =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&input), &dims, &src_strides, 0)
            .unwrap();
    let mut dest =
        ErasedRawStridedMut::new(KernelDType::F64, as_bytes_mut(&mut output), &[], &[], 0).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();

    assert_eq!(output, [36.0]);
}

#[test]
fn erased_reduce_plan_executes_c64_product() {
    let dims = [3usize];
    let strides = [1isize];
    let input = [
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, -1.0),
        Complex64::new(0.5, 1.0),
    ];
    let mut output = [Complex64::new(0.0, 0.0)];

    let plan =
        ErasedReducePlan::compile(KernelDType::C64, ReduceOp::Product, &dims, &strides).unwrap();
    let source =
        ErasedRawStridedRef::new(KernelDType::C64, as_bytes(&input), &dims, &strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::new(KernelDType::C64, as_bytes_mut(&mut output), &[], &[], 0).unwrap();

    plan.execute(&ExecContext::max_threads(1).unwrap(), &mut dest, &source)
        .unwrap();

    assert_eq!(output, [input[0] * input[1] * input[2]]);
}

#[test]
fn erased_reduce_plan_executes_i32_sum_with_ambient_context() {
    let dims = [4usize];
    let strides = [1isize];
    let input = [1i32, -2, 3, 4];
    let mut output = [0i32];

    let plan = ErasedReducePlan::compile(KernelDType::I32, ReduceOp::Sum, &dims, &strides).unwrap();
    let source =
        ErasedRawStridedRef::new(KernelDType::I32, as_bytes(&input), &dims, &strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::new(KernelDType::I32, as_bytes_mut(&mut output), &[], &[], 0).unwrap();

    plan.execute(&ExecContext::ambient(), &mut dest, &source)
        .unwrap();

    assert_eq!(output, [6]);
}

#[test]
fn erased_reduce_plan_executes_remaining_supported_dtype_set() {
    let dims = [2usize];
    let strides = [1isize];

    let input_f32 = [1.5f32, 2.5];
    let mut output_f32 = [0.0f32];
    let plan = ErasedReducePlan::compile(KernelDType::F32, ReduceOp::Sum, &dims, &strides).unwrap();
    assert_eq!(plan.dtype(), KernelDType::F32);
    assert_eq!(plan.op(), ReduceOp::Sum);
    let source =
        ErasedRawStridedRef::new(KernelDType::F32, as_bytes(&input_f32), &dims, &strides, 0)
            .unwrap();
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::F32,
        as_bytes_mut(&mut output_f32),
        &[1],
        &[1],
        0,
    )
    .unwrap();
    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();
    assert_eq!(output_f32, [4.0]);

    let input_i64 = [2i64, -3];
    let mut output_i64 = [0i64];
    let plan =
        ErasedReducePlan::compile(KernelDType::I64, ReduceOp::Product, &dims, &strides).unwrap();
    let source =
        ErasedRawStridedRef::new(KernelDType::I64, as_bytes(&input_i64), &dims, &strides, 0)
            .unwrap();
    let mut dest =
        ErasedRawStridedMut::new(KernelDType::I64, as_bytes_mut(&mut output_i64), &[], &[], 0)
            .unwrap();
    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();
    assert_eq!(output_i64, [-6]);

    let input_c32 = [Complex32::new(1.0, 1.0), Complex32::new(2.0, -1.0)];
    let mut output_c32 = [Complex32::new(0.0, 0.0)];
    let plan =
        ErasedReducePlan::compile(KernelDType::C32, ReduceOp::Product, &dims, &strides).unwrap();
    let source =
        ErasedRawStridedRef::new(KernelDType::C32, as_bytes(&input_c32), &dims, &strides, 0)
            .unwrap();
    let mut dest =
        ErasedRawStridedMut::new(KernelDType::C32, as_bytes_mut(&mut output_c32), &[], &[], 0)
            .unwrap();
    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();
    assert_eq!(output_c32, [input_c32[0] * input_c32[1]]);
}

#[test]
fn erased_reduce_plan_empty_input_returns_operation_identity() {
    let dims = [0usize, 3];
    let strides = [1isize, 0];
    let input: [f64; 0] = [];
    let mut sum_output = [9.0f64];
    let mut product_output = [9.0f64];

    let source =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&input), &dims, &strides, 0).unwrap();
    let mut sum_dest =
        ErasedRawStridedMut::new(KernelDType::F64, as_bytes_mut(&mut sum_output), &[], &[], 0)
            .unwrap();
    let mut product_dest = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut product_output),
        &[],
        &[],
        0,
    )
    .unwrap();

    ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Sum, &dims, &strides)
        .unwrap()
        .execute(&ExecContext::serial(), &mut sum_dest, &source)
        .unwrap();
    ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Product, &dims, &strides)
        .unwrap()
        .execute(&ExecContext::serial(), &mut product_dest, &source)
        .unwrap();

    assert_eq!(sum_output, [0.0]);
    assert_eq!(product_output, [1.0]);
}

#[test]
fn erased_reduce_plan_executes_single_axis_sum_into_strided_output() {
    let src_dims = [2usize, 3];
    let src_strides = [1isize, 2];
    let input = [0.0f64, 1.0, 10.0, 11.0, 20.0, 21.0];
    let dest_dims = [3usize];
    let dest_strides = [1isize];
    let mut output = [0.0f64; 3];

    let plan = ErasedReducePlan::compile_axes(
        KernelDType::F64,
        ReduceOp::Sum,
        &src_dims,
        &src_strides,
        &dest_dims,
        &dest_strides,
        &[0],
    )
    .unwrap();
    let source = ErasedRawStridedRef::new(
        KernelDType::F64,
        as_bytes(&input),
        &src_dims,
        &src_strides,
        0,
    )
    .unwrap();
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut output),
        &dest_dims,
        &dest_strides,
        0,
    )
    .unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();

    assert_eq!(output, [1.0, 21.0, 41.0]);
}

#[test]
fn erased_reduce_plan_executes_multi_axis_sum_with_kept_middle_axis() {
    let src_dims = [2usize, 3, 4];
    let src_strides = [1isize, 2, 6];
    let input: Vec<f64> = (0..src_dims[2])
        .flat_map(|k| {
            (0..src_dims[1]).flat_map(move |j| {
                (0..src_dims[0]).map(move |i| i as f64 + 10.0 * j as f64 + 100.0 * k as f64)
            })
        })
        .collect();
    let dest_dims = [3usize];
    let dest_strides = [1isize];
    let mut output = [0.0f64; 3];

    let plan = ErasedReducePlan::compile_axes(
        KernelDType::F64,
        ReduceOp::Sum,
        &src_dims,
        &src_strides,
        &dest_dims,
        &dest_strides,
        &[0, 2],
    )
    .unwrap();
    let source = ErasedRawStridedRef::new(
        KernelDType::F64,
        as_bytes(&input),
        &src_dims,
        &src_strides,
        0,
    )
    .unwrap();
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut output),
        &dest_dims,
        &dest_strides,
        0,
    )
    .unwrap();

    plan.execute(&ExecContext::max_threads(2).unwrap(), &mut dest, &source)
        .unwrap();

    assert_eq!(output, [1204.0, 1284.0, 1364.0]);
}

#[test]
fn erased_reduce_plan_executes_all_axes_sum_into_rank0_scalar() {
    let src_dims = [2usize, 3];
    let src_strides = [1isize, 2];
    let input = [0.0f64, 1.0, 10.0, 11.0, 20.0, 21.0];
    let dest_dims: [usize; 0] = [];
    let dest_strides: [isize; 0] = [];
    let mut output = [0.0f64];

    let plan = ErasedReducePlan::compile_axes(
        KernelDType::F64,
        ReduceOp::Sum,
        &src_dims,
        &src_strides,
        &dest_dims,
        &dest_strides,
        &[0, 1],
    )
    .unwrap();
    let source = ErasedRawStridedRef::new(
        KernelDType::F64,
        as_bytes(&input),
        &src_dims,
        &src_strides,
        0,
    )
    .unwrap();
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut output),
        &dest_dims,
        &dest_strides,
        0,
    )
    .unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();

    assert_eq!(output, [63.0]);
}

#[test]
fn erased_reduce_plan_axis_reduction_writes_identity_for_empty_reduced_domain() {
    let src_dims = [2usize, 0];
    let src_strides = [1isize, 2];
    let input: [f64; 0] = [];
    let dest_dims = [2usize];
    let dest_strides = [1isize];
    let mut output = [9.0f64, 10.0];

    let plan = ErasedReducePlan::compile_axes(
        KernelDType::F64,
        ReduceOp::Product,
        &src_dims,
        &src_strides,
        &dest_dims,
        &dest_strides,
        &[1],
    )
    .unwrap();
    let source = ErasedRawStridedRef::new(
        KernelDType::F64,
        as_bytes(&input),
        &src_dims,
        &src_strides,
        0,
    )
    .unwrap();
    let mut dest = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut output),
        &dest_dims,
        &dest_strides,
        0,
    )
    .unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();

    assert_eq!(output, [1.0, 1.0]);
}

#[test]
fn erased_reduce_plan_axis_compile_rejects_invalid_contracts() {
    let src_dims = [2usize, 3];
    let src_strides = [1isize, 2];
    let dest_dims = [3usize];
    let dest_strides = [1isize];

    let duplicate_axis = ErasedReducePlan::compile_axes(
        KernelDType::F64,
        ReduceOp::Sum,
        &src_dims,
        &src_strides,
        &dest_dims,
        &dest_strides,
        &[0, 0],
    )
    .unwrap_err();
    assert!(matches!(
        duplicate_axis,
        StridedError::InvalidAxis { axis: 0, rank: 2 }
    ));

    let shape_mismatch = ErasedReducePlan::compile_axes(
        KernelDType::F64,
        ReduceOp::Sum,
        &src_dims,
        &src_strides,
        &[2],
        &[1],
        &[0],
    )
    .unwrap_err();
    assert!(matches!(shape_mismatch, StridedError::ShapeMismatch(_, _)));

    let non_injective_dest = ErasedReducePlan::compile_axes(
        KernelDType::F64,
        ReduceOp::Sum,
        &src_dims,
        &src_strides,
        &dest_dims,
        &[0],
        &[0],
    )
    .unwrap_err();
    assert!(matches!(
        non_injective_dest,
        StridedError::NonInjectiveOutputLayout
    ));
}

#[test]
fn erased_reduce_plan_rejects_dtype_mismatch_before_writing() {
    let dims = [2usize];
    let strides = [1isize];
    let input = [1.0f64, 2.0];
    let mut output = [9.0f32];

    let plan = ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Sum, &dims, &strides).unwrap();
    let source =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&input), &dims, &strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::new(KernelDType::F32, as_bytes_mut(&mut output), &[], &[], 0).unwrap();

    let err = plan
        .execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap_err();

    assert!(matches!(err, StridedError::DTypeMismatch { .. }));
    assert_eq!(output, [9.0]);
}

#[test]
fn erased_reduce_plan_rejects_layout_mismatch_before_writing() {
    let compiled_dims = [2usize, 2];
    let compiled_strides = [1isize, 2];
    let runtime_strides = [2isize, 1];
    let input = [1.0f64, 2.0, 3.0, 4.0];
    let mut output = [9.0f64];

    let plan = ErasedReducePlan::compile(
        KernelDType::F64,
        ReduceOp::Sum,
        &compiled_dims,
        &compiled_strides,
    )
    .unwrap();
    let source = ErasedRawStridedRef::new(
        KernelDType::F64,
        as_bytes(&input),
        &compiled_dims,
        &runtime_strides,
        0,
    )
    .unwrap();
    let mut dest =
        ErasedRawStridedMut::new(KernelDType::F64, as_bytes_mut(&mut output), &[], &[], 0).unwrap();

    let err = plan
        .execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap_err();

    assert!(matches!(err, StridedError::PlanLayoutMismatch));
    assert_eq!(output, [9.0]);
}

#[test]
fn erased_reduce_plan_rejects_non_scalar_output_and_unsupported_dtype() {
    let dims = [2usize];
    let strides = [1isize];
    let input = [1.0f64, 2.0];
    let mut output = [9.0f64, 10.0];

    let plan = ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Sum, &dims, &strides).unwrap();
    let source =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&input), &dims, &strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::new(KernelDType::F64, as_bytes_mut(&mut output), &[2], &[1], 0)
            .unwrap();

    let err = plan
        .execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap_err();
    assert!(matches!(err, StridedError::RankMismatch(2, 1)));
    assert_eq!(output, [9.0, 10.0]);

    let unsupported =
        ErasedReducePlan::compile(KernelDType::Bool, ReduceOp::Sum, &dims, &strides).unwrap_err();
    assert!(matches!(
        unsupported,
        StridedError::UnsupportedDType { dtype: "bool" }
    ));
}

#[test]
fn erased_reduce_plan_rejects_invalid_compile_layout() {
    let err =
        ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Sum, &[2, 3], &[1]).unwrap_err();

    assert!(matches!(err, StridedError::StrideLengthMismatch));
}
