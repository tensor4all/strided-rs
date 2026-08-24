use num_complex::{Complex32, Complex64};
use strided_kernel::{
    ErasedRawStridedMut, ErasedRawStridedRef, ErasedReducePlan, ExecContext, KernelDType,
    MaybeSimdOps, ReduceOp, StridedError,
};

#[test]
fn erased_reduce_plan_executes_f64_sum_transposed_layout() {
    let dims = [2usize, 3];
    let src_strides = [1isize, 2];
    let input = [0.0f64, 10.0, 1.0, 11.0, 2.0, 12.0];
    let mut output = [0.0f64];

    let plan =
        ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Sum, &dims, &src_strides).unwrap();
    let source = ErasedRawStridedRef::from_slice(&input, &dims, &src_strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[], &[], 0).unwrap();

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
    let source = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[], &[], 0).unwrap();

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
    let source = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[], &[], 0).unwrap();

    plan.execute(&ExecContext::ambient(), &mut dest, &source)
        .unwrap();

    assert_eq!(output, [6]);
}

#[test]
fn erased_reduce_plan_integer_full_reductions_use_wrapping_arithmetic() {
    let dims = [2usize];
    let strides = [1isize];

    let input_i32 = [i32::MAX, 1];
    let mut sum_output = [0i32];
    let sum_plan =
        ErasedReducePlan::compile(KernelDType::I32, ReduceOp::Sum, &dims, &strides).unwrap();
    let sum_source = ErasedRawStridedRef::from_slice(&input_i32, &dims, &strides, 0).unwrap();
    let mut sum_dest = ErasedRawStridedMut::from_slice_mut(&mut sum_output, &[], &[], 0).unwrap();

    sum_plan
        .execute(&ExecContext::serial(), &mut sum_dest, &sum_source)
        .unwrap();

    assert_eq!(sum_output, [i32::MIN]);

    let input_i64 = [i64::MAX, 2];
    let mut product_output = [0i64];
    let product_plan =
        ErasedReducePlan::compile(KernelDType::I64, ReduceOp::Product, &dims, &strides).unwrap();
    let product_source = ErasedRawStridedRef::from_slice(&input_i64, &dims, &strides, 0).unwrap();
    let mut product_dest =
        ErasedRawStridedMut::from_slice_mut(&mut product_output, &[], &[], 0).unwrap();

    product_plan
        .execute(&ExecContext::serial(), &mut product_dest, &product_source)
        .unwrap();

    assert_eq!(product_output, [i64::MAX.wrapping_mul(2)]);
}

#[test]
fn erased_reduce_plan_serial_full_sum_uses_fixed_multi_accumulator_order() {
    let dims = [9usize];
    let strides = [1isize];
    let input = [1.0e16f64, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0e16];
    let plan = ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Sum, &dims, &strides).unwrap();
    let source = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();

    let mut serial_output = [0.0f64];
    let mut serial_dest =
        ErasedRawStridedMut::from_slice_mut(&mut serial_output, &[], &[], 0).unwrap();
    plan.execute(&ExecContext::serial(), &mut serial_dest, &source)
        .unwrap();

    let mut one_thread_output = [0.0f64];
    let mut one_thread_dest =
        ErasedRawStridedMut::from_slice_mut(&mut one_thread_output, &[], &[], 0).unwrap();
    plan.execute(
        &ExecContext::max_threads(1).unwrap(),
        &mut one_thread_dest,
        &source,
    )
    .unwrap();

    let expected = <f64 as MaybeSimdOps>::try_simd_sum(&input).unwrap_or(7.0);
    assert_eq!(serial_output[0].to_bits(), expected.to_bits());
    assert_eq!(one_thread_output[0].to_bits(), serial_output[0].to_bits());
}

#[cfg(feature = "parallel")]
#[test]
fn erased_reduce_plan_fixed_bounded_context_is_bitwise_repeatable() {
    let dims = [131_073usize];
    let strides = [1isize];
    let input = (0..dims[0])
        .map(|index| match index % 4 {
            0 => 1.0e12,
            1 => 1.0,
            2 => -1.0e12,
            _ => -0.5,
        })
        .collect::<Vec<f64>>();
    let plan = ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Sum, &dims, &strides).unwrap();
    let source = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let ctx = ExecContext::max_threads(4).unwrap();
    let mut expected = None;

    for _ in 0..8 {
        let mut output = [0.0f64];
        let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[], &[], 0).unwrap();
        plan.execute(&ctx, &mut dest, &source).unwrap();
        let bits = output[0].to_bits();
        assert_eq!(*expected.get_or_insert(bits), bits);
    }
}

#[cfg(feature = "parallel")]
#[test]
fn erased_sum_squares_fixed_bounded_context_is_bitwise_repeatable() {
    let dims = [131_073usize];
    let strides = [1isize];
    let input = (0..dims[0])
        .map(|index| match index % 4 {
            0 => 1.0e6,
            1 => 1.0,
            2 => -1.0e6,
            _ => -0.5,
        })
        .collect::<Vec<f64>>();
    let plan =
        ErasedReducePlan::compile(KernelDType::F64, ReduceOp::SumSquares, &dims, &strides).unwrap();
    let source = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let ctx = ExecContext::max_threads(4).unwrap();
    let mut expected = None;

    for _ in 0..8 {
        let mut output = [0.0f64];
        let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[], &[], 0).unwrap();
        plan.execute(&ctx, &mut dest, &source).unwrap();
        let bits = output[0].to_bits();
        assert_eq!(*expected.get_or_insert(bits), bits);
    }
}

#[test]
fn erased_reduce_plan_preserves_noncompact_and_nonfinite_semantics() {
    let dims = [2usize, 2];
    let strides = [1isize, 3];
    let input = [
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, -1.0),
        Complex64::new(99.0, 99.0),
        Complex64::new(-2.0, 0.5),
        Complex64::new(4.0, 1.5),
    ];
    let plan = ErasedReducePlan::compile(KernelDType::C64, ReduceOp::Sum, &dims, &strides).unwrap();
    let source = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let mut output = [Complex64::new(0.0, 0.0)];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[], &[], 0).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();

    assert_eq!(output, [input[0] + input[1] + input[3] + input[4]]);

    let nonfinite_input = [f64::INFINITY, 2.0, f64::NEG_INFINITY];
    let nonfinite_dims = [nonfinite_input.len()];
    let nonfinite_strides = [1isize];
    let nonfinite_plan = ErasedReducePlan::compile(
        KernelDType::F64,
        ReduceOp::Sum,
        &nonfinite_dims,
        &nonfinite_strides,
    )
    .unwrap();
    let nonfinite_source =
        ErasedRawStridedRef::from_slice(&nonfinite_input, &nonfinite_dims, &nonfinite_strides, 0)
            .unwrap();
    let mut nonfinite_output = [0.0f64];
    let mut nonfinite_dest =
        ErasedRawStridedMut::from_slice_mut(&mut nonfinite_output, &[], &[], 0).unwrap();

    nonfinite_plan
        .execute(
            &ExecContext::serial(),
            &mut nonfinite_dest,
            &nonfinite_source,
        )
        .unwrap();

    assert!(nonfinite_output[0].is_nan());
}

#[test]
fn erased_reduce_plan_contiguous_product_covers_vector_body_and_tail() {
    let dims = [65usize];
    let strides = [1isize];
    let input = (0..dims[0])
        .map(|index| if index % 2 == 0 { 2.0f64 } else { 0.5 })
        .collect::<Vec<_>>();
    let plan =
        ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Product, &dims, &strides).unwrap();
    let source = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let mut output = [0.0f64];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[], &[], 0).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();

    assert_eq!(output, [2.0]);
}

#[test]
fn erased_reduce_plan_contiguous_fast_path_honors_offset_and_empty_layout() {
    let storage = [99.0f64, 88.0, 1.0, 2.0, 3.0, 77.0];
    let dims = [3usize];
    let strides = [1isize];
    let plan = ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Sum, &dims, &strides).unwrap();
    let source = ErasedRawStridedRef::from_slice(&storage, &dims, &strides, 2).unwrap();
    let mut output = [0.0f64];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[], &[], 0).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();

    assert_eq!(output, [6.0]);

    let empty_dims = [0usize];
    let empty_plan =
        ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Product, &empty_dims, &strides)
            .unwrap();
    let empty_source =
        ErasedRawStridedRef::from_slice::<f64>(&[], &empty_dims, &strides, isize::MAX).unwrap();
    let mut empty_output = [0.0f64];
    let mut empty_dest =
        ErasedRawStridedMut::from_slice_mut(&mut empty_output, &[], &[], 0).unwrap();

    empty_plan
        .execute(&ExecContext::serial(), &mut empty_dest, &empty_source)
        .unwrap();

    assert_eq!(empty_output, [1.0]);
}

#[test]
fn erased_reduce_plan_product_documents_reassociated_overflow_classification() {
    let dims = [65usize];
    let strides = [1isize];
    let mut input = [1.0f64; 65];
    input[0] = f64::MAX;
    input[1] = f64::MIN_POSITIVE;
    input[2] = f64::MIN_POSITIVE;
    input[3] = f64::MIN_POSITIVE;
    input[32] = f64::MAX;
    let left_fold = input.iter().copied().fold(1.0, |acc, value| acc * value);
    let plan =
        ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Product, &dims, &strides).unwrap();
    let source = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let mut output = [0.0f64];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[], &[], 0).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();

    assert_eq!(left_fold, 0.0);
    assert!(!output[0].is_finite());
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
    let source = ErasedRawStridedRef::from_slice(&input_f32, &dims, &strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output_f32, &[1], &[1], 0).unwrap();
    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();
    assert_eq!(output_f32, [4.0]);

    let input_i64 = [2i64, -3];
    let mut output_i64 = [0i64];
    let plan =
        ErasedReducePlan::compile(KernelDType::I64, ReduceOp::Product, &dims, &strides).unwrap();
    let source = ErasedRawStridedRef::from_slice(&input_i64, &dims, &strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output_i64, &[], &[], 0).unwrap();
    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();
    assert_eq!(output_i64, [-6]);

    let input_c32 = [Complex32::new(1.0, 1.0), Complex32::new(2.0, -1.0)];
    let mut output_c32 = [Complex32::new(0.0, 0.0)];
    let plan =
        ErasedReducePlan::compile(KernelDType::C32, ReduceOp::Product, &dims, &strides).unwrap();
    let source = ErasedRawStridedRef::from_slice(&input_c32, &dims, &strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output_c32, &[], &[], 0).unwrap();
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

    let source = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let mut sum_dest = ErasedRawStridedMut::from_slice_mut(&mut sum_output, &[], &[], 0).unwrap();
    let mut product_dest =
        ErasedRawStridedMut::from_slice_mut(&mut product_output, &[], &[], 0).unwrap();

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
    let source = ErasedRawStridedRef::from_slice(&input, &src_dims, &src_strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();

    assert_eq!(output, [1.0, 21.0, 41.0]);
}

#[test]
fn erased_reduce_plan_integer_axis_reductions_use_wrapping_arithmetic() {
    let src_dims = [2usize, 2];
    let src_strides = [1isize, 2];
    let dest_dims = [2usize];
    let dest_strides = [1isize];
    let input = [i32::MAX, 1, i32::MIN, -1];
    let mut output = [0i32; 2];

    let plan = ErasedReducePlan::compile_axes(
        KernelDType::I32,
        ReduceOp::Sum,
        &src_dims,
        &src_strides,
        &dest_dims,
        &dest_strides,
        &[0],
    )
    .unwrap();
    let source = ErasedRawStridedRef::from_slice(&input, &src_dims, &src_strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();

    assert_eq!(output, [i32::MIN, i32::MAX]);
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
    let source = ErasedRawStridedRef::from_slice(&input, &src_dims, &src_strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0).unwrap();

    plan.execute(&ExecContext::max_threads(2).unwrap(), &mut dest, &source)
        .unwrap();

    assert_eq!(output, [1204.0, 1284.0, 1364.0]);
}

#[test]
fn erased_reduce_plan_rank8_reordered_axes_preserve_fold_order() {
    let src_dims = [2usize; 8];
    let src_strides = [1isize, 2, 4, 8, 16, 32, 64, 128];
    let input: Vec<f64> = (0..256)
        .map(|index| match index % 4 {
            0 => 1.0e16,
            1 => index as f64,
            2 => -1.0e16,
            _ => -(index as f64),
        })
        .collect();
    let axes = [6usize, 0, 3];
    let kept_axes = [1usize, 2, 4, 5, 7];
    let dest_dims = [2usize; 5];
    let dest_strides = [1isize, 2, 4, 8, 16];
    let mut output = [0.0f64; 32];
    let plan = ErasedReducePlan::compile_axes(
        KernelDType::F64,
        ReduceOp::Sum,
        &src_dims,
        &src_strides,
        &dest_dims,
        &dest_strides,
        &axes,
    )
    .unwrap();
    let source = ErasedRawStridedRef::from_slice(&input, &src_dims, &src_strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();

    let mut expected = [0.0f64; 32];
    for (output_linear, expected_value) in expected.iter_mut().enumerate() {
        let mut source_index = [0usize; 8];
        let mut linear = output_linear;
        for &axis in &kept_axes {
            source_index[axis] = linear % 2;
            linear /= 2;
        }
        let mut acc = 0.0f64;
        for reduced_linear in 0..8 {
            let mut linear = reduced_linear;
            for &axis in &axes {
                source_index[axis] = linear % 2;
                linear /= 2;
            }
            let offset = source_index
                .iter()
                .zip(src_strides)
                .map(|(&coord, stride)| coord as isize * stride)
                .sum::<isize>() as usize;
            acc += input[offset];
        }
        *expected_value = acc;
    }
    assert_eq!(
        output.map(f64::to_bits),
        expected.map(f64::to_bits),
        "caller-supplied reduced-axis order defines the fold order"
    );
}

#[test]
fn erased_reduce_plan_fused_rank8_axes_match_contiguous_order() {
    let src_dims = [2usize; 8];
    let src_strides = [1isize, 2, 4, 8, 16, 32, 64, 128];
    let input: Vec<i32> = (0..256).collect();
    let dest_dims = [2usize];
    let dest_strides = [1isize];
    let mut output = [0i32; 2];
    let plan = ErasedReducePlan::compile_axes(
        KernelDType::I32,
        ReduceOp::Sum,
        &src_dims,
        &src_strides,
        &dest_dims,
        &dest_strides,
        &[0, 1, 2, 3, 4, 5, 6],
    )
    .unwrap();
    let source = ErasedRawStridedRef::from_slice(&input, &src_dims, &src_strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();

    assert_eq!(output, [8128, 24512]);
}

#[test]
fn erased_reduce_plan_partially_fused_inner_axes_preserve_order() {
    let src_dims = [2usize, 3, 2, 2];
    let src_strides = [1isize, 2, 6, 12];
    let input: Vec<f64> = (0..24)
        .map(|index| match index % 3 {
            0 => 1.0e16,
            1 => index as f64,
            _ => -1.0e16,
        })
        .collect();
    let dest_dims = [2usize];
    let dest_strides = [1isize];
    let axes = [0usize, 1, 3];
    let mut output = [0.0f64; 2];
    let plan = ErasedReducePlan::compile_axes(
        KernelDType::F64,
        ReduceOp::Sum,
        &src_dims,
        &src_strides,
        &dest_dims,
        &dest_strides,
        &axes,
    )
    .unwrap();
    let source = ErasedRawStridedRef::from_slice(&input, &src_dims, &src_strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();

    let mut expected = [0.0f64; 2];
    for kept in 0..2 {
        let mut acc = 0.0;
        for linear in 0..12 {
            let axis0 = linear % 2;
            let axis1 = (linear / 2) % 3;
            let axis3 = linear / 6;
            let offset = axis0 + 2 * axis1 + 6 * kept + 12 * axis3;
            acc += input[offset];
        }
        expected[kept] = acc;
    }
    assert_eq!(output.map(f64::to_bits), expected.map(f64::to_bits));
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
    let source = ErasedRawStridedRef::from_slice(&input, &src_dims, &src_strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0).unwrap();

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
    let source = ErasedRawStridedRef::from_slice(&input, &src_dims, &src_strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0).unwrap();

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
    let source = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[], &[], 0).unwrap();

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
    let source =
        ErasedRawStridedRef::from_slice(&input, &compiled_dims, &runtime_strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[], &[], 0).unwrap();

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
    let source = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[2], &[1], 0).unwrap();

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

#[test]
fn erased_reduce_plan_sum_squares_matches_materialized_full_reduction() {
    let dims = [5usize];
    let strides = [-1isize];
    let input = [1.5f64, -2.0, 3.25, -4.5, 0.125, 99.0];
    let expected = input[..5].iter().map(|&value| value * value).sum::<f64>();
    let mut serial_output = [0.0f64];
    let mut max_one_output = [0.0f64];
    let plan =
        ErasedReducePlan::compile(KernelDType::F64, ReduceOp::SumSquares, &dims, &strides).unwrap();
    let source = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 4).unwrap();

    let mut serial_dest =
        ErasedRawStridedMut::from_slice_mut(&mut serial_output, &[], &[], 0).unwrap();
    plan.execute(&ExecContext::serial(), &mut serial_dest, &source)
        .unwrap();

    let mut max_one_dest =
        ErasedRawStridedMut::from_slice_mut(&mut max_one_output, &[], &[], 0).unwrap();
    plan.execute(
        &ExecContext::max_threads(1).unwrap(),
        &mut max_one_dest,
        &source,
    )
    .unwrap();

    assert_eq!(serial_output, [expected]);
    assert_eq!(max_one_output[0].to_bits(), serial_output[0].to_bits());
}

#[test]
fn erased_reduce_plan_sum_squares_is_bitwise_materialized_sum() {
    let dims = [257usize];
    let strides = [1isize];
    let input = (0..dims[0])
        .map(|index| match index % 5 {
            0 => 1.0e100,
            1 => -1.0e-100,
            2 => index as f64 * 0.25,
            3 => -3.5,
            _ => 0.0,
        })
        .collect::<Vec<_>>();
    let squared = input
        .iter()
        .copied()
        .map(|value| value * value)
        .collect::<Vec<_>>();
    let source = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let squared_source = ErasedRawStridedRef::from_slice(&squared, &dims, &strides, 0).unwrap();
    let sum_squares_plan =
        ErasedReducePlan::compile(KernelDType::F64, ReduceOp::SumSquares, &dims, &strides).unwrap();
    let sum_plan =
        ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Sum, &dims, &strides).unwrap();
    let mut fused_output = [0.0f64];
    let mut materialized_output = [0.0f64];
    let mut fused_dest =
        ErasedRawStridedMut::from_slice_mut(&mut fused_output, &[], &[], 0).unwrap();
    let mut materialized_dest =
        ErasedRawStridedMut::from_slice_mut(&mut materialized_output, &[], &[], 0).unwrap();

    sum_squares_plan
        .execute(&ExecContext::serial(), &mut fused_dest, &source)
        .unwrap();
    sum_plan
        .execute(
            &ExecContext::serial(),
            &mut materialized_dest,
            &squared_source,
        )
        .unwrap();

    assert_eq!(fused_output[0].to_bits(), materialized_output[0].to_bits());
}

#[test]
fn erased_reduce_plan_sum_squares_matches_materialized_axis_reduction() {
    let src_dims = [2usize, 3, 2];
    let src_strides = [1isize, 2, 6];
    let input = [
        1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0,
    ];
    let dest_dims = [3usize];
    let dest_strides = [2isize];
    let mut output = [-1.0f32; 5];
    let plan = ErasedReducePlan::compile_axes(
        KernelDType::F32,
        ReduceOp::SumSquares,
        &src_dims,
        &src_strides,
        &dest_dims,
        &dest_strides,
        &[0, 2],
    )
    .unwrap();
    let source = ErasedRawStridedRef::from_slice(&input, &src_dims, &src_strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0).unwrap();

    plan.execute(&ExecContext::max_threads(2).unwrap(), &mut dest, &source)
        .unwrap();

    let expected = [
        1.0f32.powi(2) + (-2.0f32).powi(2) + 7.0f32.powi(2) + (-8.0f32).powi(2),
        3.0f32.powi(2) + (-4.0f32).powi(2) + 9.0f32.powi(2) + (-10.0f32).powi(2),
        5.0f32.powi(2) + (-6.0f32).powi(2) + 11.0f32.powi(2) + (-12.0f32).powi(2),
    ];
    assert_eq!(output, [expected[0], -1.0, expected[1], -1.0, expected[2]]);
}

#[test]
fn erased_reduce_plan_sum_squares_axis_handles_negative_stride_offset_and_zero_extent() {
    let src_dims = [2usize, 3];
    let src_strides = [-1isize, 2];
    let input = [1.0f64, -2.0, 3.0, -4.0, 5.0, -6.0];
    let dest_dims = [3usize];
    let dest_strides = [2isize];
    let mut output = [-1.0f64; 6];
    let plan = ErasedReducePlan::compile_axes(
        KernelDType::F64,
        ReduceOp::SumSquares,
        &src_dims,
        &src_strides,
        &dest_dims,
        &dest_strides,
        &[0],
    )
    .unwrap();
    let source = ErasedRawStridedRef::from_slice(&input, &src_dims, &src_strides, 1).unwrap();
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 1).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();
    assert_eq!(output, [-1.0, 5.0, -1.0, 25.0, -1.0, 61.0]);

    let empty_src_dims = [2usize, 0];
    let empty_src_strides = [-1isize, 2];
    let empty_dest_dims = [2usize];
    let empty_dest_strides = [1isize];
    let empty_plan = ErasedReducePlan::compile_axes(
        KernelDType::F64,
        ReduceOp::SumSquares,
        &empty_src_dims,
        &empty_src_strides,
        &empty_dest_dims,
        &empty_dest_strides,
        &[1],
    )
    .unwrap();
    let empty_source = ErasedRawStridedRef::from_slice::<f64>(
        &[],
        &empty_src_dims,
        &empty_src_strides,
        isize::MAX,
    )
    .unwrap();
    let mut empty_output = [9.0f64, 10.0];
    let mut empty_dest = ErasedRawStridedMut::from_slice_mut(
        &mut empty_output,
        &empty_dest_dims,
        &empty_dest_strides,
        0,
    )
    .unwrap();

    empty_plan
        .execute(&ExecContext::serial(), &mut empty_dest, &empty_source)
        .unwrap();
    assert_eq!(empty_output, [0.0, 0.0]);
}

#[test]
fn erased_reduce_plan_sum_squares_handles_empty_rank_zero_and_nonfinite_values() {
    let empty_dims = [0usize];
    let strides = [1isize];
    let empty_plan = ErasedReducePlan::compile(
        KernelDType::F64,
        ReduceOp::SumSquares,
        &empty_dims,
        &strides,
    )
    .unwrap();
    let empty_source =
        ErasedRawStridedRef::from_slice::<f64>(&[], &empty_dims, &strides, isize::MAX).unwrap();
    let mut empty_output = [9.0f64];
    let mut empty_dest =
        ErasedRawStridedMut::from_slice_mut(&mut empty_output, &[], &[], 0).unwrap();
    empty_plan
        .execute(&ExecContext::serial(), &mut empty_dest, &empty_source)
        .unwrap();
    assert_eq!(empty_output, [0.0]);

    let scalar = [-0.0f64];
    let scalar_plan =
        ErasedReducePlan::compile(KernelDType::F64, ReduceOp::SumSquares, &[], &[]).unwrap();
    let scalar_source = ErasedRawStridedRef::from_slice(&scalar, &[], &[], 0).unwrap();
    let mut scalar_output = [1.0f64];
    let mut scalar_dest =
        ErasedRawStridedMut::from_slice_mut(&mut scalar_output, &[], &[], 0).unwrap();
    scalar_plan
        .execute(&ExecContext::serial(), &mut scalar_dest, &scalar_source)
        .unwrap();
    assert_eq!(scalar_output[0].to_bits(), 0.0f64.to_bits());

    let nonfinite = [f64::INFINITY, f64::NAN];
    let nonfinite_plan =
        ErasedReducePlan::compile(KernelDType::F64, ReduceOp::SumSquares, &[2], &[1]).unwrap();
    let nonfinite_source = ErasedRawStridedRef::from_slice(&nonfinite, &[2], &[1], 0).unwrap();
    let mut nonfinite_output = [0.0f64];
    let mut nonfinite_dest =
        ErasedRawStridedMut::from_slice_mut(&mut nonfinite_output, &[], &[], 0).unwrap();
    nonfinite_plan
        .execute(
            &ExecContext::serial(),
            &mut nonfinite_dest,
            &nonfinite_source,
        )
        .unwrap();
    assert!(nonfinite_output[0].is_nan());
}

#[test]
fn erased_reduce_plan_sum_squares_rounds_multiplication_before_accumulation() {
    let dims = [2usize];
    let strides = [1isize];
    let input = [f64::MAX, f64::MIN_POSITIVE / 2.0];
    let plan =
        ErasedReducePlan::compile(KernelDType::F64, ReduceOp::SumSquares, &dims, &strides).unwrap();
    let source = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let mut output = [0.0f64];
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut output, &[], &[], 0).unwrap();

    plan.execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap();

    assert!(output[0].is_infinite());
    assert_eq!((input[1] * input[1]).to_bits(), 0.0f64.to_bits());
}

#[test]
fn erased_reduce_plan_sum_squares_rejects_non_float_dtypes() {
    for dtype in [
        KernelDType::I32,
        KernelDType::I64,
        KernelDType::C32,
        KernelDType::C64,
        KernelDType::Bool,
    ] {
        let err = ErasedReducePlan::compile(dtype, ReduceOp::SumSquares, &[1], &[1]).unwrap_err();
        assert!(matches!(err, StridedError::UnsupportedDType { .. }));
    }
}
