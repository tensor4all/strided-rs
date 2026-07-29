use num_complex::{Complex32, Complex64};
use strided_kernel::{
    ErasedConcatenatePlan, ErasedPadPlan, ErasedRawStridedMut, ErasedRawStridedRef,
    ErasedReversePlan, ErasedSlicePlan, ExecContext, KernelDType, StridedError,
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
fn erased_slice_plan_executes_strided_static_slice() {
    let operand_dims = [4usize, 5];
    let operand_strides = [1isize, 4];
    let dest_dims = [2usize, 2];
    let dest_strides = [1isize, 2];
    let starts = [1usize, 1];
    let limits = [4usize, 5];
    let slice_strides = [2usize, 2];
    let operand = [
        0.0f64, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0, 30.0, 31.0, 32.0,
        33.0, 40.0, 41.0, 42.0, 43.0,
    ];
    let mut dest = [0.0f64; 4];

    let plan = ErasedSlicePlan::compile(
        KernelDType::F64,
        &operand_dims,
        &operand_strides,
        &dest_dims,
        &dest_strides,
        &starts,
        &limits,
        &slice_strides,
    )
    .unwrap();
    let operand_ref = ErasedRawStridedRef::new(
        KernelDType::F64,
        as_bytes(&operand),
        &operand_dims,
        &operand_strides,
        0,
    )
    .unwrap();
    let mut dest_ref = ErasedRawStridedMut::new(
        KernelDType::F64,
        as_bytes_mut(&mut dest),
        &dest_dims,
        &dest_strides,
        0,
    )
    .unwrap();

    plan.execute(&ExecContext::serial(), &mut dest_ref, &operand_ref)
        .unwrap();

    assert_eq!(dest, [11.0, 13.0, 31.0, 33.0]);
}

#[test]
fn erased_reverse_plan_executes_multi_axis_reverse() {
    let dims = [2usize, 3, 2];
    let strides = [1isize, 2, 6];
    let axes = [0usize, 2];
    let operand = [0i32, 1, 10, 11, 20, 21, 100, 101, 110, 111, 120, 121];
    let mut dest = [0i32; 12];

    let plan =
        ErasedReversePlan::compile(KernelDType::I32, &dims, &strides, &strides, &axes).unwrap();
    let operand_ref =
        ErasedRawStridedRef::new(KernelDType::I32, as_bytes(&operand), &dims, &strides, 0).unwrap();
    let mut dest_ref = ErasedRawStridedMut::new(
        KernelDType::I32,
        as_bytes_mut(&mut dest),
        &dims,
        &strides,
        0,
    )
    .unwrap();

    plan.execute(
        &ExecContext::max_threads(1).unwrap(),
        &mut dest_ref,
        &operand_ref,
    )
    .unwrap();

    assert_eq!(dest, [101, 100, 111, 110, 121, 120, 1, 0, 11, 10, 21, 20]);
}

#[test]
fn erased_pad_plan_fills_output_and_copies_cropped_interior_padded_input() {
    let operand_dims = [3usize];
    let operand_strides = [1isize];
    let dest_dims = [6usize];
    let dest_strides = [1isize];
    let edge_low = [-1i64];
    let edge_high = [2i64];
    let interior = [1i64];
    let operand = [10i32, 11, 12];
    let fill = [-1i32];
    let mut dest = [0i32; 6];

    let plan = ErasedPadPlan::compile(
        KernelDType::I32,
        &operand_dims,
        &operand_strides,
        &dest_dims,
        &dest_strides,
        &edge_low,
        &edge_high,
        &interior,
    )
    .unwrap();
    let operand_ref = ErasedRawStridedRef::new(
        KernelDType::I32,
        as_bytes(&operand),
        &operand_dims,
        &operand_strides,
        0,
    )
    .unwrap();
    let mut dest_ref = ErasedRawStridedMut::new(
        KernelDType::I32,
        as_bytes_mut(&mut dest),
        &dest_dims,
        &dest_strides,
        0,
    )
    .unwrap();

    plan.execute(
        &ExecContext::serial(),
        &mut dest_ref,
        &operand_ref,
        as_bytes(&fill),
    )
    .unwrap();

    assert_eq!(dest, [-1, 11, -1, 12, -1, -1]);
}

fn assert_dense_edge_pad<T>(dtype: KernelDType, operand: &[T], fill: T, expected: &[T])
where
    T: Copy + core::fmt::Debug + PartialEq,
{
    let operand_dims = [operand.len()];
    let operand_strides = [1isize];
    let dest_dims = [expected.len()];
    let dest_strides = [1isize];
    let edge_low = [2i64];
    let edge_high = [1i64];
    let interior = [0i64];
    let mut dest = vec![fill; expected.len()];
    let fill = [fill];

    let plan = ErasedPadPlan::compile(
        dtype,
        &operand_dims,
        &operand_strides,
        &dest_dims,
        &dest_strides,
        &edge_low,
        &edge_high,
        &interior,
    )
    .unwrap();
    let operand_ref =
        ErasedRawStridedRef::new(dtype, as_bytes(operand), &operand_dims, &operand_strides, 0)
            .unwrap();
    let mut dest_ref =
        ErasedRawStridedMut::new(dtype, as_bytes_mut(&mut dest), &dest_dims, &dest_strides, 0)
            .unwrap();

    plan.execute(
        &ExecContext::max_threads(4).unwrap(),
        &mut dest_ref,
        &operand_ref,
        as_bytes(&fill),
    )
    .unwrap();
    assert_eq!(dest, expected);
}

#[test]
fn erased_pad_contiguous_run_matches_expected_for_every_kernel_dtype() {
    assert_dense_edge_pad(
        KernelDType::F32,
        &[1.0f32, 2.0],
        -1.0,
        &[-1.0, -1.0, 1.0, 2.0, -1.0],
    );
    assert_dense_edge_pad(
        KernelDType::F64,
        &[1.0f64, 2.0],
        -1.0,
        &[-1.0, -1.0, 1.0, 2.0, -1.0],
    );
    assert_dense_edge_pad(KernelDType::I32, &[1i32, 2], -1, &[-1, -1, 1, 2, -1]);
    assert_dense_edge_pad(KernelDType::I64, &[1i64, 2], -1, &[-1, -1, 1, 2, -1]);
    assert_dense_edge_pad(
        KernelDType::Bool,
        &[true, false],
        false,
        &[false, false, true, false, false],
    );
    assert_dense_edge_pad(
        KernelDType::C32,
        &[Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)],
        Complex32::new(-1.0, 0.0),
        &[
            Complex32::new(-1.0, 0.0),
            Complex32::new(-1.0, 0.0),
            Complex32::new(1.0, 2.0),
            Complex32::new(3.0, 4.0),
            Complex32::new(-1.0, 0.0),
        ],
    );
    assert_dense_edge_pad(
        KernelDType::C64,
        &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
        Complex64::new(-1.0, 0.0),
        &[
            Complex64::new(-1.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(-1.0, 0.0),
        ],
    );
}

#[test]
fn erased_pad_handles_empty_rank_zero_and_noncontiguous_fallbacks() {
    let empty_dims = [0usize];
    let empty_strides = [1isize];
    let padded_dims = [3usize];
    let padded_strides = [1isize];
    let edge_low = [1i64];
    let edge_high = [2i64];
    let interior = [0i64];
    let empty: [i32; 0] = [];
    let fill = [-7i32];
    let mut padded = [0i32; 3];
    let plan = ErasedPadPlan::compile(
        KernelDType::I32,
        &empty_dims,
        &empty_strides,
        &padded_dims,
        &padded_strides,
        &edge_low,
        &edge_high,
        &interior,
    )
    .unwrap();
    let empty_ref = ErasedRawStridedRef::new(
        KernelDType::I32,
        as_bytes(&empty),
        &empty_dims,
        &empty_strides,
        0,
    )
    .unwrap();
    let mut padded_ref = ErasedRawStridedMut::new(
        KernelDType::I32,
        as_bytes_mut(&mut padded),
        &padded_dims,
        &padded_strides,
        0,
    )
    .unwrap();
    plan.execute(
        &ExecContext::serial(),
        &mut padded_ref,
        &empty_ref,
        as_bytes(&fill),
    )
    .unwrap();
    assert_eq!(padded, [-7; 3]);

    let scalar = [42i32];
    let mut scalar_dest = [0i32];
    let scalar_plan =
        ErasedPadPlan::compile(KernelDType::I32, &[], &[], &[], &[], &[], &[], &[]).unwrap();
    let scalar_ref =
        ErasedRawStridedRef::new(KernelDType::I32, as_bytes(&scalar), &[], &[], 0).unwrap();
    let mut scalar_dest_ref = ErasedRawStridedMut::new(
        KernelDType::I32,
        as_bytes_mut(&mut scalar_dest),
        &[],
        &[],
        0,
    )
    .unwrap();
    scalar_plan
        .execute(
            &ExecContext::serial(),
            &mut scalar_dest_ref,
            &scalar_ref,
            as_bytes(&fill),
        )
        .unwrap();
    assert_eq!(scalar_dest, [42]);

    let operand = [10i32, 20];
    let mut strided_dest = [-9i32; 4];
    let dims = [2usize];
    let operand_strides = [1isize];
    let dest_strides = [2isize];
    let edge = [0i64];
    let fallback = ErasedPadPlan::compile(
        KernelDType::I32,
        &dims,
        &operand_strides,
        &dims,
        &dest_strides,
        &edge,
        &edge,
        &interior,
    )
    .unwrap();
    let operand_ref = ErasedRawStridedRef::new(
        KernelDType::I32,
        as_bytes(&operand),
        &dims,
        &operand_strides,
        0,
    )
    .unwrap();
    let mut dest_ref = ErasedRawStridedMut::new(
        KernelDType::I32,
        as_bytes_mut(&mut strided_dest),
        &dims,
        &dest_strides,
        0,
    )
    .unwrap();
    fallback
        .execute(
            &ExecContext::serial(),
            &mut dest_ref,
            &operand_ref,
            as_bytes(&fill),
        )
        .unwrap();
    assert_eq!(strided_dest, [10, -9, 20, -9]);
}

#[test]
fn erased_pad_contiguous_runs_honor_offsets_and_outer_axis_cropping() {
    let dims = [3usize];
    let strides = [1isize];
    let padded_dims = [5usize];
    let edge = [1i64];
    let interior = [0i64];
    let operand = [99i32, 10, 20, 30, 99];
    let mut dest = [99i32, 0, 0, 0, 0, 0, 99];
    let fill = [-7i32];
    let plan = ErasedPadPlan::compile(
        KernelDType::I32,
        &dims,
        &strides,
        &padded_dims,
        &strides,
        &edge,
        &edge,
        &interior,
    )
    .unwrap();
    let operand_ref =
        ErasedRawStridedRef::new(KernelDType::I32, as_bytes(&operand), &dims, &strides, 1).unwrap();
    let mut dest_ref = ErasedRawStridedMut::new(
        KernelDType::I32,
        as_bytes_mut(&mut dest),
        &padded_dims,
        &strides,
        1,
    )
    .unwrap();
    plan.execute(
        &ExecContext::serial(),
        &mut dest_ref,
        &operand_ref,
        as_bytes(&fill),
    )
    .unwrap();
    assert_eq!(dest, [99, -7, 10, 20, 30, -7, 99]);

    let operand_dims = [2usize, 3];
    let operand_strides = [1isize, 2];
    let dest_dims = [4usize, 2];
    let dest_strides = [1isize, 4];
    let edge_low = [1i64, -1];
    let edge_high = [1i64, 0];
    let interior = [0i64, 0];
    let operand = [1i32, 2, 3, 4, 5, 6];
    let mut dest = [0i32; 8];
    let plan = ErasedPadPlan::compile(
        KernelDType::I32,
        &operand_dims,
        &operand_strides,
        &dest_dims,
        &dest_strides,
        &edge_low,
        &edge_high,
        &interior,
    )
    .unwrap();
    let operand_ref = ErasedRawStridedRef::new(
        KernelDType::I32,
        as_bytes(&operand),
        &operand_dims,
        &operand_strides,
        0,
    )
    .unwrap();
    let mut dest_ref = ErasedRawStridedMut::new(
        KernelDType::I32,
        as_bytes_mut(&mut dest),
        &dest_dims,
        &dest_strides,
        0,
    )
    .unwrap();
    plan.execute(
        &ExecContext::serial(),
        &mut dest_ref,
        &operand_ref,
        as_bytes(&fill),
    )
    .unwrap();
    assert_eq!(dest, [-7, 3, 4, -7, -7, 5, 6, -7]);
}

#[test]
fn erased_concatenate_plan_executes_three_input_axis_concatenate() {
    let input0_dims = [2usize, 1];
    let input1_dims = [2usize, 2];
    let input2_dims = [2usize, 1];
    let input_strides = [1isize, 2];
    let dest_dims = [2usize, 4];
    let dest_strides = [1isize, 2];
    let axis = 1usize;
    let input0 = [0i64, 1];
    let input1 = [10i64, 11, 20, 21];
    let input2 = [30i64, 31];
    let mut dest = [0i64; 8];
    let input_dims = [&input0_dims[..], &input1_dims[..], &input2_dims[..]];
    let input_strides_list = [&input_strides[..], &input_strides[..], &input_strides[..]];

    let plan = ErasedConcatenatePlan::compile(
        KernelDType::I64,
        &input_dims,
        &input_strides_list,
        &dest_dims,
        &dest_strides,
        axis,
    )
    .unwrap();
    let input0_ref = ErasedRawStridedRef::new(
        KernelDType::I64,
        as_bytes(&input0),
        &input0_dims,
        &input_strides,
        0,
    )
    .unwrap();
    let input1_ref = ErasedRawStridedRef::new(
        KernelDType::I64,
        as_bytes(&input1),
        &input1_dims,
        &input_strides,
        0,
    )
    .unwrap();
    let input2_ref = ErasedRawStridedRef::new(
        KernelDType::I64,
        as_bytes(&input2),
        &input2_dims,
        &input_strides,
        0,
    )
    .unwrap();
    let inputs = [input0_ref, input1_ref, input2_ref];
    let mut dest_ref = ErasedRawStridedMut::new(
        KernelDType::I64,
        as_bytes_mut(&mut dest),
        &dest_dims,
        &dest_strides,
        0,
    )
    .unwrap();

    plan.execute(&ExecContext::serial(), &mut dest_ref, &inputs)
        .unwrap();

    assert_eq!(dest, [0, 1, 10, 11, 20, 21, 30, 31]);
}

#[test]
fn erased_static_indexing_plans_expose_dtype_and_inner_plan() {
    let dims = [2usize];
    let strides = [1isize];
    let starts = [0usize];
    let limits = [2usize];
    let slice_strides = [1usize];
    let edge = [0i64];
    let interior = [0i64];
    let input_dims = [&dims[..], &dims[..]];
    let input_strides = [&strides[..], &strides[..]];
    let dest_dims = [4usize];

    let slice = ErasedSlicePlan::compile(
        KernelDType::F32,
        &dims,
        &strides,
        &dims,
        &strides,
        &starts,
        &limits,
        &slice_strides,
    )
    .unwrap();
    let reverse =
        ErasedReversePlan::compile(KernelDType::F64, &dims, &strides, &strides, &[0]).unwrap();
    let pad = ErasedPadPlan::compile(
        KernelDType::Bool,
        &dims,
        &strides,
        &dims,
        &strides,
        &edge,
        &edge,
        &interior,
    )
    .unwrap();
    let concat = ErasedConcatenatePlan::compile(
        KernelDType::C32,
        &input_dims,
        &input_strides,
        &dest_dims,
        &strides,
        0,
    )
    .unwrap();

    assert_eq!(slice.dtype(), KernelDType::F32);
    assert!(format!("{:?}", slice.plan()).contains("SlicePlan"));
    assert_eq!(reverse.dtype(), KernelDType::F64);
    assert!(format!("{:?}", reverse.plan()).contains("ReversePlan"));
    assert_eq!(pad.dtype(), KernelDType::Bool);
    assert!(format!("{:?}", pad.plan()).contains("PadPlan"));
    assert_eq!(concat.dtype(), KernelDType::C32);
    assert!(format!("{:?}", concat.plan()).contains("ConcatenatePlan"));
}

#[test]
fn erased_static_indexing_plans_reject_mismatches_before_writing() {
    let dims = [2usize];
    let strides = [1isize];
    let starts = [0usize];
    let limits = [2usize];
    let slice_strides = [1usize];
    let operand = [1.0f64, 2.0];
    let mut f32_dest = [9.0f32, 9.0];

    let slice = ErasedSlicePlan::compile(
        KernelDType::F64,
        &dims,
        &strides,
        &dims,
        &strides,
        &starts,
        &limits,
        &slice_strides,
    )
    .unwrap();
    let operand_ref =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&operand), &dims, &strides, 0).unwrap();
    let mut dest_ref = ErasedRawStridedMut::new(
        KernelDType::F32,
        as_bytes_mut(&mut f32_dest),
        &dims,
        &strides,
        0,
    )
    .unwrap();

    let err = slice
        .execute(&ExecContext::serial(), &mut dest_ref, &operand_ref)
        .unwrap_err();
    assert!(matches!(err, StridedError::DTypeMismatch { .. }));
    assert_eq!(f32_dest, [9.0, 9.0]);

    let edge = [0i64];
    let interior = [0i64];
    let pad = ErasedPadPlan::compile(
        KernelDType::Bool,
        &dims,
        &strides,
        &dims,
        &strides,
        &edge,
        &edge,
        &interior,
    )
    .unwrap();
    let bool_operand = [true, false];
    let mut bool_dest = [false, false];
    let operand_ref = ErasedRawStridedRef::new(
        KernelDType::Bool,
        as_bytes(&bool_operand),
        &dims,
        &strides,
        0,
    )
    .unwrap();
    let mut dest_ref = ErasedRawStridedMut::new(
        KernelDType::Bool,
        as_bytes_mut(&mut bool_dest),
        &dims,
        &strides,
        0,
    )
    .unwrap();

    let err = pad
        .execute(&ExecContext::serial(), &mut dest_ref, &operand_ref, &[2u8])
        .unwrap_err();
    assert!(matches!(err, StridedError::InvalidBoolByte { value: 2 }));
    assert_eq!(bool_dest, [false, false]);
}
