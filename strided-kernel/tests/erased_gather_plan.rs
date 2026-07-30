use num_complex::{Complex32, Complex64};
use strided_kernel::{
    ErasedGatherPlan, ErasedRawStridedMut, ErasedRawStridedRef, ExecContext, GatherSpec,
    KernelDType, StridedError,
};

#[test]
fn erased_gather_plan_executes_f64_column_gather_with_i64_indices() {
    let operand_dims = [3usize, 4];
    let operand_strides = [1isize, 3];
    let index_dims = [2usize, 1];
    let index_strides = [1isize, 2];
    let dest_dims = [2usize, 3];
    let dest_strides = [1isize, 2];
    let operand = [
        0.0f64, 1.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0,
    ];
    let indices = [2i64, 0];
    let mut dest = [0.0f64; 6];
    let spec = GatherSpec {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![1],
        start_index_map: vec![1],
        index_vector_dim: 1,
        slice_sizes: vec![3, 1],
    };

    let plan = ErasedGatherPlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &operand_dims,
        &operand_strides,
        &index_dims,
        &index_strides,
        &dest_dims,
        &dest_strides,
        spec,
    )
    .unwrap();
    let operand_ref =
        ErasedRawStridedRef::from_slice(&operand, &operand_dims, &operand_strides, 0).unwrap();
    let index_ref =
        ErasedRawStridedRef::from_slice(&indices, &index_dims, &index_strides, 0).unwrap();
    let mut dest_ref =
        ErasedRawStridedMut::from_slice_mut(&mut dest, &dest_dims, &dest_strides, 0).unwrap();

    plan.execute(
        &ExecContext::serial(),
        &mut dest_ref,
        &operand_ref,
        &index_ref,
    )
    .unwrap();

    assert_eq!(dest, [20.0, 0.0, 21.0, 1.0, 22.0, 2.0]);
}

#[test]
fn erased_gather_plan_executes_f32_windows_with_i32_indices_and_clamping() {
    let operand_dims = [4usize];
    let operand_strides = [1isize];
    let index_dims = [3usize];
    let index_strides = [1isize];
    let dest_dims = [3usize, 2];
    let dest_strides = [1isize, 3];
    let operand = [10.0f32, 11.0, 12.0, 13.0];
    let indices = [-1i32, 2, 99];
    let mut dest = [0.0f32; 6];
    let spec = GatherSpec {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![2],
    };

    let plan = ErasedGatherPlan::compile(
        KernelDType::F32,
        KernelDType::I32,
        &operand_dims,
        &operand_strides,
        &index_dims,
        &index_strides,
        &dest_dims,
        &dest_strides,
        spec,
    )
    .unwrap();
    let operand_ref =
        ErasedRawStridedRef::from_slice(&operand, &operand_dims, &operand_strides, 0).unwrap();
    let index_ref =
        ErasedRawStridedRef::from_slice(&indices, &index_dims, &index_strides, 0).unwrap();
    let mut dest_ref =
        ErasedRawStridedMut::from_slice_mut(&mut dest, &dest_dims, &dest_strides, 0).unwrap();

    plan.execute(
        &ExecContext::max_threads(1).unwrap(),
        &mut dest_ref,
        &operand_ref,
        &index_ref,
    )
    .unwrap();

    assert_eq!(dest, [10.0, 12.0, 12.0, 11.0, 13.0, 13.0]);
}

fn assert_supported_take<T>(dtype: KernelDType, input: &[T])
where
    T: Copy + core::fmt::Debug + Default + PartialEq + strided_kernel::KernelStorageElement,
{
    let operand_dims = [input.len()];
    let operand_strides = [1isize];
    let index_dims = [2usize];
    let index_strides = [1isize];
    let dest_dims = [2usize];
    let dest_strides = [1isize];
    let indices = [1i64, 0];
    let mut output = vec![T::default(); 2];
    let spec = GatherSpec {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    };

    let plan = ErasedGatherPlan::compile(
        dtype,
        KernelDType::I64,
        &operand_dims,
        &operand_strides,
        &index_dims,
        &index_strides,
        &dest_dims,
        &dest_strides,
        spec,
    )
    .unwrap();
    let operand_ref =
        ErasedRawStridedRef::from_slice(input, &operand_dims, &operand_strides, 0).unwrap();
    let index_ref =
        ErasedRawStridedRef::from_slice(&indices, &index_dims, &index_strides, 0).unwrap();
    let mut dest_ref =
        ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &dest_strides, 0).unwrap();

    plan.execute(
        &ExecContext::ambient(),
        &mut dest_ref,
        &operand_ref,
        &index_ref,
    )
    .unwrap();

    assert_eq!(output, vec![input[1], input[0]]);
}

#[test]
fn erased_gather_plan_executes_supported_value_dtype_set() {
    assert_supported_take(KernelDType::F64, &[1.0f64, 2.0]);
    assert_supported_take(KernelDType::I32, &[1i32, 2]);
    assert_supported_take(KernelDType::I64, &[1i64, 2]);
    assert_supported_take(KernelDType::Bool, &[false, true]);
    assert_supported_take(
        KernelDType::C32,
        &[Complex32::new(1.0, -1.0), Complex32::new(2.0, 3.0)],
    );
    assert_supported_take(
        KernelDType::C64,
        &[Complex64::new(1.0, -1.0), Complex64::new(2.0, 3.0)],
    );
}

#[test]
fn erased_gather_plan_rejects_dtype_and_layout_mismatch_before_writing() {
    let operand_dims = [2usize];
    let strides = [1isize];
    let index_dims = [1usize];
    let dest_dims = [1usize];
    let operand = [1.0f64, 2.0];
    let indices = [0i64];
    let mut output = [9.0f32];
    let spec = GatherSpec {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    };

    let plan = ErasedGatherPlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &operand_dims,
        &strides,
        &index_dims,
        &strides,
        &dest_dims,
        &strides,
        spec,
    )
    .unwrap();
    let operand_ref =
        ErasedRawStridedRef::from_slice(&operand, &operand_dims, &strides, 0).unwrap();
    let index_ref = ErasedRawStridedRef::from_slice(&indices, &index_dims, &strides, 0).unwrap();
    let mut dest_ref =
        ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &strides, 0).unwrap();

    let err = plan
        .execute(
            &ExecContext::serial(),
            &mut dest_ref,
            &operand_ref,
            &index_ref,
        )
        .unwrap_err();

    assert!(matches!(err, StridedError::DTypeMismatch { .. }));
    assert_eq!(output, [9.0]);

    let bad_index_ref = ErasedRawStridedRef::from_slice(&indices, &[1], &[0], 0).unwrap();
    let mut output = [9.0f64];
    let mut dest_ref =
        ErasedRawStridedMut::from_slice_mut(&mut output, &dest_dims, &strides, 0).unwrap();
    let err = plan
        .execute(
            &ExecContext::serial(),
            &mut dest_ref,
            &operand_ref,
            &bad_index_ref,
        )
        .unwrap_err();

    assert!(matches!(err, StridedError::PlanLayoutMismatch));
    assert_eq!(output, [9.0]);
}

#[test]
fn erased_gather_plan_rejects_invalid_compile_specs() {
    let operand_dims = [2usize, 3];
    let operand_strides = [1isize, 2];
    let index_dims = [2usize];
    let index_strides = [1isize];
    let dest_dims = [2usize];
    let dest_strides = [1isize];

    let unsupported_index_dtype = ErasedGatherPlan::compile(
        KernelDType::F64,
        KernelDType::F64,
        &operand_dims,
        &operand_strides,
        &index_dims,
        &index_strides,
        &dest_dims,
        &dest_strides,
        GatherSpec {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0, 1],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1, 1],
        },
    )
    .unwrap_err();
    assert!(matches!(
        unsupported_index_dtype,
        StridedError::UnsupportedDType { dtype: "f64" }
    ));

    let bad_dest_shape = ErasedGatherPlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &operand_dims,
        &operand_strides,
        &index_dims,
        &index_strides,
        &[3],
        &[1],
        GatherSpec {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0, 1],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1, 1],
        },
    )
    .unwrap_err();
    assert!(matches!(bad_dest_shape, StridedError::ShapeMismatch(_, _)));

    let duplicate_axis = ErasedGatherPlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &operand_dims,
        &operand_strides,
        &index_dims,
        &index_strides,
        &dest_dims,
        &dest_strides,
        GatherSpec {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0, 0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1, 1],
        },
    )
    .unwrap_err();
    assert!(matches!(duplicate_axis, StridedError::InvalidAxis { .. }));

    let oversized_window = ErasedGatherPlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &operand_dims,
        &operand_strides,
        &index_dims,
        &index_strides,
        &dest_dims,
        &dest_strides,
        GatherSpec {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0, 1],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1, 4],
        },
    )
    .unwrap_err();
    assert!(matches!(oversized_window, StridedError::InvalidAxis { .. }));
}
