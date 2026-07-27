use strided_kernel::{
    ErasedDynamicSlicePlan, ErasedDynamicUpdateSlicePlan, ErasedRawStridedMut, ErasedRawStridedRef,
    ErasedScatterPlan, ExecContext, KernelDType, ScatterSpec, StridedError,
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
fn erased_dynamic_slice_plan_executes_fixed_window_with_clamped_starts() {
    let operand_dims = [4usize, 3];
    let operand_strides = [1isize, 4];
    let starts_dims = [2usize];
    let starts_strides = [1isize];
    let dest_dims = [2usize, 2];
    let dest_strides = [1isize, 2];
    let slice_sizes = [2usize, 2];
    let operand = [
        0.0f64, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0,
    ];
    let starts = [-1i64, 2];
    let mut dest = [0.0f64; 4];

    let plan = ErasedDynamicSlicePlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &operand_dims,
        &operand_strides,
        &starts_dims,
        &starts_strides,
        &dest_dims,
        &dest_strides,
        &slice_sizes,
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
    let starts_ref = ErasedRawStridedRef::new(
        KernelDType::I64,
        as_bytes(&starts),
        &starts_dims,
        &starts_strides,
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

    plan.execute(
        &ExecContext::serial(),
        &mut dest_ref,
        &operand_ref,
        &starts_ref,
    )
    .unwrap();

    assert_eq!(dest, [10.0, 11.0, 20.0, 21.0]);
}

#[test]
fn erased_dynamic_update_slice_plan_overwrites_clamped_window_after_copy() {
    let operand_dims = [5usize];
    let operand_strides = [1isize];
    let starts_dims = [1usize];
    let starts_strides = [1isize];
    let update_dims = [2usize];
    let update_strides = [1isize];
    let dest_dims = [5usize];
    let dest_strides = [1isize];
    let operand = [0i32, 1, 2, 3, 4];
    let starts = [99i32];
    let update = [7i32, 8];
    let mut dest = [0i32; 5];

    let plan = ErasedDynamicUpdateSlicePlan::compile(
        KernelDType::I32,
        KernelDType::I32,
        &operand_dims,
        &operand_strides,
        &starts_dims,
        &starts_strides,
        &update_dims,
        &update_strides,
        &dest_dims,
        &dest_strides,
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
    let starts_ref = ErasedRawStridedRef::new(
        KernelDType::I32,
        as_bytes(&starts),
        &starts_dims,
        &starts_strides,
        0,
    )
    .unwrap();
    let update_ref = ErasedRawStridedRef::new(
        KernelDType::I32,
        as_bytes(&update),
        &update_dims,
        &update_strides,
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
        &update_ref,
        &starts_ref,
    )
    .unwrap();

    assert_eq!(dest, [0, 1, 2, 7, 8]);
}

#[test]
fn erased_scatter_plan_adds_overlapping_updates_in_col_major_order() {
    let operand_dims = [4usize];
    let operand_strides = [1isize];
    let index_dims = [3usize, 1];
    let index_strides = [1isize, 3];
    let update_dims = [3usize];
    let update_strides = [1isize];
    let dest_dims = [4usize];
    let dest_strides = [1isize];
    let operand = [1.0f64, 2.0, 3.0, 4.0];
    let indices = [1i64, 1, -1];
    let updates = [10.0f64, 20.0, 30.0];
    let mut dest = [0.0f64; 4];
    let spec = ScatterSpec {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };

    let plan = ErasedScatterPlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &operand_dims,
        &operand_strides,
        &index_dims,
        &index_strides,
        &update_dims,
        &update_strides,
        &dest_dims,
        &dest_strides,
        spec,
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
    let index_ref = ErasedRawStridedRef::new(
        KernelDType::I64,
        as_bytes(&indices),
        &index_dims,
        &index_strides,
        0,
    )
    .unwrap();
    let update_ref = ErasedRawStridedRef::new(
        KernelDType::F64,
        as_bytes(&updates),
        &update_dims,
        &update_strides,
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

    plan.execute(
        &ExecContext::max_threads(2).unwrap(),
        &mut dest_ref,
        &operand_ref,
        &index_ref,
        &update_ref,
    )
    .unwrap();

    assert_eq!(dest, [31.0, 32.0, 3.0, 4.0]);
}

#[test]
fn erased_indexed_write_plans_reject_invalid_contracts() {
    let dynamic_slice_err = ErasedDynamicSlicePlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &[3],
        &[1],
        &[1],
        &[1],
        &[4],
        &[1],
        &[4],
    )
    .unwrap_err();
    assert!(matches!(
        dynamic_slice_err,
        StridedError::InvalidAxis { .. }
    ));

    let dynamic_update_err = ErasedDynamicUpdateSlicePlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &[3],
        &[1],
        &[1],
        &[1],
        &[2],
        &[1],
        &[2],
        &[1],
    )
    .unwrap_err();
    assert!(matches!(
        dynamic_update_err,
        StridedError::ShapeMismatch(_, _)
    ));

    let scatter_bool_err = ErasedScatterPlan::compile(
        KernelDType::Bool,
        KernelDType::I64,
        &[4],
        &[1],
        &[1, 1],
        &[1, 1],
        &[1],
        &[1],
        &[4],
        &[1],
        ScatterSpec {
            update_window_dims: vec![],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        },
    )
    .unwrap_err();
    assert!(matches!(
        scatter_bool_err,
        StridedError::UnsupportedDType { dtype: "bool" }
    ));
}
