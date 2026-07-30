use num_complex::{Complex32, Complex64};
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

fn strided_storage<T: Copy + Default>(values: &[T]) -> Vec<T> {
    let mut storage = vec![T::default(); values.len().saturating_mul(2).saturating_sub(1)];
    for (slot, &value) in storage.iter_mut().step_by(2).zip(values) {
        *slot = value;
    }
    storage
}

fn assert_dynamic_fast_paths_match_generic<T>(dtype: KernelDType, values: &[T], updates: &[T])
where
    T: Copy + core::fmt::Debug + Default + PartialEq,
{
    let starts_dims = [1usize];
    let starts_strides = [1isize];
    let starts = [99i64];
    let starts_ref = ErasedRawStridedRef::new(
        KernelDType::I64,
        as_bytes(&starts),
        &starts_dims,
        &starts_strides,
        0,
    )
    .unwrap();

    let slice_dims = [4usize];
    let contiguous_strides = [1isize];
    let strided_strides = [2isize];
    let operand_dims = [values.len()];
    let contiguous_operand = values.to_vec();
    let strided_operand = strided_storage(values);
    let mut contiguous_slice = vec![T::default(); slice_dims[0]];
    let mut strided_slice = strided_storage(&contiguous_slice);
    let contiguous_slice_plan = ErasedDynamicSlicePlan::compile(
        dtype,
        KernelDType::I64,
        &operand_dims,
        &contiguous_strides,
        &starts_dims,
        &starts_strides,
        &slice_dims,
        &contiguous_strides,
        &slice_dims,
    )
    .unwrap();
    let strided_slice_plan = ErasedDynamicSlicePlan::compile(
        dtype,
        KernelDType::I64,
        &operand_dims,
        &strided_strides,
        &starts_dims,
        &starts_strides,
        &slice_dims,
        &strided_strides,
        &slice_dims,
    )
    .unwrap();
    let contiguous_operand_ref = ErasedRawStridedRef::new(
        dtype,
        as_bytes(&contiguous_operand),
        &operand_dims,
        &contiguous_strides,
        0,
    )
    .unwrap();
    let strided_operand_ref = ErasedRawStridedRef::new(
        dtype,
        as_bytes(&strided_operand),
        &operand_dims,
        &strided_strides,
        0,
    )
    .unwrap();
    let mut contiguous_slice_ref = ErasedRawStridedMut::new(
        dtype,
        as_bytes_mut(&mut contiguous_slice),
        &slice_dims,
        &contiguous_strides,
        0,
    )
    .unwrap();
    let mut strided_slice_ref = ErasedRawStridedMut::new(
        dtype,
        as_bytes_mut(&mut strided_slice),
        &slice_dims,
        &strided_strides,
        0,
    )
    .unwrap();
    contiguous_slice_plan
        .execute(
            &ExecContext::max_threads(4).unwrap(),
            &mut contiguous_slice_ref,
            &contiguous_operand_ref,
            &starts_ref,
        )
        .unwrap();
    strided_slice_plan
        .execute(
            &ExecContext::max_threads(4).unwrap(),
            &mut strided_slice_ref,
            &strided_operand_ref,
            &starts_ref,
        )
        .unwrap();
    let strided_slice_values = strided_slice.iter().step_by(2).copied().collect::<Vec<_>>();
    assert_eq!(contiguous_slice, strided_slice_values);

    let update_dims = [updates.len()];
    let contiguous_update = updates.to_vec();
    let strided_update = strided_storage(updates);
    let mut contiguous_dest = vec![T::default(); values.len()];
    let mut strided_dest = strided_storage(&contiguous_dest);
    let contiguous_update_plan = ErasedDynamicUpdateSlicePlan::compile(
        dtype,
        KernelDType::I64,
        &operand_dims,
        &contiguous_strides,
        &starts_dims,
        &starts_strides,
        &update_dims,
        &contiguous_strides,
        &operand_dims,
        &contiguous_strides,
    )
    .unwrap();
    let strided_update_plan = ErasedDynamicUpdateSlicePlan::compile(
        dtype,
        KernelDType::I64,
        &operand_dims,
        &strided_strides,
        &starts_dims,
        &starts_strides,
        &update_dims,
        &strided_strides,
        &operand_dims,
        &strided_strides,
    )
    .unwrap();
    let contiguous_update_ref = ErasedRawStridedRef::new(
        dtype,
        as_bytes(&contiguous_update),
        &update_dims,
        &contiguous_strides,
        0,
    )
    .unwrap();
    let strided_update_ref = ErasedRawStridedRef::new(
        dtype,
        as_bytes(&strided_update),
        &update_dims,
        &strided_strides,
        0,
    )
    .unwrap();
    let mut contiguous_dest_ref = ErasedRawStridedMut::new(
        dtype,
        as_bytes_mut(&mut contiguous_dest),
        &operand_dims,
        &contiguous_strides,
        0,
    )
    .unwrap();
    let mut strided_dest_ref = ErasedRawStridedMut::new(
        dtype,
        as_bytes_mut(&mut strided_dest),
        &operand_dims,
        &strided_strides,
        0,
    )
    .unwrap();
    contiguous_update_plan
        .execute(
            &ExecContext::max_threads(4).unwrap(),
            &mut contiguous_dest_ref,
            &contiguous_operand_ref,
            &contiguous_update_ref,
            &starts_ref,
        )
        .unwrap();
    strided_update_plan
        .execute(
            &ExecContext::max_threads(4).unwrap(),
            &mut strided_dest_ref,
            &strided_operand_ref,
            &strided_update_ref,
            &starts_ref,
        )
        .unwrap();
    let strided_dest_values = strided_dest.iter().step_by(2).copied().collect::<Vec<_>>();
    assert_eq!(contiguous_dest, strided_dest_values);
}

#[test]
fn erased_dynamic_fast_paths_match_generic_replay_for_every_value_dtype() {
    assert_dynamic_fast_paths_match_generic(
        KernelDType::F32,
        &[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        &[10.0f32, 11.0, 12.0],
    );
    assert_dynamic_fast_paths_match_generic(
        KernelDType::F64,
        &[0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        &[10.0f64, 11.0, 12.0],
    );
    assert_dynamic_fast_paths_match_generic(
        KernelDType::I32,
        &[0i32, 1, 2, 3, 4, 5, 6, 7],
        &[10i32, 11, 12],
    );
    assert_dynamic_fast_paths_match_generic(
        KernelDType::I64,
        &[0i64, 1, 2, 3, 4, 5, 6, 7],
        &[10i64, 11, 12],
    );
    assert_dynamic_fast_paths_match_generic(
        KernelDType::Bool,
        &[false, true, false, true, true, false, true, false],
        &[true, true, false],
    );
    assert_dynamic_fast_paths_match_generic(
        KernelDType::C32,
        &[
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, -1.0),
            Complex32::new(2.0, -2.0),
            Complex32::new(3.0, -3.0),
            Complex32::new(4.0, -4.0),
            Complex32::new(5.0, -5.0),
            Complex32::new(6.0, -6.0),
            Complex32::new(7.0, -7.0),
        ],
        &[
            Complex32::new(10.0, 1.0),
            Complex32::new(11.0, 2.0),
            Complex32::new(12.0, 3.0),
        ],
    );
    assert_dynamic_fast_paths_match_generic(
        KernelDType::C64,
        &[
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(3.0, -3.0),
            Complex64::new(4.0, -4.0),
            Complex64::new(5.0, -5.0),
            Complex64::new(6.0, -6.0),
            Complex64::new(7.0, -7.0),
        ],
        &[
            Complex64::new(10.0, 1.0),
            Complex64::new(11.0, 2.0),
            Complex64::new(12.0, 3.0),
        ],
    );
}

#[test]
fn erased_dynamic_slice_preserves_empty_and_negative_stride_generic_cases() {
    let empty_plan = ErasedDynamicSlicePlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &[4],
        &[1],
        &[1],
        &[1],
        &[0],
        &[1],
        &[0],
    )
    .unwrap();
    let operand = [0.0f64, 1.0, 2.0, 3.0];
    let starts = [0i64];
    let mut empty = Vec::<f64>::new();
    let operand_ref =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&operand), &[4], &[1], 0).unwrap();
    let starts_ref =
        ErasedRawStridedRef::new(KernelDType::I64, as_bytes(&starts), &[1], &[1], 0).unwrap();
    let mut empty_ref =
        ErasedRawStridedMut::new(KernelDType::F64, as_bytes_mut(&mut empty), &[0], &[1], 0)
            .unwrap();
    empty_plan
        .execute(
            &ExecContext::serial(),
            &mut empty_ref,
            &operand_ref,
            &starts_ref,
        )
        .unwrap();

    let reverse_plan = ErasedDynamicSlicePlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &[4],
        &[-1],
        &[1],
        &[1],
        &[4],
        &[1],
        &[4],
    )
    .unwrap();
    let reverse_operand =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&operand), &[4], &[-1], 3).unwrap();
    let mut reversed = [0.0f64; 4];
    let mut reversed_ref =
        ErasedRawStridedMut::new(KernelDType::F64, as_bytes_mut(&mut reversed), &[4], &[1], 0)
            .unwrap();
    reverse_plan
        .execute(
            &ExecContext::serial(),
            &mut reversed_ref,
            &reverse_operand,
            &starts_ref,
        )
        .unwrap();
    assert_eq!(reversed, [3.0, 2.0, 1.0, 0.0]);
}
