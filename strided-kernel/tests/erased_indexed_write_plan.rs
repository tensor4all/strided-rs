use core::mem::MaybeUninit;
use num_complex::{Complex32, Complex64};
use strided_kernel::{
    col_major_strides, ErasedDynamicSlicePlan, ErasedDynamicUpdateSlicePlan, ErasedRawStridedMut,
    ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut, ErasedScatterPlan,
    ExecContext, KernelDType, ScatterSpec, StridedError,
};

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
    let operand_ref =
        ErasedRawStridedRef::from_slice(&operand, &operand_dims, &operand_strides, 0).unwrap();
    let starts_ref =
        ErasedRawStridedRef::from_slice(&starts, &starts_dims, &starts_strides, 0).unwrap();
    let mut dest_ref =
        ErasedRawStridedMut::from_slice_mut(&mut dest, &dest_dims, &dest_strides, 0).unwrap();

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
    let operand_ref =
        ErasedRawStridedRef::from_slice(&operand, &operand_dims, &operand_strides, 0).unwrap();
    let starts_ref =
        ErasedRawStridedRef::from_slice(&starts, &starts_dims, &starts_strides, 0).unwrap();
    let update_ref =
        ErasedRawStridedRef::from_slice(&update, &update_dims, &update_strides, 0).unwrap();
    let mut dest_ref =
        ErasedRawStridedMut::from_slice_mut(&mut dest, &dest_dims, &dest_strides, 0).unwrap();

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
    let operand_ref =
        ErasedRawStridedRef::from_slice(&operand, &operand_dims, &operand_strides, 0).unwrap();
    let index_ref =
        ErasedRawStridedRef::from_slice(&indices, &index_dims, &index_strides, 0).unwrap();
    let update_ref =
        ErasedRawStridedRef::from_slice(&updates, &update_dims, &update_strides, 0).unwrap();
    let mut dest_ref =
        ErasedRawStridedMut::from_slice_mut(&mut dest, &dest_dims, &dest_strides, 0).unwrap();

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

fn strided_storage<T: Copy + Default + strided_kernel::KernelStorageElement>(
    values: &[T],
) -> Vec<T> {
    let mut storage = vec![T::default(); values.len().saturating_mul(2).saturating_sub(1)];
    for (slot, &value) in storage.iter_mut().step_by(2).zip(values) {
        *slot = value;
    }
    storage
}

fn assert_dynamic_fast_paths_match_generic<T>(dtype: KernelDType, values: &[T], updates: &[T])
where
    T: Copy + core::fmt::Debug + Default + PartialEq + strided_kernel::KernelStorageElement,
{
    let starts_dims = [1usize];
    let starts_strides = [1isize];
    let starts = [99i64];
    let starts_ref =
        ErasedRawStridedRef::from_slice(&starts, &starts_dims, &starts_strides, 0).unwrap();

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
    let contiguous_operand_ref =
        ErasedRawStridedRef::from_slice(&contiguous_operand, &operand_dims, &contiguous_strides, 0)
            .unwrap();
    let strided_operand_ref =
        ErasedRawStridedRef::from_slice(&strided_operand, &operand_dims, &strided_strides, 0)
            .unwrap();
    let mut contiguous_slice_ref = ErasedRawStridedMut::from_slice_mut(
        &mut contiguous_slice,
        &slice_dims,
        &contiguous_strides,
        0,
    )
    .unwrap();
    let mut strided_slice_ref =
        ErasedRawStridedMut::from_slice_mut(&mut strided_slice, &slice_dims, &strided_strides, 0)
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
    let contiguous_update_ref =
        ErasedRawStridedRef::from_slice(&contiguous_update, &update_dims, &contiguous_strides, 0)
            .unwrap();
    let strided_update_ref =
        ErasedRawStridedRef::from_slice(&strided_update, &update_dims, &strided_strides, 0)
            .unwrap();
    let mut contiguous_dest_ref = ErasedRawStridedMut::from_slice_mut(
        &mut contiguous_dest,
        &operand_dims,
        &contiguous_strides,
        0,
    )
    .unwrap();
    let mut strided_dest_ref =
        ErasedRawStridedMut::from_slice_mut(&mut strided_dest, &operand_dims, &strided_strides, 0)
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
fn dynamic_indexed_compile_rejects_replay_delta_overflow() {
    let slice_error = ErasedDynamicSlicePlan::compile(
        KernelDType::I32,
        KernelDType::I32,
        &[2],
        &[isize::MIN],
        &[1],
        &[1],
        &[2],
        &[1],
        &[2],
    )
    .unwrap_err();
    assert!(matches!(slice_error, StridedError::OffsetOverflow));

    let update_error = ErasedDynamicUpdateSlicePlan::compile(
        KernelDType::I32,
        KernelDType::I32,
        &[2],
        &[1],
        &[1],
        &[1],
        &[2],
        &[isize::MIN],
        &[2],
        &[1],
    )
    .unwrap_err();
    assert!(matches!(update_error, StridedError::OffsetOverflow));
}

fn layout_bounds(dims: &[usize], strides: &[isize]) -> (isize, isize) {
    let mut min = 0isize;
    let mut max = 0isize;
    for (&dim, &stride) in dims.iter().zip(strides) {
        let extent = (dim - 1) as isize * stride;
        if extent < 0 {
            min += extent;
        } else {
            max += extent;
        }
    }
    (min, max)
}

fn layout_storage<T: Copy + Default>(dims: &[usize], strides: &[isize]) -> (Vec<T>, isize) {
    let (min, max) = layout_bounds(dims, strides);
    let offset = 3 - min;
    let len = usize::try_from(offset + max + 1).unwrap();
    (vec![T::default(); len], offset)
}

fn layout_offset(base: isize, strides: &[isize], coords: &[usize]) -> usize {
    let offset = coords
        .iter()
        .zip(strides)
        .fold(base, |offset, (&coord, &stride)| {
            offset + coord as isize * stride
        });
    usize::try_from(offset).unwrap()
}

fn decode_col_major(mut linear: usize, dims: &[usize]) -> Vec<usize> {
    dims.iter()
        .map(|&dim| {
            let coord = linear % dim;
            linear /= dim;
            coord
        })
        .collect()
}

fn logical_total(dims: &[usize]) -> usize {
    dims.iter().product()
}

fn assert_initialized_layout(
    actual: &[i32],
    expected: &[i32],
    dims: &[usize],
    strides: &[isize],
    offset: isize,
) {
    for linear in 0..logical_total(dims) {
        let coords = decode_col_major(linear, dims);
        let slot = layout_offset(offset, strides, &coords);
        assert_eq!(actual[slot], expected[slot], "logical element {linear}");
    }
}

fn assert_uninitialized_layout(
    actual: &[MaybeUninit<i32>],
    expected: &[i32],
    dims: &[usize],
    strides: &[isize],
    offset: isize,
) {
    for linear in 0..logical_total(dims) {
        let coords = decode_col_major(linear, dims);
        let slot = layout_offset(offset, strides, &coords);
        assert_eq!(unsafe { actual[slot].assume_init_ref() }, &expected[slot]);
    }
}

fn run_dynamic_layout_case<I>(rank: usize, index_dtype: KernelDType, start_value: i32)
where
    I: Copy + Default + From<i32> + strided_kernel::KernelStorageElement,
{
    let operand_dims = vec![3usize; rank];
    let window_dims = vec![2usize; rank];
    let operand_strides: Vec<_> = (0..rank)
        .map(|axis| {
            let stride = 2 * 3isize.pow(u32::try_from(axis).unwrap());
            if axis == 0 {
                -stride
            } else {
                stride
            }
        })
        .collect();
    let window_strides: Vec<_> = (0..rank)
        .map(|axis| {
            let stride = 2 * 2isize.pow(u32::try_from(axis).unwrap());
            if axis == 0 {
                -stride
            } else {
                stride
            }
        })
        .collect();
    let dest_strides = operand_strides.clone();
    let starts_dims = [rank];
    let starts_strides = [1isize];

    let (mut operand_data, operand_offset) = layout_storage::<i32>(&operand_dims, &operand_strides);
    for linear in 0..logical_total(&operand_dims) {
        let coords = decode_col_major(linear, &operand_dims);
        let slot = layout_offset(operand_offset, &operand_strides, &coords);
        operand_data[slot] = 10 + linear as i32;
    }
    let operand = ErasedRawStridedRef::from_slice(
        &operand_data,
        &operand_dims,
        &operand_strides,
        operand_offset,
    )
    .unwrap();

    let mut starts_data = vec![I::default(); rank + 4];
    for (axis, value) in starts_data[2..2 + rank].iter_mut().enumerate() {
        let _ = axis;
        *value = I::from(start_value);
    }
    let starts =
        ErasedRawStridedRef::from_slice(&starts_data, &starts_dims, &starts_strides, 2).unwrap();

    let slice = ErasedDynamicSlicePlan::compile(
        KernelDType::I32,
        index_dtype,
        &operand_dims,
        &operand_strides,
        &starts_dims,
        &starts_strides,
        &window_dims,
        &window_strides,
        &window_dims,
    )
    .unwrap();
    let (mut slice_data, slice_offset) = layout_storage::<i32>(&window_dims, &window_strides);
    slice_data.fill(-777);
    let mut expected_slice = slice_data.clone();
    let clamped = if start_value < 0 { 0 } else { 1 };
    for linear in 0..logical_total(&window_dims) {
        let coords = decode_col_major(linear, &window_dims);
        let source_coords: Vec<_> = coords.iter().map(|&coord| coord + clamped).collect();
        let source_linear = source_coords
            .iter()
            .zip(&operand_dims)
            .enumerate()
            .map(|(axis, (&coord, &dim))| coord * dim.pow(u32::try_from(axis).unwrap()))
            .sum::<usize>();
        let slot = layout_offset(slice_offset, &window_strides, &coords);
        expected_slice[slot] = 10 + source_linear as i32;
    }
    let mut initialized_slice = slice_data;
    let mut initialized_slice_ref = ErasedRawStridedMut::from_slice_mut(
        &mut initialized_slice,
        &window_dims,
        &window_strides,
        slice_offset,
    )
    .unwrap();
    slice
        .execute(
            &ExecContext::serial(),
            &mut initialized_slice_ref,
            &operand,
            &starts,
        )
        .unwrap();
    assert_initialized_layout(
        &initialized_slice,
        &expected_slice,
        &window_dims,
        &window_strides,
        slice_offset,
    );

    let mut raw_slice = vec![MaybeUninit::<i32>::uninit(); expected_slice.len()];
    let mut raw_slice_ref = ErasedRawStridedUninitMut::from_uninit_slice(
        &mut raw_slice,
        &window_dims,
        &window_strides,
        slice_offset,
    )
    .unwrap();
    slice
        .execute_uninit(
            &ExecContext::serial(),
            &mut raw_slice_ref,
            &ErasedRawStridedPtr::from_ref(&operand),
            &ErasedRawStridedPtr::from_ref(&starts),
        )
        .unwrap();
    assert_uninitialized_layout(
        &raw_slice,
        &expected_slice,
        &window_dims,
        &window_strides,
        slice_offset,
    );

    let update = ErasedDynamicUpdateSlicePlan::compile(
        KernelDType::I32,
        index_dtype,
        &operand_dims,
        &operand_strides,
        &starts_dims,
        &starts_strides,
        &window_dims,
        &window_strides,
        &operand_dims,
        &dest_strides,
    )
    .unwrap();
    let (mut update_data, update_offset) = layout_storage::<i32>(&window_dims, &window_strides);
    for linear in 0..logical_total(&window_dims) {
        let coords = decode_col_major(linear, &window_dims);
        let slot = layout_offset(update_offset, &window_strides, &coords);
        update_data[slot] = 1000 + linear as i32;
    }
    let update_ref =
        ErasedRawStridedRef::from_slice(&update_data, &window_dims, &window_strides, update_offset)
            .unwrap();
    let (mut initialized_update, update_dest_offset) =
        layout_storage::<i32>(&operand_dims, &dest_strides);
    initialized_update.fill(-777);
    let mut expected_update = initialized_update.clone();
    for linear in 0..logical_total(&operand_dims) {
        let coords = decode_col_major(linear, &operand_dims);
        let slot = layout_offset(update_dest_offset, &dest_strides, &coords);
        expected_update[slot] = 10 + linear as i32;
    }
    for linear in 0..logical_total(&window_dims) {
        let coords = decode_col_major(linear, &window_dims);
        let dest_coords: Vec<_> = coords.iter().map(|&coord| coord + clamped).collect();
        let slot = layout_offset(update_dest_offset, &dest_strides, &dest_coords);
        expected_update[slot] = 1000 + linear as i32;
    }
    let mut initialized_update_ref = ErasedRawStridedMut::from_slice_mut(
        &mut initialized_update,
        &operand_dims,
        &dest_strides,
        update_dest_offset,
    )
    .unwrap();
    update
        .execute(
            &ExecContext::serial(),
            &mut initialized_update_ref,
            &operand,
            &update_ref,
            &starts,
        )
        .unwrap();
    assert_initialized_layout(
        &initialized_update,
        &expected_update,
        &operand_dims,
        &dest_strides,
        update_dest_offset,
    );

    let mut raw_update = vec![MaybeUninit::<i32>::uninit(); expected_update.len()];
    let mut raw_update_ref = ErasedRawStridedUninitMut::from_uninit_slice(
        &mut raw_update,
        &operand_dims,
        &dest_strides,
        update_dest_offset,
    )
    .unwrap();
    update
        .execute_uninit(
            &ExecContext::serial(),
            &mut raw_update_ref,
            &ErasedRawStridedPtr::from_ref(&operand),
            &ErasedRawStridedPtr::from_ref(&update_ref),
            &ErasedRawStridedPtr::from_ref(&starts),
        )
        .unwrap();
    assert_uninitialized_layout(
        &raw_update,
        &expected_update,
        &operand_dims,
        &dest_strides,
        update_dest_offset,
    );
}

#[test]
fn erased_dynamic_generic_rank_layouts_clamp_and_preserve_updates() {
    for &rank in &[2usize, 4, 8] {
        run_dynamic_layout_case::<i32>(rank, KernelDType::I32, -1);
        run_dynamic_layout_case::<i64>(rank, KernelDType::I64, 99);
    }
}

#[test]
fn erased_dynamic_update_zero_window_still_copies_operand() {
    let operand_dims = [2usize, 2];
    let zero_dims = [0usize, 2];
    let starts_dims = [2usize];
    let operand = [1i32, 2, 3, 4];
    let starts = [0i32, 0];
    let update: [i32; 0] = [];
    let plan = ErasedDynamicUpdateSlicePlan::compile(
        KernelDType::I32,
        KernelDType::I32,
        &operand_dims,
        &[1, 2],
        &starts_dims,
        &[1],
        &zero_dims,
        &[1, 0],
        &operand_dims,
        &[1, 2],
    )
    .unwrap();
    let source = ErasedRawStridedRef::from_slice(&operand, &operand_dims, &[1, 2], 0).unwrap();
    let update_ref = ErasedRawStridedRef::from_slice(&update, &zero_dims, &[1, 0], 0).unwrap();
    let starts_ref = ErasedRawStridedRef::from_slice(&starts, &starts_dims, &[1], 0).unwrap();
    let mut initialized = [0i32; 4];
    let mut initialized_ref =
        ErasedRawStridedMut::from_slice_mut(&mut initialized, &operand_dims, &[1, 2], 0).unwrap();
    plan.execute(
        &ExecContext::serial(),
        &mut initialized_ref,
        &source,
        &update_ref,
        &starts_ref,
    )
    .unwrap();
    assert_eq!(initialized, operand);

    let mut raw = vec![MaybeUninit::<i32>::uninit(); 4];
    let mut raw_ref =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &operand_dims, &[1, 2], 0).unwrap();
    plan.execute_uninit(
        &ExecContext::serial(),
        &mut raw_ref,
        &ErasedRawStridedPtr::from_ref(&source),
        &ErasedRawStridedPtr::from_ref(&update_ref),
        &ErasedRawStridedPtr::from_ref(&starts_ref),
    )
    .unwrap();
    for (actual, expected) in raw.iter().zip(operand) {
        assert_eq!(unsafe { *actual.assume_init_ref() }, expected);
    }
}

fn scatter_window_spec_for_rank(rank: usize) -> ScatterSpec {
    ScatterSpec {
        update_window_dims: (1..rank).collect(),
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    }
}

fn scatter_window_spec() -> ScatterSpec {
    scatter_window_spec_for_rank(2)
}

fn run_compact_windowed_scatter<I>()
where
    I: Copy + Default + From<i32> + strided_kernel::KernelStorageElement,
{
    for &rank in &[2usize, 4, 8] {
        let batch = 4usize;
        let window_elems = 1usize << (rank - 1);
        let mut dims = vec![batch];
        dims.extend((0..rank - 1).map(|_| 2usize));
        let strides = col_major_strides(&dims);
        let total = batch * window_elems;
        let operand: Vec<i32> = (0..total).map(|value| 1000 + value as i32).collect();
        let updates: Vec<i32> = (0..total).map(|value| 1 + value as i32).collect();
        let index_values = [-1, 0, 3, 1];
        let indices: Vec<I> = index_values.into_iter().map(I::from).collect();
        let index_dims = [batch, 1usize];
        let index_strides = [1isize, batch as isize];
        let mut expected = operand.clone();
        for (batch_index, &start) in index_values.iter().enumerate() {
            let start = start.clamp(0, (batch - 1) as i32) as usize;
            for window_index in 0..window_elems {
                let dest_linear = start + batch * window_index;
                let update_linear = batch_index + batch * window_index;
                expected[dest_linear] = expected[dest_linear].wrapping_add(updates[update_linear]);
            }
        }

        let plan = ErasedScatterPlan::compile(
            KernelDType::I32,
            if core::mem::size_of::<I>() == core::mem::size_of::<i32>() {
                KernelDType::I32
            } else {
                KernelDType::I64
            },
            &dims,
            &strides,
            &index_dims,
            &index_strides,
            &dims,
            &strides,
            &dims,
            &strides,
            scatter_window_spec_for_rank(rank),
        )
        .unwrap();
        let operand_ref = ErasedRawStridedRef::from_slice(&operand, &dims, &strides, 0).unwrap();
        let index_ref =
            ErasedRawStridedRef::from_slice(&indices, &index_dims, &index_strides, 0).unwrap();
        let update_ref = ErasedRawStridedRef::from_slice(&updates, &dims, &strides, 0).unwrap();

        let mut initialized = vec![0i32; total];
        let mut initialized_ref =
            ErasedRawStridedMut::from_slice_mut(&mut initialized, &dims, &strides, 0).unwrap();
        plan.execute(
            &ExecContext::serial(),
            &mut initialized_ref,
            &operand_ref,
            &index_ref,
            &update_ref,
        )
        .unwrap();
        assert_eq!(initialized, expected);

        let mut uninitialized = vec![MaybeUninit::<i32>::uninit(); total];
        let mut uninitialized_ref =
            ErasedRawStridedUninitMut::from_uninit_slice(&mut uninitialized, &dims, &strides, 0)
                .unwrap();
        plan.execute_uninit(
            &ExecContext::max_threads(4).unwrap(),
            &mut uninitialized_ref,
            &ErasedRawStridedPtr::from_ref(&operand_ref),
            &ErasedRawStridedPtr::from_ref(&index_ref),
            &ErasedRawStridedPtr::from_ref(&update_ref),
        )
        .unwrap();
        assert_uninitialized_layout(&uninitialized, &expected, &dims, &strides, 0);
    }
}

#[test]
fn erased_scatter_windowed_compact_ranks_match_exact_clamped_replay() {
    run_compact_windowed_scatter::<i32>();
    run_compact_windowed_scatter::<i64>();
}

#[test]
fn erased_scatter_replays_imaginary_vector_axis_and_reordered_windows() {
    let operand_dims = [2usize, 3, 2];
    let operand_strides = [1isize, 2, 6];
    let index_dims = [2usize];
    let index_strides = [1isize];
    let update_dims = [2usize, 2, 2];
    let update_strides = [1isize, 2, 4];
    let indices = [0i64, 2];
    let operand: Vec<i32> = (0..12).map(|value| 100 + value).collect();
    let updates: Vec<i32> = (0..8).map(|value| 10 + value).collect();
    let spec = ScatterSpec {
        update_window_dims: vec![2, 0],
        inserted_window_dims: vec![1],
        scatter_dims_to_operand_dims: vec![1],
        index_vector_dim: 1,
    };
    let plan = ErasedScatterPlan::compile(
        KernelDType::I32,
        KernelDType::I64,
        &operand_dims,
        &operand_strides,
        &index_dims,
        &index_strides,
        &update_dims,
        &update_strides,
        &operand_dims,
        &operand_strides,
        spec,
    )
    .unwrap();
    let operand_ref =
        ErasedRawStridedRef::from_slice(&operand, &operand_dims, &operand_strides, 0).unwrap();
    let index_ref =
        ErasedRawStridedRef::from_slice(&indices, &index_dims, &index_strides, 0).unwrap();
    let update_ref =
        ErasedRawStridedRef::from_slice(&updates, &update_dims, &update_strides, 0).unwrap();

    let mut expected = operand.clone();
    for (batch, &start) in indices.iter().enumerate() {
        let start = start as usize;
        for window_axis_1 in 0..2 {
            for window_axis_0 in 0..2 {
                let update_slot = window_axis_1 + batch * 2 + window_axis_0 * 4;
                let dest_slot = window_axis_0 + start * 2 + window_axis_1 * 6;
                expected[dest_slot] = expected[dest_slot].wrapping_add(updates[update_slot]);
            }
        }
    }

    let mut actual = vec![0i32; operand.len()];
    let mut dest =
        ErasedRawStridedMut::from_slice_mut(&mut actual, &operand_dims, &operand_strides, 0)
            .unwrap();
    plan.execute(
        &ExecContext::serial(),
        &mut dest,
        &operand_ref,
        &index_ref,
        &update_ref,
    )
    .unwrap();
    assert_eq!(actual, expected);
}

fn run_rank2_layout_scatter<I>(
    update_strides: [isize; 2],
    update_offset: isize,
    update_len: usize,
    dest_strides: [isize; 2],
    dest_offset: isize,
    dest_len: usize,
) where
    I: Copy + Default + From<i32> + strided_kernel::KernelStorageElement,
{
    let dims = [4usize, 2];
    let operand_strides = [1isize, 4];
    let operand: Vec<i32> = (0..8).map(|value| 100 + value).collect();
    let indices: Vec<I> = [0, 0, 3, 1].into_iter().map(I::from).collect();
    let updates: Vec<i32> = vec![0; update_len];
    let mut updates = updates;
    for linear in 0..8 {
        let slot = layout_offset(
            update_offset,
            &update_strides,
            &decode_col_major(linear, &dims),
        );
        updates[slot] = 10 + linear as i32;
    }
    let index_dims = [4usize, 1];
    let index_strides = [1isize, 4];
    let mut expected = vec![-777i32; dest_len];
    for linear in 0..8 {
        let coords = decode_col_major(linear, &dims);
        expected[layout_offset(dest_offset, &dest_strides, &coords)] = operand[linear];
    }
    for (batch, &start) in [0usize, 0, 3, 1].iter().enumerate() {
        for window in 0..2 {
            let dest_coords = [start, window];
            let dest_slot = layout_offset(dest_offset, &dest_strides, &dest_coords);
            let update_slot = layout_offset(update_offset, &update_strides, &[batch, window]);
            expected[dest_slot] = expected[dest_slot].wrapping_add(updates[update_slot]);
        }
    }

    let plan = ErasedScatterPlan::compile(
        KernelDType::I32,
        if core::mem::size_of::<I>() == core::mem::size_of::<i32>() {
            KernelDType::I32
        } else {
            KernelDType::I64
        },
        &dims,
        &operand_strides,
        &index_dims,
        &index_strides,
        &dims,
        &update_strides,
        &dims,
        &dest_strides,
        scatter_window_spec(),
    )
    .unwrap();
    let operand_ref =
        ErasedRawStridedRef::from_slice(&operand, &dims, &operand_strides, 0).unwrap();
    let index_ref =
        ErasedRawStridedRef::from_slice(&indices, &index_dims, &index_strides, 0).unwrap();
    let update_ref =
        ErasedRawStridedRef::from_slice(&updates, &dims, &update_strides, update_offset).unwrap();

    let mut initialized = vec![-777i32; dest_len];
    let mut initialized_ref =
        ErasedRawStridedMut::from_slice_mut(&mut initialized, &dims, &dest_strides, dest_offset)
            .unwrap();
    plan.execute(
        &ExecContext::serial(),
        &mut initialized_ref,
        &operand_ref,
        &index_ref,
        &update_ref,
    )
    .unwrap();
    assert_eq!(initialized, expected);

    let mut uninitialized = vec![MaybeUninit::<i32>::uninit(); dest_len];
    let mut uninitialized_ref = ErasedRawStridedUninitMut::from_uninit_slice(
        &mut uninitialized,
        &dims,
        &dest_strides,
        dest_offset,
    )
    .unwrap();
    plan.execute_uninit(
        &ExecContext::serial(),
        &mut uninitialized_ref,
        &ErasedRawStridedPtr::from_ref(&operand_ref),
        &ErasedRawStridedPtr::from_ref(&index_ref),
        &ErasedRawStridedPtr::from_ref(&update_ref),
    )
    .unwrap();
    assert_uninitialized_layout(&uninitialized, &expected, &dims, &dest_strides, dest_offset);
}

#[test]
fn erased_scatter_windowed_negative_nonunit_and_hole_layouts_match() {
    run_rank2_layout_scatter::<i32>([-1, 4], 3, 8, [1, 4], 0, 8);
    run_rank2_layout_scatter::<i64>([-1, 4], 3, 8, [1, 4], 0, 8);
    run_rank2_layout_scatter::<i32>([2, 1], 0, 8, [2, 1], 1, 9);
    run_rank2_layout_scatter::<i64>([2, 1], 0, 8, [2, 1], 1, 9);
}

#[test]
fn erased_scatter_empty_domain_is_a_noop_for_initialized_and_uninitialized() {
    let dims = [4usize, 0];
    let strides = [1isize, 4];
    let index_dims = [4usize, 1];
    let index_strides = [1isize, 4];
    let plan = ErasedScatterPlan::compile(
        KernelDType::I32,
        KernelDType::I64,
        &dims,
        &strides,
        &index_dims,
        &index_strides,
        &dims,
        &strides,
        &dims,
        &strides,
        scatter_window_spec(),
    )
    .unwrap();
    let operand: [i32; 0] = [];
    let indices = [0i64; 4];
    let updates: [i32; 0] = [];
    let operand_ref = ErasedRawStridedRef::from_slice(&operand, &dims, &strides, 0).unwrap();
    let index_ref =
        ErasedRawStridedRef::from_slice(&indices, &index_dims, &index_strides, 0).unwrap();
    let update_ref = ErasedRawStridedRef::from_slice(&updates, &dims, &strides, 0).unwrap();
    let mut initialized: [i32; 0] = [];
    let mut initialized_ref =
        ErasedRawStridedMut::from_slice_mut(&mut initialized, &dims, &strides, 0).unwrap();
    plan.execute(
        &ExecContext::serial(),
        &mut initialized_ref,
        &operand_ref,
        &index_ref,
        &update_ref,
    )
    .unwrap();
    let mut uninitialized = Vec::<MaybeUninit<i32>>::new();
    let mut uninitialized_ref =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut uninitialized, &dims, &strides, 0)
            .unwrap();
    plan.execute_uninit(
        &ExecContext::serial(),
        &mut uninitialized_ref,
        &ErasedRawStridedPtr::from_ref(&operand_ref),
        &ErasedRawStridedPtr::from_ref(&index_ref),
        &ErasedRawStridedPtr::from_ref(&update_ref),
    )
    .unwrap();
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
    let operand_ref = ErasedRawStridedRef::from_slice(&operand, &[4], &[1], 0).unwrap();
    let starts_ref = ErasedRawStridedRef::from_slice(&starts, &[1], &[1], 0).unwrap();
    let mut empty_ref = ErasedRawStridedMut::from_slice_mut(&mut empty, &[0], &[1], 0).unwrap();
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
    let reverse_operand = ErasedRawStridedRef::from_slice(&operand, &[4], &[-1], 3).unwrap();
    let mut reversed = [0.0f64; 4];
    let mut reversed_ref =
        ErasedRawStridedMut::from_slice_mut(&mut reversed, &[4], &[1], 0).unwrap();
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
