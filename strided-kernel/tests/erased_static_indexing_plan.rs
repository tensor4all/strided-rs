use core::mem::MaybeUninit;
use num_complex::{Complex32, Complex64};
use strided_kernel::{
    ErasedConcatenatePlan, ErasedPadPlan, ErasedRawStridedMut, ErasedRawStridedPtr,
    ErasedRawStridedRef, ErasedRawStridedUninitMut, ErasedReversePlan, ErasedSlicePlan,
    ExecContext, KernelDType, StridedError,
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

fn as_uninit_bytes_mut<T>(data: &mut [MaybeUninit<T>]) -> &mut [MaybeUninit<u8>] {
    unsafe {
        core::slice::from_raw_parts_mut(
            data.as_mut_ptr().cast::<MaybeUninit<u8>>(),
            core::mem::size_of_val(data),
        )
    }
}

fn assume_init_vec<T>(data: Vec<MaybeUninit<T>>) -> Vec<T> {
    let mut data = core::mem::ManuallyDrop::new(data);
    unsafe { Vec::from_raw_parts(data.as_mut_ptr().cast::<T>(), data.len(), data.capacity()) }
}

fn assert_uninit_slice<T>(dtype: KernelDType, operand: Vec<T>, expected: Vec<T>)
where
    T: Copy + core::fmt::Debug + PartialEq,
{
    let dims = [operand.len()];
    let strides = [1isize];
    let starts = [0usize];
    let limits = [operand.len()];
    let slice_strides = [1usize];
    let plan = ErasedSlicePlan::compile(
        dtype,
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
        ErasedRawStridedRef::new(dtype, as_bytes(&operand), &dims, &strides, 0).unwrap();
    let operand_ptr = ErasedRawStridedPtr::from_ref(&operand_ref);
    let mut dest = Vec::<MaybeUninit<T>>::with_capacity(operand.len());
    unsafe {
        dest.set_len(operand.len());
    }
    let mut dest_ref =
        ErasedRawStridedUninitMut::new(dtype, as_uninit_bytes_mut(&mut dest), &dims, &strides, 0)
            .unwrap();

    plan.execute_uninit(
        &ExecContext::max_threads(2).unwrap(),
        &mut dest_ref,
        &operand_ptr,
    )
    .unwrap();

    assert_eq!(assume_init_vec(dest), expected);
}

fn assert_empty_uninit_slice<T>(dtype: KernelDType)
where
    T: Copy,
{
    let dims = [0usize];
    let strides = [1isize];
    let starts = [0usize];
    let limits = [0usize];
    let slice_strides = [1usize];
    let plan = ErasedSlicePlan::compile(
        dtype,
        &dims,
        &strides,
        &dims,
        &strides,
        &starts,
        &limits,
        &slice_strides,
    )
    .unwrap();
    let operand: [T; 0] = [];
    let operand_ref =
        ErasedRawStridedRef::new(dtype, as_bytes(&operand), &dims, &strides, 0).unwrap();
    let operand_ptr = ErasedRawStridedPtr::from_ref(&operand_ref);
    let mut empty_bytes: [MaybeUninit<u8>; 0] = [];
    let mut dest_ref =
        ErasedRawStridedUninitMut::new(dtype, &mut empty_bytes, &dims, &strides, 0).unwrap();

    plan.execute_uninit(&ExecContext::serial(), &mut dest_ref, &operand_ptr)
        .unwrap();
}

#[test]
fn erased_slice_uninit_executes_every_kernel_dtype() {
    assert_uninit_slice(KernelDType::F32, vec![1.0f32, -2.0], vec![1.0, -2.0]);
    assert_uninit_slice(KernelDType::F64, vec![1.0f64, -2.0], vec![1.0, -2.0]);
    assert_uninit_slice(KernelDType::I32, vec![1i32, -2], vec![1, -2]);
    assert_uninit_slice(KernelDType::I64, vec![1i64, -2], vec![1, -2]);
    assert_uninit_slice(KernelDType::Bool, vec![true, false], vec![true, false]);
    assert_uninit_slice(
        KernelDType::C32,
        vec![Complex32::new(1.0, 2.0), Complex32::new(-2.0, 3.0)],
        vec![Complex32::new(1.0, 2.0), Complex32::new(-2.0, 3.0)],
    );
    assert_uninit_slice(
        KernelDType::C64,
        vec![Complex64::new(1.0, 2.0), Complex64::new(-2.0, 3.0)],
        vec![Complex64::new(1.0, 2.0), Complex64::new(-2.0, 3.0)],
    );
}

#[test]
fn erased_slice_uninit_accepts_unaligned_empty_storage_for_every_kernel_dtype() {
    assert_empty_uninit_slice::<f32>(KernelDType::F32);
    assert_empty_uninit_slice::<f64>(KernelDType::F64);
    assert_empty_uninit_slice::<i32>(KernelDType::I32);
    assert_empty_uninit_slice::<i64>(KernelDType::I64);
    assert_empty_uninit_slice::<bool>(KernelDType::Bool);
    assert_empty_uninit_slice::<Complex32>(KernelDType::C32);
    assert_empty_uninit_slice::<Complex64>(KernelDType::C64);
}

#[test]
fn erased_reverse_uninit_handles_rank_above_inline_limit() {
    let dims = [2usize; 9];
    let strides = [1isize, 2, 4, 8, 16, 32, 64, 128, 256];
    let axes = [0usize, 8];
    let operand: Vec<i32> = (0..512).collect();
    let plan =
        ErasedReversePlan::compile(KernelDType::I32, &dims, &strides, &strides, &axes).unwrap();
    let operand_ref =
        ErasedRawStridedRef::new(KernelDType::I32, as_bytes(&operand), &dims, &strides, 0).unwrap();
    let operand_ptr = ErasedRawStridedPtr::from_ref(&operand_ref);
    let mut dest = vec![MaybeUninit::<i32>::uninit(); operand.len()];
    let mut dest_ref = ErasedRawStridedUninitMut::new(
        KernelDType::I32,
        as_uninit_bytes_mut(&mut dest),
        &dims,
        &strides,
        0,
    )
    .unwrap();

    plan.execute_uninit(&ExecContext::serial(), &mut dest_ref, &operand_ptr)
        .unwrap();

    let dest = assume_init_vec(dest);
    assert_eq!(dest[0], operand[257]);
    assert_eq!(dest[511], operand[254]);
}

#[test]
fn erased_pad_and_concatenate_uninit_cover_noncontiguous_and_multi_input_outputs() {
    let operand_dims = [2usize];
    let operand_strides = [1isize];
    let padded_dims = [4usize];
    let padded_strides = [2isize];
    let edge_low = [1i64];
    let edge_high = [1i64];
    let interior = [0i64];
    let operand = [10i32, 20];
    let fill = [-1i32];
    let pad = ErasedPadPlan::compile(
        KernelDType::I32,
        &operand_dims,
        &operand_strides,
        &padded_dims,
        &padded_strides,
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
    let operand_ptr = ErasedRawStridedPtr::from_ref(&operand_ref);
    let mut padded = vec![MaybeUninit::<i32>::uninit(); 7];
    let mut padded_ref = ErasedRawStridedUninitMut::new(
        KernelDType::I32,
        as_uninit_bytes_mut(&mut padded),
        &padded_dims,
        &padded_strides,
        0,
    )
    .unwrap();
    pad.execute_uninit(
        &ExecContext::serial(),
        &mut padded_ref,
        &operand_ptr,
        as_bytes(&fill),
    )
    .unwrap();
    assert_eq!(
        [0usize, 2, 4, 6].map(|index| unsafe { padded[index].assume_init() }),
        [-1, 10, 20, -1]
    );

    let lhs = [1i32, 2];
    let rhs = [3i32, 4];
    let input_dims: [&[usize]; 2] = [&operand_dims, &operand_dims];
    let input_strides: [&[isize]; 2] = [&operand_strides, &operand_strides];
    let concat_dims = [4usize];
    let concat_strides = [1isize];
    let concat = ErasedConcatenatePlan::compile(
        KernelDType::I32,
        &input_dims,
        &input_strides,
        &concat_dims,
        &concat_strides,
        0,
    )
    .unwrap();
    let lhs_ref = ErasedRawStridedRef::new(
        KernelDType::I32,
        as_bytes(&lhs),
        &operand_dims,
        &operand_strides,
        0,
    )
    .unwrap();
    let rhs_ref = ErasedRawStridedRef::new(
        KernelDType::I32,
        as_bytes(&rhs),
        &operand_dims,
        &operand_strides,
        0,
    )
    .unwrap();
    let inputs = [
        ErasedRawStridedPtr::from_ref(&lhs_ref),
        ErasedRawStridedPtr::from_ref(&rhs_ref),
    ];
    let mut joined = vec![MaybeUninit::<i32>::uninit(); 4];
    let mut joined_ref = ErasedRawStridedUninitMut::new(
        KernelDType::I32,
        as_uninit_bytes_mut(&mut joined),
        &concat_dims,
        &concat_strides,
        0,
    )
    .unwrap();
    concat
        .execute_uninit(&ExecContext::serial(), &mut joined_ref, &inputs)
        .unwrap();
    assert_eq!(assume_init_vec(joined), [1, 2, 3, 4]);
}

#[test]
fn erased_uninit_static_replay_rejects_input_output_overlap_before_writing() {
    let dims = [2usize];
    let strides = [1isize];
    let starts = [0usize];
    let limits = [2usize];
    let slice_strides = [1usize];
    let plan = ErasedSlicePlan::compile(
        KernelDType::I32,
        &dims,
        &strides,
        &dims,
        &strides,
        &starts,
        &limits,
        &slice_strides,
    )
    .unwrap();
    let mut storage = vec![MaybeUninit::new(7i32), MaybeUninit::new(9i32)];
    let input_ptr = unsafe {
        ErasedRawStridedPtr::new(
            KernelDType::I32,
            core::ptr::NonNull::new(storage.as_mut_ptr().cast::<u8>()).unwrap(),
            core::mem::size_of_val(storage.as_slice()),
            &dims,
            &strides,
            0,
        )
        .unwrap()
    };
    let mut dest = ErasedRawStridedUninitMut::new(
        KernelDType::I32,
        as_uninit_bytes_mut(&mut storage),
        &dims,
        &strides,
        0,
    )
    .unwrap();

    let error = plan
        .execute_uninit(&ExecContext::serial(), &mut dest, &input_ptr)
        .unwrap_err();

    assert!(matches!(
        error,
        StridedError::OverlappingInputOutput { input: 0 }
    ));
    assert_eq!(
        storage
            .into_iter()
            .map(|value| unsafe { value.assume_init() })
            .collect::<Vec<_>>(),
        [7, 9]
    );
}

#[test]
fn erased_concatenate_validates_all_segment_offsets_before_writing() {
    let first_dims = [1usize];
    let empty_dims = [0usize];
    let input_strides = [1isize];
    let dest_dims = [1usize];
    let dest_strides = [isize::MAX];
    let input_dims: [&[usize]; 2] = [&first_dims, &empty_dims];
    let input_stride_sets: [&[isize]; 2] = [&input_strides, &input_strides];
    let plan = ErasedConcatenatePlan::compile(
        KernelDType::I32,
        &input_dims,
        &input_stride_sets,
        &dest_dims,
        &dest_strides,
        0,
    )
    .unwrap();
    let first = [11i32];
    let empty: [i32; 0] = [];
    let first_ref = ErasedRawStridedRef::new(
        KernelDType::I32,
        as_bytes(&first),
        &first_dims,
        &input_strides,
        0,
    )
    .unwrap();
    let empty_ref = ErasedRawStridedRef::new(
        KernelDType::I32,
        as_bytes(&empty),
        &empty_dims,
        &input_strides,
        0,
    )
    .unwrap();
    let inputs = [
        ErasedRawStridedPtr::from_ref(&first_ref),
        ErasedRawStridedPtr::from_ref(&empty_ref),
    ];
    let mut dest = [MaybeUninit::new(3i32), MaybeUninit::new(7i32)];
    let mut dest_ref = ErasedRawStridedUninitMut::new(
        KernelDType::I32,
        as_uninit_bytes_mut(&mut dest),
        &dest_dims,
        &dest_strides,
        1,
    )
    .unwrap();

    let error = plan
        .execute_uninit(&ExecContext::serial(), &mut dest_ref, &inputs)
        .unwrap_err();

    assert!(matches!(error, StridedError::OffsetOverflow));
    assert_eq!(
        dest.map(|value| unsafe { value.assume_init() }),
        [3, 7],
        "validation failure must precede every segment write"
    );
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
