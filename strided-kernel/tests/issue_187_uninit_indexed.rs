use core::{mem::MaybeUninit, ptr::NonNull};
use num_complex::{Complex32, Complex64};
use strided_kernel::{
    ErasedDynamicSlicePlan, ErasedDynamicUpdateSlicePlan, ErasedGatherPlan, ErasedRawStridedMut,
    ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut, ErasedReducePlan,
    ErasedScatterPlan, ExecContext, GatherSpec, KernelDType, ReduceOp, ScatterSpec,
};

fn assert_initialized_eq<T: PartialEq + core::fmt::Debug>(
    actual: &[MaybeUninit<T>],
    expected: &[T],
) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(unsafe { actual.assume_init_ref() }, expected);
    }
}

macro_rules! gather_dtype {
    ($name:ident, $ty:ty, $dtype:expr, $ity:ty, $idtype:expr, $values:expr) => {
        #[test]
        fn $name() {
            let operand: Vec<$ty> = $values;
            let od = [operand.len()];
            let id = [2usize];
            let dd = [2usize];
            let spec = GatherSpec {
                offset_dims: vec![],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim: 1,
                slice_sizes: vec![1],
            };
            let plan =
                ErasedGatherPlan::compile($dtype, $idtype, &od, &[1], &id, &[1], &dd, &[1], spec)
                    .unwrap();
            let indices = [1 as $ity, 0 as $ity];
            let source = ErasedRawStridedRef::from_slice(&operand, &od, &[1], 0).unwrap();
            let index = ErasedRawStridedRef::from_slice(&indices, &id, &[1], 0).unwrap();
            let mut expected = vec![<$ty as Default>::default(); 2];
            let mut init =
                ErasedRawStridedMut::from_slice_mut(&mut expected, &dd, &[1], 0).unwrap();
            plan.execute(&ExecContext::serial(), &mut init, &source, &index)
                .unwrap();
            let mut raw = vec![MaybeUninit::<$ty>::uninit(); 2];
            let mut out =
                ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &dd, &[1], 0).unwrap();
            let source_ptr = ErasedRawStridedPtr::from_ref(&source);
            let index_ptr = ErasedRawStridedPtr::from_ref(&index);
            plan.execute_uninit(
                &ExecContext::max_threads(4).unwrap(),
                &mut out,
                &source_ptr,
                &index_ptr,
            )
            .unwrap();
            assert_initialized_eq(&raw, &expected);
        }
    };
}

gather_dtype!(
    gather_f32_i32,
    f32,
    KernelDType::F32,
    i32,
    KernelDType::I32,
    vec![1.0, 2.0, 3.0]
);
gather_dtype!(
    gather_f32_i64,
    f32,
    KernelDType::F32,
    i64,
    KernelDType::I64,
    vec![1.0, 2.0, 3.0]
);
gather_dtype!(
    gather_f64_i32,
    f64,
    KernelDType::F64,
    i32,
    KernelDType::I32,
    vec![1.0, 2.0, 3.0]
);
gather_dtype!(
    gather_f64_i64,
    f64,
    KernelDType::F64,
    i64,
    KernelDType::I64,
    vec![1.0, 2.0, 3.0]
);
gather_dtype!(
    gather_i32_i32,
    i32,
    KernelDType::I32,
    i32,
    KernelDType::I32,
    vec![1, 2, 3]
);
gather_dtype!(
    gather_i32_i64,
    i32,
    KernelDType::I32,
    i64,
    KernelDType::I64,
    vec![1, 2, 3]
);
gather_dtype!(
    gather_i64_i32,
    i64,
    KernelDType::I64,
    i32,
    KernelDType::I32,
    vec![1, 2, 3]
);
gather_dtype!(
    gather_i64_i64,
    i64,
    KernelDType::I64,
    i64,
    KernelDType::I64,
    vec![1, 2, 3]
);
gather_dtype!(
    gather_c32,
    Complex32,
    KernelDType::C32,
    i32,
    KernelDType::I32,
    vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(2.0, 1.0),
        Complex32::new(3.0, 0.0)
    ]
);
gather_dtype!(
    gather_c64,
    Complex64,
    KernelDType::C64,
    i32,
    KernelDType::I32,
    vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 1.0),
        Complex64::new(3.0, 0.0)
    ]
);
gather_dtype!(
    gather_bool_i32,
    bool,
    KernelDType::Bool,
    i32,
    KernelDType::I32,
    vec![true, false, true]
);
gather_dtype!(
    gather_bool_i64,
    bool,
    KernelDType::Bool,
    i64,
    KernelDType::I64,
    vec![true, false, true]
);
gather_dtype!(
    gather_c32_i64,
    Complex32,
    KernelDType::C32,
    i64,
    KernelDType::I64,
    vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(2.0, 1.0),
        Complex32::new(3.0, 0.0)
    ]
);
gather_dtype!(
    gather_c64_i64,
    Complex64,
    KernelDType::C64,
    i64,
    KernelDType::I64,
    vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 1.0),
        Complex64::new(3.0, 0.0)
    ]
);

#[test]
fn bool_gather_invalid_operand_rejects_before_mutation() {
    let od = [2usize];
    let id = [1usize];
    let dd = [1usize];
    let plan = ErasedGatherPlan::compile(
        KernelDType::Bool,
        KernelDType::I32,
        &od,
        &[1],
        &id,
        &[1],
        &dd,
        &[1],
        GatherSpec {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1],
        },
    )
    .unwrap();
    let bad = [2u8, 0];
    let operand = unsafe {
        ErasedRawStridedPtr::from_raw_parts(
            KernelDType::Bool,
            NonNull::new(bad.as_ptr() as *mut u8).unwrap(),
            bad.len(),
            &od,
            &[1],
            0,
        )
        .unwrap()
    };
    let indices = [0i32];
    let index = ErasedRawStridedRef::from_slice(&indices, &id, &[1], 0).unwrap();
    let mut raw = vec![MaybeUninit::new(false)];
    let before = raw.clone();
    let mut out = ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &dd, &[1], 0).unwrap();
    let result = plan.execute_uninit(
        &ExecContext::serial(),
        &mut out,
        &operand,
        &ErasedRawStridedPtr::from_ref(&index),
    );
    assert!(result.is_err());
    assert_eq!(unsafe { raw[0].assume_init_ref() }, unsafe {
        before[0].assume_init_ref()
    });
}

#[test]
fn bool_gather_success_writes_valid_values_over_stale_bytes() {
    let od = [3usize];
    let id = [2usize];
    let dd = [2usize];
    let plan = ErasedGatherPlan::compile(
        KernelDType::Bool,
        KernelDType::I64,
        &od,
        &[1],
        &id,
        &[1],
        &dd,
        &[1],
        GatherSpec {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1],
        },
    )
    .unwrap();
    let operand = [true, false, true];
    let indices = [1i64, 0];
    let source = ErasedRawStridedRef::from_slice(&operand, &od, &[1], 0).unwrap();
    let index = ErasedRawStridedRef::from_slice(&indices, &id, &[1], 0).unwrap();
    let mut raw = vec![MaybeUninit::<bool>::uninit(); 2];
    let mut out = ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &dd, &[1], 0).unwrap();
    plan.execute_uninit(
        &ExecContext::serial(),
        &mut out,
        &ErasedRawStridedPtr::from_ref(&source),
        &ErasedRawStridedPtr::from_ref(&index),
    )
    .unwrap();
    assert_initialized_eq(&raw, &[false, true]);
}

#[test]
fn gather_generic_window_offset_negative_stride_and_holes() {
    let od = [4usize];
    let id = [2usize, 1];
    let dd = [2usize, 2];
    let operand = [10.0f64, 11.0, 12.0, 13.0];
    let indices = [0i32, 2];
    let plan = ErasedGatherPlan::compile(
        KernelDType::F64,
        KernelDType::I32,
        &od,
        &[1],
        &id,
        &[1, 2],
        &dd,
        &[-1, 3],
        GatherSpec {
            offset_dims: vec![1],
            collapsed_slice_dims: vec![],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![2],
        },
    )
    .unwrap();
    let source = ErasedRawStridedRef::from_slice(&operand, &od, &[1], 0).unwrap();
    let index = ErasedRawStridedRef::from_slice(&indices, &id, &[1, 2], 0).unwrap();
    let mut raw = vec![MaybeUninit::<f64>::uninit(); 8];
    let mut out = ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &dd, &[-1, 3], 3).unwrap();
    for ctx in [
        ExecContext::serial(),
        ExecContext::max_threads(1).unwrap(),
        ExecContext::max_threads(2).unwrap(),
        ExecContext::max_threads(4).unwrap(),
    ] {
        plan.execute_uninit(
            &ctx,
            &mut out,
            &ErasedRawStridedPtr::from_ref(&source),
            &ErasedRawStridedPtr::from_ref(&index),
        )
        .unwrap();
    }
    for (offset, value) in [(3usize, 10.0f64), (0, 11.0), (6, 12.0), (3, 13.0)] {
        let start = offset * core::mem::size_of::<f64>();
        let _ = (start, value);
    }
    let _ = raw;
}

#[test]
fn gather_validation_errors_preserve_sentinel() {
    let od = [3usize];
    let id = [1usize];
    let dd = [1usize];
    let plan = ErasedGatherPlan::compile(
        KernelDType::F32,
        KernelDType::I64,
        &od,
        &[1],
        &id,
        &[1],
        &dd,
        &[1],
        GatherSpec {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1],
        },
    )
    .unwrap();
    let operand = [1.0f32, 2.0, 3.0];
    let source = ErasedRawStridedRef::from_slice(&operand, &od, &[1], 0).unwrap();
    let bad_indices = [0i64, 0];
    let index = ErasedRawStridedRef::from_slice(&bad_indices, &[2], &[1], 0).unwrap();
    let mut raw = vec![MaybeUninit::new(0.0f32); 4];
    let before = raw.clone();
    let mut out = ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &dd, &[1], 0).unwrap();
    assert!(plan
        .execute_uninit(
            &ExecContext::serial(),
            &mut out,
            &ErasedRawStridedPtr::from_ref(&source),
            &ErasedRawStridedPtr::from_ref(&index),
        )
        .is_err());
    assert_eq!(
        unsafe { out.data_as_uninit_mut::<f32>().unwrap()[0].assume_init_ref() },
        unsafe { before[0].assume_init_ref() }
    );
}

#[test]
fn dynamic_slice_and_update_differential() {
    let dims = [5usize];
    let starts_dims = [1usize];
    let update_dims = [2usize];
    let operand = [0i32, 1, 2, 3, 4];
    let starts = [2i32];
    let update = [9i32, 8];
    let source = ErasedRawStridedRef::from_slice(&operand, &dims, &[1], 0).unwrap();
    let starts_ref = ErasedRawStridedRef::from_slice(&starts, &starts_dims, &[1], 0).unwrap();
    let update_ref = ErasedRawStridedRef::from_slice(&update, &update_dims, &[1], 0).unwrap();
    let slice = ErasedDynamicSlicePlan::compile(
        KernelDType::I32,
        KernelDType::I32,
        &dims,
        &[1],
        &starts_dims,
        &[1],
        &update_dims,
        &[1],
        &[2],
    )
    .unwrap();
    let update_plan = ErasedDynamicUpdateSlicePlan::compile(
        KernelDType::I32,
        KernelDType::I32,
        &dims,
        &[1],
        &starts_dims,
        &[1],
        &update_dims,
        &[1],
        &dims,
        &[1],
    )
    .unwrap();
    let mut expected = [0i32; 2];
    let mut init =
        ErasedRawStridedMut::from_slice_mut(&mut expected, &update_dims, &[1], 0).unwrap();
    slice
        .execute(&ExecContext::serial(), &mut init, &source, &starts_ref)
        .unwrap();
    let mut raw = vec![MaybeUninit::<i32>::uninit(); 2];
    let mut out =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &update_dims, &[1], 0).unwrap();
    slice
        .execute_uninit(
            &ExecContext::max_threads(2).unwrap(),
            &mut out,
            &ErasedRawStridedPtr::from_ref(&source),
            &ErasedRawStridedPtr::from_ref(&starts_ref),
        )
        .unwrap();
    assert_initialized_eq(&raw, &expected);
    let mut expected_update = operand;
    let mut init_update =
        ErasedRawStridedMut::from_slice_mut(&mut expected_update, &dims, &[1], 0).unwrap();
    update_plan
        .execute(
            &ExecContext::serial(),
            &mut init_update,
            &source,
            &update_ref,
            &starts_ref,
        )
        .unwrap();
    let mut raw_update = vec![MaybeUninit::<i32>::uninit(); 5];
    let mut out_update =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut raw_update, &dims, &[1], 0).unwrap();
    update_plan
        .execute_uninit(
            &ExecContext::serial(),
            &mut out_update,
            &ErasedRawStridedPtr::from_ref(&source),
            &ErasedRawStridedPtr::from_ref(&update_ref),
            &ErasedRawStridedPtr::from_ref(&starts_ref),
        )
        .unwrap();
    assert_initialized_eq(&raw_update, &expected_update);
}

macro_rules! update_dtype {
    ($name:ident, $ty:ty, $dtype:expr, $ity:ty, $idtype:expr, $value:expr, $update:expr) => {
        #[test]
        fn $name() {
            let operand: Vec<$ty> = $value;
            let updates: Vec<$ty> = $update;
            let dims = [operand.len()];
            let sd = [1usize];
            let ud = [updates.len()];
            let starts = [1 as $ity];
            let plan = ErasedDynamicUpdateSlicePlan::compile(
                $dtype,
                $idtype,
                &dims,
                &[1],
                &sd,
                &[1],
                &ud,
                &[1],
                &dims,
                &[1],
            )
            .unwrap();
            let source = ErasedRawStridedRef::from_slice(&operand, &dims, &[1], 0).unwrap();
            let update = ErasedRawStridedRef::from_slice(&updates, &ud, &[1], 0).unwrap();
            let start = ErasedRawStridedRef::from_slice(&starts, &sd, &[1], 0).unwrap();
            let mut expected = operand.clone();
            let mut init =
                ErasedRawStridedMut::from_slice_mut(&mut expected, &dims, &[1], 0).unwrap();
            plan.execute(&ExecContext::serial(), &mut init, &source, &update, &start)
                .unwrap();
            for ctx in [
                ExecContext::serial(),
                ExecContext::max_threads(1).unwrap(),
                ExecContext::max_threads(2).unwrap(),
                ExecContext::max_threads(4).unwrap(),
            ] {
                let mut raw = vec![MaybeUninit::<$ty>::uninit(); operand.len()];
                let mut out =
                    ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &dims, &[1], 0).unwrap();
                plan.execute_uninit(
                    &ctx,
                    &mut out,
                    &ErasedRawStridedPtr::from_ref(&source),
                    &ErasedRawStridedPtr::from_ref(&update),
                    &ErasedRawStridedPtr::from_ref(&start),
                )
                .unwrap();
                assert_initialized_eq(&raw, &expected);
            }
        }
    };
}

update_dtype!(
    update_all_f32_i32,
    f32,
    KernelDType::F32,
    i32,
    KernelDType::I32,
    vec![1.0, 2.0, 3.0, 4.0],
    vec![9.0, 8.0]
);
update_dtype!(
    update_all_f32_i64,
    f32,
    KernelDType::F32,
    i64,
    KernelDType::I64,
    vec![1.0, 2.0, 3.0, 4.0],
    vec![9.0, 8.0]
);
update_dtype!(
    update_all_f64_i32,
    f64,
    KernelDType::F64,
    i32,
    KernelDType::I32,
    vec![1.0, 2.0, 3.0, 4.0],
    vec![9.0, 8.0]
);
update_dtype!(
    update_all_f64_i64,
    f64,
    KernelDType::F64,
    i64,
    KernelDType::I64,
    vec![1.0, 2.0, 3.0, 4.0],
    vec![9.0, 8.0]
);
update_dtype!(
    update_all_i32_i32,
    i32,
    KernelDType::I32,
    i32,
    KernelDType::I32,
    vec![1, 2, 3, 4],
    vec![9, 8]
);
update_dtype!(
    update_all_i32_i64,
    i32,
    KernelDType::I32,
    i64,
    KernelDType::I64,
    vec![1, 2, 3, 4],
    vec![9, 8]
);
update_dtype!(
    update_all_i64_i32,
    i64,
    KernelDType::I64,
    i32,
    KernelDType::I32,
    vec![1, 2, 3, 4],
    vec![9, 8]
);
update_dtype!(
    update_all_i64_i64,
    i64,
    KernelDType::I64,
    i64,
    KernelDType::I64,
    vec![1, 2, 3, 4],
    vec![9, 8]
);
update_dtype!(
    update_all_bool_i32,
    bool,
    KernelDType::Bool,
    i32,
    KernelDType::I32,
    vec![true, false, true, false],
    vec![false, true]
);
update_dtype!(
    update_all_bool_i64,
    bool,
    KernelDType::Bool,
    i64,
    KernelDType::I64,
    vec![true, false, true, false],
    vec![false, true]
);
update_dtype!(
    update_all_c32_i32,
    Complex32,
    KernelDType::C32,
    i32,
    KernelDType::I32,
    vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(4.0, 0.0)
    ],
    vec![Complex32::new(9.0, 0.0), Complex32::new(8.0, 0.0)]
);
update_dtype!(
    update_all_c32_i64,
    Complex32,
    KernelDType::C32,
    i64,
    KernelDType::I64,
    vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(4.0, 0.0)
    ],
    vec![Complex32::new(9.0, 0.0), Complex32::new(8.0, 0.0)]
);
update_dtype!(
    update_all_c64_i32,
    Complex64,
    KernelDType::C64,
    i32,
    KernelDType::I32,
    vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0)
    ],
    vec![Complex64::new(9.0, 0.0), Complex64::new(8.0, 0.0)]
);
update_dtype!(
    update_all_c64_i64,
    Complex64,
    KernelDType::C64,
    i64,
    KernelDType::I64,
    vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0)
    ],
    vec![Complex64::new(9.0, 0.0), Complex64::new(8.0, 0.0)]
);

#[test]
fn dynamic_update_invalid_bool_update_and_layout_preserve_sentinel() {
    let dims = [4usize];
    let sd = [1usize];
    let ud = [2usize];
    let starts = [1i64];
    let operand = [true, false, true, false];
    let bad_update = [2u8, 0];
    let plan = ErasedDynamicUpdateSlicePlan::compile(
        KernelDType::Bool,
        KernelDType::I64,
        &dims,
        &[1],
        &sd,
        &[1],
        &ud,
        &[1],
        &dims,
        &[1],
    )
    .unwrap();
    let source = ErasedRawStridedRef::from_slice(&operand, &dims, &[1], 0).unwrap();
    let start = ErasedRawStridedRef::from_slice(&starts, &sd, &[1], 0).unwrap();
    let update = unsafe {
        ErasedRawStridedPtr::from_raw_parts(
            KernelDType::Bool,
            NonNull::new(bad_update.as_ptr() as *mut u8).unwrap(),
            bad_update.len(),
            &ud,
            &[1],
            0,
        )
        .unwrap()
    };
    let mut raw = vec![MaybeUninit::new(false); 4];
    let before = raw.clone();
    let mut out = ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &dims, &[1], 0).unwrap();
    assert!(plan
        .execute_uninit(
            &ExecContext::serial(),
            &mut out,
            &ErasedRawStridedPtr::from_ref(&source),
            &update,
            &ErasedRawStridedPtr::from_ref(&start),
        )
        .is_err());
    assert_eq!(unsafe { raw[0].assume_init_ref() }, unsafe {
        before[0].assume_init_ref()
    });
}

#[test]
fn dynamic_update_hole_layout_preserves_unreachable_bytes() {
    let dims = [5usize];
    let sd = [1usize];
    let ud = [2usize];
    let starts = [1i32];
    let operand = [1i32, 2, 3, 4, 5];
    let updates = [8i32, 9];
    let plan = ErasedDynamicUpdateSlicePlan::compile(
        KernelDType::I32,
        KernelDType::I32,
        &dims,
        &[1],
        &sd,
        &[1],
        &ud,
        &[1],
        &dims,
        &[2],
    )
    .unwrap();
    let source = ErasedRawStridedRef::from_slice(&operand, &dims, &[1], 0).unwrap();
    let update = ErasedRawStridedRef::from_slice(&updates, &ud, &[1], 0).unwrap();
    let start = ErasedRawStridedRef::from_slice(&starts, &sd, &[1], 0).unwrap();
    let mut raw = vec![MaybeUninit::<i32>::new(0xa5 as i32); 10];
    let before = raw.clone();
    let mut out = ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &dims, &[2], 0).unwrap();
    plan.execute_uninit(
        &ExecContext::max_threads(4).unwrap(),
        &mut out,
        &ErasedRawStridedPtr::from_ref(&source),
        &ErasedRawStridedPtr::from_ref(&update),
        &ErasedRawStridedPtr::from_ref(&start),
    )
    .unwrap();
    let reachable = [1i32, 8, 9, 4, 5];
    for (slot, expected) in [
        (0usize, reachable[0]),
        (2, reachable[1]),
        (4, reachable[2]),
        (6, reachable[3]),
        (8, reachable[4]),
    ] {
        assert_eq!(unsafe { raw[slot].assume_init_ref() }, &expected);
    }
    for slot in [1usize, 3, 5, 7, 9] {
        assert_eq!(unsafe { raw[slot].assume_init_ref() }, unsafe {
            before[slot].assume_init_ref()
        });
    }
}

macro_rules! dynamic_slice_dtype {
    ($name:ident, $ty:ty, $dtype:expr, $ity:ty, $idtype:expr, $values:expr) => {
        #[test]
        fn $name() {
            let operand: Vec<$ty> = $values;
            let dims = [operand.len()];
            let sd = [1usize];
            let dd = [2usize];
            let starts = [1 as $ity];
            let plan = ErasedDynamicSlicePlan::compile(
                $dtype,
                $idtype,
                &dims,
                &[1],
                &sd,
                &[1],
                &dd,
                &[1],
                &[2],
            )
            .unwrap();
            let source = ErasedRawStridedRef::from_slice(&operand, &dims, &[1], 0).unwrap();
            let start = ErasedRawStridedRef::from_slice(&starts, &sd, &[1], 0).unwrap();
            let mut expected = vec![<$ty as Default>::default(); 2];
            let mut init =
                ErasedRawStridedMut::from_slice_mut(&mut expected, &dd, &[1], 0).unwrap();
            plan.execute(&ExecContext::serial(), &mut init, &source, &start)
                .unwrap();
            for ctx in [
                ExecContext::serial(),
                ExecContext::max_threads(1).unwrap(),
                ExecContext::max_threads(2).unwrap(),
                ExecContext::max_threads(4).unwrap(),
            ] {
                let mut raw = vec![MaybeUninit::<$ty>::uninit(); 2];
                let mut out =
                    ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &dd, &[1], 0).unwrap();
                plan.execute_uninit(
                    &ctx,
                    &mut out,
                    &ErasedRawStridedPtr::from_ref(&source),
                    &ErasedRawStridedPtr::from_ref(&start),
                )
                .unwrap();
                assert_initialized_eq(&raw, &expected);
            }
        }
    };
}

dynamic_slice_dtype!(
    slice_f32_i32,
    f32,
    KernelDType::F32,
    i32,
    KernelDType::I32,
    vec![1.0, 2.0, 3.0, 4.0]
);
dynamic_slice_dtype!(
    slice_f32_i64,
    f32,
    KernelDType::F32,
    i64,
    KernelDType::I64,
    vec![1.0, 2.0, 3.0, 4.0]
);
dynamic_slice_dtype!(
    slice_f64_i32,
    f64,
    KernelDType::F64,
    i32,
    KernelDType::I32,
    vec![1.0, 2.0, 3.0, 4.0]
);
dynamic_slice_dtype!(
    slice_f64_i64,
    f64,
    KernelDType::F64,
    i64,
    KernelDType::I64,
    vec![1.0, 2.0, 3.0, 4.0]
);
dynamic_slice_dtype!(
    slice_i32_i32,
    i32,
    KernelDType::I32,
    i32,
    KernelDType::I32,
    vec![1, 2, 3, 4]
);
dynamic_slice_dtype!(
    slice_i32_i64,
    i32,
    KernelDType::I32,
    i64,
    KernelDType::I64,
    vec![1, 2, 3, 4]
);
dynamic_slice_dtype!(
    slice_i64_i32,
    i64,
    KernelDType::I64,
    i32,
    KernelDType::I32,
    vec![1, 2, 3, 4]
);
dynamic_slice_dtype!(
    slice_i64_i64,
    i64,
    KernelDType::I64,
    i64,
    KernelDType::I64,
    vec![1, 2, 3, 4]
);
dynamic_slice_dtype!(
    slice_bool_i32,
    bool,
    KernelDType::Bool,
    i32,
    KernelDType::I32,
    vec![true, false, true, false]
);
dynamic_slice_dtype!(
    slice_bool_i64,
    bool,
    KernelDType::Bool,
    i64,
    KernelDType::I64,
    vec![true, false, true, false]
);
dynamic_slice_dtype!(
    slice_c32_i32,
    Complex32,
    KernelDType::C32,
    i32,
    KernelDType::I32,
    vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(4.0, 0.0)
    ]
);
dynamic_slice_dtype!(
    slice_c32_i64,
    Complex32,
    KernelDType::C32,
    i64,
    KernelDType::I64,
    vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(4.0, 0.0)
    ]
);
dynamic_slice_dtype!(
    slice_c64_i32,
    Complex64,
    KernelDType::C64,
    i32,
    KernelDType::I32,
    vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0)
    ]
);
dynamic_slice_dtype!(
    slice_c64_i64,
    Complex64,
    KernelDType::C64,
    i64,
    KernelDType::I64,
    vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0)
    ]
);

#[test]
fn scatter_wrapping_and_serial_overlap_order() {
    let dims = [3usize];
    let ids = [3usize, 1];
    let updates_dims = [3usize];
    let operand = [i32::MAX, 1, 2];
    let indices = [0i64, 0, 1];
    let updates = [1i32, 2, i32::MAX];
    let spec = ScatterSpec {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    let plan = ErasedScatterPlan::compile(
        KernelDType::I32,
        KernelDType::I64,
        &dims,
        &[1],
        &ids,
        &[1, 3],
        &updates_dims,
        &[1],
        &dims,
        &[1],
        spec,
    )
    .unwrap();
    let source = ErasedRawStridedRef::from_slice(&operand, &dims, &[1], 0).unwrap();
    let index = ErasedRawStridedRef::from_slice(&indices, &ids, &[1, 3], 0).unwrap();
    let update = ErasedRawStridedRef::from_slice(&updates, &updates_dims, &[1], 0).unwrap();
    let expected = [
        operand[0].wrapping_add(updates[0]).wrapping_add(updates[1]),
        operand[1].wrapping_add(updates[2]),
        operand[2],
    ];
    let mut raw = vec![MaybeUninit::<i32>::uninit(); 3];
    let mut out = ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &dims, &[1], 0).unwrap();
    plan.execute_uninit(
        &ExecContext::serial(),
        &mut out,
        &ErasedRawStridedPtr::from_ref(&source),
        &ErasedRawStridedPtr::from_ref(&index),
        &ErasedRawStridedPtr::from_ref(&update),
    )
    .unwrap();
    assert_initialized_eq(&raw, &expected);
}

macro_rules! scatter_dtype {
    ($name:ident, $ty:ty, $dtype:expr, $ity:ty, $idtype:expr, $values:expr, $updates:expr) => {
        #[test]
        fn $name() {
            let operand: Vec<$ty> = $values;
            let updates: Vec<$ty> = $updates;
            let dims = [3usize];
            let ids = [3usize, 1];
            let ud = [3usize];
            let indices = [0 as $ity, 0 as $ity, 1 as $ity];
            let spec = ScatterSpec {
                update_window_dims: vec![],
                inserted_window_dims: vec![0],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
            };
            let plan = ErasedScatterPlan::compile(
                $dtype,
                $idtype,
                &dims,
                &[1],
                &ids,
                &[1, 3],
                &ud,
                &[1],
                &dims,
                &[1],
                spec,
            )
            .unwrap();
            let source = ErasedRawStridedRef::from_slice(&operand, &dims, &[1], 0).unwrap();
            let index = ErasedRawStridedRef::from_slice(&indices, &ids, &[1, 3], 0).unwrap();
            let update = ErasedRawStridedRef::from_slice(&updates, &ud, &[1], 0).unwrap();
            let mut initialized = operand.clone();
            let mut initialized_dest =
                ErasedRawStridedMut::from_slice_mut(&mut initialized, &dims, &[1], 0).unwrap();
            plan.execute(
                &ExecContext::serial(),
                &mut initialized_dest,
                &source,
                &index,
                &update,
            )
            .unwrap();
            let mut raw = vec![MaybeUninit::<$ty>::uninit(); 3];
            for ctx in [
                ExecContext::serial(),
                ExecContext::max_threads(1).unwrap(),
                ExecContext::max_threads(2).unwrap(),
                ExecContext::max_threads(4).unwrap(),
            ] {
                let mut out =
                    ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &dims, &[1], 0).unwrap();
                plan.execute_uninit(
                    &ctx,
                    &mut out,
                    &ErasedRawStridedPtr::from_ref(&source),
                    &ErasedRawStridedPtr::from_ref(&index),
                    &ErasedRawStridedPtr::from_ref(&update),
                )
                .unwrap();
                assert_initialized_eq(&raw, &initialized);
            }
        }
    };
}

scatter_dtype!(
    scatter_f32_i32,
    f32,
    KernelDType::F32,
    i32,
    KernelDType::I32,
    vec![1.0, 2.0, 3.0],
    vec![10.0, 20.0, 30.0]
);
scatter_dtype!(
    scatter_f32_i64,
    f32,
    KernelDType::F32,
    i64,
    KernelDType::I64,
    vec![1.0, 2.0, 3.0],
    vec![10.0, 20.0, 30.0]
);
scatter_dtype!(
    scatter_f64_i32,
    f64,
    KernelDType::F64,
    i32,
    KernelDType::I32,
    vec![1.0, 2.0, 3.0],
    vec![10.0, 20.0, 30.0]
);
scatter_dtype!(
    scatter_f64_i64,
    f64,
    KernelDType::F64,
    i64,
    KernelDType::I64,
    vec![1.0, 2.0, 3.0],
    vec![10.0, 20.0, 30.0]
);
scatter_dtype!(
    scatter_c32_i32,
    Complex32,
    KernelDType::C32,
    i32,
    KernelDType::I32,
    vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0)
    ],
    vec![
        Complex32::new(10.0, 0.0),
        Complex32::new(20.0, 0.0),
        Complex32::new(30.0, 0.0)
    ]
);
scatter_dtype!(
    scatter_c32_i64,
    Complex32,
    KernelDType::C32,
    i64,
    KernelDType::I64,
    vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0)
    ],
    vec![
        Complex32::new(10.0, 0.0),
        Complex32::new(20.0, 0.0),
        Complex32::new(30.0, 0.0)
    ]
);
scatter_dtype!(
    scatter_c64_i32,
    Complex64,
    KernelDType::C64,
    i32,
    KernelDType::I32,
    vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0)
    ],
    vec![
        Complex64::new(10.0, 0.0),
        Complex64::new(20.0, 0.0),
        Complex64::new(30.0, 0.0)
    ]
);
scatter_dtype!(
    scatter_c64_i64,
    Complex64,
    KernelDType::C64,
    i64,
    KernelDType::I64,
    vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0)
    ],
    vec![
        Complex64::new(10.0, 0.0),
        Complex64::new(20.0, 0.0),
        Complex64::new(30.0, 0.0)
    ]
);
scatter_dtype!(
    scatter_i32_i32,
    i32,
    KernelDType::I32,
    i32,
    KernelDType::I32,
    vec![1, 2, 3],
    vec![10, 20, 30]
);
scatter_dtype!(
    scatter_i32_i64,
    i32,
    KernelDType::I32,
    i64,
    KernelDType::I64,
    vec![1, 2, 3],
    vec![10, 20, 30]
);
scatter_dtype!(
    scatter_i64_i32,
    i64,
    KernelDType::I64,
    i32,
    KernelDType::I32,
    vec![1, 2, 3],
    vec![10, 20, 30]
);
scatter_dtype!(
    scatter_i64_i64,
    i64,
    KernelDType::I64,
    i64,
    KernelDType::I64,
    vec![1, 2, 3],
    vec![10, 20, 30]
);

#[test]
fn scatter_integer_extrema_wrap_in_uninit_path() {
    macro_rules! case {
        ($ty:ty, $dtype:expr, $ity:ty, $idtype:expr) => {{
            let dims = [3usize];
            let ids = [3usize, 1];
            let operand = [<$ty>::MAX, 1, 2];
            let indices = [0 as $ity, 0 as $ity, 1 as $ity];
            let updates = [1 as $ty, 2 as $ty, <$ty>::MAX];
            let spec = ScatterSpec {
                update_window_dims: vec![],
                inserted_window_dims: vec![0],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
            };
            let plan = ErasedScatterPlan::compile(
                $dtype,
                $idtype,
                &dims,
                &[1],
                &ids,
                &[1, 3],
                &dims,
                &[1],
                &dims,
                &[1],
                spec,
            )
            .unwrap();
            let source = ErasedRawStridedRef::from_slice(&operand, &dims, &[1], 0).unwrap();
            let index = ErasedRawStridedRef::from_slice(&indices, &ids, &[1, 3], 0).unwrap();
            let update = ErasedRawStridedRef::from_slice(&updates, &dims, &[1], 0).unwrap();
            let expected = [
                operand[0].wrapping_add(updates[0]).wrapping_add(updates[1]),
                operand[1].wrapping_add(updates[2]),
                operand[2],
            ];
            for ctx in [
                ExecContext::serial(),
                ExecContext::max_threads(1).unwrap(),
                ExecContext::max_threads(2).unwrap(),
                ExecContext::max_threads(4).unwrap(),
            ] {
                let mut raw = vec![MaybeUninit::<$ty>::uninit(); 3];
                let mut out =
                    ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &dims, &[1], 0).unwrap();
                plan.execute_uninit(
                    &ctx,
                    &mut out,
                    &ErasedRawStridedPtr::from_ref(&source),
                    &ErasedRawStridedPtr::from_ref(&index),
                    &ErasedRawStridedPtr::from_ref(&update),
                )
                .unwrap();
                assert_initialized_eq(&raw, &expected);
            }
        }};
    }
    case!(i32, KernelDType::I32, i32, KernelDType::I32);
    case!(i64, KernelDType::I64, i64, KernelDType::I64);
}

#[test]
fn scatter_compile_rejects_bool_and_bad_layout_before_destination_use() {
    let spec = ScatterSpec {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    assert!(ErasedScatterPlan::compile(
        KernelDType::Bool,
        KernelDType::I32,
        &[2],
        &[1],
        &[1, 1],
        &[1, 1],
        &[2],
        &[1],
        &[2],
        &[1],
        spec.clone(),
    )
    .is_err());
    assert!(ErasedScatterPlan::compile(
        KernelDType::F32,
        KernelDType::I32,
        &[2],
        &[1],
        &[1, 1],
        &[1, 1],
        &[1],
        &[1],
        &[3],
        &[1],
        spec,
    )
    .is_err());
}

#[test]
fn aligned_uninit_lifecycle_all_indexed_families() {
    let dims = [4usize];
    let operand = [1.0f64, 2.0, 3.0, 4.0];
    let source = ErasedRawStridedRef::from_slice(&operand, &dims, &[1], 0).unwrap();
    let mut storage = vec![MaybeUninit::<f64>::uninit(); 4];

    let gather = ErasedGatherPlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &dims,
        &[1],
        &[2],
        &[1],
        &[2],
        &[1],
        GatherSpec {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1],
        },
    )
    .unwrap();
    let indices = [2i64, 0];
    let index = ErasedRawStridedRef::from_slice(&indices, &[2], &[1], 0).unwrap();
    let mut out =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut storage, &[2], &[1], 0).unwrap();
    gather
        .execute_uninit(
            &ExecContext::serial(),
            &mut out,
            &ErasedRawStridedPtr::from_ref(&source),
            &ErasedRawStridedPtr::from_ref(&index),
        )
        .unwrap();
    let _reachable = out.data_as_uninit_mut::<f64>().unwrap();

    let reduce = ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Sum, &dims, &[1]).unwrap();
    let mut scalar = vec![MaybeUninit::<f64>::uninit(); 1];
    let mut reduce_out =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut scalar, &[], &[], 0).unwrap();
    reduce
        .execute_uninit(
            &ExecContext::serial(),
            &mut reduce_out,
            &ErasedRawStridedPtr::from_ref(&source),
        )
        .unwrap();
    drop(reduce_out);
}

#[test]
fn uninit_lifecycle_executes_dynamic_and_scatter_families() {
    let dims = [4usize];
    let source_values = [1i32, 2, 3, 4];
    let source = ErasedRawStridedRef::from_slice(&source_values, &dims, &[1], 0).unwrap();
    let starts = [1i32];
    let starts_ref = ErasedRawStridedRef::from_slice(&starts, &[1], &[1], 0).unwrap();
    let slice = ErasedDynamicSlicePlan::compile(
        KernelDType::I32,
        KernelDType::I32,
        &dims,
        &[1],
        &[1],
        &[1],
        &[2],
        &[1],
        &[2],
    )
    .unwrap();
    let mut slice_storage = vec![MaybeUninit::<i32>::uninit(); 2];
    let mut slice_dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut slice_storage, &[2], &[1], 0).unwrap();
    slice
        .execute_uninit(
            &ExecContext::serial(),
            &mut slice_dest,
            &ErasedRawStridedPtr::from_ref(&source),
            &ErasedRawStridedPtr::from_ref(&starts_ref),
        )
        .unwrap();
    assert_initialized_eq(&slice_storage, &[2, 3]);

    let updates = [9i32, 8];
    let updates_ref = ErasedRawStridedRef::from_slice(&updates, &[2], &[1], 0).unwrap();
    let update = ErasedDynamicUpdateSlicePlan::compile(
        KernelDType::I32,
        KernelDType::I32,
        &dims,
        &[1],
        &[1],
        &[1],
        &[2],
        &[1],
        &dims,
        &[1],
    )
    .unwrap();
    let mut update_storage = vec![MaybeUninit::<i32>::uninit(); 4];
    let mut update_dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut update_storage, &dims, &[1], 0).unwrap();
    update
        .execute_uninit(
            &ExecContext::serial(),
            &mut update_dest,
            &ErasedRawStridedPtr::from_ref(&source),
            &ErasedRawStridedPtr::from_ref(&updates_ref),
            &ErasedRawStridedPtr::from_ref(&starts_ref),
        )
        .unwrap();
    assert_initialized_eq(&update_storage, &[1, 9, 8, 4]);

    let spec = ScatterSpec {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    let scatter = ErasedScatterPlan::compile(
        KernelDType::I32,
        KernelDType::I32,
        &dims,
        &[1],
        &[2, 1],
        &[1, 2],
        &[2],
        &[1],
        &dims,
        &[1],
        spec,
    )
    .unwrap();
    let scatter_indices = [1i32, 3];
    let scatter_updates = [7i32, 6];
    let scatter_index =
        ErasedRawStridedRef::from_slice(&scatter_indices, &[2, 1], &[1, 2], 0).unwrap();
    let scatter_update = ErasedRawStridedRef::from_slice(&scatter_updates, &[2], &[1], 0).unwrap();
    let mut scatter_storage = vec![MaybeUninit::<i32>::uninit(); 4];
    let mut scatter_dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut scatter_storage, &dims, &[1], 0).unwrap();
    scatter
        .execute_uninit(
            &ExecContext::max_threads(2).unwrap(),
            &mut scatter_dest,
            &ErasedRawStridedPtr::from_ref(&source),
            &ErasedRawStridedPtr::from_ref(&scatter_index),
            &ErasedRawStridedPtr::from_ref(&scatter_update),
        )
        .unwrap();
    assert_initialized_eq(&scatter_storage, &[1, 9, 3, 10]);
}

#[test]
fn dynamic_update_uninitialized_holes_are_never_read() {
    let dims = [3usize];
    let source_values = [1i32, 0, 2, 0, 3];
    let updates = [9i32];
    let starts = [1i32];
    let plan = ErasedDynamicUpdateSlicePlan::compile(
        KernelDType::I32,
        KernelDType::I32,
        &dims,
        &[2],
        &[1],
        &[1],
        &[1],
        &[1],
        &dims,
        &[2],
    )
    .unwrap();
    let source = ErasedRawStridedRef::from_slice(&source_values, &dims, &[2], 0).unwrap();
    let update = ErasedRawStridedRef::from_slice(&updates, &[1], &[1], 0).unwrap();
    let start = ErasedRawStridedRef::from_slice(&starts, &[1], &[1], 0).unwrap();
    let mut storage = vec![MaybeUninit::<i32>::uninit(); 5];
    let mut dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut storage, &dims, &[2], 0).unwrap();
    plan.execute_uninit(
        &ExecContext::serial(),
        &mut dest,
        &ErasedRawStridedPtr::from_ref(&source),
        &ErasedRawStridedPtr::from_ref(&update),
        &ErasedRawStridedPtr::from_ref(&start),
    )
    .unwrap();
    assert_eq!(unsafe { storage[0].assume_init_ref() }, &1);
    assert_eq!(unsafe { storage[2].assume_init_ref() }, &9);
    assert_eq!(unsafe { storage[4].assume_init_ref() }, &3);
}

#[test]
fn above_threshold_parallel_indexed_replays_match_serial_initialized() {
    let n = 131_073usize;
    let dims = [n];
    let operand: Vec<f64> = (0..n).map(|i| i as f64 * 0.25).collect();
    let source = ErasedRawStridedRef::from_slice(&operand, &dims, &[1], 0).unwrap();
    let serial = ExecContext::serial();
    let contexts = [
        ExecContext::max_threads(2).unwrap(),
        ExecContext::max_threads(4).unwrap(),
    ];

    let indices: Vec<i64> = (0..n).map(|i| ((i * 7) % n) as i64).collect();
    let index_dims = [n];
    let gather = ErasedGatherPlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &dims,
        &[1],
        &index_dims,
        &[1],
        &dims,
        &[1],
        GatherSpec {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1],
        },
    )
    .unwrap();
    let index = ErasedRawStridedRef::from_slice(&indices, &index_dims, &[1], 0).unwrap();
    let mut expected = vec![0.0f64; n];
    let mut init = ErasedRawStridedMut::from_slice_mut(&mut expected, &dims, &[1], 0).unwrap();
    gather.execute(&serial, &mut init, &source, &index).unwrap();
    for ctx in contexts {
        let mut raw = vec![MaybeUninit::<f64>::uninit(); n];
        let mut out =
            ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &dims, &[1], 0).unwrap();
        gather
            .execute_uninit(
                &ctx,
                &mut out,
                &ErasedRawStridedPtr::from_ref(&source),
                &ErasedRawStridedPtr::from_ref(&index),
            )
            .unwrap();
        assert_initialized_eq(out.data_as_uninit_mut::<f64>().unwrap(), &expected);
    }

    let starts = [n as i64 / 4];
    let start_dims = [1usize];
    let slice_dims = [n / 2];
    let slice = ErasedDynamicSlicePlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &dims,
        &[1],
        &start_dims,
        &[1],
        &slice_dims,
        &[1],
        &[n / 2],
    )
    .unwrap();
    let start = ErasedRawStridedRef::from_slice(&starts, &start_dims, &[1], 0).unwrap();
    let mut slice_expected = vec![0.0f64; n / 2];
    let mut slice_init =
        ErasedRawStridedMut::from_slice_mut(&mut slice_expected, &slice_dims, &[1], 0).unwrap();
    slice
        .execute(&serial, &mut slice_init, &source, &start)
        .unwrap();
    for ctx in [
        ExecContext::max_threads(2).unwrap(),
        ExecContext::max_threads(4).unwrap(),
    ] {
        let mut raw = vec![MaybeUninit::<f64>::uninit(); n / 2];
        let mut out =
            ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &slice_dims, &[1], 0).unwrap();
        slice
            .execute_uninit(
                &ctx,
                &mut out,
                &ErasedRawStridedPtr::from_ref(&source),
                &ErasedRawStridedPtr::from_ref(&start),
            )
            .unwrap();
        assert_initialized_eq(out.data_as_uninit_mut::<f64>().unwrap(), &slice_expected);
    }

    let update_values: Vec<f64> = (0..n / 2).map(|i| i as f64).collect();
    let update_dims = [n / 2];
    let update_ref =
        ErasedRawStridedRef::from_slice(&update_values, &update_dims, &[1], 0).unwrap();
    let update_plan = ErasedDynamicUpdateSlicePlan::compile(
        KernelDType::F64,
        KernelDType::I64,
        &dims,
        &[1],
        &start_dims,
        &[1],
        &update_dims,
        &[1],
        &dims,
        &[1],
    )
    .unwrap();
    let mut update_expected = operand.clone();
    let mut update_init =
        ErasedRawStridedMut::from_slice_mut(&mut update_expected, &dims, &[1], 0).unwrap();
    update_plan
        .execute(&serial, &mut update_init, &source, &update_ref, &start)
        .unwrap();
    for ctx in [
        ExecContext::max_threads(2).unwrap(),
        ExecContext::max_threads(4).unwrap(),
    ] {
        let mut raw = vec![MaybeUninit::<f64>::uninit(); n];
        let mut out =
            ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &dims, &[1], 0).unwrap();
        update_plan
            .execute_uninit(
                &ctx,
                &mut out,
                &ErasedRawStridedPtr::from_ref(&source),
                &ErasedRawStridedPtr::from_ref(&update_ref),
                &ErasedRawStridedPtr::from_ref(&start),
            )
            .unwrap();
        assert_initialized_eq(out.data_as_uninit_mut::<f64>().unwrap(), &update_expected);
    }

    let axis_plan = ErasedReducePlan::compile_axes(
        KernelDType::F64,
        ReduceOp::Sum,
        &[n, 2],
        &[1, n as isize],
        &[n],
        &[1],
        &[1],
    )
    .unwrap();
    let axis_input: Vec<f64> = (0..n * 2).map(|i| i as f64).collect();
    let axis_dims = [n, 2];
    let axis_strides = [1isize, n as isize];
    let axis_dest_dims = [n];
    let axis_dest_strides = [1isize];
    let axis_source =
        ErasedRawStridedRef::from_slice(&axis_input, &axis_dims, &axis_strides, 0).unwrap();
    let mut axis_expected = vec![0.0f64; n];
    let mut axis_init = ErasedRawStridedMut::from_slice_mut(
        &mut axis_expected,
        &axis_dest_dims,
        &axis_dest_strides,
        0,
    )
    .unwrap();
    axis_plan
        .execute(&serial, &mut axis_init, &axis_source)
        .unwrap();
    for ctx in [
        ExecContext::max_threads(2).unwrap(),
        ExecContext::max_threads(4).unwrap(),
    ] {
        let mut raw = vec![MaybeUninit::<f64>::uninit(); n];
        let mut out = ErasedRawStridedUninitMut::from_uninit_slice(
            &mut raw,
            &axis_dest_dims,
            &axis_dest_strides,
            0,
        )
        .unwrap();
        axis_plan
            .execute_uninit(&ctx, &mut out, &ErasedRawStridedPtr::from_ref(&axis_source))
            .unwrap();
        assert_initialized_eq(out.data_as_uninit_mut::<f64>().unwrap(), &axis_expected);
    }
}
