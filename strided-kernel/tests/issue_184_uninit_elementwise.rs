use core::mem::MaybeUninit;
use num_complex::{Complex32, Complex64};

use strided_kernel::{
    batched_outer_product_into_uninit, broadcast_mul_into_uninit, compare_into_uninit,
    mul_into_uninit, CompareOp, ErasedFusedPlan, ErasedRawStridedRef, ErasedRawStridedUninitMut,
    ExecContext, FusedInst, FusedOp, FusedPlan, Identity, KernelDType, StridedView, StridedViewMut,
};

fn uninit_view<'a, T>(
    data: &'a mut [MaybeUninit<T>],
    dims: &[usize],
) -> StridedViewMut<'a, MaybeUninit<T>> {
    StridedViewMut::new(data, dims, &[1], 0).unwrap()
}

fn assume_init<T>(data: Vec<MaybeUninit<T>>) -> Vec<T> {
    let mut data = core::mem::ManuallyDrop::new(data);
    unsafe { Vec::from_raw_parts(data.as_mut_ptr().cast(), data.len(), data.capacity()) }
}

fn assert_mul_dtype<T>(lhs: &[T], rhs: &[T], expected: &[T])
where
    T: Copy
        + core::fmt::Debug
        + PartialEq
        + core::ops::Mul<Output = T>
        + strided_kernel::MaybeSendSync
        + 'static,
{
    let dims = [lhs.len()];
    let lhs_view = StridedView::<_, Identity>::new(lhs, &dims, &[1], 0).unwrap();
    let rhs_view = StridedView::<_, Identity>::new(rhs, &dims, &[1], 0).unwrap();
    let mut output = vec![MaybeUninit::uninit(); lhs.len()];
    mul_into_uninit(&mut uninit_view(&mut output, &dims), &lhs_view, &rhs_view).unwrap();
    assert_eq!(assume_init(output), expected);
}

fn as_bytes<T>(data: &[T]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(data.as_ptr().cast(), core::mem::size_of_val(data)) }
}

fn as_uninit_bytes<T>(data: &mut [MaybeUninit<T>]) -> &mut [MaybeUninit<u8>] {
    unsafe {
        core::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), core::mem::size_of_val(data))
    }
}

#[test]
fn typed_uninitialized_full_overwrite_siblings_match_initialized_paths() {
    let lhs = [1i32, 2, 3, 4];
    let rhs = [5i32, 6, 7, 8];
    let lhs_view = StridedView::<_, Identity>::new(&lhs, &[4], &[1], 0).unwrap();
    let rhs_view = StridedView::<_, Identity>::new(&rhs, &[4], &[1], 0).unwrap();

    let mut mul_out = vec![MaybeUninit::uninit(); 4];
    mul_into_uninit(&mut uninit_view(&mut mul_out, &[4]), &lhs_view, &rhs_view).unwrap();
    assert_eq!(assume_init(mul_out), [5, 12, 21, 32]);

    let mut compare_out = vec![MaybeUninit::uninit(); 4];
    compare_into_uninit(
        &mut uninit_view(&mut compare_out, &[4]),
        &lhs_view,
        &rhs_view,
        CompareOp::Lt,
    )
    .unwrap();
    assert_eq!(assume_init(compare_out), [true; 4]);

    let mut broadcast_out = vec![MaybeUninit::uninit(); 4];
    broadcast_mul_into_uninit(
        &mut uninit_view(&mut broadcast_out, &[4]),
        &lhs_view,
        &[0],
        &rhs_view,
        &[0],
    )
    .unwrap();
    assert_eq!(assume_init(broadcast_out), [5, 12, 21, 32]);

    let lhs_outer = [2i32, 3];
    let rhs_outer = [10i32, 20, 30];
    let lhs_outer_view = StridedView::<_, Identity>::new(&lhs_outer, &[2], &[1], 0).unwrap();
    let rhs_outer_view = StridedView::<_, Identity>::new(&rhs_outer, &[3], &[1], 0).unwrap();
    let mut outer_out = vec![MaybeUninit::uninit(); 6];
    let mut outer_view = StridedViewMut::new(&mut outer_out, &[2, 3], &[1, 2], 0).unwrap();
    batched_outer_product_into_uninit(&mut outer_view, &lhs_outer_view, &rhs_outer_view, 1, 1)
        .unwrap();
    assert_eq!(assume_init(outer_out), [20, 30, 40, 60, 60, 90]);

    assert_mul_dtype(&[1.0f32, -2.0], &[3.0, 4.0], &[3.0, -8.0]);
    assert_mul_dtype(&[1.0f64, -2.0], &[3.0, 4.0], &[3.0, -8.0]);
    assert_mul_dtype(&[3i64, -4], &[5, 6], &[15, -24]);
    assert_mul_dtype(
        &[Complex32::new(1.0, 2.0), Complex32::new(3.0, -1.0)],
        &[Complex32::new(2.0, -1.0), Complex32::new(0.5, 2.0)],
        &[Complex32::new(4.0, 3.0), Complex32::new(3.5, 5.5)],
    );
    assert_mul_dtype(
        &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)],
        &[Complex64::new(2.0, -1.0), Complex64::new(0.5, 2.0)],
        &[Complex64::new(4.0, 3.0), Complex64::new(3.5, 5.5)],
    );
}

#[test]
fn erased_fused_uninitialized_replay_writes_valid_bool_output() {
    let lhs = [true, false, true, false];
    let dims = [4usize];
    let strides = [1isize];
    let plan = ErasedFusedPlan::compile(
        KernelDType::Bool,
        FusedPlan {
            input_count: 1,
            outputs: vec![1],
            ops: vec![FusedInst {
                op: FusedOp::Conj,
                inputs: vec![0],
            }],
        },
    )
    .unwrap();
    let inputs = [
        ErasedRawStridedRef::new(KernelDType::Bool, as_bytes(&lhs), &dims, &strides, 0).unwrap(),
    ];
    let mut output = vec![MaybeUninit::<bool>::uninit(); 4];
    let mut dest = ErasedRawStridedUninitMut::new(
        KernelDType::Bool,
        as_uninit_bytes(&mut output),
        &dims,
        &strides,
        0,
    )
    .unwrap();
    plan.execute_uninit(&ExecContext::max_threads(1).unwrap(), &mut dest, &inputs)
        .unwrap();
    assert_eq!(assume_init(output), lhs);
}

#[test]
fn typed_uninitialized_validation_errors_leave_sentinels_untouched() {
    let lhs = [2i32, 3];
    let rhs = [5i32, 7];
    let lhs_view = StridedView::<_, Identity>::new(&lhs, &[2], &[1], 0).unwrap();
    let rhs_view = StridedView::<_, Identity>::new(&rhs, &[2], &[1], 0).unwrap();
    let sentinel = MaybeUninit::new(99i32);
    let mut output = vec![sentinel; 2];
    let err = mul_into_uninit(
        &mut StridedViewMut::new(&mut output, &[2], &[0], 0).unwrap(),
        &lhs_view,
        &rhs_view,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        strided_kernel::StridedError::NonInjectiveOutputLayout
    ));
    assert!(output
        .iter()
        .all(|value| unsafe { value.assume_init() == 99 }));
}

#[cfg(feature = "parallel")]
#[test]
fn erased_fused_uninitialized_parallel_replay_preserves_wrapping_integer_semantics() {
    let len = 32 * 1024;
    let lhs = vec![i32::MAX; len];
    let rhs = vec![2i32; len];
    let dims = [len];
    let strides = [1isize];
    let plan = ErasedFusedPlan::compile(
        KernelDType::I32,
        FusedPlan {
            input_count: 2,
            outputs: vec![2],
            ops: vec![FusedInst {
                op: FusedOp::Multiply,
                inputs: vec![0, 1],
            }],
        },
    )
    .unwrap();
    let inputs = [
        ErasedRawStridedRef::new(KernelDType::I32, as_bytes(&lhs), &dims, &strides, 0).unwrap(),
        ErasedRawStridedRef::new(KernelDType::I32, as_bytes(&rhs), &dims, &strides, 0).unwrap(),
    ];
    let mut output = vec![MaybeUninit::<i32>::uninit(); len];
    let mut dest = ErasedRawStridedUninitMut::new(
        KernelDType::I32,
        as_uninit_bytes(&mut output),
        &dims,
        &strides,
        0,
    )
    .unwrap();
    plan.execute_uninit(&ExecContext::max_threads(2).unwrap(), &mut dest, &inputs)
        .unwrap();
    assert!(output
        .iter()
        .all(|value| unsafe { value.assume_init() == -2 }));
}
