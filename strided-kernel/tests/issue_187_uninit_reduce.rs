use core::fmt::Debug;
use core::mem::MaybeUninit;
use num_complex::{Complex32, Complex64};
use strided_kernel::{
    ErasedRawStridedMut, ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut,
    ErasedReducePlan, ExecContext, KernelDType, KernelStorageElement, ReduceOp,
};

fn assert_uninit_replay<T>(input: Vec<T>, dtype: KernelDType, op: ReduceOp, expected: T)
where
    T: KernelStorageElement + Default + PartialEq + Debug,
{
    let dims = [input.len()];
    let strides = [1isize];
    let plan = ErasedReducePlan::compile(dtype, op, &dims, &strides).unwrap();
    let source = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let mut initialized = [T::default()];
    let mut initialized_dest =
        ErasedRawStridedMut::from_slice_mut(&mut initialized, &[], &[], 0).unwrap();
    plan.execute(&ExecContext::serial(), &mut initialized_dest, &source)
        .unwrap();
    assert_eq!(initialized[0], expected);

    let mut uninitialized = vec![MaybeUninit::new(T::default()); 1];
    let mut uninitialized_dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut uninitialized, &[], &[], 0).unwrap();
    plan.execute_uninit(
        &ExecContext::serial(),
        &mut uninitialized_dest,
        &ErasedRawStridedPtr::from_ref(&source),
    )
    .unwrap();
    assert_eq!(
        unsafe { uninitialized[0].assume_init_ref() },
        &initialized[0]
    );
}

macro_rules! differential {
    ($name:ident, $ty:ty, $dtype:expr, $values:expr) => {
        #[test]
        fn $name() {
            let input: Vec<$ty> = $values;
            let dims = [2usize, input.len() / 2];
            let strides = [1isize, 2];
            let plan = ErasedReducePlan::compile($dtype, ReduceOp::Sum, &dims, &strides).unwrap();
            let source = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
            let contexts = [
                ExecContext::serial(),
                ExecContext::max_threads(1).unwrap(),
                ExecContext::max_threads(2).unwrap(),
                ExecContext::max_threads(4).unwrap(),
            ];
            for ctx in contexts {
                let mut expected = [<$ty as Default>::default()];
                let mut initialized =
                    ErasedRawStridedMut::from_slice_mut(&mut expected, &[], &[], 0).unwrap();
                plan.execute(&ctx, &mut initialized, &source).unwrap();

                let mut raw = vec![MaybeUninit::<$ty>::uninit(); 1];
                let mut uninit =
                    ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &[], &[], 0).unwrap();
                let source_ptr = ErasedRawStridedPtr::from_ref(&source);
                plan.execute_uninit(&ctx, &mut uninit, &source_ptr).unwrap();
                assert_eq!(unsafe { raw[0].assume_init_ref() }, &expected[0]);
            }
        }
    };
}

differential!(f32_full, f32, KernelDType::F32, vec![1.0, -2.0, 3.0, 4.0]);
differential!(
    f64_full,
    f64,
    KernelDType::F64,
    vec![1.0e16, 1.0, 1.0, -1.0e16, -0.0, 0.0]
);
differential!(i32_full, i32, KernelDType::I32, vec![i32::MAX, 1, -3, 4]);
differential!(i64_full, i64, KernelDType::I64, vec![i64::MAX, 2, -3, 4]);
differential!(
    c32_full,
    Complex32,
    KernelDType::C32,
    vec![
        Complex32::new(1.0, 2.0),
        Complex32::new(-2.0, 1.0),
        Complex32::new(3.0, -1.0),
        Complex32::new(4.0, 0.5)
    ]
);
differential!(
    c64_full,
    Complex64,
    KernelDType::C64,
    vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(-2.0, 1.0),
        Complex64::new(3.0, -1.0),
        Complex64::new(4.0, 0.5)
    ]
);

#[test]
fn axis_holes_negative_stride_and_identity_match() {
    let input = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let src_dims = [2usize, 3];
    let src_strides = [1isize, -2];
    let dest_dims = [2usize];
    let dest_strides = [2isize];
    let plan = ErasedReducePlan::compile_axes(
        KernelDType::F64,
        ReduceOp::SumSquares,
        &src_dims,
        &src_strides,
        &dest_dims,
        &dest_strides,
        &[1],
    )
    .unwrap();
    let source = ErasedRawStridedRef::from_slice(&input, &src_dims, &src_strides, 4).unwrap();
    let mut expected = [0.0f64; 4];
    expected[0] = input[4] * input[4] + input[2] * input[2] + input[0] * input[0];
    expected[2] = input[5] * input[5] + input[3] * input[3] + input[1] * input[1];
    let mut initialized =
        ErasedRawStridedMut::from_slice_mut(&mut expected, &dest_dims, &dest_strides, 0).unwrap();
    plan.execute(&ExecContext::serial(), &mut initialized, &source)
        .unwrap();

    let mut raw = vec![MaybeUninit::<f64>::new(0x5a as f64); 4];
    let before = raw.clone();
    let mut uninit =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &dest_dims, &dest_strides, 0)
            .unwrap();
    let source_ptr = ErasedRawStridedPtr::from_ref(&source);
    plan.execute_uninit(
        &ExecContext::max_threads(4).unwrap(),
        &mut uninit,
        &source_ptr,
    )
    .unwrap();
    assert_eq!(unsafe { raw[0].assume_init_ref() }, &expected[0]);
    assert_eq!(unsafe { raw[2].assume_init_ref() }, &expected[2]);
    assert_eq!(unsafe { raw[1].assume_init_ref() }, unsafe {
        before[1].assume_init_ref()
    });
    assert_eq!(unsafe { raw[3].assume_init_ref() }, unsafe {
        before[3].assume_init_ref()
    });
}

#[test]
fn uninit_reduce_empty_product_uses_identity() {
    assert_uninit_replay(Vec::<f64>::new(), KernelDType::F64, ReduceOp::Product, 1.0);
}

#[test]
fn uninit_reduce_product_and_nonfinite_match_initialized_replay() {
    assert_uninit_replay(
        vec![2.0f64, 0.5, 2.0, 0.5],
        KernelDType::F64,
        ReduceOp::Product,
        1.0,
    );
    let input = vec![f64::INFINITY, 2.0, f64::NEG_INFINITY];
    let dims = [input.len()];
    let source = ErasedRawStridedRef::from_slice(&input, &dims, &[1], 0).unwrap();
    let plan = ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Sum, &dims, &[1]).unwrap();
    let mut initialized = [0.0f64];
    let mut initialized_dest =
        ErasedRawStridedMut::from_slice_mut(&mut initialized, &[], &[], 0).unwrap();
    plan.execute(&ExecContext::serial(), &mut initialized_dest, &source)
        .unwrap();
    let mut uninitialized = vec![MaybeUninit::new(0.0f64)];
    let mut uninitialized_dest =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut uninitialized, &[], &[], 0).unwrap();
    plan.execute_uninit(
        &ExecContext::serial(),
        &mut uninitialized_dest,
        &ErasedRawStridedPtr::from_ref(&source),
    )
    .unwrap();
    assert!(initialized[0].is_nan());
    assert!(unsafe { *uninitialized[0].assume_init_ref() }.is_nan());
}

#[test]
fn uninit_reduce_simd_tail_and_sum_squares_match_initialized_replay() {
    let input = (0..65)
        .map(|i| if i % 2 == 0 { 2.0 } else { 0.5 })
        .collect::<Vec<f64>>();
    assert_uninit_replay(input, KernelDType::F64, ReduceOp::Product, 2.0);
    let input = (0..17).map(|i| i as f64 - 8.0).collect::<Vec<f64>>();
    let expected = input.iter().map(|value| value * value).sum();
    assert_uninit_replay(input, KernelDType::F64, ReduceOp::SumSquares, expected);
}

#[test]
fn validation_errors_leave_uninitialized_bytes_untouched() {
    let input = [1.0f64, 2.0];
    let dims = [2usize];
    let source = ErasedRawStridedRef::from_slice(&input, &dims, &[1], 0).unwrap();
    let plan = ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Sum, &dims, &[1]).unwrap();
    let mut raw = vec![MaybeUninit::<f64>::new(0xa5 as f64); 1];
    let before = raw.clone();
    let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(&mut raw, &[], &[], 0).unwrap();
    let wrong = ErasedRawStridedRef::from_slice(&[1i32, 2], &dims, &[1], 0).unwrap();
    assert!(plan
        .execute_uninit(
            &ExecContext::serial(),
            &mut dest,
            &ErasedRawStridedPtr::from_ref(&wrong)
        )
        .is_err());
    assert_eq!(
        unsafe { dest.data_as_uninit_mut::<f64>().unwrap()[0].assume_init_ref() },
        unsafe { before[0].assume_init_ref() }
    );
    assert!(plan
        .execute_uninit(
            &ExecContext::serial(),
            &mut dest,
            &ErasedRawStridedPtr::from_ref(&source)
        )
        .is_ok());
}
