use core::mem::MaybeUninit;
use num_complex::{Complex32, Complex64};
use strided_kernel::{
    ErasedRawStridedMut, ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut,
    ErasedReducePlan, ExecContext, KernelDType, ReduceOp,
};

fn bytes<T>(value: &[T]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(value.as_ptr().cast(), core::mem::size_of_val(value)) }
}

fn bytes_mut<T>(value: &mut [T]) -> &mut [u8] {
    unsafe {
        core::slice::from_raw_parts_mut(value.as_mut_ptr().cast(), core::mem::size_of_val(value))
    }
}

fn uninit_bytes_mut(value: &mut [MaybeUninit<u8>]) -> &mut [u8] {
    unsafe { core::slice::from_raw_parts_mut(value.as_mut_ptr().cast(), value.len()) }
}

macro_rules! differential {
    ($name:ident, $ty:ty, $dtype:expr, $values:expr) => {
        #[test]
        fn $name() {
            let input: Vec<$ty> = $values;
            let dims = [2usize, input.len() / 2];
            let strides = [1isize, 2];
            let plan = ErasedReducePlan::compile($dtype, ReduceOp::Sum, &dims, &strides).unwrap();
            let source =
                ErasedRawStridedRef::new($dtype, bytes(&input), &dims, &strides, 0).unwrap();
            let contexts = [
                ExecContext::serial(),
                ExecContext::max_threads(1).unwrap(),
                ExecContext::max_threads(2).unwrap(),
                ExecContext::max_threads(4).unwrap(),
            ];
            for ctx in contexts {
                let mut expected = [<$ty as Default>::default()];
                let mut initialized =
                    ErasedRawStridedMut::new($dtype, bytes_mut(&mut expected), &[], &[], 0)
                        .unwrap();
                plan.execute(&ctx, &mut initialized, &source).unwrap();

                let mut raw = vec![MaybeUninit::new(0xa5u8); core::mem::size_of::<$ty>()];
                let mut uninit =
                    ErasedRawStridedUninitMut::new($dtype, &mut raw, &[], &[], 0).unwrap();
                let source_ptr = ErasedRawStridedPtr::from_ref(&source);
                plan.execute_uninit(&ctx, &mut uninit, &source_ptr).unwrap();
                assert_eq!(uninit_bytes_mut(&mut raw), bytes(&expected));
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
    let src_strides = [1isize, 2];
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
    let source =
        ErasedRawStridedRef::new(KernelDType::F64, bytes(&input), &src_dims, &src_strides, 0)
            .unwrap();
    let mut expected = [0.0f64; 4];
    expected[0] = input[0] * input[0] + input[2] * input[2] + input[4] * input[4];
    expected[2] = input[1] * input[1] + input[3] * input[3] + input[5] * input[5];
    let mut initialized = ErasedRawStridedMut::new(
        KernelDType::F64,
        bytes_mut(&mut expected),
        &dest_dims,
        &dest_strides,
        0,
    )
    .unwrap();
    plan.execute(&ExecContext::serial(), &mut initialized, &source)
        .unwrap();

    let mut raw = vec![MaybeUninit::new(0x5au8); 4 * core::mem::size_of::<f64>()];
    let before = raw.clone();
    let mut uninit =
        ErasedRawStridedUninitMut::new(KernelDType::F64, &mut raw, &dest_dims, &dest_strides, 0)
            .unwrap();
    let source_ptr = ErasedRawStridedPtr::from_ref(&source);
    plan.execute_uninit(
        &ExecContext::max_threads(4).unwrap(),
        &mut uninit,
        &source_ptr,
    )
    .unwrap();
    assert_eq!(
        &uninit_bytes_mut(&mut raw)[8..16],
        &uninit_bytes_mut(&mut before.clone())[8..16]
    );
    assert_eq!(
        &uninit_bytes_mut(&mut raw)[24..32],
        &uninit_bytes_mut(&mut before.clone())[24..32]
    );
}

#[test]
fn validation_errors_leave_uninitialized_bytes_untouched() {
    let input = [1.0f64, 2.0];
    let dims = [2usize];
    let source = ErasedRawStridedRef::new(KernelDType::F64, bytes(&input), &dims, &[1], 0).unwrap();
    let plan = ErasedReducePlan::compile(KernelDType::F64, ReduceOp::Sum, &dims, &[1]).unwrap();
    let mut raw = vec![MaybeUninit::new(0xa5u8); core::mem::size_of::<f64>()];
    let before = raw.clone();
    let mut dest = ErasedRawStridedUninitMut::new(KernelDType::F64, &mut raw, &[], &[], 0).unwrap();
    let wrong =
        ErasedRawStridedRef::new(KernelDType::I32, bytes(&[1i32, 2]), &dims, &[1], 0).unwrap();
    assert!(plan
        .execute_uninit(
            &ExecContext::serial(),
            &mut dest,
            &ErasedRawStridedPtr::from_ref(&wrong)
        )
        .is_err());
    assert_eq!(
        uninit_bytes_mut(dest.data_mut()),
        uninit_bytes_mut(&mut before.clone())
    );
    assert!(plan
        .execute_uninit(
            &ExecContext::serial(),
            &mut dest,
            &ErasedRawStridedPtr::from_ref(&source)
        )
        .is_ok());
}
