use core::fmt::Debug;

use num_complex::{Complex32, Complex64};
use strided_kernel::{
    ErasedCopyPlan, ErasedRawStridedMut, ErasedRawStridedRef, ExecContext, KernelDType,
    StridedError,
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
fn erased_copy_plan_executes_f64_transposed_layout() {
    let dims = [2usize, 3];
    let src_strides = [3isize, 1];
    let dst_strides = [1isize, 2];
    let src = [0.0f64, 1.0, 2.0, 10.0, 11.0, 12.0];
    let mut dst = [0.0f64; 6];

    let plan =
        ErasedCopyPlan::compile(KernelDType::F64, &dims, &dst_strides, &src_strides).unwrap();
    {
        let source =
            ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&src), &dims, &src_strides, 0)
                .unwrap();
        let mut dest = ErasedRawStridedMut::new(
            KernelDType::F64,
            as_bytes_mut(&mut dst),
            &dims,
            &dst_strides,
            0,
        )
        .unwrap();

        plan.execute(&ExecContext::serial(), &mut dest, &source)
            .unwrap();
    }

    assert_eq!(dst, [0.0, 10.0, 1.0, 11.0, 2.0, 12.0]);
}

#[test]
fn erased_copy_plan_rejects_dtype_mismatch() {
    let dims = [2usize];
    let strides = [1isize];
    let src = [1.0f64, 2.0];
    let mut dst = [0.0f32; 2];

    let plan = ErasedCopyPlan::compile(KernelDType::F64, &dims, &strides, &strides).unwrap();
    let source =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&src), &dims, &strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::new(KernelDType::F32, as_bytes_mut(&mut dst), &dims, &strides, 0)
            .unwrap();

    let err = plan
        .execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap_err();
    assert!(matches!(err, StridedError::DTypeMismatch { .. }));
}

fn assert_supported_copy<T>(dtype: KernelDType, input: &[T])
where
    T: Copy + Debug + Default + PartialEq,
{
    let dims = [input.len()];
    let strides = [1isize];
    let mut output = vec![T::default(); input.len()];

    let plan = ErasedCopyPlan::compile(dtype, &dims, &strides, &strides).unwrap();
    {
        let source = ErasedRawStridedRef::new(dtype, as_bytes(input), &dims, &strides, 0).unwrap();
        let mut dest =
            ErasedRawStridedMut::new(dtype, as_bytes_mut(&mut output), &dims, &strides, 0).unwrap();
        plan.execute(&ExecContext::serial(), &mut dest, &source)
            .unwrap();
    }

    assert_eq!(output, input);
}

#[test]
fn erased_copy_plan_executes_supported_scalar_set() {
    assert_supported_copy(KernelDType::F32, &[1.0f32, -2.0, 3.5]);
    assert_supported_copy(KernelDType::F64, &[1.0f64, -2.0, 3.5]);
    assert_supported_copy(KernelDType::I32, &[1i32, -2, 3]);
    assert_supported_copy(KernelDType::I64, &[1i64, -2, 3]);
    assert_supported_copy(KernelDType::Bool, &[true, false, true]);
    assert_supported_copy(
        KernelDType::C32,
        &[Complex32::new(1.0, -2.0), Complex32::new(3.5, 4.0)],
    );
    assert_supported_copy(
        KernelDType::C64,
        &[Complex64::new(1.0, -2.0), Complex64::new(3.5, 4.0)],
    );
}

#[test]
fn erased_raw_descriptors_reject_invalid_byte_layouts() {
    let dims = [1usize];
    let strides = [1isize];
    let bytes = [0u8; 9];

    let err = ErasedRawStridedRef::new(KernelDType::F64, &bytes, &dims, &strides, 0).unwrap_err();
    assert!(matches!(err, StridedError::ByteLengthMismatch { .. }));

    let aligned = [0.0f64; 2];
    let aligned_bytes = as_bytes(&aligned);
    let misaligned = &aligned_bytes[1..1 + core::mem::size_of::<f64>()];
    let err =
        ErasedRawStridedRef::new(KernelDType::F64, misaligned, &dims, &strides, 0).unwrap_err();
    assert!(matches!(err, StridedError::DataAlignmentMismatch { .. }));

    let invalid_bool = [2u8];
    let err =
        ErasedRawStridedRef::new(KernelDType::Bool, &invalid_bool, &dims, &strides, 0).unwrap_err();
    assert!(matches!(err, StridedError::InvalidBoolByte { value: 2 }));
}

#[test]
fn erased_copy_plan_revalidates_bool_output_bytes_before_replay() {
    let dims = [1usize];
    let strides = [1isize];
    let source_bytes = [1u8];
    let mut dest_bytes = [0u8];

    let plan = ErasedCopyPlan::compile(KernelDType::Bool, &dims, &strides, &strides).unwrap();
    let source =
        ErasedRawStridedRef::new(KernelDType::Bool, &source_bytes, &dims, &strides, 0).unwrap();
    let mut dest =
        ErasedRawStridedMut::new(KernelDType::Bool, &mut dest_bytes, &dims, &strides, 0).unwrap();

    dest.data_mut()[0] = 2;
    let err = plan
        .execute(&ExecContext::serial(), &mut dest, &source)
        .unwrap_err();
    assert!(matches!(err, StridedError::InvalidBoolByte { value: 2 }));
}

#[test]
fn erased_copy_plan_accepts_explicit_execution_contexts() {
    let dims = [2usize];
    let strides = [1isize];
    let src = [3.0f64, 4.0];
    let mut serial_dst = [0.0f64; 2];
    let mut bounded_dst = [0.0f64; 2];
    let mut ambient_dst = [0.0f64; 2];

    let plan = ErasedCopyPlan::compile(KernelDType::F64, &dims, &strides, &strides).unwrap();
    let source =
        ErasedRawStridedRef::new(KernelDType::F64, as_bytes(&src), &dims, &strides, 0).unwrap();

    {
        let mut dest = ErasedRawStridedMut::new(
            KernelDType::F64,
            as_bytes_mut(&mut serial_dst),
            &dims,
            &strides,
            0,
        )
        .unwrap();
        plan.execute(&ExecContext::serial(), &mut dest, &source)
            .unwrap();
    }
    {
        let mut dest = ErasedRawStridedMut::new(
            KernelDType::F64,
            as_bytes_mut(&mut bounded_dst),
            &dims,
            &strides,
            0,
        )
        .unwrap();
        let ctx = ExecContext::max_threads(1).unwrap();
        plan.execute(&ctx, &mut dest, &source).unwrap();
    }
    {
        let mut dest = ErasedRawStridedMut::new(
            KernelDType::F64,
            as_bytes_mut(&mut ambient_dst),
            &dims,
            &strides,
            0,
        )
        .unwrap();
        plan.execute(&ExecContext::ambient(), &mut dest, &source)
            .unwrap();
    }

    assert!(matches!(
        ExecContext::max_threads(0).unwrap_err(),
        StridedError::InvalidThreadBudget { max_threads: 0 }
    ));
    assert_eq!(serial_dst, src);
    assert_eq!(bounded_dst, src);
    assert_eq!(ambient_dst, src);
}
