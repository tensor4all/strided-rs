//! Erased `ErasedUpdatePlan` beta-zero and uninitialized-output semantics.
//!
//! BLAS treats `beta = 0` as "do not read the destination"; these tests poison
//! the destination with NaN (or leave it uninitialized) and require the erased
//! update to produce finite results anyway.

#[cfg(feature = "parallel")]
mod common;
#[cfg(feature = "parallel")]
use common::Budget;
use core::mem::MaybeUninit;

use num_complex::{Complex32, Complex64};
use strided_basic::Exec;
use strided_basic::{
    ElementOpApply, ErasedRawStridedMut, ErasedRawStridedPtr, ErasedRawStridedRef,
    ErasedRawStridedUninitMut, ErasedUpdatePlan, KernelDType, KernelStorageElement, StridedError,
    UpdateOp,
};

/// Independent reference for `dst = alpha * src` over a logical layout.
fn reference_scale<T>(
    dst: &mut [T],
    src: &[T],
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    alpha: T,
) where
    T: Copy + core::ops::Mul<Output = T>,
{
    let total: usize = dims.iter().product();
    for linear in 0..total {
        let mut rest = linear;
        let mut d = 0isize;
        let mut s = 0isize;
        for axis in 0..dims.len() {
            let coord = (rest % dims[axis]) as isize;
            rest /= dims[axis];
            d += coord * dst_strides[axis];
            s += coord * src_strides[axis];
        }
        dst[d as usize] = alpha * src[s as usize];
    }
}

/// Poison the initialized destination, run the erased update, and require the
/// write-only `alpha * src` result for a beta that is zero.
fn assert_beta_zero_case<T>(
    dtype: KernelDType,
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    src: &[T],
    poison: T,
    alpha: T,
    beta: T,
) where
    T: Copy
        + PartialEq
        + core::fmt::Debug
        + KernelStorageElement
        + Send
        + Sync
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>
        + num_traits::Zero
        + num_traits::One
        + ElementOpApply,
{
    let plan = ErasedUpdatePlan::compile(dtype, dims, dst_strides, src_strides).unwrap();
    let len: usize = dims.iter().product();
    let mut dst = vec![poison; len];
    {
        let source = ErasedRawStridedRef::from_slice(src, dims, src_strides, 0).unwrap();
        let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, dims, dst_strides, 0).unwrap();
        plan.execute(
            &mut dest,
            &source,
            alpha,
            beta,
            UpdateOp::Identity,
            Exec::serial(),
        )
        .unwrap();
    }
    let mut expected = vec![poison; len];
    reference_scale(&mut expected, src, dims, dst_strides, src_strides, alpha);
    assert_eq!(dst, expected);
}

/// `execute_uninit` must initialize every reachable slot; a poisoned hole stays
/// untouched.
fn assert_uninit_case<T>(
    dtype: KernelDType,
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    src: &[T],
    poison: T,
    alpha: T,
) where
    T: Copy
        + PartialEq
        + core::fmt::Debug
        + KernelStorageElement
        + Send
        + Sync
        + core::ops::Mul<Output = T>
        + num_traits::One
        + ElementOpApply,
{
    let plan = ErasedUpdatePlan::compile(dtype, dims, dst_strides, src_strides).unwrap();
    let len: usize = dims.iter().product();
    let mut storage = vec![MaybeUninit::new(poison); len];
    {
        let source = ErasedRawStridedRef::from_slice(src, dims, src_strides, 0).unwrap();
        let source_ptr = ErasedRawStridedPtr::from_ref(&source);
        let mut dest =
            ErasedRawStridedUninitMut::from_uninit_slice(&mut storage, dims, dst_strides, 0)
                .unwrap();
        plan.execute_uninit(
            &mut dest,
            &source_ptr,
            alpha,
            UpdateOp::Identity,
            Exec::serial(),
        )
        .unwrap();
    }
    let mut expected = vec![poison; len];
    reference_scale(&mut expected, src, dims, dst_strides, src_strides, alpha);
    for (got, want) in storage.iter().zip(&expected) {
        // SAFETY: every slot was initialized with `poison` above, so even a
        // hole stays a valid readable value.
        assert_eq!(unsafe { got.assume_init() }, *want);
    }
}

#[test]
fn erased_update_plan_beta_zero_never_reads_nan_destination() {
    let dims = [2usize, 3];
    let contiguous = [3isize, 1];
    let src_permuted = [3isize, 1];
    let dst_permuted = [1isize, 2];

    let f64_src = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_beta_zero_case(
        KernelDType::F64,
        &dims,
        &contiguous,
        &contiguous,
        &f64_src,
        f64::NAN,
        2.0,
        0.0,
    );
    assert_beta_zero_case(
        KernelDType::F64,
        &dims,
        &dst_permuted,
        &src_permuted,
        &f64_src,
        f64::NAN,
        1.0,
        0.0,
    );
    // -0.0 == 0.0, so this is the write-only branch too.
    assert_beta_zero_case(
        KernelDType::F64,
        &dims,
        &dst_permuted,
        &src_permuted,
        &f64_src,
        f64::NAN,
        3.0,
        -0.0,
    );

    let f32_src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_beta_zero_case(
        KernelDType::F32,
        &dims,
        &contiguous,
        &contiguous,
        &f32_src,
        f32::NAN,
        2.0,
        0.0,
    );
    assert_beta_zero_case(
        KernelDType::F32,
        &dims,
        &dst_permuted,
        &src_permuted,
        &f32_src,
        f32::NAN,
        1.0,
        0.0,
    );

    let c64_src: Vec<Complex64> = (0..6)
        .map(|value| Complex64::new(value as f64, -(value as f64)))
        .collect();
    let c64_nan = Complex64::new(f64::NAN, f64::NAN);
    assert_beta_zero_case(
        KernelDType::C64,
        &dims,
        &contiguous,
        &contiguous,
        &c64_src,
        c64_nan,
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 0.0),
    );
    assert_beta_zero_case(
        KernelDType::C64,
        &dims,
        &dst_permuted,
        &src_permuted,
        &c64_src,
        c64_nan,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    );
}

#[test]
fn erased_update_plan_conj_beta_zero_ignores_nan_destination() {
    let dims = [2usize, 3];
    let src_strides = [3isize, 1];
    let dst_strides = [1isize, 2];
    let src: Vec<Complex32> = (0..6)
        .map(|value| Complex32::new(value as f32, -(value as f32)))
        .collect();
    let plan =
        ErasedUpdatePlan::compile(KernelDType::C32, &dims, &dst_strides, &src_strides).unwrap();
    let mut dst = vec![Complex32::new(f32::NAN, f32::NAN); 6];
    {
        let source = ErasedRawStridedRef::from_slice(&src, &dims, &src_strides, 0).unwrap();
        let mut dest =
            ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &dst_strides, 0).unwrap();
        plan.execute(
            &mut dest,
            &source,
            Complex32::new(2.0, 0.0),
            Complex32::new(0.0, 0.0),
            UpdateOp::Conj,
            Exec::serial(),
        )
        .unwrap();
    }
    let mut expected = vec![Complex32::new(f32::NAN, f32::NAN); 6];
    let conjugated: Vec<Complex32> = src.iter().map(|value| value.conj()).collect();
    reference_scale(
        &mut expected,
        &conjugated,
        &dims,
        &dst_strides,
        &src_strides,
        Complex32::new(2.0, 0.0),
    );
    assert_eq!(dst, expected);
}

#[test]
fn erased_update_plan_beta_nonzero_reads_destination() {
    let dims = [2usize, 2];
    let strides = [1isize, 2];
    let plan = ErasedUpdatePlan::compile(KernelDType::F64, &dims, &strides, &strides).unwrap();
    let src = [1.0f64, 2.0, 3.0, 4.0];
    let initial = [10.0f64, 20.0, 30.0, 40.0];
    let mut dst = initial;
    let source = ErasedRawStridedRef::from_slice(&src, &dims, &strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &strides, 0).unwrap();
    plan.execute(
        &mut dest,
        &source,
        2.0,
        0.5,
        UpdateOp::Identity,
        Exec::serial(),
    )
    .unwrap();
    for index in 0..4 {
        assert_eq!(dst[index], 2.0 * src[index] + 0.5 * initial[index]);
    }
}

#[test]
fn erased_update_plan_nan_beta_reads_destination() {
    let dims = [2usize];
    let strides = [1isize];
    let plan = ErasedUpdatePlan::compile(KernelDType::F64, &dims, &strides, &strides).unwrap();
    let src = [1.0f64, 2.0];
    let mut dst = [3.0f64, 4.0];
    let source = ErasedRawStridedRef::from_slice(&src, &dims, &strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &strides, 0).unwrap();
    plan.execute(
        &mut dest,
        &source,
        1.0,
        f64::NAN,
        UpdateOp::Identity,
        Exec::serial(),
    )
    .unwrap();
    assert!(dst.iter().all(|value| value.is_nan()));
}

#[test]
fn erased_update_plan_execute_uninit_initializes_every_reachable_slot() {
    let dims = [2usize, 3];
    let contiguous = [3isize, 1];
    let src_permuted = [3isize, 1];
    let dst_permuted = [1isize, 2];

    let f64_src = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_uninit_case(
        KernelDType::F64,
        &dims,
        &contiguous,
        &contiguous,
        &f64_src,
        f64::NAN,
        2.0,
    );
    assert_uninit_case(
        KernelDType::F64,
        &dims,
        &dst_permuted,
        &src_permuted,
        &f64_src,
        f64::NAN,
        1.0,
    );

    let f32_src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_uninit_case(
        KernelDType::F32,
        &dims,
        &contiguous,
        &contiguous,
        &f32_src,
        f32::NAN,
        2.0,
    );
    assert_uninit_case(
        KernelDType::F32,
        &dims,
        &dst_permuted,
        &src_permuted,
        &f32_src,
        f32::NAN,
        1.0,
    );

    let c64_src: Vec<Complex64> = (0..6)
        .map(|value| Complex64::new(value as f64, -(value as f64)))
        .collect();
    let c64_nan = Complex64::new(f64::NAN, f64::NAN);
    assert_uninit_case(
        KernelDType::C64,
        &dims,
        &contiguous,
        &contiguous,
        &c64_src,
        c64_nan,
        Complex64::new(2.0, 0.0),
    );
    assert_uninit_case(
        KernelDType::C64,
        &dims,
        &dst_permuted,
        &src_permuted,
        &c64_src,
        c64_nan,
        Complex64::new(1.0, 0.0),
    );
}

#[test]
fn erased_update_plan_rejects_before_any_write() {
    let dims = [2usize];
    let strides = [1isize];
    let src = [1.0f64, 2.0];

    // dtype mismatch
    let plan = ErasedUpdatePlan::compile(KernelDType::F64, &dims, &strides, &strides).unwrap();
    let mut dst32 = [0.0f32; 2];
    let source = ErasedRawStridedRef::from_slice(&src, &dims, &strides, 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst32, &dims, &strides, 0).unwrap();
    assert!(matches!(
        plan.execute(
            &mut dest,
            &source,
            1.0,
            0.0,
            UpdateOp::Identity,
            Exec::serial()
        ),
        Err(StridedError::DTypeMismatch { .. })
    ));
    let mut storage = [MaybeUninit::new(f32::NAN); 2];
    let source_ptr = ErasedRawStridedPtr::from_ref(&source);
    let mut uninit =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut storage, &dims, &strides, 0).unwrap();
    assert!(matches!(
        plan.execute_uninit(
            &mut uninit,
            &source_ptr,
            1.0,
            UpdateOp::Identity,
            Exec::serial()
        ),
        Err(StridedError::DTypeMismatch { .. })
    ));
    drop(uninit);
    // SAFETY: every slot was initialized with NaN before the failed call.
    assert!(unsafe { storage[0].assume_init() }.is_nan());

    // Plan layout mismatch on a differently strided source.
    let mut dst = [0.0f64; 4];
    let source_other =
        ErasedRawStridedRef::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &dims, &[2], 0).unwrap();
    let mut dest = ErasedRawStridedMut::from_slice_mut(&mut dst, &dims, &strides, 0).unwrap();
    assert!(matches!(
        plan.execute(
            &mut dest,
            &source_other,
            1.0,
            0.0,
            UpdateOp::Identity,
            Exec::serial()
        ),
        Err(StridedError::PlanLayoutMismatch)
    ));

    // The uninit entry rejects a source inside the destination allocation
    // before writing anything.
    let mut storage = [MaybeUninit::new(1.0f64); 4];
    let raw_source = unsafe {
        ErasedRawStridedPtr::from_raw_parts(
            KernelDType::F64,
            core::ptr::NonNull::new(storage.as_mut_ptr().cast::<u8>()).unwrap(),
            32,
            &dims,
            &strides,
            0,
        )
    }
    .unwrap();
    let mut uninit =
        ErasedRawStridedUninitMut::from_uninit_slice(&mut storage, &dims, &strides, 2).unwrap();
    assert!(matches!(
        plan.execute_uninit(
            &mut uninit,
            &raw_source,
            1.0,
            UpdateOp::Identity,
            Exec::serial()
        ),
        Err(StridedError::OverlappingInputOutput { input: 0 })
    ));
}

#[cfg(feature = "parallel")]
#[cfg(feature = "parallel")]
fn with_two_workers<R: Send>(f: impl FnOnce(&Budget) -> R + Send) -> R {
    f(&Budget::Pooled(2))
}

#[cfg(feature = "parallel")]
#[test]
fn erased_update_plan_two_worker_matches_serial() {
    // Above the threading threshold so the two-worker replay really splits.
    let dims = [40_001usize];
    let strides = [1isize];
    let plan = ErasedUpdatePlan::compile(KernelDType::F64, &dims, &strides, &strides).unwrap();
    let src: Vec<f64> = (0..40_001).map(|value| value as f64).collect();

    let execute = |ctx: &Budget| -> Vec<f64> {
        let mut out = vec![f64::NAN; 40_001];
        let source = ErasedRawStridedRef::from_slice(&src, &dims, &strides, 0).unwrap();
        let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &dims, &strides, 0).unwrap();
        plan.execute(&mut dest, &source, 2.0, 0.0, UpdateOp::Identity, ctx.exec())
            .unwrap();
        out
    };
    let execute_uninit = |ctx: &Budget| -> Vec<f64> {
        let mut storage = vec![MaybeUninit::new(f64::NAN); 40_001];
        let source = ErasedRawStridedRef::from_slice(&src, &dims, &strides, 0).unwrap();
        let source_ptr = ErasedRawStridedPtr::from_ref(&source);
        let mut dest =
            ErasedRawStridedUninitMut::from_uninit_slice(&mut storage, &dims, &strides, 0).unwrap();
        plan.execute_uninit(&mut dest, &source_ptr, 2.0, UpdateOp::Identity, ctx.exec())
            .unwrap();
        storage
            .iter()
            .map(|value| unsafe { value.assume_init() })
            .collect()
    };

    let expected: Vec<f64> = (0..40_001).map(|value| 2.0 * value as f64).collect();
    assert_eq!(execute(&Budget::Serial), expected);
    assert_eq!(with_two_workers(execute), expected);
    assert_eq!(execute_uninit(&Budget::Serial), expected);
    assert_eq!(with_two_workers(execute_uninit), expected);
}
