use super::*;
use num_complex::{Complex32, Complex64};

#[test]
fn uninit_then_receipt_drops_after_panic() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let plan = CopyPlan::compile(&[2], &[1], &[1]).unwrap();
    let source_data = [3i32, 5];
    let source = RawStridedRef::new(&source_data, &[2], &[1], 0).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut storage = vec![MaybeUninit::<i32>::uninit(); 3];
        let mut dest = RawStridedMut::new(&mut storage, &[2], &[1], 0).unwrap();
        let _: () = plan
            .execute_uninit_then(&mut dest, &source, |_receipt| {
                panic!("post-copy update failure");
            })
            .unwrap();
    }));
    assert!(result.is_err());
}

/// Reference: the per-call raw kernel (which itself is differential-tested
/// against the view kernels in raw_ops.rs).
fn plan_matches_direct<T>(dims: &[usize], dst_strides: &[isize], src_strides: &[isize], src: &[T])
where
    T: Copy
        + PartialEq
        + core::fmt::Debug
        + Default
        + Mul<T, Output = T>
        + ElementOpApply
        + MaybeSendSync
        + num_traits::One,
{
    let len = src.len();
    let plan = CopyPlan::compile(dims, dst_strides, src_strides).unwrap();

    let mut expected = vec![T::default(); len];
    {
        let mut dest = RawStridedMut::new(&mut expected, dims, dst_strides, 0).unwrap();
        let source = RawStridedRef::new(src, dims, src_strides, 0).unwrap();
        crate::copy_scale_raw(&mut dest, &source, T::one()).unwrap();
    }

    let mut actual = vec![T::default(); len];
    {
        let mut dest = RawStridedMut::new(&mut actual, dims, dst_strides, 0).unwrap();
        let source = RawStridedRef::new(src, dims, src_strides, 0).unwrap();
        plan.execute(&mut dest, &source).unwrap();
    }
    assert_eq!(actual, expected);
}

fn fill_f64(len: usize) -> Vec<f64> {
    (0..len).map(|value| value as f64 - 2.5).collect()
}

#[test]
fn plan_copy_matches_direct_rank0() {
    plan_matches_direct::<f64>(&[], &[], &[], &[7.0]);
}

#[test]
fn plan_copy_matches_direct_rank1() {
    plan_matches_direct::<f64>(&[5], &[1], &[1], &fill_f64(5));
}

#[test]
fn plan_copy_matches_direct_rank2_transposed() {
    plan_matches_direct::<f64>(&[3, 4], &[1, 3], &[4, 1], &fill_f64(12));
}

#[test]
fn plan_copy_matches_direct_rank4() {
    plan_matches_direct::<f64>(&[2, 3, 2, 2], &[12, 4, 2, 1], &[1, 2, 6, 12], &fill_f64(24));
}

#[test]
fn plan_copy_matches_direct_rank8() {
    let dims = [2usize; 8];
    let dst: Vec<isize> = (0..8).map(|axis| 1isize << axis).collect();
    let src: Vec<isize> = (0..8).rev().map(|axis| 1isize << axis).collect();
    plan_matches_direct::<f64>(&dims, &dst, &src, &fill_f64(256));
}

#[test]
fn plan_copy_matches_direct_zero_size() {
    plan_matches_direct::<f64>(&[2, 0, 3], &[3, 3, 1], &[1, 6, 2], &fill_f64(6));
}

#[test]
fn plan_copy_matches_direct_f32_and_complex() {
    let dims = [2usize, 3];
    let dst = [1isize, 2];
    let src = [3isize, 1];
    plan_matches_direct::<f32>(&dims, &dst, &src, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let complex: Vec<Complex32> = (0..6)
        .map(|value| Complex32::new(value as f32, -(value as f32)))
        .collect();
    plan_matches_direct::<Complex32>(&dims, &dst, &src, &complex);
    let complex: Vec<Complex64> = (0..6)
        .map(|value| Complex64::new(value as f64, 1.0 - value as f64))
        .collect();
    plan_matches_direct::<Complex64>(&dims, &dst, &src, &complex);
}

#[test]
fn plan_copy_negative_stride_matches_view_kernel() {
    // Negative source stride: src viewed reversed, offset at the end.
    let dims = [4usize];
    let src_strides = [-1isize];
    let dst_strides = [1isize];
    let src = [1.0f64, 2.0, 3.0, 4.0];
    let plan = CopyPlan::compile(&dims, &dst_strides, &src_strides).unwrap();

    let mut actual = [0.0f64; 4];
    let mut dest = RawStridedMut::new(&mut actual, &dims, &dst_strides, 0).unwrap();
    let source = RawStridedRef::new(&src, &dims, &src_strides, 3).unwrap();
    plan.execute(&mut dest, &source).unwrap();
    assert_eq!(actual, [4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn plan_execute_scale_and_conj() {
    let dims = [2usize, 2];
    let strides = [2isize, 1];
    let src = [
        Complex64::new(1.0, 2.0),
        Complex64::new(-3.0, 4.0),
        Complex64::new(0.5, -1.0),
        Complex64::new(2.0, 0.0),
    ];
    let plan = CopyPlan::compile(&dims, &strides, &strides).unwrap();

    let mut scaled = [Complex64::default(); 4];
    let mut dest = RawStridedMut::new(&mut scaled, &dims, &strides, 0).unwrap();
    let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
    plan.execute_scale(&mut dest, &source, Complex64::new(2.0, 0.0))
        .unwrap();
    assert_eq!(scaled[1], Complex64::new(-6.0, 8.0));

    let mut conjugated = [Complex64::default(); 4];
    let mut dest = RawStridedMut::new(&mut conjugated, &dims, &strides, 0).unwrap();
    let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
    plan.execute_conj(&mut dest, &source).unwrap();
    assert_eq!(conjugated[0], Complex64::new(1.0, -2.0));
    assert_eq!(conjugated[3], Complex64::new(2.0, 0.0));
}

#[test]
fn plan_rank_above_limit_falls_back_to_view_kernels() {
    let dims = [2usize; 9];
    let dst: Vec<isize> = (0..9).map(|axis| 1isize << axis).collect();
    let src: Vec<isize> = (0..9).rev().map(|axis| 1isize << axis).collect();
    let source_data = fill_f64(512);
    let plan = CopyPlan::compile(&dims, &dst, &src).unwrap();
    assert!(plan.fused.is_none());

    let mut expected = vec![0.0f64; 512];
    {
        let mut dest = RawStridedMut::new(&mut expected, &dims, &dst, 0).unwrap();
        let source = RawStridedRef::new(&source_data, &dims, &src, 0).unwrap();
        crate::copy_scale_raw(&mut dest, &source, 1.0).unwrap();
    }
    let mut actual = vec![0.0f64; 512];
    let mut dest = RawStridedMut::new(&mut actual, &dims, &dst, 0).unwrap();
    let source = RawStridedRef::new(&source_data, &dims, &src, 0).unwrap();
    plan.execute(&mut dest, &source).unwrap();
    assert_eq!(actual, expected);

    // Fallback also serves scale and conj.
    let mut scaled = vec![0.0f64; 512];
    let mut dest = RawStridedMut::new(&mut scaled, &dims, &dst, 0).unwrap();
    plan.execute_scale(&mut dest, &source, 2.0).unwrap();
    assert_eq!(scaled[0], 2.0 * actual[0]);
    let mut conjugated = vec![0.0f64; 512];
    let mut dest = RawStridedMut::new(&mut conjugated, &dims, &dst, 0).unwrap();
    plan.execute_conj(&mut dest, &source).unwrap();
    assert_eq!(conjugated, actual);
}

#[test]
fn compile_rejects_length_mismatch() {
    let err = CopyPlan::compile(&[2, 3], &[3, 1], &[1]).unwrap_err();
    assert!(matches!(err, StridedError::StrideLengthMismatch));
    let err = CopyPlan::compile(&[2, 3], &[3], &[1, 2]).unwrap_err();
    assert!(matches!(err, StridedError::StrideLengthMismatch));
}

#[test]
fn compile_rejects_extent_overflow() {
    let err = CopyPlan::compile(&[usize::MAX, 2], &[1, 1], &[1, 1]).unwrap_err();
    assert!(matches!(err, StridedError::OffsetOverflow));
}

#[test]
fn compile_rejects_unrepresentable_positive_and_negative_offset_spans() {
    for strides in [
        [isize::MAX / 2 + 1, isize::MAX],
        [isize::MIN / 2 - 1, isize::MIN],
    ] {
        let err = CopyPlan::compile(&[2, 2], &strides, &strides).unwrap_err();
        assert!(matches!(err, StridedError::NonInjectiveOutputLayout));
    }
}

#[test]
fn compile_accepts_representable_mixed_sign_span_without_fusion_overflow() {
    let positive = isize::MAX / 4;
    let negative = -(isize::MAX - positive);
    let strides = [positive, negative];
    CopyPlan::compile(&[2, 2], &strides, &strides).unwrap();
}

#[test]
fn compile_rejects_non_injective_destination() {
    // Two logical columns land on the same offsets: forbidden overlap in
    // the mutable destination.
    let err = CopyPlan::compile(&[2, 2], &[1, 0], &[2, 1]).unwrap_err();
    assert!(matches!(err, StridedError::NonInjectiveOutputLayout));
    // Broadcast-like (stride 0) source layouts remain allowed.
    CopyPlan::compile(&[2, 2], &[2, 1], &[0, 1]).unwrap();
}

#[test]
fn execute_rejects_layout_drift() {
    let dims = [2usize, 3];
    let strides = [3isize, 1];
    let plan = CopyPlan::compile(&dims, &strides, &strides).unwrap();
    let src = fill_f64(6);
    let mut dst = vec![0.0f64; 6];

    // Different dims than compiled.
    let other_dims = [3usize, 2];
    let other_strides = [2isize, 1];
    let mut dest = RawStridedMut::new(&mut dst, &other_dims, &other_strides, 0).unwrap();
    let source = RawStridedRef::new(&src, &other_dims, &other_strides, 0).unwrap();
    let err = plan.execute(&mut dest, &source).unwrap_err();
    assert!(matches!(err, StridedError::PlanLayoutMismatch));

    // Same dims, different source strides.
    let column_major = [1isize, 2];
    let mut dest = RawStridedMut::new(&mut dst, &dims, &strides, 0).unwrap();
    let source = RawStridedRef::new(&src, &dims, &column_major, 0).unwrap();
    let err = plan.execute_scale(&mut dest, &source, 1.0).unwrap_err();
    assert!(matches!(err, StridedError::PlanLayoutMismatch));

    // Same dims, different destination strides.
    let mut dest = RawStridedMut::new(&mut dst, &dims, &column_major, 0).unwrap();
    let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
    let err = plan.execute_conj(&mut dest, &source).unwrap_err();
    assert!(matches!(err, StridedError::PlanLayoutMismatch));
}

#[test]
fn identity_layout_uses_single_fused_axis() {
    let plan = CopyPlan::compile(&[2, 3, 4], &[12, 4, 1], &[12, 4, 1]).unwrap();
    let fused = plan.fused.expect("rank 3 stays on the fused path");
    assert_eq!(fused.rank, 1);
    assert_eq!(fused.dims[0], 24);
}
