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
fn plan_rank_above_inline_limit_uses_spilled_fused_traversal() {
    let dims = [2usize; 9];
    let dst: Vec<isize> = (0..9).map(|axis| 1isize << axis).collect();
    let src: Vec<isize> = (0..9).rev().map(|axis| 1isize << axis).collect();
    let source_data = fill_f64(512);
    let plan = CopyPlan::compile(&dims, &dst, &src).unwrap();
    // Every axis is padded (stride 2*2^k), so fusion cannot reduce the rank:
    // the traversal spills instead of leaving the fused path.
    assert_eq!(plan.fused.rank(), 9);

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

    // The spilled traversal also serves scale and conj.
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
    let fused = plan.fused.inline().expect("rank 3 stays on the fused path");
    assert_eq!(fused.rank, 1);
    assert_eq!(fused.dims[0], 24);
}

#[test]
fn update_plan_computes_alpha_f_plus_beta_self_on_permuted_layout() {
    // 2x3 source, destination transposed, src strides make the source
    // non-contiguous so the update replays through the fused pair traversal.
    let dims = [2usize, 3];
    let dst_strides = [1isize, 2]; // column-major 2x3
    let src_strides = [3isize, 1]; // row-major window
    let plan = UpdatePlan::compile(&dims, &dst_strides, &src_strides).unwrap();

    let src = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let initial = [10.0f64, 20.0, 30.0, 40.0, 50.0, 60.0];
    let mut dst = initial;
    let source = RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap();
    let mut dest = RawStridedMut::new(&mut dst, &dims, &dst_strides, 0).unwrap();
    // dst = 2 * (src + 1) + 3 * dst
    plan.execute(&mut dest, &source, 2.0, 3.0, |x| x + 1.0)
        .unwrap();

    // Logical (i, j): src offset i*3 + j, dst offset i + j*2. Every logical
    // destination is written exactly once, so the expected value uses the
    // pre-update destination.
    for i in 0..2 {
        for j in 0..3 {
            let s = src[i * 3 + j];
            let expected = 2.0 * (s + 1.0) + 3.0 * initial[i + j * 2];
            assert_eq!(dst[i + j * 2], expected, "({i},{j})");
        }
    }
}

#[test]
fn update_plan_rejects_non_injective_destination_and_layout_drift() {
    let dims = [2usize, 2];
    let src_strides = [1isize, 2];
    let err = UpdatePlan::compile(&dims, &[1, 0], &src_strides).unwrap_err();
    assert!(matches!(err, StridedError::NonInjectiveOutputLayout));

    let plan = UpdatePlan::compile(&dims, &[1, 2], &src_strides).unwrap();
    let src = [1.0f64, 2.0, 3.0, 4.0];
    let mut dst = [0.0f64; 4];
    let source = RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap();
    let mut dest = RawStridedMut::new(&mut dst, &dims, &[2, 1], 0).unwrap();
    let err = plan
        .execute(&mut dest, &source, 1.0, 1.0, |x| x)
        .unwrap_err();
    assert!(matches!(err, StridedError::PlanLayoutMismatch));
}

/// Independent reference for the write-only update `dst = alpha * src` over a
/// logical layout, used to check that `beta = 0` never reads the destination.
fn reference_scale<T>(
    dst: &mut [T],
    src: &[T],
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    alpha: T,
) where
    T: Copy + Mul<Output = T>,
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

/// Poison a destination with `poison`, run `dest = alpha * f(src) + beta * dest`
/// and assert the result equals `alpha * f(src)` for a beta that is zero.
fn assert_update_beta_zero_case<T>(
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
        + MaybeSendSync
        + Add<Output = T>
        + Mul<Output = T>
        + num_traits::Zero
        + num_traits::One,
{
    let len: usize = dims.iter().product();
    let plan = UpdatePlan::compile(dims, dst_strides, src_strides).unwrap();
    let mut dst = vec![poison; len];
    {
        let source = RawStridedRef::new(src, dims, src_strides, 0).unwrap();
        let mut dest = RawStridedMut::new(&mut dst, dims, dst_strides, 0).unwrap();
        plan.execute(&mut dest, &source, alpha, beta, |x| x)
            .unwrap();
    }
    let mut expected = vec![poison; len];
    reference_scale(&mut expected, src, dims, dst_strides, src_strides, alpha);
    assert_eq!(dst, expected);
}

/// Poisoned `MaybeUninit` destination: `execute_uninit` must write every
/// reachable slot exactly as the write-only reference does, leaving holes as
/// they were.
fn assert_execute_uninit_case<T>(
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
        + MaybeSendSync
        + Add<Output = T>
        + Mul<Output = T>
        + num_traits::Zero
        + num_traits::One,
{
    let len: usize = dims.iter().product();
    let plan = UpdatePlan::compile(dims, dst_strides, src_strides).unwrap();
    let mut storage = vec![MaybeUninit::new(poison); len];
    {
        let source = RawStridedRef::new(src, dims, src_strides, 0).unwrap();
        let mut dest = RawStridedMut::new(&mut storage, dims, dst_strides, 0).unwrap();
        plan.execute_uninit(&mut dest, &source, alpha, |x| x)
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
fn update_plan_beta_zero_never_reads_nan_destination() {
    let dims = [2usize, 3];
    let contiguous = [3isize, 1];
    let src_permuted = [3isize, 1];
    let dst_permuted = [1isize, 2];

    let f64_src = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_update_beta_zero_case(
        &dims,
        &contiguous,
        &contiguous,
        &f64_src,
        f64::NAN,
        2.0,
        0.0,
    );
    assert_update_beta_zero_case(
        &dims,
        &dst_permuted,
        &src_permuted,
        &f64_src,
        f64::NAN,
        2.0,
        0.0,
    );
    // alpha == 1 takes the exact write-only copy branch.
    assert_update_beta_zero_case(
        &dims,
        &dst_permuted,
        &src_permuted,
        &f64_src,
        f64::NAN,
        1.0,
        0.0,
    );

    let f32_src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_update_beta_zero_case(
        &dims,
        &contiguous,
        &contiguous,
        &f32_src,
        f32::NAN,
        2.0,
        0.0,
    );
    assert_update_beta_zero_case(
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
    assert_update_beta_zero_case(
        &dims,
        &contiguous,
        &contiguous,
        &c64_src,
        c64_nan,
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 0.0),
    );
    assert_update_beta_zero_case(
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
fn update_plan_negative_zero_beta_is_write_only() {
    // `-0.0 == 0.0`, so beta = -0.0 must also skip the destination read.
    let dims = [2usize, 3];
    let src_strides = [3isize, 1];
    let dst_strides = [1isize, 2];
    let src = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_update_beta_zero_case(&dims, &dst_strides, &src_strides, &src, f64::NAN, 3.0, -0.0);
}

#[test]
fn update_plan_beta_nonzero_reads_destination() {
    let dims = [2usize, 2];
    let strides = [1isize, 2];
    let plan = UpdatePlan::compile(&dims, &strides, &strides).unwrap();
    let src = [1.0f64, 2.0, 3.0, 4.0];
    let initial = [10.0f64, 20.0, 30.0, 40.0];
    let mut dst = initial;
    let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
    let mut dest = RawStridedMut::new(&mut dst, &dims, &strides, 0).unwrap();
    plan.execute(&mut dest, &source, 2.0, 0.5, |x| x + 1.0)
        .unwrap();
    for index in 0..4 {
        assert_eq!(dst[index], 2.0 * (src[index] + 1.0) + 0.5 * initial[index]);
    }
}

#[test]
fn update_plan_nan_beta_reads_destination() {
    // NaN is not zero, so the destination read path stays active.
    let dims = [2usize];
    let strides = [1isize];
    let plan = UpdatePlan::compile(&dims, &strides, &strides).unwrap();
    let src = [1.0f64, 2.0];
    let mut dst = [3.0f64, 4.0];
    let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
    let mut dest = RawStridedMut::new(&mut dst, &dims, &strides, 0).unwrap();
    plan.execute(&mut dest, &source, 1.0, f64::NAN, |x| x)
        .unwrap();
    assert!(dst.iter().all(|value| value.is_nan()));
}

#[test]
fn update_plan_execute_uninit_initializes_every_reachable_slot() {
    let dims = [2usize, 3];
    let contiguous = [3isize, 1];
    let src_permuted = [3isize, 1];
    let dst_permuted = [1isize, 2];

    let f64_src = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_execute_uninit_case(&dims, &contiguous, &contiguous, &f64_src, f64::NAN, 2.0);
    assert_execute_uninit_case(&dims, &contiguous, &contiguous, &f64_src, f64::NAN, 1.0);
    assert_execute_uninit_case(&dims, &dst_permuted, &src_permuted, &f64_src, f64::NAN, 2.0);

    let f32_src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_execute_uninit_case(&dims, &contiguous, &contiguous, &f32_src, f32::NAN, 2.0);
    assert_execute_uninit_case(&dims, &dst_permuted, &src_permuted, &f32_src, f32::NAN, 1.0);

    let c64_src: Vec<Complex64> = (0..6)
        .map(|value| Complex64::new(value as f64, -(value as f64)))
        .collect();
    let c64_nan = Complex64::new(f64::NAN, f64::NAN);
    assert_execute_uninit_case(
        &dims,
        &contiguous,
        &contiguous,
        &c64_src,
        c64_nan,
        Complex64::new(2.0, 0.0),
    );
    assert_execute_uninit_case(
        &dims,
        &dst_permuted,
        &src_permuted,
        &c64_src,
        c64_nan,
        Complex64::new(1.0, 0.0),
    );
}

#[test]
fn update_plan_execute_uninit_leaves_gapped_holes_untouched() {
    // Rank-1 destination with stride 2 reaches offsets 0 and 2; the holes at 1
    // and 3 keep their poison and must not be written.
    let dims = [2usize];
    let dst_strides = [2isize];
    let src_strides = [1isize];
    let src = [7.0f64, 8.0];
    let plan = UpdatePlan::compile(&dims, &dst_strides, &src_strides).unwrap();
    let mut storage = vec![MaybeUninit::new(f64::NAN); 4];
    {
        let source = RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap();
        let mut dest = RawStridedMut::new(&mut storage, &dims, &dst_strides, 0).unwrap();
        plan.execute_uninit(&mut dest, &source, 1.0, |x| x).unwrap();
    }
    // SAFETY: every slot was initialized with NaN before the call.
    unsafe {
        assert_eq!(storage[0].assume_init(), 7.0);
        assert!(storage[1].assume_init().is_nan());
        assert_eq!(storage[2].assume_init(), 8.0);
        assert!(storage[3].assume_init().is_nan());
    }
}
