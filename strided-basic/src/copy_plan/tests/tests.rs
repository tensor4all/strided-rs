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
            .execute_uninit_then(&mut dest, &source, Exec::serial(), |_receipt, _exec| {
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
        + Send
        + Sync
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
        plan.execute(&mut dest, &source, Exec::serial()).unwrap();
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
    plan.execute(&mut dest, &source, Exec::serial()).unwrap();
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
    plan.execute_scale(&mut dest, &source, Complex64::new(2.0, 0.0), Exec::serial())
        .unwrap();
    assert_eq!(scaled[1], Complex64::new(-6.0, 8.0));

    let mut conjugated = [Complex64::default(); 4];
    let mut dest = RawStridedMut::new(&mut conjugated, &dims, &strides, 0).unwrap();
    let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
    plan.execute_conj(&mut dest, &source, Exec::serial())
        .unwrap();
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
    plan.execute(&mut dest, &source, Exec::serial()).unwrap();
    assert_eq!(actual, expected);

    // The spilled traversal also serves scale and conj.
    let mut scaled = vec![0.0f64; 512];
    let mut dest = RawStridedMut::new(&mut scaled, &dims, &dst, 0).unwrap();
    plan.execute_scale(&mut dest, &source, 2.0, Exec::serial())
        .unwrap();
    assert_eq!(scaled[0], 2.0 * actual[0]);
    let mut conjugated = vec![0.0f64; 512];
    let mut dest = RawStridedMut::new(&mut conjugated, &dims, &dst, 0).unwrap();
    plan.execute_conj(&mut dest, &source, Exec::serial())
        .unwrap();
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
    let err = plan
        .execute(&mut dest, &source, Exec::serial())
        .unwrap_err();
    assert!(matches!(err, StridedError::PlanLayoutMismatch));

    // Same dims, different source strides.
    let column_major = [1isize, 2];
    let mut dest = RawStridedMut::new(&mut dst, &dims, &strides, 0).unwrap();
    let source = RawStridedRef::new(&src, &dims, &column_major, 0).unwrap();
    let err = plan
        .execute_scale(&mut dest, &source, 1.0, Exec::serial())
        .unwrap_err();
    assert!(matches!(err, StridedError::PlanLayoutMismatch));

    // Same dims, different destination strides.
    let mut dest = RawStridedMut::new(&mut dst, &dims, &column_major, 0).unwrap();
    let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
    let err = plan
        .execute_conj(&mut dest, &source, Exec::serial())
        .unwrap_err();
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
    plan.execute(&mut dest, &source, 2.0, 3.0, |x| x + 1.0, Exec::serial())
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
        .execute(&mut dest, &source, 1.0, 1.0, |x| x, Exec::serial())
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
        + Send
        + Sync
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
        plan.execute(&mut dest, &source, alpha, beta, |x| x, Exec::serial())
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
        + Send
        + Sync
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
        plan.execute_uninit(&mut dest, &source, alpha, |x| x, Exec::serial())
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
    plan.execute(&mut dest, &source, 2.0, 0.5, |x| x + 1.0, Exec::serial())
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
    plan.execute(&mut dest, &source, 1.0, f64::NAN, |x| x, Exec::serial())
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
        plan.execute_uninit(&mut dest, &source, 1.0, |x| x, Exec::serial())
            .unwrap();
    }
    // SAFETY: every slot was initialized with NaN before the call.
    unsafe {
        assert_eq!(storage[0].assume_init(), 7.0);
        assert!(storage[1].assume_init().is_nan());
        assert_eq!(storage[2].assume_init(), 8.0);
        assert!(storage[3].assume_init().is_nan());
    }
}

/// A `CopyPlan` prices itself by its footprint: past the class's L2 cache a
/// strided (one-element-per-cache-line) plan costs the class's beyond-L2 copy
/// rate, while a plan that is one unit-stride run keeps the `Copy` row at every
/// size, so no contiguous caller's width moves (strided-rs#364).
#[test]
fn a_strided_plan_past_l2_prices_itself_above_the_contiguous_row() {
    use crate::cost::parallel_width;
    use threadteam::HostClass;
    let row = KernelClass::Copy.ns_per_element();
    let dims = [512usize, 512];
    // 2 MiB of source and 2 MiB of destination, transposed / as one run.
    let strided = CopyPlan::compile(&dims, &[512, 1], &[1, 512]).unwrap();
    let contiguous = CopyPlan::compile(&dims, &[1, 512], &[1, 512]).unwrap();
    let rate = strided.ns_per_element::<f64, f64>();

    // Each side is counted at its own element size: 256 x 256 f64 is exactly the
    // 1 MiB the class's L2 holds, so it stays on the row (twice 1 MiB would not).
    let at_l2 = CopyPlan::compile(&[256usize, 256], &[256, 1], &[1, 256]).unwrap();
    assert_eq!(at_l2.ns_per_element::<f64, f64>(), row);
    // A streamed run keeps the row at every footprint, reversed included.
    assert_eq!(contiguous.ns_per_element::<f64, f64>(), row);
    let reversed = CopyPlan::compile(&[4096usize], &[-1], &[-1]).unwrap();
    assert_eq!(reversed.ns_per_element::<f64, f64>(), row);
    // An L2-resident transposing plan (128 x 128 f64 is 256 KiB) keeps the row.
    let small = CopyPlan::compile(&[128usize, 128], &[128, 1], &[1, 128]).unwrap();
    assert_eq!(small.ns_per_element::<f64, f64>(), row);

    if HostClass::detect() == HostClass::X86_64Avx512 {
        // The table of strided-rs#363 measures 3.09-3.24 ns per element for the
        // 512 x 512 transposing case, so the plan must be within a factor of two
        // of it; and one byte past the class's L2 is already priced there.
        assert!(
            (3.09 / 2.0..=3.24 * 2.0).contains(&rate),
            "the plan prices itself at {rate} ns per element, outside a factor of two \
             of the 3.09-3.24 measured for this case"
        );
        let just_over = CopyPlan::compile(&[257usize, 257], &[257, 1], &[1, 257]).unwrap();
        assert!(
            just_over.ns_per_element::<f64, f64>() > row,
            "257 x 257 f64 is 1.01 MiB of source and destination, past the class's L2"
        );
    } else {
        assert_eq!(rate, row, "this class has no beyond-L2 copy rate");
    }

    // The contiguous plan asks the gate for the width the class's public rule
    // gives, so the footprint term cannot have moved a contiguous caller.
    let pool = crate::threading::test_pool(4);
    let total = 512 * 512;
    let mut exec = Exec::new(&pool, 4).unwrap();
    let plan_width = crate::cost::for_len_at_rate(
        exec.reborrow(),
        total,
        contiguous.ns_per_element::<f64, f64>(),
    )
    .budget();
    assert_eq!(
        plan_width.get(),
        parallel_width(&exec, KernelClass::Copy, total).get()
    );
}

/// The working set of a layout pair in bytes, as the plan forms it, for the
/// tests that must sit past the class's L2.
fn plan_footprint(
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    element: usize,
) -> usize {
    span_elements(dims, dst_strides) * element + span_elements(dims, src_strides) * element
}

/// A plan whose footprint is past its class's L2 forks at its own rate on the
/// replay that is not tiled: 1024 elements of a 4 MiB working set are 175 ns at
/// the `Copy` row, below the 300 ns this pool asks for a fork, and 3.07 us at the
/// plan's beyond-L2 rate (strided-rs#364).
///
/// Executed, so the rate `CopyPlan::replay` computes is the one under test: a
/// replay that passed the row here would install no worker and the assertion
/// below fails. Without the `parallel` feature every token is serial, so there is
/// no fork to observe.
#[cfg(feature = "parallel")]
#[test]
fn a_strided_plan_past_l2_forks_untiled_at_its_own_rate() {
    use threadteam::ledger;
    // A class with no beyond-L2 copy rate prices every plan at the row, so there is
    // no fork to attribute to the plan's rate. Which class that is depends on the
    // machine; the policy is pinned host-independently by
    // `cost::tests::a_footprint_past_l2_is_priced_above_the_row_of_its_class` and
    // the mechanism by
    // `the_splitter_asks_its_gate_at_the_rate_it_is_given`.
    if crate::cost::copy_ns_per_element_for_footprint(
        threadteam::HostClass::detect(),
        usize::MAX >> 8,
    ) == KernelClass::Copy.ns_per_element()
    {
        eprintln!("skipped: this class has no beyond-L2 copy rate");
        return;
    }
    let pool = crate::threading::test_pool(4);
    let dims = [1024usize];
    let strides = [256isize];
    let len = 1 + 1023 * 256;
    // 8 bytes * 2 * 261889 = 4.0 MiB of source and destination.
    assert!(
        plan_footprint(&dims, &strides, &strides, 8) > (1 << 20),
        "the case must sit past the class's L2 for the rate to differ"
    );
    let plan = CopyPlan::compile(&dims, &strides, &strides).unwrap();
    let src = vec![1.5f64; len];
    let mut dst = vec![0.0f64; len];
    let src_ref = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
    let mut dst_mut = RawStridedMut::new(&mut dst, &dims, &strides, 0).unwrap();
    let exec = Exec::new(&pool, 4).unwrap();
    let root = exec.ledger_root().unwrap();
    plan.execute(&mut dst_mut, &src_ref, exec).unwrap();
    assert!(
        ledger::installs(root) > 0,
        "the plan forked nowhere: the replay priced it at the row, not at {:.3} ns per element",
        plan.ns_per_element::<f64, f64>()
    );
    assert!(ledger::max_live(root) <= 4);
    assert_eq!(dst[0], 1.5);
    assert_eq!(dst[1023 * 256], 1.5);
}

/// The same, on the tiled replay of strided-rs#422: a 128 x 32 transposing copy
/// (2.6 MiB) is 700 ns at the row and 12.3 us at the plan's rate, against a pool
/// whose section costs 3 us for the first fork, so only the plan's rate opens the
/// split at all.
#[cfg(feature = "parallel")]
#[test]
fn a_strided_plan_past_l2_splits_its_tiled_replay_at_its_own_rate() {
    use threadteam::{DispatchCosts, DispatchKind, HostClass, Pool, WidthCosts};
    // A class with no beyond-L2 copy rate prices every plan at the row, so there is
    // no fork to attribute to the plan's rate. Which class that is depends on the
    // machine; the policy is pinned host-independently by
    // `cost::tests::a_footprint_past_l2_is_priced_above_the_row_of_its_class` and
    // the mechanism by
    // `the_splitter_asks_its_gate_at_the_rate_it_is_given`.
    if crate::cost::copy_ns_per_element_for_footprint(
        threadteam::HostClass::detect(),
        usize::MAX >> 8,
    ) == KernelClass::Copy.ns_per_element()
    {
        eprintln!("skipped: this class has no beyond-L2 copy rate");
        return;
    }
    let micro = core::time::Duration::from_micros(1);
    let flat = WidthCosts::flat(micro);
    let costs = DispatchCosts::for_class(HostClass::Generic)
        .with(DispatchKind::Team, flat)
        .with(DispatchKind::InPool, flat)
        .with(DispatchKind::Outside, flat);
    let pool = Pool::builder()
        .threads(4)
        .dispatch_costs(costs)
        .build()
        .unwrap();
    let dims = [128usize, 32];
    let dst_strides = [1isize, 2048];
    let src_strides = [2048isize, 1];
    let len = 128 * 2048;
    assert!(
        plan_footprint(&dims, &dst_strides, &src_strides, 8) > (1 << 20),
        "the case must sit past the class's L2 for the rate to differ"
    );
    let plan = CopyPlan::compile(&dims, &dst_strides, &src_strides).unwrap();
    // The layout must reach the tiled replay, or this test proves nothing.
    assert!(
        plan.fused
            .inline()
            .and_then(pair_tiles::<f64, f64>)
            .is_some(),
        "the layout must have tiles for `replay_tiled` to see the rate"
    );
    let src = vec![1.5f64; len];
    let mut dst = vec![0.0f64; len];
    let src_ref = RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap();
    let mut dst_mut = RawStridedMut::new(&mut dst, &dims, &dst_strides, 0).unwrap();
    let exec = Exec::new(&pool, 4).unwrap();
    let root = exec.ledger_root().unwrap();
    crate::threading::take_tiled_leaves();
    plan.execute(&mut dst_mut, &src_ref, exec).unwrap();
    let leaves = crate::threading::take_tiled_leaves();
    let installs = threadteam::ledger::installs(root);
    // More than one leaf: the *splitter* forked at the plan's rate. One leaf is
    // what a splitter gated at the row records, and an empty list is what a gate at
    // the row records before it.
    assert!(
        leaves.len() > 1,
        "the splitter did not fork: {} leaf/leaves and {} worker install(s) for a plan priced at \
         {:.3} ns per element",
        leaves.len(),
        installs,
        plan.ns_per_element::<f64, f64>()
    );
    assert!(
        installs > 0,
        "no worker was installed: the splitter asked at the row, not at {:.3} ns per element",
        plan.ns_per_element::<f64, f64>()
    );
    assert_eq!(dst[0], 1.5);
    assert_eq!(dst[127 + 31 * 2048], 1.5);
}

/// A dense reversed buffer over several axes streams as one run too: fusion
/// sorts by signed destination stride, so `[-1, -512]` over `[512, 512]` keeps
/// two axes and the rank-one check alone priced it as strided ([#364]).
#[test]
fn a_reversed_dense_buffer_keeps_the_copy_row() {
    let dims = [512usize, 512];
    let strides = [-1isize, -512];
    let plan = CopyPlan::compile(&dims, &strides, &strides).unwrap();
    assert!(
        plan.is_contiguous_run(),
        "a dense reversed buffer is one streamed run"
    );
    assert_eq!(
        plan.ns_per_element::<f64, f64>(),
        crate::cost::KernelClass::Copy.ns_per_element(),
        "the copy row is the calibrated rate of a streamed copy"
    );
    // A sign flip between the axes makes the walk jump: two runs, priced as strided.
    let mixed = CopyPlan::compile(&dims, &[-1isize, 512], &strides).unwrap();
    assert!(!mixed.is_contiguous_run(), "mixed signs are not one run");
}

/// A reversed unit-stride run streams like a forward one, so it keeps the `Copy`
/// row even when its working set is past the class's L2: 131072 `f64` at stride
/// -1 on both sides is 2 MiB together. Treating it as strided would price it at
/// 3.0 ns per element (strided-rs#364).
#[test]
fn a_reversed_run_past_l2_keeps_the_copy_row() {
    let dims = [131_072usize];
    let strides = [-1isize];
    assert!(
        plan_footprint(&dims, &strides, &strides, 8) > (1 << 20),
        "the reversed case must sit past the class's L2 for the exemption to matter"
    );
    let plan = CopyPlan::compile(&dims, &strides, &strides).unwrap();
    assert_eq!(
        plan.ns_per_element::<f64, f64>(),
        KernelClass::Copy.ns_per_element(),
        "a reversed single run is a streamed copy, not a strided one"
    );
}

/// The splitter asks its forks at the rate it is given, not at a class's: 1024
/// elements are 92-175 ns at any class's row (below the 300 ns this test pool asks
/// for a fork) and 3.1 us at 3.0 ns per element (above it), on every host. This is
/// the seam the tiled replay of strided-rs#422 uses, so it is pinned where
/// the plans themselves cannot be (a class without a beyond-L2 rate prices every
/// plan at the row).
#[cfg(feature = "parallel")]
#[test]
fn the_splitter_asks_its_gate_at_the_rate_it_is_given() {
    use threadteam::ledger;
    let pool = crate::threading::test_pool(4);
    let dims = [32usize, 32];
    let blocks = [8usize, 8];
    let strides = [vec![1isize, 4096], vec![4096isize, 1]];
    let costs = crate::fuse::compute_costs(&[&strides[0][..], &strides[1][..]]);
    let mut installs = Vec::new();
    for rate in [KernelClass::Copy.ns_per_element(), 3.0] {
        let exec = Exec::new(&pool, 4).unwrap();
        let root = exec.ledger_root().unwrap();
        crate::threading::mapreduce_threaded_at_rate(
            &dims,
            &blocks,
            &strides,
            &[0, 0],
            &costs,
            rate,
            exec,
            &|_, _, _, _| Ok(()),
        )
        .unwrap();
        installs.push(ledger::installs(root));
    }
    assert_eq!(
        installs[0], 0,
        "1024 elements at the row must not pay this pool's fork"
    );
    assert!(
        installs[1] > 0,
        "1024 elements at 3.0 ns per element must split, whatever the class"
    );
}
