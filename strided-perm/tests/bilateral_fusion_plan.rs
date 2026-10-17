use strided_perm::{plan_bilateral_fusion, BilateralFusionPlan, FusionPlanError};

fn plan(dims: &[usize], src_strides: &[isize], dst_strides: &[isize]) -> BilateralFusionPlan {
    plan_bilateral_fusion(dims, src_strides, dst_strides).unwrap()
}

/// Build the inline-vector expected value for a fusion plan.
fn fusion(dims: &[usize], src_strides: &[isize], dst_strides: &[isize]) -> BilateralFusionPlan {
    BilateralFusionPlan {
        dims: dims.iter().copied().collect(),
        src_strides: src_strides.iter().copied().collect(),
        dst_strides: dst_strides.iter().copied().collect(),
    }
}

#[test]
fn empty_metadata_stays_rank_zero() {
    assert_eq!(plan(&[], &[], &[]), fusion(&[], &[], &[]));
}

#[test]
fn size_one_axes_are_removed_including_scalar_like_shapes() {
    assert_eq!(plan(&[1, 1], &[7, -3], &[9, 4]), fusion(&[], &[], &[]));
    assert_eq!(
        plan(&[1, 5, 1], &[100, 1, -20], &[80, 1, 9]),
        fusion(&[5], &[1], &[1])
    );
}

#[test]
fn identity_layout_collapses_to_one_axis() {
    assert_eq!(
        plan(&[2, 3, 4], &[1, 2, 6], &[1, 2, 6]),
        fusion(&[24], &[1], &[1])
    );
}

#[test]
fn only_bilaterally_contiguous_axes_fuse() {
    assert_eq!(
        plan(&[2, 3, 4], &[1, 2, 100], &[1, 2, 6]),
        fusion(&[6, 4], &[1, 100], &[1, 6])
    );
    assert_eq!(
        plan(&[2, 3, 4], &[1, 10, 40], &[1, 2, 6]),
        fusion(&[2, 3, 4], &[1, 10, 40], &[1, 2, 6])
    );
}

#[test]
fn two_dimensional_transpose_does_not_fuse() {
    assert_eq!(
        plan(&[3, 2], &[2, 1], &[1, 3]),
        fusion(&[3, 2], &[2, 1], &[1, 3])
    );
}

#[test]
fn negative_contiguous_strides_can_fuse() {
    assert_eq!(plan(&[2, 3], &[-1, -2], &[1, 2]), fusion(&[6], &[-1], &[1]));
}

#[test]
fn rank_24_contiguous_tensor_collapses_to_one_axis() {
    let mut dims = vec![64];
    dims.extend([2; 23]);
    let mut strides = vec![1isize];
    for axis in 1..dims.len() {
        strides.push(strides[axis - 1] * dims[axis - 1] as isize);
    }
    assert_eq!(
        plan(&dims, &strides, &strides),
        fusion(&[dims.iter().product()], &[1], &[1])
    );
}

#[test]
fn mismatched_metadata_lengths_are_rejected() {
    assert_eq!(
        plan_bilateral_fusion(&[2, 3], &[1], &[1, 2]),
        Err(FusionPlanError::LengthMismatch {
            dims: 2,
            src_strides: 1,
            dst_strides: 2,
        })
    );
}

#[test]
fn unrepresentable_or_overflowing_dimensions_are_rejected() {
    assert_eq!(
        plan_bilateral_fusion(&[usize::MAX, 2], &[1, isize::MAX], &[1, isize::MAX]),
        Err(FusionPlanError::DimensionOverflow)
    );
}
