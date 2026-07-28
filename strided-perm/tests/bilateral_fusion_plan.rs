use strided_perm::{plan_bilateral_fusion, BilateralFusionPlan, FusionPlanError};

fn plan(dims: &[usize], src_strides: &[isize], dst_strides: &[isize]) -> BilateralFusionPlan {
    plan_bilateral_fusion(dims, src_strides, dst_strides).unwrap()
}

#[test]
fn empty_metadata_stays_rank_zero() {
    assert_eq!(
        plan(&[], &[], &[]),
        BilateralFusionPlan {
            dims: vec![],
            src_strides: vec![],
            dst_strides: vec![],
        }
    );
}

#[test]
fn size_one_axes_are_removed_including_scalar_like_shapes() {
    assert_eq!(
        plan(&[1, 1], &[7, -3], &[9, 4]),
        BilateralFusionPlan {
            dims: vec![],
            src_strides: vec![],
            dst_strides: vec![],
        }
    );
    assert_eq!(
        plan(&[1, 5, 1], &[100, 1, -20], &[80, 1, 9]),
        BilateralFusionPlan {
            dims: vec![5],
            src_strides: vec![1],
            dst_strides: vec![1],
        }
    );
}

#[test]
fn identity_layout_collapses_to_one_axis() {
    assert_eq!(
        plan(&[2, 3, 4], &[1, 2, 6], &[1, 2, 6]),
        BilateralFusionPlan {
            dims: vec![24],
            src_strides: vec![1],
            dst_strides: vec![1],
        }
    );
}

#[test]
fn only_bilaterally_contiguous_axes_fuse() {
    assert_eq!(
        plan(&[2, 3, 4], &[1, 2, 100], &[1, 2, 6]),
        BilateralFusionPlan {
            dims: vec![6, 4],
            src_strides: vec![1, 100],
            dst_strides: vec![1, 6],
        }
    );
    assert_eq!(
        plan(&[2, 3, 4], &[1, 10, 40], &[1, 2, 6]),
        BilateralFusionPlan {
            dims: vec![2, 3, 4],
            src_strides: vec![1, 10, 40],
            dst_strides: vec![1, 2, 6],
        }
    );
}

#[test]
fn two_dimensional_transpose_does_not_fuse() {
    assert_eq!(
        plan(&[3, 2], &[2, 1], &[1, 3]),
        BilateralFusionPlan {
            dims: vec![3, 2],
            src_strides: vec![2, 1],
            dst_strides: vec![1, 3],
        }
    );
}

#[test]
fn negative_contiguous_strides_can_fuse() {
    assert_eq!(
        plan(&[2, 3], &[-1, -2], &[1, 2]),
        BilateralFusionPlan {
            dims: vec![6],
            src_strides: vec![-1],
            dst_strides: vec![1],
        }
    );
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
        BilateralFusionPlan {
            dims: vec![dims.iter().product()],
            src_strides: vec![1],
            dst_strides: vec![1],
        }
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
