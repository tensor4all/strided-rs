use super::*;

#[test]
fn test_fuse_dims_bilateral_all_contiguous() {
    // 24-dim all-size-2 col-major: all contiguous -> fuses to single dim
    let dims = vec![2, 2, 2, 2];
    let src_strides = vec![1, 2, 4, 8];
    let dst_strides = vec![1, 2, 4, 8];
    let plan = plan_bilateral_fusion(&dims, &src_strides, &dst_strides).unwrap();
    assert_eq!(&plan.dims[..], &[16]);
    assert_eq!(&plan.src_strides[..], &[1]);
    assert_eq!(&plan.dst_strides[..], &[1]);
}

#[test]
fn test_fuse_dims_bilateral_partial() {
    // src: contiguous 0-1, not 1-2
    // dst: contiguous 0-1-2
    let dims = vec![2, 3, 4];
    let src_strides = vec![1, 2, 100]; // 0-1 contiguous, 1-2 not
    let dst_strides = vec![1, 2, 6]; // all contiguous
    let plan = plan_bilateral_fusion(&dims, &src_strides, &dst_strides).unwrap();
    assert_eq!(&plan.dims[..], &[6, 4]); // first two fuse
    assert_eq!(&plan.src_strides[..], &[1, 100]);
    assert_eq!(&plan.dst_strides[..], &[1, 6]);
}

#[test]
fn test_fuse_dims_bilateral_scattered() {
    // The benchmark case: scattered strides, nothing fuses
    let dims = vec![2, 2, 2];
    let src_strides = vec![1, 4194304, 2]; // scattered
    let dst_strides = vec![1, 2, 4]; // contiguous
    let plan = plan_bilateral_fusion(&dims, &src_strides, &dst_strides).unwrap();
    assert_eq!(&plan.dims[..], &[2, 2, 2]); // nothing fuses
    assert_eq!(&plan.src_strides[..], &[1, 4194304, 2]);
    assert_eq!(&plan.dst_strides[..], &[1, 2, 4]);
}
