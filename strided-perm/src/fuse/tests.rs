use super::*;

#[test]
fn test_fuse_dims_contiguous() {
    let dims = [3, 4];
    let strides1 = [1isize, 3];
    let strides2 = [1isize, 3];
    let all_strides: Vec<&[isize]> = vec![&strides1, &strides2];
    let fused = fuse_dims(&dims, &all_strides);
    assert_eq!(fused, vec![12, 1]);
}

#[test]
fn test_fuse_dims_non_contiguous() {
    let dims = [3, 4];
    let strides1 = [1isize, 10];
    let all_strides: Vec<&[isize]> = vec![&strides1];
    let fused = fuse_dims(&dims, &all_strides);
    assert_eq!(fused, vec![3, 4]);
}

#[test]
fn test_fuse_dims_bilateral_all_contiguous() {
    // 24-dim all-size-2 col-major: all contiguous -> fuses to single dim
    let dims = vec![2, 2, 2, 2];
    let src_strides = vec![1, 2, 4, 8];
    let dst_strides = vec![1, 2, 4, 8];
    let (fd, fs, fds) = fuse_dims_bilateral(&dims, &src_strides, &dst_strides);
    assert_eq!(&fd[..], &[16]);
    assert_eq!(&fs[..], &[1]);
    assert_eq!(&fds[..], &[1]);
}

#[test]
fn test_fuse_dims_bilateral_partial() {
    // src: contiguous 0-1, not 1-2
    // dst: contiguous 0-1-2
    let dims = vec![2, 3, 4];
    let src_strides = vec![1, 2, 100]; // 0-1 contiguous, 1-2 not
    let dst_strides = vec![1, 2, 6]; // all contiguous
    let (fd, fs, fds) = fuse_dims_bilateral(&dims, &src_strides, &dst_strides);
    assert_eq!(&fd[..], &[6, 4]); // first two fuse
    assert_eq!(&fs[..], &[1, 100]);
    assert_eq!(&fds[..], &[1, 6]);
}

#[test]
fn test_fuse_dims_bilateral_scattered() {
    // The benchmark case: scattered strides, nothing fuses
    let dims = vec![2, 2, 2];
    let src_strides = vec![1, 4194304, 2]; // scattered
    let dst_strides = vec![1, 2, 4]; // contiguous
    let (fd, fs, fds) = fuse_dims_bilateral(&dims, &src_strides, &dst_strides);
    assert_eq!(&fd[..], &[2, 2, 2]); // nothing fuses
    assert_eq!(&fs[..], &[1, 4194304, 2]);
    assert_eq!(&fds[..], &[1, 2, 4]);
}

#[test]
fn test_compress_dims_removes_fused() {
    let dims = vec![12usize, 1];
    let strides = vec![vec![1isize, 3]];
    let (cd, cs) = compress_dims(&dims, &strides);
    assert_eq!(cd, vec![12]);
    assert_eq!(cs, vec![vec![1]]);
}

#[test]
fn test_compute_costs() {
    let strides1 = [1isize, 4, 0];
    let strides2 = [2isize, 1, 0];
    let all_strides: Vec<&[isize]> = vec![&strides1, &strides2];
    let costs = compute_costs(&all_strides);
    assert_eq!(costs, vec![2, 2, 1]);
}
