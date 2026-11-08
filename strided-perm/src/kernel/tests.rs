use super::*;

#[test]
fn test_build_plan_fused_compresses() {
    let dims = [2usize, 3];
    let strides = [1isize, 2];
    let strides_list: Vec<&[isize]> = vec![&strides];
    let (fused_dims, fused_strides, plan) = build_plan_fused(&dims, &strides_list, Some(0), 8);
    assert_eq!(fused_dims, vec![6]);
    assert_eq!(fused_strides.len(), 1);
    assert_eq!(fused_strides[0], vec![1]);
    assert_eq!(plan.block.len(), 1);
}

#[test]
fn test_for_each_inner_block_preordered_total() {
    let dims = vec![3, 4, 2];
    let blocks = vec![3, 4, 2];
    let strides = vec![vec![1, 3, 12], vec![1, 3, 12]];
    let offsets = vec![0, 0];
    let mut total = 0usize;
    for_each_inner_block_preordered(&dims, &blocks, &strides, &offsets, |_, len, _| {
        total += len;
        Ok(())
    })
    .unwrap();
    assert_eq!(total, 24);
}

#[test]
fn test_build_plan_fused_non_contiguous() {
    // Row-major [4,5]: strides [5, 1]
    let dims = [4usize, 5];
    let strides = [5isize, 1];
    let strides_list: Vec<&[isize]> = vec![&strides];
    let (fused_dims, fused_strides, plan) = build_plan_fused(&dims, &strides_list, Some(0), 8);
    // Row-major: should reorder so stride-1 is first, then fuse to 20
    assert_eq!(fused_dims, vec![20]);
    assert_eq!(fused_strides[0], vec![1]);
    assert_eq!(plan.block.len(), 1);
}

#[test]
fn test_build_plan_fused_multi_array() {
    // Two arrays: one col-major, one row-major
    let dims = [4usize, 5];
    let dst_strides = [1isize, 4];
    let src_strides = [5isize, 1];
    let strides_list: Vec<&[isize]> = vec![&dst_strides, &src_strides];
    let (fused_dims, fused_strides, plan) = build_plan_fused(&dims, &strides_list, Some(0), 8);
    // Conflicting strides means no fusion possible
    assert_eq!(fused_dims.len(), 2);
    assert_eq!(fused_strides.len(), 2);
    assert!(!plan.block.is_empty());
}

#[test]
fn test_build_plan_fused_small_basic() {
    let dims = [2usize, 3];
    let strides = [1isize, 2];
    let strides_list: Vec<&[isize]> = vec![&strides];
    let (fused_dims, fused_strides, plan) = build_plan_fused_small(&dims, &strides_list);
    assert_eq!(fused_dims, vec![6]);
    assert_eq!(fused_strides[0], vec![1]);
    // Small plan: block = dims (no blocking)
    assert_eq!(plan.block, fused_dims);
}

#[test]
fn test_build_plan_fused_small_non_contiguous() {
    let dims = [4usize, 5];
    let strides = [5isize, 1];
    let strides_list: Vec<&[isize]> = vec![&strides];
    let (fused_dims, fused_strides, plan) = build_plan_fused_small(&dims, &strides_list);
    // No ordering in small path, but fusion should still work on contiguous groups
    assert!(!fused_dims.is_empty());
    assert_eq!(fused_strides.len(), 1);
    assert_eq!(plan.block, fused_dims);
}

#[test]
fn test_for_each_rank0() {
    let dims = vec![];
    let blocks = vec![];
    let strides: Vec<Vec<isize>> = vec![vec![]];
    let offsets = vec![0isize];
    let mut called = false;
    for_each_inner_block_preordered(&dims, &blocks, &strides, &offsets, |_, len, _| {
        called = true;
        assert_eq!(len, 1);
        Ok(())
    })
    .unwrap();
    assert!(called);
}

/// Helper: count total elements iterated for given rank
fn count_elements(dims: &[usize], blocks: &[usize]) -> usize {
    let n_arrays = 1;
    let strides: Vec<Vec<isize>> = {
        let mut s = vec![vec![0isize; dims.len()]; n_arrays];
        let mut stride = 1isize;
        for d in 0..dims.len() {
            for row in &mut s {
                row[d] = stride;
            }
            stride *= dims[d] as isize;
        }
        s
    };
    let offsets = vec![0isize; n_arrays];
    let mut total = 0usize;
    for_each_inner_block_preordered(dims, blocks, &strides, &offsets, |_, len, _| {
        total += len;
        Ok(())
    })
    .unwrap();
    total
}

#[test]
fn test_for_each_rank4() {
    assert_eq!(count_elements(&[2, 3, 4, 5], &[2, 3, 4, 5]), 120);
}

#[test]
fn test_for_each_rank5() {
    assert_eq!(count_elements(&[2, 2, 2, 2, 2], &[2, 2, 2, 2, 2]), 32);
}

#[test]
fn test_for_each_rank6() {
    assert_eq!(count_elements(&[2, 2, 2, 2, 2, 3], &[2, 2, 2, 2, 2, 3]), 96);
}

#[test]
fn test_for_each_rank7() {
    assert_eq!(
        count_elements(&[2, 2, 2, 2, 2, 2, 3], &[2, 2, 2, 2, 2, 2, 3]),
        192
    );
}

#[test]
fn test_for_each_rank8() {
    assert_eq!(
        count_elements(&[2, 2, 2, 2, 2, 2, 2, 3], &[2, 2, 2, 2, 2, 2, 2, 3]),
        384
    );
}

#[test]
fn test_for_each_rank9_iterative() {
    // Rank 9 triggers kernel_nd_inner_iterative
    assert_eq!(
        count_elements(&[2, 2, 2, 2, 2, 2, 2, 2, 3], &[2, 2, 2, 2, 2, 2, 2, 2, 3]),
        768
    );
}

#[test]
fn test_for_each_rank10_iterative() {
    assert_eq!(
        count_elements(
            &[2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
            &[2, 2, 2, 2, 2, 2, 2, 2, 2, 3]
        ),
        1536
    );
}

#[test]
fn test_total_len_empty() {
    assert_eq!(total_len(&[]), 1);
}

#[test]
fn test_total_len_basic() {
    assert_eq!(total_len(&[2, 3, 4]), 24);
}
