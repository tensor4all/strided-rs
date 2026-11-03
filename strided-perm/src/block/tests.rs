use super::*;

#[test]
fn test_total_memory_region_contiguous() {
    let dims = [100usize];
    let strides = [8isize];
    let byte_strides: Vec<&[isize]> = vec![&strides];
    let region = total_memory_region(&dims, &byte_strides);
    assert_eq!(region, 832);
}

#[test]
fn test_compute_blocks_small() {
    let dims = [10usize, 10];
    let costs = [2isize, 2];
    let strides = [8isize, 80];
    let orders = [1usize, 2];
    let byte_strides: Vec<&[isize]> = vec![&strides];
    let stride_orders: Vec<&[usize]> = vec![&orders];
    let blocks = compute_blocks(
        &dims,
        &costs,
        &byte_strides,
        &stride_orders,
        BLOCK_MEMORY_SIZE,
    );
    assert_eq!(blocks, vec![10, 10]);
}

#[test]
fn test_compute_blocks_large() {
    let dims = [1000usize, 1000];
    let costs = [2isize, 2];
    let strides = [8isize, 8000];
    let orders = [1usize, 2];
    let byte_strides: Vec<&[isize]> = vec![&strides];
    let stride_orders: Vec<&[usize]> = vec![&orders];
    let blocks = compute_blocks(
        &dims,
        &costs,
        &byte_strides,
        &stride_orders,
        BLOCK_MEMORY_SIZE,
    );
    assert!(blocks[0] <= dims[0]);
    assert!(blocks[1] <= dims[1]);
    assert!(blocks[0] >= 1);
    assert!(blocks[1] >= 1);
}

#[test]
fn test_compute_blocks_empty() {
    let blocks = compute_blocks(&[], &[], &[], &[], BLOCK_MEMORY_SIZE);
    assert!(blocks.is_empty());
}

#[test]
fn test_last_argmax_weighted_basic() {
    // blocks [10, 5], costs [2, 4]
    // scores: (10-1)*2=18, (5-1)*4=16 → first wins
    assert_eq!(last_argmax_weighted(&[10, 5], &[2, 4]), Some(0));
}

#[test]
fn test_last_argmax_weighted_ties() {
    // Equal scores: last wins (>= semantics)
    assert_eq!(last_argmax_weighted(&[5, 5], &[2, 2]), Some(1));
}

#[test]
fn test_last_argmax_weighted_all_one() {
    // All blocks are 1: no valid candidate
    assert_eq!(last_argmax_weighted(&[1, 1, 1], &[1, 1, 1]), None);
}

#[test]
fn test_last_argmax_weighted_empty() {
    assert_eq!(last_argmax_weighted(&[], &[]), None);
}

#[test]
fn test_total_memory_region_multi_array() {
    // Two arrays with different strides
    let dims = [100usize, 100];
    let s1 = [8isize, 800]; // col-major f64
    let s2 = [800isize, 8]; // row-major f64
    let byte_strides: Vec<&[isize]> = vec![&s1, &s2];
    let region = total_memory_region(&dims, &byte_strides);
    // Should sum contributions from both arrays
    assert!(region > 0);
}

#[test]
fn test_total_memory_region_large_stride() {
    // Stride >= cache line triggers block multiplication
    let dims = [10usize, 10];
    let strides = [8isize, 800]; // second dim stride 800 >= 64
    let byte_strides: Vec<&[isize]> = vec![&strides];
    let region = total_memory_region(&dims, &byte_strides);
    assert!(region > 0);
}

#[test]
fn test_compute_blocks_min_stride_exceeds_block_size() {
    // Both strides very large (> block_size=64): min_stride > block_size → all blocks = 1
    // Two arrays with conflicting stride orders to prevent recursive tail path
    let dims = [10usize, 10];
    let costs = [1isize, 1];
    let s1 = [100000isize, 1000000]; // array 1: order [0, 1]
    let s2 = [1000000isize, 100000]; // array 2: order [1, 0]
    let o1 = [0usize, 1];
    let o2 = [1usize, 0]; // conflicting: o1[0]=0, o2[0]=1, min_order=0 but o2[0]!=0
    let byte_strides: Vec<&[isize]> = vec![&s1, &s2];
    let stride_orders: Vec<&[usize]> = vec![&o1, &o2];
    let blocks = compute_blocks(&dims, &costs, &byte_strides, &stride_orders, 64);
    assert_eq!(blocks, vec![1, 1]);
}

#[test]
fn test_compute_block_sizes_3d() {
    // 3D col-major
    let dims = [10usize, 20, 30];
    let order = [0usize, 1, 2];
    let strides = [8isize, 80, 1600];
    let strides_list: Vec<&[isize]> = vec![&strides];
    let blocks = compute_block_sizes(&dims, &order, &strides_list, 8);
    assert_eq!(blocks.len(), 3);
    for i in 0..3 {
        assert!(blocks[i] >= 1 && blocks[i] <= dims[i]);
    }
}

#[test]
fn test_compute_blocks_first_stride_order_matches() {
    // stride_orders[0][0] == min_order → triggers recursive tail path
    let dims = [4usize, 100, 100];
    let costs = [1isize, 10, 10];
    // Stride order: [0, 1, 2] and min_order = 0
    let strides = [8isize, 32, 3200];
    let orders = [0usize, 1, 2];
    let byte_strides: Vec<&[isize]> = vec![&strides];
    let stride_orders: Vec<&[usize]> = vec![&orders];
    let blocks = compute_blocks(
        &dims,
        &costs,
        &byte_strides,
        &stride_orders,
        BLOCK_MEMORY_SIZE,
    );
    // First dim should be kept at full extent
    assert_eq!(blocks[0], 4);
}
