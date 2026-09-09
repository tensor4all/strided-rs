use super::*;
use strided_view::auxiliary::index_order;

#[test]
fn test_fuse_dims_contiguous() {
    // Two contiguous dimensions: [3, 4] with strides [1, 3] -> fused to [12, 1]
    let dims = [3, 4];
    let strides1 = [1isize, 3];
    let strides2 = [1isize, 3];
    let all_strides: Vec<&[isize]> = vec![&strides1, &strides2];

    let fused = fuse_dims(&dims, &all_strides);
    assert_eq!(fused, vec![12, 1]);
}

#[test]
fn test_fuse_dims_allows_broadcast_operand_across_fused_axes() {
    let dims = [16usize, 16, 64, 64];
    let out = [1isize, 16, 256, 16_384];
    let lhs = [1isize, 16, 0, 256];
    let rhs = [0isize, 0, 1, 64];
    let all_strides: Vec<&[isize]> = vec![&out, &lhs, &rhs];

    let fused = fuse_dims(&dims, &all_strides);

    assert_eq!(fused, vec![256, 1, 64, 64]);
}

#[test]
fn test_fuse_dims_non_contiguous() {
    // Non-contiguous: strides don't match
    let dims = [3, 4];
    let strides1 = [1isize, 10]; // Not contiguous (should be 3)
    let all_strides: Vec<&[isize]> = vec![&strides1];

    let fused = fuse_dims(&dims, &all_strides);
    assert_eq!(fused, vec![3, 4]); // No fusion
}

#[test]
fn test_fuse_dims_partial() {
    // 3D: first two fuse, third doesn't
    let dims = [2, 3, 4];
    let strides = [1isize, 2, 100]; // dims[0]*strides[0]=2=strides[1], but 6≠100
    let all_strides: Vec<&[isize]> = vec![&strides];

    let fused = fuse_dims(&dims, &all_strides);
    assert_eq!(fused, vec![6, 1, 4]); // Fused first two
}

#[test]
fn test_fuse_dims_multiple_arrays() {
    // Only fuse if ALL arrays are contiguous
    let dims = [3, 4];
    let strides1 = [1isize, 3]; // Contiguous
    let strides2 = [1isize, 10]; // Not contiguous
    let all_strides: Vec<&[isize]> = vec![&strides1, &strides2];

    let fused = fuse_dims(&dims, &all_strides);
    assert_eq!(fused, vec![3, 4]); // No fusion because strides2 isn't contiguous
}

#[test]
fn test_compute_importance_2_arrays() {
    // Example with 2 arrays, dims [4, 5]
    let dims = [4usize, 5];
    let strides1 = [1isize, 4]; // Column-major output
    let strides2 = [5isize, 1]; // Row-major input
    let all_strides: Vec<&[isize]> = vec![&strides1, &strides2];

    let order1 = index_order(&strides1);
    let order2 = index_order(&strides2);
    let index_orders = vec![order1, order2];

    let importance = compute_importance(&dims, &all_strides, &index_orders);

    // With output weighted 2x, dimension 0 should have higher importance
    // since it has smaller stride in the output array
    assert!(importance[0] > importance[1]);
}

#[test]
fn test_sort_by_importance() {
    let importance = vec![100u64, 50, 200, 10];
    let perm = sort_by_importance(&importance);
    assert_eq!(perm, vec![2, 0, 1, 3]); // Indices sorted by descending importance
}

#[test]
fn test_compute_costs() {
    let strides1 = [1isize, 4, 0];
    let strides2 = [2isize, 1, 0];
    let all_strides: Vec<&[isize]> = vec![&strides1, &strides2];

    let costs = compute_costs(&all_strides);
    // min strides: [1, 1, 0], transformed: [2, 2, 1]
    assert_eq!(costs, vec![2, 2, 1]);
}

#[test]
fn test_compute_importance_with_zero_stride() {
    // Zero stride (broadcast) still gets index_order = 1 for compatibility,
    // but it does not contribute to ordering importance.
    let dims = [4usize, 5];
    let strides1 = [0isize, 1]; // First dim is broadcast
    let all_strides: Vec<&[isize]> = vec![&strides1];

    let order1 = index_order(&strides1);
    // For stride 0: order = 1 (zero strides always get 1)
    // For stride 1: order = 1 (no non-zero stride < 1)
    assert_eq!(order1, vec![1, 1]);

    let index_orders = vec![order1];
    let importance = compute_importance(&dims, &all_strides, &index_orders);

    assert_eq!(importance[0], 0);
    assert!(importance[1] > 0);
}

#[test]
fn test_compute_importance_size_one_dim() {
    // Size-1 dimensions get zero importance
    let dims = [4usize, 1, 5];
    let strides1 = [1isize, 4, 4];
    let all_strides: Vec<&[isize]> = vec![&strides1];

    let order1 = index_order(&strides1);
    let index_orders = vec![order1];
    let importance = compute_importance(&dims, &all_strides, &index_orders);

    // Dimension 1 has size 1 -> importance = 0
    assert_eq!(importance[1], 0);
    // Other dimensions should have non-zero importance
    assert!(importance[0] > 0);
    assert!(importance[2] > 0);
}

#[test]
fn test_compute_importance_output_weight() {
    // Output (first array) is strongly weighted
    // With same strides, dimension with smaller stride in output wins
    let dims = [4usize, 5];
    let out_strides = [1isize, 4]; // Column-major output
    let in_strides = [5isize, 1]; // Row-major input
    let all_strides: Vec<&[isize]> = vec![&out_strides, &in_strides];

    let order_out = index_order(&out_strides); // [1, 2]
    let order_in = index_order(&in_strides); // [2, 1]
    let index_orders = vec![order_out, order_in];

    let importance = compute_importance(&dims, &all_strides, &index_orders);

    // Output weighting makes dimension 0 (smaller stride in output) win.
    assert!(importance[0] > importance[1]);
}

#[test]
fn test_compute_costs_owned_vecs() {
    // Ported from threading.rs: verify compute_costs works with Vec<Vec<isize>>
    let strides_list: Vec<Vec<isize>> = vec![vec![1, 0, 3], vec![2, 0, 4]];
    let costs = compute_costs(&strides_list);
    assert_eq!(costs, vec![2, 1, 6]);
}

#[test]
fn test_compute_costs_with_zero() {
    // Zero strides become cost 1, non-zero become 2*abs
    let strides1 = [0isize, 2, -3];
    let strides2 = [1isize, 0, 2];
    let all_strides: Vec<&[isize]> = vec![&strides1, &strides2];

    let costs = compute_costs(&all_strides);
    // min abs: [0, 0, 2]
    // transform: [1, 1, 4]
    assert_eq!(costs, vec![1, 1, 4]);
}

// ---- compress_dims tests ----

#[test]
fn test_compress_dims_removes_fused() {
    let dims = vec![12usize, 1];
    let strides = vec![vec![1isize, 3]];
    let (cd, cs) = compress_dims(&dims, &strides);
    assert_eq!(cd, vec![12]);
    assert_eq!(cs, vec![vec![1]]);
}

#[test]
fn test_compress_dims_removes_multiple() {
    let dims = vec![6usize, 1, 4];
    let strides = vec![vec![1isize, 2, 100]];
    let (cd, cs) = compress_dims(&dims, &strides);
    assert_eq!(cd, vec![6, 4]);
    assert_eq!(cs, vec![vec![1, 100]]);
}

#[test]
fn test_compress_dims_no_removal() {
    let dims = vec![3usize, 4];
    let strides = vec![vec![1isize, 3]];
    let (cd, cs) = compress_dims(&dims, &strides);
    assert_eq!(cd, vec![3, 4]);
    assert_eq!(cs, vec![vec![1, 3]]);
}

#[test]
fn test_compress_dims_all_ones() {
    let dims = vec![1usize, 1, 1];
    let strides = vec![vec![1isize, 1, 1]];
    let (cd, cs) = compress_dims(&dims, &strides);
    assert_eq!(cd, vec![1]);
    assert_eq!(cs, vec![vec![1]]);
}

#[test]
fn test_compress_dims_multi_arrays() {
    let dims = vec![6usize, 1, 4];
    let strides = vec![vec![1isize, 6, 6], vec![4isize, 24, 1]];
    let (cd, cs) = compress_dims(&dims, &strides);
    assert_eq!(cd, vec![6, 4]);
    assert_eq!(cs, vec![vec![1, 6], vec![4, 1]]);
}

#[test]
fn test_compress_dims_single_dim() {
    let dims = vec![5usize];
    let strides = vec![vec![1isize]];
    let (cd, cs) = compress_dims(&dims, &strides);
    assert_eq!(cd, vec![5]);
    assert_eq!(cs, vec![vec![1]]);
}

#[test]
fn test_compress_dims_single_dim_one() {
    let dims = vec![1usize];
    let strides = vec![vec![1isize]];
    let (cd, cs) = compress_dims(&dims, &strides);
    assert_eq!(cd, vec![1]);
    assert_eq!(cs, vec![vec![1]]);
}

#[test]
fn test_compress_dims_empty() {
    let dims: Vec<usize> = vec![];
    let strides: Vec<Vec<isize>> = vec![];
    let (cd, cs) = compress_dims(&dims, &strides);
    assert!(cd.is_empty());
    assert!(cs.is_empty());
}
