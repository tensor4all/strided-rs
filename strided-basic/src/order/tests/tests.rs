use super::*;

#[test]
fn test_compute_order_handles_empty_inputs_and_destination_reordering() {
    let dims = [2usize, 3];
    assert_eq!(compute_order(&dims, &[], None), vec![0, 1]);

    let first = [1isize, 2];
    let second = [2isize, 1];
    let strides = vec![&first[..], &second[..]];
    for dest_index in [None, Some(0), Some(1), Some(2)] {
        let order = compute_order(&dims, &strides, dest_index);
        assert_eq!(order.len(), dims.len());
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1]);
    }
}

#[test]
fn test_compute_order_column_major() {
    // Column-major array: strides [1, 4]
    let dims = [4usize, 5];
    let strides = [1isize, 4];
    let strides_list: Vec<&[isize]> = vec![&strides];

    let order = compute_order(&dims, &strides_list, Some(0));

    // Dimension 0 has smallest stride -> highest importance -> first
    assert_eq!(order[0], 0);
    assert_eq!(order[1], 1);
}

#[test]
fn test_compute_order_row_major() {
    // Row-major array: strides [5, 1]
    let dims = [4usize, 5];
    let strides = [5isize, 1];
    let strides_list: Vec<&[isize]> = vec![&strides];

    let order = compute_order(&dims, &strides_list, Some(0));

    // Dimension 1 has smallest stride -> highest importance -> first
    assert_eq!(order[0], 1);
    assert_eq!(order[1], 0);
}

#[test]
fn test_compute_order_mixed() {
    // Output column-major, input row-major
    // Output weighted 2x, so dimension 0 should be first
    let dims = [4usize, 5];
    let out_strides = [1isize, 4]; // Column-major output
    let in_strides = [5isize, 1]; // Row-major input
    let strides_list: Vec<&[isize]> = vec![&out_strides, &in_strides];

    let order = compute_order(&dims, &strides_list, Some(0));

    // Output has 2x weight, so column-major wins -> dim 0 first
    assert_eq!(order[0], 0);
    assert_eq!(order[1], 1);
}

#[test]
fn test_compute_order_3d() {
    // 3D array: want smallest stride dimension first
    let dims = [3usize, 4, 5];
    let strides = [20isize, 5, 1]; // Last dimension is contiguous
    let strides_list: Vec<&[isize]> = vec![&strides];

    let order = compute_order(&dims, &strides_list, Some(0));

    // Dimension 2 has smallest stride -> first in order
    assert_eq!(order[0], 2);
}

#[test]
fn test_compute_order_size_one_dims() {
    // Size-1 dimensions should have zero importance -> go to back
    let dims = [4usize, 1, 5];
    let strides = [1isize, 4, 4];
    let strides_list: Vec<&[isize]> = vec![&strides];

    let order = compute_order(&dims, &strides_list, Some(0));

    // Dimension 1 has size 1 -> should be last
    assert_eq!(order[2], 1);
}

#[test]
fn test_compute_order_empty() {
    let dims: [usize; 0] = [];
    let strides: [isize; 0] = [];
    let strides_list: Vec<&[isize]> = vec![&strides];

    let order = compute_order(&dims, &strides_list, Some(0));
    assert!(order.is_empty());
}

#[test]
fn test_compute_order_with_zero_stride_broadcast() {
    // Zero stride indicates broadcasting and should not pull an axis inward.
    let dims = [4usize, 5, 3];
    let strides = [0isize, 1, 5]; // First dim is broadcast (stride 0)
    let strides_list: Vec<&[isize]> = vec![&strides];

    let order = compute_order(&dims, &strides_list, Some(0));

    assert_eq!(order, vec![1, 2, 0]);
}

#[test]
fn test_compute_order_negative_strides() {
    // Negative strides should be handled correctly
    let dims = [4usize, 5];
    let strides = [-1isize, -4]; // Reversed column-major
    let strides_list: Vec<&[isize]> = vec![&strides];

    let order = compute_order(&dims, &strides_list, Some(0));

    // abs: [1, 4], so dimension 0 has smaller stride -> higher importance -> first
    assert_eq!(order[0], 0);
    assert_eq!(order[1], 1);
}

#[test]
fn test_compute_order_4d_permuted() {
    // 4D array with various strides (Issue #5 related)
    let dims = [2usize, 3, 4, 5];
    let out_strides = [60isize, 20, 5, 1]; // Column-major-ish
    let in_strides = [1isize, 2, 6, 24]; // Row-major-ish
    let strides_list: Vec<&[isize]> = vec![&out_strides, &in_strides];

    let order = compute_order(&dims, &strides_list, Some(0));

    // Output is strongly weighted, so its stride order dominates
    // Output index_order: [4, 3, 2, 1] (60 > 20 > 5 > 1)
    // Input index_order: [1, 2, 3, 4] (1 < 2 < 6 < 24)
    // With output weighting, dimension 3 (stride 1 in output) should be first
    assert_eq!(order[0], 3);
}
