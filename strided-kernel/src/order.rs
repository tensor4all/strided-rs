//! Loop ordering algorithm ported from Strided.jl
//!
//! This module computes the optimal dimension iteration order using
//! the index_order + importance bit-packing algorithm from Julia.

use crate::fuse::{compute_importance, sort_by_importance};
use strided_view::auxiliary::index_order;

/// Compute the optimal iteration order for dimensions.
///
/// This implementation follows Julia's `_mapreduce_order!` algorithm:
/// 1. Compute `index_order` for each array's strides
/// 2. Compute importance scores using bit-packing with output weighted 2x
/// 3. Sort dimensions by importance (descending)
///
/// # Arguments
/// * `dims` - The dimensions of the arrays
/// * `strides_list` - Slice of stride arrays, one per array
/// * `dest_index` - Index of the destination array (weighted 2x, typically 0)
///
/// # Returns
/// Permutation of dimension indices in optimal iteration order
///
/// # Julia equivalent
/// ```julia
/// g = 8 * sizeof(Int) - leading_zeros(M + 1)
/// importance = 2 .* (1 .<< (g .* (N .- indexorder(strides[1]))))
/// for k in 2:M
///     importance = importance .+ (1 .<< (g .* (N .- indexorder(strides[k]))))
/// end
/// importance = importance .* (dims .> 1)
/// p = sortperm(importance; rev=true)
/// ```
pub(crate) fn compute_order(
    dims: &[usize],
    strides_list: &[&[isize]],
    dest_index: Option<usize>,
) -> Vec<usize> {
    let rank = dims.len();
    if rank == 0 {
        return Vec::new();
    }

    if strides_list.is_empty() {
        return (0..rank).collect();
    }

    // Compute index_order for each stride array
    let mut index_orders: Vec<Vec<usize>> = Vec::with_capacity(strides_list.len());
    for strides in strides_list {
        index_orders.push(index_order(strides));
    }

    // Reorder so destination array is first (gets 2x weight)
    let reordered_strides: Vec<&[isize]>;
    let reordered_orders: Vec<Vec<usize>>;

    if let Some(dest_idx) = dest_index {
        if dest_idx < strides_list.len() && dest_idx != 0 {
            // Move destination to front
            let mut strides_vec: Vec<&[isize]> = strides_list.to_vec();
            let mut orders_vec = index_orders;

            let dest_strides = strides_vec.remove(dest_idx);
            let dest_order = orders_vec.remove(dest_idx);

            strides_vec.insert(0, dest_strides);
            orders_vec.insert(0, dest_order);

            reordered_strides = strides_vec;
            reordered_orders = orders_vec;
        } else {
            reordered_strides = strides_list.to_vec();
            reordered_orders = index_orders;
        }
    } else {
        reordered_strides = strides_list.to_vec();
        reordered_orders = index_orders;
    }

    // Compute importance using the Julia algorithm
    let importance = compute_importance(dims, &reordered_strides, &reordered_orders);

    // Sort by importance (descending)
    sort_by_importance(&importance)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Zero stride indicates broadcasting
        // Julia: zero strides get index_order = 1 (highest priority for iteration)
        let dims = [4usize, 5, 3];
        let strides = [0isize, 1, 5]; // First dim is broadcast (stride 0)
        let strides_list: Vec<&[isize]> = vec![&strides];

        let order = compute_order(&dims, &strides_list, Some(0));

        // With index_order: [1, 1, 2] for strides [0, 1, 5]
        // importance for dim 0: shift = g * (3 - 1) = high
        // importance for dim 1: shift = g * (3 - 1) = high (same as dim 0)
        // importance for dim 2: shift = g * (3 - 2) = lower
        // So dim 2 (largest stride) should be last
        assert_eq!(order[2], 2);
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

        // Output is weighted 2x, so its stride order dominates
        // Output index_order: [4, 3, 2, 1] (60 > 20 > 5 > 1)
        // Input index_order: [1, 2, 3, 4] (1 < 2 < 6 < 24)
        // With output 2x weight, dimension 3 (stride 1 in output) should be first
        assert_eq!(order[0], 3);
    }
}
