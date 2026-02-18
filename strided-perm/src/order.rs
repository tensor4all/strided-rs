//! Loop ordering algorithm ported from Strided.jl

use crate::fuse::{compute_importance, sort_by_importance};
use strided_view::auxiliary::index_order;

/// Compute the optimal iteration order for dimensions.
///
/// This implementation follows Julia's `_mapreduce_order!` algorithm:
/// 1. Compute `index_order` for each array's strides
/// 2. Compute importance scores using bit-packing with output weighted 2x
/// 3. Sort dimensions by importance (descending)
pub fn compute_order(
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
        let dims = [4usize, 5];
        let strides = [1isize, 4];
        let strides_list: Vec<&[isize]> = vec![&strides];
        let order = compute_order(&dims, &strides_list, Some(0));
        assert_eq!(order[0], 0);
        assert_eq!(order[1], 1);
    }

    #[test]
    fn test_compute_order_row_major() {
        let dims = [4usize, 5];
        let strides = [5isize, 1];
        let strides_list: Vec<&[isize]> = vec![&strides];
        let order = compute_order(&dims, &strides_list, Some(0));
        assert_eq!(order[0], 1);
        assert_eq!(order[1], 0);
    }

    #[test]
    fn test_compute_order_mixed() {
        let dims = [4usize, 5];
        let out_strides = [1isize, 4];
        let in_strides = [5isize, 1];
        let strides_list: Vec<&[isize]> = vec![&out_strides, &in_strides];
        let order = compute_order(&dims, &strides_list, Some(0));
        assert_eq!(order[0], 0);
        assert_eq!(order[1], 1);
    }

    #[test]
    fn test_compute_order_empty() {
        let dims: [usize; 0] = [];
        let strides: [isize; 0] = [];
        let strides_list: Vec<&[isize]> = vec![&strides];
        let order = compute_order(&dims, &strides_list, Some(0));
        assert!(order.is_empty());
    }
}
