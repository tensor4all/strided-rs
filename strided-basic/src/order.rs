//! Loop ordering algorithm derived from Strided.jl
//!
//! This module computes the optimal dimension iteration order using
//! an index_order + importance bit-packing algorithm.

use crate::fuse::{compute_importance, sort_by_importance};
use strided_view::auxiliary::index_order;

/// Compute the optimal iteration order for dimensions.
///
/// This implementation follows Julia's `_mapreduce_order!` structure:
/// 1. Compute `index_order` for each array's strides
/// 2. Compute importance scores using bit-packing, with strong output locality
///    preference and zero-stride broadcast axes ignored for ordering
/// 3. Sort dimensions by importance (descending)
///
/// # Arguments
/// * `dims` - The dimensions of the arrays
/// * `strides_list` - Slice of stride arrays, one per array
/// * `dest_index` - Index of the destination array (strongly weighted, typically 0)
///
/// # Returns
/// Permutation of dimension indices in optimal iteration order
///
/// The baseline comes from Strided.jl. This version differs by ignoring
/// zero-stride broadcast axes for ordering and giving the destination a
/// stronger weight so contiguous stores stay in the inner loop.
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

    // Reorder so destination array is first (gets strong store-locality weight)
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
#[path = "order/tests/tests.rs"]
mod tests;
