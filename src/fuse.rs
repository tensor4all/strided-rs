//! Dimension fusion logic ported from Strided.jl/src/mapreduce.jl
//!
//! This module implements the core dimension fusion algorithm that merges
//! contiguous dimensions to reduce iteration complexity.

// Some helper functions are ported for completeness but not yet used.
#![allow(dead_code)]

/// Fuse contiguous dimensions across multiple arrays.
///
/// This function fuses subsequent dimensions that are contiguous in memory
/// for all arrays. If `strides[k][i] == dims[i-1] * strides[k][i-1]` for all k,
/// dimensions i-1 and i can be merged.
///
/// # Arguments
/// * `dims` - The shared dimensions of all arrays
/// * `all_strides` - Vector of stride tuples, one per array
///
/// # Returns
/// The fused dimensions (stride values remain unchanged, caller must recompute)
///
/// # Julia equivalent
/// ```julia
/// function _mapreduce_fuse!(f, op, initop, dims, arrays)
///     allstrides = map(strides, arrays)
///     @inbounds for i in length(dims):-1:2
///         merge = true
///         for s in allstrides
///             if s[i] != dims[i - 1] * s[i - 1]
///                 merge = false
///                 break
///             end
///         end
///         if merge
///             dims = setindex(dims, dims[i - 1] * dims[i], i - 1)
///             dims = setindex(dims, 1, i)
///         end
///     end
///     return dims
/// end
/// ```
pub fn fuse_dims(dims: &[usize], all_strides: &[&[isize]]) -> Vec<usize> {
    let n = dims.len();
    if n <= 1 || all_strides.is_empty() {
        return dims.to_vec();
    }

    let mut result = dims.to_vec();

    // Work from the end towards the beginning (Julia: for i in length(dims):-1:2)
    for i in (1..n).rev() {
        let mut can_merge = true;

        // Check all arrays for contiguity
        for strides in all_strides {
            // s[i] should equal dims[i-1] * s[i-1] for fusion
            let expected = result[i - 1] as isize * strides[i - 1];
            if strides[i] != expected {
                can_merge = false;
                break;
            }
        }

        if can_merge {
            // Fuse dimensions: merge dimension i into i-1
            result[i - 1] *= result[i];
            result[i] = 1;
        }
    }

    result
}

/// Compute the "importance" of each dimension for loop ordering.
///
/// This encodes stride order information into importance scores that determine
/// the optimal iteration order. The output array's strides are weighted 2x.
///
/// # Julia equivalent
/// ```julia
/// g = 8 * sizeof(Int) - leading_zeros(M + 1)  # ceil(log2(M+2))
/// importance = 2 .* (1 .<< (g .* (N .- indexorder(strides[1]))))
/// for k in 2:M
///     importance = importance .+ (1 .<< (g .* (N .- indexorder(strides[k]))))
/// end
/// importance = importance .* (dims .> 1)
/// ```
///
/// # Arguments
/// * `dims` - The dimensions
/// * `all_strides` - Vector of stride tuples
/// * `index_orders` - Pre-computed index orders for each stride tuple
///
/// # Returns
/// Importance scores for each dimension
pub fn compute_importance(
    dims: &[usize],
    all_strides: &[&[isize]],
    index_orders: &[Vec<usize>],
) -> Vec<u64> {
    let n = dims.len();
    let m = all_strides.len();

    if n == 0 || m == 0 {
        return vec![];
    }

    // g = ceil(log2(M + 2)) = number of bits needed to encode array count
    let g = (64 - (m as u64 + 1).leading_zeros()) as u64;

    let mut importance = vec![0u64; n];

    // First array (output) is weighted 2x
    for i in 0..n {
        let shift = g * (n - index_orders[0][i]) as u64;
        importance[i] = 2 * (1u64 << shift);
    }

    // Add contributions from remaining arrays
    for k in 1..m {
        for i in 0..n {
            let shift = g * (n - index_orders[k][i]) as u64;
            importance[i] += 1u64 << shift;
        }
    }

    // Zero importance for size-1 dimensions (put them at the back)
    for i in 0..n {
        if dims[i] <= 1 {
            importance[i] = 0;
        }
    }

    importance
}

/// Get the permutation that sorts by importance (descending).
///
/// Returns indices that would sort the importance array in descending order.
pub fn sort_by_importance(importance: &[u64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..importance.len()).collect();
    indices.sort_by(|&a, &b| importance[b].cmp(&importance[a]));
    indices
}

/// Apply a permutation to a slice.
pub fn permute_by<T: Clone>(data: &[T], perm: &[usize]) -> Vec<T> {
    perm.iter().map(|&i| data[i].clone()).collect()
}

/// Compute the minimum stride cost for each dimension.
///
/// Julia: `costs = map(a -> ifelse(iszero(a), 1, a << 1), map(min, strides...))`
pub fn compute_costs(all_strides: &[&[isize]]) -> Vec<isize> {
    if all_strides.is_empty() {
        return vec![];
    }

    let n = all_strides[0].len();
    let mut costs = vec![isize::MAX; n];

    for strides in all_strides {
        for i in 0..n {
            costs[i] = costs[i].min(strides[i].abs());
        }
    }

    // Transform: zero -> 1, nonzero -> 2*abs
    for cost in &mut costs {
        if *cost == 0 {
            *cost = 1;
        } else {
            *cost *= 2;
        }
    }

    costs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auxiliary::index_order;

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
    fn test_permute_by() {
        let data = vec![10, 20, 30, 40];
        let perm = vec![2, 0, 3, 1];
        let result = permute_by(&data, &perm);
        assert_eq!(result, vec![30, 10, 40, 20]);
    }
}
