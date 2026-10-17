//! Dimension fusion logic ported from Strided.jl/src/mapreduce.jl
//!
//! This module implements the core dimension fusion algorithm that merges
//! contiguous dimensions to reduce iteration complexity.

use crate::inline::{InlineVec, OperandVec};

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
pub fn fuse_dims(dims: &[usize], all_strides: &[&[isize]]) -> InlineVec<usize> {
    let n = dims.len();
    if n <= 1 || all_strides.is_empty() {
        return dims.iter().copied().collect();
    }

    let mut result: InlineVec<usize> = dims.iter().copied().collect();

    // Work from the end towards the beginning (Julia: for i in length(dims):-1:2)
    for i in (1..n).rev() {
        let mut can_merge = true;

        // Check all arrays for contiguity. An operand broadcast across both
        // axes has stride 0 for both axes and does not prevent fusing the
        // iteration space.
        for strides in all_strides {
            if strides[i - 1] == 0 && strides[i] == 0 {
                continue;
            }

            // s[i] should equal dims[i-1] * s[i-1] for fusion. A product
            // that does not fit in isize cannot equal a stored stride.
            let expected = isize::try_from(result[i - 1])
                .ok()
                .and_then(|dim| dim.checked_mul(strides[i - 1]));
            if expected != Some(strides[i]) {
                can_merge = false;
                break;
            }
        }

        if can_merge {
            // Fuse dimensions: merge dimension i into i-1. An extent product
            // that overflows (possible for zero-sized or broadcast views,
            // which validate without forming it) leaves the axes unfused.
            if let Some(merged) = result[i - 1].checked_mul(result[i]) {
                result[i - 1] = merged;
                result[i] = 1;
            }
        }
    }

    result
}

/// Remove size-1 dimensions from fused dims and all corresponding strides.
///
/// After `fuse_dims()`, many dimensions may be 1 (either originally size-1
/// or merged into a neighbor). These contribute nothing to iteration but
/// increase loop depth. This function strips them out.
///
/// If ALL dimensions are 1 (scalar-like), a single dimension of size 1
/// is preserved so the kernel has something to iterate over.
///
/// Both the returned dimensions and the per-operand stride lists are inline
/// collections: rank `<= INLINE_RANK` with at most [`crate::inline::OPERAND_CAP`]
/// operands does not allocate (#306 P1).
pub fn compress_dims<S: AsRef<[isize]>>(
    dims: &[usize],
    all_strides: &[S],
) -> (InlineVec<usize>, OperandVec<InlineVec<isize>>) {
    let kept: InlineVec<usize> = (0..dims.len()).filter(|&i| dims[i] != 1).collect();

    if kept.is_empty() {
        // All dims are 1 (or empty). Preserve a single trivial dimension.
        if dims.is_empty() {
            let strides = all_strides
                .iter()
                .map(|s| s.as_ref().iter().copied().collect())
                .collect();
            return (InlineVec::new(), strides);
        }
        let new_strides = all_strides
            .iter()
            .map(|s| {
                let mut v = InlineVec::new();
                v.push(s.as_ref()[0]);
                v
            })
            .collect();
        let mut dims_out = InlineVec::new();
        dims_out.push(1);
        return (dims_out, new_strides);
    }

    let new_dims: InlineVec<usize> = kept.iter().map(|&i| dims[i]).collect();
    let new_strides: OperandVec<InlineVec<isize>> = all_strides
        .iter()
        .map(|s| kept.iter().map(|&i| s.as_ref()[i]).collect())
        .collect();

    (new_dims, new_strides)
}

/// Compute the "importance" of each dimension for loop ordering.
///
/// This encodes stride order information into importance scores that determine
/// the optimal iteration order. Zero-stride broadcast axes do not contribute to
/// importance, and the output array gets strong weight so contiguous stores
/// remain in the inner loop.
///
/// # Arguments
/// * `dims` - The dimensions
/// * `all_strides` - Vector of stride tuples
/// * `index_orders` - Pre-computed index orders for each stride tuple
///
/// # Returns
/// Importance scores for each dimension
pub fn compute_importance<I: AsRef<[usize]>>(
    dims: &[usize],
    all_strides: &[&[isize]],
    index_orders: &[I],
) -> InlineVec<u64> {
    let n = dims.len();
    let m = all_strides.len();

    if n == 0 || m == 0 {
        return InlineVec::new();
    }

    // g = ceil(log2(M + 2)) = number of bits needed to encode array count
    let g = (64 - (m as u64 + 1).leading_zeros()) as u64;

    let mut importance: InlineVec<u64> = (0..n).map(|_| 0u64).collect();

    let output_weight = 1u64 << (g + 1);

    // First array (output) gets a strong weight. For elementwise kernels the
    // store stream determines whether the inner loop can be emitted as a
    // contiguous vector loop, so a broadcasted input's local stride should not
    // break a contiguous output group.
    for i in 0..n {
        if all_strides[0][i] != 0 {
            let shift = g * (n - index_orders[0].as_ref()[i]) as u64;
            importance[i] = output_weight * (1u64 << shift);
        }
    }

    // Add contributions from remaining arrays
    #[allow(clippy::needless_range_loop)]
    for k in 1..m {
        for i in 0..n {
            if all_strides[k][i] != 0 {
                let shift = g * (n - index_orders[k].as_ref()[i]) as u64;
                importance[i] += 1u64 << shift;
            }
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
pub fn sort_by_importance(importance: &[u64]) -> InlineVec<usize> {
    let mut indices: InlineVec<usize> = (0..importance.len()).collect();
    indices.sort_by(|&a, &b| importance[b].cmp(&importance[a]));
    indices
}

/// Compute the minimum stride cost for each dimension.
///
/// Julia: `costs = map(a -> ifelse(iszero(a), 1, a << 1), map(min, strides...))`
pub(crate) fn compute_costs<S: AsRef<[isize]>>(all_strides: &[S]) -> InlineVec<isize> {
    if all_strides.is_empty() {
        return InlineVec::new();
    }

    let n = all_strides[0].as_ref().len();
    let mut costs: InlineVec<isize> = (0..n).map(|_| isize::MAX).collect();

    for strides in all_strides {
        let strides = strides.as_ref();
        for i in 0..n {
            costs[i] = costs[i].min(strides[i].checked_abs().unwrap_or(isize::MAX));
        }
    }

    // Transform: zero -> 1, nonzero -> 2*abs. Costs only rank axes, so a
    // stride beyond isize::MAX / 2 (reachable with zero-sized elements)
    // saturates instead of overflowing.
    for cost in &mut costs {
        if *cost == 0 {
            *cost = 1;
        } else {
            *cost = cost.saturating_mul(2);
        }
    }

    costs
}

#[cfg(test)]
#[path = "fuse/tests/tests.rs"]
mod tests;
