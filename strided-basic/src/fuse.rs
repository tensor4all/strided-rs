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
/// the optimal iteration order. Zero-stride broadcast axes and size-1 axes do
/// not contribute to importance, and the output array gets strong weight so
/// contiguous stores remain in the inner loop.
///
/// Each array's level `n - order` is packed `g = ceil(log2(m + 2))` bits apart,
/// with the output's level carrying `g + 1` bits on top. That packing is used
/// whenever it fits in the 64-bit score, so the plan order is the same as it
/// always was. At high rank `g * n` does not fit (tensor4all/strided-rs#450),
/// and the per-level width narrows to the widest one that does: every level
/// stays distinct and only the weight between levels changes. From rank ~62 on
/// one bit per level still overflows, and the levels above the cap share the
/// top weight.
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

    // Weights for one per-level width and level cap, or `None` if any shift,
    // product or sum would overflow. `(g, u64::MAX)` is the original packing.
    let weights = |bits: u64, level_cap: u64| -> Option<InlineVec<u64>> {
        // 2^(bits * level), or `None` when that shift or width is too large.
        let term = |k: usize, i: usize| -> Option<u64> {
            let level = (n - index_orders[k].as_ref()[i]) as u64;
            let shift = level.min(level_cap).checked_mul(bits)?;
            1u64.checked_shl(u32::try_from(shift).ok()?)
        };
        let output_weight = 1u64.checked_shl((bits + 1) as u32)?;

        let mut weights: InlineVec<u64> = (0..n).map(|_| 0u64).collect();
        for i in 0..n {
            // Size-1 axes carry no ordering information; skipping them keeps
            // their strides out of the weights.
            if dims[i] <= 1 {
                continue;
            }
            // The first array (output) gets a strong weight. For elementwise
            // kernels the store stream determines whether the inner loop can be
            // emitted as a contiguous vector loop, so a broadcasted input's
            // local stride should not break a contiguous output group.
            if all_strides[0][i] != 0 {
                weights[i] = term(0, i)?.checked_mul(output_weight)?;
            }
            for (k, strides) in all_strides.iter().enumerate().skip(1) {
                if strides[i] != 0 {
                    weights[i] = weights[i].checked_add(term(k, i)?)?;
                }
            }
        }
        Some(weights)
    };

    // Keep the original packing whenever it fits, so no plan order changes;
    // narrow the per-level width only on an actual overflow (#450).
    let mut bits = g;
    let mut packed = weights(bits, u64::MAX);
    while packed.is_none() && bits > 1 {
        bits -= 1;
        packed = weights(bits, u64::MAX);
    }

    packed
        .or_else(|| {
            // One bit per level still overflows from rank ~62 on: share the top
            // weight, with the widest level cap that fits. The bound is `u64`,
            // not a fixed 61: the cap that fits depends on the shape (a cap of
            // 61 is forced only when the output stream is live, because its
            // weight is `4 * 2^cap`; with an all-zero output stride `63` fits),
            // so let the checked arithmetic choose per shape.
            (1..=63u64)
                .rev()
                .find_map(|level_cap| weights(1, level_cap))
        })
        // Unreachable for any operand count a slice can hold; ties every axis.
        .unwrap_or_else(|| (0..n).map(|_| 0u64).collect())
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
    let Some(first) = all_strides.first() else {
        return InlineVec::new();
    };
    let n = first.as_ref().len();
    // Zero -> 1, nonzero -> 2*abs. Costs only rank axes, so a stride beyond
    // isize::MAX / 2 (reachable with zero-sized elements) saturates instead
    // of overflowing, and `isize::MIN`'s magnitude counts as `isize::MAX`.
    let cost = |axis: usize| {
        let min = all_strides
            .iter()
            .map(|strides| strides.as_ref()[axis].checked_abs().unwrap_or(isize::MAX))
            .min()
            .unwrap_or(isize::MAX);
        if min == 0 {
            1
        } else {
            min.saturating_mul(2)
        }
    };
    // Filled in place at the inline rank: collecting element by element
    // through the inline vector measured as most of this function's cost.
    if n <= strided_view::INLINE_RANK {
        let mut costs = [0isize; strided_view::INLINE_RANK];
        for (axis, slot) in costs[..n].iter_mut().enumerate() {
            *slot = cost(axis);
        }
        return InlineVec::from_array_len(costs, n);
    }
    (0..n).map(cost).collect()
}

#[cfg(test)]
#[path = "fuse/tests/tests.rs"]
mod tests;
