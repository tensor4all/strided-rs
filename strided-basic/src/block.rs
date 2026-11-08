//! Block size computation ported from Strided.jl
//!
//! This module computes optimal block sizes for cache-efficient iteration.
//! It uses Julia's `totalmemoryregion` for accurate memory region estimation
//! and `_computeblocks` for iterative block size reduction.
//!
//! # Arithmetic contract
//!
//! Byte strides, memory regions and halving scores are evaluated in `u128`
//! from exact byte magnitudes `|s| · elem_size`. For extents `d ≥ 1` whose
//! product fits in `usize` (callers check it with `kernel::total_len`), every
//! intermediate is exact, so the result is the block vector of the
//! mathematical-integer model, for any element size and any stride. Outside
//! that precondition the region arithmetic saturates: both budget comparisons
//! keep their unbounded outcome and nothing panics.

use crate::fuse::compute_costs;
use crate::inline::{InlineVec, OperandVec};
use crate::{BLOCK_MEMORY_SIZE, CACHE_LINE_SIZE};
use strided_view::auxiliary::index_order;

/// Compute block sizes for tiled iteration.
///
/// This implementation follows Julia's `_computeblocks` algorithm on axes
/// that are already in iteration order:
/// 1. If the memory region fits in the block budget, use full dimensions
/// 2. Otherwise compute stride orders and costs, and reduce blocks by
///    cost-weighted halving and decrementing
///
/// Two exits return full dimensions before the orders are computed, both
/// with the same result as the full algorithm: a zero `elem_size` (every
/// byte stride is 0, so every stride order is 1 and every axis is kept), and
/// operands whose strides are all nonzero and strictly increasing in
/// magnitude (each operand's stride order is `1..=r`, so every axis is kept).
///
/// # Arguments
/// * `dims` - The dimensions, in iteration order, each at least 1
/// * `strides_list` - Slice of element stride arrays, one per array, in the
///   same order as `dims`
/// * `elem_size` - Size of each element in bytes
///
/// # Returns
/// Block sizes in iteration order
pub(crate) fn compute_block_sizes(
    dims: &[usize],
    strides_list: &[&[isize]],
    elem_size: usize,
) -> InlineVec<usize> {
    if dims.is_empty() || elem_size == 0 {
        return dims.iter().copied().collect();
    }

    let elem_size = elem_size as u128;
    let byte_strides: OperandVec<InlineVec<u128>> = strides_list
        .iter()
        .map(|strides| {
            strides
                .iter()
                .map(|&s| s.unsigned_abs() as u128 * elem_size)
                .collect()
        })
        .collect();
    let byte_stride_refs: OperandVec<&[u128]> = byte_strides.iter().map(|s| s.as_slice()).collect();

    if total_memory_region_at(dims, &byte_stride_refs, 0) <= BLOCK_MEMORY_SIZE as u128
        || strides_list.iter().all(|s| strictly_increasing(s))
    {
        return dims.iter().copied().collect();
    }

    // With `elem_size > 0` the byte strides are an exact positive multiple of
    // the element strides, so both have the same stride order.
    let stride_orders: OperandVec<InlineVec<usize>> =
        strides_list.iter().map(|s| index_order(s)).collect();
    let stride_order_refs: OperandVec<&[usize]> =
        stride_orders.iter().map(|s| s.as_slice()).collect();
    let costs = compute_costs(strides_list);

    compute_blocks(
        dims,
        &costs,
        &byte_stride_refs,
        &stride_order_refs,
        BLOCK_MEMORY_SIZE,
    )
}

/// Whether every stride is nonzero and strictly larger in magnitude than the
/// previous one.
fn strictly_increasing(strides: &[isize]) -> bool {
    strides.first().is_some_and(|&s| s != 0)
        && strides
            .windows(2)
            .all(|w| w[0].unsigned_abs() < w[1].unsigned_abs())
}

/// Compute block sizes using the Julia algorithm.
///
/// # Julia equivalent
/// ```julia
/// function _computeblocks(dims, costs, bytestrides, strideorders, blocksize)
///     if totalmemoryregion(dims, bytestrides) <= blocksize
///         return dims
///     end
///     # ... reduction logic
/// end
/// ```
fn compute_blocks(
    dims: &[usize],
    costs: &[isize],
    byte_strides: &[&[u128]],
    stride_orders: &[&[usize]],
    block_size: usize,
) -> InlineVec<usize> {
    compute_blocks_at(dims, 0, costs, byte_strides, stride_orders, block_size)
}

/// Recursive core of [`compute_blocks`] with `offset` tracking how many
/// leading axes have already been fixed. Indexing each operand's stride slice
/// from `offset` keeps the recursion allocation-free at the inline bound
/// (#306 P1).
fn compute_blocks_at(
    dims: &[usize],
    offset: usize,
    costs: &[isize],
    byte_strides: &[&[u128]],
    stride_orders: &[&[usize]],
    block_size: usize,
) -> InlineVec<usize> {
    let n = dims.len();
    if n == 0 {
        return InlineVec::new();
    }

    let block_size_bytes = block_size as u128;

    // If everything fits in cache, use full dims
    if total_memory_region_at(dims, byte_strides, offset) <= block_size_bytes {
        return dims.iter().copied().collect();
    }

    // Check if first dimension is smallest stride for all arrays
    let min_order = stride_orders
        .iter()
        .filter_map(|orders| orders[offset..].iter().min().copied())
        .min()
        .unwrap_or(1);

    if stride_orders
        .iter()
        .all(|orders| !orders[offset..].is_empty() && orders[offset] == min_order)
    {
        // First dimension has smallest stride in all arrays
        // Keep first dimension, recurse on rest
        let tail_blocks = compute_blocks_at(
            &dims[1..],
            offset + 1,
            &costs[1..],
            byte_strides,
            stride_orders,
            block_size,
        );

        let mut result: InlineVec<usize> = InlineVec::new();
        result.push(dims[0]);
        result.extend(tail_blocks);
        return result;
    }

    // Check if minimum stride is larger than block size
    let min_stride = byte_strides
        .iter()
        .filter_map(|s| s[offset..].iter().min().copied())
        .min()
        .unwrap_or(0);

    if min_stride > block_size_bytes {
        return (0..n).map(|_| 1).collect();
    }

    // Iteratively reduce blocks
    let mut blocks: InlineVec<usize> = dims.iter().copied().collect();

    // Phase 1: Halve until within 2x of target
    while total_memory_region_at(&blocks, byte_strides, offset) >= 2 * block_size_bytes {
        let i = last_argmax_weighted(&blocks, costs);
        if i.is_none() || blocks[i.unwrap()] <= 1 {
            break;
        }
        let i = i.unwrap();
        blocks[i] = blocks[i].div_ceil(2);
    }

    // Phase 2: Decrement until within target
    while total_memory_region_at(&blocks, byte_strides, offset) > block_size_bytes {
        let i = last_argmax_weighted(&blocks, costs);
        if i.is_none() || blocks[i.unwrap()] <= 1 {
            break;
        }
        let i = i.unwrap();
        blocks[i] -= 1;
    }

    blocks
}

/// Compute total memory region for cache considerations.
///
/// This estimates the memory footprint considering cache line effects.
/// Strides smaller than cache line contribute to contiguous region,
/// larger strides multiply the number of cache line blocks.
///
/// # Julia equivalent
/// ```julia
/// function totalmemoryregion(dims, bytestrides)
///     memoryregion = 0
///     for i in 1:length(bytestrides)
///         strides = bytestrides[i]
///         numcontigeouscachelines = 0
///         numcachelineblocks = 1
///         for (d, s) in zip(dims, strides)
///             if s < _cachelinelength
///                 numcontigeouscachelines += (d - 1) * s
///             else
///                 numcachelineblocks *= d
///             end
///         end
///         numcontigeouscachelines = div(numcontigeouscachelines, _cachelinelength) + 1
///         memoryregion += _cachelinelength * numcontigeouscachelines * numcachelineblocks
///     end
///     return memoryregion
/// end
/// ```
///
/// `byte_strides` holds byte stride magnitudes. The sums and products
/// saturate, which is exact whenever the extents' product fits in `usize`.
#[cfg(test)]
fn total_memory_region(dims: &[usize], byte_strides: &[&[u128]]) -> u128 {
    total_memory_region_at(dims, byte_strides, 0)
}

/// As [`total_memory_region`], but with the leading `offset` axes fixed. The
/// recursion in `compute_blocks_at` uses it to avoid materializing tail slices.
fn total_memory_region_at(dims: &[usize], byte_strides: &[&[u128]], offset: usize) -> u128 {
    let cache_line = CACHE_LINE_SIZE as u128;
    let mut memory_region = 0u128;

    for strides in byte_strides {
        let strides = &strides[offset..];
        let mut num_contiguous_bytes = 0u128;
        let mut num_cache_line_blocks = 1u128;

        for (&d, &s) in dims.iter().zip(strides.iter()) {
            if s < cache_line {
                // Small stride: contributes to contiguous region. The product
                // is below 2^70, so only the sum can saturate.
                num_contiguous_bytes =
                    num_contiguous_bytes.saturating_add(d.saturating_sub(1) as u128 * s);
            } else {
                // Large stride: multiplies cache line blocks
                num_cache_line_blocks = num_cache_line_blocks.saturating_mul(d as u128);
            }
        }

        // Convert to cache lines
        let contiguous_lines = num_contiguous_bytes / cache_line + 1;

        memory_region = memory_region.saturating_add(
            cache_line
                .saturating_mul(contiguous_lines)
                .saturating_mul(num_cache_line_blocks),
        );
    }

    memory_region
}

/// Find the last index with maximum (value - 1) * cost.
///
/// Julia: `_lastargmax((blocks .- 1) .* costs)`, over blocks above 1 only.
/// Costs are nonnegative (`compute_costs` gives at least 1); the score is
/// exact in `u128`.
fn last_argmax_weighted(blocks: &[usize], costs: &[isize]) -> Option<usize> {
    if blocks.is_empty() {
        return None;
    }

    let mut max_score = 0u128;
    let mut max_idx = None;

    for (i, (&b, &c)) in blocks.iter().zip(costs.iter()).enumerate() {
        if b <= 1 {
            continue;
        }
        let score = (b - 1) as u128 * c as u128;
        if score >= max_score {
            max_score = score;
            max_idx = Some(i);
        }
    }

    max_idx
}

#[cfg(test)]
#[path = "block/tests/tests.rs"]
mod tests;
