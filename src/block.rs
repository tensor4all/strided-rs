//! Block size computation ported from Strided.jl
//!
//! This module computes optimal block sizes for cache-efficient iteration.
//! It uses Julia's `totalmemoryregion` for accurate memory region estimation
//! and `_computeblocks` for iterative block size reduction.

use crate::auxiliary::index_order;
use crate::fuse::compute_costs;
use crate::{BLOCK_MEMORY_SIZE, CACHE_LINE_SIZE};

/// Compute block sizes for tiled iteration.
///
/// This implementation follows Julia's `_computeblocks` algorithm:
/// 1. Compute byte strides and stride orders
/// 2. If total memory fits in cache, use full dimensions
/// 3. Otherwise, iteratively reduce blocks using cost-weighted halving
///
/// # Arguments
/// * `dims` - The dimensions (in iteration order)
/// * `order` - The iteration order permutation
/// * `strides_list` - Slice of stride arrays, one per array
/// * `elem_size` - Size of each element in bytes
///
/// # Returns
/// Block sizes in iteration order
pub(crate) fn compute_block_sizes(
    dims: &[usize],
    order: &[usize],
    strides_list: &[&[isize]],
    elem_size: usize,
) -> Vec<usize> {
    if order.is_empty() {
        return Vec::new();
    }

    // Reorder dims to iteration order
    let ordered_dims: Vec<usize> = order.iter().map(|&i| dims[i]).collect();

    // Compute byte strides in iteration order
    let byte_strides: Vec<Vec<isize>> = strides_list
        .iter()
        .map(|strides| {
            order
                .iter()
                .map(|&i| strides[i] * elem_size as isize)
                .collect()
        })
        .collect();

    // Compute stride orders (in iteration order)
    let stride_orders: Vec<Vec<usize>> = byte_strides
        .iter()
        .map(|bs| index_order(bs))
        .collect();

    // Reorder strides for cost computation
    let reordered_strides: Vec<Vec<isize>> = strides_list
        .iter()
        .map(|strides| order.iter().map(|&i| strides[i]).collect())
        .collect();

    let reordered_refs: Vec<&[isize]> = reordered_strides.iter().map(|s| s.as_slice()).collect();
    let costs = compute_costs(&reordered_refs);

    // Convert byte_strides to slices
    let byte_stride_refs: Vec<&[isize]> = byte_strides.iter().map(|s| s.as_slice()).collect();
    let stride_order_refs: Vec<&[usize]> = stride_orders.iter().map(|s| s.as_slice()).collect();

    compute_blocks(
        &ordered_dims,
        &costs,
        &byte_stride_refs,
        &stride_order_refs,
        BLOCK_MEMORY_SIZE,
    )
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
    byte_strides: &[&[isize]],
    stride_orders: &[&[usize]],
    block_size: usize,
) -> Vec<usize> {
    let n = dims.len();
    if n == 0 {
        return vec![];
    }

    // If everything fits in cache, use full dims
    if total_memory_region(dims, byte_strides) <= block_size {
        return dims.to_vec();
    }

    // Check if first dimension is smallest stride for all arrays
    let min_order = stride_orders
        .iter()
        .filter_map(|orders| orders.iter().min().copied())
        .min()
        .unwrap_or(1);

    if stride_orders.iter().all(|orders| !orders.is_empty() && orders[0] == min_order) {
        // First dimension has smallest stride in all arrays
        // Keep first dimension, recurse on rest
        let tail_dims: Vec<usize> = dims[1..].to_vec();
        let tail_costs: Vec<isize> = costs[1..].to_vec();
        let tail_byte_strides: Vec<&[isize]> = byte_strides
            .iter()
            .map(|s| &s[1..])
            .collect();
        let tail_stride_orders: Vec<&[usize]> = stride_orders
            .iter()
            .map(|s| &s[1..])
            .collect();

        let tail_blocks = compute_blocks(
            &tail_dims,
            &tail_costs,
            &tail_byte_strides,
            &tail_stride_orders,
            block_size,
        );

        let mut result = vec![dims[0]];
        result.extend(tail_blocks);
        return result;
    }

    // Check if minimum stride is larger than block size
    let min_stride = byte_strides
        .iter()
        .filter_map(|s| s.iter().map(|x| x.unsigned_abs()).min())
        .min()
        .unwrap_or(0);

    if min_stride > block_size {
        return vec![1; n];
    }

    // Iteratively reduce blocks
    let mut blocks = dims.to_vec();

    // Phase 1: Halve until within 2x of target
    while total_memory_region(&blocks, byte_strides) >= 2 * block_size {
        let i = last_argmax_weighted(&blocks, costs);
        if i.is_none() || blocks[i.unwrap()] <= 1 {
            break;
        }
        let i = i.unwrap();
        blocks[i] = (blocks[i] + 1) / 2;
    }

    // Phase 2: Decrement until within target
    while total_memory_region(&blocks, byte_strides) > block_size {
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
fn total_memory_region(dims: &[usize], byte_strides: &[&[isize]]) -> usize {
    let cache_line = CACHE_LINE_SIZE;
    let mut memory_region = 0usize;

    for strides in byte_strides {
        let mut num_contiguous_cache_lines = 0isize;
        let mut num_cache_line_blocks = 1usize;

        for (&d, &s) in dims.iter().zip(strides.iter()) {
            let s_abs = s.unsigned_abs();
            if s_abs < cache_line {
                // Small stride: contributes to contiguous region
                num_contiguous_cache_lines += (d.saturating_sub(1) as isize) * (s_abs as isize);
            } else {
                // Large stride: multiplies cache line blocks
                num_cache_line_blocks *= d;
            }
        }

        // Convert to cache lines
        let contiguous_lines =
            (num_contiguous_cache_lines as usize / cache_line) + 1;

        memory_region += cache_line * contiguous_lines * num_cache_line_blocks;
    }

    memory_region
}

/// Find the last index with maximum (value - 1) * cost.
///
/// Julia: `_lastargmax((blocks .- 1) .* costs)`
fn last_argmax_weighted(blocks: &[usize], costs: &[isize]) -> Option<usize> {
    if blocks.is_empty() {
        return None;
    }

    let mut max_score = 0isize;
    let mut max_idx = None;

    for (i, (&b, &c)) in blocks.iter().zip(costs.iter()).enumerate() {
        if b <= 1 {
            continue;
        }
        let score = (b as isize - 1) * c;
        if score >= max_score {
            max_score = score;
            max_idx = Some(i);
        }
    }

    max_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_total_memory_region_contiguous() {
        // Contiguous array: 100 elements * 8 bytes = 800 bytes
        // Should be about 800 bytes (plus cache line rounding)
        let dims = [100usize];
        let strides = [8isize]; // 8 bytes per element
        let byte_strides: Vec<&[isize]> = vec![&strides];

        let region = total_memory_region(&dims, &byte_strides);

        // 99 * 8 = 792 bytes contiguous
        // 792 / 64 + 1 = 13 cache lines
        // 64 * 13 = 832 bytes
        assert_eq!(region, 832);
    }

    #[test]
    fn test_total_memory_region_strided() {
        // Non-contiguous: stride larger than cache line
        let dims = [10usize];
        let strides = [128isize]; // Stride >= cache line
        let byte_strides: Vec<&[isize]> = vec![&strides];

        let region = total_memory_region(&dims, &byte_strides);

        // Each element touches separate cache line blocks
        // 64 * 1 * 10 = 640 bytes
        assert_eq!(region, 640);
    }

    #[test]
    fn test_compute_blocks_small() {
        // Small array that fits in cache
        let dims = [10usize, 10];
        let costs = [2isize, 2];
        let strides = [8isize, 80];
        let orders = [1usize, 2];
        let byte_strides: Vec<&[isize]> = vec![&strides];
        let stride_orders: Vec<&[usize]> = vec![&orders];

        let blocks = compute_blocks(&dims, &costs, &byte_strides, &stride_orders, BLOCK_MEMORY_SIZE);

        // Should use full dimensions since it fits
        assert_eq!(blocks, vec![10, 10]);
    }

    #[test]
    fn test_compute_blocks_large() {
        // Large array that needs blocking
        let dims = [1000usize, 1000];
        let costs = [2isize, 2];
        let strides = [8isize, 8000]; // 8 bytes per f64
        let orders = [1usize, 2];
        let byte_strides: Vec<&[isize]> = vec![&strides];
        let stride_orders: Vec<&[usize]> = vec![&orders];

        let blocks = compute_blocks(&dims, &costs, &byte_strides, &stride_orders, BLOCK_MEMORY_SIZE);

        // Should reduce block sizes
        assert!(blocks[0] <= dims[0]);
        assert!(blocks[1] <= dims[1]);
        // But should still be reasonable
        assert!(blocks[0] >= 1);
        assert!(blocks[1] >= 1);
    }

    #[test]
    fn test_last_argmax_weighted() {
        let blocks = [10usize, 20, 5];
        let costs = [1isize, 1, 2];

        let idx = last_argmax_weighted(&blocks, &costs);

        // (10-1)*1=9, (20-1)*1=19, (5-1)*2=8
        // Maximum is 19 at index 1
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn test_last_argmax_weighted_tie() {
        // When tied, returns last index
        let blocks = [10usize, 10];
        let costs = [1isize, 1];

        let idx = last_argmax_weighted(&blocks, &costs);

        // Both have score 9, return last (index 1)
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn test_compute_block_sizes_full_pipeline() {
        // Test the full compute_block_sizes function
        let dims = [100usize, 100];
        let order = [0usize, 1];
        let strides = [1isize, 100];
        let strides_list: Vec<&[isize]> = vec![&strides];

        let blocks = compute_block_sizes(&dims, &order, &strides_list, 8);

        // Should return valid block sizes
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0] >= 1 && blocks[0] <= 100);
        assert!(blocks[1] >= 1 && blocks[1] <= 100);
    }
}
