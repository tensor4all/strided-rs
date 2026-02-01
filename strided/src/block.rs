//! Block size computation ported from Strided.jl
//!
//! This module computes optimal block sizes for cache-efficient iteration.
//! It uses Julia's `totalmemoryregion` for accurate memory region estimation
//! and `_computeblocks` for iterative block size reduction.

use crate::fuse::compute_costs;
use crate::{BLOCK_MEMORY_SIZE, CACHE_LINE_SIZE};
use stridedview::auxiliary::index_order;

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
    let stride_orders: Vec<Vec<usize>> = byte_strides.iter().map(|bs| index_order(bs)).collect();

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

    if stride_orders
        .iter()
        .all(|orders| !orders.is_empty() && orders[0] == min_order)
    {
        // First dimension has smallest stride in all arrays
        // Keep first dimension, recurse on rest
        let tail_dims: Vec<usize> = dims[1..].to_vec();
        let tail_costs: Vec<isize> = costs[1..].to_vec();
        let tail_byte_strides: Vec<&[isize]> = byte_strides.iter().map(|s| &s[1..]).collect();
        let tail_stride_orders: Vec<&[usize]> = stride_orders.iter().map(|s| &s[1..]).collect();

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
        blocks[i] = blocks[i].div_ceil(2);
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
        let contiguous_lines = (num_contiguous_cache_lines as usize / cache_line) + 1;

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

        let blocks = compute_blocks(
            &dims,
            &costs,
            &byte_strides,
            &stride_orders,
            BLOCK_MEMORY_SIZE,
        );

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

        let blocks = compute_blocks(
            &dims,
            &costs,
            &byte_strides,
            &stride_orders,
            BLOCK_MEMORY_SIZE,
        );

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

    // ========== Julia-comparison tests ==========

    #[test]
    fn test_total_memory_region_julia_match_2d() {
        // Julia: dims=(10,10), bytestrides=((8,80),)
        // numcontigeouscachelines: 9*8 + 9*80 = 72 + 720 = 792 (but 80 >= 64!)
        // Actually: stride 8 < 64, so += (10-1)*8 = 72
        //          stride 80 >= 64, so numcachelineblocks *= 10
        // contiguous_lines = 72/64 + 1 = 2
        // memory = 64 * 2 * 10 = 1280
        let dims = [10usize, 10];
        let strides = [8isize, 80];
        let byte_strides: Vec<&[isize]> = vec![&strides];

        let region = total_memory_region(&dims, &byte_strides);
        assert_eq!(region, 1280);
    }

    #[test]
    fn test_total_memory_region_julia_match_multiple_arrays() {
        // Two arrays with different strides
        let dims = [10usize, 10];
        let strides1 = [8isize, 80]; // Column-major-ish
        let strides2 = [80isize, 8]; // Row-major-ish
        let byte_strides: Vec<&[isize]> = vec![&strides1, &strides2];

        let region = total_memory_region(&dims, &byte_strides);

        // Array 1: stride 8 < 64 -> contiguous += 72, stride 80 >= 64 -> blocks *= 10
        //          contiguous_lines = 2, region1 = 64 * 2 * 10 = 1280
        // Array 2: stride 80 >= 64 -> blocks *= 10, stride 8 < 64 -> contiguous += 72
        //          contiguous_lines = 2, region2 = 64 * 2 * 10 = 1280
        // Total = 2560
        assert_eq!(region, 2560);
    }

    #[test]
    fn test_total_memory_region_all_contiguous() {
        // All strides < cache line
        let dims = [10usize, 5];
        let strides = [8isize, 40]; // Both < 64
        let byte_strides: Vec<&[isize]> = vec![&strides];

        let region = total_memory_region(&dims, &byte_strides);

        // contiguous = (10-1)*8 + (5-1)*40 = 72 + 160 = 232
        // contiguous_lines = 232/64 + 1 = 4
        // blocks = 1 (no large strides)
        // region = 64 * 4 * 1 = 256
        assert_eq!(region, 256);
    }

    #[test]
    fn test_total_memory_region_all_large_strides() {
        // All strides >= cache line
        let dims = [5usize, 4];
        let strides = [64isize, 320]; // Both >= 64
        let byte_strides: Vec<&[isize]> = vec![&strides];

        let region = total_memory_region(&dims, &byte_strides);

        // contiguous = 0
        // contiguous_lines = 0/64 + 1 = 1
        // blocks = 5 * 4 = 20
        // region = 64 * 1 * 20 = 1280
        assert_eq!(region, 1280);
    }

    #[test]
    fn test_compute_blocks_first_dim_smallest_stride() {
        // When first dimension has smallest stride for all arrays,
        // Julia keeps d1 and recurses on tail
        let dims = [100usize, 10, 10];
        let costs = [2isize, 2, 2];
        // All arrays have first dim with smallest stride
        let strides1 = [8isize, 800, 8000];
        let orders1 = [1usize, 2, 3]; // First dim has order 1 (smallest)
        let byte_strides: Vec<&[isize]> = vec![&strides1];
        let stride_orders: Vec<&[usize]> = vec![&orders1];

        let blocks = compute_blocks(
            &dims,
            &costs,
            &byte_strides,
            &stride_orders,
            BLOCK_MEMORY_SIZE,
        );

        // First dimension should be kept as-is (100)
        assert_eq!(blocks[0], 100);
    }

    #[test]
    fn test_compute_blocks_min_stride_larger_than_blocksize() {
        // When minimum stride > blocksize AND total memory > blocksize, return all 1s
        // Need:
        // 1. Total memory > blocksize
        // 2. First dim does NOT have smallest stride (to skip the special case)
        // 3. All strides > blocksize
        let dims = [100usize, 100];
        let costs = [2isize, 2];
        // Stride order: dim 1 has smaller stride than dim 0, but both > blocksize
        let strides = [40000000isize, 40000]; // dim 1 is smaller but still > 32768
        let orders = [2usize, 1]; // dim 1 has smaller stride order
        let byte_strides: Vec<&[isize]> = vec![&strides];
        let stride_orders: Vec<&[usize]> = vec![&orders];

        // Verify total memory exceeds blocksize first
        let initial_mem = total_memory_region(&dims, &byte_strides);
        assert!(
            initial_mem > BLOCK_MEMORY_SIZE,
            "Initial memory {} should exceed {}",
            initial_mem,
            BLOCK_MEMORY_SIZE
        );

        // Verify minimum stride > blocksize
        let min_stride = strides.iter().map(|s| s.unsigned_abs()).min().unwrap();
        assert!(
            min_stride > BLOCK_MEMORY_SIZE,
            "Min stride {} should exceed {}",
            min_stride,
            BLOCK_MEMORY_SIZE
        );

        let blocks = compute_blocks(
            &dims,
            &costs,
            &byte_strides,
            &stride_orders,
            BLOCK_MEMORY_SIZE,
        );

        // With minimum stride > blocksize, should return all 1s
        assert_eq!(blocks, vec![1, 1]);
    }

    #[test]
    fn test_compute_blocks_4d_array_first_dim_smallest() {
        // 4D array where first dim has smallest stride (column-major case)
        // Julia's algorithm keeps first dimension when it has smallest stride for cache locality
        let dims = [10usize, 10, 10, 10];
        let costs = [2isize, 2, 2, 2];
        // Column-major strides: first dim has smallest stride
        let strides = [8isize, 80, 800, 8000];
        let orders = [1usize, 2, 3, 4];
        let byte_strides: Vec<&[isize]> = vec![&strides];
        let stride_orders: Vec<&[usize]> = vec![&orders];

        let blocks = compute_blocks(
            &dims,
            &costs,
            &byte_strides,
            &stride_orders,
            BLOCK_MEMORY_SIZE,
        );

        // When first dim has smallest stride, Julia keeps full dims via recursion
        // This is by design for cache efficiency
        assert_eq!(blocks.len(), 4);
        for (i, &b) in blocks.iter().enumerate() {
            assert!(b >= 1 && b <= dims[i], "Block {} out of range", i);
        }
    }

    #[test]
    fn test_compute_blocks_4d_needs_reduction() {
        // 4D array where blocking reduction is actually needed
        // Mix strides so first dim doesn't have globally smallest stride
        let dims = [100usize, 100, 100, 100];
        let costs = [2isize, 2, 2, 2];
        // Mixed strides: dim 1 has smallest stride, not dim 0
        let strides = [80isize, 8, 8000, 800];
        let orders = [2usize, 1, 4, 3]; // Reflects actual stride magnitudes
        let byte_strides: Vec<&[isize]> = vec![&strides];
        let stride_orders: Vec<&[usize]> = vec![&orders];

        // Verify total memory exceeds blocksize
        let initial_mem = total_memory_region(&dims, &byte_strides);
        assert!(initial_mem > BLOCK_MEMORY_SIZE);

        let blocks = compute_blocks(
            &dims,
            &costs,
            &byte_strides,
            &stride_orders,
            BLOCK_MEMORY_SIZE,
        );

        // Blocks should be reduced from original dims
        assert_eq!(blocks.len(), 4);
        let total_elements: usize = blocks.iter().product();
        let original_elements: usize = dims.iter().product();
        assert!(
            total_elements < original_elements,
            "Blocks should be smaller than original"
        );
    }

    #[test]
    fn test_compute_blocks_4d_permuted() {
        // 4D array with permuted strides (Issue #5 scenario)
        let dims = [10usize, 10, 10, 10];
        let costs = [2isize, 4, 8, 16]; // Different costs
                                        // Permuted strides: last dim has smallest stride
        let strides = [8000isize, 800, 80, 8];
        let orders = [4usize, 3, 2, 1];
        let byte_strides: Vec<&[isize]> = vec![&strides];
        let stride_orders: Vec<&[usize]> = vec![&orders];

        let blocks = compute_blocks(
            &dims,
            &costs,
            &byte_strides,
            &stride_orders,
            BLOCK_MEMORY_SIZE,
        );

        assert_eq!(blocks.len(), 4);
        // Verify blocks are valid
        for (i, &b) in blocks.iter().enumerate() {
            assert!(b >= 1 && b <= dims[i]);
        }
    }

    #[test]
    fn test_compute_blocks_mixed_strides_two_arrays() {
        // Two arrays with conflicting stride patterns
        let dims = [100usize, 100];
        let costs = [2isize, 2];
        let strides1 = [8isize, 800]; // Column-major
        let strides2 = [800isize, 8]; // Row-major
        let orders1 = [1usize, 2];
        let orders2 = [2usize, 1];
        let byte_strides: Vec<&[isize]> = vec![&strides1, &strides2];
        let stride_orders: Vec<&[usize]> = vec![&orders1, &orders2];

        let blocks = compute_blocks(
            &dims,
            &costs,
            &byte_strides,
            &stride_orders,
            BLOCK_MEMORY_SIZE,
        );

        // Should not use the first-dim-smallest special case
        // since arrays have different stride orders
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0] >= 1 && blocks[0] <= 100);
        assert!(blocks[1] >= 1 && blocks[1] <= 100);
    }

    #[test]
    fn test_last_argmax_weighted_all_ones() {
        // When all blocks are 1, should return None
        let blocks = [1usize, 1, 1];
        let costs = [1isize, 2, 3];

        let idx = last_argmax_weighted(&blocks, &costs);
        assert_eq!(idx, None);
    }

    #[test]
    fn test_last_argmax_weighted_mixed() {
        // Mix of 1s and larger values
        let blocks = [1usize, 5, 3];
        let costs = [100isize, 1, 1];

        let idx = last_argmax_weighted(&blocks, &costs);

        // (1-1)*100=0, (5-1)*1=4, (3-1)*1=2
        // Max is 4 at index 1
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn test_last_argmax_weighted_cost_matters() {
        // Higher cost can outweigh larger block
        let blocks = [3usize, 10];
        let costs = [10isize, 1];

        let idx = last_argmax_weighted(&blocks, &costs);

        // (3-1)*10=20, (10-1)*1=9
        // Max is 20 at index 0
        assert_eq!(idx, Some(0));
    }

    #[test]
    fn test_compute_blocks_negative_strides() {
        // Negative strides should be handled via absolute value
        let dims = [10usize, 10];
        let costs = [2isize, 2];
        let strides = [-8isize, -80]; // Negative strides
        let orders = [1usize, 2];
        let byte_strides: Vec<&[isize]> = vec![&strides];
        let stride_orders: Vec<&[usize]> = vec![&orders];

        let blocks = compute_blocks(
            &dims,
            &costs,
            &byte_strides,
            &stride_orders,
            BLOCK_MEMORY_SIZE,
        );

        // Should behave same as positive strides
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0] >= 1 && blocks[0] <= 10);
        assert!(blocks[1] >= 1 && blocks[1] <= 10);
    }

    #[test]
    fn test_compute_block_sizes_4d_column_major() {
        // Full pipeline test for 4D arrays (Issue #5)
        // Column-major layout: first dim has smallest stride
        let dims = [32usize, 32, 32, 32];
        let order = [0usize, 1, 2, 3]; // Natural order
        let strides = [1isize, 32, 1024, 32768]; // Column-major
        let strides_list: Vec<&[isize]> = vec![&strides];

        let blocks = compute_block_sizes(&dims, &order, &strides_list, 8);

        assert_eq!(blocks.len(), 4);
        // Verify blocks are valid
        for (i, &b) in blocks.iter().enumerate() {
            assert!(b >= 1 && b <= dims[i], "Block {} = {} out of range", i, b);
        }
    }

    #[test]
    fn test_compute_block_sizes_4d_permuted_strides() {
        // Full pipeline test for 4D arrays with permuted strides
        // This exercises the halving/decrementing reduction path
        let dims = [32usize, 32, 32, 32];
        let order = [3usize, 2, 1, 0]; // Reversed order
                                       // Permuted strides: smallest stride is in last position of original
        let strides = [32768isize, 1024, 32, 1];
        let strides_list: Vec<&[isize]> = vec![&strides];

        let blocks = compute_block_sizes(&dims, &order, &strides_list, 8);

        assert_eq!(blocks.len(), 4);
        for (i, &b) in blocks.iter().enumerate() {
            assert!(b >= 1, "Block {} must be >= 1", i);
        }
    }
}
