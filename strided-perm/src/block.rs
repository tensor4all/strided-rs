//! Block size computation ported from Strided.jl

use crate::fuse::compute_costs;
use crate::{BLOCK_MEMORY_SIZE, CACHE_LINE_SIZE};
use strided_view::auxiliary::index_order;

/// Compute block sizes for tiled iteration.
pub fn compute_block_sizes(
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

    if total_memory_region(dims, byte_strides) <= block_size {
        return dims.to_vec();
    }

    let min_order = stride_orders
        .iter()
        .filter_map(|orders| orders.iter().min().copied())
        .min()
        .unwrap_or(1);

    if stride_orders
        .iter()
        .all(|orders| !orders.is_empty() && orders[0] == min_order)
    {
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

    let min_stride = byte_strides
        .iter()
        .filter_map(|s| s.iter().map(|x| x.unsigned_abs()).min())
        .min()
        .unwrap_or(0);

    if min_stride > block_size {
        return vec![1; n];
    }

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

fn total_memory_region(dims: &[usize], byte_strides: &[&[isize]]) -> usize {
    let cache_line = CACHE_LINE_SIZE;
    let mut memory_region = 0usize;

    for strides in byte_strides {
        let mut num_contiguous_cache_lines = 0isize;
        let mut num_cache_line_blocks = 1usize;

        for (&d, &s) in dims.iter().zip(strides.iter()) {
            let s_abs = s.unsigned_abs();
            if s_abs < cache_line {
                num_contiguous_cache_lines += (d.saturating_sub(1) as isize) * (s_abs as isize);
            } else {
                num_cache_line_blocks *= d;
            }
        }

        let contiguous_lines = (num_contiguous_cache_lines as usize / cache_line) + 1;
        memory_region += cache_line * contiguous_lines * num_cache_line_blocks;
    }

    memory_region
}

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
        let dims = [100usize];
        let strides = [8isize];
        let byte_strides: Vec<&[isize]> = vec![&strides];
        let region = total_memory_region(&dims, &byte_strides);
        assert_eq!(region, 832);
    }

    #[test]
    fn test_compute_blocks_small() {
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
        assert_eq!(blocks, vec![10, 10]);
    }

    #[test]
    fn test_compute_blocks_large() {
        let dims = [1000usize, 1000];
        let costs = [2isize, 2];
        let strides = [8isize, 8000];
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
        assert!(blocks[0] <= dims[0]);
        assert!(blocks[1] <= dims[1]);
        assert!(blocks[0] >= 1);
        assert!(blocks[1] >= 1);
    }
}
