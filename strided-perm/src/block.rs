//! Block size computation ported from Strided.jl

use crate::fuse::compute_costs;
use crate::{BLOCK_MEMORY_SIZE, CACHE_LINE_SIZE};
use strided_view::auxiliary::index_order;

/// Compute block sizes for tiled iteration, on axes already in iteration
/// order (each extent at least 1).
///
/// Byte strides and halving scores are exact `u128` values from
/// `|s| · elem_size`. Memory regions are exact whenever the extents' product
/// fits in `usize`, and saturate outside it. A zero `elem_size`, and operands whose strides are all
/// nonzero and strictly increasing in magnitude, return full dimensions before
/// the stride orders are computed; the full algorithm gives the same result.
pub fn compute_block_sizes(
    dims: &[usize],
    strides_list: &[&[isize]],
    elem_size: usize,
) -> Vec<usize> {
    if dims.is_empty() || elem_size == 0 {
        return dims.to_vec();
    }

    let elem_size = elem_size as u128;
    let byte_strides: Vec<Vec<u128>> = strides_list
        .iter()
        .map(|strides| {
            strides
                .iter()
                .map(|&s| s.unsigned_abs() as u128 * elem_size)
                .collect()
        })
        .collect();
    let byte_stride_refs: Vec<&[u128]> = byte_strides.iter().map(|s| s.as_slice()).collect();

    if total_memory_region(dims, &byte_stride_refs) <= BLOCK_MEMORY_SIZE as u128
        || strides_list.iter().all(|s| strictly_increasing(s))
    {
        return dims.to_vec();
    }

    // With `elem_size > 0` the byte strides are an exact positive multiple of
    // the element strides, so both have the same stride order.
    let stride_orders: Vec<Vec<usize>> = strides_list
        .iter()
        .map(|s| index_order(s).into_vec())
        .collect();
    let stride_order_refs: Vec<&[usize]> = stride_orders.iter().map(|s| s.as_slice()).collect();
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

fn compute_blocks(
    dims: &[usize],
    costs: &[isize],
    byte_strides: &[&[u128]],
    stride_orders: &[&[usize]],
    block_size: usize,
) -> Vec<usize> {
    let n = dims.len();
    if n == 0 {
        return vec![];
    }

    let block_size_bytes = block_size as u128;
    if total_memory_region(dims, byte_strides) <= block_size_bytes {
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
        let tail_byte_strides: Vec<&[u128]> = byte_strides.iter().map(|s| &s[1..]).collect();
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
        .filter_map(|s| s.iter().min().copied())
        .min()
        .unwrap_or(0);

    if min_stride > block_size_bytes {
        return vec![1; n];
    }

    let mut blocks = dims.to_vec();

    // Phase 1: Halve until within 2x of target
    while total_memory_region(&blocks, byte_strides) >= 2 * block_size_bytes {
        let i = last_argmax_weighted(&blocks, costs);
        if i.is_none() || blocks[i.unwrap()] <= 1 {
            break;
        }
        let i = i.unwrap();
        blocks[i] = blocks[i].div_ceil(2);
    }

    // Phase 2: Decrement until within target
    while total_memory_region(&blocks, byte_strides) > block_size_bytes {
        let i = last_argmax_weighted(&blocks, costs);
        if i.is_none() || blocks[i.unwrap()] <= 1 {
            break;
        }
        let i = i.unwrap();
        blocks[i] -= 1;
    }

    blocks
}

/// Memory region of byte stride magnitudes `byte_strides`; the sums and
/// products saturate.
fn total_memory_region(dims: &[usize], byte_strides: &[&[u128]]) -> u128 {
    let cache_line = CACHE_LINE_SIZE as u128;
    let mut memory_region = 0u128;

    for strides in byte_strides {
        let mut num_contiguous_bytes = 0u128;
        let mut num_cache_line_blocks = 1u128;

        for (&d, &s) in dims.iter().zip(strides.iter()) {
            if s < cache_line {
                // The product is below 2^70, so only the sum can saturate.
                num_contiguous_bytes =
                    num_contiguous_bytes.saturating_add(d.saturating_sub(1) as u128 * s);
            } else {
                num_cache_line_blocks = num_cache_line_blocks.saturating_mul(d as u128);
            }
        }

        let contiguous_lines = num_contiguous_bytes / cache_line + 1;
        memory_region = memory_region.saturating_add(
            cache_line
                .saturating_mul(contiguous_lines)
                .saturating_mul(num_cache_line_blocks),
        );
    }

    memory_region
}

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
mod tests;
