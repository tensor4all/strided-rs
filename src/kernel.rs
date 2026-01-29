//! Kernel iteration engine ported from Julia's Strided.jl/src/mapreduce.jl
//!
//! This module implements the core iteration engine that follows Julia's
//! `_mapreduce_kernel!` pattern for cache-optimized strided array operations.

use crate::fuse::fuse_dims;
use crate::{block, order, Result};

pub(crate) struct KernelPlan {
    pub(crate) order: Vec<usize>, // outer -> inner
    pub(crate) block: Vec<usize>,
}

/// Build an execution plan for strided iteration.
///
/// This follows Julia's `_mapreduce_fuse!` -> `_mapreduce_order!` -> `_mapreduce_block!` pipeline:
/// 1. Fuse contiguous dimensions
/// 2. Compute optimal iteration order
/// 3. Compute block sizes for cache efficiency
pub(crate) fn build_plan(
    dims: &[usize],
    strides_list: &[&[isize]],
    dest_index: Option<usize>,
    elem_size: usize,
) -> KernelPlan {
    let order = order::compute_order(dims, strides_list, dest_index);
    let block = block::compute_block_sizes(dims, &order, strides_list, elem_size);
    KernelPlan { order, block }
}

/// Build an execution plan with dimension fusion.
///
/// This is the Julia-faithful version that fuses contiguous dimensions
/// before computing the iteration order.
pub(crate) fn build_plan_fused(
    dims: &[usize],
    strides_list: &[&[isize]],
    dest_index: Option<usize>,
    elem_size: usize,
) -> (Vec<usize>, KernelPlan) {
    // Fuse contiguous dimensions
    let fused_dims = fuse_dims(dims, strides_list);

    // Compute order and blocks on fused dimensions
    let order = order::compute_order(&fused_dims, strides_list, dest_index);
    let block = block::compute_block_sizes(&fused_dims, &order, strides_list, elem_size);

    (fused_dims, KernelPlan { order, block })
}

// ============================================================================
// Block-based iteration with inner stride callback
// ============================================================================

/// Iterate over blocks, calling f with (offsets, block_len, inner_strides).
///
/// The callback receives the current byte offsets for each array, the number
/// of elements in the innermost block, and the innermost strides for each array.
/// This allows the caller to implement vectorized inner loops.
#[inline]
pub(crate) fn for_each_inner_block<F>(
    dims: &[usize],
    plan: &KernelPlan,
    strides_list: &[&[isize]],
    mut f: F,
) -> Result<()>
where
    F: FnMut(&[isize], usize, &[isize]) -> Result<()>,
{
    let rank = dims.len();
    if rank == 0 {
        let offsets = vec![0isize; strides_list.len()];
        return f(&offsets, 1, &[]);
    }

    // Reorder dimensions and strides according to plan
    let ordered_dims: Vec<usize> = plan.order.iter().map(|&d| dims[d]).collect();
    let ordered_blocks: Vec<usize> = plan.block.clone();

    // Build stride vectors for each array, ordered by plan
    let num_arrays = strides_list.len();
    let mut ordered_strides: Vec<Vec<isize>> = Vec::with_capacity(num_arrays);
    for strides in strides_list {
        let s: Vec<isize> = plan.order.iter().map(|&d| strides[d]).collect();
        ordered_strides.push(s);
    }

    // Initial offsets (all zero)
    let mut offsets = vec![0isize; num_arrays];

    // Call the specialized kernel based on rank
    match rank {
        1 => kernel_1d_inner(
            &ordered_dims,
            &ordered_blocks,
            &ordered_strides,
            &mut offsets,
            &mut f,
        ),
        2 => kernel_2d_inner(
            &ordered_dims,
            &ordered_blocks,
            &ordered_strides,
            &mut offsets,
            &mut f,
        ),
        3 => kernel_3d_inner(
            &ordered_dims,
            &ordered_blocks,
            &ordered_strides,
            &mut offsets,
            &mut f,
        ),
        4 => kernel_4d_inner(
            &ordered_dims,
            &ordered_blocks,
            &ordered_strides,
            &mut offsets,
            &mut f,
        ),
        _ => kernel_nd_inner(
            &ordered_dims,
            &ordered_blocks,
            &ordered_strides,
            &mut offsets,
            &mut f,
        ),
    }
}

// ============================================================================
// Specialized kernels (inner-block callback)
// ============================================================================

/// 1D kernel with inner block callback
#[inline]
fn kernel_1d_inner<F>(
    dims: &[usize],
    blocks: &[usize],
    strides: &[Vec<isize>],
    offsets: &mut [isize],
    f: &mut F,
) -> Result<()>
where
    F: FnMut(&[isize], usize, &[isize]) -> Result<()>,
{
    let d0 = dims[0];
    let b0 = blocks[0].max(1).min(d0);

    // Extract inner strides
    let inner_strides: Vec<isize> = strides.iter().map(|s| s[0]).collect();

    let mut j0 = 0usize;
    while j0 < d0 {
        let block_len = b0.min(d0 - j0);

        // Call with block info
        f(offsets, block_len, &inner_strides)?;

        // Advance offsets by block_len
        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
            *offset += (block_len as isize) * s[0];
        }
        j0 += block_len;
    }

    // Reset offsets
    for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
        *offset -= (d0 as isize) * s[0];
    }

    Ok(())
}

/// 2D kernel with inner block callback
///
/// Loop nesting (matches Julia): outer=d1 (lowest importance), inner callback=d0 (highest importance)
#[inline]
fn kernel_2d_inner<F>(
    dims: &[usize],
    blocks: &[usize],
    strides: &[Vec<isize>],
    offsets: &mut [isize],
    f: &mut F,
) -> Result<()>
where
    F: FnMut(&[isize], usize, &[isize]) -> Result<()>,
{
    let d0 = dims[0];
    let d1 = dims[1];
    let b0 = blocks[0].max(1).min(d0);
    let b1 = blocks[1].max(1).min(d1);

    // Inner strides are for dim 0 (highest importance = smallest stride)
    let inner_strides: Vec<isize> = strides.iter().map(|s| s[0]).collect();

    // Outer block loop: d1 (lowest importance)
    let mut j1 = 0usize;
    while j1 < d1 {
        let blen1 = b1.min(d1 - j1);

        // Inner block loop: d0 (highest importance)
        let mut j0 = 0usize;
        while j0 < d0 {
            let blen0 = b0.min(d0 - j0);

            // Element loops: outer=d1, inner callback=d0
            for _ in 0..blen1 {
                f(offsets, blen0, &inner_strides)?;
                for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                    *offset += s[1];
                }
            }
            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                *offset -= (blen1 as isize) * s[1];
                *offset += (blen0 as isize) * s[0];
            }
            j0 += blen0;
        }

        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
            *offset -= (d0 as isize) * s[0];
            *offset += (blen1 as isize) * s[1];
        }
        j1 += blen1;
    }

    for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
        *offset -= (d1 as isize) * s[1];
    }

    Ok(())
}

/// 3D kernel with inner block callback
///
/// Loop nesting (matches Julia): outer=d2, mid=d1, inner callback=d0 (highest importance)
#[inline]
fn kernel_3d_inner<F>(
    dims: &[usize],
    blocks: &[usize],
    strides: &[Vec<isize>],
    offsets: &mut [isize],
    f: &mut F,
) -> Result<()>
where
    F: FnMut(&[isize], usize, &[isize]) -> Result<()>,
{
    let d0 = dims[0];
    let d1 = dims[1];
    let d2 = dims[2];
    let b0 = blocks[0].max(1).min(d0);
    let b1 = blocks[1].max(1).min(d1);
    let b2 = blocks[2].max(1).min(d2);

    // Inner strides are for dim 0 (highest importance)
    let inner_strides: Vec<isize> = strides.iter().map(|s| s[0]).collect();

    // Outer block loop: d2 (lowest importance)
    let mut j2 = 0usize;
    while j2 < d2 {
        let blen2 = b2.min(d2 - j2);

        let mut j1 = 0usize;
        while j1 < d1 {
            let blen1 = b1.min(d1 - j1);

            // Innermost block loop: d0 (highest importance)
            let mut j0 = 0usize;
            while j0 < d0 {
                let blen0 = b0.min(d0 - j0);

                // Element loops: outer=d2, mid=d1, inner callback=d0
                for _ in 0..blen2 {
                    for _ in 0..blen1 {
                        f(offsets, blen0, &inner_strides)?;
                        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                            *offset += s[1];
                        }
                    }
                    for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                        *offset -= (blen1 as isize) * s[1];
                        *offset += s[2];
                    }
                }
                for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                    *offset -= (blen2 as isize) * s[2];
                    *offset += (blen0 as isize) * s[0];
                }
                j0 += blen0;
            }

            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                *offset -= (d0 as isize) * s[0];
                *offset += (blen1 as isize) * s[1];
            }
            j1 += blen1;
        }

        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
            *offset -= (d1 as isize) * s[1];
            *offset += (blen2 as isize) * s[2];
        }
        j2 += blen2;
    }

    for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
        *offset -= (d2 as isize) * s[2];
    }

    Ok(())
}

/// 4D kernel with inner block callback
///
/// Loop nesting (matches Julia): outer=d3, d2, d1, inner callback=d0 (highest importance)
#[inline]
fn kernel_4d_inner<F>(
    dims: &[usize],
    blocks: &[usize],
    strides: &[Vec<isize>],
    offsets: &mut [isize],
    f: &mut F,
) -> Result<()>
where
    F: FnMut(&[isize], usize, &[isize]) -> Result<()>,
{
    let d0 = dims[0];
    let d1 = dims[1];
    let d2 = dims[2];
    let d3 = dims[3];
    let b0 = blocks[0].max(1).min(d0);
    let b1 = blocks[1].max(1).min(d1);
    let b2 = blocks[2].max(1).min(d2);
    let b3 = blocks[3].max(1).min(d3);

    // Inner strides are for dim 0 (highest importance)
    let inner_strides: Vec<isize> = strides.iter().map(|s| s[0]).collect();

    // Outer block loop: d3 (lowest importance)
    let mut j3 = 0usize;
    while j3 < d3 {
        let blen3 = b3.min(d3 - j3);

        let mut j2 = 0usize;
        while j2 < d2 {
            let blen2 = b2.min(d2 - j2);

            let mut j1 = 0usize;
            while j1 < d1 {
                let blen1 = b1.min(d1 - j1);

                // Innermost block loop: d0 (highest importance)
                let mut j0 = 0usize;
                while j0 < d0 {
                    let blen0 = b0.min(d0 - j0);

                    // Element loops: outer=d3, d2, d1, inner callback=d0
                    for _ in 0..blen3 {
                        for _ in 0..blen2 {
                            for _ in 0..blen1 {
                                f(offsets, blen0, &inner_strides)?;
                                for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                                    *offset += s[1];
                                }
                            }
                            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                                *offset -= (blen1 as isize) * s[1];
                                *offset += s[2];
                            }
                        }
                        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                            *offset -= (blen2 as isize) * s[2];
                            *offset += s[3];
                        }
                    }
                    for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                        *offset -= (blen3 as isize) * s[3];
                        *offset += (blen0 as isize) * s[0];
                    }
                    j0 += blen0;
                }

                for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                    *offset -= (d0 as isize) * s[0];
                    *offset += (blen1 as isize) * s[1];
                }
                j1 += blen1;
            }

            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                *offset -= (d1 as isize) * s[1];
                *offset += (blen2 as isize) * s[2];
            }
            j2 += blen2;
        }

        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
            *offset -= (d2 as isize) * s[2];
            *offset += (blen3 as isize) * s[3];
        }
        j3 += blen3;
    }

    for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
        *offset -= (d3 as isize) * s[3];
    }

    Ok(())
}

/// N-dimensional kernel with inner block callback (recursive fallback)
///
/// Recursion starts from the last level (outermost = lowest importance)
/// and descends to level 0 (innermost = highest importance = callback).
#[inline]
fn kernel_nd_inner<F>(
    dims: &[usize],
    blocks: &[usize],
    strides: &[Vec<isize>],
    offsets: &mut [isize],
    f: &mut F,
) -> Result<()>
where
    F: FnMut(&[isize], usize, &[isize]) -> Result<()>,
{
    // Inner strides are for dim 0 (highest importance)
    let inner_strides: Vec<isize> = strides.iter().map(|s| s[0]).collect();
    let last = dims.len() - 1;
    kernel_nd_inner_level(last, dims, blocks, strides, &inner_strides, offsets, f)
}

/// Recursive level handler.
/// `level` counts down from `rank-1` (outermost) to `0` (innermost callback).
#[inline]
fn kernel_nd_inner_level<F>(
    level: usize,
    dims: &[usize],
    blocks: &[usize],
    strides: &[Vec<isize>],
    inner_strides: &[isize],
    offsets: &mut [isize],
    f: &mut F,
) -> Result<()>
where
    F: FnMut(&[isize], usize, &[isize]) -> Result<()>,
{
    let d = dims[level];
    let b = blocks[level].max(1).min(d);

    if level == 0 {
        // Innermost level (highest importance) - call callback with block info
        let mut j = 0usize;
        while j < d {
            let blen = b.min(d - j);
            f(offsets, blen, inner_strides)?;
            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                *offset += (blen as isize) * s[0];
            }
            j += blen;
        }
        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
            *offset -= (d as isize) * s[0];
        }
    } else {
        // Outer level — block loop then element loop stepping through this dimension
        let mut j = 0usize;
        while j < d {
            let blen = b.min(d - j);

            // Element loop for this dimension, recurse into next-inner level
            for _ in 0..blen {
                kernel_nd_inner_level(level - 1, dims, blocks, strides, inner_strides, offsets, f)?;
                for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                    *offset += s[level];
                }
            }
            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                *offset -= (blen as isize) * s[level];
                *offset += (blen as isize) * s[level];
            }
            j += blen;
        }
        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
            *offset -= (d as isize) * s[level];
        }
    }

    Ok(())
}

// ============================================================================
// Utility functions
// ============================================================================

/// Reorder by plan, apply a second fusion pass, and recompute blocks.
///
/// The standard pipeline (fuse → order → block) only fuses dimensions in their
/// original order, which works for col-major but misses row-major contiguity.
/// After ordering puts smallest-stride dimensions first, a second `fuse_dims`
/// pass can merge dimensions that are now adjacent and contiguous.
///
/// Returns `(double_fused_dims, ordered_strides, blocks)` all in iteration order.
pub(crate) fn double_fuse_for_parallel(
    fused_dims: &[usize],
    strides_list: &[&[isize]],
    plan: &KernelPlan,
    elem_size: usize,
) -> (Vec<usize>, Vec<Vec<isize>>, Vec<usize>) {
    // Step 1: Reorder by plan (smallest stride first)
    let (ordered_dims, ordered_strides) = reorder_by_plan(fused_dims, strides_list, plan);

    // Step 2: Second fuse on ordered dims/strides
    let ordered_strides_refs: Vec<&[isize]> =
        ordered_strides.iter().map(|s| s.as_slice()).collect();
    let double_fused_dims = fuse_dims(&ordered_dims, &ordered_strides_refs);

    // Step 3: Recompute blocks with identity ordering on double-fused dims
    let identity: Vec<usize> = (0..double_fused_dims.len()).collect();
    let blocks = block::compute_block_sizes(
        &double_fused_dims,
        &identity,
        &ordered_strides_refs,
        elem_size,
    );

    (double_fused_dims, ordered_strides, blocks)
}

/// Reorder dimensions and strides according to a plan.
///
/// Returns (ordered_dims, ordered_strides_list) where each is reordered
/// by `plan.order`.
pub(crate) fn reorder_by_plan(
    dims: &[usize],
    strides_list: &[&[isize]],
    plan: &KernelPlan,
) -> (Vec<usize>, Vec<Vec<isize>>) {
    let ordered_dims: Vec<usize> = plan.order.iter().map(|&d| dims[d]).collect();
    let ordered_strides: Vec<Vec<isize>> = strides_list
        .iter()
        .map(|strides| plan.order.iter().map(|&d| strides[d]).collect())
        .collect();
    (ordered_dims, ordered_strides)
}

pub(crate) fn ensure_same_shape(a: &[usize], b: &[usize]) -> Result<()> {
    if a.len() != b.len() {
        return Err(crate::StridedError::RankMismatch(a.len(), b.len()));
    }
    if a != b {
        return Err(crate::StridedError::ShapeMismatch(a.to_vec(), b.to_vec()));
    }
    Ok(())
}

pub(crate) fn is_contiguous(dims: &[usize], strides: &[isize]) -> bool {
    if dims.len() != strides.len() {
        return false;
    }
    if dims.is_empty() {
        return true;
    }
    let mut expected = 1isize;
    for (&dim, &stride) in dims.iter().rev().zip(strides.iter().rev()) {
        if dim <= 1 {
            continue;
        }
        if stride != expected {
            return false;
        }
        expected = expected.saturating_mul(dim as isize);
    }
    true
}

pub(crate) fn total_len(dims: &[usize]) -> usize {
    if dims.is_empty() {
        return 1;
    }
    dims.iter().product()
}

/// Whether the sequential contiguous fast path should be used.
///
/// When the `parallel` feature is enabled and the total element count exceeds
/// the threading threshold, we must *not* take the contiguous fast path so that
/// the parallel kernel path can be reached.  Julia has no separate contiguous
/// fast path — everything flows through fuse → order → block → threaded →
/// kernel, so skipping it here matches Julia's branching.
#[inline]
pub(crate) fn use_sequential_fast_path(total: usize) -> bool {
    #[cfg(feature = "parallel")]
    {
        total <= crate::threading::MINTHREADLENGTH
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = total;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_inner_block() {
        let dims = vec![2, 4];
        let strides1 = vec![4isize, 1];
        let strides2 = vec![4isize, 1];
        let strides_list: Vec<&[isize]> = vec![&strides1, &strides2];
        let plan = build_plan(&dims, &strides_list, Some(0), 8);

        let mut total_elements = 0usize;
        for_each_inner_block(&dims, &plan, &strides_list, |_offsets, len, _strides| {
            total_elements += len;
            Ok(())
        })
        .unwrap();

        assert_eq!(total_elements, 8);
    }
}
