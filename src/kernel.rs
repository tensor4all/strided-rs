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

    // Inner strides are for the innermost dimension (dim 1)
    let inner_strides: Vec<isize> = strides.iter().map(|s| s[1]).collect();

    let mut j0 = 0usize;
    while j0 < d0 {
        let blen0 = b0.min(d0 - j0);

        let mut j1 = 0usize;
        while j1 < d1 {
            let blen1 = b1.min(d1 - j1);

            // Iterate over outer dimension within block
            for _ in 0..blen0 {
                // Call with inner block info
                f(offsets, blen1, &inner_strides)?;

                // Advance by inner block and step outer
                for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                    *offset += (blen1 as isize) * s[1];
                    *offset -= (blen1 as isize) * s[1]; // Will be reset below
                    *offset += s[0];
                }
            }
            // Reset outer stride
            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                *offset -= (blen0 as isize) * s[0];
            }

            // Move to next block in dimension 1
            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                *offset += (blen1 as isize) * s[1];
            }
            j1 += blen1;
        }

        // Reset dimension 1, advance dimension 0
        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
            *offset -= (d1 as isize) * s[1];
            *offset += (blen0 as isize) * s[0];
        }
        j0 += blen0;
    }

    // Reset all
    for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
        *offset -= (d0 as isize) * s[0];
    }

    Ok(())
}

/// 3D kernel with inner block callback
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

    let inner_strides: Vec<isize> = strides.iter().map(|s| s[2]).collect();

    let mut j0 = 0usize;
    while j0 < d0 {
        let blen0 = b0.min(d0 - j0);

        let mut j1 = 0usize;
        while j1 < d1 {
            let blen1 = b1.min(d1 - j1);

            let mut j2 = 0usize;
            while j2 < d2 {
                let blen2 = b2.min(d2 - j2);

                for _ in 0..blen0 {
                    for _ in 0..blen1 {
                        f(offsets, blen2, &inner_strides)?;
                        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                            *offset += (blen2 as isize) * s[2];
                            *offset -= (blen2 as isize) * s[2];
                            *offset += s[1];
                        }
                    }
                    for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                        *offset -= (blen1 as isize) * s[1];
                        *offset += s[0];
                    }
                }
                for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                    *offset -= (blen0 as isize) * s[0];
                    *offset += (blen2 as isize) * s[2];
                }
                j2 += blen2;
            }

            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                *offset -= (d2 as isize) * s[2];
                *offset += (blen1 as isize) * s[1];
            }
            j1 += blen1;
        }

        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
            *offset -= (d1 as isize) * s[1];
            *offset += (blen0 as isize) * s[0];
        }
        j0 += blen0;
    }

    for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
        *offset -= (d0 as isize) * s[0];
    }

    Ok(())
}

/// 4D kernel with inner block callback
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

    let inner_strides: Vec<isize> = strides.iter().map(|s| s[3]).collect();

    let mut j0 = 0usize;
    while j0 < d0 {
        let blen0 = b0.min(d0 - j0);

        let mut j1 = 0usize;
        while j1 < d1 {
            let blen1 = b1.min(d1 - j1);

            let mut j2 = 0usize;
            while j2 < d2 {
                let blen2 = b2.min(d2 - j2);

                let mut j3 = 0usize;
                while j3 < d3 {
                    let blen3 = b3.min(d3 - j3);

                    for _ in 0..blen0 {
                        for _ in 0..blen1 {
                            for _ in 0..blen2 {
                                f(offsets, blen3, &inner_strides)?;
                                for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                                    *offset += (blen3 as isize) * s[3];
                                    *offset -= (blen3 as isize) * s[3];
                                    *offset += s[2];
                                }
                            }
                            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                                *offset -= (blen2 as isize) * s[2];
                                *offset += s[1];
                            }
                        }
                        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                            *offset -= (blen1 as isize) * s[1];
                            *offset += s[0];
                        }
                    }
                    for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                        *offset -= (blen0 as isize) * s[0];
                        *offset += (blen3 as isize) * s[3];
                    }
                    j3 += blen3;
                }

                for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                    *offset -= (d3 as isize) * s[3];
                    *offset += (blen2 as isize) * s[2];
                }
                j2 += blen2;
            }

            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                *offset -= (d2 as isize) * s[2];
                *offset += (blen1 as isize) * s[1];
            }
            j1 += blen1;
        }

        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
            *offset -= (d1 as isize) * s[1];
            *offset += (blen0 as isize) * s[0];
        }
        j0 += blen0;
    }

    for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
        *offset -= (d0 as isize) * s[0];
    }

    Ok(())
}

/// N-dimensional kernel with inner block callback (recursive fallback)
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
    let inner_strides: Vec<isize> = strides.iter().map(|s| s[dims.len() - 1]).collect();
    kernel_nd_inner_level(0, dims, blocks, strides, &inner_strides, offsets, f)
}

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

    if level == dims.len() - 1 {
        // Innermost level - call callback with block info
        let mut j = 0usize;
        while j < d {
            let blen = b.min(d - j);
            f(offsets, blen, inner_strides)?;
            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                *offset += (blen as isize) * s[level];
            }
            j += blen;
        }
        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
            *offset -= (d as isize) * s[level];
        }
    } else {
        // Outer level
        let mut j = 0usize;
        while j < d {
            let blen = b.min(d - j);

            // Inner block loop
            for _ in 0..blen {
                kernel_nd_inner_level(level + 1, dims, blocks, strides, inner_strides, offsets, f)?;
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
