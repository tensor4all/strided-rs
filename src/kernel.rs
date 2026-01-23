//! Kernel iteration engine ported from Julia's Strided.jl/src/mapreduce.jl
//!
//! This module implements the core iteration engine that follows Julia's
//! `_mapreduce_kernel!` pattern for cache-optimized strided array operations.

use crate::fuse::fuse_dims;
use crate::{block, order, Result, StridedError};
use mdarray::{Layout, Shape, Slice};

pub(crate) struct StridedView<T> {
    pub(crate) ptr: *const T,
    pub(crate) dims: Vec<usize>,
    pub(crate) strides: Vec<isize>,
}

pub(crate) struct StridedViewMut<T> {
    pub(crate) ptr: *mut T,
    pub(crate) dims: Vec<usize>,
    pub(crate) strides: Vec<isize>,
}

impl<T> StridedView<T> {
    pub(crate) fn from_slice<S: Shape, L: Layout>(slice: &Slice<T, S, L>) -> Result<Self> {
        let (dims, strides) = dims_and_strides(slice);
        validate_layout(&dims, &strides)?;
        Ok(Self {
            ptr: slice.as_ptr(),
            dims,
            strides,
        })
    }

    pub(crate) fn from_parts(ptr: *const T, dims: &[usize], strides: &[isize]) -> Result<Self> {
        validate_layout(dims, strides)?;
        Ok(Self {
            ptr,
            dims: dims.to_vec(),
            strides: strides.to_vec(),
        })
    }
}

impl<T> StridedViewMut<T> {
    pub(crate) fn from_slice<S: Shape, L: Layout>(slice: &mut Slice<T, S, L>) -> Result<Self> {
        let (dims, strides) = dims_and_strides(&*slice);
        validate_layout(&dims, &strides)?;
        Ok(Self {
            ptr: slice.as_mut_ptr(),
            dims,
            strides,
        })
    }
}

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
#[allow(dead_code)]
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
// Julia-faithful kernel implementation
// ============================================================================

/// Execute a map operation using Julia's _mapreduce_kernel! pattern.
///
/// This is the core iteration engine that implements the Julia algorithm:
/// - Outer loops iterate in blocks (for cache efficiency)
/// - Inner loops iterate within blocks (for vectorization)
/// - Strides are used for offset updates (no multiplication per element)
///
/// # Julia equivalent
/// ```julia
/// @generated function _mapreduce_kernel!(f, op, initop, dims, blocks, arrays, strides, offsets)
///     # Generated nested loops with stride-based indexing
/// end
/// ```
#[inline]
pub(crate) fn mapreduce_kernel<F>(
    dims: &[usize],
    plan: &KernelPlan,
    strides_list: &[&[isize]],
    mut f: F,
) -> Result<()>
where
    F: FnMut(&[isize]) -> Result<()>,
{
    let rank = dims.len();
    if rank == 0 {
        let offsets = vec![0isize; strides_list.len()];
        return f(&offsets);
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
        1 => kernel_1d(&ordered_dims, &ordered_blocks, &ordered_strides, &mut offsets, &mut f),
        2 => kernel_2d(&ordered_dims, &ordered_blocks, &ordered_strides, &mut offsets, &mut f),
        3 => kernel_3d(&ordered_dims, &ordered_blocks, &ordered_strides, &mut offsets, &mut f),
        4 => kernel_4d(&ordered_dims, &ordered_blocks, &ordered_strides, &mut offsets, &mut f),
        _ => kernel_nd(&ordered_dims, &ordered_blocks, &ordered_strides, &mut offsets, &mut f),
    }
}

/// Execute a map operation with inner block callback.
///
/// This is similar to `mapreduce_kernel` but calls the callback with
/// (offsets, block_length, inner_strides) for the innermost dimension,
/// allowing vectorization of the inner loop.
#[inline]
pub(crate) fn mapreduce_kernel_inner_block<F>(
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
        1 => kernel_1d_inner(&ordered_dims, &ordered_blocks, &ordered_strides, &mut offsets, &mut f),
        2 => kernel_2d_inner(&ordered_dims, &ordered_blocks, &ordered_strides, &mut offsets, &mut f),
        3 => kernel_3d_inner(&ordered_dims, &ordered_blocks, &ordered_strides, &mut offsets, &mut f),
        4 => kernel_4d_inner(&ordered_dims, &ordered_blocks, &ordered_strides, &mut offsets, &mut f),
        _ => kernel_nd_inner(&ordered_dims, &ordered_blocks, &ordered_strides, &mut offsets, &mut f),
    }
}

// ============================================================================
// Specialized kernels for common dimensions (element-wise callback)
// ============================================================================

/// 1D kernel - single dimension
#[inline]
fn kernel_1d<F>(
    dims: &[usize],
    blocks: &[usize],
    strides: &[Vec<isize>],
    offsets: &mut [isize],
    f: &mut F,
) -> Result<()>
where
    F: FnMut(&[isize]) -> Result<()>,
{
    let d0 = dims[0];
    let b0 = blocks[0].max(1).min(d0);

    let mut j0 = 0usize;
    while j0 < d0 {
        let block_len = b0.min(d0 - j0);

        // Inner loop - iterate within block using stride addition
        for _ in 0..block_len {
            f(offsets)?;

            // Step all offsets by their strides
            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                *offset += s[0];
            }
        }

        // Return offsets (subtract block_len * stride, already done by step)
        // No need - next block continues from current position

        j0 += block_len;
    }

    // Reset offsets to original
    for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
        *offset -= (d0 as isize) * s[0];
    }

    Ok(())
}

/// 2D kernel
#[inline]
fn kernel_2d<F>(
    dims: &[usize],
    blocks: &[usize],
    strides: &[Vec<isize>],
    offsets: &mut [isize],
    f: &mut F,
) -> Result<()>
where
    F: FnMut(&[isize]) -> Result<()>,
{
    let d0 = dims[0];
    let d1 = dims[1];
    let b0 = blocks[0].max(1).min(d0);
    let b1 = blocks[1].max(1).min(d1);

    let mut j0 = 0usize;
    while j0 < d0 {
        let block_len0 = b0.min(d0 - j0);

        let mut j1 = 0usize;
        while j1 < d1 {
            let block_len1 = b1.min(d1 - j1);

            // Inner loops
            for _ in 0..block_len0 {
                for _ in 0..block_len1 {
                    f(offsets)?;
                    for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                        *offset += s[1];
                    }
                }
                // Return inner stride
                for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                    *offset -= (block_len1 as isize) * s[1];
                    *offset += s[0];
                }
            }
            // Return outer stride for this block
            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                *offset -= (block_len0 as isize) * s[0];
            }

            // Move to next block in dimension 1
            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                *offset += (block_len1 as isize) * s[1];
            }
            j1 += block_len1;
        }

        // Reset dimension 1, advance dimension 0
        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
            *offset -= (d1 as isize) * s[1];
            *offset += (block_len0 as isize) * s[0];
        }
        j0 += block_len0;
    }

    // Reset all offsets
    for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
        *offset -= (d0 as isize) * s[0];
    }

    Ok(())
}

/// 3D kernel
#[inline]
fn kernel_3d<F>(
    dims: &[usize],
    blocks: &[usize],
    strides: &[Vec<isize>],
    offsets: &mut [isize],
    f: &mut F,
) -> Result<()>
where
    F: FnMut(&[isize]) -> Result<()>,
{
    let d0 = dims[0];
    let d1 = dims[1];
    let d2 = dims[2];
    let b0 = blocks[0].max(1).min(d0);
    let b1 = blocks[1].max(1).min(d1);
    let b2 = blocks[2].max(1).min(d2);

    let mut j0 = 0usize;
    while j0 < d0 {
        let blen0 = b0.min(d0 - j0);

        let mut j1 = 0usize;
        while j1 < d1 {
            let blen1 = b1.min(d1 - j1);

            let mut j2 = 0usize;
            while j2 < d2 {
                let blen2 = b2.min(d2 - j2);

                // Inner loops
                for _ in 0..blen0 {
                    for _ in 0..blen1 {
                        for _ in 0..blen2 {
                            f(offsets)?;
                            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
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

/// 4D kernel
#[inline]
fn kernel_4d<F>(
    dims: &[usize],
    blocks: &[usize],
    strides: &[Vec<isize>],
    offsets: &mut [isize],
    f: &mut F,
) -> Result<()>
where
    F: FnMut(&[isize]) -> Result<()>,
{
    let d0 = dims[0];
    let d1 = dims[1];
    let d2 = dims[2];
    let d3 = dims[3];
    let b0 = blocks[0].max(1).min(d0);
    let b1 = blocks[1].max(1).min(d1);
    let b2 = blocks[2].max(1).min(d2);
    let b3 = blocks[3].max(1).min(d3);

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

                    // Inner loops
                    for _ in 0..blen0 {
                        for _ in 0..blen1 {
                            for _ in 0..blen2 {
                                for _ in 0..blen3 {
                                    f(offsets)?;
                                    for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                                        *offset += s[3];
                                    }
                                }
                                for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
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

/// N-dimensional kernel (recursive fallback)
#[inline]
fn kernel_nd<F>(
    dims: &[usize],
    blocks: &[usize],
    strides: &[Vec<isize>],
    offsets: &mut [isize],
    f: &mut F,
) -> Result<()>
where
    F: FnMut(&[isize]) -> Result<()>,
{
    kernel_nd_level(0, dims, blocks, strides, offsets, f)
}

#[inline]
fn kernel_nd_level<F>(
    level: usize,
    dims: &[usize],
    blocks: &[usize],
    strides: &[Vec<isize>],
    offsets: &mut [isize],
    f: &mut F,
) -> Result<()>
where
    F: FnMut(&[isize]) -> Result<()>,
{
    if level == dims.len() {
        return f(offsets);
    }

    let d = dims[level];
    let b = blocks[level].max(1).min(d);

    let mut j = 0usize;
    while j < d {
        let blen = b.min(d - j);

        // Inner loop for this block
        kernel_nd_block(level, blen, dims, blocks, strides, offsets, f)?;

        // Advance to next block
        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
            *offset += (blen as isize) * s[level];
        }
        j += blen;
    }

    // Reset this dimension
    for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
        *offset -= (d as isize) * s[level];
    }

    Ok(())
}

#[inline]
fn kernel_nd_block<F>(
    level: usize,
    block_len: usize,
    dims: &[usize],
    blocks: &[usize],
    strides: &[Vec<isize>],
    offsets: &mut [isize],
    f: &mut F,
) -> Result<()>
where
    F: FnMut(&[isize]) -> Result<()>,
{
    if level == dims.len() - 1 {
        // Innermost level - just iterate
        for _ in 0..block_len {
            f(offsets)?;
            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                *offset += s[level];
            }
        }
        // Return stride
        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
            *offset -= (block_len as isize) * s[level];
        }
    } else {
        // Outer level - iterate and recurse
        for _ in 0..block_len {
            kernel_nd_level(level + 1, dims, blocks, strides, offsets, f)?;
            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                *offset += s[level];
            }
        }
        // Return stride
        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
            *offset -= (block_len as isize) * s[level];
        }
    }
    Ok(())
}

// ============================================================================
// Specialized kernels for common dimensions (inner-block callback)
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
// Legacy API (for backward compatibility)
// ============================================================================

/// Iterate over all elements, calling f with the current offsets.
///
/// This is the legacy API that wraps the new Julia-faithful kernel.
#[inline]
pub(crate) fn for_each_offset<F>(
    dims: &[usize],
    plan: &KernelPlan,
    strides_list: &[&[isize]],
    f: F,
) -> Result<()>
where
    F: FnMut(&[isize]) -> Result<()>,
{
    mapreduce_kernel(dims, plan, strides_list, f)
}

/// Iterate over blocks, calling f with (offsets, block_len, inner_strides).
///
/// This is the legacy API that wraps the new Julia-faithful kernel.
#[inline]
pub(crate) fn for_each_inner_block<F>(
    dims: &[usize],
    plan: &KernelPlan,
    strides_list: &[&[isize]],
    f: F,
) -> Result<()>
where
    F: FnMut(&[isize], usize, &[isize]) -> Result<()>,
{
    mapreduce_kernel_inner_block(dims, plan, strides_list, f)
}

// ============================================================================
// Utility functions
// ============================================================================

pub(crate) fn validate_layout(dims: &[usize], strides: &[isize]) -> Result<()> {
    if dims.len() != strides.len() {
        return Err(StridedError::StrideLengthMismatch);
    }
    for (i, (&dim, &stride)) in dims.iter().zip(strides.iter()).enumerate() {
        if dim > 1 && stride == 0 {
            return Err(StridedError::ZeroStride { dim: i });
        }
    }
    Ok(())
}

fn dims_and_strides<T, S: Shape, L: Layout>(slice: &Slice<T, S, L>) -> (Vec<usize>, Vec<isize>) {
    let rank = slice.rank();
    let mut dims = Vec::with_capacity(rank);
    let mut strides = Vec::with_capacity(rank);
    for i in 0..rank {
        dims.push(slice.dim(i));
        strides.push(slice.stride(i));
    }
    (dims, strides)
}

pub(crate) fn ensure_same_shape(a: &[usize], b: &[usize]) -> Result<()> {
    if a.len() != b.len() {
        return Err(StridedError::RankMismatch(a.len(), b.len()));
    }
    if a != b {
        return Err(StridedError::ShapeMismatch(a.to_vec(), b.to_vec()));
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
    fn test_kernel_1d() {
        let dims = vec![5];
        let strides1 = vec![1isize];
        let strides2 = vec![2isize];
        let strides_list: Vec<&[isize]> = vec![&strides1, &strides2];
        let plan = build_plan(&dims, &strides_list, Some(0), 8);

        let mut collected = Vec::new();
        mapreduce_kernel(&dims, &plan, &strides_list, |offsets| {
            collected.push(offsets.to_vec());
            Ok(())
        })
        .unwrap();

        assert_eq!(collected.len(), 5);
        // First array: 0, 1, 2, 3, 4
        // Second array: 0, 2, 4, 6, 8
        assert_eq!(collected[0], vec![0, 0]);
        assert_eq!(collected[1], vec![1, 2]);
        assert_eq!(collected[4], vec![4, 8]);
    }

    #[test]
    fn test_kernel_2d() {
        let dims = vec![2, 3];
        let strides1 = vec![3isize, 1]; // Row-major
        let strides2 = vec![1isize, 2]; // Column-major
        let strides_list: Vec<&[isize]> = vec![&strides1, &strides2];
        let plan = build_plan(&dims, &strides_list, Some(0), 8);

        let mut collected = Vec::new();
        mapreduce_kernel(&dims, &plan, &strides_list, |offsets| {
            collected.push(offsets.to_vec());
            Ok(())
        })
        .unwrap();

        assert_eq!(collected.len(), 6);
    }

    #[test]
    fn test_kernel_inner_block() {
        let dims = vec![2, 4];
        let strides1 = vec![4isize, 1];
        let strides2 = vec![4isize, 1];
        let strides_list: Vec<&[isize]> = vec![&strides1, &strides2];
        let plan = build_plan(&dims, &strides_list, Some(0), 8);

        let mut total_elements = 0usize;
        mapreduce_kernel_inner_block(&dims, &plan, &strides_list, |_offsets, len, _strides| {
            total_elements += len;
            Ok(())
        })
        .unwrap();

        assert_eq!(total_elements, 8);
    }
}
