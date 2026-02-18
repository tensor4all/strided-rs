//! HPTT-inspired cache-efficient tensor permutation.
//!
//! Key techniques:
//! 1. Bilateral dimension fusion (fuse dims contiguous in both src and dst)
//! 2. Cache-aware blocking (L1-sized tiles)
//! 3. Optimal loop ordering (stride-1 innermost)

use crate::fuse::fuse_dims_bilateral;
use crate::{BLOCK_MEMORY_SIZE, CACHE_LINE_SIZE};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Target tile size for permutation blocking.
/// Use full L1 (32KB) since permutation is pure copy with no computation.
const TILE_TARGET: usize = BLOCK_MEMORY_SIZE;

/// Minimum number of elements to justify multi-threaded execution.
#[cfg(feature = "parallel")]
const MINTHREADLENGTH: usize = 1 << 15; // 32768

/// Plan for a blocked permutation copy.
#[derive(Debug)]
pub struct PermutePlan {
    /// Fused dimensions (after bilateral fusion).
    pub fused_dims: Vec<usize>,
    /// Fused source strides.
    pub src_strides: Vec<isize>,
    /// Fused destination strides.
    pub dst_strides: Vec<isize>,
    /// Block (tile) sizes per fused dimension.
    pub block_sizes: Vec<usize>,
    /// Loop iteration order (outermost first, innermost last).
    pub loop_order: Vec<usize>,
}

/// Build a permutation plan using bilateral fusion and cache-aware blocking.
pub fn build_permute_plan(
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
    elem_size: usize,
) -> PermutePlan {
    // Phase 1: Bilateral dimension fusion
    let (fused_dims, fused_src, fused_dst) = fuse_dims_bilateral(dims, src_strides, dst_strides);

    let rank = fused_dims.len();
    if rank == 0 {
        return PermutePlan {
            fused_dims,
            src_strides: fused_src,
            dst_strides: fused_dst,
            block_sizes: vec![],
            loop_order: vec![],
        };
    }

    // Phase 2: Compute optimal loop order
    // Put stride-1 (or smallest stride) dimension innermost (last in loop_order).
    // For permutation: prefer dimension where EITHER src or dst has stride 1,
    // with preference for dst stride 1 (sequential writes).
    let loop_order = compute_perm_order(&fused_dims, &fused_src, &fused_dst);

    // Phase 3: Compute block sizes to fit in L1 cache
    let block_sizes =
        compute_perm_blocks(&fused_dims, &fused_src, &fused_dst, &loop_order, elem_size);

    PermutePlan {
        fused_dims,
        src_strides: fused_src,
        dst_strides: fused_dst,
        block_sizes,
        loop_order,
    }
}

/// Compute iteration order for permutation.
///
/// Strategy: the innermost dimension (last in returned order) should be the one
/// with the smallest stride in either src or dst, preferring dst (writes).
/// Outer dimensions are sorted by descending stride magnitude so that
/// larger strides are in the outermost loops (better for blocking).
fn compute_perm_order(dims: &[usize], src_strides: &[isize], dst_strides: &[isize]) -> Vec<usize> {
    let rank = dims.len();
    if rank <= 1 {
        return (0..rank).collect();
    }

    // Find the dimension with the smallest min-stride (preferring dst for writes).
    // Tie-break: prefer dst stride 1 over src stride 1.
    let mut inner_dim = 0;
    let mut inner_score = score_for_inner(src_strides[0], dst_strides[0], dims[0]);

    for d in 1..rank {
        if dims[d] <= 1 {
            continue;
        }
        let s = score_for_inner(src_strides[d], dst_strides[d], dims[d]);
        if s < inner_score || (s == inner_score && dims[d] > dims[inner_dim]) {
            inner_score = s;
            inner_dim = d;
        }
    }

    // Build order: outer dims sorted by max stride magnitude (descending),
    // inner dim last.
    let mut outer: Vec<usize> = (0..rank).filter(|&d| d != inner_dim).collect();
    outer.sort_by(|&a, &b| {
        let sa = src_strides[a]
            .unsigned_abs()
            .max(dst_strides[a].unsigned_abs());
        let sb = src_strides[b]
            .unsigned_abs()
            .max(dst_strides[b].unsigned_abs());
        sb.cmp(&sa) // descending
    });
    outer.push(inner_dim);
    outer
}

/// Score a dimension for being the innermost loop.
/// Lower score = better for inner.
///
/// Strategy: minimize the minimum stride in the inner dimension
/// (to enable contiguous access on at least one side).
/// Tiebreak: prefer dst stride 1 (sequential writes, write-combining).
fn score_for_inner(src_stride: isize, dst_stride: isize, dim: usize) -> u64 {
    if dim <= 1 {
        return u64::MAX;
    }
    let sa = src_stride.unsigned_abs() as u64;
    let da = dst_stride.unsigned_abs() as u64;
    let min_stride = sa.min(da);
    // Primary: smallest min-stride wins (at least one side is contiguous)
    // Secondary: prefer dst stride 1 for write-combining
    let bonus = if da <= sa { 0u64 } else { 1u64 };
    min_stride * 4 + bonus
}

/// Compute block sizes for cache-aware tiling.
fn compute_perm_blocks(
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
    loop_order: &[usize],
    elem_size: usize,
) -> Vec<usize> {
    let rank = dims.len();
    if rank == 0 {
        return vec![];
    }

    let mut blocks = dims.to_vec();

    // Compute total memory footprint of current blocks
    let footprint = |blk: &[usize]| -> usize {
        tile_memory_footprint(blk, src_strides, dst_strides, elem_size)
    };

    if footprint(&blocks) <= TILE_TARGET {
        return blocks;
    }

    // The innermost dimension (last in loop_order) keeps its full extent
    // to maximize vectorization.  Reduce outer dimensions first.
    //
    // Iterate from outermost to innermost-1, halving until we fit.
    // We use a multi-pass approach: first halve the outermost dims,
    // then if still too big, reduce the inner dim.

    // Phase 1: Halve outer dimensions (outermost first)
    for pass in 0..20 {
        if footprint(&blocks) <= TILE_TARGET {
            break;
        }
        let mut changed = false;
        // loop_order[0] is outermost; loop_order[rank-1] is innermost
        // Skip the innermost in early passes
        let limit = if pass < 10 { rank - 1 } else { rank };
        for &d in &loop_order[..limit] {
            if blocks[d] <= 1 {
                continue;
            }
            if footprint(&blocks) <= TILE_TARGET {
                break;
            }
            blocks[d] = (blocks[d] + 1) / 2;
            changed = true;
        }
        if !changed {
            break;
        }
    }

    // Phase 2: Fine-tune - decrement the largest block until we fit
    while footprint(&blocks) > TILE_TARGET {
        // Find the dimension with the largest block (preferring outer dims)
        let mut best = None;
        let mut best_size = 0;
        for &d in loop_order {
            if blocks[d] > 1 && blocks[d] > best_size {
                best_size = blocks[d];
                best = Some(d);
            }
        }
        match best {
            Some(d) => blocks[d] -= 1,
            None => break,
        }
    }

    // Ensure innermost block is at least the cache line width (if dimension allows)
    let inner_dim = loop_order[rank - 1];
    let inner_min = (CACHE_LINE_SIZE / elem_size).max(1).min(dims[inner_dim]);
    if blocks[inner_dim] < inner_min {
        blocks[inner_dim] = inner_min;
    }

    blocks
}

/// Estimate the memory footprint of a tile.
///
/// For each of src and dst, compute the memory region touched:
///   sum over dims of (block[d] - 1) * |stride[d]| * elem_size + elem_size
fn tile_memory_footprint(
    blocks: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
    elem_size: usize,
) -> usize {
    let src_region = stride_footprint(blocks, src_strides, elem_size);
    let dst_region = stride_footprint(blocks, dst_strides, elem_size);
    src_region + dst_region
}

/// Memory region touched by one array with given block sizes and strides.
fn stride_footprint(blocks: &[usize], strides: &[isize], elem_size: usize) -> usize {
    let cl = CACHE_LINE_SIZE;
    let mut contiguous_bytes = 0isize;
    let mut cache_line_blocks = 1usize;

    for (&b, &s) in blocks.iter().zip(strides.iter()) {
        let s_bytes = (s.unsigned_abs() * elem_size) as isize;
        if s_bytes < cl as isize {
            contiguous_bytes += (b.saturating_sub(1) as isize) * s_bytes;
        } else {
            cache_line_blocks *= b;
        }
    }

    let lines = (contiguous_bytes as usize / cl) + 1;
    cl * lines * cache_line_blocks
}

/// Execute the blocked permutation copy.
///
/// # Safety
/// - `src` must be valid for reads at all offsets determined by dims/src_strides
/// - `dst` must be valid for writes at all offsets determined by dims/dst_strides
/// - src and dst must not overlap
pub unsafe fn execute_permute_blocked<T: Copy>(src: *const T, dst: *mut T, plan: &PermutePlan) {
    let rank = plan.fused_dims.len();
    if rank == 0 {
        *dst = *src;
        return;
    }

    let dims = &plan.fused_dims;
    let src_s = &plan.src_strides;
    let dst_s = &plan.dst_strides;
    let blocks = &plan.block_sizes;
    let order = &plan.loop_order;

    // Reorder everything to loop_order so that iteration is 0..rank
    // with dimension 0 = outermost, rank-1 = innermost.
    let mut o_dims = vec![0usize; rank];
    let mut o_blocks = vec![0usize; rank];
    let mut o_src_s = vec![0isize; rank];
    let mut o_dst_s = vec![0isize; rank];
    for (i, &d) in order.iter().enumerate() {
        o_dims[i] = dims[d];
        o_blocks[i] = blocks[d];
        o_src_s[i] = src_s[d];
        o_dst_s[i] = dst_s[d];
    }

    blocked_copy_ordered(src, dst, &o_dims, &o_src_s, &o_dst_s, &o_blocks);
}

/// Execute the blocked permutation copy with Rayon parallelism.
///
/// Parallelizes the outermost block loop using `rayon::par_iter`.
/// Falls back to single-threaded for small tensors (< MINTHREADLENGTH elements).
///
/// # Safety
/// Same requirements as `execute_permute_blocked`.
#[cfg(feature = "parallel")]
pub unsafe fn execute_permute_blocked_par<T: Copy + Send + Sync>(
    src: *const T,
    dst: *mut T,
    plan: &PermutePlan,
) {
    let rank = plan.fused_dims.len();
    let total: usize = plan.fused_dims.iter().product();

    // Fall back to single-threaded for small tensors or rank 0
    if rank == 0 || total < MINTHREADLENGTH {
        execute_permute_blocked(src, dst, plan);
        return;
    }

    let dims = &plan.fused_dims;
    let src_s = &plan.src_strides;
    let dst_s = &plan.dst_strides;
    let blocks = &plan.block_sizes;
    let order = &plan.loop_order;

    // Reorder to loop_order
    let mut o_dims = vec![0usize; rank];
    let mut o_blocks = vec![0usize; rank];
    let mut o_src_s = vec![0isize; rank];
    let mut o_dst_s = vec![0isize; rank];
    for (i, &d) in order.iter().enumerate() {
        o_dims[i] = dims[d];
        o_blocks[i] = blocks[d];
        o_src_s[i] = src_s[d];
        o_dst_s[i] = dst_s[d];
    }

    // Parallelize over outermost block loop (dim 0).
    let n_outer = (o_dims[0] + o_blocks[0] - 1) / o_blocks[0];

    if n_outer <= 1 || rank <= 1 {
        // Not enough outer blocks to parallelize
        blocked_copy_ordered(src, dst, &o_dims, &o_src_s, &o_dst_s, &o_blocks);
        return;
    }

    // Convert pointers to usize to avoid raw-pointer Send/Sync issues in closures.
    let src_addr = src as usize;
    let dst_addr = dst as usize;
    let outer_block = o_blocks[0];
    let outer_dim = o_dims[0];
    let outer_src_stride = o_src_s[0];
    let outer_dst_stride = o_dst_s[0];
    let elem_size = std::mem::size_of::<T>();

    (0..n_outer).into_par_iter().for_each(|block_idx| {
        let start = block_idx * outer_block;
        let extent = outer_block.min(outer_dim - start);

        // Compute byte offsets and reconstruct pointers
        let src_byte_off = (start as isize) * outer_src_stride * (elem_size as isize);
        let dst_byte_off = (start as isize) * outer_dst_stride * (elem_size as isize);
        let sub_src = (src_addr as isize + src_byte_off) as *const T;
        let sub_dst = (dst_addr as isize + dst_byte_off) as *mut T;

        let mut sub_dims = o_dims.clone();
        sub_dims[0] = extent;

        unsafe {
            blocked_copy_ordered(sub_src, sub_dst, &sub_dims, &o_src_s, &o_dst_s, &o_blocks);
        }
    });
}

/// Blocked copy with dimensions already in iteration order.
///
/// Dim 0 is outermost, dim rank-1 is innermost.
/// Dispatches to rank-specialized kernels for rank 1-3.
unsafe fn blocked_copy_ordered<T: Copy>(
    src: *const T,
    dst: *mut T,
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
    blocks: &[usize],
) {
    match dims.len() {
        1 => blocked_copy_1d(src, dst, dims[0], blocks[0], src_strides[0], dst_strides[0]),
        2 => blocked_copy_2d(
            src,
            dst,
            [dims[0], dims[1]],
            [blocks[0], blocks[1]],
            [src_strides[0], src_strides[1]],
            [dst_strides[0], dst_strides[1]],
        ),
        3 => blocked_copy_3d(
            src,
            dst,
            [dims[0], dims[1], dims[2]],
            [blocks[0], blocks[1], blocks[2]],
            [src_strides[0], src_strides[1], src_strides[2]],
            [dst_strides[0], dst_strides[1], dst_strides[2]],
        ),
        _ => blocked_copy_nd(src, dst, dims, src_strides, dst_strides, blocks),
    }
}

/// 1D blocked copy.
#[inline]
unsafe fn blocked_copy_1d<T: Copy>(
    src: *const T,
    dst: *mut T,
    dim: usize,
    block: usize,
    src_stride: isize,
    dst_stride: isize,
) {
    let mut j = 0usize;
    while j < dim {
        let len = block.min(dim - j);
        copy_inner_loop(
            src.offset((j as isize) * src_stride),
            dst.offset((j as isize) * dst_stride),
            len,
            src_stride,
            dst_stride,
        );
        j += block;
    }
}

/// 2D blocked copy — the most important case for transpositions.
///
/// After bilateral fusion, most transpositions reduce to a 2D problem.
/// This uses tiled iteration with tight inner loops.
#[inline]
unsafe fn blocked_copy_2d<T: Copy>(
    src: *const T,
    dst: *mut T,
    dims: [usize; 2],
    blocks: [usize; 2],
    src_s: [isize; 2],
    dst_s: [isize; 2],
) {
    let mut j0 = 0usize;
    while j0 < dims[0] {
        let b0 = blocks[0].min(dims[0] - j0);
        let src_row = src.offset((j0 as isize) * src_s[0]);
        let dst_row = dst.offset((j0 as isize) * dst_s[0]);

        let mut j1 = 0usize;
        while j1 < dims[1] {
            let b1 = blocks[1].min(dims[1] - j1);
            let src_tile = src_row.offset((j1 as isize) * src_s[1]);
            let dst_tile = dst_row.offset((j1 as isize) * dst_s[1]);

            // Copy tile [b0 x b1]: outer loop over dim 0, inner over dim 1
            let mut sp = src_tile;
            let mut dp = dst_tile;
            for _ in 0..b0 {
                copy_inner_loop(sp, dp, b1, src_s[1], dst_s[1]);
                sp = sp.offset(src_s[0]);
                dp = dp.offset(dst_s[0]);
            }

            j1 += blocks[1];
        }
        j0 += blocks[0];
    }
}

/// 3D blocked copy.
#[inline]
unsafe fn blocked_copy_3d<T: Copy>(
    src: *const T,
    dst: *mut T,
    dims: [usize; 3],
    blocks: [usize; 3],
    src_s: [isize; 3],
    dst_s: [isize; 3],
) {
    let mut j0 = 0usize;
    while j0 < dims[0] {
        let b0 = blocks[0].min(dims[0] - j0);
        let src0 = src.offset((j0 as isize) * src_s[0]);
        let dst0 = dst.offset((j0 as isize) * dst_s[0]);

        let mut j1 = 0usize;
        while j1 < dims[1] {
            let b1 = blocks[1].min(dims[1] - j1);
            let src1 = src0.offset((j1 as isize) * src_s[1]);
            let dst1 = dst0.offset((j1 as isize) * dst_s[1]);

            let mut j2 = 0usize;
            while j2 < dims[2] {
                let b2 = blocks[2].min(dims[2] - j2);
                let src_tile = src1.offset((j2 as isize) * src_s[2]);
                let dst_tile = dst1.offset((j2 as isize) * dst_s[2]);

                // Copy tile: outer over dim 0, mid over dim 1, inner over dim 2
                let mut sp0 = src_tile;
                let mut dp0 = dst_tile;
                for _ in 0..b0 {
                    let mut sp1 = sp0;
                    let mut dp1 = dp0;
                    for _ in 0..b1 {
                        copy_inner_loop(sp1, dp1, b2, src_s[2], dst_s[2]);
                        sp1 = sp1.offset(src_s[1]);
                        dp1 = dp1.offset(dst_s[1]);
                    }
                    sp0 = sp0.offset(src_s[0]);
                    dp0 = dp0.offset(dst_s[0]);
                }

                j2 += blocks[2];
            }
            j1 += blocks[1];
        }
        j0 += blocks[0];
    }
}

/// N-dimensional blocked copy (generic fallback for rank >= 4).
unsafe fn blocked_copy_nd<T: Copy>(
    src: *const T,
    dst: *mut T,
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
    blocks: &[usize],
) {
    let rank = dims.len();

    let mut block_counts = vec![0usize; rank];
    for d in 0..rank {
        block_counts[d] = (dims[d] + blocks[d] - 1) / blocks[d];
    }

    let mut blk_idx = vec![0usize; rank];
    let mut src_blk_off = 0isize;
    let mut dst_blk_off = 0isize;
    let mut elem_idx = vec![0usize; rank];
    let mut tile_ext = vec![0usize; rank];

    let inner_src_stride = src_strides[rank - 1];
    let inner_dst_stride = dst_strides[rank - 1];
    let outer_rank = rank - 1;

    loop {
        let inner_extent =
            blocks[rank - 1].min(dims[rank - 1] - blk_idx[rank - 1] * blocks[rank - 1]);
        tile_ext[rank - 1] = inner_extent;

        let mut total_outer = 1usize;
        for d in 0..outer_rank {
            let start = blk_idx[d] * blocks[d];
            let tile_d = blocks[d].min(dims[d] - start);
            tile_ext[d] = tile_d;
            elem_idx[d] = 0;
            total_outer *= tile_d;
        }

        let mut src_elem_off = 0isize;
        let mut dst_elem_off = 0isize;

        for _ in 0..total_outer {
            copy_inner_loop(
                src.offset(src_blk_off + src_elem_off),
                dst.offset(dst_blk_off + dst_elem_off),
                inner_extent,
                inner_src_stride,
                inner_dst_stride,
            );

            for d in (0..outer_rank).rev() {
                elem_idx[d] += 1;
                src_elem_off += src_strides[d];
                dst_elem_off += dst_strides[d];
                if elem_idx[d] < tile_ext[d] {
                    break;
                }
                src_elem_off -= (tile_ext[d] as isize) * src_strides[d];
                dst_elem_off -= (tile_ext[d] as isize) * dst_strides[d];
                elem_idx[d] = 0;
            }
        }

        let mut carry = true;
        for d in (0..rank).rev() {
            if !carry {
                break;
            }
            blk_idx[d] += 1;
            src_blk_off += (blocks[d] as isize) * src_strides[d];
            dst_blk_off += (blocks[d] as isize) * dst_strides[d];

            if blk_idx[d] < block_counts[d] {
                carry = false;
            } else {
                src_blk_off -= (blk_idx[d] as isize) * (blocks[d] as isize) * src_strides[d];
                dst_blk_off -= (blk_idx[d] as isize) * (blocks[d] as isize) * dst_strides[d];
                blk_idx[d] = 0;
            }
        }

        if carry {
            break;
        }
    }
}

/// Inner copy loop along a single dimension.
///
/// This is the hot path that gets auto-vectorized by the compiler.
#[inline(always)]
unsafe fn copy_inner_loop<T: Copy>(
    src: *const T,
    dst: *mut T,
    count: usize,
    src_stride: isize,
    dst_stride: isize,
) {
    if src_stride == 1 && dst_stride == 1 {
        // Both contiguous: use memcpy
        std::ptr::copy_nonoverlapping(src, dst, count);
    } else if dst_stride == 1 {
        // Sequential writes (gather pattern)
        let mut s = src;
        let mut d = dst;
        for _ in 0..count {
            *d = *s;
            s = s.offset(src_stride);
            d = d.add(1);
        }
    } else if src_stride == 1 {
        // Sequential reads (scatter pattern)
        let mut s = src;
        let mut d = dst;
        for _ in 0..count {
            *d = *s;
            s = s.add(1);
            d = d.offset(dst_stride);
        }
    } else {
        // Both non-unit stride
        let mut s = src;
        let mut d = dst;
        for _ in 0..count {
            *d = *s;
            s = s.offset(src_stride);
            d = d.offset(dst_stride);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_plan_identity() {
        // Identity permutation: should fuse everything into 1 dim
        let plan = build_permute_plan(&[2, 3, 4], &[1, 2, 6], &[1, 2, 6], 8);
        assert_eq!(plan.fused_dims, vec![24]);
        assert_eq!(plan.src_strides, vec![1]);
        assert_eq!(plan.dst_strides, vec![1]);
    }

    #[test]
    fn test_build_plan_transpose_2d() {
        // [4, 5] with src col-major [1, 4], dst row-major [5, 1]
        let plan = build_permute_plan(&[4, 5], &[1, 4], &[5, 1], 8);
        // No bilateral fusion possible (strides differ)
        assert_eq!(plan.fused_dims, vec![4, 5]);
    }

    #[test]
    fn test_execute_identity_copy() {
        let src = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0f64; 6];
        let plan = build_permute_plan(&[2, 3], &[1, 2], &[1, 2], 8);
        unsafe {
            execute_permute_blocked(src.as_ptr(), dst.as_mut_ptr(), &plan);
        }
        assert_eq!(dst, src);
    }

    #[test]
    fn test_execute_transpose_2d() {
        // src [3, 2] col-major: [[1,4],[2,5],[3,6]]
        let src = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0f64; 6];
        // Transpose: dst should be [2, 3] col-major
        // dst strides for "permuted" dims: src[1,3] -> dst[1,2]
        // But we're doing: dst[i,j] = src[j,i]
        // src dims [3,2], src strides [1,3]
        // permuted view: dims [2,3], strides [3,1]
        // dst dims [2,3], strides [1,2]
        let plan = build_permute_plan(&[2, 3], &[3, 1], &[1, 2], 8);
        unsafe {
            execute_permute_blocked(src.as_ptr(), dst.as_mut_ptr(), &plan);
        }
        // Expected: dst = [1, 4, 2, 5, 3, 6] (col-major [2,3])
        assert_eq!(dst, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_execute_3d_permute() {
        // src [2,3,4] col-major, permute [2,0,1]
        let dims = [2usize, 3, 4];
        let total: usize = dims.iter().product();
        let src: Vec<f64> = (0..total).map(|i| i as f64).collect();
        let mut dst = vec![0.0f64; total];

        // src strides (col-major): [1, 2, 6]
        // After permute [2,0,1]: dims [4,2,3], strides [6,1,2]
        // dst col-major for [4,2,3]: strides [1, 4, 8]
        let plan = build_permute_plan(&[4, 2, 3], &[6, 1, 2], &[1, 4, 8], 8);
        unsafe {
            execute_permute_blocked(src.as_ptr(), dst.as_mut_ptr(), &plan);
        }

        // Verify: dst[k, i, j] should equal src[i*1 + j*2 + k*6]
        for k in 0..4 {
            for i in 0..2 {
                for j in 0..3 {
                    let dst_idx = k + i * 4 + j * 8;
                    let src_idx = i + j * 2 + k * 6;
                    assert_eq!(
                        dst[dst_idx], src[src_idx],
                        "mismatch at k={k}, i={i}, j={j}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_score_for_inner() {
        // Prefer dst stride 1 over src stride 1
        assert!(score_for_inner(4, 1, 10) < score_for_inner(1, 4, 10));
        // Both stride 1 is best
        assert!(score_for_inner(1, 1, 10) <= score_for_inner(4, 1, 10));
        // Size-1 dims should not be inner
        assert_eq!(score_for_inner(1, 1, 1), u64::MAX);
    }

    #[test]
    fn test_loop_order_prefers_stride1() {
        // Dims [4, 5], src strides [1, 4], dst strides [5, 1]
        // Dim 0: min(1,5)=1, dim 1: min(4,1)=1 → tiebreak on dst: dim 1 (dst stride 1)
        let order = compute_perm_order(&[4, 5], &[1, 4], &[5, 1]);
        assert_eq!(*order.last().unwrap(), 1);
    }

    #[test]
    fn test_execute_4d_permute() {
        // 4D: dims [2, 3, 4, 5], col-major src, permute [3, 1, 0, 2]
        let dims = [2usize, 3, 4, 5];
        let total: usize = dims.iter().product();
        let src: Vec<f64> = (0..total).map(|i| i as f64).collect();
        let mut dst = vec![0.0f64; total];

        // src col-major strides: [1, 2, 6, 24]
        // permuted dims: [5, 3, 2, 4], strides: [24, 2, 1, 6]
        // dst col-major for [5, 3, 2, 4]: [1, 5, 15, 30]
        let plan = build_permute_plan(&[5, 3, 2, 4], &[24, 2, 1, 6], &[1, 5, 15, 30], 8);
        unsafe {
            execute_permute_blocked(src.as_ptr(), dst.as_mut_ptr(), &plan);
        }

        // Verify sample: multi-index (i0, i1, i2, i3)
        // src offset = i0*24 + i1*2 + i2*1 + i3*6
        // dst offset = i0*1 + i1*5 + i2*15 + i3*30
        for i0 in 0..5 {
            for i1 in 0..3 {
                for i2 in 0..2 {
                    for i3 in 0..4 {
                        let src_idx = i0 * 24 + i1 * 2 + i2 + i3 * 6;
                        let dst_idx = i0 + i1 * 5 + i2 * 15 + i3 * 30;
                        assert_eq!(
                            dst[dst_idx], src[src_idx],
                            "4D mismatch at ({i0},{i1},{i2},{i3})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_execute_5d_permute() {
        // 5D: dims [2, 2, 2, 2, 3], permute [4, 0, 1, 2, 3]
        let dims = [2usize, 2, 2, 2, 3];
        let total: usize = dims.iter().product();
        let src: Vec<f64> = (0..total).map(|i| i as f64).collect();
        let mut dst = vec![0.0f64; total];

        // src col-major: [1, 2, 4, 8, 16]
        // permuted: dims [3, 2, 2, 2, 2], strides [16, 1, 2, 4, 8]
        // dst col-major for [3, 2, 2, 2, 2]: [1, 3, 6, 12, 24]
        let plan = build_permute_plan(&[3, 2, 2, 2, 2], &[16, 1, 2, 4, 8], &[1, 3, 6, 12, 24], 8);
        unsafe {
            execute_permute_blocked(src.as_ptr(), dst.as_mut_ptr(), &plan);
        }

        // Verify all elements
        for i0 in 0..3 {
            for i1 in 0..2 {
                for i2 in 0..2 {
                    for i3 in 0..2 {
                        for i4 in 0..2 {
                            let src_idx = i0 * 16 + i1 + i2 * 2 + i3 * 4 + i4 * 8;
                            let dst_idx = i0 + i1 * 3 + i2 * 6 + i3 * 12 + i4 * 24;
                            assert_eq!(
                                dst[dst_idx], src[src_idx],
                                "5D mismatch at ({i0},{i1},{i2},{i3},{i4})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_execute_rank0_scalar() {
        let src = vec![42.0f64];
        let mut dst = vec![0.0f64];
        let plan = build_permute_plan(&[], &[], &[], 8);
        unsafe {
            execute_permute_blocked(src.as_ptr(), dst.as_mut_ptr(), &plan);
        }
        assert_eq!(dst[0], 42.0);
    }

    #[test]
    fn test_tile_memory_footprint_basic() {
        // 2D tile [8, 8], strides [1, 8] and [8, 1], elem 8 bytes
        let blocks = [8usize, 8];
        let src_s = [1isize, 8];
        let dst_s = [8isize, 1];
        let fp = tile_memory_footprint(&blocks, &src_s, &dst_s, 8);
        assert!(fp > 0);
    }

    #[test]
    fn test_stride_footprint_contiguous() {
        // Contiguous: blocks [100], stride [1], elem 8
        let fp = stride_footprint(&[100], &[1], 8);
        // 99 * 8 = 792 bytes in contiguous region → 792/64 + 1 = 13 cache lines
        assert_eq!(fp, 64 * 13);
    }

    #[test]
    fn test_stride_footprint_large_stride() {
        // Large stride >= cache line: each block element is a separate cache line block
        let fp = stride_footprint(&[10], &[100], 8);
        // stride 100*8 = 800 bytes >= 64 → cache_line_blocks = 10
        // contiguous_bytes = 0 → lines = 1
        assert_eq!(fp, 64 * 1 * 10);
    }

    #[test]
    fn test_compute_perm_order_single_dim() {
        let order = compute_perm_order(&[10], &[1], &[1]);
        assert_eq!(order, vec![0]);
    }

    #[test]
    fn test_compute_perm_order_3d() {
        // 3D: src [1, 10, 100], dst [100, 10, 1]
        // Min strides: dim 0 → min(1,100)=1, dim 1 → min(10,10)=10, dim 2 → min(100,1)=1
        // Dim 0 and 2 tie with min=1. Dim 0: dst=100>src=1 → bonus=1. Dim 2: dst=1<=src=100 → bonus=0.
        // Dim 2 wins (lower score).
        let order = compute_perm_order(&[5, 5, 5], &[1, 10, 100], &[100, 10, 1]);
        assert_eq!(*order.last().unwrap(), 2);
    }

    #[test]
    fn test_scattered_strides_plan() {
        // Simplified scattered case: 4 dims of size 2
        let dims = vec![2, 2, 2, 2];
        let src_strides = vec![1, 8, 2, 4]; // scattered
        let dst_strides = vec![1, 2, 4, 8]; // col-major
        let plan = build_permute_plan(&dims, &src_strides, &dst_strides, 8);
        // Dims 2-3 fuse bilaterally (src: 2→4 contiguous, dst: 4→8 contiguous)
        // So we get 3 fused dims: [2, 2, 4]
        assert_eq!(plan.fused_dims.len(), 3);
        // Inner dim should be dim 0 (both have stride 1)
        assert_eq!(*plan.loop_order.last().unwrap(), 0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_execute_par_transpose_2d() {
        let src = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0f64; 6];
        let plan = build_permute_plan(&[2, 3], &[3, 1], &[1, 2], 8);
        unsafe {
            execute_permute_blocked_par(src.as_ptr(), dst.as_mut_ptr(), &plan);
        }
        assert_eq!(dst, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_execute_par_large() {
        // Large enough to trigger parallel execution (> MINTHREADLENGTH)
        let n = 256;
        let total = n * n * n;
        let src: Vec<f64> = (0..total).map(|i| i as f64).collect();
        let mut dst = vec![0.0f64; total];

        // [256, 256, 256] col-major, transpose [2, 0, 1]
        // src strides: [1, 256, 65536]
        // permuted dims: [256, 256, 256], strides: [65536, 1, 256]
        // dst col-major: [1, 256, 65536]
        let plan = build_permute_plan(&[n, n, n], &[65536, 1, 256], &[1, 256, 65536], 8);
        unsafe {
            execute_permute_blocked_par(src.as_ptr(), dst.as_mut_ptr(), &plan);
        }

        // Verify: for multi-index (i0, i1, i2),
        //   src offset = i0*65536 + i1*1 + i2*256
        //   dst offset = i0*1 + i1*256 + i2*65536
        for i0 in [0, 1, 127, 255] {
            for i1 in [0, 1, 127, 255] {
                for i2 in [0, 1, 127, 255] {
                    let dst_idx = i0 + i1 * n + i2 * n * n;
                    let src_idx = i0 * 65536 + i1 + i2 * 256;
                    assert_eq!(
                        dst[dst_idx], src[src_idx],
                        "mismatch at i0={i0}, i1={i1}, i2={i2}"
                    );
                }
            }
        }
    }
}
