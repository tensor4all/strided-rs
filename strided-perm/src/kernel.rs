//! Block-based iteration engine for strided permutation operations.

use crate::fuse::{compress_dims, fuse_dims};
use crate::{block, order};
use strided_view::Result;

pub struct KernelPlan {
    pub block: Vec<usize>,
}

/// Build an execution plan with dimension fusion.
///
/// Pipeline: order -> reorder -> fuse -> block.
pub fn build_plan_fused(
    dims: &[usize],
    strides_list: &[&[isize]],
    dest_index: Option<usize>,
    elem_size: usize,
) -> (Vec<usize>, Vec<Vec<isize>>, KernelPlan) {
    let order = order::compute_order(dims, strides_list, dest_index);

    let ordered_dims: Vec<usize> = order.iter().map(|&d| dims[d]).collect();
    let ordered_strides: Vec<Vec<isize>> = strides_list
        .iter()
        .map(|strides| order.iter().map(|&d| strides[d]).collect())
        .collect();
    let ordered_strides_refs: Vec<&[isize]> =
        ordered_strides.iter().map(|s| s.as_slice()).collect();

    let fused_dims = fuse_dims(&ordered_dims, &ordered_strides_refs);
    let (compressed_dims, compressed_strides) = compress_dims(&fused_dims, &ordered_strides);
    let compressed_strides_refs: Vec<&[isize]> =
        compressed_strides.iter().map(|s| s.as_slice()).collect();

    let block = block::compute_block_sizes(&compressed_dims, &compressed_strides_refs, elem_size);

    (compressed_dims, compressed_strides, KernelPlan { block })
}

/// Simplified plan for small tensors that fit in L1 cache.
pub fn build_plan_fused_small(
    dims: &[usize],
    strides_list: &[&[isize]],
) -> (Vec<usize>, Vec<Vec<isize>>, KernelPlan) {
    let strides_owned: Vec<Vec<isize>> = strides_list.iter().map(|s| s.to_vec()).collect();

    let fused = fuse_dims(dims, strides_list);
    let (fused_dims, fused_strides) = compress_dims(&fused, &strides_owned);

    let block = fused_dims.clone();

    (fused_dims, fused_strides, KernelPlan { block })
}

// ============================================================================
// Macro-generated rank-specialized kernels
// ============================================================================

macro_rules! elem_loops {
    ($offsets:ident, $strides:ident, $f:ident, $blens:ident, $is:ident; $lv:literal) => {
        for _ in 0..$blens[$lv] {
            $f($offsets, $blens[0], &$is)?;
            for (o, s) in $offsets.iter_mut().zip($strides.iter()) {
                *o += s[$lv];
            }
        }
    };
    ($offsets:ident, $strides:ident, $f:ident, $blens:ident, $is:ident;
     $lv:literal, $next:literal $(, $rest:literal)*) => {
        for _ in 0..$blens[$lv] {
            elem_loops!($offsets, $strides, $f, $blens, $is; $next $(, $rest)*);
            for (o, s) in $offsets.iter_mut().zip($strides.iter()) {
                *o -= ($blens[$next] as isize) * s[$next];
                *o += s[$lv];
            }
        }
    };
}

macro_rules! block_loop {
    ($dims:ident, $blocks:ident, $strides:ident, $offsets:ident, $f:ident,
     $blens:ident, $is:ident; elem=[$($el:literal),+]; $lv0:literal; top=$top:literal) => {{
        let mut _j = 0usize;
        while _j < $dims[$lv0] {
            $blens[$lv0] = $blocks[$lv0].max(1).min($dims[$lv0]).min($dims[$lv0] - _j);
            elem_loops!($offsets, $strides, $f, $blens, $is; $($el),+);
            for (o, s) in $offsets.iter_mut().zip($strides.iter()) {
                *o -= ($blens[$top] as isize) * s[$top];
                *o += ($blens[$lv0] as isize) * s[$lv0];
            }
            _j += $blens[$lv0];
        }
    }};
    ($dims:ident, $blocks:ident, $strides:ident, $offsets:ident, $f:ident,
     $blens:ident, $is:ident; elem=[$($el:literal),+];
     $lv:literal, $next:literal $(, $rest:literal)*; top=$top:literal) => {{
        let mut _j = 0usize;
        while _j < $dims[$lv] {
            $blens[$lv] = $blocks[$lv].max(1).min($dims[$lv]).min($dims[$lv] - _j);
            block_loop!($dims, $blocks, $strides, $offsets, $f, $blens, $is;
                elem=[$($el),+]; $next $(, $rest)*; top=$top);
            for (o, s) in $offsets.iter_mut().zip($strides.iter()) {
                *o -= ($dims[$next] as isize) * s[$next];
                *o += ($blens[$lv] as isize) * s[$lv];
            }
            _j += $blens[$lv];
        }
    }};
}

macro_rules! make_kernel {
    ($name:ident, rank=1) => {
        #[inline]
        fn $name<F>(
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
            let inner_strides: Vec<isize> = strides.iter().map(|s| s[0]).collect();
            let mut j0 = 0usize;
            while j0 < d0 {
                let blen0 = b0.min(d0 - j0);
                f(offsets, blen0, &inner_strides)?;
                for (o, s) in offsets.iter_mut().zip(strides.iter()) {
                    *o += (blen0 as isize) * s[0];
                }
                j0 += blen0;
            }
            for (o, s) in offsets.iter_mut().zip(strides.iter()) {
                *o -= (d0 as isize) * s[0];
            }
            Ok(())
        }
    };
    ($name:ident, rank=$rank:literal,
     block=[$($blk:literal),+], elem=[$($el:literal),+], top=$top:literal) => {
        #[inline]
        fn $name<F>(
            dims: &[usize],
            blocks: &[usize],
            strides: &[Vec<isize>],
            offsets: &mut [isize],
            f: &mut F,
        ) -> Result<()>
        where
            F: FnMut(&[isize], usize, &[isize]) -> Result<()>,
        {
            let inner_strides: Vec<isize> = strides.iter().map(|s| s[0]).collect();
            let mut blens = [0usize; $rank];
            block_loop!(dims, blocks, strides, offsets, f, blens, inner_strides;
                elem=[$($el),+]; $($blk),+; top=$top);
            for (o, s) in offsets.iter_mut().zip(strides.iter()) {
                *o -= (dims[$top] as isize) * s[$top];
            }
            Ok(())
        }
    };
}

make_kernel!(kernel_1d_inner, rank = 1);
make_kernel!(
    kernel_2d_inner,
    rank = 2,
    block = [1, 0],
    elem = [1],
    top = 1
);
make_kernel!(
    kernel_3d_inner,
    rank = 3,
    block = [2, 1, 0],
    elem = [2, 1],
    top = 2
);
make_kernel!(
    kernel_4d_inner,
    rank = 4,
    block = [3, 2, 1, 0],
    elem = [3, 2, 1],
    top = 3
);
make_kernel!(
    kernel_5d_inner,
    rank = 5,
    block = [4, 3, 2, 1, 0],
    elem = [4, 3, 2, 1],
    top = 4
);
make_kernel!(
    kernel_6d_inner,
    rank = 6,
    block = [5, 4, 3, 2, 1, 0],
    elem = [5, 4, 3, 2, 1],
    top = 5
);
make_kernel!(
    kernel_7d_inner,
    rank = 7,
    block = [6, 5, 4, 3, 2, 1, 0],
    elem = [6, 5, 4, 3, 2, 1],
    top = 6
);
make_kernel!(
    kernel_8d_inner,
    rank = 8,
    block = [7, 6, 5, 4, 3, 2, 1, 0],
    elem = [7, 6, 5, 4, 3, 2, 1],
    top = 7
);

/// N-dimensional kernel (iterative form, fallback for rank >= 9).
#[inline]
fn kernel_nd_inner_iterative<F>(
    dims: &[usize],
    blocks: &[usize],
    strides: &[Vec<isize>],
    offsets: &mut [isize],
    f: &mut F,
) -> Result<()>
where
    F: FnMut(&[isize], usize, &[isize]) -> Result<()>,
{
    let rank = dims.len();
    debug_assert!(rank >= 9);

    let d0 = dims[0];
    let b0 = blocks[0].max(1).min(d0);
    let inner_strides: Vec<isize> = strides.iter().map(|s| s[0]).collect();

    let mut idx = vec![0usize; rank];

    loop {
        let mut j0 = 0usize;
        while j0 < d0 {
            let blen0 = b0.min(d0 - j0);
            f(offsets, blen0, &inner_strides)?;
            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                *offset += (blen0 as isize) * s[0];
            }
            j0 += blen0;
        }
        for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
            *offset -= (d0 as isize) * s[0];
        }

        let mut level = 1usize;
        loop {
            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                *offset += s[level];
            }
            idx[level] += 1;
            if idx[level] < dims[level] {
                break;
            }

            idx[level] = 0;
            for (offset, s) in offsets.iter_mut().zip(strides.iter()) {
                *offset -= (dims[level] as isize) * s[level];
            }
            level += 1;
            if level == rank {
                return Ok(());
            }
        }
    }
}

/// Iterate over blocks with pre-ordered dimensions and initial offsets.
///
/// # Errors
///
/// The traversal itself cannot fail: every return value is `Ok(())`, or
/// exactly the error returned by `f`, which this function propagates
/// unchanged. Non-zero ranks dispatch to the specialized 1D..8D kernels and
/// higher ranks to the generic iterator.
#[inline]
pub fn for_each_inner_block_preordered<F>(
    dims: &[usize],
    blocks: &[usize],
    strides: &[Vec<isize>],
    initial_offsets: &[isize],
    mut f: F,
) -> Result<()>
where
    F: FnMut(&[isize], usize, &[isize]) -> Result<()>,
{
    let rank = dims.len();
    if rank == 0 {
        return f(initial_offsets, 1, &[]);
    }

    let mut offsets = initial_offsets.to_vec();

    match rank {
        1 => kernel_1d_inner(dims, blocks, strides, &mut offsets, &mut f),
        2 => kernel_2d_inner(dims, blocks, strides, &mut offsets, &mut f),
        3 => kernel_3d_inner(dims, blocks, strides, &mut offsets, &mut f),
        4 => kernel_4d_inner(dims, blocks, strides, &mut offsets, &mut f),
        5 => kernel_5d_inner(dims, blocks, strides, &mut offsets, &mut f),
        6 => kernel_6d_inner(dims, blocks, strides, &mut offsets, &mut f),
        7 => kernel_7d_inner(dims, blocks, strides, &mut offsets, &mut f),
        8 => kernel_8d_inner(dims, blocks, strides, &mut offsets, &mut f),
        _ => kernel_nd_inner_iterative(dims, blocks, strides, &mut offsets, &mut f),
    }
}

/// Utility: total number of elements.
#[inline]
pub fn total_len(dims: &[usize]) -> usize {
    if dims.is_empty() {
        return 1;
    }
    dims.iter().product()
}

#[cfg(test)]
mod tests;
