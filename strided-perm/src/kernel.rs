//! Block-based iteration engine for strided permutation operations.

use crate::fuse::{compress_dims, fuse_dims};
use crate::{block, order};
use strided_view::Result;

/// Maximum total elements for the small tensor fast path.
pub const SMALL_TENSOR_THRESHOLD: usize = 1024;

pub struct KernelPlan {
    pub order: Vec<usize>,
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

    let identity: Vec<usize> = (0..compressed_dims.len()).collect();
    let block = block::compute_block_sizes(
        &compressed_dims,
        &identity,
        &compressed_strides_refs,
        elem_size,
    );

    (
        compressed_dims,
        compressed_strides,
        KernelPlan {
            order: identity,
            block,
        },
    )
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
    let identity: Vec<usize> = (0..fused_dims.len()).collect();

    (
        fused_dims,
        fused_strides,
        KernelPlan {
            order: identity,
            block,
        },
    )
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
mod tests {
    use super::*;

    #[test]
    fn test_build_plan_fused_compresses() {
        let dims = [2usize, 3];
        let strides = [1isize, 2];
        let strides_list: Vec<&[isize]> = vec![&strides];
        let (fused_dims, fused_strides, plan) = build_plan_fused(&dims, &strides_list, Some(0), 8);
        assert_eq!(fused_dims, vec![6]);
        assert_eq!(fused_strides.len(), 1);
        assert_eq!(fused_strides[0], vec![1]);
        assert_eq!(plan.block.len(), 1);
    }

    #[test]
    fn test_for_each_inner_block_preordered_total() {
        let dims = vec![3, 4, 2];
        let blocks = vec![3, 4, 2];
        let strides = vec![vec![1, 3, 12], vec![1, 3, 12]];
        let offsets = vec![0, 0];
        let mut total = 0usize;
        for_each_inner_block_preordered(&dims, &blocks, &strides, &offsets, |_, len, _| {
            total += len;
            Ok(())
        })
        .unwrap();
        assert_eq!(total, 24);
    }

    #[test]
    fn test_build_plan_fused_non_contiguous() {
        // Row-major [4,5]: strides [5, 1]
        let dims = [4usize, 5];
        let strides = [5isize, 1];
        let strides_list: Vec<&[isize]> = vec![&strides];
        let (fused_dims, fused_strides, plan) = build_plan_fused(&dims, &strides_list, Some(0), 8);
        // Row-major: should reorder so stride-1 is first, then fuse to 20
        assert_eq!(fused_dims, vec![20]);
        assert_eq!(fused_strides[0], vec![1]);
        assert_eq!(plan.block.len(), 1);
    }

    #[test]
    fn test_build_plan_fused_multi_array() {
        // Two arrays: one col-major, one row-major
        let dims = [4usize, 5];
        let dst_strides = [1isize, 4];
        let src_strides = [5isize, 1];
        let strides_list: Vec<&[isize]> = vec![&dst_strides, &src_strides];
        let (fused_dims, fused_strides, plan) = build_plan_fused(&dims, &strides_list, Some(0), 8);
        // Conflicting strides means no fusion possible
        assert_eq!(fused_dims.len(), 2);
        assert_eq!(fused_strides.len(), 2);
        assert!(plan.block.len() >= 1);
    }

    #[test]
    fn test_build_plan_fused_small_basic() {
        let dims = [2usize, 3];
        let strides = [1isize, 2];
        let strides_list: Vec<&[isize]> = vec![&strides];
        let (fused_dims, fused_strides, plan) = build_plan_fused_small(&dims, &strides_list);
        assert_eq!(fused_dims, vec![6]);
        assert_eq!(fused_strides[0], vec![1]);
        // Small plan: block = dims (no blocking)
        assert_eq!(plan.block, fused_dims);
    }

    #[test]
    fn test_build_plan_fused_small_non_contiguous() {
        let dims = [4usize, 5];
        let strides = [5isize, 1];
        let strides_list: Vec<&[isize]> = vec![&strides];
        let (fused_dims, fused_strides, plan) = build_plan_fused_small(&dims, &strides_list);
        // No ordering in small path, but fusion should still work on contiguous groups
        assert!(!fused_dims.is_empty());
        assert_eq!(fused_strides.len(), 1);
        assert_eq!(plan.block, fused_dims);
    }

    #[test]
    fn test_for_each_rank0() {
        let dims = vec![];
        let blocks = vec![];
        let strides: Vec<Vec<isize>> = vec![vec![]];
        let offsets = vec![0isize];
        let mut called = false;
        for_each_inner_block_preordered(&dims, &blocks, &strides, &offsets, |_, len, _| {
            called = true;
            assert_eq!(len, 1);
            Ok(())
        })
        .unwrap();
        assert!(called);
    }

    /// Helper: count total elements iterated for given rank
    fn count_elements(dims: &[usize], blocks: &[usize]) -> usize {
        let n_arrays = 1;
        let strides: Vec<Vec<isize>> = {
            let mut s = vec![vec![0isize; dims.len()]; n_arrays];
            let mut stride = 1isize;
            for d in 0..dims.len() {
                for a in 0..n_arrays {
                    s[a][d] = stride;
                }
                stride *= dims[d] as isize;
            }
            s
        };
        let offsets = vec![0isize; n_arrays];
        let mut total = 0usize;
        for_each_inner_block_preordered(dims, blocks, &strides, &offsets, |_, len, _| {
            total += len;
            Ok(())
        })
        .unwrap();
        total
    }

    #[test]
    fn test_for_each_rank4() {
        assert_eq!(count_elements(&[2, 3, 4, 5], &[2, 3, 4, 5]), 120);
    }

    #[test]
    fn test_for_each_rank5() {
        assert_eq!(count_elements(&[2, 2, 2, 2, 2], &[2, 2, 2, 2, 2]), 32);
    }

    #[test]
    fn test_for_each_rank6() {
        assert_eq!(count_elements(&[2, 2, 2, 2, 2, 3], &[2, 2, 2, 2, 2, 3]), 96);
    }

    #[test]
    fn test_for_each_rank7() {
        assert_eq!(
            count_elements(&[2, 2, 2, 2, 2, 2, 3], &[2, 2, 2, 2, 2, 2, 3]),
            192
        );
    }

    #[test]
    fn test_for_each_rank8() {
        assert_eq!(
            count_elements(&[2, 2, 2, 2, 2, 2, 2, 3], &[2, 2, 2, 2, 2, 2, 2, 3]),
            384
        );
    }

    #[test]
    fn test_for_each_rank9_iterative() {
        // Rank 9 triggers kernel_nd_inner_iterative
        assert_eq!(
            count_elements(&[2, 2, 2, 2, 2, 2, 2, 2, 3], &[2, 2, 2, 2, 2, 2, 2, 2, 3]),
            768
        );
    }

    #[test]
    fn test_for_each_rank10_iterative() {
        assert_eq!(
            count_elements(
                &[2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
                &[2, 2, 2, 2, 2, 2, 2, 2, 2, 3]
            ),
            1536
        );
    }

    #[test]
    fn test_total_len_empty() {
        assert_eq!(total_len(&[]), 1);
    }

    #[test]
    fn test_total_len_basic() {
        assert_eq!(total_len(&[2, 3, 4]), 24);
    }
}
