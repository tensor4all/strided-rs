//! Kernel iteration engine ported from Julia's Strided.jl/src/mapreduce.jl
//!
//! This module implements the core iteration engine that follows Julia's
//! `_mapreduce_kernel!` pattern for cache-optimized strided array operations.

use crate::fuse::{compress_dims, fuse_dims};
use crate::{block, order, Result};

pub(crate) struct KernelPlan {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) order: Vec<usize>, // outer -> inner
    pub(crate) block: Vec<usize>,
}

/// Build an execution plan for strided iteration (used only in tests).
///
/// This follows Julia's `_mapreduce_fuse!` -> `_mapreduce_order!` -> `_mapreduce_block!` pipeline:
/// 1. Fuse contiguous dimensions
/// 2. Compute optimal iteration order
/// 3. Compute block sizes for cache efficiency
#[cfg(test)]
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
/// Pipeline: order -> reorder -> fuse -> block.
///
/// Ordering first ensures that dimensions are sorted by stride importance
/// (smallest stride innermost). Fusing *after* ordering catches contiguous
/// dimensions regardless of the original memory layout (column-major,
/// row-major, or any permutation).
///
/// Returns `(fused_dims, ordered_strides, KernelPlan)` where dimensions
/// and strides are already in iteration order (plan.order is identity).
pub(crate) fn build_plan_fused(
    dims: &[usize],
    strides_list: &[&[isize]],
    dest_index: Option<usize>,
    elem_size: usize,
) -> (Vec<usize>, Vec<Vec<isize>>, KernelPlan) {
    // 1. Compute optimal iteration order on original dims
    let order = order::compute_order(dims, strides_list, dest_index);

    // 2. Reorder dims and strides
    let ordered_dims: Vec<usize> = order.iter().map(|&d| dims[d]).collect();
    let ordered_strides: Vec<Vec<isize>> = strides_list
        .iter()
        .map(|strides| order.iter().map(|&d| strides[d]).collect())
        .collect();
    let ordered_strides_refs: Vec<&[isize]> =
        ordered_strides.iter().map(|s| s.as_slice()).collect();

    // 3. Fuse contiguous dimensions in ordered space
    let fused_dims = fuse_dims(&ordered_dims, &ordered_strides_refs);

    // 4. Compress: remove size-1 dimensions to reduce loop depth
    let (compressed_dims, compressed_strides) = compress_dims(&fused_dims, &ordered_strides);
    let compressed_strides_refs: Vec<&[isize]> =
        compressed_strides.iter().map(|s| s.as_slice()).collect();

    // 5. Compute blocks with identity ordering (already ordered)
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

// ============================================================================
// Block-based iteration with inner stride callback
// ============================================================================

/// Iterate over blocks, calling f with (offsets, block_len, inner_strides).
///
/// The callback receives the current byte offsets for each array, the number
/// of elements in the innermost block, and the innermost strides for each array.
/// This allows the caller to implement vectorized inner loops.
#[cfg(test)]
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
        5 => kernel_5d_inner(
            &ordered_dims,
            &ordered_blocks,
            &ordered_strides,
            &mut offsets,
            &mut f,
        ),
        6 => kernel_6d_inner(
            &ordered_dims,
            &ordered_blocks,
            &ordered_strides,
            &mut offsets,
            &mut f,
        ),
        7 => kernel_7d_inner(
            &ordered_dims,
            &ordered_blocks,
            &ordered_strides,
            &mut offsets,
            &mut f,
        ),
        8 => kernel_8d_inner(
            &ordered_dims,
            &ordered_blocks,
            &ordered_strides,
            &mut offsets,
            &mut f,
        ),
        _ => kernel_nd_inner_iterative(
            &ordered_dims,
            &ordered_blocks,
            &ordered_strides,
            &mut offsets,
            &mut f,
        ),
    }
}

// ============================================================================
// Macro-generated rank-specialized kernels (inner-block callback)
// ============================================================================

/// Element-level nested loops for ranks >= 2.
///
/// Iterates over the element block, calling `f` at each innermost position.
/// Levels are listed from outermost to innermost (excluding level 0 which is
/// the callback level). For a rank-N kernel, element levels are N-1, N-2, ..., 1.
macro_rules! elem_loops {
    // Base case: single level (innermost element level).
    // Iterates blens[$lv] times, calling f then advancing by stride[$lv].
    ($offsets:ident, $strides:ident, $f:ident, $blens:ident, $is:ident; $lv:literal) => {
        for _ in 0..$blens[$lv] {
            $f($offsets, $blens[0], &$is)?;
            for (o, s) in $offsets.iter_mut().zip($strides.iter()) {
                *o += s[$lv];
            }
        }
    };
    // Recursive case: outermost level wraps inner levels.
    // After the inner loop, resets the inner level's offset and advances this level.
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

/// Block-level nested while loops for ranks >= 2.
///
/// Each block level iterates over tiles of the corresponding dimension.
/// At level 0 (innermost), element loops are executed. After each inner body,
/// offsets are adjusted to reset the inner dimension and advance the current level.
macro_rules! block_loop {
    // Base case: single block level (the innermost).
    // Runs element loops, then resets top element level and advances this block level.
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
    // Recursive case: outer block level wraps inner block levels.
    // After the inner body, resets the next-inner dimension and advances this level.
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

/// Generate a rank-specialized kernel function.
///
/// For rank 1 there are no element loops; only a single block loop on dim 0.
/// For rank >= 2, block loops nest from outermost to innermost (dim 0), and
/// element loops nest from dim N-1 down to dim 1 inside the innermost block.
macro_rules! make_kernel {
    // Rank 1: single block loop, no element nesting.
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
    // Rank >= 2: nested block loops + element loops.
    //   block=[outermost, ..., 0]  - block loop levels from outermost to innermost
    //   elem=[N-1, ..., 1]         - element loop levels from outermost to innermost
    //   top=N-1                    - topmost element level (= rank - 1)
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

make_kernel!(kernel_1d_inner, rank=1);
make_kernel!(kernel_2d_inner, rank=2, block=[1, 0], elem=[1], top=1);
make_kernel!(kernel_3d_inner, rank=3, block=[2, 1, 0], elem=[2, 1], top=2);
make_kernel!(kernel_4d_inner, rank=4, block=[3, 2, 1, 0], elem=[3, 2, 1], top=3);
make_kernel!(kernel_5d_inner, rank=5, block=[4, 3, 2, 1, 0], elem=[4, 3, 2, 1], top=4);
make_kernel!(kernel_6d_inner, rank=6, block=[5, 4, 3, 2, 1, 0], elem=[5, 4, 3, 2, 1], top=5);
make_kernel!(kernel_7d_inner, rank=7, block=[6, 5, 4, 3, 2, 1, 0], elem=[6, 5, 4, 3, 2, 1], top=6);
make_kernel!(kernel_8d_inner, rank=8, block=[7, 6, 5, 4, 3, 2, 1, 0], elem=[7, 6, 5, 4, 3, 2, 1], top=7);

/// N-dimensional kernel with inner block callback (iterative form).
///
/// Fallback for rank >= 9. Uses a carry-style increment over outer levels
/// instead of compile-time unrolled nested loops.
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

    // Current position for each outer level (1..rank-1). Level 0 uses block loop.
    let mut idx = vec![0usize; rank];

    loop {
        // Level 0: callback over contiguous block fragments.
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

        // Carry-style increment for outer levels.
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

// ============================================================================
// Pre-ordered iteration (for threaded leaf functions)
// ============================================================================

/// Iterate over blocks with pre-ordered dimensions and initial offsets.
///
/// Unlike `for_each_inner_block`, this function assumes that `dims`, `blocks`,
/// and `strides` are **already in iteration order** (i.e., identity ordering).
/// It also accepts `initial_offsets` which are added to the starting offsets
/// before iteration begins.
///
/// This avoids the redundant re-ordering and per-callback `Vec` allocation
/// that `for_each_inner_block_with_offsets` previously incurred.
#[inline]
pub(crate) fn for_each_inner_block_preordered<F>(
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

    // Start from initial_offsets (kernel functions reset to starting values at end)
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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum ContiguousLayout {
    /// C-like layout: last axis varies fastest.
    RowMajor,
    /// Julia/Fortran-like layout: first axis varies fastest.
    ColMajor,
}

/// Returns the contiguous memory layout kind for the given (dims, strides).
///
/// Notes:
/// - Ignores axes with `dim <= 1` since they do not affect addressability.
/// - Does not treat negative-stride views as contiguous for fast-path purposes.
pub(crate) fn contiguous_layout(dims: &[usize], strides: &[isize]) -> Option<ContiguousLayout> {
    if dims.len() != strides.len() {
        return None;
    }
    if dims.is_empty() {
        return Some(ContiguousLayout::RowMajor);
    }

    // Row-major: check from last to first.
    let mut expected = 1isize;
    let mut row_ok = true;
    for (&dim, &stride) in dims.iter().rev().zip(strides.iter().rev()) {
        if dim <= 1 {
            continue;
        }
        if stride != expected {
            row_ok = false;
            break;
        }
        expected = expected.saturating_mul(dim as isize);
    }
    if row_ok {
        return Some(ContiguousLayout::RowMajor);
    }

    // Col-major: check from first to last.
    let mut expected = 1isize;
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        if dim <= 1 {
            continue;
        }
        if stride != expected {
            return None;
        }
        expected = expected.saturating_mul(dim as isize);
    }
    Some(ContiguousLayout::ColMajor)
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
/// fast path -- everything flows through fuse -> order -> block -> threaded ->
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

/// Returns the common contiguous layout if **all** provided stride arrays
/// share the same contiguous layout for the given `dims`.
///
/// Returns `None` if `strides_list` is empty, any array is not contiguous,
/// or any two arrays have different contiguous layouts.
#[inline]
pub(crate) fn same_contiguous_layout(
    dims: &[usize],
    strides_list: &[&[isize]],
) -> Option<ContiguousLayout> {
    let first = contiguous_layout(dims, strides_list.first()?)?;
    for strides in &strides_list[1..] {
        if contiguous_layout(dims, strides)? != first {
            return None;
        }
    }
    Some(first)
}

/// Returns the common contiguous layout only when the sequential fast path
/// should be used (total elements <= threading threshold).
#[inline]
pub(crate) fn sequential_contiguous_layout(
    dims: &[usize],
    strides_list: &[&[isize]],
) -> Option<ContiguousLayout> {
    if !use_sequential_fast_path(total_len(dims)) {
        return None;
    }
    same_contiguous_layout(dims, strides_list)
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

    #[test]
    fn test_contiguous_layout_row_vs_col() {
        let dims = [3usize, 4];
        let row = [4isize, 1];
        let col = [1isize, 3];
        assert_eq!(
            contiguous_layout(&dims, &row),
            Some(ContiguousLayout::RowMajor)
        );
        assert_eq!(
            contiguous_layout(&dims, &col),
            Some(ContiguousLayout::ColMajor)
        );
        assert!(contiguous_layout(&dims, &row).is_some());
        assert!(contiguous_layout(&dims, &col).is_some());
    }

    #[test]
    fn test_contiguous_layout_ignores_dim1_axes() {
        let dims = [2usize, 1, 3];
        // Middle stride is irrelevant since that axis never varies.
        let strides = [3isize, 999, 1];
        assert_eq!(
            contiguous_layout(&dims, &strides),
            Some(ContiguousLayout::RowMajor)
        );
    }

    // ---- same_contiguous_layout tests ----

    #[test]
    fn test_same_contiguous_layout_all_row_major() {
        let dims = [3usize, 4];
        let s1 = [4isize, 1];
        let s2 = [4isize, 1];
        assert_eq!(
            same_contiguous_layout(&dims, &[&s1, &s2]),
            Some(ContiguousLayout::RowMajor)
        );
    }

    #[test]
    fn test_same_contiguous_layout_all_col_major() {
        let dims = [3usize, 4];
        let s1 = [1isize, 3];
        let s2 = [1isize, 3];
        assert_eq!(
            same_contiguous_layout(&dims, &[&s1, &s2]),
            Some(ContiguousLayout::ColMajor)
        );
    }

    #[test]
    fn test_same_contiguous_layout_mixed_layouts() {
        let dims = [3usize, 4];
        let row = [4isize, 1];
        let col = [1isize, 3];
        assert_eq!(same_contiguous_layout(&dims, &[&row, &col]), None);
    }

    #[test]
    fn test_same_contiguous_layout_one_noncontiguous() {
        let dims = [3usize, 4];
        let row = [4isize, 1];
        let bad = [8isize, 2];
        assert_eq!(same_contiguous_layout(&dims, &[&row, &bad]), None);
    }

    #[test]
    fn test_same_contiguous_layout_empty_strides_list() {
        let dims = [3usize, 4];
        let empty: &[&[isize]] = &[];
        assert_eq!(same_contiguous_layout(&dims, empty), None);
    }

    #[test]
    fn test_same_contiguous_layout_single_array() {
        let dims = [3usize, 4];
        let s = [4isize, 1];
        assert_eq!(
            same_contiguous_layout(&dims, &[&s[..]]),
            Some(ContiguousLayout::RowMajor)
        );
    }

    #[test]
    fn test_same_contiguous_layout_many_arrays() {
        let dims = [2usize, 3];
        let s = [3isize, 1];
        assert_eq!(
            same_contiguous_layout(&dims, &[&s[..], &s[..], &s[..], &s[..], &s[..]]),
            Some(ContiguousLayout::RowMajor)
        );
    }

    #[test]
    fn test_same_contiguous_layout_empty_dims() {
        let dims: [usize; 0] = [];
        let s: [isize; 0] = [];
        assert_eq!(
            same_contiguous_layout(&dims, &[&s[..], &s[..]]),
            Some(ContiguousLayout::RowMajor)
        );
    }

    // ---- sequential_contiguous_layout tests ----

    #[test]
    fn test_sequential_contiguous_layout_small_array() {
        let dims = [3usize, 4];
        let s1 = [4isize, 1];
        let s2 = [4isize, 1];
        assert_eq!(
            sequential_contiguous_layout(&dims, &[&s1, &s2]),
            Some(ContiguousLayout::RowMajor)
        );
    }

    #[test]
    fn test_sequential_contiguous_layout_noncontiguous() {
        let dims = [3usize, 4];
        let s1 = [4isize, 1];
        let s2 = [8isize, 2];
        assert_eq!(sequential_contiguous_layout(&dims, &[&s1, &s2]), None);
    }

    #[test]
    fn test_sequential_contiguous_layout_col_major() {
        let dims = [3usize, 4];
        let col = [1isize, 3];
        assert_eq!(
            sequential_contiguous_layout(&dims, &[&col]),
            Some(ContiguousLayout::ColMajor)
        );
    }

    #[test]
    fn test_build_plan_fused_compresses() {
        // A contiguous 2x3 column-major array fuses [2,3] -> [6,1] -> compress -> [6]
        let dims = [2usize, 3];
        let strides = [1isize, 2];
        let strides_list: Vec<&[isize]> = vec![&strides];
        let (fused_dims, fused_strides, plan) = build_plan_fused(&dims, &strides_list, Some(0), 8);
        // After fusion + compression, should be 1D
        assert_eq!(fused_dims, vec![6]);
        assert_eq!(fused_strides.len(), 1);
        assert_eq!(fused_strides[0], vec![1]);
        assert_eq!(plan.block.len(), 1);
    }

    #[test]
    fn test_kernel_nd_iterative_total_elements_match() {
        // Use rank 9 to test kernel_nd_inner_iterative (fallback for rank >= 9)
        let dims = vec![2usize, 2, 2, 2, 2, 2, 2, 2, 3];
        let blocks = vec![2usize, 1, 1, 1, 1, 1, 1, 1, 1];
        let mut stride_val = 1isize;
        let mut sv = Vec::new();
        for &d in &dims {
            sv.push(stride_val);
            stride_val *= d as isize;
        }
        let strides = vec![sv.clone(), sv];
        let mut offsets = vec![0isize, 0isize];
        let mut total = 0usize;

        kernel_nd_inner_iterative(
            &dims,
            &blocks,
            &strides,
            &mut offsets,
            &mut |_off, len, _s| {
                total += len;
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(total, dims.iter().product::<usize>());
        assert_eq!(offsets, vec![0isize, 0isize]);
    }

    /// Helper: run a macro-generated kernel and verify total elements and offset reset.
    fn verify_kernel_total_and_offsets(rank: usize) {
        // Build dims: dim[0] = 3 for block variety, rest = 2
        let mut dims = vec![3usize];
        for _ in 1..rank {
            dims.push(2);
        }
        let blocks = vec![2usize; rank];

        // Column-major strides
        let mut stride_val = 1isize;
        let mut sv = Vec::new();
        for &d in &dims {
            sv.push(stride_val);
            stride_val *= d as isize;
        }
        let strides = vec![sv.clone(), sv];
        let mut offsets = vec![0isize, 0isize];
        let mut total = 0usize;
        let expected: usize = dims.iter().product();

        let result = match rank {
            1 => kernel_1d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
                total += l;
                Ok(())
            }),
            2 => kernel_2d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
                total += l;
                Ok(())
            }),
            3 => kernel_3d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
                total += l;
                Ok(())
            }),
            4 => kernel_4d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
                total += l;
                Ok(())
            }),
            5 => kernel_5d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
                total += l;
                Ok(())
            }),
            6 => kernel_6d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
                total += l;
                Ok(())
            }),
            7 => kernel_7d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
                total += l;
                Ok(())
            }),
            8 => kernel_8d_inner(&dims, &blocks, &strides, &mut offsets, &mut |_o, l, _s| {
                total += l;
                Ok(())
            }),
            _ => panic!("unsupported rank"),
        };
        result.unwrap();
        assert_eq!(total, expected, "rank={rank}: total mismatch");
        assert_eq!(offsets, vec![0, 0], "rank={rank}: offsets not reset");
    }

    #[test]
    fn test_macro_kernels_total_elements_1d() {
        verify_kernel_total_and_offsets(1);
    }

    #[test]
    fn test_macro_kernels_total_elements_2d() {
        verify_kernel_total_and_offsets(2);
    }

    #[test]
    fn test_macro_kernels_total_elements_3d() {
        verify_kernel_total_and_offsets(3);
    }

    #[test]
    fn test_macro_kernels_total_elements_4d() {
        verify_kernel_total_and_offsets(4);
    }

    #[test]
    fn test_macro_kernels_total_elements_5d() {
        verify_kernel_total_and_offsets(5);
    }

    #[test]
    fn test_macro_kernels_total_elements_6d() {
        verify_kernel_total_and_offsets(6);
    }

    #[test]
    fn test_macro_kernels_total_elements_7d() {
        verify_kernel_total_and_offsets(7);
    }

    #[test]
    fn test_macro_kernels_total_elements_8d() {
        verify_kernel_total_and_offsets(8);
    }

    /// Verify that macro-generated kernels visit every element exactly once
    /// by collecting all linear offsets and checking them against the expected set.
    fn verify_kernel_visits_all_elements(rank: usize) {
        assert!(rank >= 2 && rank <= 8);
        let mut dims = vec![3usize];
        for _ in 1..rank {
            dims.push(2);
        }
        let blocks = vec![2usize; rank];

        // Column-major strides (single array for simplicity)
        let mut stride_val = 1isize;
        let mut sv = Vec::new();
        for &d in &dims {
            sv.push(stride_val);
            stride_val *= d as isize;
        }
        let strides = vec![sv];

        // Collect all linear offsets visited
        let mut visited = std::collections::HashSet::new();
        let mut offsets = vec![0isize];

        let result = match rank {
            2 => kernel_2d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
                for i in 0..len {
                    visited.insert(o[0] + (i as isize) * s[0]);
                }
                Ok(())
            }),
            3 => kernel_3d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
                for i in 0..len {
                    visited.insert(o[0] + (i as isize) * s[0]);
                }
                Ok(())
            }),
            4 => kernel_4d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
                for i in 0..len {
                    visited.insert(o[0] + (i as isize) * s[0]);
                }
                Ok(())
            }),
            5 => kernel_5d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
                for i in 0..len {
                    visited.insert(o[0] + (i as isize) * s[0]);
                }
                Ok(())
            }),
            6 => kernel_6d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
                for i in 0..len {
                    visited.insert(o[0] + (i as isize) * s[0]);
                }
                Ok(())
            }),
            7 => kernel_7d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
                for i in 0..len {
                    visited.insert(o[0] + (i as isize) * s[0]);
                }
                Ok(())
            }),
            8 => kernel_8d_inner(&dims, &blocks, &strides, &mut offsets, &mut |o, len, s| {
                for i in 0..len {
                    visited.insert(o[0] + (i as isize) * s[0]);
                }
                Ok(())
            }),
            _ => unreachable!(),
        };
        result.unwrap();

        // Expected: all offsets 0..total
        let total: usize = dims.iter().product();
        let expected: std::collections::HashSet<isize> = (0..total as isize).collect();
        assert_eq!(visited, expected, "rank={rank}: not all elements visited exactly once");
    }

    #[test]
    fn test_macro_kernel_5d_visits_all_elements() {
        verify_kernel_visits_all_elements(5);
    }

    #[test]
    fn test_macro_kernel_6d_visits_all_elements() {
        verify_kernel_visits_all_elements(6);
    }

    #[test]
    fn test_macro_kernel_7d_visits_all_elements() {
        verify_kernel_visits_all_elements(7);
    }

    #[test]
    fn test_macro_kernel_8d_visits_all_elements() {
        verify_kernel_visits_all_elements(8);
    }
}
