//! Threading support ported from Julia's Strided.jl
//!
//! This module implements the divide-and-conquer threading algorithm from
//! Julia's `_mapreduce_threaded!` (mapreduce.jl:195-227).
//!
//! Key features:
//! - Recursive dimension splitting using cost-weighted selection
//! - False-sharing avoidance for parallel reductions
//! - Integration with rayon for work-stealing parallelism

use crate::block::compute_block_sizes;
use crate::fuse::compute_costs;
use crate::order::compute_order;
use crate::MIN_THREAD_LENGTH;

#[cfg(feature = "parallel")]

/// Context for threaded mapreduce operations.
///
/// This struct holds all the information needed for recursive divide-and-conquer
/// parallelization, matching Julia's `_mapreduce_threaded!` signature.
#[derive(Clone)]
pub struct ThreadedContext<'a> {
    /// Dimensions of the array region to process
    pub dims: Vec<usize>,
    /// Block sizes for cache-efficient iteration
    pub blocks: Vec<usize>,
    /// Strides for each array (one inner slice per array)
    pub strides: Vec<&'a [isize]>,
    /// Current offsets into each array
    pub offsets: Vec<isize>,
    /// Cost weights for dimension selection
    pub costs: Vec<isize>,
    /// Number of threads available for this subtask
    pub nthreads: usize,
    /// Spacing for false-sharing avoidance (reduction only)
    pub spacing: usize,
    /// Task index for computing output offset
    pub task_index: usize,
}

impl<'a> ThreadedContext<'a> {
    /// Create a new threaded context.
    pub fn new(
        dims: Vec<usize>,
        strides: Vec<&'a [isize]>,
        offsets: Vec<isize>,
        elem_size: usize,
    ) -> Self {
        let strides_refs: Vec<&[isize]> = strides.iter().copied().collect();

        // Compute dimension ordering
        let order = compute_order(&dims, &strides_refs, Some(0));

        // Compute block sizes
        let blocks = compute_block_sizes(&dims, &order, &strides_refs, elem_size);

        // Compute costs
        let costs = compute_costs(&strides_refs);

        #[cfg(feature = "parallel")]
        let nthreads = rayon::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let nthreads = 1;

        Self {
            dims,
            blocks,
            strides,
            offsets,
            costs,
            nthreads,
            spacing: 0,
            task_index: 1,
        }
    }

    /// Total number of elements in the current region.
    pub fn total_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Check if we should run sequentially.
    ///
    /// Julia: `nthreads == 1 || prod(dims) <= MINTHREADLENGTH`
    pub fn should_run_sequential(&self) -> bool {
        self.nthreads <= 1 || self.total_elements() <= MIN_THREAD_LENGTH
    }
}

/// Find the best dimension to split for threading.
///
/// Julia equivalent: `i = _lastargmax((dims .- 1) .* costs)`
///
/// Returns the index of the dimension with the highest (dims - 1) * cost score,
/// preferring later indices on ties (last argmax).
///
/// Returns None if no dimension can be split (all dims <= 1 or costs are 0).
pub fn find_split_dimension(dims: &[usize], costs: &[isize]) -> Option<usize> {
    if dims.is_empty() {
        return None;
    }

    let mut max_score = 0isize;
    let mut max_idx = None;

    for (i, (&d, &c)) in dims.iter().zip(costs.iter()).enumerate() {
        if d <= 1 {
            continue;
        }
        let score = (d as isize - 1) * c;
        // Use >= for "last argmax" behavior
        if score >= max_score {
            max_score = score;
            max_idx = Some(i);
        }
    }

    max_idx
}

/// Check if a dimension should be split.
///
/// Julia: `costs[i] == 0 || dims[i] <= min(blocks[i], 1024)`
pub fn should_split_dimension(
    dim_idx: usize,
    dims: &[usize],
    blocks: &[usize],
    costs: &[isize],
) -> bool {
    let cost = costs.get(dim_idx).copied().unwrap_or(0);
    let dim = dims.get(dim_idx).copied().unwrap_or(0);
    let block = blocks.get(dim_idx).copied().unwrap_or(0);

    // Don't split if cost is 0 (would cause race conditions for reductions)
    // or if dimension is already small enough
    cost != 0 && dim > block.min(1024)
}

/// Split context into two halves for parallel execution.
///
/// Julia equivalent:
/// ```julia
/// di = dims[i]
/// ndi = di >> 1
/// nnthreads = nthreads >> 1
/// newdims = setindex(dims, ndi, i)
/// stridesi = getindex.(strides, i)
/// newoffsets2 = offsets .+ ndi .* stridesi
/// newdims2 = setindex(dims, di - ndi, i)
/// ```
pub fn split_context<'a>(
    ctx: &ThreadedContext<'a>,
    split_dim: usize,
) -> (ThreadedContext<'a>, ThreadedContext<'a>) {
    let di = ctx.dims[split_dim];
    let ndi = di >> 1; // First half size
    let ndi2 = di - ndi; // Second half size (handles odd dimensions)

    let nnthreads = ctx.nthreads >> 1;
    let nnthreads2 = ctx.nthreads - nnthreads;

    // First context: first half of split dimension
    let mut dims1 = ctx.dims.clone();
    dims1[split_dim] = ndi;

    let ctx1 = ThreadedContext {
        dims: dims1,
        blocks: ctx.blocks.clone(),
        strides: ctx.strides.clone(),
        offsets: ctx.offsets.clone(),
        costs: ctx.costs.clone(),
        nthreads: nnthreads,
        spacing: ctx.spacing,
        task_index: ctx.task_index,
    };

    // Second context: second half of split dimension
    let mut dims2 = ctx.dims.clone();
    dims2[split_dim] = ndi2;

    // Compute new offsets: offsets + ndi * strides[split_dim]
    let mut offsets2 = ctx.offsets.clone();
    for (offset, strides) in offsets2.iter_mut().zip(ctx.strides.iter()) {
        let stride_at_split = strides[split_dim];
        *offset += ndi as isize * stride_at_split;
    }

    let ctx2 = ThreadedContext {
        dims: dims2,
        blocks: ctx.blocks.clone(),
        strides: ctx.strides.clone(),
        offsets: offsets2,
        costs: ctx.costs.clone(),
        nthreads: nnthreads2,
        spacing: ctx.spacing,
        task_index: ctx.task_index + nnthreads,
    };

    (ctx1, ctx2)
}

/// Mask costs for dimensions with zero stride in output array.
///
/// Julia: `costs = costs .* .!(iszero.(strides[1]))`
///
/// This prevents splitting dimensions that would cause race conditions
/// when the output has zero stride (reduction dimensions).
pub fn mask_reduction_costs(costs: &[isize], output_strides: &[isize]) -> Vec<isize> {
    costs
        .iter()
        .zip(output_strides.iter())
        .map(|(&c, &s)| if s == 0 { 0 } else { c })
        .collect()
}

/// Compute spacing for false-sharing avoidance in parallel reductions.
///
/// Julia: `spacing = isbitstype(T) ? max(1, div(64, sizeof(T))) : 1`
///
/// Returns the number of elements to space out partial results to avoid
/// cache line contention between threads.
#[allow(dead_code)] // Will be used for parallel reductions
pub fn compute_reduction_spacing(elem_size: usize) -> usize {
    if elem_size == 0 {
        return 1;
    }
    (crate::CACHE_LINE_SIZE / elem_size).max(1)
}

// Generic threaded_map is not used directly; par_zip_map2_into uses its own
// recursive implementation for type-safety with raw pointers.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_split_dimension_basic() {
        let dims = [10usize, 20, 5];
        let costs = [1isize, 1, 1];

        let idx = find_split_dimension(&dims, &costs);

        // (10-1)*1=9, (20-1)*1=19, (5-1)*1=4
        // Max is 19 at index 1
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn test_find_split_dimension_with_costs() {
        let dims = [10usize, 5];
        let costs = [1isize, 10];

        let idx = find_split_dimension(&dims, &costs);

        // (10-1)*1=9, (5-1)*10=40
        // Max is 40 at index 1
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn test_find_split_dimension_last_argmax() {
        // Tied scores should return last index
        let dims = [10usize, 10];
        let costs = [1isize, 1];

        let idx = find_split_dimension(&dims, &costs);

        // Both have score 9, should return 1 (last)
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn test_find_split_dimension_all_ones() {
        let dims = [1usize, 1, 1];
        let costs = [1isize, 1, 1];

        let idx = find_split_dimension(&dims, &costs);

        // No dimension can be split
        assert_eq!(idx, None);
    }

    #[test]
    fn test_mask_reduction_costs() {
        let costs = [2isize, 4, 6];
        let strides = [1isize, 0, 2]; // Zero stride at dim 1 (reduction)

        let masked = mask_reduction_costs(&costs, &strides);

        assert_eq!(masked, vec![2, 0, 6]); // Cost at dim 1 masked to 0
    }

    #[test]
    fn test_compute_reduction_spacing() {
        // f64: 8 bytes -> spacing = 64/8 = 8
        assert_eq!(compute_reduction_spacing(8), 8);

        // f32: 4 bytes -> spacing = 64/4 = 16
        assert_eq!(compute_reduction_spacing(4), 16);

        // u8: 1 byte -> spacing = 64/1 = 64
        assert_eq!(compute_reduction_spacing(1), 64);
    }

    #[test]
    fn test_split_context() {
        let strides1: Vec<isize> = vec![1, 10];
        let strides2: Vec<isize> = vec![10, 1];
        let strides: Vec<&[isize]> = vec![&strides1, &strides2];

        let ctx = ThreadedContext {
            dims: vec![100, 100],
            blocks: vec![50, 50],
            strides: strides.clone(),
            offsets: vec![0, 0],
            costs: vec![2, 2],
            nthreads: 4,
            spacing: 0,
            task_index: 1,
        };

        let (ctx1, ctx2) = split_context(&ctx, 0);

        // First context: first half of dim 0
        assert_eq!(ctx1.dims, vec![50, 100]);
        assert_eq!(ctx1.offsets, vec![0, 0]);
        assert_eq!(ctx1.nthreads, 2);
        assert_eq!(ctx1.task_index, 1);

        // Second context: second half of dim 0
        assert_eq!(ctx2.dims, vec![50, 100]);
        // Offset = 0 + 50 * stride[0] = 50 * 1 = 50 for first array
        // Offset = 0 + 50 * stride[0] = 50 * 10 = 500 for second array
        assert_eq!(ctx2.offsets, vec![50, 500]);
        assert_eq!(ctx2.nthreads, 2);
        assert_eq!(ctx2.task_index, 3);
    }

    #[test]
    fn test_split_context_odd_dimension() {
        let strides: Vec<isize> = vec![1, 10];
        let strides_ref: Vec<&[isize]> = vec![&strides];

        let ctx = ThreadedContext {
            dims: vec![101, 100], // Odd first dimension
            blocks: vec![50, 50],
            strides: strides_ref,
            offsets: vec![0],
            costs: vec![2, 2],
            nthreads: 4,
            spacing: 0,
            task_index: 1,
        };

        let (ctx1, ctx2) = split_context(&ctx, 0);

        // First half: 50, second half: 51
        assert_eq!(ctx1.dims, vec![50, 100]);
        assert_eq!(ctx2.dims, vec![51, 100]);
    }

    #[test]
    fn test_should_split_dimension() {
        let dims = [100usize, 10];
        let blocks = [50usize, 50];
        let costs = [2isize, 2];

        // Dim 0: 100 > min(50, 1024) = 50, cost != 0 -> should split
        assert!(should_split_dimension(0, &dims, &blocks, &costs));

        // Dim 1: 10 <= min(50, 1024) = 50 -> should not split
        assert!(!should_split_dimension(1, &dims, &blocks, &costs));
    }

    #[test]
    fn test_should_split_dimension_zero_cost() {
        let dims = [100usize];
        let blocks = [50usize];
        let costs = [0isize]; // Zero cost (reduction dimension)

        // Zero cost -> never split (would cause race condition)
        assert!(!should_split_dimension(0, &dims, &blocks, &costs));
    }

    #[test]
    fn test_threaded_context_sequential() {
        let strides: Vec<isize> = vec![1, 100];
        let strides_ref: Vec<&[isize]> = vec![&strides];

        let ctx = ThreadedContext {
            dims: vec![10, 10], // 100 elements < MIN_THREAD_LENGTH
            blocks: vec![10, 10],
            strides: strides_ref,
            offsets: vec![0],
            costs: vec![2, 2],
            nthreads: 4,
            spacing: 0,
            task_index: 1,
        };

        // Small array should run sequentially
        assert!(ctx.should_run_sequential());
    }

    #[test]
    fn test_threaded_context_parallel() {
        let strides: Vec<isize> = vec![1, 1000];
        let strides_ref: Vec<&[isize]> = vec![&strides];

        let ctx = ThreadedContext {
            dims: vec![1000, 1000], // 1M elements > MIN_THREAD_LENGTH
            blocks: vec![100, 100],
            strides: strides_ref,
            offsets: vec![0],
            costs: vec![2, 2],
            nthreads: 4,
            spacing: 0,
            task_index: 1,
        };

        // Large array with multiple threads should run in parallel
        assert!(!ctx.should_run_sequential());
    }

    #[test]
    fn test_threaded_context_single_thread() {
        let strides: Vec<isize> = vec![1, 1000];
        let strides_ref: Vec<&[isize]> = vec![&strides];

        let ctx = ThreadedContext {
            dims: vec![1000, 1000],
            blocks: vec![100, 100],
            strides: strides_ref,
            offsets: vec![0],
            costs: vec![2, 2],
            nthreads: 1, // Only one thread
            spacing: 0,
            task_index: 1,
        };

        // Single thread should run sequentially
        assert!(ctx.should_run_sequential());
    }
}
