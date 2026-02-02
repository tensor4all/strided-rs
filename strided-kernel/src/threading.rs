//! Rayon-based parallel execution for strided operations.
//!
//! Faithfully ports Julia Strided.jl's `_mapreduce_threaded!` recursive
//! dimension-splitting strategy using `rayon::join`.

use smallvec::SmallVec;

use crate::kernel::for_each_inner_block_preordered;
use crate::Result;

/// Stack-allocated Vec for dims/offsets in recursive threading.
/// 8 elements covers up to 8-dimensional arrays (after fusion, typically 2-4).
type SVec<T> = SmallVec<[T; 8]>;

/// A raw pointer wrapper that is `Send` + `Sync`.
///
/// # Safety
/// The caller must guarantee that the pointed-to data is valid for the
/// lifetime of any parallel operation and that no data races occur
/// (e.g., different threads write to disjoint regions).
pub(crate) struct SendPtr<T>(pub(crate) *mut T);

impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SendPtr<T> {}

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    pub(crate) fn as_ptr(self) -> *mut T {
        self.0
    }

    pub(crate) fn as_const(self) -> *const T {
        self.0 as *const T
    }
}

/// Minimum number of elements to justify multi-threaded execution.
/// Matches Julia's `MINTHREADLENGTH = 1 << 15`.
pub(crate) const MINTHREADLENGTH: usize = 1 << 15;

/// Recursive dimension-splitting parallel execution.
///
/// Faithfully ports Julia's `_mapreduce_threaded!` (mapreduce.jl L195-227).
///
/// Parameters:
/// - `dims`: Ordered dimensions (after fuse/order/block)
/// - `blocks`: Block sizes per dimension
/// - `strides_list`: Per-array strides, ordered by plan
/// - `offsets`: Per-array byte offsets into the data
/// - `costs`: Per-dimension splitting costs
/// - `nthreads`: Number of threads available for this subtree
/// - `spacing`: For complete reduction — stride between thread-local output slots (0 for map)
/// - `taskindex`: 1-based task index for complete reduction output slot addressing
/// - `f`: Leaf function — called when we've reached a single-thread region
///
/// The leaf function `f` receives `(dims, blocks, strides_list, offsets)` describing
/// the sub-region to process.
pub(crate) fn mapreduce_threaded<F>(
    dims: &[usize],
    blocks: &[usize],
    strides_list: &[Vec<isize>],
    offsets: &[isize],
    costs: &[isize],
    nthreads: usize,
    spacing: isize,
    taskindex: usize,
    f: &F,
) -> Result<()>
where
    F: Fn(&[usize], &[usize], &[Vec<isize>], &[isize]) -> Result<()> + Sync,
{
    let total: usize = dims.iter().product();

    // Base case: single thread or below threshold
    if nthreads <= 1 || total <= MINTHREADLENGTH {
        if spacing != 0 {
            let mut spaced: SVec<isize> = SmallVec::from_slice(offsets);
            spaced[0] += spacing * (taskindex as isize - 1);
            return f(dims, blocks, strides_list, &spaced);
        }
        return f(dims, blocks, strides_list, offsets);
    }

    // Select split dimension: _lastargmax((dims .- 1) .* costs)
    // Streaming argmax avoids allocating a scores Vec.
    // Uses >= to match Julia's `_lastargmax` (ties broken by last index).
    let (i, _) = dims.iter().zip(costs.iter()).enumerate().fold(
        (0, isize::MIN),
        |(best_i, best_v), (idx, (&d, &c))| {
            let score = (d as isize - 1) * c;
            if score >= best_v {
                (idx, score)
            } else {
                (best_i, best_v)
            }
        },
    );

    // Guard: costs[i] == 0 || dims[i] <= min(blocks[i], 1024)
    if costs[i] == 0 || dims[i] <= blocks[i].min(1024) {
        if spacing != 0 {
            let mut spaced: SVec<isize> = SmallVec::from_slice(offsets);
            spaced[0] += spacing * (taskindex as isize - 1);
            return f(dims, blocks, strides_list, &spaced);
        }
        return f(dims, blocks, strides_list, offsets);
    }

    // Split dimension i in half
    let di = dims[i];
    let ndi = di / 2;
    let nt_left = nthreads / 2;
    let nt_right = nthreads - nt_left;

    // Left half: dims[i] = ndi, same offsets
    let mut left_dims: SVec<usize> = SmallVec::from_slice(dims);
    left_dims[i] = ndi;

    // Right half: dims[i] = di - ndi, offsets advanced by ndi * stride[i]
    let mut right_dims: SVec<usize> = SmallVec::from_slice(dims);
    right_dims[i] = di - ndi;
    let mut right_offsets: SVec<isize> = SmallVec::from_slice(offsets);
    for (k, strides) in strides_list.iter().enumerate() {
        right_offsets[k] += ndi as isize * strides[i];
    }

    let left_offsets: SVec<isize> = SmallVec::from_slice(offsets);

    // rayon::join for parallel left/right execution
    let (r1, r2) = rayon::join(
        || {
            mapreduce_threaded(
                &left_dims,
                blocks,
                strides_list,
                &left_offsets,
                costs,
                nt_left,
                spacing,
                taskindex,
                f,
            )
        },
        || {
            mapreduce_threaded(
                &right_dims,
                blocks,
                strides_list,
                &right_offsets,
                costs,
                nt_right,
                spacing,
                taskindex + nt_left,
                f,
            )
        },
    );
    r1?;
    r2?;
    Ok(())
}

/// Execute the kernel on a sub-region defined by initial offsets.
///
/// Delegates to `for_each_inner_block_preordered` which directly calls
/// kernel functions with the initial offsets, avoiding redundant re-ordering
/// and per-callback `Vec` allocation.
pub(crate) fn for_each_inner_block_with_offsets<F>(
    dims: &[usize],
    blocks: &[usize],
    strides_list: &[Vec<isize>],
    initial_offsets: &[isize],
    f: F,
) -> Result<()>
where
    F: FnMut(&[isize], usize, &[isize]) -> Result<()>,
{
    for_each_inner_block_preordered(dims, blocks, strides_list, initial_offsets, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: compute lastargmax via streaming fold (same logic as in mapreduce_threaded).
    fn streaming_lastargmax(dims: &[usize], costs: &[isize]) -> usize {
        let (i, _) = dims.iter().zip(costs.iter()).enumerate().fold(
            (0, isize::MIN),
            |(best_i, best_v), (idx, (&d, &c))| {
                let score = (d as isize - 1) * c;
                if score >= best_v {
                    (idx, score)
                } else {
                    (best_i, best_v)
                }
            },
        );
        i
    }

    #[test]
    fn test_streaming_lastargmax() {
        // Basic: scores = (9*2, 19*1, 4*3) = (18, 19, 12) → max at index 1
        assert_eq!(streaming_lastargmax(&[10, 20, 5], &[2, 1, 3]), 1);

        // Ties: last index wins (>= semantics)
        // scores: (10-1)*1=9, (10-1)*1=9, (10-1)*1=9 → all equal → last wins
        assert_eq!(streaming_lastargmax(&[10, 10, 10], &[1, 1, 1]), 2);

        // All dims=1: scores are all 0 → last wins
        assert_eq!(streaming_lastargmax(&[1, 1, 1], &[1, 1, 1]), 2);

        // Single dimension
        assert_eq!(streaming_lastargmax(&[100], &[2]), 0);
    }

    #[test]
    fn test_mapreduce_threaded_single_thread() {
        // With nthreads=1, should just call f directly
        let dims = vec![10, 10];
        let blocks = vec![10, 10];
        let strides = vec![vec![1isize, 10], vec![1, 10]];
        let offsets = vec![0isize, 0];
        let costs = vec![2, 20];

        let called = std::sync::atomic::AtomicBool::new(false);
        mapreduce_threaded(
            &dims,
            &blocks,
            &strides,
            &offsets,
            &costs,
            1,
            0,
            1,
            &|_dims, _blocks, _strides, _offsets| {
                called.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(())
            },
        )
        .unwrap();
        assert!(called.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn test_mapreduce_threaded_splits_cover_all_elements() {
        // Verify that parallel splitting covers all elements
        use std::sync::atomic::{AtomicUsize, Ordering};
        let dims = vec![100, 100];
        let blocks = vec![100, 100];
        let strides = vec![vec![1isize, 100], vec![1, 100]];
        let offsets = vec![0isize, 0];
        let costs = vec![2, 200];

        let total_elements = AtomicUsize::new(0);
        mapreduce_threaded(
            &dims,
            &blocks,
            &strides,
            &offsets,
            &costs,
            4,
            0,
            1,
            &|dims, _blocks, _strides, _offsets| {
                let n: usize = dims.iter().product();
                total_elements.fetch_add(n, Ordering::Relaxed);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(total_elements.load(Ordering::SeqCst), 10000);
    }

    #[test]
    fn test_mapreduce_threaded_with_spacing() {
        // Verify spacing/taskindex base case applies offsets correctly
        use std::sync::atomic::{AtomicI64, Ordering};
        let dims = vec![10];
        let blocks = vec![10];
        let strides = vec![vec![0isize], vec![1]];
        let offsets = vec![0isize, 0];
        let costs = vec![2];

        let received_offset = AtomicI64::new(0);
        mapreduce_threaded(
            &dims,
            &blocks,
            &strides,
            &offsets,
            &costs,
            1,
            8,
            3, // spacing=8, taskindex=3
            &|_dims, _blocks, _strides, offsets| {
                received_offset.store(offsets[0] as i64, Ordering::SeqCst);
                Ok(())
            },
        )
        .unwrap();
        // offset[0] should be 8 * (3 - 1) = 16
        assert_eq!(received_offset.load(Ordering::SeqCst), 16);
    }
}
