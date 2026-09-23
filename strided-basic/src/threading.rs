//! Rayon-based parallel execution for strided operations.
//!
//! Faithfully ports Julia Strided.jl's `_mapreduce_threaded!` recursive
//! dimension-splitting strategy using `rayon::join`.

#[cfg(feature = "parallel")]
use smallvec::SmallVec;
#[cfg(feature = "parallel")]
use std::ops::Range;

#[cfg(feature = "parallel")]
use crate::kernel::for_each_inner_block_preordered;
#[cfg(feature = "parallel")]
use crate::Result;

/// Stack-allocated Vec for dims/offsets in recursive threading.
/// 8 elements covers up to 8-dimensional arrays (after fusion, typically 2-4).
#[cfg(feature = "parallel")]
type SVec<T> = SmallVec<[T; 8]>;

/// A raw pointer wrapper that is `Send` + `Sync`.
///
/// # Safety
/// The caller must guarantee that the pointed-to data is valid for the
/// lifetime of any parallel operation and that no data races occur
/// (e.g., different threads write to disjoint regions).
#[cfg(feature = "parallel")]
/// Raw address transport for scoped kernel workers, without dereference or ownership.
/// Storage liveness, aliasing and cross-thread access must be established at use.
///
/// # Examples
/// ```
/// use strided_basic::execution::SendPtr;
/// let mut value = 1_i32;
/// let address = &mut value as *mut i32;
/// assert_eq!(SendPtr(address).as_ptr(), address);
/// ```
pub struct SendPtr<T>(pub *mut T);

#[cfg(feature = "parallel")]
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

#[cfg(feature = "parallel")]
impl<T> Copy for SendPtr<T> {}

#[cfg(feature = "parallel")]
unsafe impl<T> Send for SendPtr<T> {}
#[cfg(feature = "parallel")]
unsafe impl<T> Sync for SendPtr<T> {}

#[cfg(feature = "parallel")]
impl<T> SendPtr<T> {
    pub fn as_ptr(self) -> *mut T {
        self.0
    }

    pub fn as_const(self) -> *const T {
        self.0 as *const T
    }
}

/// Minimum number of elements to justify multi-threaded execution.
///
/// See `docs/design/erased-execution-policy.md` for the benchmark and decision
/// record. Matches Julia's `MINTHREADLENGTH = 1 << 15`.
#[cfg(feature = "parallel")]
pub const MINTHREADLENGTH: usize = 1 << 15;

#[cfg(feature = "parallel")]
pub fn parallel_threads_for_len(len: usize) -> usize {
    if len <= MINTHREADLENGTH {
        return 1;
    }
    let nthreads = crate::execution_policy::rayon_threads();
    if nthreads > 1 {
        nthreads
    } else {
        1
    }
}

#[cfg(feature = "parallel")]
fn join_with_policy<A, B, RA, RB>(policy: crate::ExecutionPolicy, left: A, right: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    match policy {
        crate::ExecutionPolicy::AmbientRayon => rayon::join(left, right),
        crate::ExecutionPolicy::Sequential | crate::ExecutionPolicy::Rayon { .. } => {
            crate::execution_policy::with_scheduler_suspended(|| {
                rayon::join(
                    || crate::execution_policy::with_owned_execution(policy, false, left),
                    || crate::execution_policy::with_owned_execution(policy, false, right),
                )
            })
        }
    }
}

#[cfg(feature = "parallel")]
pub fn parallel_map_reduce<R, Map, Reduce>(
    range: Range<usize>,
    nthreads: usize,
    map: &Map,
    reduce: &Reduce,
) -> R
where
    R: Send,
    Map: Fn(Range<usize>) -> R + Sync,
    Reduce: Fn(R, R) -> R + Sync,
{
    parallel_map_reduce_with_policy(
        range,
        nthreads,
        crate::execution_policy::active_policy(),
        false,
        map,
        reduce,
    )
}

#[cfg(feature = "parallel")]
pub(crate) fn parallel_for_each<F>(range: Range<usize>, nthreads: usize, operation: &F)
where
    F: Fn(Range<usize>) + Sync,
{
    parallel_map_reduce(
        range,
        nthreads,
        &|subrange| operation(subrange),
        &|(), ()| (),
    );
}

pub(crate) fn copy_into_col_major<T: Copy + crate::MaybeSendSync>(
    dest: &mut crate::StridedViewMut<T>,
    src: &crate::StridedView<T>,
) -> crate::Result<()> {
    strided_perm::copy_into_col_major(dest, src)
}

pub(crate) fn copy_permuted_serial<T: Copy + crate::MaybeSendSync>(
    dest: &mut crate::StridedViewMut<T>,
    src: &crate::StridedView<T>,
) -> crate::Result<()> {
    strided_perm::copy_into(dest, src)
}

#[cfg(test)]
#[path = "threading/tests/default_tests.rs"]
mod default_tests;

#[cfg(feature = "parallel")]
pub(crate) fn current_pool_threads() -> usize {
    rayon::current_num_threads()
}

#[cfg(feature = "parallel")]
fn with_permutation_copy_scheduler<R>(operation: impl FnOnce() -> R) -> R {
    crate::execution_policy::with_scheduler_suspended(operation)
}

#[cfg(feature = "parallel")]
pub(crate) fn copy_permuted_with_active_policy<T: Copy + crate::MaybeSendSync>(
    dest: &mut crate::StridedViewMut<T>,
    src: &crate::StridedView<T>,
) -> crate::Result<()> {
    let total = crate::kernel::total_len(dest.dims())?;
    let current_pool_threads = current_pool_threads();
    let parallel_eligible = crate::execution_policy::permutation_copy_parallel_eligible(
        crate::execution_policy::active_policy(),
        crate::execution_policy::fanout_active(),
        current_pool_threads,
    );
    if total > MINTHREADLENGTH && parallel_eligible {
        with_permutation_copy_scheduler(|| strided_perm::copy_into_par(dest, src))
    } else {
        copy_permuted_serial(dest, src)
    }
}

#[cfg(feature = "parallel")]
fn parallel_map_reduce_with_policy<R, Map, Reduce>(
    range: Range<usize>,
    nthreads: usize,
    policy: crate::ExecutionPolicy,
    in_fanout: bool,
    map: &Map,
    reduce: &Reduce,
) -> R
where
    R: Send,
    Map: Fn(Range<usize>) -> R + Sync,
    Reduce: Fn(R, R) -> R + Sync,
{
    let len = range.end - range.start;
    if nthreads <= 1 || len <= 1 {
        return crate::execution_policy::with_owned_execution(policy, in_fanout, || map(range));
    }

    let left_threads = nthreads / 2;
    let right_threads = nthreads - left_threads;
    let left_len = len * left_threads / nthreads;
    let middle = range.start + left_len.max(1).min(len - 1);
    let left_range = range.start..middle;
    let right_range = middle..range.end;

    let (left, right) = join_with_policy(
        policy,
        || parallel_map_reduce_with_policy(left_range, left_threads, policy, true, map, reduce),
        || parallel_map_reduce_with_policy(right_range, right_threads, policy, true, map, reduce),
    );
    crate::execution_policy::with_owned_execution(policy, true, || reduce(left, right))
}

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
#[cfg(feature = "parallel")]
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
    ThreadedMapReduce {
        blocks,
        strides_list,
        costs,
        spacing,
        policy: crate::execution_policy::active_policy(),
        operation: f,
    }
    .run(dims, offsets, nthreads, taskindex, false)
}

#[cfg(feature = "parallel")]
struct ThreadedMapReduce<'a, F> {
    blocks: &'a [usize],
    strides_list: &'a [Vec<isize>],
    costs: &'a [isize],
    spacing: isize,
    policy: crate::ExecutionPolicy,
    operation: &'a F,
}

#[cfg(feature = "parallel")]
impl<F> ThreadedMapReduce<'_, F>
where
    F: Fn(&[usize], &[usize], &[Vec<isize>], &[isize]) -> Result<()> + Sync,
{
    fn run(
        &self,
        dims: &[usize],
        offsets: &[isize],
        nthreads: usize,
        taskindex: usize,
        in_fanout: bool,
    ) -> Result<()> {
        let total = crate::kernel::total_len(dims)?;

        // Base case: single thread or below threshold
        if nthreads <= 1 || total <= MINTHREADLENGTH {
            if self.spacing != 0 {
                let mut spaced: SVec<isize> = SmallVec::from_slice(offsets);
                spaced[0] += self.spacing * (taskindex as isize - 1);
                return crate::execution_policy::with_owned_execution(
                    self.policy,
                    in_fanout,
                    || (self.operation)(dims, self.blocks, self.strides_list, &spaced),
                );
            }
            return crate::execution_policy::with_owned_execution(self.policy, in_fanout, || {
                (self.operation)(dims, self.blocks, self.strides_list, offsets)
            });
        }

        // Select split dimension: _lastargmax((dims .- 1) .* costs)
        // Streaming argmax avoids allocating a scores Vec.
        // Uses >= to match Julia's `_lastargmax` (ties broken by last index).
        let (i, _) = dims.iter().zip(self.costs.iter()).enumerate().fold(
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
        if self.costs[i] == 0 || dims[i] <= self.blocks[i].min(1024) {
            if self.spacing != 0 {
                let mut spaced: SVec<isize> = SmallVec::from_slice(offsets);
                spaced[0] += self.spacing * (taskindex as isize - 1);
                return crate::execution_policy::with_owned_execution(
                    self.policy,
                    in_fanout,
                    || (self.operation)(dims, self.blocks, self.strides_list, &spaced),
                );
            }
            return crate::execution_policy::with_owned_execution(self.policy, in_fanout, || {
                (self.operation)(dims, self.blocks, self.strides_list, offsets)
            });
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
        for (k, strides) in self.strides_list.iter().enumerate() {
            right_offsets[k] += ndi as isize * strides[i];
        }

        let left_offsets: SVec<isize> = SmallVec::from_slice(offsets);

        let (r1, r2) = join_with_policy(
            self.policy,
            || self.run(&left_dims, &left_offsets, nt_left, taskindex, true),
            || {
                self.run(
                    &right_dims,
                    &right_offsets,
                    nt_right,
                    taskindex + nt_left,
                    true,
                )
            },
        );
        r1?;
        r2?;
        Ok(())
    }
}

/// Execute the kernel on a sub-region defined by initial offsets.
///
/// Delegates to `for_each_inner_block_preordered` which directly calls
/// kernel functions with the initial offsets, avoiding redundant re-ordering
/// and per-callback `Vec` allocation.
#[cfg(feature = "parallel")]
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

#[cfg(all(test, feature = "parallel"))]
pub(crate) fn test_pool(threads: usize) -> std::sync::Arc<rayon::ThreadPool> {
    std::sync::Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap(),
    )
}

#[cfg(all(test, feature = "parallel"))]
#[path = "threading/tests/tests.rs"]
mod tests;

#[cfg(feature = "parallel")]
impl<T> core::fmt::Debug for SendPtr<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("SendPtr").field(&self.0).finish()
    }
}
