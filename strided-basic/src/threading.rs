//! Parallel execution for strided operations.
//!
//! Faithfully ports Julia Strided.jl's `_mapreduce_threaded!` recursive
//! dimension-splitting strategy. Every split goes through
//! [`threadteam::Exec::fork`] or [`threadteam::Exec::for_ranges`]: the
//! token carries the budget and the pool, so this module holds no placement
//! state of its own.

use crate::cost::KernelClass;
pub(crate) use crate::cost::{for_len, is_parallel};
use crate::inline::InlineVec;
use crate::kernel::for_each_inner_block_preordered;
use crate::Result;
use core::num::NonZeroUsize;
use std::ops::Range;
use threadteam::Exec;

/// Stack-allocated Vec for dims/offsets in recursive threading.
/// 8 elements covers up to 8-dimensional arrays (after fusion, typically 2-4).
type SVec<T> = InlineVec<T>;

const ONE: NonZeroUsize = NonZeroUsize::MIN;

/// A raw pointer wrapper that is `Send` + `Sync`.
///
/// # Safety
/// The caller must guarantee that the pointed-to data is valid for the
/// lifetime of any parallel operation and that no data races occur
/// (e.g., different threads write to disjoint regions).
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

impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SendPtr<T> {}

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    pub fn as_ptr(self) -> *mut T {
        self.0
    }

    pub fn as_const(self) -> *const T {
        self.0 as *const T
    }
}

impl<T> core::fmt::Debug for SendPtr<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("SendPtr").field(&self.0).finish()
    }
}

/// Block sizes handed to the parallel splitter.
///
/// [`split_axis`] declines to split a dimension that fits in one cache block
/// (`dims[i] <= blocks[i].min(1024)`), so a region whose block covers a small
/// whole tensor (for example `1024x64` with `block == dims`) never opens even
/// at four threads. Halving such a block lets the splitter open; the leaf
/// merely walks two tiles instead of one. Dimensions that already exceed the
/// guard are left unchanged so other shapes keep their tiles.
fn scheduler_blocks(dims: &[usize], blocks: &[usize]) -> SVec<usize> {
    dims.iter()
        .zip(blocks)
        .map(|(&dim, &block)| {
            if dim > 1 && dim <= block.min(1024) {
                (dim / 2).max(1)
            } else {
                block
            }
        })
        .collect()
}

/// Runs `left` and `right` as the two halves of `exec` and returns both results.
///
/// Each half receives a child token carrying its share of the budget.
fn fork2<R: Send>(
    exec: Exec<'_>,
    left: impl Fn(Exec<'_>) -> R + Sync,
    right: impl Fn(Exec<'_>) -> R + Sync,
) -> (R, R) {
    // Mutex-free result slots: a `Mutex` allocates its lock on first use on
    // some platforms, which made every call allocate.
    struct Slot<T>(core::cell::UnsafeCell<Option<T>>);
    // SAFETY: job `i` is the only writer of slot `i`, and the slots are read only
    // after `fork` has returned, which waits for every job.
    unsafe impl<T: Send> Sync for Slot<T> {}
    let slots = [Slot(Default::default()), Slot(Default::default())];
    exec.fork(2, |i, child| {
        let value = if i == 0 { left(child) } else { right(child) };
        // SAFETY: see `Slot`; each index is written by exactly one job.
        unsafe { *slots[i].0.get() = Some(value) };
    });
    let [left, right] = slots.map(|slot| {
        slot.0
            .into_inner()
            .expect("every fork job stores its result")
    });
    (left, right)
}

/// Recursive halving map-reduce over `range` within the budget of `exec`.
///
/// `map` runs on the leaf ranges and `reduce` combines two neighbouring
/// results left to right, so the combination order depends only on the budget.
/// A serial token, a budget of one or a single-element range runs `map` on the
/// whole range on the calling thread.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::parallel_map_reduce;
/// use strided_basic::Exec;
///
/// let sum = parallel_map_reduce(0..100, Exec::serial(), &|r| r.sum::<usize>(), &|a, b| a + b);
/// assert_eq!(sum, 4950);
/// ```
pub fn parallel_map_reduce<R, Map, Reduce>(
    range: Range<usize>,
    exec: Exec<'_>,
    map: &Map,
    reduce: &Reduce,
) -> R
where
    R: Send,
    Map: Fn(Range<usize>) -> R + Sync,
    Reduce: Fn(R, R) -> R + Sync,
{
    let len = range.end - range.start;
    let budget = exec.budget().get();
    if budget <= 1 || len <= 1 {
        return map(range);
    }

    let left_threads = budget.div_ceil(2);
    let left_len = len * left_threads / budget;
    let middle = range.start + left_len.max(1).min(len - 1);
    let left_range = range.start..middle;
    let right_range = middle..range.end;

    let (left, right) = fork2(
        exec,
        |child| parallel_map_reduce(left_range.clone(), child, map, reduce),
        |child| parallel_map_reduce(right_range.clone(), child, map, reduce),
    );
    reduce(left, right)
}

/// Runs `operation` over contiguous sub-ranges of `range`, one per slot of
/// `exec` at most.
pub(crate) fn parallel_for_each<F>(range: Range<usize>, exec: Exec<'_>, operation: &F)
where
    F: Fn(Range<usize>) + Sync,
{
    let start = range.start;
    exec.for_ranges(range.end - range.start, ONE, |chunk| {
        operation(start + chunk.start..start + chunk.end)
    });
}

/// Copy `len` contiguous elements, chunked across the budget of `exec` when
/// one section pays for a copy of that size (`KernelClass::Copy`) and as one
/// direct `copy_nonoverlapping` otherwise.
///
/// # Safety
///
/// `src..src + len` must be readable and `dst..dst + len` writable for the
/// whole call, the two ranges must not overlap, and no other thread may
/// access the destination range concurrently.
pub(crate) unsafe fn copy_contiguous<T: Copy + Send + Sync>(
    src: *const T,
    dst: *mut T,
    len: usize,
    exec: Exec<'_>,
) {
    let exec = for_len(exec, KernelClass::Copy, len);
    if !exec.is_serial() {
        let src = SendPtr(src as *mut T);
        let dst = SendPtr(dst);
        parallel_for_each(0..len, exec, &|range| {
            // SAFETY: `range` lies inside `0..len`, and worker ranges are
            // disjoint, so each worker copies an exclusive destination
            // chunk from the caller-validated source range.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    src.as_const().add(range.start),
                    dst.as_ptr().add(range.start),
                    range.len(),
                );
            }
        });
        return;
    }
    // SAFETY: the caller guarantees both ranges are valid and disjoint.
    unsafe { core::ptr::copy_nonoverlapping(src, dst, len) };
}

/// Fill a contiguous slice, chunked across the budget of `exec` when a section
/// pays for it (`KernelClass::Copy`) and as one serial `fill` otherwise.
pub(crate) fn fill_contiguous<T: Copy + Send + Sync>(dst: &mut [T], value: T, exec: Exec<'_>) {
    let len = dst.len();
    let exec = for_len(exec, KernelClass::Copy, len);
    if !exec.is_serial() {
        let dst = SendPtr(dst.as_mut_ptr());
        parallel_for_each(0..len, exec, &|range| {
            // SAFETY: `range` lies inside the exclusively borrowed slice
            // and worker ranges are disjoint.
            let chunk = unsafe {
                core::slice::from_raw_parts_mut(dst.as_ptr().add(range.start), range.len())
            };
            chunk.fill(value);
        });
        return;
    }
    dst.fill(value);
}

/// Fill `len` contiguous destination elements that may be uninitialized.
///
/// [`fill_contiguous`] forms a `&mut [T]`, which is undefined behaviour over
/// uninitialized storage (for example a fresh `bool` buffer). This sibling
/// views the same span as `[MaybeUninit<T>]`, which has no validity invariant,
/// and reuses the existing chunking.
///
/// # Safety
///
/// `dst..dst + len` must be writable for the whole call and no other thread
/// may access the destination range concurrently.
pub(crate) unsafe fn fill_contiguous_uninit<T: Copy + Send + Sync>(
    dst: *mut T,
    len: usize,
    value: T,
    exec: Exec<'_>,
) {
    // SAFETY: `MaybeUninit<T>` has no validity invariant, so the span is a
    // valid `[MaybeUninit<T>]`; `fill_contiguous` writes every element.
    let span =
        unsafe { core::slice::from_raw_parts_mut(dst.cast::<core::mem::MaybeUninit<T>>(), len) };
    fill_contiguous(span, core::mem::MaybeUninit::new(value), exec);
}

/// Permuting copy into a destination of any layout, split across `exec`.
///
/// The permutation engine decides the width itself (copy class, one section),
/// so this is the only route: a serial token (or a copy too small to repay a
/// section) takes its calling-thread path.
pub(crate) fn copy_permuted<T: Copy + Send + Sync + 'static>(
    dest: &mut crate::StridedViewMut<T>,
    src: &crate::StridedView<T>,
    exec: Exec<'_>,
) -> crate::Result<()> {
    strided_perm::copy_into(dest, src, exec)
}

/// [`copy_permuted`] into an uninitialized destination.
///
/// The permutation engine is entered on the source element type, so its
/// native `f32`/`f64` paths apply: a destination reinterpreted as
/// `MaybeUninit<T>` falls to the generic element path, measured 2.5x slower
/// on a 4096x4096 `f64` transpose.
pub(crate) fn copy_permuted_uninit<T: Copy + Send + Sync + 'static>(
    dest: &mut crate::StridedViewMut<core::mem::MaybeUninit<T>>,
    src: &crate::StridedView<T>,
    exec: Exec<'_>,
) -> crate::Result<()> {
    strided_perm::copy_into_uninit(dest, src, exec)
}

/// Dimension to halve: `_lastargmax((dims .- 1) .* costs)`, or `None` when
/// that dimension is not worth splitting.
///
/// Ties break towards the last index, matching Julia's `_lastargmax`. A
/// dimension is not split when its cost is zero or it fits in one block
/// (`dims[i] <= min(blocks[i], 1024)`).
fn split_axis(dims: &[usize], blocks: &[usize], costs: &[isize]) -> Option<usize> {
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
    (costs[i] != 0 && dims[i] > blocks[i].min(1024)).then_some(i)
}

/// The two halves of `dims` split at axis `i`: left dims, right dims and the
/// right offsets advanced by the left extent.
fn halves(
    dims: &[usize],
    offsets: &[isize],
    strides_list: &[&[isize]],
    i: usize,
) -> (SVec<usize>, SVec<usize>, SVec<isize>) {
    let di = dims[i];
    let ndi = di / 2;
    let mut left_dims: SVec<usize> = dims.iter().copied().collect();
    left_dims[i] = ndi;
    let mut right_dims: SVec<usize> = dims.iter().copied().collect();
    right_dims[i] = di - ndi;
    let mut right_offsets: SVec<isize> = offsets.iter().copied().collect();
    for (k, strides) in strides_list.iter().enumerate() {
        right_offsets[k] += ndi as isize * strides[i];
    }
    (left_dims, right_dims, right_offsets)
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
/// - `exec`: Placement token; its budget is the number of slots the whole
///   region may occupy, divided between the halves at every split
/// - `f`: Leaf function — called when we've reached a single-slot region
///
/// The leaf function `f` receives `(dims, blocks, strides_list, offsets)` describing
/// the sub-region to process.
pub(crate) fn mapreduce_threaded<S: AsRef<[isize]>, F>(
    dims: &[usize],
    blocks: &[usize],
    strides_list: &[S],
    offsets: &[isize],
    costs: &[isize],
    class: KernelClass,
    exec: Exec<'_>,
    f: &F,
) -> Result<()>
where
    F: Fn(&[usize], &[usize], &[&[isize]], &[isize]) -> Result<()> + Sync,
{
    // The recursive splitter is compiled once; the leaf is `dyn`, called once
    // per leaf region. Borrowed per-operand stride slices are built once here
    // so the splitter and its leaves share them without owning a
    // `Vec<Vec<isize>>` (#306 P1).
    let refs: SVec<&[isize]> = strided_slices(strides_list);
    mapreduce_threaded_dyn(dims, blocks, &refs, offsets, costs, class, exec, f)
}

fn strided_slices<S: AsRef<[isize]>>(strides_list: &[S]) -> SVec<&[isize]> {
    strides_list.iter().map(|s| s.as_ref()).collect()
}

/// The leaf callback of a threaded region.
type LeafFn<'a> = &'a (dyn Fn(&[usize], &[usize], &[&[isize]], &[isize]) -> Result<()> + Sync);

#[inline(never)]
fn mapreduce_threaded_dyn(
    dims: &[usize],
    blocks: &[usize],
    strides_list: &[&[isize]],
    offsets: &[isize],
    costs: &[isize],
    class: KernelClass,
    exec: Exec<'_>,
    f: LeafFn<'_>,
) -> Result<()> {
    // Normalize before the splitter: the guard in `split_axis` uses these tile
    // sizes, so a block that covers its whole dimension would otherwise
    // suppress the split. The shared layer applies this for every map, zip and
    // fused caller.
    let blocks = scheduler_blocks(dims, blocks);
    ThreadedMapReduce {
        blocks: &blocks,
        strides_list,
        costs,
        class,
        operation: f,
    }
    .run(dims, offsets, exec)
}

struct ThreadedMapReduce<'a> {
    blocks: &'a [usize],
    strides_list: &'a [&'a [isize]],
    costs: &'a [isize],
    class: KernelClass,
    operation: LeafFn<'a>,
}

impl ThreadedMapReduce<'_> {
    fn run(&self, dims: &[usize], offsets: &[isize], exec: Exec<'_>) -> Result<()> {
        let total = crate::kernel::total_len(dims)?;

        // Base case: a single slot, or a fork of this region does not pay.
        if !is_parallel(self.class, total, &exec) {
            return (self.operation)(dims, self.blocks, self.strides_list, offsets);
        }
        let Some(i) = split_axis(dims, self.blocks, self.costs) else {
            return (self.operation)(dims, self.blocks, self.strides_list, offsets);
        };

        let (left_dims, right_dims, right_offsets) = halves(dims, offsets, self.strides_list, i);
        let (r1, r2) = fork2(
            exec,
            |child| self.run(&left_dims, offsets, child),
            |child| self.run(&right_dims, &right_offsets, child),
        );
        r1?;
        r2?;
        Ok(())
    }
}

/// Recursive dimension-splitting parallel fold.
///
/// Like [`mapreduce_threaded`], but the leaf returns a partial result and each
/// split combines the two children with `combine`. A full reduction therefore
/// never needs a worker-count-sized output-slot vector (#306 scope addition).
/// `f` must reduce the whole sub-region it is handed, using the supplied
/// `for_each_inner_block_with_offsets` traversal.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mapreduce_threaded_fold<S, F, C, R>(
    dims: &[usize],
    blocks: &[usize],
    strides_list: &[S],
    offsets: &[isize],
    costs: &[isize],
    class: KernelClass,
    exec: Exec<'_>,
    f: &F,
    combine: &C,
) -> Result<R>
where
    S: AsRef<[isize]>,
    F: Fn(&[usize], &[usize], &[&[isize]], &[isize]) -> Result<R> + Sync,
    C: Fn(R, R) -> R + Sync,
    R: Send,
{
    let refs: SVec<&[isize]> = strided_slices(strides_list);
    // Normalize whole-dimension blocks before splitting, matching the
    // `mapreduce_threaded` path (#313): a block that covers its dimension would
    // otherwise suppress the split for a small whole tensor.
    let blocks = scheduler_blocks(dims, blocks);
    FoldMapReduce {
        blocks: &blocks,
        strides_list: &refs,
        costs,
        class,
        f,
        combine,
    }
    .run(dims, offsets, exec)
}

struct FoldMapReduce<'a, F, C> {
    blocks: &'a [usize],
    strides_list: &'a [&'a [isize]],
    costs: &'a [isize],
    class: KernelClass,
    f: &'a F,
    combine: &'a C,
}

impl<F, C> FoldMapReduce<'_, F, C> {
    fn run<R: Send>(&self, dims: &[usize], offsets: &[isize], exec: Exec<'_>) -> Result<R>
    where
        F: Fn(&[usize], &[usize], &[&[isize]], &[isize]) -> Result<R> + Sync,
        C: Fn(R, R) -> R + Sync,
    {
        let total = crate::kernel::total_len(dims)?;

        // Base case: a single slot, or a fork of this region does not pay.
        if !is_parallel(self.class, total, &exec) {
            return (self.f)(dims, self.blocks, self.strides_list, offsets);
        }
        let Some(i) = split_axis(dims, self.blocks, self.costs) else {
            return (self.f)(dims, self.blocks, self.strides_list, offsets);
        };

        // Split dimension i in half and combine the two partial results.
        let (left_dims, right_dims, right_offsets) = halves(dims, offsets, self.strides_list, i);
        let (r1, r2) = fork2(
            exec,
            |child| self.run(&left_dims, offsets, child),
            |child| self.run(&right_dims, &right_offsets, child),
        );
        Ok((self.combine)(r1?, r2?))
    }
}

/// Execute the kernel on a sub-region defined by initial offsets.
///
/// Delegates to `for_each_inner_block_preordered` which directly calls
/// kernel functions with the initial offsets, avoiding redundant re-ordering
/// and per-callback `Vec` allocation.
pub(crate) fn for_each_inner_block_with_offsets<S: AsRef<[isize]>, F>(
    dims: &[usize],
    blocks: &[usize],
    strides_list: &[S],
    initial_offsets: &[isize],
    f: F,
) -> Result<()>
where
    F: FnMut(&[isize], usize, &[isize]) -> Result<()>,
{
    for_each_inner_block_preordered(dims, blocks, strides_list, initial_offsets, f)
}

/// A pool of `threads` workers for unit tests.
#[cfg(test)]
pub(crate) fn test_pool(threads: usize) -> threadteam::Pool {
    threadteam::Pool::builder()
        .threads(threads)
        .dispatch_costs(test_dispatch_costs())
        .build()
        .unwrap()
}

/// Dispatch costs of the unit-test pools: 100 nanoseconds per section whatever the
/// token, so that regions of a few thousand elements open the parallel branches
/// the tests compare against the serial ones.
#[cfg(test)]
pub(crate) fn test_dispatch_costs() -> threadteam::DispatchCosts {
    use threadteam::{DispatchKind, WidthCosts};
    let cheap = WidthCosts::flat(core::time::Duration::from_nanos(100));
    threadteam::DispatchCosts::for_class(threadteam::HostClass::Generic)
        .with(DispatchKind::Team, cheap)
        .with(DispatchKind::InPool, cheap)
        .with(DispatchKind::Outside, cheap)
}

/// Run `operation` once with a serial token and once with a four-slot token of
/// a four-worker pool, and return both results.
///
/// The parallel run asserts that the gate actually opens for a region of a
/// million elements, so equivalence tests with such sizes exercise the parallel
/// branches. Without the `parallel` feature the pool has
/// width one and both runs are serial.
#[cfg(test)]
pub(crate) fn run_serial_and_parallel<R: Send>(operation: impl Fn(Exec<'_>) -> R + Sync) -> (R, R) {
    let serial = operation(Exec::serial());
    let pool = test_pool(4);
    let exec = Exec::new(&pool, 4).unwrap();
    assert_eq!(
        crate::cost::parallel_width(&exec, KernelClass::Map, 1 << 20).get(),
        exec.budget().get()
    );
    let parallel = operation(exec);
    (serial, parallel)
}

#[cfg(test)]
#[path = "threading/tests/tests.rs"]
mod tests;
