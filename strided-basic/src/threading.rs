//! Rayon-based parallel execution for strided operations.
//!
//! Faithfully ports Julia Strided.jl's `_mapreduce_threaded!` recursive
//! dimension-splitting strategy using `rayon::join`.

#[cfg(feature = "parallel")]
use crate::inline::InlineVec;
#[cfg(feature = "parallel")]
use std::ops::Range;

#[cfg(feature = "parallel")]
use core::num::NonZeroUsize;

#[cfg(feature = "parallel")]
use crate::kernel::for_each_inner_block_preordered;
#[cfg(feature = "parallel")]
use crate::Result;

/// Stack-allocated Vec for dims/offsets in recursive threading.
/// 8 elements covers up to 8-dimensional arrays (after fusion, typically 2-4).
#[cfg(feature = "parallel")]
type SVec<T> = InlineVec<T>;

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

/// Block sizes handed to the parallel splitter.
///
/// [`ThreadedMapReduce::run`] declines to split a dimension that fits in one
/// cache block (`dims[i] <= blocks[i].min(1024)`), so a region whose block
/// covers a small whole tensor (for example `1024x64` with `block == dims`)
/// never opens even at four threads. Halving such a block lets the splitter
/// open; the leaf merely walks two tiles instead of one. Dimensions that
/// already exceed the guard are left unchanged so other shapes keep their
/// tiles.
#[cfg(feature = "parallel")]
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
    let run = || match policy {
        crate::ExecutionPolicy::AmbientRayon => rayon::join(left, right),
        // A sequential policy owns no fanout, even when a low-level caller
        // passes a thread budget above one. Do not let `rayon::join` resolve
        // the global pool through the scheduler suspension below; keep the
        // same owned policy/fanout state the join branch would have installed.
        crate::ExecutionPolicy::Sequential => (
            crate::execution_policy::with_owned_execution(policy, false, left),
            crate::execution_policy::with_owned_execution(policy, false, right),
        ),
        crate::ExecutionPolicy::Rayon { .. } => {
            crate::execution_policy::with_scheduler_suspended(|| {
                rayon::join(
                    || crate::execution_policy::with_owned_execution(policy, false, left),
                    || crate::execution_policy::with_owned_execution(policy, false, right),
                )
            })
        }
    };
    // A target-pool scope owns placement even for the ambient-policy branch:
    // enter the target pool at the first real split, not once per recursion
    // level. A sequential policy never enters.
    if matches!(policy, crate::ExecutionPolicy::Sequential) {
        run()
    } else {
        enter_target_pool(run)
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
    let policy = crate::execution_policy::active_policy();
    parallel_map_reduce_with_policy(
        range,
        crate::execution_policy::clamp_fanout_width(nthreads, policy),
        policy,
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

/// Copy `len` contiguous elements, chunked across workers above the
/// repository threshold ([`parallel_threads_for_len`]) and as one direct
/// `copy_nonoverlapping` otherwise (always, without the `parallel` feature).
///
/// # Safety
///
/// `src..src + len` must be readable and `dst..dst + len` writable for the
/// whole call, the two ranges must not overlap, and no other thread may
/// access the destination range concurrently.
pub(crate) unsafe fn copy_contiguous<T: Copy + crate::MaybeSendSync>(
    src: *const T,
    dst: *mut T,
    len: usize,
) {
    #[cfg(feature = "parallel")]
    {
        let nthreads = parallel_threads_for_len(len);
        if nthreads > 1 {
            let src = SendPtr(src as *mut T);
            let dst = SendPtr(dst);
            parallel_for_each(0..len, nthreads, &|range| {
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
    }
    // SAFETY: the caller guarantees both ranges are valid and disjoint.
    unsafe { core::ptr::copy_nonoverlapping(src, dst, len) };
}

/// Fill a contiguous slice, chunked across workers above the repository
/// threshold and as one serial `fill` otherwise.
pub(crate) fn fill_contiguous<T: Copy + crate::MaybeSendSync>(dst: &mut [T], value: T) {
    #[cfg(feature = "parallel")]
    {
        let len = dst.len();
        let nthreads = parallel_threads_for_len(len);
        if nthreads > 1 {
            let dst = SendPtr(dst.as_mut_ptr());
            parallel_for_each(0..len, nthreads, &|range| {
                // SAFETY: `range` lies inside the exclusively borrowed slice
                // and worker ranges are disjoint.
                let chunk = unsafe {
                    core::slice::from_raw_parts_mut(dst.as_ptr().add(range.start), range.len())
                };
                chunk.fill(value);
            });
            return;
        }
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
pub(crate) unsafe fn fill_contiguous_uninit<T: Copy + crate::MaybeSendSync>(
    dst: *mut T,
    len: usize,
    value: T,
) {
    // SAFETY: `MaybeUninit<T>` has no validity invariant, so the span is a
    // valid `[MaybeUninit<T>]`; `fill_contiguous` writes every element.
    let span =
        unsafe { core::slice::from_raw_parts_mut(dst.cast::<core::mem::MaybeUninit<T>>(), len) };
    fill_contiguous(span, core::mem::MaybeUninit::new(value));
}

pub(crate) fn copy_into_col_major<T: Copy + crate::MaybeSendSync + 'static>(
    dest: &mut crate::StridedViewMut<T>,
    src: &crate::StridedView<T>,
) -> crate::Result<()> {
    strided_perm::copy_into_col_major(dest, src)
}

pub(crate) fn copy_permuted_serial<T: Copy + crate::MaybeSendSync + 'static>(
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

/// A caller-supplied pool that strided fan-out must use instead of the ambient
/// one. The pointer is valid only for the guarding scope on this thread.
#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct TargetPool {
    pool: *const rayon::ThreadPool,
    max_threads: NonZeroUsize,
}

#[cfg(feature = "parallel")]
std::thread_local! {
    static TARGET_POOL: core::cell::Cell<Option<TargetPool>> = const { core::cell::Cell::new(None) };
}

#[cfg(all(test, feature = "parallel"))]
std::thread_local! {
    static TARGET_POOL_INSTALLS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

#[cfg(all(test, feature = "parallel"))]
pub(crate) fn target_pool_install_count() -> usize {
    TARGET_POOL_INSTALLS.with(core::cell::Cell::get)
}

#[cfg(all(test, feature = "parallel"))]
pub(crate) fn reset_target_pool_install_count() {
    TARGET_POOL_INSTALLS.with(|count| count.set(0));
}

#[cfg(all(test, feature = "parallel"))]
static PERMUTATION_FALLBACK_WORKERS: std::sync::Mutex<Vec<Option<String>>> =
    std::sync::Mutex::new(Vec::new());

#[cfg(all(test, feature = "parallel"))]
fn record_permutation_fallback_worker() {
    let name = std::thread::current().name().map(str::to_owned);
    let mut workers = PERMUTATION_FALLBACK_WORKERS.lock().unwrap();
    if workers.len() < 32 && !workers.iter().any(|entry| *entry == name) {
        workers.push(name);
    }
}

#[cfg(all(test, feature = "parallel"))]
pub(crate) fn take_permutation_fallback_workers() -> Vec<Option<String>> {
    std::mem::take(&mut *PERMUTATION_FALLBACK_WORKERS.lock().unwrap())
}

#[cfg(feature = "parallel")]
struct TargetPoolGuard {
    previous: Option<TargetPool>,
}

#[cfg(feature = "parallel")]
impl Drop for TargetPoolGuard {
    fn drop(&mut self) {
        set_target_pool(self.previous);
    }
}

#[cfg(feature = "parallel")]
fn target_pool() -> Option<TargetPool> {
    TARGET_POOL.with(core::cell::Cell::get)
}

#[cfg(feature = "parallel")]
fn set_target_pool(pool: Option<TargetPool>) {
    TARGET_POOL.with(|cell| cell.set(pool));
}

#[cfg(feature = "parallel")]
pub(crate) fn with_execution_pool<R>(
    pool: &rayon::ThreadPool,
    max_threads: NonZeroUsize,
    operation: impl FnOnce() -> R,
) -> R {
    let previous = target_pool();
    // Nested scopes stay conservative: the inner pool takes over placement,
    // but it cannot enlarge an enclosing thread budget.
    let max_threads = previous
        .map(|outer| outer.max_threads.min(max_threads))
        .unwrap_or(max_threads);
    set_target_pool(Some(TargetPool {
        pool: pool as *const _,
        max_threads,
    }));
    let _guard = TargetPoolGuard { previous };
    operation()
}

/// Suspend the target-pool scope while the caller may run unrelated Rayon
/// work (an owned install or a scheduler boundary).
#[cfg(feature = "parallel")]
pub(crate) fn with_target_pool_suspended<R>(operation: impl FnOnce() -> R) -> R {
    let previous = target_pool();
    set_target_pool(None);
    let _guard = TargetPoolGuard { previous };
    operation()
}

/// Worker count of the scoped target pool, clamped by its thread budget.
#[cfg(feature = "parallel")]
pub(crate) fn target_pool_width() -> Option<usize> {
    target_pool().map(|target| {
        // SAFETY: the pointer was installed by `with_execution_pool` and is
        // valid until its guard drops, which cannot happen while this
        // thread-local borrow is executing.
        let pool = unsafe { &*target.pool };
        target.max_threads.get().min(pool.current_num_threads())
    })
}

/// Physical worker count of the scoped target pool, ignoring the thread budget.
///
/// Permutation eligibility needs this: the HPTT entry parallelizes over every
/// worker in the installed pool, so a pool larger than the cap must use the
/// bounded map fallback instead.
#[cfg(feature = "parallel")]
pub(crate) fn target_pool_physical_threads() -> Option<usize> {
    target_pool().map(|target| {
        // SAFETY: see `target_pool_width`.
        let pool = unsafe { &*target.pool };
        pool.current_num_threads()
    })
}

/// Enter the scoped target pool when the caller is outside it and the target
/// width permits fan-out. Without a scope this is a plain call.
#[cfg(feature = "parallel")]
pub(crate) fn enter_target_pool<R: Send>(operation: impl FnOnce() -> R + Send) -> R {
    let Some(target) = target_pool() else {
        return operation();
    };
    // SAFETY: see `target_pool_width`.
    let pool = unsafe { &*target.pool };
    let width = target.max_threads.get().min(pool.current_num_threads());
    if width <= 1 || pool.current_thread_index().is_some() {
        operation()
    } else {
        #[cfg(all(test, feature = "parallel"))]
        TARGET_POOL_INSTALLS.with(|count| count.set(count.get() + 1));
        // The caller waits here and Rayon may let it run unrelated work from its
        // own pool. Suspend both the execution policy and the target-pool scope
        // so that work cannot inherit this operation's cap. The installed child
        // receives the captured policy explicitly.
        crate::execution_policy::with_scheduler_suspended(|| pool.install(operation))
    }
}

/// Runs `operation` on `pool`, clamping `budget` to the pool's worker count and
/// entering the pool only when the caller is not already one of its workers and
/// the effective width permits fanout.
///
/// This is the only place [`crate::ExecContext::run_on`] touches the Rayon
/// pool; the policy construction stays in `exec_context.rs` and the placement
/// stays next to the rest of the repository's Rayon interaction.
#[cfg(feature = "parallel")]
pub(crate) fn run_on_pool<T: Send>(
    pool: &rayon::ThreadPool,
    budget: core::num::NonZeroUsize,
    operation: impl FnOnce(usize) -> T + Send,
) -> T {
    let width = budget.get().min(pool.current_num_threads());
    let run = move || operation(width);
    if width <= 1 || pool.current_thread_index().is_some() {
        run()
    } else {
        pool.install(run)
    }
}

#[cfg(feature = "parallel")]
fn with_permutation_copy_scheduler<R>(operation: impl FnOnce() -> R) -> R {
    crate::execution_policy::with_scheduler_suspended(operation)
}

/// Which entry a permutation copy takes under the active policy.
#[cfg(feature = "parallel")]
enum PermutationRoute {
    /// The permutation engine's own parallel entry, inside the target pool.
    Parallel,
    /// A bounded policy (`Rayon { max_threads }` below the pool size) is not
    /// eligible for the permutation engine's own parallel entry, but the
    /// element map honours the same policy budget, so it keeps more workers
    /// than the serial permutation copy would.
    BoundedMap,
    Serial,
}

/// Resolves the policy before touching Rayon. A sequential policy must not
/// probe the ambient pool width: outside a worker, `current_num_threads`
/// resolves the global registry and starts the global Rayon pool. A target
/// pool scope reports its own width and never consults the ambient registry.
#[cfg(feature = "parallel")]
fn permutation_route(dims: &[usize]) -> crate::Result<PermutationRoute> {
    let policy = crate::execution_policy::active_policy();
    let total = crate::kernel::total_len(dims)?;
    if policy == crate::ExecutionPolicy::Sequential {
        return Ok(PermutationRoute::Serial);
    }
    let physical_pool_threads = target_pool_physical_threads().unwrap_or_else(current_pool_threads);
    let effective_pool_threads = target_pool_width().unwrap_or(physical_pool_threads);
    let parallel_eligible = crate::execution_policy::permutation_copy_parallel_eligible(
        policy,
        crate::execution_policy::fanout_active(),
        physical_pool_threads,
    );
    Ok(if total > MINTHREADLENGTH && parallel_eligible {
        PermutationRoute::Parallel
    } else if total > MINTHREADLENGTH && effective_pool_threads > 1 {
        PermutationRoute::BoundedMap
    } else {
        PermutationRoute::Serial
    })
}

#[cfg(feature = "parallel")]
pub(crate) fn copy_permuted_with_active_policy<T: Copy + crate::MaybeSendSync + 'static>(
    dest: &mut crate::StridedViewMut<T>,
    src: &crate::StridedView<T>,
) -> crate::Result<()> {
    match permutation_route(dest.dims())? {
        // The permutation engine resolves Rayon's current pool internally, so
        // the target pool must be installed around the whole parallel entry.
        PermutationRoute::Parallel => enter_target_pool(|| {
            with_permutation_copy_scheduler(|| strided_perm::copy_into_par(dest, src))
        }),
        PermutationRoute::BoundedMap => crate::map_view::map_into(dest, src, |value| {
            #[cfg(all(test, feature = "parallel"))]
            record_permutation_fallback_worker();
            value
        }),
        PermutationRoute::Serial => copy_permuted_serial(dest, src),
    }
}

/// [`copy_permuted_with_active_policy`] into an uninitialized destination.
///
/// The permutation engine is entered on the source element type, so its
/// native `f32`/`f64` paths apply: a destination reinterpreted as
/// `MaybeUninit<T>` fell to the generic element path, measured 2.5x slower on
/// a 4096x4096 `f64` transpose.
#[cfg(feature = "parallel")]
pub(crate) fn copy_permuted_uninit_with_active_policy<T: Copy + crate::MaybeSendSync + 'static>(
    dest: &mut crate::StridedViewMut<core::mem::MaybeUninit<T>>,
    src: &crate::StridedView<T>,
) -> crate::Result<()> {
    match permutation_route(dest.dims())? {
        PermutationRoute::Parallel => enter_target_pool(|| {
            with_permutation_copy_scheduler(|| strided_perm::copy_into_uninit_par(dest, src))
        }),
        PermutationRoute::BoundedMap => {
            crate::map_view::map_into(dest, src, core::mem::MaybeUninit::new)
        }
        PermutationRoute::Serial => strided_perm::copy_into_uninit(dest, src),
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
pub(crate) fn mapreduce_threaded<S: AsRef<[isize]>, F>(
    dims: &[usize],
    blocks: &[usize],
    strides_list: &[S],
    offsets: &[isize],
    costs: &[isize],
    nthreads: usize,
    spacing: isize,
    taskindex: usize,
    f: &F,
) -> Result<()>
where
    F: Fn(&[usize], &[usize], &[&[isize]], &[isize]) -> Result<()> + Sync,
{
    // The recursive splitter (rayon join) is compiled once; the leaf is `dyn`,
    // called once per leaf region. Borrowed per-operand stride slices are built
    // once here so the splitter and its leaves share them without owning a
    // `Vec<Vec<isize>>` (#306 P1).
    let refs: SVec<&[isize]> = strides_list.iter().map(|s| s.as_ref()).collect();
    let nthreads = crate::execution_policy::clamp_fanout_width(
        nthreads,
        crate::execution_policy::active_policy(),
    );
    mapreduce_threaded_dyn(
        dims, blocks, &refs, offsets, costs, nthreads, spacing, taskindex, f,
    )
}

/// The leaf callback of a threaded region.
#[cfg(feature = "parallel")]
type LeafFn<'a> = &'a (dyn Fn(&[usize], &[usize], &[&[isize]], &[isize]) -> Result<()> + Sync);

#[cfg(feature = "parallel")]
#[inline(never)]
fn mapreduce_threaded_dyn(
    dims: &[usize],
    blocks: &[usize],
    strides_list: &[&[isize]],
    offsets: &[isize],
    costs: &[isize],
    nthreads: usize,
    spacing: isize,
    taskindex: usize,
    f: LeafFn<'_>,
) -> Result<()> {
    // Normalize before the splitter: the guard below uses these tile sizes, so
    // a block that covers its whole dimension would otherwise suppress the
    // split. The shared layer applies this for every map, zip and fused caller.
    let blocks = scheduler_blocks(dims, blocks);
    ThreadedMapReduce {
        blocks: &blocks,
        strides_list,
        costs,
        spacing,
        policy: crate::execution_policy::active_policy(),
        operation: f,
    }
    .run(dims, offsets, nthreads, taskindex, false)
}

#[cfg(feature = "parallel")]
struct ThreadedMapReduce<'a> {
    blocks: &'a [usize],
    strides_list: &'a [&'a [isize]],
    costs: &'a [isize],
    spacing: isize,
    policy: crate::ExecutionPolicy,
    operation: LeafFn<'a>,
}

#[cfg(feature = "parallel")]
impl ThreadedMapReduce<'_> {
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
                let mut spaced: SVec<isize> = offsets.iter().copied().collect();
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
                let mut spaced: SVec<isize> = offsets.iter().copied().collect();
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
        let mut left_dims: SVec<usize> = dims.iter().copied().collect();
        left_dims[i] = ndi;

        // Right half: dims[i] = di - ndi, offsets advanced by ndi * stride[i]
        let mut right_dims: SVec<usize> = dims.iter().copied().collect();
        right_dims[i] = di - ndi;
        let mut right_offsets: SVec<isize> = offsets.iter().copied().collect();
        for (k, strides) in self.strides_list.iter().enumerate() {
            right_offsets[k] += ndi as isize * strides[i];
        }

        let left_offsets: SVec<isize> = offsets.iter().copied().collect();

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

/// Recursive dimension-splitting parallel fold.
///
/// Like [`mapreduce_threaded`], but the leaf returns a partial result and each
/// split combines the two children with `combine`. A full reduction therefore
/// never needs a worker-count-sized output-slot vector (#306 scope addition).
/// `f` must reduce the whole sub-region it is handed, using the supplied
/// `for_each_inner_block_with_offsets` traversal.
#[cfg(feature = "parallel")]
pub(crate) fn mapreduce_threaded_fold<S, F, C, R>(
    dims: &[usize],
    blocks: &[usize],
    strides_list: &[S],
    offsets: &[isize],
    costs: &[isize],
    nthreads: usize,
    f: &F,
    combine: &C,
) -> Result<R>
where
    S: AsRef<[isize]>,
    F: Fn(&[usize], &[usize], &[&[isize]], &[isize]) -> Result<R> + Sync,
    C: Fn(R, R) -> R + Sync,
    R: Send,
{
    let refs: SVec<&[isize]> = strides_list.iter().map(|s| s.as_ref()).collect();
    let policy = crate::execution_policy::active_policy();
    let nthreads = crate::execution_policy::clamp_fanout_width(nthreads, policy);
    // Normalize whole-dimension blocks before splitting, matching the
    // `mapreduce_threaded` path (#313): a block that covers its dimension would
    // otherwise suppress the split for a small whole tensor.
    let blocks = scheduler_blocks(dims, blocks);
    FoldMapReduce {
        blocks: &blocks,
        strides_list: &refs,
        costs,
        policy,
        f,
        combine,
    }
    .run(dims, offsets, nthreads, false)
}

#[cfg(feature = "parallel")]
struct FoldMapReduce<'a, F, C> {
    blocks: &'a [usize],
    strides_list: &'a [&'a [isize]],
    costs: &'a [isize],
    policy: crate::ExecutionPolicy,
    f: &'a F,
    combine: &'a C,
}

#[cfg(feature = "parallel")]
impl<F, C> FoldMapReduce<'_, F, C> {
    fn run<R: Send>(
        &self,
        dims: &[usize],
        offsets: &[isize],
        nthreads: usize,
        in_fanout: bool,
    ) -> Result<R>
    where
        F: Fn(&[usize], &[usize], &[&[isize]], &[isize]) -> Result<R> + Sync,
        C: Fn(R, R) -> R + Sync,
    {
        let total = crate::kernel::total_len(dims)?;

        // Base case: single thread or below threshold.
        if nthreads <= 1 || total <= MINTHREADLENGTH {
            return crate::execution_policy::with_owned_execution(self.policy, in_fanout, || {
                (self.f)(dims, self.blocks, self.strides_list, offsets)
            });
        }

        // Select split dimension: _lastargmax((dims .- 1) .* costs).
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

        // Guard: costs[i] == 0 || dims[i] <= min(blocks[i], 1024).
        if self.costs[i] == 0 || dims[i] <= self.blocks[i].min(1024) {
            return crate::execution_policy::with_owned_execution(self.policy, in_fanout, || {
                (self.f)(dims, self.blocks, self.strides_list, offsets)
            });
        }

        // Split dimension i in half and combine the two partial results.
        let di = dims[i];
        let ndi = di / 2;
        let nt_left = nthreads / 2;
        let nt_right = nthreads - nt_left;

        let mut left_dims: SVec<usize> = dims.iter().copied().collect();
        left_dims[i] = ndi;
        let mut right_dims: SVec<usize> = dims.iter().copied().collect();
        right_dims[i] = di - ndi;
        let mut right_offsets: SVec<isize> = offsets.iter().copied().collect();
        for (k, strides) in self.strides_list.iter().enumerate() {
            right_offsets[k] += ndi as isize * strides[i];
        }
        let left_offsets: SVec<isize> = offsets.iter().copied().collect();

        let (r1, r2) = join_with_policy(
            self.policy,
            || self.run(&left_dims, &left_offsets, nt_left, true),
            || self.run(&right_dims, &right_offsets, nt_right, true),
        );
        Ok((self.combine)(r1?, r2?))
    }
}

/// Execute the kernel on a sub-region defined by initial offsets.
///
/// Delegates to `for_each_inner_block_preordered` which directly calls
/// kernel functions with the initial offsets, avoiding redundant re-ordering
/// and per-callback `Vec` allocation.
#[cfg(feature = "parallel")]
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

#[cfg(all(test, feature = "parallel"))]
pub(crate) fn test_pool(threads: usize) -> std::sync::Arc<rayon::ThreadPool> {
    std::sync::Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap(),
    )
}

/// Run `operation` once under a serial [`crate::ExecContext`] and once under a
/// four-thread one inside a four-worker pool, and return both results.
///
/// The parallel run asserts that the repository gate actually opens for a
/// length above [`MINTHREADLENGTH`], so equivalence tests with such sizes
/// exercise the parallel branches.
#[cfg(all(test, feature = "parallel"))]
pub(crate) fn run_serial_and_parallel<R: Send>(operation: impl Fn() -> R + Sync) -> (R, R) {
    let serial = crate::ExecContext::serial().run(|| {
        assert_eq!(parallel_threads_for_len(MINTHREADLENGTH + 1), 1);
        operation()
    });
    let parallel = test_pool(4).install(|| {
        crate::ExecContext::max_threads(4).unwrap().run(|| {
            assert_eq!(parallel_threads_for_len(MINTHREADLENGTH + 1), 4);
            assert_eq!(parallel_threads_for_len(MINTHREADLENGTH), 1);
            operation()
        })
    });
    (serial, parallel)
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
