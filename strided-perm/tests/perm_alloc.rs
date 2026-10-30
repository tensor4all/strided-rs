//! Asserts the per-call allocation contract of the `strided-perm` copy entry
//! points (#306 item 7 / P4, #281).
//!
//! `copy_into` builds their `PermutePlan` (fused metadata,
//! flattened loop levels, grouped-tile offset tables) per call. The plan
//! metadata is inline up to [`INLINE_RANK`], so a rank-≤ 8 plan allocates
//! nothing. The grouped-tile offset tables are inline up to
//! `GROUP_OFFSETS_INLINE` entries; a larger group walks its remaining axes as
//! outer loops instead of growing the table, so a grouped plan allocates
//! nothing either (#337). The arms below cover rank 1..=8, including rank-8
//! grouped tiles larger than the table bound.
//!
//! This lives in its own integration-test binary because it installs a
//! counting global allocator. Serial windows read the calling thread's counter
//! (the plan is built there). Parallel windows read a process-wide counter so
//! allocations made on stolen Rayon tasks are counted on every worker, not
//! only the initiating one. The whole matrix runs in one test so no other test
//! in this binary can allocate concurrently with a process-wide window.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicUsize, Ordering};

use strided_perm::copy_into;
use strided_view::{StridedArray, StridedViewMut, INLINE_RANK};
use threadteam::Exec;
#[cfg(feature = "parallel")]
use threadteam::Pool;

struct CountingAllocator;

#[cfg(feature = "parallel")]
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static THREAD_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn record_allocation() {
    #[cfg(feature = "parallel")]
    ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
    // `try_with` fails only during thread teardown, when nothing is counted.
    let _ = THREAD_ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Allocations made by the calling thread while `run` executes.
fn count_thread_allocations(run: impl FnOnce()) -> usize {
    let before = THREAD_ALLOCATIONS.with(Cell::get);
    run();
    THREAD_ALLOCATIONS.with(Cell::get) - before
}

/// Allocations made by every thread while `run` executes.
///
/// Parallel windows use this so allocations in stolen Rayon tasks are counted.
#[cfg(feature = "parallel")]
fn count_process_allocations(run: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.load(Ordering::SeqCst);
    run();
    ALLOCATIONS.load(Ordering::SeqCst) - before
}

fn col_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut stride = 1isize;
    dims.iter()
        .map(|&dim| {
            let current = stride;
            stride *= dim as isize;
            current
        })
        .collect()
}

fn max_offset(dims: &[usize], strides: &[isize]) -> isize {
    dims.iter()
        .zip(strides)
        .map(|(&dim, &stride)| (dim as isize - 1) * stride)
        .sum()
}

/// Source array over a dense buffer with the given strides.
fn source_array(dims: &[usize], strides: &[isize]) -> StridedArray<f64> {
    let data: Vec<f64> = (0..=(max_offset(dims, strides) as usize))
        .map(|value| value as f64)
        .collect();
    StridedArray::from_parts(data, dims, strides, 0).unwrap()
}

/// The rank-8 extents-2 permutation with the first two axes swapped. This is a
/// dense permutation but still takes the grouped-transpose path, and both
/// groups fit the inline offset bound.
fn grouped_rank8_strides() -> (Vec<usize>, Vec<isize>) {
    let dims = vec![2usize; 8];
    let src = vec![2isize, 1, 4, 8, 16, 32, 64, 128];
    (dims, src)
}

/// Run `copy_into` for one source/destination layout pair and report
/// calling-thread allocations. Views and buffers are built outside the counted
/// window.
fn serial_allocations(dims: &[usize], src_strides: &[isize], dst_strides: &[isize]) -> usize {
    let src = source_array(dims, src_strides);
    let mut dst_data = vec![0.0f64; max_offset(dims, dst_strides) as usize + 1];
    let mut dst = StridedViewMut::new(&mut dst_data, dims, dst_strides, 0).unwrap();
    let src_view = src.view();
    count_thread_allocations(|| copy_into(&mut dst, &src_view, Exec::serial()).unwrap())
}

fn check_serial_contract() {
    for rank in 1..=INLINE_RANK {
        // Extent-2 layouts keep every grouped table within the inline bound,
        // so the whole plan is allocation-free.
        let dims = vec![2usize; rank];
        let dst_strides = col_major_strides(&dims);

        // Contiguous source: the memcpy fast path.
        assert_eq!(
            serial_allocations(&dims, &dst_strides, &dst_strides),
            0,
            "contiguous rank {rank}"
        );

        // Permuted source: a reversed-stride view over a col-major buffer.
        let mut permuted = dst_strides.clone();
        permuted.reverse();
        assert_eq!(
            serial_allocations(&dims, &permuted, &dst_strides),
            0,
            "permuted rank {rank}"
        );

        // Scattered source: rotate the destination stride order.
        if rank >= 2 {
            let mut scattered = dst_strides.clone();
            scattered.rotate_left(1);
            assert_eq!(
                serial_allocations(&dims, &scattered, &dst_strides),
                0,
                "scattered rank {rank}"
            );
        }
    }

    // Dense extent-2 permutation that takes the grouped-transpose path at rank
    // 8 with both groups inside the inline offset bound.
    let (dims, src_strides) = grouped_rank8_strides();
    let dst_strides = col_major_strides(&dims);
    assert_eq!(
        serial_allocations(&dims, &src_strides, &dst_strides),
        0,
        "grouped rank 8"
    );

    // A rank-9 plan spills its fused metadata instead of erroring.
    let dims9 = vec![2usize; INLINE_RANK + 1];
    let mut strides9 = col_major_strides(&dims9);
    strides9.reverse();
    assert!(
        serial_allocations(&dims9, &strides9, &col_major_strides(&dims9)) > 0,
        "rank 9 must spill"
    );

    // A grouped tile larger than the inline offset-table bound keeps its
    // table inline and walks the rest of the group as outer loops
    // (tensor4all/strided-rs#337). Source group along the 100-element first
    // axis (100 > 64 inline), while the destination keeps a 32-element
    // contiguous group.
    let dims = [100usize, 2, 2, 2, 2, 2, 2, 2];
    let src_strides = [1isize, 101, 1000, 1001, 1002, 1003, 1004, 1005];
    let dst_strides = [128isize, 1, 2, 4, 8, 16, 32, 64];
    assert_eq!(
        serial_allocations(&dims, &src_strides, &dst_strides),
        0,
        "a 100-entry grouped tile allocated"
    );

    // A rank-8 mixed-extent scattered layout whose destination-contiguous
    // group reaches 120 elements (> 64) after the planner has selected the
    // grouped tile.
    let dims = [2usize, 3, 4, 5, 6, 7, 8, 9];
    let dst_strides = col_major_strides(&dims);
    let mut src_strides = dst_strides.clone();
    src_strides.swap(0, 7);
    assert_eq!(
        serial_allocations(&dims, &src_strides, &dst_strides),
        0,
        "a 120-entry grouped tile allocated"
    );
}

/// Pooled windows. The process-wide counter sees every worker, so a per-call
/// allocation on a worker that took a task fails the assertion.
///
/// A fan-out itself costs the scheduler's per-task bookkeeping (one boxed job
/// per spawned task), which is not the permutation plan's. The windows below
/// therefore compare a pooled call against the same call with a budget-one
/// token, which runs inline on the calling thread: the plan contributes the same
/// allocations to both, and the difference is bounded by the number of tasks the
/// pool may start.
#[cfg(feature = "parallel")]
fn check_parallel_contract() {
    let pool = Pool::builder().threads(2).build().unwrap();
    // Let the workers finish their own start-up allocations before any counted
    // window opens: the process-wide counter sees them.
    Exec::new(&pool, 2).unwrap().fork(2, |_, _| {});
    std::thread::sleep(std::time::Duration::from_millis(100));

    let pooled_extra = |dst: &mut StridedViewMut<f64>, src: &strided_view::StridedView<f64>| {
        // Tokens are minted outside the windows: with the `ledger` feature a
        // root token registers itself, which allocates.
        let one = Exec::new(&pool, 1).unwrap();
        let two = Exec::new(&pool, 2).unwrap();
        let inline = count_thread_allocations(|| copy_into(dst, src, one).unwrap());
        let pooled = count_process_allocations(|| copy_into(dst, src, two).unwrap());
        (inline, pooled)
    };

    // Ranks 1..=INLINE_RANK with extent-2 layouts whose grouped tables stay
    // inside the inline bound.
    for rank in 1..=INLINE_RANK {
        let dims = vec![2usize; rank];
        let dst_strides = col_major_strides(&dims);
        let mut permuted = dst_strides.clone();
        permuted.reverse();
        let mut scattered = dst_strides.clone();
        if rank >= 2 {
            scattered.rotate_left(1);
        }
        for (label, src_strides) in [("permuted", permuted), ("scattered", scattered)] {
            let src = source_array(&dims, &src_strides);
            let mut dst_data = vec![0.0f64; max_offset(&dims, &dst_strides) as usize + 1];
            let mut dst = StridedViewMut::new(&mut dst_data, &dims, &dst_strides, 0).unwrap();
            let src_view = src.view();
            // Warm up outside the counted window.
            copy_into(&mut dst, &src_view, Exec::new(&pool, 2).unwrap()).unwrap();
            let (inline, pooled) = pooled_extra(&mut dst, &src_view);
            assert_eq!(inline, 0, "inline {label} rank {rank}");
            assert!(
                pooled <= MAX_SCHEDULER_ALLOCATIONS,
                "pooled {label} rank {rank}: {pooled} allocations"
            );
        }
    }

    // A layout large enough to split, with a transposing source so the plan is
    // non-trivial.
    let dims = [32usize, 32, 32];
    let src_strides = [1024isize, 1, 32];
    let src = source_array(&dims, &src_strides);
    let mut dst_data = vec![0.0f64; max_offset(&dims, &col_major_strides(&dims)) as usize + 1];
    let dst_strides = col_major_strides(&dims);
    let mut dst = StridedViewMut::new(&mut dst_data, &dims, &dst_strides, 0).unwrap();
    let src_view = src.view();
    for _ in 0..4 {
        copy_into(&mut dst, &src_view, Exec::new(&pool, 2).unwrap()).unwrap();
    }
    let (inline, pooled) = pooled_extra(&mut dst, &src_view);
    assert_eq!(inline, 0, "inline copy at rank 3");
    assert!(
        pooled <= MAX_SCHEDULER_ALLOCATIONS,
        "pooled copy at rank 3: {pooled} allocations"
    );

    // The rank-8 grouped layout.
    let (dims8, src8) = grouped_rank8_strides();
    let src8 = source_array(&dims8, &src8);
    let dst8_strides = col_major_strides(&dims8);
    let mut dst8_data = vec![0.0f64; max_offset(&dims8, &dst8_strides) as usize + 1];
    let mut dst8 = StridedViewMut::new(&mut dst8_data, &dims8, &dst8_strides, 0).unwrap();
    let src8_view = src8.view();
    copy_into(&mut dst8, &src8_view, Exec::new(&pool, 2).unwrap()).unwrap();
    let (inline, pooled) = pooled_extra(&mut dst8, &src8_view);
    assert_eq!(inline, 0, "inline grouped rank-8 copy");
    assert!(
        pooled <= MAX_SCHEDULER_ALLOCATIONS,
        "pooled grouped rank-8 copy: {pooled} allocations"
    );
}

/// Upper bound on the scheduler's own allocations for one two-task fan-out
/// (boxed jobs and the occasional queue block). The plan itself contributes
/// none: the inline windows above assert exactly zero.
#[cfg(feature = "parallel")]
const MAX_SCHEDULER_ALLOCATIONS: usize = 16;

#[test]
fn copy_entries_are_allocation_free_within_the_contract() {
    check_serial_contract();
    #[cfg(feature = "parallel")]
    check_parallel_contract();
}
