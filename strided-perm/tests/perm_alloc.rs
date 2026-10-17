//! Asserts the per-call allocation contract of the `strided-perm` copy entry
//! points (#306 item 7 / P4, #281).
//!
//! `copy_into` and `copy_into_par` build their `PermutePlan` (fused metadata,
//! flattened loop levels, grouped-tile offset tables) per call. The plan
//! metadata is inline up to [`INLINE_RANK`], so a rank-≤ 8 plan allocates
//! nothing. The grouped-tile offset tables are sized by the product of a
//! tile's extents, not by rank: they are inline up to `GROUP_OFFSETS_INLINE`
//! and spill to the heap above it (P1/P4 allow allocation above the bound).
//! The arms below cover the inline case at rank 1..=8 and record the rank-8
//! grouped cases whose tables spill.
//!
//! This lives in its own integration-test binary because it installs a
//! counting global allocator. Every counting window reads the calling thread's
//! counter: the serial windows run on the test thread, and each parallel
//! window runs inside a dedicated 2-worker pool and reads the worker thread's
//! counter (the plan is built there). A process-global counter was flaky
//! because an unrelated thread can allocate inside a short window.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicUsize, Ordering};

use strided_perm::copy_into;
#[cfg(feature = "parallel")]
use strided_perm::copy_into_par;
use strided_view::{StridedArray, StridedViewMut, INLINE_RANK};

struct CountingAllocator;

thread_local! {
    static THREAD_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn record_allocation() {
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
fn count_allocations(run: impl FnOnce()) -> usize {
    let before = THREAD_ALLOCATIONS.with(Cell::get);
    run();
    THREAD_ALLOCATIONS.with(Cell::get) - before
}

/// Allocations made by the pool worker thread that `run` executes on.
///
/// `run` is handed to the pool, so the plan build inside `copy_into_par` is
/// counted on the worker, not on the test thread.
#[cfg(feature = "parallel")]
fn count_worker_allocations(pool: &rayon::ThreadPool, run: impl FnOnce() + Send) -> usize {
    let result = AtomicUsize::new(0);
    pool.install(|| {
        let before = THREAD_ALLOCATIONS.with(Cell::get);
        run();
        result.store(
            THREAD_ALLOCATIONS.with(Cell::get) - before,
            Ordering::SeqCst,
        );
    });
    result.load(Ordering::SeqCst)
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
    count_allocations(|| copy_into(&mut dst, &src_view).unwrap())
}

#[test]
fn copy_into_is_allocation_free_up_to_inline_rank() {
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
}

/// A grouped tile whose offset table is larger than the inline bound spills
/// that table to the heap. The tile algorithm is unchanged: the long
/// source-contiguous group is still selected, it just needs a heap table.
#[test]
fn grouped_offset_tables_spill_above_the_inline_bound() {
    // Source group along the 100-element first axis (100 > 64 inline), while
    // the destination keeps a 32-element contiguous group. `dim_a = 0`,
    // `dim_b = 1`, `legacy_area = 200 < 256`, so the grouped path is chosen.
    // The source strides are deliberately non-doubling so bilateral fusion
    // does not collapse the group_b axes into `dim_b` and enlarge
    // `legacy_area` past the grouped threshold.
    let dims = [100usize, 2, 2, 2, 2, 2, 2, 2];
    let src_strides = [1isize, 101, 1000, 1001, 1002, 1003, 1004, 1005];
    let dst_strides = [128isize, 1, 2, 4, 8, 16, 32, 64];
    let allocations = serial_allocations(&dims, &src_strides, &dst_strides);
    assert!(
        allocations > 0,
        "a 100-entry grouped table must spill, got {allocations}"
    );
    println!("spill case [100,2x7] allocations = {allocations}");

    // A rank-8 mixed-extent scattered layout whose destination-contiguous
    // group reaches 120 elements (> 64) after the planner has selected the
    // grouped tile; its table spills too.
    let dims = [2usize, 3, 4, 5, 6, 7, 8, 9];
    let dst_strides = col_major_strides(&dims);
    let mut src_strides = dst_strides.clone();
    src_strides.swap(0, 7);
    let allocations = serial_allocations(&dims, &src_strides, &dst_strides);
    assert!(
        allocations > 0,
        "a 120-entry grouped table must spill, got {allocations}"
    );
    println!("spill case [2,3,..,9] swapped allocations = {allocations}");
}

#[cfg(feature = "parallel")]
#[test]
fn copy_into_par_is_allocation_free_in_a_two_worker_pool() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();

    // Ranks 1..=INLINE_RANK with extent-2 layouts whose grouped tables stay
    // inside the inline bound. The totals are below the threading threshold,
    // so the parallel entry falls back to the serial engine after building the
    // plan; the plan build still runs on a pool worker.
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
            pool.install(|| copy_into_par(&mut dst, &src_view).unwrap());
            let allocations = count_worker_allocations(&pool, || {
                copy_into_par(&mut dst, &src_view).unwrap();
            });
            assert_eq!(allocations, 0, "parallel {label} rank {rank}");
        }
    }

    // A layout large enough to enter the parallel branch (>= 32768 elements),
    // with a transposing source so the plan is non-trivial.
    let dims = [32usize, 32, 32];
    let src_strides = [1024isize, 1, 32];
    let src = source_array(&dims, &src_strides);
    let mut dst_data = vec![0.0f64; max_offset(&dims, &col_major_strides(&dims)) as usize + 1];
    let dst_strides = col_major_strides(&dims);
    let mut dst = StridedViewMut::new(&mut dst_data, &dims, &dst_strides, 0).unwrap();
    let src_view = src.view();

    // Warm the pool and both one-time codepaths outside the counted window.
    for _ in 0..4 {
        pool.install(|| copy_into_par(&mut dst, &src_view).unwrap());
    }
    let allocations = count_worker_allocations(&pool, || {
        copy_into_par(&mut dst, &src_view).unwrap();
    });
    assert_eq!(
        allocations, 0,
        "parallel copy at rank 3 must not allocate in a 2-worker pool"
    );

    // The rank-8 grouped layout also stays allocation-free in the pool.
    let (dims8, src8) = grouped_rank8_strides();
    let src8 = source_array(&dims8, &src8);
    let dst8_strides = col_major_strides(&dims8);
    let mut dst8_data = vec![0.0f64; max_offset(&dims8, &dst8_strides) as usize + 1];
    let mut dst8 = StridedViewMut::new(&mut dst8_data, &dims8, &dst8_strides, 0).unwrap();
    let src8_view = src8.view();
    pool.install(|| copy_into_par(&mut dst8, &src8_view).unwrap());
    let allocations = count_worker_allocations(&pool, || {
        copy_into_par(&mut dst8, &src8_view).unwrap();
    });
    assert_eq!(
        allocations, 0,
        "parallel grouped rank-8 copy must not allocate in a 2-worker pool"
    );
}
