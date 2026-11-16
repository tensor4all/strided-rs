//! Zero-allocation counter for the sorted-run reduction's execute path
//! (tensor4all/strided-rs#479).
//!
//! `RunIndex::from_dest` and `RunSumPlan::compile` allocate at topology time,
//! outside every counted window. `execute_into` / `execute_acc` must not
//! allocate: the payload walk's offset and stride collections hold one entry per
//! operand (two here), and payload rank 0..=8 dispatches to the rank-specialized
//! kernels. Payload rank 9 and above takes `kernel_nd_inner_iterative`, whose
//! odometer is a `Vec` by contract, and is not gated here.
//!
//! Both windows count the calling thread only. That is the whole execute path
//! here: the sizes below stay far under the cost gate (at most 16k elements,
//! ~2us of `Reduce` work against a microsecond-scale dispatch), so the leaves
//! never run on a pool worker. A *forking* window is deliberately out of scope:
//! in a debug build threadteam's token instrumentation allocates on the
//! dispatching side of every `fork`, which is not a per-call allocation of this
//! kernel.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use strided_basic::{Exec, Pool, RawStridedMut, RawStridedRef, RunIndex, RunSumPlan};

thread_local! {
    static THREAD_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn record_allocation() {
    let _ = THREAD_ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
}

struct CountingAllocator;

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

/// Allocations made by the calling thread while `run` executes, so a
/// concurrently idle pool cannot leak background allocations into the count.
fn count_thread_allocations(run: impl FnOnce()) -> usize {
    let before = THREAD_ALLOCATIONS.with(Cell::get);
    run();
    THREAD_ALLOCATIONS.with(Cell::get) - before
}

/// Extent of a layout whose axes are deliberately not contiguous, so the payload
/// walk keeps one axis per declared payload axis instead of fusing them.
fn gapped_len(dims: &[usize]) -> usize {
    let mut span = 1usize;
    for (axis, &dim) in dims.iter().enumerate() {
        span += (dim - 1) * 3usize.pow(axis as u32 + 1);
    }
    span
}

fn gapped_strides(dims: &[usize], row_stride: isize) -> Vec<isize> {
    let mut strides = vec![row_stride];
    for axis in 0..dims.len() {
        strides.push(3isize.pow(axis as u32 + 1));
    }
    strides
}

fn exercise(pool: Option<&Pool>, budget: usize, payload_rank: usize) {
    let roots = 64usize;
    let runs_count = 5usize;
    let payload: Vec<usize> = vec![2; payload_rank];
    let span = gapped_len(&payload);

    let mut src_dims = vec![roots];
    src_dims.extend_from_slice(&payload);
    let mut dest_dims = vec![runs_count];
    dest_dims.extend_from_slice(&payload);
    let src_strides = gapped_strides(&payload, span as isize);
    let dest_strides = gapped_strides(&payload, span as isize);

    let dest_of_row: Vec<i32> = (0..roots)
        .map(|row| ((row * 7) % runs_count) as i32)
        .collect();
    let src: Vec<f64> = (0..roots * span).map(|index| index as f64 * 0.25).collect();
    let mut out = vec![f64::NAN; runs_count * span];

    let runs = RunIndex::from_dest(&dest_of_row, runs_count).unwrap();
    let plan =
        RunSumPlan::compile(&src_dims, &src_strides, &dest_dims, &dest_strides, &runs).unwrap();
    let src_ref = RawStridedRef::new(&src, &src_dims, &src_strides, 0).unwrap();

    // The token is minted outside the counted windows: threadteam's debug-build
    // token carries `Track` bookkeeping that allocates once, and minting one
    // token per region is the caller's job, not the kernel's.
    let mut exec = match pool {
        Some(pool) => Exec::new(pool, budget).unwrap(),
        None => Exec::serial(),
    };

    // `_into` and `_acc` take the same path; count each on its own window.
    let into = count_thread_allocations(|| {
        let mut dest = RawStridedMut::new(&mut out, &dest_dims, &dest_strides, 0).unwrap();
        plan.execute_into(&mut dest, &src_ref, &runs, exec.reborrow())
            .unwrap();
    });
    assert_eq!(
        into,
        0,
        "execute_into allocated {into} times (pool: {}, payload rank {payload_rank})",
        pool.is_some()
    );

    let acc = count_thread_allocations(|| {
        let mut dest = RawStridedMut::new(&mut out, &dest_dims, &dest_strides, 0).unwrap();
        plan.execute_acc(&mut dest, &src_ref, &runs, exec.reborrow())
            .unwrap();
    });
    assert_eq!(
        acc,
        0,
        "execute_acc allocated {acc} times (pool: {}, payload rank {payload_rank})",
        pool.is_some()
    );
}

#[test]
fn execute_allocates_nothing() {
    for payload_rank in 0..=8 {
        exercise(None, 1, payload_rank);
    }
    let pool = Pool::builder().threads(2).build().unwrap();
    for payload_rank in 0..=8 {
        exercise(Some(&pool), 2, payload_rank);
    }
}
