//! Counting allocator shared by the `strided-fused` allocation tests.
//!
//! Each test binary includes this module and uses a subset of it (the serial
//! tests never touch `count_global_allocations`, the process-wide one), so
//! unused items are expected here.
#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) struct CountingAllocator;

pub(crate) static GLOBAL_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    pub(crate) static THREAD_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    pub(crate) static THREAD_FREES: Cell<usize> = const { Cell::new(0) };
}

pub(crate) fn record_allocation() {
    GLOBAL_ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
    // `try_with` fails only during thread teardown, when nothing is counted.
    let _ = THREAD_ALLOCATIONS.try_with(|count| count.set(count.get().wrapping_add(1)));
}

pub(crate) fn record_deallocation() {
    let _ = THREAD_FREES.try_with(|count| count.set(count.get().wrapping_add(1)));
}

// SAFETY: each method forwards its arguments unchanged to `System`, which meets the `GlobalAlloc`
// contract, and the counting side effects neither allocate nor unwind. `fetch_add` on the atomic
// wraps on overflow. The thread-local `Cell`s are `const`-initialized and have no destructor, so
// `try_with` registers nothing and allocates nothing, and it returns `Err` during teardown
// instead of panicking. Their updates use `wrapping_add`, so they cannot panic either, even with
// overflow checks enabled; a plain `+ 1` would panic at `usize::MAX`, and a panic unwinding out
// of a `GlobalAlloc` method violates its contract.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        // SAFETY: forwards `alloc`'s caller contract (a non-zero-size `layout`) unchanged to
        // `System`, which allocated every block this allocator hands out.
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record_deallocation();
        // SAFETY: forwards `dealloc`'s caller contract (`ptr` allocated by this allocator with
        // `layout`) unchanged to `System`, which allocated every block this allocator hands out.
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        // SAFETY: forwards `realloc`'s caller contract (`ptr` from this allocator with `layout`,
        // a valid non-zero `new_size`) unchanged to `System`, which allocated every block this
        // allocator hands out.
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
pub(crate) static GLOBAL: CountingAllocator = CountingAllocator;

pub(crate) fn count_allocations(run: impl FnOnce()) -> usize {
    let before = THREAD_ALLOCATIONS.with(Cell::get);
    run();
    THREAD_ALLOCATIONS.with(Cell::get).wrapping_sub(before)
}

/// Net live allocations (allocations minus frees) made by the calling thread
/// while `run` executes. Used to catch a dropped-metadata leak.
pub(crate) fn count_net_allocations(run: impl FnOnce()) -> isize {
    let before_alloc = THREAD_ALLOCATIONS.with(Cell::get);
    let before_free = THREAD_FREES.with(Cell::get);
    run();
    let after_alloc = THREAD_ALLOCATIONS.with(Cell::get);
    let after_free = THREAD_FREES.with(Cell::get);
    // Wrap in `usize` before the cast, so a window that crosses the wrap still
    // yields the small true delta (1) instead of a huge or negative one.
    let alloc_delta = after_alloc.wrapping_sub(before_alloc) as isize;
    let free_delta = after_free.wrapping_sub(before_free) as isize;
    alloc_delta.wrapping_sub(free_delta)
}

/// Steady-state allocations made by every thread while `run` executes.
///
/// The window runs three times and reports the minimum. A pool worker the
/// warm-up never scheduled pays one-time first-touch costs (thread-local
/// registration allocates on macOS) the first time it picks up a split, which
/// is not a per-call allocation; a per-call allocation shows up in every round.
/// (#321)
#[cfg(feature = "parallel")]
pub(crate) fn count_global_allocations(mut run: impl FnMut()) -> usize {
    (0..3)
        .map(|_| {
            let before = GLOBAL_ALLOCATIONS.load(Ordering::SeqCst);
            run();
            GLOBAL_ALLOCATIONS
                .load(Ordering::SeqCst)
                .wrapping_sub(before)
        })
        .min()
        .unwrap_or(0)
}
