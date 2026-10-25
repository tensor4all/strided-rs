//! Steady-state entries allocate nothing (without the ledger).
#![cfg(all(feature = "parallel", not(feature = "ledger")))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use strided_exec::{Exec, Pool};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.realloc(p, l, n)
    }
}

#[global_allocator]
static A: Counting = Counting;

fn entries(pool: &Pool) {
    let sink = AtomicUsize::new(0);
    let grain = NonZeroUsize::new(1).unwrap();
    Exec::new(pool, 3).unwrap().fork(7, |i, _| {
        sink.fetch_add(i, Ordering::Relaxed);
    });
    Exec::new(pool, 3).unwrap().for_ranges(100, grain, |r| {
        sink.fetch_add(r.len(), Ordering::Relaxed);
    });
    let w = Exec::new(pool, 3).unwrap().external(|w| w.get());
    assert_eq!(w, 3);
    assert!(sink.load(Ordering::Relaxed) > 0);
}

/// Smallest allocation count over `rounds` windows of one call each, after warm-up.
///
/// From a plain thread every entry pushes one job into rayon's injector queue,
/// whose block allocation (one per 31 pushes) is amortized and outside this
/// crate; a per-call allocation shows up in every window, so the minimum
/// separates the two.
fn measure(rounds: usize, f: impl Fn()) -> usize {
    for _ in 0..20 {
        f();
    }
    (0..rounds)
        .map(|_| {
            let before = ALLOCS.load(Ordering::SeqCst);
            f();
            ALLOCS.load(Ordering::SeqCst) - before
        })
        .min()
        .unwrap_or(0)
}

// One test: the counter is process-global.
#[test]
fn fork_for_ranges_external_do_not_allocate() {
    let pool = Pool::builder().threads(3).build().unwrap();
    let plain = measure(10, || entries(&pool));
    assert_eq!(plain, 0, "plain-thread entries allocated");

    // Inside the pool: a worker of `pool` enters again, which installs nothing,
    // so every window is strict.
    let inside = AtomicUsize::new(usize::MAX);
    Exec::new(&pool, 3).unwrap().fork(2, |i, _| {
        if i == 0 {
            inside.store(measure(10, || entries(&pool)), Ordering::SeqCst);
        }
    });
    assert_eq!(
        inside.load(Ordering::SeqCst),
        0,
        "in-pool entries allocated"
    );
}
