//! Asserts that constructing a view from caller-owned metadata does not touch
//! the allocator, and that the two constructors differ exactly that way.
//!
//! This lives in its own integration-test binary because it installs a
//! counting global allocator; keeping it isolated avoids counting noise from
//! unrelated tests. The count is per calling thread, because the test harness
//! and runtime threads allocate at any time and a process-wide count is flaky.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use strided_view::{StridedView, StridedViewMut};

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

#[test]
fn borrowed_metadata_construction_allocates_nothing() {
    let data = vec![1.0f64; 24];
    let mut out = vec![0.0f64; 24];
    let dims = [2usize, 3, 4];
    let strides = [1isize, 2, 6];

    let allocations = count_allocations(|| {
        for _ in 0..1000 {
            let view = StridedView::<f64>::new(&data, &dims, &strides, 0).unwrap();
            let mut out_view = StridedViewMut::new(&mut out, &dims, &strides, 0).unwrap();
            // Clone, reborrow, and an operation that only composes the element
            // op keep the borrowed metadata.
            let clone = view.clone();
            let reborrow = out_view.as_view();
            let conj = clone.conj();
            assert_eq!(reborrow.dims(), &dims);
            assert_eq!(conj.dims(), &dims);
            out_view.set(&[0, 0, 0], view.get(&[0, 0, 0]));
        }
    });

    assert_eq!(allocations, 0, "borrowed construction must not allocate");
}

#[test]
fn owned_metadata_construction_allocates_the_metadata_only() {
    let data = vec![1.0f64; 24];
    let dims = vec![2usize, 3, 4];
    let strides = vec![1isize, 2, 6];

    // The `Vec`s are moved into the view, so the view itself allocates nothing
    // on top of them.
    let allocations = count_allocations(|| {
        let dims = dims.clone();
        let strides = strides.clone();
        let view = StridedView::<f64>::new_owned(&data, dims, strides, 0).unwrap();
        assert_eq!(view.dims(), &[2, 3, 4]);
    });

    assert_eq!(allocations, 2, "one allocation per owned metadata vector");
}

#[test]
fn derived_views_own_their_metadata_without_an_atomic() {
    let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let dims = [2usize, 3];
    let strides = [1isize, 2];

    let allocations = count_allocations(|| {
        let view = StridedView::<f64>::new(&data, &dims, &strides, 0).unwrap();
        let permuted = view.permute(&[1, 0]).unwrap();
        assert_eq!(permuted.dims(), &[3, 2]);
        assert_eq!(permuted.strides(), &[2, 1]);
    });

    // `permute` builds seen-marker, dims and strides vectors; the permuted view
    // owns the last two. No atomic refcount is touched either way.
    assert_eq!(allocations, 3, "permute's own metadata vectors");
}

#[test]
fn reborrowing_a_derived_mutable_view_shares_its_metadata() {
    let mut data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let dims = [2usize, 3];
    let strides = [1isize, 2];

    // `permute` returns a view that owns its metadata; the immutable reborrow
    // must borrow that metadata rather than copy it.
    let mut permuted = StridedViewMut::new(&mut data, &dims, &strides, 0)
        .unwrap()
        .permute(&[1, 0])
        .unwrap();

    let allocations = count_allocations(|| {
        let read = permuted.as_view();
        assert_eq!(read.dims(), &[3, 2]);
    });

    assert_eq!(allocations, 0, "a reborrow borrows the metadata");
}
