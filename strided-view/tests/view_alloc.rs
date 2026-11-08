//! Asserts the inline-metadata allocation contract of `strided-view` (#306 P3/P5).
//!
//! View construction, clones, metadata transforms, the stride/axis helpers and
//! `StridedArray` metadata are allocation-free up to [`INLINE_RANK`]; a higher
//! rank spills instead of erroring.
//!
//! This lives in its own integration-test binary because it installs a
//! counting global allocator; keeping it isolated avoids counting noise from
//! unrelated tests. The count is per calling thread, because the test harness
//! and runtime threads allocate at any time and a process-wide count is flaky.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use strided_view::auxiliary::index_order;
use strided_view::{
    col_major_strides, row_major_strides, StrideVec, StridedArray, StridedView, StridedViewMut,
    INLINE_RANK,
};

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

fn dims_for(rank: usize) -> Vec<usize> {
    (0..rank).map(|i| i + 2).collect()
}

/// Rank `rank` contiguous layout and a data slice large enough for it.
fn contiguous_case(rank: usize) -> (Vec<usize>, StrideVec, Vec<f64>) {
    let dims = dims_for(rank);
    let strides = col_major_strides(&dims);
    let total: usize = dims.iter().product();
    let data = vec![1.0f64; total.max(1)];
    (dims, strides, data)
}

#[test]
fn inline_rank_is_eight() {
    assert_eq!(INLINE_RANK, 8);
}

#[test]
fn construction_allocates_nothing_up_to_inline_rank() {
    for rank in 1..=INLINE_RANK {
        let (dims, strides, data) = contiguous_case(rank);
        let allocations = count_allocations(|| {
            let view = StridedView::<f64>::new(black_box(&data), &dims, &strides, 0).unwrap();
            black_box(view.dims().len());
        });
        assert_eq!(allocations, 0, "StridedView::new rank {rank} allocated");

        let mut data_mut = data.clone();
        let allocations = count_allocations(|| {
            let view =
                StridedViewMut::<f64>::new(black_box(&mut data_mut), &dims, &strides, 0).unwrap();
            black_box(view.dims().len());
        });
        assert_eq!(allocations, 0, "StridedViewMut::new rank {rank} allocated");
    }
}

#[test]
fn clone_conj_and_derived_views_allocate_nothing_up_to_inline_rank() {
    for rank in 1..=INLINE_RANK {
        let (dims, strides, data) = contiguous_case(rank);
        let view = StridedView::<f64>::new(&data, &dims, &strides, 0).unwrap();
        let perm: Vec<usize> = (0..rank).rev().collect();

        let allocations = count_allocations(|| {
            black_box(view.clone());
        });
        assert_eq!(allocations, 0, "clone rank {rank} allocated");

        let allocations = count_allocations(|| {
            black_box(view.conj());
        });
        assert_eq!(allocations, 0, "conj rank {rank} allocated");

        let allocations = count_allocations(|| {
            black_box(view.permute(&perm).unwrap());
        });
        assert_eq!(allocations, 0, "permute rank {rank} allocated");

        // Identity broadcast: every target dim equals the source dim.
        let allocations = count_allocations(|| {
            black_box(view.broadcast(&dims).unwrap());
        });
        assert_eq!(allocations, 0, "broadcast rank {rank} allocated");

        // `transpose_2d` is the rank-2 special case of `permute`.
        if rank == 2 {
            let (dims2, strides2, data2) = contiguous_case(2);
            let view2 = StridedView::<f64>::new(&data2, &dims2, &strides2, 0).unwrap();
            let allocations = count_allocations(|| {
                black_box(view2.transpose_2d().unwrap());
            });
            assert_eq!(allocations, 0, "transpose_2d allocated");
        }

        // A diagonal view needs at least two axes.
        if rank >= 2 {
            let allocations = count_allocations(|| {
                black_box(view.diagonal_view(&[(0, 1)]).unwrap());
            });
            assert_eq!(allocations, 0, "diagonal_view rank {rank} allocated");
        }
    }
}

#[test]
fn diagonal_view_with_repeated_pairs_allocates_nothing() {
    // Nine copies of the same removal axis must not grow the axis list past
    // the inline bound: insertion is unique, so a rank-2 view still allocates
    // nothing even though the pair list is longer than INLINE_RANK. The layout
    // is the review's reproducer (dims [2, 2], strides [0, 0]).
    let dims = [2usize, 2];
    let strides = [0isize, 0];
    let data = [0.0f64; 1];
    let view = StridedView::<f64>::new(&data, &dims, &strides, 0).unwrap();
    let pairs = [(0usize, 1usize); INLINE_RANK + 1];
    let allocations = count_allocations(|| {
        let diagonal = view.diagonal_view(&pairs).unwrap();
        assert_eq!(diagonal.dims(), &[2]);
        black_box(diagonal.dims().len());
    });
    assert_eq!(allocations, 0, "duplicate diagonal pairs allocated");
}

#[test]
fn mutable_permute_and_reborrow_allocate_nothing_up_to_inline_rank() {
    for rank in 1..=INLINE_RANK {
        let (dims, strides, data) = contiguous_case(rank);
        let perm: Vec<usize> = (0..rank).rev().collect();
        let mut data_mut = data.clone();

        let allocations = count_allocations(|| {
            let view = StridedViewMut::<f64>::new(&mut data_mut, &dims, &strides, 0)
                .unwrap()
                .permute(&perm)
                .unwrap();
            black_box(view.dims().len());
        });
        assert_eq!(
            allocations, 0,
            "StridedViewMut::permute rank {rank} allocated"
        );

        let view = StridedViewMut::<f64>::new(&mut data_mut, &dims, &strides, 0).unwrap();
        let allocations = count_allocations(|| {
            black_box(view.as_view());
        });
        assert_eq!(allocations, 0, "as_view rank {rank} allocated");
    }
}

#[test]
fn stride_and_axis_helpers_are_inline_up_to_inline_rank() {
    for rank in 1..=INLINE_RANK {
        let dims = dims_for(rank);
        let mut strides: Vec<isize> = Vec::new();
        let allocations = count_allocations(|| {
            black_box(col_major_strides(&dims));
            black_box(row_major_strides(&dims));
        });
        assert_eq!(allocations, 0, "stride helpers rank {rank} allocated");
        strides.extend(col_major_strides(&dims));
        let allocations = count_allocations(|| {
            black_box(index_order(&strides));
        });
        assert_eq!(allocations, 0, "index_order rank {rank} allocated");
    }
}

#[test]
fn strided_array_metadata_is_inline() {
    for rank in 1..=INLINE_RANK {
        let (dims, strides, data) = contiguous_case(rank);
        // Only the moved-in data buffer is owned; `from_parts` copies metadata
        // inline.
        let allocations = count_allocations(|| {
            let array =
                StridedArray::from_parts(black_box(data.clone()), &dims, &strides, 0).unwrap();
            black_box(array.dims().len());
        });
        // `data.clone()` inside the closure allocates the element buffer; that
        // is user data, not library metadata. Subtracting it leaves zero.
        assert_eq!(
            allocations, 1,
            "from_parts rank {rank} allocated beyond the element buffer"
        );

        let mut array = StridedArray::from_parts(data.clone(), &dims, &strides, 0).unwrap();
        let allocations = count_allocations(|| {
            black_box(array.view());
        });
        assert_eq!(allocations, 0, "StridedArray::view rank {rank} allocated");
        let allocations = count_allocations(|| {
            black_box(array.view_mut());
        });
        assert_eq!(
            allocations, 0,
            "StridedArray::view_mut rank {rank} allocated"
        );
    }
}

#[test]
fn strided_array_permuted_metadata_is_inline() {
    for rank in 1..=INLINE_RANK {
        let (dims, strides, data) = contiguous_case(rank);
        let perm: Vec<usize> = (0..rank).rev().collect();
        let owned = data.clone();
        let allocations = count_allocations(move || {
            let array = StridedArray::from_parts(owned, &dims, &strides, 0).unwrap();
            let permuted = array.permuted(&perm).unwrap();
            black_box(permuted.dims().len());
        });
        assert_eq!(
            allocations, 0,
            "StridedArray::permuted rank {rank} allocated"
        );
    }
}

#[test]
fn col_major_allocates_only_its_element_buffer_up_to_inline_rank() {
    for rank in 1..=INLINE_RANK {
        let dims = dims_for(rank);
        let allocations = count_allocations(|| {
            black_box(StridedArray::<f64>::col_major(&dims));
        });
        assert_eq!(
            allocations, 1,
            "StridedArray::col_major rank {rank} allocated metadata"
        );
    }
}

#[test]
fn above_inline_rank_spills_gracefully() {
    let rank = INLINE_RANK + 1;
    let (dims, strides, data) = contiguous_case(rank);

    // `new` boxes `SpilledMeta` plus one slice per metadata vector, and each
    // spills at rank 9.
    let allocations = count_allocations(|| {
        let view = StridedView::<f64>::new(&data, &dims, &strides, 0).unwrap();
        black_box(view.dims().len());
    });
    assert_eq!(allocations, 3, "rank 9 new should spill the boxed metadata");

    let view = StridedView::<f64>::new(&data, &dims, &strides, 0).unwrap();
    let allocations = count_allocations(|| {
        black_box(view.clone());
    });
    assert_eq!(
        allocations, 3,
        "rank 9 clone should spill the boxed metadata"
    );

    let perm: Vec<usize> = (0..rank).rev().collect();
    let allocations = count_allocations(|| {
        black_box(view.permute(&perm).unwrap());
    });
    assert!(
        allocations >= 2,
        "rank 9 permute should spill, got {allocations} allocations"
    );

    // P5 helpers spill above the inline capacity rather than erroring.
    let allocations = count_allocations(|| {
        black_box(col_major_strides(&dims));
    });
    assert_eq!(allocations, 1, "rank 9 col_major_strides should spill once");
    let allocations = count_allocations(|| {
        black_box(index_order(&strides));
    });
    assert_eq!(allocations, 1, "rank 9 index_order should spill once");
}
