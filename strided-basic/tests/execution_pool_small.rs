//! A small operation inside a target-pool scope stays on the calling thread and
//! never installs the pool. This binary is independent so the placement
//! observation cannot be disturbed by another test's Rayon state.

#![cfg(feature = "parallel")]

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};

use strided_basic::execution::rayon_threads;
use strided_basic::{map_into, with_execution_pool, StridedArray};

#[test]
fn small_operation_does_not_enter_the_target_pool() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let source = StridedArray::<usize>::from_fn_col_major(&[8, 8], |index| index[0] + 8 * index[1]);
    let mut destination = StridedArray::<usize>::col_major(&[8, 8]);
    let caller = std::thread::current().id();

    let left_caller = AtomicBool::new(false);
    let saw_pool_worker = AtomicBool::new(false);

    with_execution_pool(&pool, NonZeroUsize::new(4).unwrap(), || {
        // Width queries report the target pool, not the ambient one.
        assert_eq!(rayon_threads(), 4);
        assert_eq!(rayon::current_thread_index(), None);

        map_into(&mut destination.view_mut(), &source.view(), |value| {
            if std::thread::current().id() != caller {
                left_caller.store(true, Ordering::SeqCst);
            }
            if pool.current_thread_index().is_some() {
                saw_pool_worker.store(true, Ordering::SeqCst);
            }
            value + 1
        })
        .unwrap();
    });

    // The scope itself ran on the caller, and the small work never installed.
    assert_eq!(std::thread::current().id(), caller);
    assert!(!left_caller.load(Ordering::SeqCst));
    assert!(!saw_pool_worker.load(Ordering::SeqCst));
    for index in 0..64 {
        assert_eq!(destination.data()[index], index + 1);
    }
}

#[test]
fn small_operation_inside_a_worker_of_the_target_pool_runs_in_place() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let source = StridedArray::<usize>::from_fn_col_major(&[8, 8], |index| index[0] + 8 * index[1]);
    let mut destination = StridedArray::<usize>::col_major(&[8, 8]);

    pool.install(|| {
        let worker = std::thread::current().id();
        let left_worker = AtomicBool::new(false);
        with_execution_pool(&pool, NonZeroUsize::new(4).unwrap(), || {
            map_into(&mut destination.view_mut(), &source.view(), |value| {
                if std::thread::current().id() != worker {
                    left_worker.store(true, Ordering::SeqCst);
                }
                value + 1
            })
            .unwrap();
        });
        assert!(!left_worker.load(Ordering::SeqCst));
    });

    for index in 0..64 {
        assert_eq!(destination.data()[index], index + 1);
    }
}
