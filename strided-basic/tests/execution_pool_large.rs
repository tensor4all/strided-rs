//! A large operation inside a target-pool scope fans out on that pool's workers
//! and never on the ambient one.

#![cfg(feature = "parallel")]

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use strided_basic::{map_into, with_execution_pool, StridedArray};

const LEN: usize = 1 << 17;
const _: () = assert!(LEN > (1 << 15));

#[test]
fn large_operation_runs_on_target_pool_workers() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let source = StridedArray::<usize>::from_fn_col_major(&[LEN], |index| index[0]);
    let mut destination = StridedArray::<usize>::col_major(&[LEN]);
    let caller = std::thread::current().id();

    let saw_worker = AtomicBool::new(false);
    let ran_on_caller = AtomicBool::new(false);
    let visited = AtomicUsize::new(0);

    with_execution_pool(&pool, NonZeroUsize::new(4).unwrap(), || {
        map_into(&mut destination.view_mut(), &source.view(), |value| {
            if pool.current_thread_index().is_some() {
                saw_worker.store(true, Ordering::SeqCst);
            }
            if std::thread::current().id() == caller {
                ran_on_caller.store(true, Ordering::SeqCst);
            }
            visited.fetch_add(1, Ordering::Relaxed);
            value + 1
        })
        .unwrap();
    });

    assert!(saw_worker.load(Ordering::SeqCst));
    assert!(!ran_on_caller.load(Ordering::SeqCst));
    assert_eq!(visited.load(Ordering::Relaxed), LEN);
    assert_eq!(destination.get(&[0]), 1);
    assert_eq!(destination.get(&[LEN - 1]), LEN);
}

#[test]
fn capped_target_pool_limits_concurrent_workers() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let source = StridedArray::<usize>::from_fn_col_major(&[LEN], |index| index[0]);
    let mut destination = StridedArray::<usize>::col_major(&[LEN]);

    let active = AtomicUsize::new(0);
    let max_active = AtomicUsize::new(0);
    let wrong_pool = AtomicBool::new(false);

    with_execution_pool(&pool, NonZeroUsize::new(2).unwrap(), || {
        map_into(&mut destination.view_mut(), &source.view(), |value| {
            if pool.current_thread_index().is_none() {
                wrong_pool.store(true, Ordering::SeqCst);
            }
            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
            max_active.fetch_max(now, Ordering::SeqCst);
            std::hint::spin_loop();
            active.fetch_sub(1, Ordering::SeqCst);
            value
        })
        .unwrap();
    });

    assert!(!wrong_pool.load(Ordering::SeqCst));
    assert!(max_active.load(Ordering::SeqCst) <= 2);
    assert_eq!(destination.get(&[LEN - 1]), LEN - 1);
}
