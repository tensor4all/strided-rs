//! A small operation handed a token minted from a target pool stays on the
//! calling thread and never installs the pool. This binary is independent so the
//! placement observation cannot be disturbed by another test's state.

#![cfg(feature = "parallel")]

use std::sync::atomic::{AtomicBool, Ordering};

use strided_basic::{map_into, Exec, Pool, StridedArray};

const WORKER_PREFIX: &str = "strided-small-worker";

fn named_pool() -> Pool {
    Pool::builder()
        .threads(4)
        .thread_name(|i| format!("{WORKER_PREFIX}-{i}"))
        .build()
        .unwrap()
}

fn on_pool_worker() -> bool {
    std::thread::current()
        .name()
        .is_some_and(|name| name.starts_with(WORKER_PREFIX))
}

#[test]
fn small_operation_does_not_enter_the_target_pool() {
    let pool = named_pool();
    let source = StridedArray::<usize>::from_fn_col_major(&[8, 8], |index| index[0] + 8 * index[1]);
    let mut destination = StridedArray::<usize>::col_major(&[8, 8]);
    let caller = std::thread::current().id();

    let left_caller = AtomicBool::new(false);
    let saw_pool_worker = AtomicBool::new(false);

    // The token is minted outside the measured operation.
    let exec = Exec::new(&pool, 4).unwrap();
    let root = exec.ledger_root().unwrap();
    map_into(
        &mut destination.view_mut(),
        &source.view(),
        |value| {
            if std::thread::current().id() != caller {
                left_caller.store(true, Ordering::SeqCst);
            }
            if on_pool_worker() {
                saw_pool_worker.store(true, Ordering::SeqCst);
            }
            value + 1
        },
        exec,
    )
    .unwrap();

    // The small work ran on the caller and never installed into the pool.
    assert!(!left_caller.load(Ordering::SeqCst));
    assert!(!saw_pool_worker.load(Ordering::SeqCst));
    assert_eq!(strided_exec::ledger::installs(root), 0);
    for index in 0..64 {
        assert_eq!(destination.data()[index], index + 1);
    }
}

#[test]
fn small_operation_inside_a_worker_of_the_target_pool_runs_in_place() {
    let pool = named_pool();
    let source = StridedArray::<usize>::from_fn_col_major(&[8, 8], |index| index[0] + 8 * index[1]);

    // Two jobs of a four-slot token: each runs on a pool worker, and the small
    // map inside a job must stay on that same worker.
    let exec = Exec::new(&pool, 4).unwrap();
    let root = exec.ledger_root().unwrap();
    let left_worker = AtomicBool::new(false);
    let saw_worker = AtomicBool::new(false);
    exec.fork(2, |_job, child| {
        assert!(
            on_pool_worker(),
            "a fork job of a pooled token runs on a worker"
        );
        saw_worker.store(true, Ordering::SeqCst);
        let worker = std::thread::current().id();
        let mut destination = StridedArray::<usize>::col_major(&[8, 8]);
        map_into(
            &mut destination.view_mut(),
            &source.view(),
            |value| {
                if std::thread::current().id() != worker {
                    left_worker.store(true, Ordering::SeqCst);
                }
                value + 1
            },
            child,
        )
        .unwrap();
        for index in 0..64 {
            assert_eq!(destination.data()[index], index + 1);
        }
    });

    assert!(saw_worker.load(Ordering::SeqCst));
    assert!(!left_worker.load(Ordering::SeqCst));
    // The plain caller entered the pool once to run the two jobs.
    assert_eq!(strided_exec::ledger::installs(root), 1);
}
