//! Ledger-backed invariants: concurrency per root token and pool entries.
#![cfg(all(feature = "parallel", feature = "ledger"))]

use std::num::NonZeroUsize;
use std::sync::Barrier;
use std::time::Duration;
use strided_exec::{ledger, Exec, Pool};

fn pool(n: usize) -> Pool {
    Pool::builder()
        .threads(n)
        .thread_name(|i| format!("ledger-{i}"))
        .build()
        .unwrap()
}

fn nap() {
    std::thread::sleep(Duration::from_millis(2));
}

#[test]
fn nested_fork_never_exceeds_the_root_budget() {
    let pool = pool(8);
    for budget in [1, 2, 3, 5, 8] {
        let exec = Exec::new(&pool, budget).unwrap();
        let root = exec.ledger_root().unwrap();
        exec.fork(5, |_, l1| {
            l1.fork(4, |_, l2| {
                l2.fork(3, |_, mut l3| {
                    nap();
                    l3.reborrow().for_ranges(10, NonZeroUsize::MIN, |_| nap());
                    l3.fork(2, |_, _| nap());
                });
            });
        });
        assert!(ledger::max_live(root) <= budget, "budget {budget}");
        assert_eq!(ledger::live(root), 0);
    }
}

#[test]
fn saturating_flat_fork_reaches_the_budget() {
    let pool = pool(8);
    let exec = Exec::new(&pool, 8).unwrap();
    let root = exec.ledger_root().unwrap();
    let barrier = Barrier::new(8);
    exec.fork(8, |_, _| {
        barrier.wait();
    });
    assert_eq!(ledger::max_live(root), 8);
}

#[test]
fn saturating_three_level_fork_reaches_exactly_the_budget() {
    let pool = pool(8);
    let exec = Exec::new(&pool, 8).unwrap();
    let root = exec.ledger_root().unwrap();
    let barrier = Barrier::new(8);
    exec.fork(2, |_, l1| {
        l1.fork(2, |_, l2| {
            l2.fork(2, |_, _| {
                barrier.wait();
            });
        });
    });
    assert_eq!(ledger::max_live(root), 8);
}

#[test]
fn for_ranges_and_broadcast_are_counted() {
    let pool = pool(4);
    let exec = Exec::new(&pool, 4).unwrap();
    let root = exec.ledger_root().unwrap();
    let barrier = Barrier::new(4);
    exec.for_ranges(4, NonZeroUsize::MIN, |_| {
        barrier.wait();
    });
    assert_eq!(ledger::max_live(root), 4);

    let exec = Exec::new(&pool, 4).unwrap();
    let root = exec.ledger_root().unwrap();
    let barrier = Barrier::new(4);
    exec.broadcast(|_| {
        barrier.wait();
    })
    .unwrap();
    assert_eq!(ledger::max_live(root), 4);
}

#[test]
fn same_pool_reentry_does_not_install() {
    let pool = pool(4);
    // Entering from a plain thread installs exactly once per entry.
    let outer = Exec::new(&pool, 2).unwrap();
    let outer_root = outer.ledger_root().unwrap();
    outer.fork(2, |_, _| {
        // Now on a worker of `pool`: nested entries of any root must not install.
        let inner = Exec::new(&pool, 2).unwrap();
        let inner_root = inner.ledger_root().unwrap();
        inner.fork(2, |_, _| {});
        let inner = Exec::new(&pool, 2).unwrap();
        let inner_root2 = inner.ledger_root().unwrap();
        inner.for_ranges(8, NonZeroUsize::MIN, |_| {});
        assert_eq!(ledger::installs(inner_root), 0);
        assert_eq!(ledger::installs(inner_root2), 0);
    });
    assert_eq!(ledger::installs(outer_root), 1);
}

#[test]
fn inline_entries_never_install() {
    let pool = pool(4);
    let exec = Exec::new(&pool, 4).unwrap();
    let root = exec.ledger_root().unwrap();
    exec.fork(1, |_, child| child.for_ranges(1, NonZeroUsize::MIN, |_| {}));
    assert_eq!(ledger::installs(root), 0);

    let exec = Exec::new(&pool, 1).unwrap();
    let root = exec.ledger_root().unwrap();
    exec.fork(9, |_, _| {});
    assert_eq!(ledger::installs(root), 0);
}
