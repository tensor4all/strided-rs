//! A large operation handed a token minted from a target pool fans out on that
//! pool's workers and never on the calling thread.

#![cfg(feature = "parallel")]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use strided_basic::{map_into, Exec, Pool, StridedArray};

const LEN: usize = 1 << 17;
const _: () = assert!(LEN > (1 << 15));
const WORKER_PREFIX: &str = "strided-large-worker";

fn named_pool() -> Pool {
    Pool::builder()
        .dispatch_costs(cheap_dispatch_costs())
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
fn large_operation_runs_on_target_pool_workers() {
    let pool = named_pool();
    let source = StridedArray::<usize>::from_fn_col_major(&[LEN], |index| index[0]);
    let mut destination = StridedArray::<usize>::col_major(&[LEN]);
    let caller = std::thread::current().id();

    let saw_worker = AtomicBool::new(false);
    let ran_on_caller = AtomicBool::new(false);
    let visited = AtomicUsize::new(0);

    let exec = Exec::new(&pool, 4).unwrap();
    let root = exec.ledger_root().unwrap();
    map_into(
        &mut destination.view_mut(),
        &source.view(),
        |value| {
            if on_pool_worker() {
                saw_worker.store(true, Ordering::SeqCst);
            }
            if std::thread::current().id() == caller {
                ran_on_caller.store(true, Ordering::SeqCst);
            }
            visited.fetch_add(1, Ordering::Relaxed);
            value + 1
        },
        exec,
    )
    .unwrap();

    assert!(saw_worker.load(Ordering::SeqCst));
    assert!(!ran_on_caller.load(Ordering::SeqCst));
    assert_eq!(visited.load(Ordering::Relaxed), LEN);
    assert_eq!(destination.get(&[0]), 1);
    assert_eq!(destination.get(&[LEN - 1]), LEN);
    assert_eq!(
        threadteam::ledger::installs(root),
        1,
        "the plain caller installed once"
    );
}

#[test]
fn capped_target_pool_limits_concurrent_workers() {
    let pool = named_pool();
    let source = StridedArray::<usize>::from_fn_col_major(&[LEN], |index| index[0]);
    let mut destination = StridedArray::<usize>::col_major(&[LEN]);

    let active = AtomicUsize::new(0);
    let max_active = AtomicUsize::new(0);
    let wrong_pool = AtomicBool::new(false);

    let exec = Exec::new(&pool, 2).unwrap();
    let root = exec.ledger_root().unwrap();
    map_into(
        &mut destination.view_mut(),
        &source.view(),
        |value| {
            if !on_pool_worker() {
                wrong_pool.store(true, Ordering::SeqCst);
            }
            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
            max_active.fetch_max(now, Ordering::SeqCst);
            std::hint::spin_loop();
            active.fetch_sub(1, Ordering::SeqCst);
            value
        },
        exec,
    )
    .unwrap();

    assert!(!wrong_pool.load(Ordering::SeqCst));
    assert!(max_active.load(Ordering::SeqCst) <= 2);
    assert!(threadteam::ledger::max_live(root) <= 2);
    assert_eq!(destination.get(&[LEN - 1]), LEN - 1);
}

/// Dispatch costs of the test pools: 100 ns per section whatever the token, so that the
/// regions of tens of thousands of elements these tests use open the parallel branches
/// (the host default costs would keep them serial).
fn cheap_dispatch_costs() -> threadteam::DispatchCosts {
    use threadteam::{DispatchKind, WidthCosts};
    let cheap = WidthCosts::flat(std::time::Duration::from_nanos(100));
    threadteam::DispatchCosts::for_class(threadteam::HostClass::Generic)
        .with(DispatchKind::Team, cheap)
        .with(DispatchKind::InPool, cheap)
        .with(DispatchKind::Outside, cheap)
}
