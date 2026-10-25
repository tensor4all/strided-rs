//! Placement behaviour that needs real workers.
#![cfg(feature = "parallel")]

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex};
use strided_exec::{Exec, Pool, Refused};

fn named(prefix: &'static str, n: usize) -> Pool {
    Pool::builder()
        .threads(n)
        .thread_name(move |i| format!("{prefix}-{i}"))
        .build()
        .unwrap()
}

fn current_name() -> String {
    std::thread::current().name().unwrap_or("").to_string()
}

#[test]
fn tasks_run_on_pool_workers_when_called_from_a_plain_thread() {
    let pool = named("plain", 3);
    let names = Mutex::new(Vec::new());
    Exec::new(&pool, 3)
        .unwrap()
        .fork(6, |_, _| names.lock().unwrap().push(current_name()));
    let names = names.into_inner().unwrap();
    assert_eq!(names.len(), 6);
    assert!(names.iter().all(|n| n.starts_with("plain-")), "{names:?}");
}

#[test]
fn cross_pool_call_runs_on_the_target_pool() {
    let a = named("alpha", 2);
    let b = named("beta", 2);
    let seen = Mutex::new(Vec::new());
    Exec::new(&a, 2).unwrap().fork(2, |_, _| {
        assert!(current_name().starts_with("alpha-"));
        Exec::new(&b, 2).unwrap().fork(4, |_, _| {
            seen.lock().unwrap().push(current_name());
        });
        Exec::new(&b, 2)
            .unwrap()
            .for_ranges(8, NonZeroUsize::MIN, |_| {
                seen.lock().unwrap().push(current_name());
            });
    });
    let seen = seen.into_inner().unwrap();
    assert!(!seen.is_empty());
    assert!(seen.iter().all(|n| n.starts_with("beta-")), "{seen:?}");
}

#[test]
fn same_pool_nested_forks_stay_on_the_pool() {
    let pool = named("same", 4);
    let names = Mutex::new(Vec::new());
    Exec::new(&pool, 4).unwrap().fork(2, |_, child| {
        child.fork(2, |_, _| names.lock().unwrap().push(current_name()));
    });
    let names = names.into_inner().unwrap();
    assert_eq!(names.len(), 4);
    assert!(names.iter().all(|n| n.starts_with("same-")));
}

#[test]
fn broadcast_co_schedules_all_participants() {
    let pool = named("bc", 4);
    for budget in 2..=4 {
        let barrier = Barrier::new(budget);
        let seen: Vec<AtomicUsize> = (0..budget).map(|_| AtomicUsize::new(0)).collect();
        Exec::new(&pool, budget)
            .unwrap()
            .broadcast(|i| {
                barrier.wait();
                seen[i].fetch_add(1, Ordering::Relaxed);
            })
            .unwrap();
        assert!(seen.iter().all(|s| s.load(Ordering::Relaxed) == 1));
    }
}

#[test]
fn concurrent_broadcasts_on_one_pool_do_not_deadlock() {
    let pool = named("bc2", 4);
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                for _ in 0..20 {
                    let barrier = Barrier::new(2);
                    Exec::new(&pool, 2)
                        .unwrap()
                        .broadcast(|_| {
                            barrier.wait();
                        })
                        .unwrap();
                }
            });
        }
    });
}

#[test]
fn broadcast_from_a_foreign_pool_worker_runs_on_the_target_pool() {
    let a = named("src", 2);
    let b = named("dst", 2);
    let names = Mutex::new(Vec::new());
    Exec::new(&a, 2).unwrap().fork(2, |_, _| {
        let barrier = Barrier::new(2);
        Exec::new(&b, 2)
            .unwrap()
            .broadcast(|_| {
                barrier.wait();
                names.lock().unwrap().push(current_name());
            })
            .unwrap();
    });
    let names = names.into_inner().unwrap();
    assert_eq!(names.len(), 4);
    assert!(names.iter().all(|n| n.starts_with("dst-")), "{names:?}");
}

#[test]
fn broadcast_refuses_fork_children_and_in_pool_callers() {
    let pool = named("refuse", 4);
    // fork(1) hands the whole budget to the single job, inline on this thread.
    Exec::new(&pool, 4).unwrap().fork(1, |_, child| {
        assert!(child.budget().get() > 1);
        assert_eq!(child.broadcast(|_| unreachable!()), Err(Refused::ForkChild));
    });
    let mut exec = Exec::new(&pool, 4).unwrap();
    exec.reborrow().fork(1, |_, mut child| {
        assert_eq!(
            child.reborrow().broadcast(|_| unreachable!()),
            Err(Refused::ForkChild)
        );
    });
    // From a worker of the same pool.
    exec.fork(2, |_, _| {
        let own = Exec::new(&pool, 2).unwrap();
        assert_eq!(
            own.broadcast(|_| unreachable!()),
            Err(Refused::CallerInPool)
        );
    });
}

#[test]
fn broadcast_with_budget_one_runs_inline_even_for_children() {
    let pool = named("one", 2);
    let here = std::thread::current().id();
    Exec::new(&pool, 1)
        .unwrap()
        .broadcast(|i| {
            assert_eq!(i, 0);
            assert_eq!(std::thread::current().id(), here);
        })
        .unwrap();
    Exec::new(&pool, 2).unwrap().fork(2, |_, child| {
        assert_eq!(child.budget().get(), 1);
        child.broadcast(|_| {}).unwrap();
    });
}

#[test]
fn limit_keeps_broadcast_eligibility_of_roots_only() {
    let pool = named("lim", 4);
    let barrier = Barrier::new(2);
    let mut exec = Exec::new(&pool, 4).unwrap();
    exec.reborrow()
        .limit(NonZeroUsize::new(2).unwrap())
        .broadcast(|_| {
            barrier.wait();
        })
        .unwrap();
    exec.fork(1, |_, child| {
        let child = child.limit(NonZeroUsize::new(2).unwrap());
        assert_eq!(child.broadcast(|_| unreachable!()), Err(Refused::ForkChild));
    });
}

#[test]
fn external_hands_the_budget_and_runs_on_the_pool() {
    let pool = named("ext", 4);
    for budget in 1..=6 {
        let (width, name) = Exec::new(&pool, budget)
            .unwrap()
            .external(|width| (width.get(), current_name()));
        assert_eq!(width, budget.min(4));
        if budget > 1 {
            assert!(name.starts_with("ext-"), "{name}");
        }
    }
    assert_eq!(Exec::serial().external(|width| width.get()), 1);
}

#[test]
fn external_from_a_worker_of_the_pool_runs_in_place() {
    let pool = named("ext-in", 2);
    Exec::new(&pool, 2).unwrap().fork(2, |_, _| {
        let caller = current_name();
        let inner = Exec::new(&pool, 2)
            .unwrap()
            .external(|width| (width.get(), current_name()));
        assert_eq!(inner, (2, caller));
    });
}

#[test]
fn can_broadcast_agrees_with_broadcast_in_every_case() {
    let pool = named("can", 4);
    let agrees = |exec: Exec<'_>, expected: bool| {
        // `can_broadcast` does not consume; `broadcast` then must match it.
        assert_eq!(exec.can_broadcast(), expected);
        assert_eq!(exec.broadcast(|_| {}).is_ok(), expected);
    };
    // Root, serial, budget one.
    agrees(Exec::new(&pool, 4).unwrap(), true);
    agrees(Exec::serial(), true);
    agrees(Exec::new(&pool, 1).unwrap(), true);
    // Reborrow and limit of a root.
    let mut root = Exec::new(&pool, 4).unwrap();
    agrees(root.reborrow(), true);
    agrees(root.reborrow().limit(NonZeroUsize::new(2).unwrap()), true);
    // Fork children: refused with a budget above one, inline with budget one.
    root.fork(1, |_, mut child| {
        agrees(child.reborrow(), false);
        agrees(child.reborrow().limit(NonZeroUsize::new(2).unwrap()), false);
        agrees(child, false);
    });
    Exec::new(&pool, 2).unwrap().fork(2, |_, child| {
        assert_eq!(child.budget().get(), 1);
        agrees(child, true);
    });
    // A worker of the same pool.
    Exec::new(&pool, 2).unwrap().fork(2, |_, _| {
        agrees(Exec::new(&pool, 2).unwrap(), false);
        agrees(Exec::new(&pool, 1).unwrap(), true);
    });
    // A worker of another pool may broadcast on this one.
    let other = named("can-other", 1);
    Exec::new(&other, 1).unwrap().external(|_| {
        agrees(Exec::new(&pool, 4).unwrap(), true);
    });
}
