//! Behaviour that holds with and without the `parallel` feature.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use strided_exec::{Error, Exec, Pool};

fn pool(n: usize) -> Pool {
    Pool::builder().threads(n).build().unwrap()
}

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

#[test]
fn new_clamps_to_pool_width_and_rejects_zero() {
    let pool = pool(3);
    let width = pool.width();
    assert_eq!(Exec::new(&pool, 1_000).unwrap().budget(), width);
    assert_eq!(Exec::new(&pool, 1).unwrap().budget().get(), 1);
    assert_eq!(Exec::new(&pool, 0).unwrap_err(), Error::ZeroBudget);
    assert_eq!(
        Pool::builder().threads(0).build().unwrap_err(),
        Error::ZeroThreads
    );
}

#[test]
fn fork_runs_every_job_once_and_shares_conserve_the_budget() {
    let pool = pool(8);
    for budget in 1..=8 {
        for jobs in 0..=20 {
            let hits: Vec<AtomicUsize> = (0..jobs).map(|_| AtomicUsize::new(0)).collect();
            let budgets = Mutex::new(vec![0usize; jobs]);
            let exec = Exec::new(&pool, budget).unwrap();
            let effective = exec.budget().get();
            exec.fork(jobs, |i, child| {
                hits[i].fetch_add(1, Ordering::Relaxed);
                budgets.lock().unwrap()[i] = child.budget().get();
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
            let budgets = budgets.into_inner().unwrap();
            assert!(budgets.iter().all(|&b| (1..=effective).contains(&b)));
            if jobs >= 1 && jobs <= effective {
                // One task per job: the child budgets are exactly the shares.
                assert_eq!(
                    budgets.iter().sum::<usize>(),
                    effective,
                    "{jobs}/{effective}"
                );
                let (lo, hi) = (budgets.iter().min().unwrap(), budgets.iter().max().unwrap());
                assert!(hi - lo <= 1);
            }
        }
    }
}

#[test]
fn for_ranges_covers_every_index_once_with_grain() {
    let pool = pool(8);
    for budget in [1, 2, 3, 8] {
        for len in [0, 1, 2, 7, 8, 9, 63, 64, 65, 1000] {
            for grain in [1, 3, 8, 64, 5000] {
                let ranges = Mutex::new(Vec::new());
                Exec::new(&pool, budget)
                    .unwrap()
                    .for_ranges(len, nz(grain), |r| ranges.lock().unwrap().push(r));
                let mut ranges = ranges.into_inner().unwrap();
                ranges.sort_by_key(|r| r.start);
                let mut next = 0;
                for (k, r) in ranges.iter().enumerate() {
                    assert_eq!(r.start, next, "len {len} grain {grain} budget {budget}");
                    assert!(!r.is_empty());
                    if k + 1 < ranges.len() {
                        assert!(r.len() >= grain);
                    }
                    next = r.end;
                }
                assert_eq!(next, len);
                assert!(ranges.len() <= budget.max(1));
            }
        }
    }
}

#[test]
fn serial_runs_everything_on_the_calling_thread() {
    let here = std::thread::current().id();
    let check = || assert_eq!(std::thread::current().id(), here);
    Exec::serial().fork(5, |_, child| {
        check();
        assert!(child.is_serial());
    });
    Exec::serial().for_ranges(100, nz(1), |_| check());
    Exec::serial()
        .broadcast(|i| {
            assert_eq!(i, 0);
            check();
        })
        .unwrap();
    // A pooled token with budget one is serial too.
    let pool = pool(4);
    Exec::new(&pool, 1).unwrap().fork(3, |_, _| check());
    Exec::new(&pool, 1)
        .unwrap()
        .for_ranges(10, nz(1), |_| check());
}

#[test]
fn reborrow_keeps_pool_and_budget() {
    let pool = pool(2);
    let mut exec = Exec::new(&pool, 2).unwrap();
    let budget = exec.budget();
    assert_eq!(exec.reborrow().budget(), budget);
    exec.reborrow().fork(2, |_, _| {});
    exec.reborrow().for_ranges(4, nz(1), |_| {});
    exec.fork(2, |_, _| {});
}

#[test]
fn limit_narrows_without_widening() {
    let pool = pool(4);
    let w = pool.width().get();
    let exec = Exec::new(&pool, 4).unwrap().limit(nz(2));
    assert_eq!(exec.budget().get(), 2.min(w));
    let exec = Exec::new(&pool, 2).unwrap().limit(nz(100));
    assert_eq!(exec.budget().get(), 2.min(w));
    assert!(Exec::serial().limit(nz(8)).is_serial());
}
