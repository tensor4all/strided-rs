use super::*;
use crate::{with_execution_policy, ExecutionPolicy};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

#[test]
fn permutation_copy_scheduler_is_ambient_and_panic_safe() {
    let two = NonZeroUsize::new(2).unwrap();
    let policy = ExecutionPolicy::Rayon { max_threads: two };

    with_execution_policy(policy, || {
        let panic = std::panic::catch_unwind(|| {
            with_permutation_copy_scheduler(|| {
                assert_eq!(
                    crate::execution_policy::active_policy(),
                    ExecutionPolicy::AmbientRayon
                );
                assert!(!crate::execution_policy::fanout_active());
                panic!("permutation scheduler boundary panic");
            });
        });
        assert!(panic.is_err());
        assert_eq!(crate::execution_policy::active_policy(), policy);
        assert!(!crate::execution_policy::fanout_active());
    });
    assert_eq!(
        crate::execution_policy::active_policy(),
        ExecutionPolicy::AmbientRayon
    );
    assert!(!crate::execution_policy::fanout_active());
}

/// Helper: compute lastargmax via streaming fold (same logic as in mapreduce_threaded).
fn streaming_lastargmax(dims: &[usize], costs: &[isize]) -> usize {
    let (i, _) = dims.iter().zip(costs.iter()).enumerate().fold(
        (0, isize::MIN),
        |(best_i, best_v), (idx, (&d, &c))| {
            let score = (d as isize - 1) * c;
            if score >= best_v {
                (idx, score)
            } else {
                (best_i, best_v)
            }
        },
    );
    i
}

#[test]
fn test_streaming_lastargmax() {
    // Basic: scores = (9*2, 19*1, 4*3) = (18, 19, 12) → max at index 1
    assert_eq!(streaming_lastargmax(&[10, 20, 5], &[2, 1, 3]), 1);

    // Ties: last index wins (>= semantics)
    // scores: (10-1)*1=9, (10-1)*1=9, (10-1)*1=9 → all equal → last wins
    assert_eq!(streaming_lastargmax(&[10, 10, 10], &[1, 1, 1]), 2);

    // All dims=1: scores are all 0 → last wins
    assert_eq!(streaming_lastargmax(&[1, 1, 1], &[1, 1, 1]), 2);

    // Single dimension
    assert_eq!(streaming_lastargmax(&[100], &[2]), 0);
}

#[test]
fn parallel_threads_for_len_honors_policy_and_threshold() {
    let two = NonZeroUsize::new(2).unwrap();
    let four = NonZeroUsize::new(4).unwrap();

    // Test the caps against an explicit pool, independent of the ambient worker count.
    test_pool(4).install(|| {
        with_execution_policy(ExecutionPolicy::Rayon { max_threads: two }, || {
            assert_eq!(parallel_threads_for_len(MINTHREADLENGTH), 1);
            assert_eq!(parallel_threads_for_len(MINTHREADLENGTH + 1), 2);
        });
        with_execution_policy(ExecutionPolicy::Rayon { max_threads: four }, || {
            assert_eq!(parallel_threads_for_len(MINTHREADLENGTH + 1), 4);
        });
        with_execution_policy(ExecutionPolicy::Sequential, || {
            assert_eq!(parallel_threads_for_len(MINTHREADLENGTH + 1), 1);
        });
    });
}

#[test]
fn test_mapreduce_threaded_single_thread() {
    // With nthreads=1, should just call f directly
    let dims = vec![10, 10];
    let blocks = vec![10, 10];
    let strides = vec![vec![1isize, 10], vec![1, 10]];
    let offsets = vec![0isize, 0];
    let costs = vec![2, 20];

    let called = std::sync::atomic::AtomicBool::new(false);
    mapreduce_threaded(
        &dims,
        &blocks,
        &strides,
        &offsets,
        &costs,
        1,
        0,
        1,
        &|_dims, _blocks, _strides, _offsets| {
            called.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        },
    )
    .unwrap();
    assert!(called.load(std::sync::atomic::Ordering::SeqCst));
}

#[test]
fn test_mapreduce_threaded_splits_cover_all_elements() {
    // Verify that parallel splitting covers all elements
    use std::sync::atomic::{AtomicUsize, Ordering};
    let dims = vec![100, 100];
    let blocks = vec![100, 100];
    let strides = vec![vec![1isize, 100], vec![1, 100]];
    let offsets = vec![0isize, 0];
    let costs = vec![2, 200];

    let total_elements = AtomicUsize::new(0);
    mapreduce_threaded(
        &dims,
        &blocks,
        &strides,
        &offsets,
        &costs,
        4,
        0,
        1,
        &|dims, _blocks, _strides, _offsets| {
            let n: usize = dims.iter().product();
            total_elements.fetch_add(n, Ordering::Relaxed);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(total_elements.load(Ordering::SeqCst), 10000);
}

#[test]
fn test_mapreduce_threaded_with_spacing() {
    // Verify spacing/taskindex base case applies offsets correctly
    use std::sync::atomic::{AtomicI64, Ordering};
    let dims = vec![10];
    let blocks = vec![10];
    let strides = vec![vec![0isize], vec![1]];
    let offsets = vec![0isize, 0];
    let costs = vec![2];

    let received_offset = AtomicI64::new(0);
    mapreduce_threaded(
        &dims,
        &blocks,
        &strides,
        &offsets,
        &costs,
        1,
        8,
        3, // spacing=8, taskindex=3
        &|_dims, _blocks, _strides, offsets| {
            received_offset.store(offsets[0] as i64, Ordering::SeqCst);
            Ok(())
        },
    )
    .unwrap();
    // offset[0] should be 8 * (3 - 1) = 16
    assert_eq!(received_offset.load(Ordering::SeqCst), 16);
}

#[test]
fn internal_join_wait_does_not_leak_policy_to_an_unrelated_ambient_job() {
    let pool = test_pool(2);
    let policy = ExecutionPolicy::Rayon {
        max_threads: NonZeroUsize::new(2).unwrap(),
    };
    let right_release = Arc::new(AtomicBool::new(false));
    let (right_started_tx, right_started_rx) = mpsc::channel();
    let (observed_tx, observed_rx) = mpsc::channel();
    let spawn_pool = Arc::clone(&pool);

    pool.install(|| {
        with_execution_policy(policy, || {
            let task_right_release = Arc::clone(&right_release);
            let waiting_right_release = Arc::clone(&right_release);
            join_with_policy(
                policy,
                move || {
                    right_started_rx
                        .recv_timeout(Duration::from_secs(5))
                        .unwrap();
                    spawn_pool.spawn(move || {
                        let observed = with_execution_policy(ExecutionPolicy::AmbientRayon, || {
                            (
                                crate::execution_policy::active_policy(),
                                crate::execution_policy::fanout_active(),
                            )
                        });
                        observed_tx.send(observed).unwrap();
                        task_right_release.store(true, Ordering::Release);
                    });
                },
                move || {
                    right_started_tx.send(()).unwrap();
                    while !waiting_right_release.load(Ordering::Acquire) {
                        std::thread::yield_now();
                    }
                },
            );

            assert_eq!(crate::execution_policy::active_policy(), policy);
            assert!(!crate::execution_policy::fanout_active());
        });
    });

    let observed = observed_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(observed.0, ExecutionPolicy::AmbientRayon);
    assert!(!observed.1);
}

/// The permutation engine resolves Rayon's current pool internally, so the
/// target-pool scope must install around the whole parallel entry: exactly once,
/// on the calling thread, with the scoped pool doing the work.
#[test]
fn target_pool_scope_installs_once_for_parallel_permutation_copy() {
    const ROWS: usize = 257;
    const COLS: usize = 129;
    const _: () = assert!(ROWS * COLS > MINTHREADLENGTH);

    let pool = test_pool(4);
    let source = crate::StridedArray::<f64>::from_fn_col_major(&[ROWS, COLS], |index| {
        (index[0] + ROWS * index[1]) as f64
    });
    let transposed = source.view().permute(&[1, 0]).unwrap();
    let mut destination = crate::StridedArray::<f64>::row_major(&[COLS, ROWS]);

    reset_target_pool_install_count();
    with_execution_pool(&pool, NonZeroUsize::new(4).unwrap(), || {
        copy_permuted_with_active_policy(&mut destination.view_mut(), &transposed).unwrap();
    });

    assert_eq!(target_pool_install_count(), 1);
    for row in 0..ROWS {
        for column in 0..COLS {
            assert_eq!(destination.get(&[column, row]), source.get(&[row, column]));
        }
    }
}

/// A one-worker target pool behaves like the sequential policy even when a
/// low-level caller passes a larger split budget: nothing is installed and no
/// ambient pool is consulted.
#[test]
fn one_worker_target_pool_forces_serial_with_a_larger_budget() {
    let pool = test_pool(1);
    let caller = std::thread::current().id();
    let off_caller = std::sync::atomic::AtomicUsize::new(0);

    reset_target_pool_install_count();
    let sum = with_execution_pool(&pool, NonZeroUsize::new(4).unwrap(), || {
        parallel_map_reduce(
            0..(MINTHREADLENGTH * 2),
            4,
            &|range| {
                if std::thread::current().id() != caller {
                    off_caller.fetch_add(range.len(), Ordering::SeqCst);
                }
                range.sum::<usize>()
            },
            &|left, right| left + right,
        )
    });

    assert_eq!(sum, (0..(MINTHREADLENGTH * 2)).sum::<usize>());
    assert_eq!(off_caller.load(Ordering::SeqCst), 0);
    assert_eq!(target_pool_install_count(), 0);
}

fn wait_for_flag(flag: &AtomicBool, timeout: Duration) {
    let deadline = std::time::Instant::now() + timeout;
    while !flag.load(Ordering::Acquire) {
        if std::time::Instant::now() >= deadline {
            return;
        }
        std::thread::yield_now();
    }
}

/// A target pool larger than the cap must not hand a permutation copy to the
/// HPTT entry, which would use every physical worker. The bounded map fallback
/// keeps the copy on at most `max_threads` distinct workers.
#[test]
fn target_pool_cap_uses_bounded_map_fallback_for_permutation_copy() {
    const ROWS: usize = 257;
    const COLS: usize = 129;
    const _: () = assert!(ROWS * COLS > MINTHREADLENGTH);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .thread_name(|index| format!("perm-cap-{index}"))
        .build()
        .unwrap();
    let source = crate::StridedArray::<f64>::from_fn_col_major(&[ROWS, COLS], |index| {
        (index[0] + ROWS * index[1]) as f64
    });
    let transposed = source.view().permute(&[1, 0]).unwrap();
    let mut destination = crate::StridedArray::<f64>::row_major(&[COLS, ROWS]);

    let _ = take_permutation_fallback_workers();
    with_execution_pool(&pool, NonZeroUsize::new(2).unwrap(), || {
        copy_permuted_with_active_policy(&mut destination.view_mut(), &transposed).unwrap();
    });

    let workers: Vec<usize> = take_permutation_fallback_workers()
        .into_iter()
        .filter_map(|name| {
            name.as_deref()
                .and_then(|name| name.strip_prefix("perm-cap-"))
                .and_then(|index| index.parse().ok())
        })
        .collect();
    assert!(
        !workers.is_empty(),
        "a 4-worker pool capped at 2 must use the bounded map fallback"
    );
    let distinct: std::collections::BTreeSet<_> = workers.iter().copied().collect();
    assert!(
        distinct.len() <= 2,
        "permutation fallback used {} distinct workers: {workers:?}",
        distinct.len()
    );

    for row in 0..ROWS {
        for column in 0..COLS {
            assert_eq!(destination.get(&[column, row]), source.get(&[row, column]));
        }
    }
}

/// The exported low-level map-reduce splitter clamps a caller-supplied budget to
/// the effective (target-capped) width before recursion.
#[test]
fn target_pool_cap_clamps_low_level_map_reduce_budget() {
    let pool = test_pool(4);
    let callback_count = std::sync::atomic::AtomicUsize::new(0);
    let active = std::sync::atomic::AtomicUsize::new(0);
    let max_active = std::sync::atomic::AtomicUsize::new(0);

    let sum = with_execution_pool(&pool, NonZeroUsize::new(2).unwrap(), || {
        parallel_map_reduce(
            0..4096,
            4,
            &|range| {
                callback_count.fetch_add(1, Ordering::SeqCst);
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                max_active.fetch_max(now, Ordering::SeqCst);
                for _ in 0..64 {
                    std::hint::spin_loop();
                }
                active.fetch_sub(1, Ordering::SeqCst);
                range.sum::<usize>()
            },
            &|left, right| left + right,
        )
    });

    assert_eq!(sum, (0..4096usize).sum::<usize>());
    assert!(
        callback_count.load(Ordering::SeqCst) <= 2,
        "low-level splitter created {} leaf partitions with a cap of 2",
        callback_count.load(Ordering::SeqCst)
    );
    assert!(max_active.load(Ordering::SeqCst) <= 2);
}

/// The threaded dimension splitter gets the same initial-budget clamp.
#[test]
fn target_pool_cap_clamps_low_level_mapreduce_threaded_budget() {
    let pool = test_pool(4);
    let leaves = std::sync::atomic::AtomicUsize::new(0);
    let covered = std::sync::atomic::AtomicUsize::new(0);
    let dims = [1usize << 17];
    let blocks = [1usize << 10];
    let strides = [vec![1isize]];
    let offsets = [0isize];
    let costs = [2isize];

    with_execution_pool(&pool, NonZeroUsize::new(2).unwrap(), || {
        mapreduce_threaded(
            &dims,
            &blocks,
            &strides,
            &offsets,
            &costs,
            4,
            0,
            1,
            &|dims, _blocks, _strides, _offsets| {
                leaves.fetch_add(1, Ordering::SeqCst);
                covered.fetch_add(dims.iter().product::<usize>(), Ordering::SeqCst);
                Ok(())
            },
        )
        .unwrap();
    });

    assert!(
        leaves.load(Ordering::SeqCst) <= 2,
        "mapreduce_threaded created {} leaves with a cap of 2",
        leaves.load(Ordering::SeqCst)
    );
    assert_eq!(covered.load(Ordering::SeqCst), 1usize << 17);
}

/// Scheduler suspension clears the target-pool scope as well as the execution
/// policy, so work run while an owned install waits cannot inherit the target.
#[test]
fn scheduler_suspension_clears_target_pool_state() {
    let pool = test_pool(4);

    with_execution_pool(&pool, NonZeroUsize::new(4).unwrap(), || {
        assert!(super::target_pool().is_some());
        crate::execution_policy::with_scheduler_suspended(|| {
            assert!(super::target_pool().is_none());
            assert_eq!(
                crate::execution_policy::active_policy(),
                ExecutionPolicy::AmbientRayon
            );
            assert!(!crate::execution_policy::fanout_active());
        });
        assert!(super::target_pool().is_some());
    });
    assert!(super::target_pool().is_none());
}

/// While an owned target-pool install blocks the caller, Rayon may run an
/// unrelated job from the caller's own pool. That job must not observe the
/// target pool or its cap.
#[test]
fn target_pool_install_wait_does_not_leak_target_state_to_an_ambient_job() {
    let target = test_pool(4);
    let ambient = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap(),
    );
    let policy = ExecutionPolicy::Rayon {
        max_threads: NonZeroUsize::new(4).unwrap(),
    };
    let release = Arc::new(AtomicBool::new(false));
    let scope_active = Arc::new(AtomicBool::new(false));
    let (right_started_tx, right_started_rx) = mpsc::channel();
    let (observed_tx, observed_rx) = mpsc::channel();
    let spawn_pool = Arc::clone(&ambient);
    let job_scope_active = Arc::clone(&scope_active);
    let left_release = Arc::clone(&release);
    let left_wait_release = Arc::clone(&release);
    let right_wait_release = Arc::clone(&release);

    ambient.install(|| {
        with_execution_pool(&target, NonZeroUsize::new(4).unwrap(), || {
            scope_active.store(true, Ordering::SeqCst);
            join_with_policy(
                policy,
                move || {
                    right_started_rx
                        .recv_timeout(Duration::from_secs(5))
                        .unwrap();
                    spawn_pool.spawn(move || {
                        let observed = with_execution_policy(ExecutionPolicy::AmbientRayon, || {
                            (
                                crate::execution_policy::active_policy(),
                                crate::execution_policy::fanout_active(),
                                super::target_pool().is_some(),
                                job_scope_active.load(Ordering::SeqCst),
                            )
                        });
                        observed_tx.send(observed).unwrap();
                        left_release.store(true, Ordering::Release);
                    });
                    wait_for_flag(&left_wait_release, Duration::from_secs(5));
                },
                move || {
                    right_started_tx.send(()).unwrap();
                    wait_for_flag(&right_wait_release, Duration::from_secs(5));
                },
            );
            assert!(super::target_pool().is_some());
            scope_active.store(false, Ordering::SeqCst);
        });
    });

    let observed = observed_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("the ambient job did not run");
    assert_eq!(observed.0, ExecutionPolicy::AmbientRayon);
    assert!(!observed.1);
    assert!(
        !observed.2,
        "target pool leaked into unrelated ambient work"
    );
    assert!(
        observed.3,
        "the ambient job did not run while the target scope was active"
    );
}

/// Nested target-pool scopes replace placement, keep the enclosing budget, and
/// restore the outer scope afterwards.
#[test]
fn nested_target_pool_scopes_restore_the_outer_pool_and_budget() {
    const LEN: usize = 1usize << 17;
    let outer = test_pool(4);
    let inner = test_pool(4);

    with_execution_pool(&outer, NonZeroUsize::new(3).unwrap(), || {
        assert_eq!(target_pool_width(), Some(3));
        with_execution_pool(&inner, NonZeroUsize::new(4).unwrap(), || {
            // The inner scope cannot enlarge the outer budget.
            assert_eq!(target_pool_width(), Some(3));

            let source = crate::StridedArray::<usize>::from_fn_col_major(&[LEN], |index| index[0]);
            let mut destination = crate::StridedArray::<usize>::col_major(&[LEN]);
            let saw_inner = AtomicBool::new(false);
            let saw_outer = AtomicBool::new(false);
            crate::map_into(&mut destination.view_mut(), &source.view(), |value| {
                if inner.current_thread_index().is_some() {
                    saw_inner.store(true, Ordering::SeqCst);
                }
                if outer.current_thread_index().is_some() {
                    saw_outer.store(true, Ordering::SeqCst);
                }
                value
            })
            .unwrap();
            assert!(saw_inner.load(Ordering::SeqCst));
            assert!(!saw_outer.load(Ordering::SeqCst));
        });
        assert_eq!(target_pool_width(), Some(3));
    });
    assert!(super::target_pool().is_none());
}

/// Panicking inside a target-pool scope restores the enclosing scope (or the
/// absence of one).
#[test]
fn target_pool_scope_restores_after_panic() {
    let pool = test_pool(4);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_execution_pool(&pool, NonZeroUsize::new(4).unwrap(), || {
            panic!("target pool scope panic");
        });
    }));
    assert!(panic.is_err());
    assert!(super::target_pool().is_none());
    assert_eq!(
        crate::execution_policy::active_policy(),
        ExecutionPolicy::AmbientRayon
    );

    with_execution_pool(&pool, NonZeroUsize::new(3).unwrap(), || {
        let inner_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_execution_pool(&pool, NonZeroUsize::new(4).unwrap(), || {
                panic!("nested target pool scope panic");
            });
        }));
        assert!(inner_panic.is_err());
        assert_eq!(target_pool_width(), Some(3));
    });
    assert!(super::target_pool().is_none());
}
