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
