use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[test]
fn permutation_copy_parallel_eligibility_is_deterministic() {
    let two = NonZeroUsize::new(2).unwrap();
    let four = NonZeroUsize::new(4).unwrap();

    assert!(permutation_copy_parallel_eligible(
        ExecutionPolicy::Rayon { max_threads: two },
        false,
        2,
    ));
    assert!(permutation_copy_parallel_eligible(
        ExecutionPolicy::Rayon { max_threads: four },
        false,
        2,
    ));
    assert!(!permutation_copy_parallel_eligible(
        ExecutionPolicy::Rayon { max_threads: two },
        false,
        4,
    ));
    assert!(!permutation_copy_parallel_eligible(
        ExecutionPolicy::Rayon { max_threads: two },
        true,
        2,
    ));
    assert!(!permutation_copy_parallel_eligible(
        ExecutionPolicy::Sequential,
        false,
        2,
    ));
    assert!(permutation_copy_parallel_eligible(
        ExecutionPolicy::AmbientRayon,
        false,
        2,
    ));
}

#[test]
fn scheduler_panic_restores_owned_policy_and_fanout_state() {
    let two = NonZeroUsize::new(2).unwrap();
    let policy = ExecutionPolicy::Rayon { max_threads: two };

    with_execution_policy(policy, || {
        with_owned_execution(policy, true, || {
            let panic = catch_unwind(AssertUnwindSafe(|| {
                with_scheduler_suspended(|| panic!("scheduler boundary panic"));
            }));
            assert!(panic.is_err());
            assert_eq!(active_policy(), policy);
            assert!(fanout_active());
        });
    });
    assert_eq!(active_policy(), ExecutionPolicy::AmbientRayon);
    assert!(!fanout_active());
}

#[test]
fn leaf_panic_restores_ambient_policy_and_inactive_fanout() {
    let two = NonZeroUsize::new(2).unwrap();
    let policy = ExecutionPolicy::Rayon { max_threads: two };

    let panic = catch_unwind(AssertUnwindSafe(|| {
        with_owned_execution(policy, true, || panic!("owned leaf panic"));
    }));
    assert!(panic.is_err());
    assert_eq!(active_policy(), ExecutionPolicy::AmbientRayon);
    assert!(!fanout_active());
}
