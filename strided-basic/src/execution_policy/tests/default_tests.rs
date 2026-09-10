use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[test]
fn ambient_scope_returns_result_without_changing_state() {
    let before = state();

    let value = with_execution_policy(ExecutionPolicy::AmbientRayon, || {
        let active = state();
        assert_eq!(active.policy, before.policy);
        assert_eq!(active.fanout_active, before.fanout_active);
        17usize
    });

    let after = state();
    assert_eq!(value, 17);
    assert_eq!(after.policy, before.policy);
    assert_eq!(after.fanout_active, before.fanout_active);
}

#[test]
fn explicit_scope_restores_state_after_return_and_panic() {
    let before = state();
    let two = NonZeroUsize::new(2).unwrap();

    let value = with_execution_policy(ExecutionPolicy::Rayon { max_threads: two }, || {
        assert_eq!(state().policy, ExecutionPolicy::Rayon { max_threads: two });
        23usize
    });
    assert_eq!(value, 23);
    assert_eq!(state().policy, before.policy);
    assert_eq!(state().fanout_active, before.fanout_active);

    let panic = catch_unwind(AssertUnwindSafe(|| {
        with_execution_policy(ExecutionPolicy::Sequential, || panic!("policy scope panic"));
    }));
    assert!(panic.is_err());
    assert_eq!(state().policy, before.policy);
    assert_eq!(state().fanout_active, before.fanout_active);
}

#[test]
fn nested_explicit_scopes_combine_conservatively() {
    let two = NonZeroUsize::new(2).unwrap();
    let four = NonZeroUsize::new(4).unwrap();

    with_execution_policy(ExecutionPolicy::Rayon { max_threads: four }, || {
        assert_eq!(state().policy, ExecutionPolicy::Rayon { max_threads: four });

        with_execution_policy(ExecutionPolicy::Rayon { max_threads: two }, || {
            assert_eq!(state().policy, ExecutionPolicy::Rayon { max_threads: two });
        });
        assert_eq!(state().policy, ExecutionPolicy::Rayon { max_threads: four });

        with_execution_policy(ExecutionPolicy::Sequential, || {
            assert_eq!(state().policy, ExecutionPolicy::Sequential);
        });
        assert_eq!(state().policy, ExecutionPolicy::Rayon { max_threads: four });
    });

    with_execution_policy(ExecutionPolicy::Sequential, || {
        with_execution_policy(ExecutionPolicy::Rayon { max_threads: four }, || {
            assert_eq!(state().policy, ExecutionPolicy::Sequential);
        });
    });
    assert_eq!(state().policy, ExecutionPolicy::AmbientRayon);
    assert!(!state().fanout_active);
}
