use super::ExecContext;
use crate::StridedError;

#[test]
fn serial_context_has_no_thread_limit() {
    let ctx = ExecContext::serial();

    assert!(ctx.is_serial());
    assert!(!ctx.is_ambient());
    assert_eq!(ctx.max_threads_limit(), None);
    assert_eq!(ExecContext::default(), ctx);
}

#[test]
fn bounded_context_rejects_zero_and_exposes_limit() {
    let ctx = ExecContext::max_threads(4).unwrap();

    assert!(!ctx.is_serial());
    assert!(!ctx.is_ambient());
    assert_eq!(ctx.max_threads_limit().map(|value| value.get()), Some(4));
    assert!(matches!(
        ExecContext::max_threads(0).unwrap_err(),
        StridedError::InvalidThreadBudget { max_threads: 0 }
    ));
}

#[test]
fn ambient_context_has_no_thread_limit() {
    let ctx = ExecContext::ambient();

    assert!(!ctx.is_serial());
    assert!(ctx.is_ambient());
    assert_eq!(ctx.max_threads_limit(), None);
}

#[cfg(feature = "parallel")]
#[test]
fn run_installs_execution_policy_and_restores_previous_policy() {
    use core::num::NonZeroUsize;

    use crate::execution_policy::{active_policy, with_execution_policy, ExecutionPolicy};

    let two = NonZeroUsize::new(2).unwrap();
    let four = NonZeroUsize::new(4).unwrap();
    let bounded = ExecContext::max_threads(2).unwrap();

    let observed_bounded =
        with_execution_policy(ExecutionPolicy::Rayon { max_threads: four }, || {
            bounded.run(active_policy)
        });
    let observed_serial =
        with_execution_policy(ExecutionPolicy::Rayon { max_threads: four }, || {
            ExecContext::serial().run(active_policy)
        });
    let observed_ambient =
        with_execution_policy(ExecutionPolicy::Rayon { max_threads: four }, || {
            ExecContext::ambient().run(active_policy)
        });

    assert_eq!(
        observed_bounded,
        ExecutionPolicy::Rayon { max_threads: two }
    );
    assert_eq!(observed_serial, ExecutionPolicy::Sequential);
    assert_eq!(
        observed_ambient,
        ExecutionPolicy::Rayon { max_threads: four }
    );
    assert_eq!(active_policy(), ExecutionPolicy::AmbientRayon);
}
