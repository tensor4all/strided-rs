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

#[cfg(feature = "parallel")]
mod run_on {
    use super::ExecContext;
    use crate::StridedError;

    fn pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    }

    /// The operation runs on the pool it is handed: the ambient pool is never
    /// consulted, so a host that owns a pool shares it without installing it.
    #[test]
    fn the_given_pool_is_the_one_in_use() {
        let pool = pool(2);

        let observed = ExecContext::run_on(&pool, 2, |ctx| {
            (
                rayon::current_num_threads(),
                ctx.max_threads_limit().map(|value| value.get()),
            )
        })
        .unwrap();

        assert_eq!(observed, (2, Some(2)));
    }

    /// Width one never enters Rayon machinery, and a one-worker pool clamps to
    /// that even when more threads are requested.
    #[test]
    fn one_worker_stays_on_the_calling_thread() {
        let two = pool(2);
        let serial = ExecContext::run_on(&two, 1, |ctx| {
            (ctx.is_serial(), rayon::current_thread_index())
        })
        .unwrap();
        assert_eq!(serial, (true, None));

        let one = pool(1);
        let clamped = ExecContext::run_on(&one, 4, |ctx| ctx.is_serial()).unwrap();
        assert!(clamped);
    }

    /// A call from inside the pool runs in place: no second team is nested.
    #[test]
    fn a_worker_of_the_pool_does_not_nest_a_team() {
        let pool = pool(2);

        let (inner_thread, inner_limit) = ExecContext::run_on(&pool, 2, |_| {
            ExecContext::run_on(&pool, 2, |ctx| {
                (
                    rayon::current_thread_index(),
                    ctx.max_threads_limit().map(|value| value.get()),
                )
            })
            .unwrap()
        })
        .unwrap();

        assert!(inner_thread.is_some());
        assert_eq!(inner_limit, Some(2));
    }

    #[test]
    fn zero_workers_is_rejected() {
        let pool = pool(2);

        assert!(matches!(
            ExecContext::run_on(&pool, 0, |_| ()).unwrap_err(),
            StridedError::InvalidThreadBudget { max_threads: 0 }
        ));
    }
}
