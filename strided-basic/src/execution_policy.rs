use core::cell::Cell;
use core::num::NonZeroUsize;

/// Controls how a strided operation may use CPU threads.
///
/// This policy controls fanout only. It does not create a Rayon pool or choose
/// CPU placement; callers that need placement control should install the
/// operation in their chosen executor and then select [`Self::Sequential`] or
/// [`Self::Rayon`].
///
/// The policy applies to fanout owned by `strided-kernel`. A bounded worker
/// partition runs nested strided operations sequentially so nested operations
/// cannot multiply the outer budget. The scope is worker-local, not a
/// task-local Rayon context, and is not propagated to threads or Rayon tasks
/// created by user callbacks.
///
/// User callbacks that rely on policy isolation must not enter their own Rayon
/// scheduling or yield boundary (`join`, `scope`, `spawn`, and similar APIs).
/// At such a callback-owned boundary, Rayon may execute unrelated work on the
/// waiting worker while its worker-local policy is active. `strided-kernel`
/// suspends the policy at every scheduler boundary it owns, but cannot do so
/// for arbitrary scheduling performed inside a callback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionPolicy {
    /// Preserve the compatibility behavior of using all threads in the current
    /// installed Rayon pool, or the global pool when no pool is installed.
    ///
    /// Explicit runtimes should prefer [`Self::Sequential`] or [`Self::Rayon`].
    AmbientRayon,
    /// Execute entirely on the calling thread with zero Rayon fanout.
    Sequential,
    /// Use the currently installed Rayon pool while limiting the operation to
    /// at most `max_threads` concurrent partitions.
    ///
    /// Without the `parallel` feature, operations remain sequential.
    Rayon { max_threads: NonZeroUsize },
}

thread_local! {
    static ACTIVE_POLICY: Cell<ExecutionPolicy> = const { Cell::new(ExecutionPolicy::AmbientRayon) };
    static ACTIVE_FANOUT: Cell<bool> = const { Cell::new(false) };
}

fn restrict(outer: ExecutionPolicy, inner: ExecutionPolicy) -> ExecutionPolicy {
    match (outer, inner) {
        (ExecutionPolicy::Sequential, _) | (_, ExecutionPolicy::Sequential) => {
            ExecutionPolicy::Sequential
        }
        (ExecutionPolicy::AmbientRayon, policy) | (policy, ExecutionPolicy::AmbientRayon) => policy,
        (
            ExecutionPolicy::Rayon { max_threads: outer },
            ExecutionPolicy::Rayon { max_threads: inner },
        ) => ExecutionPolicy::Rayon {
            max_threads: outer.min(inner),
        },
    }
}

#[derive(Clone, Copy)]
struct ExecutionState {
    policy: ExecutionPolicy,
    fanout_active: bool,
}

struct StateGuard {
    previous: ExecutionState,
}

impl Drop for StateGuard {
    fn drop(&mut self) {
        set_state(self.previous);
    }
}

fn state() -> ExecutionState {
    ExecutionState {
        policy: ACTIVE_POLICY.with(Cell::get),
        fanout_active: ACTIVE_FANOUT.with(Cell::get),
    }
}

fn set_state(state: ExecutionState) {
    ACTIVE_POLICY.with(|active| active.set(state.policy));
    ACTIVE_FANOUT.with(|active| active.set(state.fanout_active));
}

#[cfg(feature = "parallel")]
fn with_state<R>(next: ExecutionState, operation: impl FnOnce() -> R) -> R {
    let previous = state();
    set_state(next);
    let _guard = StateGuard { previous };
    operation()
}

/// Execute `operation` under an explicit strided-kernel execution policy.
///
/// Nested scopes combine conservatively: sequential execution dominates and
/// nested Rayon budgets use the smaller limit. The previous policy is restored
/// even if `operation` panics.
///
/// The contract covers strided operations and their library-owned fanout. It
/// does not turn Rayon worker-local state into task-local state. In particular,
/// callbacks should not invoke Rayon scheduling or yielding APIs if they depend
/// on isolation from unrelated work; see [`ExecutionPolicy`] for details.
///
/// # Examples
///
/// ```rust
/// use strided_basic::{
///     map_into, with_execution_policy, ExecutionPolicy, StridedArray,
/// };
///
/// let source = StridedArray::<f64>::from_fn_col_major(&[3], |index| index[0] as f64);
/// let mut destination = StridedArray::<f64>::col_major(&[3]);
/// with_execution_policy(ExecutionPolicy::Sequential, || {
///     map_into(&mut destination.view_mut(), &source.view(), |value| value + 1.0)
///         .unwrap();
/// });
/// assert_eq!(destination.into_data(), vec![1.0, 2.0, 3.0]);
/// ```
#[inline]
pub fn with_execution_policy<R>(policy: ExecutionPolicy, operation: impl FnOnce() -> R) -> R {
    let policy = match policy {
        ExecutionPolicy::AmbientRayon => return operation(),
        policy => policy,
    };
    let previous = state();
    set_state(ExecutionState {
        policy: restrict(previous.policy, policy),
        fanout_active: previous.fanout_active,
    });
    let _guard = StateGuard { previous };
    operation()
}

#[cfg(feature = "parallel")]
pub(crate) fn active_policy() -> ExecutionPolicy {
    ACTIVE_POLICY.with(Cell::get)
}

#[cfg(feature = "parallel")]
pub(crate) fn fanout_active() -> bool {
    ACTIVE_FANOUT.with(Cell::get)
}

#[cfg(feature = "parallel")]
#[inline(always)]
pub(crate) fn with_owned_execution<R>(
    policy: ExecutionPolicy,
    fanout_active: bool,
    operation: impl FnOnce() -> R,
) -> R {
    match policy {
        ExecutionPolicy::AmbientRayon => operation(),
        ExecutionPolicy::Sequential | ExecutionPolicy::Rayon { .. } => {
            let previous = state();
            with_state(
                ExecutionState {
                    policy: restrict(previous.policy, policy),
                    fanout_active: previous.fanout_active || fanout_active,
                },
                operation,
            )
        }
    }
}

#[cfg(feature = "parallel")]
pub(crate) fn with_scheduler_suspended<R>(operation: impl FnOnce() -> R) -> R {
    with_state(
        ExecutionState {
            policy: ExecutionPolicy::AmbientRayon,
            fanout_active: false,
        },
        operation,
    )
}

#[cfg(feature = "parallel")]
pub(crate) fn permutation_copy_parallel_eligible(
    policy: ExecutionPolicy,
    fanout_active: bool,
    current_pool_threads: usize,
) -> bool {
    if fanout_active || current_pool_threads <= 1 {
        return false;
    }
    match policy {
        ExecutionPolicy::AmbientRayon => true,
        ExecutionPolicy::Sequential => false,
        ExecutionPolicy::Rayon { max_threads } => current_pool_threads <= max_threads.get(),
    }
}

#[cfg(feature = "parallel")]
pub fn rayon_threads() -> usize {
    if fanout_active() {
        return 1;
    }
    match active_policy() {
        ExecutionPolicy::Sequential => 1,
        ExecutionPolicy::AmbientRayon => crate::threading::current_pool_threads(),
        ExecutionPolicy::Rayon { max_threads } => {
            crate::threading::current_pool_threads().min(max_threads.get())
        }
    }
}

#[cfg(test)]
#[path = "execution_policy/tests/default_tests.rs"]
mod default_tests;

#[cfg(all(test, feature = "parallel"))]
#[path = "execution_policy/tests/tests.rs"]
mod tests;
