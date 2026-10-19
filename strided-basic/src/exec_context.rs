//! Execution policy passed through erased kernel replay boundaries.
//!
//! `ExecContext` is explicit even for kernels that are currently serial so
//! downstream runtimes do not accidentally bake ambient thread-pool state into
//! their prepared-kernel ABI.

use core::num::NonZeroUsize;

use crate::{with_execution_policy, ExecutionPolicy, Result, StridedError};

/// Caller-selected execution policy for prepared kernel replay.
///
/// This type intentionally hides its representation so future replay families
/// can add provider-owned pools or scheduling scopes without forcing downstream
/// crates to pattern-match a closed enum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecContext {
    kind: ExecContextKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExecContextKind {
    Serial,
    MaxThreads(NonZeroUsize),
    Ambient,
}

impl ExecContext {
    /// Execute without entering a parallel worker pool.
    #[inline]
    pub const fn serial() -> Self {
        Self {
            kind: ExecContextKind::Serial,
        }
    }

    /// Execute with an operation-local upper bound on worker threads.
    #[inline]
    pub fn max_threads(max_threads: usize) -> Result<Self> {
        match NonZeroUsize::new(max_threads) {
            Some(max_threads) => Ok(Self {
                kind: ExecContextKind::MaxThreads(max_threads),
            }),
            None => Err(StridedError::InvalidThreadBudget { max_threads }),
        }
    }

    /// Execute using the ambient runtime policy.
    ///
    /// This is useful for direct `strided-kernel` users. Runtime crates that
    /// own CPU resources should prefer [`ExecContext::serial`] or
    /// [`ExecContext::max_threads`] so thread ownership remains explicit.
    #[inline]
    pub const fn ambient() -> Self {
        Self {
            kind: ExecContextKind::Ambient,
        }
    }

    /// Returns `true` when this context requires serial execution.
    #[inline]
    pub fn is_serial(&self) -> bool {
        matches!(self.kind, ExecContextKind::Serial)
    }

    /// Returns `true` when this context delegates to ambient runtime policy.
    #[inline]
    pub fn is_ambient(&self) -> bool {
        matches!(self.kind, ExecContextKind::Ambient)
    }

    /// Returns the configured worker-thread upper bound, if any.
    #[inline]
    pub fn max_threads_limit(&self) -> Option<NonZeroUsize> {
        match self.kind {
            ExecContextKind::MaxThreads(max_threads) => Some(max_threads),
            ExecContextKind::Serial | ExecContextKind::Ambient => None,
        }
    }

    #[inline]
    /// Run an operation under this context, preserving enclosing execution limits.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::ExecContext;
    /// assert_eq!(ExecContext::serial().run(|| 42), 42);
    /// ```
    pub fn run<R>(&self, operation: impl FnOnce() -> R) -> R {
        match self.kind {
            ExecContextKind::Serial => {
                with_execution_policy(ExecutionPolicy::Sequential, operation)
            }
            ExecContextKind::MaxThreads(max_threads) => {
                with_execution_policy(ExecutionPolicy::Rayon { max_threads }, operation)
            }
            ExecContextKind::Ambient => operation(),
        }
    }
}

#[cfg(feature = "parallel")]
impl ExecContext {
    /// Enter `pool` for `operation`, bounding fanout to `max_threads` workers.
    ///
    /// Placement belongs to the caller: this never creates a pool, and the
    /// operation runs on the pool it is handed rather than on whichever pool
    /// happens to be installed, so a host that already owns a pool shares that
    /// one with these kernels instead of risking a second team.
    ///
    /// The width is clamped to the pool's worker count. A width of one enters no
    /// Rayon machinery at all and `operation` sees [`ExecContext::serial`]. When
    /// the calling thread is already a worker of `pool`, the operation runs in
    /// place rather than nesting a second team.
    ///
    /// # Errors
    ///
    /// [`StridedError::InvalidThreadBudget`] when `max_threads` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::ExecContext;
    ///
    /// let pool = rayon::ThreadPoolBuilder::new()
    ///     .num_threads(2)
    ///     .build()
    ///     .unwrap();
    /// assert!(ExecContext::run_on(&pool, 1, |ctx| ctx.is_serial()).unwrap());
    /// let limit = ExecContext::run_on(&pool, 2, |ctx| ctx.max_threads_limit()).unwrap();
    /// assert_eq!(limit.map(|value| value.get()), Some(2));
    /// ```
    pub fn run_on<R: Send>(
        pool: &rayon::ThreadPool,
        max_threads: usize,
        operation: impl FnOnce(ExecContext) -> R + Send,
    ) -> Result<R> {
        let budget = NonZeroUsize::new(max_threads)
            .ok_or(StridedError::InvalidThreadBudget { max_threads })?;
        crate::threading::run_on_pool(pool, budget, move |width| {
            let ctx = if width <= 1 {
                Self::serial()
            } else {
                Self::max_threads(width)?
            };
            Ok(ctx.run(|| operation(ctx)))
        })
    }
}

impl Default for ExecContext {
    #[inline]
    fn default() -> Self {
        Self::serial()
    }
}

/// Execute `operation` against a caller-supplied Rayon pool.
///
/// This is the placement-carrying counterpart of [`with_execution_policy`]:
/// inside the scope the effective policy behaves like [`ExecutionPolicy::Rayon`]
/// with `max_threads` clamped to the pool's worker count, and width queries such
/// as [`crate::execution::rayon_threads`] report against `pool` rather than the
/// ambient pool. Fan-out sites decide their width first on the calling thread;
/// they enter `pool` with [`rayon::ThreadPool::install`] exactly once, and only
/// when that width exceeds one and the caller is not already one of the pool's
/// workers. A width of one, or a call already inside the pool, runs in place.
/// The scope never resolves the global Rayon registry.
///
/// The scope is thread-local in the same way as [`with_execution_policy`]: it is
/// restored even if `operation` panics, but it is not propagated into Rayon
/// tasks or user callbacks. Callers that need isolation at such a boundary must
/// not enter their own Rayon scheduling there.
///
/// # Examples
///
/// ```
/// use core::num::NonZeroUsize;
/// use strided_basic::{map_into, with_execution_pool, StridedArray};
///
/// let pool = rayon::ThreadPoolBuilder::new()
///     .num_threads(2)
///     .build()
///     .unwrap();
/// let source = StridedArray::<f64>::from_fn_col_major(&[4], |index| index[0] as f64);
/// let mut destination = StridedArray::<f64>::col_major(&[4]);
///
/// with_execution_pool(&pool, NonZeroUsize::new(2).unwrap(), || {
///     map_into(&mut destination.view_mut(), &source.view(), |value| value + 1.0).unwrap();
/// });
///
/// assert_eq!(destination.into_data(), vec![1.0, 2.0, 3.0, 4.0]);
/// ```
#[cfg(feature = "parallel")]
#[inline]
pub fn with_execution_pool<R>(
    pool: &rayon::ThreadPool,
    max_threads: NonZeroUsize,
    operation: impl FnOnce() -> R,
) -> R {
    crate::threading::with_execution_pool(pool, max_threads, operation)
}

#[cfg(test)]
#[path = "exec_context/tests/tests.rs"]
mod tests;
