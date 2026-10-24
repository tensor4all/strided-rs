//! Error types of the crate.

/// Failure to build a [`Pool`](crate::Pool) or an [`Exec`](crate::Exec).
///
/// The enum is `#[non_exhaustive]`: new refusals may be added in a minor
/// release, so match with a wildcard arm.
///
/// # Examples
///
/// ```
/// use strided_exec::{Error, Exec, Pool};
///
/// let pool = Pool::builder().threads(1).build()?;
/// assert_eq!(Exec::new(&pool, 0).unwrap_err(), Error::ZeroBudget);
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A token was requested with a budget of zero slots.
    #[error("an execution budget must be at least one slot")]
    ZeroBudget,
    /// A pool was requested with zero worker threads.
    #[error("a pool needs at least one worker thread")]
    ZeroThreads,
    /// The operating system or the scheduler refused to start the workers.
    #[error("failed to start the pool workers: {0}")]
    PoolBuild(String),
}

/// A [`broadcast`](crate::Exec::broadcast) that cannot guarantee co-scheduling.
///
/// Nothing was run when this is returned. The caller falls back to a kernel
/// that does not need all participants at once.
///
/// # Examples
///
/// ```
/// use strided_exec::Refused;
///
/// assert!(Refused::ForkChild.to_string().contains("fork"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Refused {
    /// The token was produced by [`fork`](crate::Exec::fork) (or by a
    /// [`reborrow`](crate::Exec::reborrow) of such a token): its slots may share
    /// workers with sibling tasks, so its participants could wait on each other.
    #[error("the token is a child produced by fork; its slots may share workers with siblings")]
    ForkChild,
    /// The caller is already a worker of the token's pool, so it would occupy
    /// one of the slots the participants need.
    #[error("the caller is a worker of the token's pool and would occupy a participant slot")]
    CallerInPool,
}
