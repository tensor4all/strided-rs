//! Splittable CPU placement for the strided stack.
//!
//! `strided-exec` is the single place that starts tasks. Kernels never see a
//! scheduler type: they take an [`Exec`], a token for `budget` slots of a
//! [`Pool`], and split it with [`Exec::fork`], [`Exec::for_ranges`],
//! [`Exec::broadcast`] or (feature `faer`) `Exec::faer`.
//!
//! * A [`Pool`] is built once by its owner with [`Pool::builder`].
//! * [`Exec::new`] mints a token from a pool; [`Exec::serial`] needs none.
//! * A token is not `Clone`/`Copy`; entries consume it; splits conserve slots.
//! * Serial, budget-one and one-task entries run inline and never reach any
//!   scheduler, so the process-global Rayon pool is never started implicitly.
//!
//! # Features
//!
//! * `parallel` (default): run on a private pool. Without it the same types
//!   exist, every pool has width 1 and every entry runs inline.
//! * `faer`: `Exec::faer`, handing faer a `Par` sized from the budget.
//! * `ledger`: test support, see the [`ledger`] module.
//!
//! # Examples
//!
//! ```
//! use std::sync::atomic::{AtomicUsize, Ordering};
//! use strided_exec::{Exec, Pool};
//!
//! let pool = Pool::builder().threads(4).build()?;
//! let mut exec = Exec::new(&pool, 4)?;
//!
//! // Phase 1: data-parallel loop.
//! let total = AtomicUsize::new(0);
//! exec.reborrow().for_ranges(1000, 64.try_into().unwrap(), |r| {
//!     total.fetch_add(r.len(), Ordering::Relaxed);
//! });
//! assert_eq!(total.into_inner(), 1000);
//!
//! // Phase 2: two independent parts, each with its share of the slots.
//! exec.fork(2, |_i, child| {
//!     assert!(child.budget().get() <= 2);
//! });
//! # Ok::<(), strided_exec::Error>(())
//! ```

mod error;
mod exec;
#[cfg(feature = "ledger")]
pub mod ledger;
mod pool;
mod sched;
mod split;
mod track;

pub use error::{Error, Refused};
pub use exec::Exec;
pub use pool::{Pool, PoolBuilder};

#[cfg(test)]
mod tests;
