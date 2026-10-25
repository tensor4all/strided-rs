//! Test support: a per-root ledger of live tasks and pool entries.
//!
//! Enabled by the `ledger` cargo feature. Every token made by
//! [`Exec::new`](crate::Exec::new) is a *root*; tokens produced from it by
//! [`fork`](crate::Exec::fork) or [`reborrow`](crate::Exec::reborrow) report to
//! the same root. A task that hands its slots to children (by entering the
//! token again) stops counting while it waits, so the number of live tasks per
//! root is the number of slots actually in use. Consumers assert
//! `max_live(root) <= budget` in their own test suites.
//!
//! # Examples
//!
//! ```
//! use strided_exec::{ledger, Exec, Pool};
//!
//! let pool = Pool::builder().threads(2).build()?;
//! let exec = Exec::new(&pool, 2)?;
//! let root = exec.ledger_root().expect("a token from `new` is a root");
//! exec.fork(4, |_, _| {});
//! assert!(ledger::max_live(root) <= 2);
//! # Ok::<(), strided_exec::Error>(())
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

pub(crate) struct Counts {
    pub(crate) id: u64,
    pub(crate) live: AtomicUsize,
    pub(crate) max: AtomicUsize,
    pub(crate) installs: AtomicUsize,
}

impl Counts {
    pub(crate) fn enter(&self) {
        let now = self.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.max.fetch_max(now, Ordering::SeqCst);
    }
}

fn registry() -> &'static Mutex<HashMap<u64, Arc<Counts>>> {
    static REG: OnceLock<Mutex<HashMap<u64, Arc<Counts>>>> = OnceLock::new();
    REG.get_or_init(Default::default)
}

pub(crate) fn register() -> Arc<Counts> {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let counts = Arc::new(Counts {
        id: NEXT.fetch_add(1, Ordering::Relaxed),
        live: AtomicUsize::new(0),
        max: AtomicUsize::new(0),
        installs: AtomicUsize::new(0),
    });
    registry()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(counts.id, counts.clone());
    counts
}

fn lookup(root: u64) -> Option<Arc<Counts>> {
    registry()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&root)
        .cloned()
}

/// Largest number of tasks of `root` that were live at the same moment.
///
/// Returns 0 for an unknown root.
///
/// # Examples
///
/// ```
/// use strided_exec::{ledger, Exec, Pool};
///
/// let pool = Pool::builder().threads(1).build()?;
/// let exec = Exec::new(&pool, 1)?;
/// let root = exec.ledger_root().unwrap();
/// exec.fork(3, |_, _| {});
/// assert_eq!(ledger::max_live(root), 1);
/// # Ok::<(), strided_exec::Error>(())
/// ```
pub fn max_live(root: u64) -> usize {
    lookup(root).map_or(0, |c| c.max.load(Ordering::SeqCst))
}

/// Number of tasks of `root` live right now.
///
/// # Examples
///
/// ```
/// use strided_exec::{ledger, Exec, Pool};
///
/// let pool = Pool::builder().threads(1).build()?;
/// let exec = Exec::new(&pool, 1)?;
/// let root = exec.ledger_root().unwrap();
/// exec.fork(2, |_, _| {});
/// assert_eq!(ledger::live(root), 0);
/// # Ok::<(), strided_exec::Error>(())
/// ```
pub fn live(root: u64) -> usize {
    lookup(root).map_or(0, |c| c.live.load(Ordering::SeqCst))
}

/// Number of times an entry of `root` had to leave its calling thread to
/// install work into the pool (a plain thread or a worker of another pool).
///
/// Entries made from a worker of the target pool never count: they spawn into
/// the current scope without installing. Inline entries never count either.
///
/// # Examples
///
/// ```
/// use strided_exec::{ledger, Exec, Pool};
///
/// let pool = Pool::builder().threads(2).build()?;
/// let exec = Exec::new(&pool, 2)?;
/// let root = exec.ledger_root().unwrap();
/// exec.fork(1, |_, _| {});
/// assert_eq!(ledger::installs(root), 0, "one task runs inline");
/// # Ok::<(), strided_exec::Error>(())
/// ```
pub fn installs(root: u64) -> usize {
    lookup(root).map_or(0, |c| c.installs.load(Ordering::SeqCst))
}

/// Whether the process-global Rayon pool is still unstarted.
///
/// Answers by starting it: the call installs a one-thread global pool, which
/// succeeds only when nothing in the process has touched the global registry
/// before. Call it once, at the end of a test binary that holds a single test,
/// so that no other test in the process can start the global pool first.
/// Without the `parallel` feature nothing can start a pool and it returns
/// `true`.
///
/// # Examples
///
/// ```no_run
/// use strided_exec::{ledger, Exec};
///
/// Exec::serial().fork(4, |_, _| {});
/// assert!(ledger::global_pool_unstarted());
/// ```
pub fn global_pool_unstarted() -> bool {
    #[cfg(feature = "parallel")]
    {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .is_ok()
    }
    #[cfg(not(feature = "parallel"))]
    {
        true
    }
}
