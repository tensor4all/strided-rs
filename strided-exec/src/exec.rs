//! The splittable placement token.

use crate::error::{Error, Refused};
use crate::pool::Pool;
use crate::sched;
use crate::split;
use crate::track::Track;
use core::fmt;
use core::num::NonZeroUsize;
use core::ops::Range;
use core::sync::atomic::{AtomicUsize, Ordering};

/// `budget` slots of a [`Pool`], or the calling thread alone.
///
/// An `Exec` is the only way kernels start parallel work. It is a *token*:
///
/// * It is neither `Clone` nor `Copy`, and every parallel entry consumes it, so
///   a kernel cannot run two phases at full width at the same time. Use
///   [`reborrow`](Exec::reborrow) to run several phases in sequence.
/// * Only [`Exec::new`] mints a token, and it needs the `&Pool`; a token never
///   hands its pool back out, so a kernel cannot mint a fresh full-budget token.
/// * [`fork`](Exec::fork) divides the budget among its tasks (shares differ by
///   at most one and sum to the budget), so the number of tasks running at once
///   under one root token is at most its budget at every nesting depth.
///
/// A parent waiting for its children keeps no slot.
///
/// # Placement
///
/// Serial, budget one, or one task: the work runs inline on the calling thread
/// and the scheduler is never called, so no global pool is started. The caller
/// is a worker of the pool: the tasks run as a join tree on the caller, nothing is
/// installed. The caller is a plain thread or a worker of a different pool: the
/// pool is entered once and the caller waits; the tasks run on the target pool.
/// No thread-local state is involved at any point.
///
/// # Examples
///
/// ```
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use strided_exec::{Exec, Pool};
///
/// let pool = Pool::builder().threads(2).build()?;
/// let exec = Exec::new(&pool, 2)?;
/// let sum = AtomicUsize::new(0);
/// exec.for_ranges(100, 1.try_into().unwrap(), |r| {
///     sum.fetch_add(r.sum::<usize>(), Ordering::Relaxed);
/// });
/// assert_eq!(sum.into_inner(), 4950);
/// # Ok::<(), strided_exec::Error>(())
/// ```
///
/// A token cannot be duplicated:
///
/// ```compile_fail
/// fn needs_clone<T: Clone>() {}
/// needs_clone::<strided_exec::Exec<'static>>();
/// ```
///
/// ```compile_fail
/// fn needs_copy<T: Copy>() {}
/// needs_copy::<strided_exec::Exec<'static>>();
/// ```
///
/// ```compile_fail
/// let exec = strided_exec::Exec::serial();
/// exec.fork(1, |_, _| {});
/// exec.fork(1, |_, _| {}); // `fork` consumed the token
/// ```
#[must_use = "an Exec is a budget of execution slots; pass it to a kernel"]
pub struct Exec<'p> {
    pool: Option<&'p Pool>,
    budget: NonZeroUsize,
    /// Produced by `fork` (directly or through `reborrow`).
    child: bool,
    track: Track,
}

impl Exec<'static> {
    /// A token that runs everything inline on the calling thread.
    ///
    /// It never touches a scheduler, so using it cannot start any pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_exec::Exec;
    ///
    /// let exec = Exec::serial();
    /// assert!(exec.is_serial());
    /// let here = std::thread::current().id();
    /// exec.fork(3, |_, _| assert_eq!(std::thread::current().id(), here));
    /// ```
    pub const fn serial() -> Exec<'static> {
        Exec {
            pool: None,
            budget: NonZeroUsize::MIN,
            child: false,
            track: Track::NONE,
        }
    }
}

impl<'p> Exec<'p> {
    /// A token of `budget` slots of `pool`, clamped to the pool's width.
    ///
    /// This is the only place a budget is clamped.
    ///
    /// # Errors
    ///
    /// [`Error::ZeroBudget`] when `budget` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_exec::{Error, Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(2).build()?;
    /// let exec = Exec::new(&pool, 64)?;
    /// assert_eq!(exec.budget(), pool.width());
    /// assert_eq!(Exec::new(&pool, 0).unwrap_err(), Error::ZeroBudget);
    /// # Ok::<(), Error>(())
    /// ```
    pub fn new(pool: &'p Pool, budget: usize) -> Result<Exec<'p>, Error> {
        let budget = NonZeroUsize::new(budget).ok_or(Error::ZeroBudget)?;
        Ok(Exec {
            pool: Some(pool),
            budget: budget.min(pool.width()),
            child: false,
            track: Track::root(),
        })
    }

    /// The number of slots this token stands for.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(strided_exec::Exec::serial().budget().get(), 1);
    /// ```
    pub fn budget(&self) -> NonZeroUsize {
        self.budget
    }

    /// Whether no parallelism is possible: the budget is one slot.
    ///
    /// Kernels use this to take their serial path directly.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(strided_exec::Exec::serial().is_serial());
    /// ```
    pub fn is_serial(&self) -> bool {
        self.budget.get() == 1
    }

    /// Narrows the token to at most `max` slots.
    ///
    /// The result has budget `min(budget, max)` and the same pool; the dropped
    /// slots are simply not used, so slots are conserved. Callers narrow before
    /// [`broadcast`](Exec::broadcast) or [`external`](Exec::external) to match a team size chosen
    /// by their own work model. Co-scheduling eligibility is unchanged by
    /// `limit`: a root token stays eligible for `broadcast`, a child produced by
    /// [`fork`](Exec::fork) stays ineligible.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use strided_exec::{Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(4).build()?;
    /// let exec = Exec::new(&pool, 4)?.limit(NonZeroUsize::new(2).unwrap());
    /// assert!(exec.budget().get() <= 2);
    /// # Ok::<(), strided_exec::Error>(())
    /// ```
    pub fn limit(self, max: NonZeroUsize) -> Exec<'p> {
        Exec {
            budget: self.budget.min(max),
            ..self
        }
    }

    /// A token with the same pool and budget, borrowed from `self`.
    ///
    /// Parallel entries consume the token, but a kernel often runs several
    /// parallel phases in sequence. Each phase takes a reborrow. The parent is
    /// exclusively borrowed and unusable while the child lives, so the slots are
    /// held by one token at a time: this conserves slots exactly as handing the
    /// token over would.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_exec::Exec;
    ///
    /// let mut exec = Exec::serial();
    /// exec.reborrow().fork(2, |_, _| {});
    /// exec.reborrow().fork(2, |_, _| {});
    /// exec.fork(2, |_, _| {});
    /// ```
    pub fn reborrow(&mut self) -> Exec<'_> {
        Exec {
            pool: self.pool,
            budget: self.budget,
            child: self.child,
            track: self.track.alias(),
        }
    }

    /// Runs `jobs` jobs on `min(jobs, budget)` tasks; job `i` gets a child token.
    ///
    /// The budget is split into one share per task, shares differing by at most
    /// one and summing to the budget. A job's child token carries its task's
    /// share and may be passed to any kernel, which can `fork` again inside it.
    /// When `jobs` exceeds the budget the tasks take job indices from a shared
    /// counter, so the load balances; otherwise job `i` runs on task `i`.
    ///
    /// One task, a serial token, or `jobs <= 1` runs inline on the calling
    /// thread without any scheduler call. Returns when every job has finished.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    /// use strided_exec::{Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(2).build()?;
    /// let hits: Vec<AtomicUsize> = (0..5).map(|_| AtomicUsize::new(0)).collect();
    /// Exec::new(&pool, 2)?.fork(5, |i, _child| {
    ///     hits[i].fetch_add(1, Ordering::Relaxed);
    /// });
    /// assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    /// # Ok::<(), strided_exec::Error>(())
    /// ```
    pub fn fork<F>(self, jobs: usize, f: F)
    where
        F: Fn(usize, Exec<'_>) + Sync,
    {
        if jobs == 0 {
            return;
        }
        let budget = self.budget.get();
        let tasks = jobs.min(budget);
        let _released = self.track.release();
        let next = AtomicUsize::new(0);
        let this = &self;
        let body = |k: usize| {
            let _live = this.track.task();
            let share = split::share(budget, tasks, k);
            let child = || Exec {
                pool: this.pool,
                // `share >= 1` because `tasks <= budget`.
                budget: NonZeroUsize::new(share).unwrap_or(NonZeroUsize::MIN),
                child: true,
                track: this.track.child(),
            };
            if jobs <= tasks {
                f(k, child());
            } else {
                loop {
                    let job = next.fetch_add(1, Ordering::Relaxed);
                    if job >= jobs {
                        break;
                    }
                    f(job, child());
                }
            }
        };
        match self.pool {
            Some(pool) if tasks > 1 => sched::run_tasks(pool, tasks, &self.track, body),
            _ => {
                for k in 0..tasks {
                    body(k);
                }
            }
        }
    }

    /// Data-parallel loop over `0..len` in contiguous ranges.
    ///
    /// Uses at most `budget` tasks; every range is at least `grain` long except
    /// possibly the last, and together the ranges cover `0..len` exactly once.
    /// When one task suffices (`len <= grain`, a serial token) `f(0..len)` runs
    /// inline on the calling thread without any scheduler call. `len == 0` does
    /// nothing.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    /// use strided_exec::{Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(2).build()?;
    /// let covered = AtomicUsize::new(0);
    /// Exec::new(&pool, 2)?.for_ranges(10, 4.try_into().unwrap(), |r| {
    ///     covered.fetch_add(r.len(), Ordering::Relaxed);
    /// });
    /// assert_eq!(covered.into_inner(), 10);
    /// # Ok::<(), strided_exec::Error>(())
    /// ```
    pub fn for_ranges<F>(self, len: usize, grain: NonZeroUsize, f: F)
    where
        F: Fn(Range<usize>) + Sync,
    {
        if len == 0 {
            return;
        }
        let (chunk, tasks) = split::range_plan(len, grain.get(), self.budget.get());
        let _released = self.track.release();
        let track = &self.track;
        let body = |k: usize| {
            let _live = track.task();
            f(split::chunk_range(len, chunk, k));
        };
        match self.pool {
            Some(pool) if tasks > 1 => sched::run_tasks(pool, tasks, track, body),
            _ => {
                for k in 0..tasks {
                    body(k);
                }
            }
        }
    }

    /// Co-scheduling for barrier-bearing kernels: runs `f(i)` for every
    /// `i in 0..budget` with all participants running concurrently, or refuses.
    ///
    /// A serial or budget-one token runs `f(0)` inline and returns `Ok`.
    /// Otherwise the pool is entered once and every participant runs on its own
    /// worker, so `f` may block on a barrier shared by the `budget`
    /// participants.
    ///
    /// # Errors
    ///
    /// Nothing runs and [`Refused`] is returned when co-scheduling cannot be
    /// guaranteed:
    ///
    /// * [`Refused::ForkChild`]: the token was produced by [`fork`](Exec::fork)
    ///   (or derived from such a token by `reborrow` or `limit`); its slots may
    ///   share workers with siblings. Roots from [`Exec::new`], and their
    ///   `reborrow`s and `limit`s, stay eligible.
    ///
    /// Callers must handle `Refused` by falling back to a barrier-free path.
    /// * [`Refused::CallerInPool`]: the calling thread is a worker of the
    ///   token's pool and would occupy a slot.
    ///
    /// # Guarantee and limits
    ///
    /// Concurrent broadcasts on one pool are queued on every worker in the same
    /// order, so two of them cannot each hold half the workers. The participants
    /// are the pool's workers `0..budget`; if unrelated tasks that never finish
    /// occupy those workers, the broadcast waits for them. Keep barrier kernels on a pool whose workers are not held
    /// by blocking work of another token.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Barrier;
    /// use strided_exec::{Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(2).build()?;
    /// let exec = Exec::new(&pool, 2)?;
    /// let barrier = Barrier::new(exec.budget().get());
    /// exec.broadcast(|_i| {
    ///     barrier.wait(); // every participant is running, or this call refused
    /// }).expect("called from a plain thread on a root token");
    /// # Ok::<(), strided_exec::Error>(())
    /// ```
    pub fn broadcast<F>(self, f: F) -> Result<(), Refused>
    where
        F: Fn(usize) + Sync,
    {
        if let Some(refused) = self.refusal() {
            return Err(refused);
        }
        let pool = match self.pool {
            Some(pool) if self.budget.get() > 1 => pool,
            _ => {
                let _live = self.track.task();
                f(0);
                return Ok(());
            }
        };
        let _released = self.track.release();
        #[cfg(feature = "parallel")]
        sched::broadcast(pool, &self.track, self.budget.get(), f);
        #[cfg(not(feature = "parallel"))]
        {
            let _ = pool;
            f(0);
        }
        Ok(())
    }

    /// Whether [`broadcast`](Exec::broadcast) would run right now, without
    /// consuming the token.
    ///
    /// True exactly when `broadcast` would not return [`Refused`]: the token is
    /// not derived from [`fork`](Exec::fork) and the calling thread is not a
    /// worker of its pool. A serial or budget-one token returns true, because
    /// `broadcast` then runs `f(0)` inline. The answer depends on the calling
    /// thread, so ask it on the thread that will broadcast.
    ///
    /// Use it before building an expensive setup that only a broadcast needs,
    /// instead of paying for the setup and then being refused.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_exec::{Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(2).build()?;
    /// let exec = Exec::new(&pool, 2)?;
    /// assert!(exec.can_broadcast());
    /// exec.fork(1, |_, child| {
    ///     let predicted = child.can_broadcast();
    ///     assert_eq!(child.broadcast(|_| {}).is_ok(), predicted);
    /// });
    /// assert!(Exec::serial().can_broadcast());
    /// # Ok::<(), strided_exec::Error>(())
    /// ```
    pub fn can_broadcast(&self) -> bool {
        self.refusal().is_none()
    }

    /// The single refusal rule shared by `broadcast` and `can_broadcast`.
    fn refusal(&self) -> Option<Refused> {
        let pool = match self.pool {
            Some(pool) if self.budget.get() > 1 => pool,
            _ => return None,
        };
        if self.child {
            return Some(Refused::ForkChild);
        }
        if sched::in_pool(pool) {
            return Some(Refused::CallerInPool);
        }
        None
    }

    /// Hands the budget as a plain width to a library outside this stack.
    ///
    /// This is the only place a bare width exists. The token is consumed, the
    /// pool is entered once (inline when the token is serial, has a budget of
    /// one, or the caller already is a worker of the pool) and `f` receives the
    /// budget, for example to build `faer::Par::rayon(width)` or to set a vendor
    /// BLAS thread count. Whatever the library schedules with that width runs on
    /// this pool, never on an ambient one. How the library divides the width is
    /// its own responsibility. Without the `parallel` feature, `f` runs inline
    /// and receives a width of one.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_exec::Exec;
    ///
    /// let width = Exec::serial().external(|width| width.get());
    /// assert_eq!(width, 1);
    /// ```
    pub fn external<R: Send>(self, f: impl FnOnce(NonZeroUsize) -> R + Send) -> R {
        #[cfg(feature = "parallel")]
        if let Some(pool) = self.pool {
            if self.budget.get() > 1 {
                let width = self.budget;
                let _released = self.track.release();
                return sched::enter(pool, &self.track, move || f(width));
            }
        }
        f(NonZeroUsize::MIN)
    }

    /// Root id of this token in the [`ledger`](crate::ledger), if it has one.
    ///
    /// Tokens from [`Exec::new`] are roots; children and reborrows report the
    /// root of their origin; [`Exec::serial`] has none.
    ///
    /// Available with the `ledger` cargo feature.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_exec::{Exec, Pool};
    ///
    /// let pool = Pool::builder().threads(1).build()?;
    /// assert!(Exec::new(&pool, 1)?.ledger_root().is_some());
    /// assert!(Exec::serial().ledger_root().is_none());
    /// # Ok::<(), strided_exec::Error>(())
    /// ```
    #[cfg(feature = "ledger")]
    pub fn ledger_root(&self) -> Option<u64> {
        self.track.root_id()
    }
}

impl fmt::Debug for Exec<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Exec")
            .field("budget", &self.budget)
            .field("pooled", &self.pool.is_some())
            .field("child", &self.child)
            .finish()
    }
}
