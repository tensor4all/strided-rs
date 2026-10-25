//! The pool type and its builder.

use crate::error::Error;
use core::fmt;
use core::num::NonZeroUsize;

/// Default worker stack size: 16 MiB.
///
/// Sized for the recursive LU and divide-and-conquer providers that run on pool
/// workers; the same value tenferro-rs used for its own pool preset.
const DEFAULT_STACK_BYTES: usize = 16 << 20;

/// A fixed set of worker threads that [`Exec`](crate::Exec) tokens place work on.
///
/// `Pool` is the only way to make a pool. It wraps the scheduler privately: no
/// scheduler type appears in any signature, there is no constructor from an
/// external pool and no accessor returning one. The pool's owner keeps the
/// `Pool` and mints tokens from it with [`Exec::new`](crate::Exec::new).
///
/// Dropping a `Pool` shuts its workers down and joins them: when `drop`
/// returns, every worker thread has exited, so the thread-local destructors of
/// the workers (including state set up by an [`on_start`](PoolBuilder::on_start)
/// hook) have already run. Without the `parallel` feature there are no workers
/// and nothing is joined.
///
/// A pool dropped from one of its own workers cannot wait for itself: that drop
/// signals shutdown and returns without joining, so the workers exit shortly
/// after instead of before the return. Drop the pool from a plain thread when
/// the guarantee matters.
///
/// # Examples
///
/// ```
/// use strided_exec::Pool;
///
/// let pool = Pool::builder().threads(2).thread_name(|i| format!("w{i}")).build()?;
/// assert!(pool.width().get() <= 2);
/// # Ok::<(), strided_exec::Error>(())
/// ```
pub struct Pool {
    // Field order matters: `inner` is dropped (shutdown signalled) before
    // `joiner` waits for the threads.
    #[cfg(feature = "parallel")]
    pub(crate) inner: rayon::ThreadPool,
    #[cfg(feature = "parallel")]
    _joiner: Joiner,
    pub(crate) width: NonZeroUsize,
}

impl Pool {
    /// Starts configuring a pool.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = strided_exec::Pool::builder().threads(1).build()?;
    /// assert_eq!(pool.width().get(), 1);
    /// # Ok::<(), strided_exec::Error>(())
    /// ```
    pub fn builder() -> PoolBuilder {
        PoolBuilder::new()
    }

    /// Number of worker threads, the largest budget a token can have.
    ///
    /// Without the `parallel` feature this is always 1.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = strided_exec::Pool::builder().threads(1).build()?;
    /// assert_eq!(pool.width().get(), 1);
    /// # Ok::<(), strided_exec::Error>(())
    /// ```
    pub fn width(&self) -> NonZeroUsize {
        self.width
    }
}

/// The worker threads of a pool, joined when it is dropped.
#[cfg(feature = "parallel")]
struct Joiner {
    handles: std::sync::Arc<std::sync::Mutex<Vec<std::thread::JoinHandle<()>>>>,
}

#[cfg(feature = "parallel")]
impl Drop for Joiner {
    fn drop(&mut self) {
        let handles = match self.handles.lock() {
            Ok(mut guard) => std::mem::take(&mut *guard),
            Err(poisoned) => std::mem::take(&mut *poisoned.into_inner()),
        };
        let me = std::thread::current().id();
        // A worker of this pool cannot join itself: skip joining entirely.
        if handles.iter().any(|h| h.thread().id() == me) {
            return;
        }
        for handle in handles {
            // A worker that panicked has still exited; nothing to report.
            let _ = handle.join();
        }
    }
}

impl fmt::Debug for Pool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pool").field("width", &self.width).finish()
    }
}

/// Configuration of a [`Pool`].
///
/// # Examples
///
/// ```
/// use strided_exec::Pool;
///
/// let pool = Pool::builder()
///     .threads(2)
///     .stack_size(8 << 20)
///     .on_start(|_worker| { /* e.g. pin the thread */ })
///     .build()?;
/// assert!(pool.width().get() >= 1);
/// # Ok::<(), strided_exec::Error>(())
/// ```
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub struct PoolBuilder {
    threads: Option<usize>,
    stack_size: usize,
    thread_name: Option<Box<dyn Fn(usize) -> String>>,
    on_start: Option<Box<dyn Fn(usize) + Send + Sync>>,
}

impl PoolBuilder {
    fn new() -> Self {
        PoolBuilder {
            threads: None,
            stack_size: DEFAULT_STACK_BYTES,
            thread_name: None,
            on_start: None,
        }
    }

    /// Number of worker threads. Defaults to the available parallelism.
    ///
    /// Zero is rejected by [`build`](Self::build); the pool never falls back to
    /// an environment-dependent default.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_exec::{Error, Pool};
    ///
    /// assert_eq!(Pool::builder().threads(0).build().unwrap_err(), Error::ZeroThreads);
    /// ```
    #[must_use]
    pub fn threads(mut self, n: usize) -> Self {
        self.threads = Some(n);
        self
    }

    /// Names the workers; the closure receives the worker index.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = strided_exec::Pool::builder()
    ///     .threads(1)
    ///     .thread_name(|i| format!("my-pool-{i}"))
    ///     .build()?;
    /// # let _ = pool;
    /// # Ok::<(), strided_exec::Error>(())
    /// ```
    #[must_use]
    pub fn thread_name(mut self, f: impl Fn(usize) -> String + 'static) -> Self {
        self.thread_name = Some(Box::new(f));
        self
    }

    /// Worker stack size in bytes. Defaults to 16 MiB, sized for recursive LU.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = strided_exec::Pool::builder().threads(1).stack_size(4 << 20).build()?;
    /// # let _ = pool;
    /// # Ok::<(), strided_exec::Error>(())
    /// ```
    #[must_use]
    pub fn stack_size(mut self, bytes: usize) -> Self {
        self.stack_size = bytes;
        self
    }

    /// Hook run on each worker when it starts, with the worker index.
    ///
    /// Typical use is pinning the worker to a core. The hook is not called
    /// without the `parallel` feature, which starts no workers.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = strided_exec::Pool::builder().threads(1).on_start(|_| {}).build()?;
    /// # let _ = pool;
    /// # Ok::<(), strided_exec::Error>(())
    /// ```
    #[must_use]
    pub fn on_start(mut self, f: impl Fn(usize) + Send + Sync + 'static) -> Self {
        self.on_start = Some(Box::new(f));
        self
    }

    /// Starts the workers.
    ///
    /// Without the `parallel` feature no thread is started and the pool has
    /// width 1.
    ///
    /// # Errors
    ///
    /// [`Error::ZeroThreads`] when `threads(0)` was requested, and
    /// [`Error::PoolBuild`] when the workers cannot be started.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = strided_exec::Pool::builder().threads(2).build()?;
    /// assert!(pool.width().get() <= 2);
    /// # Ok::<(), strided_exec::Error>(())
    /// ```
    pub fn build(self) -> Result<Pool, Error> {
        if self.threads == Some(0) {
            return Err(Error::ZeroThreads);
        }
        self.build_checked()
    }

    #[cfg(feature = "parallel")]
    fn build_checked(self) -> Result<Pool, Error> {
        let n = self
            .threads
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, NonZeroUsize::get));
        let handles = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = std::sync::Arc::clone(&handles);
        let mut builder = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .stack_size(self.stack_size)
            .spawn_handler(move |thread| {
                let mut b = std::thread::Builder::new();
                if let Some(name) = thread.name() {
                    b = b.name(name.to_owned());
                }
                if let Some(size) = thread.stack_size() {
                    b = b.stack_size(size);
                }
                let handle = b.spawn(move || thread.run())?;
                match sink.lock() {
                    Ok(mut guard) => guard.push(handle),
                    Err(poisoned) => poisoned.into_inner().push(handle),
                }
                Ok(())
            });
        if let Some(name) = self.thread_name {
            builder = builder.thread_name(name);
        }
        if let Some(hook) = self.on_start {
            builder = builder.start_handler(hook);
        }
        // On failure `builder` drops its handler, but threads already started
        // are signalled by rayon and detached; join them here for determinism.
        let built = builder.build();
        let inner = match built {
            Ok(inner) => inner,
            Err(e) => {
                drop(Joiner { handles });
                return Err(Error::PoolBuild(e.to_string()));
            }
        };
        let width = NonZeroUsize::new(inner.current_num_threads()).unwrap_or(NonZeroUsize::MIN);
        Ok(Pool {
            inner,
            _joiner: Joiner { handles },
            width,
        })
    }

    #[cfg(not(feature = "parallel"))]
    fn build_checked(self) -> Result<Pool, Error> {
        Ok(Pool {
            width: NonZeroUsize::MIN,
        })
    }
}

impl fmt::Debug for PoolBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PoolBuilder")
            .field("threads", &self.threads)
            .field("stack_size", &self.stack_size)
            .finish_non_exhaustive()
    }
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    /// The last reference is dropped on a worker of the pool itself: the drop
    /// must not wait for its own thread.
    #[test]
    fn drop_from_a_worker_of_the_pool_does_not_hang() {
        let pool = Arc::new(Pool::builder().threads(2).build().unwrap());
        let last = Arc::clone(&pool);
        let (gate_tx, gate_rx) = mpsc::channel::<()>();
        let (done_tx, done_rx) = mpsc::channel::<()>();
        pool.inner.spawn(move || {
            gate_rx.recv().unwrap();
            drop(last); // the last reference: runs `Pool::drop` on this worker
            done_tx.send(()).unwrap();
        });
        drop(pool);
        gate_tx.send(()).unwrap();
        done_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("drop on a worker hung");
    }
}
