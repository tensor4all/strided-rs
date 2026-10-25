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
/// Dropping a `Pool` shuts its workers down.
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
    #[cfg(feature = "parallel")]
    pub(crate) inner: rayon::ThreadPool,
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
        let mut builder = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .stack_size(self.stack_size);
        if let Some(name) = self.thread_name {
            builder = builder.thread_name(name);
        }
        if let Some(hook) = self.on_start {
            builder = builder.start_handler(hook);
        }
        let inner = builder
            .build()
            .map_err(|e| Error::PoolBuild(e.to_string()))?;
        let width = NonZeroUsize::new(inner.current_num_threads()).unwrap_or(NonZeroUsize::MIN);
        Ok(Pool { inner, width })
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
