//! Execution policy passed through erased kernel replay boundaries.
//!
//! `ExecContext` is explicit even for kernels that are currently serial so
//! downstream runtimes do not accidentally bake ambient thread-pool state into
//! their prepared-kernel ABI.

use core::num::NonZeroUsize;

use crate::{Result, StridedError};

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
}

impl Default for ExecContext {
    #[inline]
    fn default() -> Self {
        Self::serial()
    }
}

#[cfg(test)]
mod tests {
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
}
