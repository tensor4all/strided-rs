//! Private bookkeeping behind the `ledger` feature.
//!
//! With the feature off, [`Track`] and its guards are zero-sized and every
//! method is an empty inline function, so the hot path carries no cost.

#[cfg(not(feature = "ledger"))]
pub(crate) use off::*;
#[cfg(feature = "ledger")]
pub(crate) use on::*;

#[cfg(not(feature = "ledger"))]
mod off {
    pub(crate) struct Track;
    pub(crate) struct Guard;

    impl Track {
        pub(crate) const NONE: Track = Track;
        #[inline(always)]
        pub(crate) fn root() -> Self {
            Track
        }
        #[inline(always)]
        pub(crate) fn child(&self) -> Self {
            Track
        }
        #[inline(always)]
        pub(crate) fn alias(&self) -> Self {
            Track
        }
        #[inline(always)]
        pub(crate) fn task(&self) -> Guard {
            Guard
        }
        #[inline(always)]
        pub(crate) fn release(&self) -> Guard {
            Guard
        }
        #[inline(always)]
        #[allow(dead_code)] // only called by the parallel scheduler paths
        pub(crate) fn note_install(&self) {}
    }
}

#[cfg(feature = "ledger")]
mod on {
    use crate::ledger::{register, Counts};
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    /// Ledger handle of a token: its root's counters and whether the thread
    /// holding the token is itself counted as a live task of that root.
    pub(crate) struct Track {
        counts: Option<Arc<Counts>>,
        counted: bool,
    }

    pub(crate) struct Guard<'a> {
        counts: Option<&'a Counts>,
        // +1 on drop (a released slot being taken back) or -1 on drop (a task ending).
        end_is_release: bool,
    }

    impl Track {
        pub(crate) const NONE: Track = Track {
            counts: None,
            counted: false,
        };

        pub(crate) fn root() -> Self {
            Track {
                counts: Some(register()),
                counted: false,
            }
        }

        /// Handle given to a job of `fork`: the job runs inside a counted task.
        pub(crate) fn child(&self) -> Self {
            Track {
                counts: self.counts.clone(),
                counted: true,
            }
        }

        /// Handle of a `reborrow`: identical accounting to the parent.
        pub(crate) fn alias(&self) -> Self {
            Track {
                counts: self.counts.clone(),
                counted: self.counted,
            }
        }

        pub(crate) fn root_id(&self) -> Option<u64> {
            self.counts.as_ref().map(|c| c.id)
        }

        /// One task of this root starts running; it ends when the guard drops.
        pub(crate) fn task(&self) -> Guard<'_> {
            if let Some(c) = &self.counts {
                c.enter();
            }
            Guard {
                counts: self.counts.as_deref(),
                end_is_release: false,
            }
        }

        /// The calling task stops counting while its slots are handed to
        /// children; it counts again when the guard drops.
        pub(crate) fn release(&self) -> Guard<'_> {
            let counts = if self.counted {
                self.counts.as_deref()
            } else {
                None
            };
            if let Some(c) = counts {
                c.live.fetch_sub(1, Ordering::SeqCst);
            }
            Guard {
                counts,
                end_is_release: true,
            }
        }

        #[allow(dead_code)] // only called by the parallel scheduler paths
        pub(crate) fn note_install(&self) {
            if let Some(c) = &self.counts {
                c.installs.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            if let Some(c) = self.counts {
                if self.end_is_release {
                    c.enter();
                } else {
                    c.live.fetch_sub(1, Ordering::SeqCst);
                }
            }
        }
    }
}
