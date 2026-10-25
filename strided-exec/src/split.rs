//! Pure arithmetic of slot shares and range chunks.

use core::ops::Range;

/// Share of task `k` when `budget` slots are split over `tasks` tasks.
///
/// Shares differ by at most one and sum to `budget` over `k in 0..tasks`.
pub(crate) fn share(budget: usize, tasks: usize, k: usize) -> usize {
    debug_assert!(tasks >= 1 && k < tasks);
    budget / tasks + usize::from(k < budget % tasks)
}

/// Chunk length and task count for a `for_ranges` loop.
///
/// Every chunk is at least `grain` long except possibly the last, and there
/// are at most `budget` chunks. `len` must be non-zero.
pub(crate) fn range_plan(len: usize, grain: usize, budget: usize) -> (usize, usize) {
    debug_assert!(len > 0 && grain > 0 && budget > 0);
    let tasks = budget.min(len.div_ceil(grain));
    let chunk = len.div_ceil(tasks).max(grain);
    (chunk, len.div_ceil(chunk))
}

/// The `k`-th chunk of a plan returned by [`range_plan`].
pub(crate) fn chunk_range(len: usize, chunk: usize, k: usize) -> Range<usize> {
    let start = k * chunk;
    start..(start + chunk).min(len)
}
