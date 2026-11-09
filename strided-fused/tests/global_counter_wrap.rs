//! Wrap crossing of the process-wide allocation counter.
//!
//! In its own test binary on purpose: `GLOBAL_ALLOCATIONS` counts every thread
//! in the process, so saturating it inside the shared allocation-test binary
//! would let other tests' allocations land in the window, and this window would
//! count theirs. One test, nothing else running.

#![cfg(feature = "parallel")]

mod common;

use std::sync::atomic::Ordering;

use common::{count_global_allocations, record_allocation, GLOBAL_ALLOCATIONS};

/// Every counted delta wraps, so a measurement window that crosses the wrap
/// reports the one operation it recorded instead of underflowing. The counter
/// is restored before anything is asserted, because a failing assertion
/// allocates and would otherwise be counted.
#[test]
fn global_delta_wraps_instead_of_underflowing() {
    let saved = GLOBAL_ALLOCATIONS.swap(usize::MAX, Ordering::SeqCst);
    let allocations = count_global_allocations(record_allocation);
    GLOBAL_ALLOCATIONS.swap(saved, Ordering::SeqCst);
    assert_eq!(allocations, 1);
}
