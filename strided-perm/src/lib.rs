//! Cache-efficient tensor permutation / transpose.
//!
//! This crate provides optimized copy and permutation operations for strided
//! multidimensional arrays. It is designed as a single-responsibility crate
//! sitting between `strided-view` (data structures) and `strided-kernel`
//! (general map/reduce/broadcast operations).
//!
//! # Dependency graph
//!
//! ```text
//! strided-view -> strided-perm -> strided-kernel
//! ```
//!
//! # Threading
//!
//! Every copy takes a [`threadteam::Exec`] placement token as its last
//! parameter and splits the permutation within that budget. The width is
//! decided by the token: the serial time of the copy
//! ([`COPY_NS_PER_ELEMENT`] times its elements) against the dispatch cost of one
//! section (`Exec::width_for`), so a copy too small to repay a fan-out runs on the
//! calling thread whatever the budget. The crate never reaches a global
//! pool; pass `Exec::serial()` for a copy on the calling thread.

#[cfg(test)]
mod block;
mod copy;
mod fuse;
mod hptt;
#[cfg(test)]
#[allow(dead_code)]
mod kernel;
#[cfg(test)]
mod order;

// Re-export primary API
pub use copy::{copy_into, copy_into_col_major, copy_into_uninit};
pub use fuse::{plan_bilateral_fusion, BilateralFusionPlan, FusionPlanError};

/// Serial throughput of a copy in nanoseconds per element.
///
/// The `Copy` class of the strided stack (`strided_basic::execution::KernelClass`
/// holds the class table; this crate sits below it and owns the one constant it
/// needs). `copy_into*` multiply it by the element count and ask the token
/// whether one section pays.
pub const COPY_NS_PER_ELEMENT: f64 = 0.090;

// Constants
pub const BLOCK_MEMORY_SIZE: usize = 32 * 1024;
pub const CACHE_LINE_SIZE: usize = 64;
