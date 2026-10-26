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
//! ([`copy_ns_per_element`] times its elements) against the dispatch cost of one
//! section (`Exec::width_for`), so a copy too small to repay a fan-out runs on the
//! calling thread whatever the budget. The crate never reaches a global
//! pool; pass `Exec::serial()` for a copy on the calling thread.

#[cfg(test)]
mod block;
mod copy;
mod fuse;
mod host;
mod hptt;
#[cfg(test)]
#[allow(dead_code)]
mod kernel;
#[cfg(test)]
mod order;

// Re-export primary API
pub use copy::{copy_into, copy_into_col_major, copy_into_uninit};
pub use fuse::{plan_bilateral_fusion, BilateralFusionPlan, FusionPlanError};
pub use host::{copy_ns_per_element, copy_ns_per_element_of, HostClass};

// Constants
pub const BLOCK_MEMORY_SIZE: usize = 32 * 1024;
pub const CACHE_LINE_SIZE: usize = 64;
