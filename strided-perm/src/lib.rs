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
//! Every copy takes a [`strided_exec::Exec`] placement token as its last
//! parameter and splits the permutation within that budget. A copy of at most
//! [`MINTHREADLENGTH`] elements runs on the calling thread whatever the budget,
//! because a fan-out costs more than the copy. The crate never reaches a global
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

/// Minimum number of elements for a copy to fan out.
///
/// The single threshold of the strided stack: `copy_into*` narrow the token to
/// one task at or below it, and `strided-basic` re-exports this constant for
/// its own entries. Matches Julia's `MINTHREADLENGTH = 1 << 15`; see
/// `docs/design/erased-execution-policy.md` for the decision record.
pub const MINTHREADLENGTH: usize = 1 << 15;

// Constants
pub const BLOCK_MEMORY_SIZE: usize = 32 * 1024;
pub const CACHE_LINE_SIZE: usize = 64;
