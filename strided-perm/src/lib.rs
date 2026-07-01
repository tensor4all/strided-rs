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
//! strided-view -> strided-perm -> strided-kernel -> strided-einsum2
//! ```

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
pub use copy::{copy_into, copy_into_col_major};
#[cfg(feature = "parallel")]
pub use copy::{copy_into_col_major_par, copy_into_par};

// Constants
pub const BLOCK_MEMORY_SIZE: usize = 32 * 1024;
pub const CACHE_LINE_SIZE: usize = 64;
