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

pub mod block;
pub mod copy;
pub mod fuse;
pub mod kernel;
pub mod order;

// Re-export primary API
pub use copy::{copy_into, copy_into_col_major, try_fuse_group};
pub use fuse::{compress_dims, compute_costs, compute_importance, fuse_dims, sort_by_importance};
pub use kernel::{
    build_plan_fused, build_plan_fused_small, for_each_inner_block_preordered, total_len,
    KernelPlan, SMALL_TENSOR_THRESHOLD,
};
pub use order::compute_order;

// Constants
pub const BLOCK_MEMORY_SIZE: usize = 32 * 1024;
pub const CACHE_LINE_SIZE: usize = 64;
