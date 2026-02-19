//! HPTT-faithful cache-efficient tensor permutation.
//!
//! Implements the key techniques from HPTT (High-Performance Tensor Transpose):
//! 1. Bilateral dimension fusion (fuse dims contiguous in both src and dst)
//! 2. 2D micro-kernel transpose (4×4 scalar for f64, 8×8 for f32)
//! 3. Macro-kernel: BLOCK × BLOCK tile via grid of micro-kernel calls
//! 4. Recursive ComputeNode loop nest (only stride-1 dims get blocked)
//! 5. ConstStride1 fast path when src and dst stride-1 dims coincide

mod execute;
mod macro_kernel;
pub(crate) mod micro_kernel;
mod plan;

pub use execute::execute_permute_blocked;
#[cfg(feature = "parallel")]
pub use execute::execute_permute_blocked_par;
pub use plan::{build_permute_plan, PermutePlan};
