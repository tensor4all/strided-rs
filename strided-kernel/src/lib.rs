//! Dtype-erased arithmetic and indexing kernels, with shared typed primitives.
//! Runtime-DAG fusion is provided separately by `strided-fused`.
pub use strided_basic::*;
mod erased;
pub use erased::*;

// strided-kernel 0.4.0 exported the runtime-DAG fused API from its crate root.
// These items moved to `strided-fused`; they stay re-exported here so the 0.4
// line remains semver compatible (issue #261). Prefer importing them from
// `strided_fused` in new code; the re-exports are scheduled for removal in the
// next breaking release.
pub use strided_fused::{
    fused_elementwise_into, ErasedFusedPlan, FusedInst, FusedOp, FusedPlan, FusedScalar,
};
