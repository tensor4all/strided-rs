//! Dtype-erased arithmetic and indexing kernels, with shared typed primitives.
//! Runtime-DAG fusion is provided separately by `strided-fused`.
pub use strided_basic::*;
mod erased;
pub use erased::*;

// strided-kernel 0.4.0 exported the runtime-DAG fused API from its crate root.
// These items moved to `strided-fused` and stay re-exported here so those names
// keep resolving for code written against the 0.4 line (issue #261). Only the
// names carry over: the calls are not source compatible, because
// `fused_elementwise_into` now requires a `threadteam::Exec` placement token as
// its fourth argument (`strided-fused/src/fused.rs:1855`) where 0.4 took three,
// so a 0.4 call site fails to compile. Prefer importing these from
// `strided_fused` in new code; the re-exports are scheduled for removal in the
// next breaking release.
pub use strided_fused::{
    fused_elementwise_into, ErasedFusedPlan, FusedInst, FusedOp, FusedPlan, FusedScalar,
};
