#![doc = include_str!("../README.md")]

//! Runtime-DAG fused elementwise execution using the shared CPU execution policy.
use strided_basic::*;
pub use strided_basic::{MaybeSendSync, Result, StridedError, StridedView, StridedViewMut};
mod erased;
mod fused;
pub use erased::ErasedFusedPlan;
pub use fused::{fused_elementwise_into, FusedInst, FusedOp, FusedPlan, FusedScalar};
