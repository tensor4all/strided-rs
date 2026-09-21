//! Dtype-erased arithmetic and indexing kernels, with shared typed primitives.
//! Runtime-DAG fusion is provided separately by `strided-fused`.
pub use strided_basic::*;
mod erased;
pub use erased::*;
