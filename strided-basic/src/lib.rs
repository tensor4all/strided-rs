#![doc = include_str!("../README.md")]

//! Cache-optimized kernels for strided multidimensional array operations.
//!
//! This crate is a Rust port of Julia's [Strided.jl](https://github.com/Jutho/Strided.jl)
//! and [StridedViews.jl](https://github.com/Jutho/StridedViews.jl) libraries, providing
//! efficient operations on strided multidimensional array views.
//!
//! # Core Types
//!
//! - [`StridedView`] / [`StridedViewMut`]: Dynamic-rank strided views over existing data
//! - [`StridedArray`]: Owned strided multidimensional array
//! - [`ElementOp`] trait and implementations ([`Identity`], [`Conj`], [`Transpose`], [`Adjoint`]):
//!   Type-level element operations applied lazily on access
//! - [`ExecutionPolicy`] / [`with_execution_policy`]: optional bounds on
//!   strided-owned CPU fanout without creating a Rayon pool
//!
//! # Primary API (view-based, Julia-compatible)
//!
//! ## Map Operations
//!
//! - [`map_into`]: Apply a function element-wise from source to destination
//! - [`zip_map2_into`], [`zip_map3_into`], [`zip_map4_into`]: Multi-array element-wise operations
//!
//! ## Reduce Operations
//!
//! - [`reduce`]: Full reduction with map function
//! - [`reduce_axis`]: Reduce along a single axis
//!
//! ## Basic Operations
//!
//! - [`copy_into`]: Copy array contents
//! - [`add`], [`mul`]: Element-wise arithmetic
//! - [`axpy`]: y = alpha*x + y (array version)
//! - [`sum`], [`dot`]: Reductions
//! - [`symmetrize_into`], [`symmetrize_conj_into`]: Matrix symmetrization
//!
//! # Example
//!
//! ```rust
//! use strided_basic::{StridedView, StridedViewMut, StridedArray, Identity, map_into};
//!
//! // Create a column-major array (Julia default)
//! let src = StridedArray::<f64>::from_fn_col_major(&[2, 3], |idx| {
//!     (idx[0] * 10 + idx[1]) as f64
//! });
//! let mut dest = StridedArray::<f64>::col_major(&[2, 3]);
//!
//! // Map with view-based API
//! map_into(&mut dest.view_mut(), &src.view(), |x| x * 2.0).unwrap();
//! assert_eq!(dest.get(&[1, 2]), 24.0); // (1*10 + 2) * 2
//! ```
//!
//! # Cache Optimization
//!
//! The library uses Julia's blocking strategy for cache efficiency:
//! - Dimensions are sorted by stride magnitude for optimal memory access
//! - Operations are blocked into tiles fitting L1 cache ([`BLOCK_MEMORY_SIZE`] = 32KB)
//! - Contiguous arrays use fast paths bypassing the blocking machinery

mod block;
mod copy_plan;
mod dense_update;
mod erased;
mod erased_common;
mod exec_context;
pub mod execution;
mod execution_policy;
mod fuse;
mod kernel;
mod layout_check;
mod map_view;
mod maybe_sync;
mod ops_view;
mod order;
mod raw_ops;
mod reduce_view;
mod simd;
mod threading;
pub use copy_plan::CopyPlan;
pub use dense_update::{axpby_accum, embed_diagonal_into_uninit, triangular_mask_into_uninit};
pub use erased::{ErasedConcatenatePlan, ErasedCopyPlan, ErasedReducePlan, ReduceOp};
pub use exec_context::ExecContext;
pub use execution_policy::{with_execution_policy, ExecutionPolicy};
pub use map_view::{
    broadcast_mul_into, broadcast_mul_into_uninit, compare_into, compare_into_uninit, map_into,
    mul_into, mul_into_uninit, zip_map2_into, zip_map3_into, zip_map4_into, CompareOp,
};
pub use maybe_sync::{MaybeSend, MaybeSendSync, MaybeSync};
pub use ops_view::{
    add, axpy, copy_conj, copy_into, copy_into_col_major, copy_into_uninit, copy_scale,
    copy_transpose_scale_into, dot, fma, mul, sum, symmetrize_conj_into, symmetrize_into,
};
pub use raw_ops::{
    axpy_conj_raw, axpy_raw, copy_scale_conj_raw, copy_scale_raw, RAW_FUSED_RANK_LIMIT,
};
pub use reduce_view::{reduce, reduce_axis};
pub use simd::MaybeSimdOps;
pub use strided_view::view;
pub use strided_view::*;
/// Block memory size for cache-optimized iteration (L1 cache target).
///
/// Operations are blocked into tiles that fit within this size to maximize cache hits.
/// Default: 32KB (typical L1 data cache size).
pub const BLOCK_MEMORY_SIZE: usize = 32 * 1024;

/// Cache line size in bytes.
///
/// Used for memory region calculations in block size computation.
pub const CACHE_LINE_SIZE: usize = 64;

mod gather_plan;
mod outer_product;
mod static_indexing_plan;
pub use gather_plan::{
    DynamicSlicePlan, DynamicUpdateSlicePlan, GatherIndex, GatherPlan, GatherSpec, ScatterPlan,
    ScatterSpec,
};
pub use outer_product::{
    batched_outer_product_into, batched_outer_product_into_uninit, plan_lazy_outer_product,
    LazyOuterProductLayout,
};
pub use static_indexing_plan::{ConcatenatePlan, PadPlan, ReversePlan, SlicePlan};
