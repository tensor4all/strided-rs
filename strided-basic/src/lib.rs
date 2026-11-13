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
//! - [`Exec`] / [`Pool`]: the placement token every fan-out entry takes as its
//!   last parameter (re-exported from `threadteam`)
//!
//! # Primary API (view-based, Julia-compatible)
//!
//! ## Map Operations
//!
//! - [`map_into`]: Apply a function element-wise from source to destination
//! - [`zip_map2_into`], [`zip_map3_into`], [`zip_map4_into`]: Multi-array element-wise operations
//!
//! ## In-place Update Operations
//!
//! - [`map_update_into`], [`zip_update2_into`], [`zip_update3_into`]: `dest[i] = f(dest[i], inputs...)`,
//!   reading the old destination without a second view of it
//!
//! ## Reduce Operations
//!
//! - [`reduce`]: Full reduction with map function
//! - [`zip_reduce`]: Two-input full reduction with a map function (the
//!   widening route, e.g. `f32` operands accumulated in `f64`)
//! - [`reduce_axis`]: Reduce along a single axis
//!
//! A reduction that can fan out (`reduce`, `zip_reduce`, `sum`) gives a result
//! that is independent of the token's budget only for a serial token; `dot`
//! never enters a parallel branch, so its result does not depend on the token at
//! all. See each entry's evaluation-order section.
//!
//! # Threading
//!
//! Every operation that can fan out takes `exec: Exec<'_>` by value as its last
//! parameter. [`Exec::serial`] runs on the calling thread; a token minted from a
//! [`Pool`] with [`Exec::new`] splits the work within its budget. Whether and how
//! wide is decided by the token: the serial time of the region (elements times the
//! throughput of its `execution::KernelClass`) against the dispatch cost of the
//! sections it opens (`Exec::pays`, `Exec::width_for`), so a region too small to
//! repay a fork stays on the calling thread whatever the budget. Without the
//! `parallel` feature every token runs inline.
//!
//! ## Suffix Vocabulary
//!
//! Accumulation is `*_acc`, overwrite is `*_into`, and `*_view` returns metadata
//! only; the table and the names still outside it are in `REPOSITORY_RULES.md`
//! ("Suffix Vocabulary"). `*_into` never reads the previous `dest[i]`, except
//! the update family ([`map_update_into`], [`zip_update2_into`],
//! [`zip_update3_into`]), which updates `dest[i]` in place from its old value.
//!
//! ## Basic Operations
//!
//! - [`copy_into`]: Copy array contents
//! - [`add_acc`], [`mul_acc`]: Accumulating element-wise arithmetic (`dest[i] += src[i]`,
//!   `dest[i] *= src[i]`)
//! - [`axpy_acc`]: y = alpha*x + y (array version)
//! - [`fma_acc`]: `dest[i] += a[i] * b[i]`
//! - [`sum`], [`dot`]: Reductions
//! - [`symmetrize_into`], [`symmetrize_conj_into`]: Matrix symmetrization
//!
//! ## Raw Strided Copies
//!
//! Over borrowed [`RawStridedRef`] / [`RawStridedMut`] layouts:
//!
//! - [`CopyPlan`], [`UpdatePlan`]: compile a layout pair once, replay it on
//!   exactly that layout
//! - [`replay_copy_raw`], [`replay_map_raw`] and their `_uninit` forms: replay
//!   already-fused dims and strides as given, compiling nothing, serially or
//!   across the token's budget
//! - [`replay_map_raw_fused`] and [`replay_map_raw_uninit_fused`]: the same
//!   replay, but fuse the pair inside the call for callers that hold unfused
//!   axes
//! - [`replay_map_raw_serial`]: the serial replay of [`replay_map_raw`] for an
//!   `FnMut` and for elements that are not `Send + Sync`
//! - [`copy_scale_raw`], [`axpy_raw`] and their `conj` forms: serial copies
//!   that fuse the layout on every call
//!
//! # Example
//!
//! ```rust
//! use strided_basic::{StridedView, StridedViewMut, StridedArray, Identity, Exec, map_into};
//!
//! // Create a column-major array (Julia default)
//! let src = StridedArray::<f64>::from_fn_col_major(&[2, 3], |idx| {
//!     (idx[0] * 10 + idx[1]) as f64
//! });
//! let mut dest = StridedArray::<f64>::col_major(&[2, 3]);
//!
//! // Map with view-based API
//! map_into(&mut dest.view_mut(), &src.view(), |x| x * 2.0, Exec::serial()).unwrap();
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
mod complex_div;
mod copy_plan;
mod cost;
mod dense_update;
mod erased;
mod erased_common;
pub mod execution;
mod fuse;
mod inline;
mod kernel;
mod layout_check;
mod map_view;
mod nan_select;
mod ops_view;
mod order;
mod raw_ops;
mod reduce_view;
mod simd;
mod threading;
mod update_view;
pub use complex_div::{robust_complex_divide_f32, robust_complex_divide_f64};
pub use copy_plan::{CopyPlan, UpdatePlan};
pub use dense_update::{
    axpby_accum, embed_diagonal_into_uninit, embed_diagonal_view_into_uninit,
    triangular_mask_into_uninit, triangular_mask_view_into_uninit,
};
pub use erased::{
    ArgReduceOp, ErasedArgReducePlan, ErasedConcatenatePlan, ErasedCopyPlan, ErasedNormPlan,
    ErasedReducePlan, ErasedScanPlan, ErasedUpdatePlan, NormKind, NormSpec, ReduceOp, ScanOp,
    ScanOptions, UpdateOp,
};
pub use inline::{InlineVec, OperandVec, OPERAND_CAP};
pub use map_view::{
    broadcast_mul_into, broadcast_mul_into_uninit, compare_into, compare_into_uninit, map_into,
    map_into_uninit_with_kernel, mul_into, mul_into_uninit, zip_map2_into, zip_map3_into,
    zip_map4_into, Classed, CompareOp, MapFn,
};
pub use nan_select::{nan_propagating_maximum, nan_propagating_minimum};
pub use ops_view::{
    add_acc, axpy_acc, copy_conj_into, copy_into, copy_into_col_major, copy_into_uninit,
    copy_scale_into, copy_transpose_scale_into, dot, fma_acc, mul_acc, sum, symmetrize_conj_into,
    symmetrize_into,
};
pub use raw_ops::{
    axpy_conj_raw, axpy_raw, copy_scale_conj_raw, copy_scale_raw, replay_copy_raw,
    replay_copy_raw_uninit, replay_map_raw, replay_map_raw_fused, replay_map_raw_serial,
    replay_map_raw_uninit, replay_map_raw_uninit_fused, RAW_FUSED_RANK_LIMIT,
};
pub use reduce_view::{reduce, reduce_axis, reduce_axis_into, zip_reduce};
pub use simd::MaybeSimdOps;
pub use strided_view::view;
pub use strided_view::*;
pub use threadteam::{Exec, Pool, PoolBuilder};
pub use update_view::{map_update_into, zip_update2_into, zip_update3_into};
/// Block memory size for cache-optimized iteration (L1 cache target).
///
/// Operations are blocked into tiles that fit within this size to maximize cache hits.
/// Default: 32KB (typical L1 data cache size).
pub const BLOCK_MEMORY_SIZE: usize = 32 * 1024;

/// Cache line size in bytes.
///
/// Used for memory region calculations in block size computation.
pub const CACHE_LINE_SIZE: usize = 64;

/// Associativity of the L1 data cache the tiles target.
///
/// With [`BLOCK_MEMORY_SIZE`] it fixes the way size, `BLOCK_MEMORY_SIZE /
/// L1_WAYS` = 4 KiB: addresses that differ by a multiple of it fall in the same
/// set, and a set holds at most `L1_WAYS` lines. 32 KiB 8-way (Zen 1-4,
/// Skylake) and 48 KiB 12-way (Ice Lake and later, Zen 5) L1d caches share the
/// 4 KiB way, and 8 is the smaller associativity, so the bound stays safe on
/// both. See `block::fit_l1_sets` (#287).
pub(crate) const L1_WAYS: usize = 8;

/// Lines a tile may keep live in one L1 set: [`L1_WAYS`] less a quarter of
/// them (6 of 8).
///
/// The quarter is headroom, not geometry: the replacement policy of these
/// caches is a pseudo-LRU that does not always evict the oldest line, and the
/// L1 stride prefetcher of a strided operand pulls its next lines into the
/// same set, so a set filled to exactly its ways still evicts lines before
/// they are reused. The value was calibrated on the #287 diagnostic; the
/// measurement is in `docs/worklogs/2026-11-10-issue-287-l1-set-tiles.md`.
pub(crate) const L1_LIVE_LINES_PER_SET: usize = L1_WAYS - L1_WAYS / 4;

/// Measurement hooks for benchmarks only; compiled out of every other build.
#[cfg(feature = "bench-hooks")]
pub mod bench_hooks {
    use core::sync::atomic::AtomicBool;

    /// Whether [`crate::block::declines_tiles`] may decline the tiles
    /// ([#428]). Clearing it times the model's tiles against the flat walk in
    /// one process; see `benches/view_decline_428.rs`.
    ///
    /// [#428]: https://github.com/tensor4all/strided-rs/issues/428
    pub static DECLINE_TILES: AtomicBool = AtomicBool::new(true);

    /// The per-set line budget `block::fit_l1_sets` enforces ([#287]); `0`
    /// turns the step off. Storing `0` and the default alternately
    /// times the model's tiles against the fitted ones in one process; see
    /// `benches/block_l1_sets_287.rs`.
    ///
    /// [#287]: https://github.com/tensor4all/strided-rs/issues/287
    pub static L1_SET_BUDGET: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(crate::L1_LIVE_LINES_PER_SET);

    /// Tiles the pointwise planner uses instead of its own when non-empty and
    /// of the fused rank: a diagnostic to time an arbitrary tile through the
    /// production walk ([#287]).
    ///
    /// [#287]: https://github.com/tensor4all/strided-rs/issues/287
    pub static FORCED_TILES: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

    /// The tiles the raw pair replay picks for `dims` as given (no fusion or
    /// reordering; extent-1 axes are dropped, as the model drops them), or
    /// `None` for its column-major walk. The extents must have a checked,
    /// nonzero product.
    pub fn pair_blocks<T>(dims: &[usize], dst: &[isize], src: &[isize]) -> Option<Vec<usize>> {
        crate::raw_ops::with_given_route::<T, T, _>(dims, dst, src, |route| {
            route.blocks().map(<[usize]>::to_vec)
        })
    }

    /// The fused dims and block vector the pointwise planner hands a kernel
    /// whose destination is operand `dest`, for the bench's report.
    pub fn pointwise_blocks(
        dims: &[usize],
        strides_list: &[&[isize]],
        elem_sizes: &[usize],
        dest: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let (dims, _, plan) =
            crate::kernel::build_plan_fused(dims, strides_list, Some(dest), elem_sizes);
        (dims.as_slice().to_vec(), plan.block.as_slice().to_vec())
    }

    /// The view path's block vector for axes that are already in iteration
    /// order, i.e. the decision the bench toggles above.
    pub fn planned_blocks(
        dims: &[usize],
        strides_list: &[&[isize]],
        elem_sizes: &[usize],
    ) -> Vec<usize> {
        crate::block::compute_block_sizes(dims, strides_list, elem_sizes)
            .as_slice()
            .to_vec()
    }
}

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
