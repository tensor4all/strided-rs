//! Execution contract for operation-family implementations.
//!
//! This module is the shared kernel-extension interface, not a public mirror
//! of implementation modules. Prevalidated entries require the caller to prove
//! the exact destination geometry and every omitted shape check. A layout marker
//! alone does not prove those obligations. Ordinary callers should use the
//! checked APIs at the crate root. Every family that can fan out takes the same
//! [`Exec`] placement token as its last parameter.
//!
//! # Examples
//!
//! ```
//! use strided_basic::{Exec, StridedArray, execution::*};
//! let src = StridedArray::<f64>::col_major(&[2]);
//! let mut dst = StridedArray::<f64>::col_major(&[2]);
//! let marker = validate_destination_layout_without_alloc(&[2], &[1]).unwrap();
//! // SAFETY: both arrays have shape [2], distinct storage, and the checked layout.
//! unsafe { map_into_validated(&mut dst.view_mut(), &src.view(), |x| x, marker, Exec::serial()) }.unwrap();
//! ```
//!
//! A marker alone does not make the prevalidated entry safe to call, which is
//! why every prevalidated entry is `unsafe`: the compiler rejects a call that
//! does not acknowledge the caller's obligation.
//!
//! ```compile_fail,E0133
//! use strided_basic::{Exec, StridedArray, execution::*};
//! let src = StridedArray::<f64>::col_major(&[2]);
//! let mut dst = StridedArray::<f64>::col_major(&[2]);
//! let marker = validate_destination_layout_without_alloc(&[2], &[1]).unwrap();
//! map_into_validated(&mut dst.view_mut(), &src.view(), |x| x, marker, Exec::serial()).unwrap();
//! ```
pub use crate::cost::{parallel_width, tree_width, KernelClass};
pub use crate::erased_common::{
    check_dtype, check_static_indexing_dtype, erased_view, validate_uninit_no_overlap,
};
pub use crate::kernel::{ensure_same_shape, KernelPlan, SMALL_TENSOR_THRESHOLD};
pub use crate::layout_check::is_injective_layout;
pub use crate::map_view::{
    validate_destination_layout, validate_destination_layout_without_alloc,
    ValidatedDestinationLayout,
};
pub use crate::threading::{parallel_map_reduce, SendPtr};

use crate::{ElementOp, Exec, InlineVec, OperandVec, Result, StridedView, StridedViewMut};

/// Prevalidated build plan fused for kernel-family implementations.
///
/// # Safety
/// All rank-indexed arrays must have matching lengths and a valid destination
/// index (when present). Shape products, stride magnitudes, cost arithmetic,
/// and every intermediate offset used by planning/iteration must be representable.
/// Iteration blocks must be positive and offsets must correspond to the supplied
/// layouts; threaded partitioning additionally requires positive
/// costs. Callback memory accesses must be valid for each generated
/// block, with disjoint mutable regions when partitions execute concurrently.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::*;
/// // SAFETY: the example supplies matching bounded layouts and disjoint storage.
/// let (dims, _, plan) = unsafe { build_plan_fused(&[2, 3], &[&[1, 2]], Some(0), 8) };
/// assert_eq!(dims.iter().product::<usize>(), 6);
/// assert_eq!(plan.block.len(), dims.len());
/// ```
#[inline]
pub unsafe fn build_plan_fused(
    dims: &[usize],
    strides_list: &[&[isize]],
    dest_index: Option<usize>,
    elem_size: usize,
) -> (InlineVec<usize>, OperandVec<InlineVec<isize>>, KernelPlan) {
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::kernel::build_plan_fused(dims, strides_list, dest_index, elem_size)
}

/// Prevalidated build plan fused small for kernel-family implementations.
///
/// # Safety
/// All rank-indexed arrays must have matching lengths and a valid destination
/// index (when present). Shape products, stride magnitudes, cost arithmetic,
/// and every intermediate offset used by planning/iteration must be representable.
/// Iteration blocks must be positive and offsets must correspond to the supplied
/// layouts; threaded partitioning additionally requires positive
/// costs. Callback memory accesses must be valid for each generated
/// block, with disjoint mutable regions when partitions execute concurrently.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::*;
/// // SAFETY: the example supplies matching bounded layouts and disjoint storage.
/// let (dims, _, _) = unsafe { build_plan_fused_small(&[2, 3], &[&[1, 2]]) };
/// assert_eq!(dims.iter().product::<usize>(), 6);
/// ```
#[inline]
pub unsafe fn build_plan_fused_small(
    dims: &[usize],
    strides_list: &[&[isize]],
) -> (InlineVec<usize>, OperandVec<InlineVec<isize>>, KernelPlan) {
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::kernel::build_plan_fused_small(dims, strides_list)
}

/// Prevalidated for each inner block preordered for kernel-family implementations.
///
/// # Safety
/// All rank-indexed arrays must have matching lengths and a valid destination
/// index (when present). Shape products, stride magnitudes, cost arithmetic,
/// and every intermediate offset used by planning/iteration must be representable.
/// Iteration blocks must be positive and offsets must correspond to the supplied
/// layouts; threaded partitioning additionally requires positive
/// costs. Callback memory accesses must be valid for each generated
/// block, with disjoint mutable regions when partitions execute concurrently.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::*;
/// // SAFETY: the example supplies matching bounded layouts and disjoint storage.
/// let mut count = 0;
/// unsafe { for_each_inner_block_preordered(&[4], &[4], &[vec![1]], &[0], |_, n, _| { count += n; Ok(()) }) }.unwrap();
/// assert_eq!(count, 4);
/// ```
///
/// # Errors
///
/// Performs no validation of its own, so the only error it can return is the
/// one `f` returns; the block walk itself cannot fail. Callers must still
/// satisfy the safety contract before invoking this prevalidated entry.
#[inline]
pub unsafe fn for_each_inner_block_preordered<S: AsRef<[isize]>, F>(
    dims: &[usize],
    blocks: &[usize],
    strides: &[S],
    initial_offsets: &[isize],
    f: F,
) -> Result<()>
where
    F: FnMut(&[isize], usize, &[isize]) -> Result<()>,
{
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::kernel::for_each_inner_block_preordered::<S, F>(
        dims,
        blocks,
        strides,
        initial_offsets,
        f,
    )
}

/// Prevalidated map into validated for kernel-family implementations.
///
/// # Safety
/// All input shapes must match the destination. `validated` must have been
/// obtained for this destination's current geometry, proving injectivity;
/// possessing a marker from another layout is not sufficient. View/descriptor
/// bounds and aliasing contracts must continue to hold throughout replay.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::*;
/// // SAFETY: the example supplies matching bounded layouts and disjoint storage.
/// use strided_basic::StridedArray;
/// let src = StridedArray::<f64>::from_parts(vec![2.0; 2], &[2], &[1], 0).unwrap();
/// let mut dst = StridedArray::<f64>::col_major(&[2]);
/// let checked = validate_destination_layout_without_alloc(&[2], &[1]).unwrap();
/// unsafe { map_into_validated(&mut dst.view_mut(), &src.view(), |a| a, checked, strided_basic::Exec::serial()) }.unwrap();
/// assert_eq!(dst.get(&[1]), 2.0);
/// ```
///
/// # Errors
///
/// Returns `StridedError::RankMismatch` or `StridedError::ShapeMismatch`
/// when the source view does not have the destination's ranks and extents,
/// the only checks the entry keeps. The destination layout is not
/// re-validated: `validated` is taken as proof of injectivity, and a marker
/// for this geometry also excludes an unrepresentable element count.
/// Callers must still satisfy the safety contract before invoking this
/// prevalidated entry.
#[inline]
pub unsafe fn map_into_validated<D: Copy + Send + Sync, A: Copy + Send + Sync, Op: ElementOp<A>>(
    dest: &mut StridedViewMut<D>,
    src: &StridedView<A, Op>,
    f: impl crate::MapFn<A, D>,
    validated: ValidatedDestinationLayout,
    exec: Exec<'_>,
) -> Result<()> {
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::map_view::map_into_validated::<D, A, Op>(dest, src, f, validated, exec)
}

/// Prevalidated zip map2 into validated for kernel-family implementations.
///
/// # Safety
/// All input shapes must match the destination. `validated` must have been
/// obtained for this destination's current geometry, proving injectivity;
/// possessing a marker from another layout is not sufficient. View/descriptor
/// bounds and aliasing contracts must continue to hold throughout replay.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::*;
/// // SAFETY: the example supplies matching bounded layouts and disjoint storage.
/// use strided_basic::StridedArray;
/// let src = StridedArray::<f64>::from_parts(vec![2.0; 2], &[2], &[1], 0).unwrap();
/// let mut dst = StridedArray::<f64>::col_major(&[2]);
/// let checked = validate_destination_layout_without_alloc(&[2], &[1]).unwrap();
/// unsafe { zip_map2_into_validated(&mut dst.view_mut(), &src.view(), &src.view(), |a, b| a + b, checked, strided_basic::Exec::serial()) }.unwrap();
/// assert_eq!(dst.get(&[1]), 4.0);
/// ```
///
/// # Errors
///
/// Performs no shape or destination-layout check of its own and returns
/// `Ok(())` for every call that satisfies the safety contract: matching
/// source extents come from that contract, destination injectivity from
/// `validated`, and a marker for this geometry also excludes an
/// unrepresentable element count. Callers must still satisfy the safety
/// contract before invoking this prevalidated entry.
#[inline]
pub unsafe fn zip_map2_into_validated<
    D: Copy + Send + Sync,
    A: Copy + Send + Sync,
    B: Copy + Send + Sync,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
>(
    dest: &mut StridedViewMut<D>,
    a: &StridedView<A, OpA>,
    b: &StridedView<B, OpB>,
    f: impl Fn(A, B) -> D + Sync,
    validated: ValidatedDestinationLayout,
    exec: Exec<'_>,
) -> Result<()> {
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::map_view::zip_map2_into_validated::<D, A, B, OpA, OpB>(dest, a, b, f, validated, exec)
}

/// Prevalidated zip map3 into validated for kernel-family implementations.
///
/// # Safety
/// All input shapes must match the destination. `validated` must have been
/// obtained for this destination's current geometry, proving injectivity;
/// possessing a marker from another layout is not sufficient. View/descriptor
/// bounds and aliasing contracts must continue to hold throughout replay.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::*;
/// // SAFETY: the example supplies matching bounded layouts and disjoint storage.
/// use strided_basic::StridedArray;
/// let src = StridedArray::<f64>::from_parts(vec![2.0; 2], &[2], &[1], 0).unwrap();
/// let mut dst = StridedArray::<f64>::col_major(&[2]);
/// let checked = validate_destination_layout_without_alloc(&[2], &[1]).unwrap();
/// unsafe { zip_map3_into_validated(&mut dst.view_mut(), &src.view(), &src.view(), &src.view(), |a, b, c| a + b + c, checked, strided_basic::Exec::serial()) }.unwrap();
/// assert_eq!(dst.get(&[1]), 6.0);
/// ```
///
/// # Errors
///
/// Returns `StridedError::RankMismatch` or `StridedError::ShapeMismatch`
/// when any source view does not have the destination's ranks and extents,
/// the only checks the entry keeps. The destination layout is not
/// re-validated: `validated` is taken as proof of injectivity, and a marker
/// for this geometry also excludes an unrepresentable element count.
/// Callers must still satisfy the safety contract before invoking this
/// prevalidated entry.
#[inline]
pub unsafe fn zip_map3_into_validated<
    D: Copy + Send + Sync,
    A: Copy + Send + Sync,
    B: Copy + Send + Sync,
    C: Copy + Send + Sync,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
    OpC: ElementOp<C>,
>(
    dest: &mut StridedViewMut<D>,
    a: &StridedView<A, OpA>,
    b: &StridedView<B, OpB>,
    c: &StridedView<C, OpC>,
    f: impl Fn(A, B, C) -> D + Sync,
    validated: ValidatedDestinationLayout,
    exec: Exec<'_>,
) -> Result<()> {
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::map_view::zip_map3_into_validated::<D, A, B, C, OpA, OpB, OpC>(
        dest, a, b, c, f, validated, exec,
    )
}

/// Prevalidated zip map4 into validated for kernel-family implementations.
///
/// # Safety
/// All input shapes must match the destination. `validated` must have been
/// obtained for this destination's current geometry, proving injectivity;
/// possessing a marker from another layout is not sufficient. View/descriptor
/// bounds and aliasing contracts must continue to hold throughout replay.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::*;
/// // SAFETY: the example supplies matching bounded layouts and disjoint storage.
/// use strided_basic::StridedArray;
/// let src = StridedArray::<f64>::from_parts(vec![2.0; 2], &[2], &[1], 0).unwrap();
/// let mut dst = StridedArray::<f64>::col_major(&[2]);
/// let checked = validate_destination_layout_without_alloc(&[2], &[1]).unwrap();
/// unsafe { zip_map4_into_validated(&mut dst.view_mut(), &src.view(), &src.view(), &src.view(), &src.view(), |a, b, c, d| a + b + c + d, checked, strided_basic::Exec::serial()) }.unwrap();
/// assert_eq!(dst.get(&[1]), 8.0);
/// ```
///
/// # Errors
///
/// Returns `StridedError::RankMismatch` or `StridedError::ShapeMismatch`
/// when any source view does not have the destination's ranks and extents,
/// the only checks the entry keeps. The destination layout is not
/// re-validated: `validated` is taken as proof of injectivity, and a marker
/// for this geometry also excludes an unrepresentable element count.
/// Callers must still satisfy the safety contract before invoking this
/// prevalidated entry.
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn zip_map4_into_validated<
    D: Copy + Send + Sync,
    A: Copy + Send + Sync,
    B: Copy + Send + Sync,
    C: Copy + Send + Sync,
    E: Copy + Send + Sync,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
    OpC: ElementOp<C>,
    OpE: ElementOp<E>,
>(
    dest: &mut StridedViewMut<D>,
    a: &StridedView<A, OpA>,
    b: &StridedView<B, OpB>,
    c: &StridedView<C, OpC>,
    e: &StridedView<E, OpE>,
    f: impl Fn(A, B, C, E) -> D + Sync,
    validated: ValidatedDestinationLayout,
    exec: Exec<'_>,
) -> Result<()> {
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::map_view::zip_map4_into_validated::<D, A, B, C, E, OpA, OpB, OpC, OpE>(
        dest, a, b, c, e, f, validated, exec,
    )
}

/// Prevalidated map raw into validated for kernel-family implementations.
///
/// # Safety
/// All input shapes must match the destination. `validated` must have been
/// obtained for this destination's current geometry, proving injectivity;
/// possessing a marker from another layout is not sufficient. View/descriptor
/// bounds and aliasing contracts must continue to hold throughout replay.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::*;
/// // SAFETY: the example supplies matching bounded layouts and disjoint storage.
/// use strided_basic::{RawStridedRef, RawStridedMut, Identity};
/// let values = [2.0_f64; 2];
/// let src = RawStridedRef::new(&values, &[2], &[1], 0).unwrap();
/// let mut output = [0.0; 2];
/// let mut dst = RawStridedMut::new(&mut output, &[2], &[1], 0).unwrap();
/// let checked = validate_destination_layout_without_alloc(&[2], &[1]).unwrap();
/// unsafe { map_raw_into_validated::<f64, f64, Identity>(&mut dst, &src, |a| a, checked, strided_basic::Exec::serial()) }.unwrap();
/// assert_eq!(output, [2.0; 2]);
/// ```
///
/// # Errors
///
/// Returns `StridedError::RankMismatch` or `StridedError::ShapeMismatch`
/// when the source does not have the destination's ranks and extents, the
/// only checks the entry keeps. The destination layout is not re-validated:
/// `validated` is taken as proof of injectivity, and a marker for this
/// geometry also excludes an unrepresentable element count. Callers must
/// still satisfy the safety contract before invoking this prevalidated
/// entry.
#[inline]
pub unsafe fn map_raw_into_validated<
    D: Copy + Send + Sync,
    A: Copy + Send + Sync,
    Op: ElementOp<A>,
>(
    dest: &mut crate::RawStridedMut<'_, D>,
    src: &crate::RawStridedRef<'_, A>,
    f: impl Fn(A) -> D + Sync,
    validated: ValidatedDestinationLayout,
    exec: Exec<'_>,
) -> Result<()> {
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::map_view::map_raw_into_validated::<D, A, Op>(dest, src, f, validated, exec)
}

/// Prevalidated zip map2 raw into validated for kernel-family implementations.
///
/// # Safety
/// All input shapes must match the destination. `validated` must have been
/// obtained for this destination's current geometry, proving injectivity;
/// possessing a marker from another layout is not sufficient. View/descriptor
/// bounds and aliasing contracts must continue to hold throughout replay.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::*;
/// // SAFETY: the example supplies matching bounded layouts and disjoint storage.
/// use strided_basic::{RawStridedRef, RawStridedMut, Identity};
/// let values = [2.0_f64; 2];
/// let src = RawStridedRef::new(&values, &[2], &[1], 0).unwrap();
/// let mut output = [0.0; 2];
/// let mut dst = RawStridedMut::new(&mut output, &[2], &[1], 0).unwrap();
/// let checked = validate_destination_layout_without_alloc(&[2], &[1]).unwrap();
/// unsafe { zip_map2_raw_into_validated::<f64, f64, f64, Identity, Identity>(&mut dst, &src, &src, |a, b| a + b, checked, strided_basic::Exec::serial()) }.unwrap();
/// assert_eq!(output, [4.0; 2]);
/// ```
///
/// # Errors
///
/// Returns `StridedError::RankMismatch` or `StridedError::ShapeMismatch`
/// when either source does not have the destination's ranks and extents,
/// the only checks the entry keeps. The destination layout is not
/// re-validated: `validated` is taken as proof of injectivity, and a marker
/// for this geometry also excludes an unrepresentable element count.
/// Callers must still satisfy the safety contract before invoking this
/// prevalidated entry.
#[inline]
pub unsafe fn zip_map2_raw_into_validated<
    D: Copy + Send + Sync,
    A: Copy + Send + Sync,
    B: Copy + Send + Sync,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
>(
    dest: &mut crate::RawStridedMut<'_, D>,
    a: &crate::RawStridedRef<'_, A>,
    b: &crate::RawStridedRef<'_, B>,
    f: impl Fn(A, B) -> D + Sync,
    validated: ValidatedDestinationLayout,
    exec: Exec<'_>,
) -> Result<()> {
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::map_view::zip_map2_raw_into_validated::<D, A, B, OpA, OpB>(
        dest, a, b, f, validated, exec,
    )
}

/// Prevalidated compute costs for kernel-family implementations.
///
/// # Safety
/// All rank-indexed arrays must have matching lengths and a valid destination
/// index (when present). Shape products, stride magnitudes, cost arithmetic,
/// and every intermediate offset used by planning/iteration must be representable.
/// Iteration blocks must be positive and offsets must correspond to the supplied
/// layouts; threaded partitioning additionally requires positive
/// costs. Callback memory accesses must be valid for each generated
/// block, with disjoint mutable regions when partitions execute concurrently.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::*;
/// // SAFETY: the example supplies matching bounded layouts and disjoint storage.
/// let costs = unsafe { compute_costs(&[vec![1, 4], vec![1, 8]]) };
/// assert_eq!(costs.as_slice(), [2, 8]);
/// ```
#[inline]
pub unsafe fn compute_costs<S: AsRef<[isize]>>(all_strides: &[S]) -> InlineVec<isize> {
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::fuse::compute_costs::<S>(all_strides)
}

/// Prevalidated mapreduce threaded for kernel-family implementations.
///
/// `class` is the serial cost class of the leaf (see [`KernelClass`]); every node
/// of the split runs in parallel only when a fork of its region pays for that
/// class.
///
/// # Safety
/// All rank-indexed arrays must have matching lengths and a valid destination
/// index (when present). Shape products, stride magnitudes, cost arithmetic,
/// and every intermediate offset used by planning/iteration must be representable.
/// Iteration blocks must be positive and offsets must correspond to the supplied
/// layouts; threaded partitioning additionally requires positive
/// costs. Callback memory accesses must be valid for each generated
/// block, with disjoint mutable regions when partitions execute concurrently.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::*;
/// // SAFETY: the example supplies matching bounded layouts and disjoint storage.
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// let count = AtomicUsize::new(0);
/// unsafe { mapreduce_threaded(&[4], &[4], &[vec![1]], &[0], &[2], KernelClass::Map, strided_basic::Exec::serial(), &|dims, _, _, _| { count.fetch_add(dims.iter().product::<usize>(), Ordering::Relaxed); Ok(()) }) }.unwrap();
/// assert_eq!(count.load(Ordering::Relaxed), 4);
/// ```
///
/// # Errors
///
/// Returns `StridedError::OffsetOverflow` when the element count of `dims`
/// cannot be represented, and otherwise only the error `f` returns; the
/// splitter itself cannot fail. Callers must still satisfy the safety
/// contract before invoking this prevalidated entry.
#[inline]
pub unsafe fn mapreduce_threaded<S: AsRef<[isize]>, F>(
    dims: &[usize],
    blocks: &[usize],
    strides_list: &[S],
    offsets: &[isize],
    costs: &[isize],
    class: KernelClass,
    exec: Exec<'_>,
    f: &F,
) -> Result<()>
where
    F: Fn(&[usize], &[usize], &[&[isize]], &[isize]) -> Result<()> + Sync,
{
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::threading::mapreduce_threaded::<S, F>(
        dims,
        blocks,
        strides_list,
        offsets,
        costs,
        class,
        exec,
        f,
    )
}

/// Full-overwrite indexed replay for an operation-family adapter.
///
/// # Safety
/// The adapter must reject input/output overlap before creating the input
/// references. All descriptor allocation and initialization contracts must
/// remain valid for the call. Output slots may be uninitialized; the owning
/// plan retains its checked layout/index validation and private initialization
/// receipt. No initialized reference to the output backing may be formed.
/// Gather into uninitialized output without exposing an initialization receipt.
///
/// # Examples
///
/// ```
/// use strided_basic::{RawStridedRef, RawStridedMut};
/// use strided_basic::execution::*;
/// use core::mem::MaybeUninit;
/// use strided_basic::{GatherPlan, GatherSpec};
/// let spec = GatherSpec { offset_dims: vec![], collapsed_slice_dims: vec![0], start_index_map: vec![0], index_vector_dim: 1, slice_sizes: vec![1] };
/// let plan = GatherPlan::compile(&[3], &[1], &[2, 1], &[1, 2], &[2], &[1], spec).unwrap();
/// let src = RawStridedRef::new(&[1_i32, 2, 3], &[3], &[1], 0).unwrap();
/// let indices = RawStridedRef::new(&[0_i32, 2], &[2, 1], &[1, 2], 0).unwrap();
/// let mut values = [MaybeUninit::<i32>::uninit(); 2];
/// let mut dst = RawStridedMut::new(&mut values, &[2], &[1], 0).unwrap();
/// // SAFETY: input and output allocations are disjoint and layouts match the plan.
/// unsafe { gather_into_uninit(&plan, &mut dst, &src, &indices, strided_basic::Exec::serial()) }.unwrap();
/// assert_eq!(unsafe { values[1].assume_init() }, 3);
/// ```
///
/// # Errors
///
/// Returns `StridedError::PlanLayoutMismatch` when the destination, operand,
/// or start-index view does not carry the plan's compiled extents and
/// strides, before any destination write, and
/// `StridedError::OffsetOverflow` when a replay offset cannot be
/// represented. Out-of-range start indices are clamped, not rejected.
#[inline]
pub unsafe fn gather_into_uninit<T, I>(
    plan: &crate::GatherPlan,
    dest: &mut crate::RawStridedMut<'_, core::mem::MaybeUninit<T>>,
    operand: &crate::RawStridedRef<'_, T>,
    start_indices: &crate::RawStridedRef<'_, I>,
    exec: Exec<'_>,
) -> Result<()>
where
    T: Copy + Send + Sync,
    I: crate::GatherIndex,
{
    plan.execute_uninit(dest, operand, start_indices, exec)
}

/// Full-overwrite indexed replay for an operation-family adapter.
///
/// # Safety
/// The adapter must reject input/output overlap before creating the input
/// references. All descriptor allocation and initialization contracts must
/// remain valid for the call. Output slots may be uninitialized; the owning
/// plan retains its checked layout/index validation and private initialization
/// receipt. No initialized reference to the output backing may be formed.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::*;
/// // SAFETY: the example supplies matching bounded layouts and disjoint storage.
/// use strided_basic::{DynamicSlicePlan, RawStridedRef, RawStridedMut};
/// use core::mem::MaybeUninit;
/// let plan = DynamicSlicePlan::compile(&[3], &[1], &[1], &[1], &[2], &[1], &[2]).unwrap();
/// let src = RawStridedRef::new(&[1_i32, 2, 3], &[3], &[1], 0).unwrap();
/// let starts = RawStridedRef::new(&[1_i32], &[1], &[1], 0).unwrap();
/// let mut values = [MaybeUninit::<i32>::uninit(); 2];
/// let mut dst = RawStridedMut::new(&mut values, &[2], &[1], 0).unwrap();
/// unsafe { dynamic_slice_into_uninit(&plan, &mut dst, &src, &starts, strided_basic::Exec::serial()) }.unwrap();
/// assert_eq!(unsafe { values[1].assume_init() }, 3);
/// ```
///
/// # Errors
///
/// Returns `StridedError::PlanLayoutMismatch` when the destination, operand,
/// or start view does not carry the plan's compiled extents and strides,
/// before any destination write. That is the only rejection: an empty
/// operand returns before any offset arithmetic, and a non-empty one carries
/// the bounds its constructor validated, so the replay arithmetic cannot
/// overflow. Out-of-range start values are clamped, not rejected.
#[inline]
pub unsafe fn dynamic_slice_into_uninit<T, I>(
    plan: &crate::DynamicSlicePlan,
    dest: &mut crate::RawStridedMut<'_, core::mem::MaybeUninit<T>>,
    operand: &crate::RawStridedRef<'_, T>,
    starts: &crate::RawStridedRef<'_, I>,
    exec: Exec<'_>,
) -> Result<()>
where
    T: Copy + Send + Sync,
    I: crate::GatherIndex,
{
    plan.execute_uninit(dest, operand, starts, exec)
}

/// Full-overwrite indexed replay for an operation-family adapter.
///
/// # Safety
/// The adapter must reject input/output overlap before creating the input
/// references. All descriptor allocation and initialization contracts must
/// remain valid for the call. Output slots may be uninitialized; the owning
/// plan retains its checked layout/index validation and private initialization
/// receipt. No initialized reference to the output backing may be formed.
/// Copy the operand and overwrite its selected window.
///
/// # Examples
///
/// ```
/// use strided_basic::{RawStridedRef, RawStridedMut};
/// use strided_basic::execution::*;
/// use core::mem::MaybeUninit;
/// use strided_basic::DynamicUpdateSlicePlan;
/// let plan = DynamicUpdateSlicePlan::compile(&[3], &[1], &[1], &[1], &[1], &[1], &[3], &[1]).unwrap();
/// let src = RawStridedRef::new(&[1_i32, 2, 3], &[3], &[1], 0).unwrap();
/// let update = RawStridedRef::new(&[9_i32], &[1], &[1], 0).unwrap();
/// let starts = RawStridedRef::new(&[1_i32], &[1], &[1], 0).unwrap();
/// let mut values = [MaybeUninit::<i32>::uninit(); 3];
/// let mut dst = RawStridedMut::new(&mut values, &[3], &[1], 0).unwrap();
/// // SAFETY: all inputs are initialized and disjoint from the matching output.
/// unsafe { dynamic_update_into_uninit(&plan, &mut dst, &src, &update, &starts, strided_basic::Exec::serial()) }.unwrap();
/// assert_eq!(unsafe { values[1].assume_init() }, 9);
/// ```
///
/// # Errors
///
/// Returns `StridedError::PlanLayoutMismatch` when the destination, operand,
/// update, or start view does not carry the plan's compiled extents and
/// strides, before any destination write. The clamped update window is
/// preflighted next, still before any destination write, and rejects a window
/// that does not fit `isize`: an empty destination returns before that offset
/// arithmetic, and a non-empty one carries the bounds its constructor
/// validated, so a checked view never reaches that rejection. Out-of-range
/// start values are clamped, not rejected.
#[inline]
pub unsafe fn dynamic_update_into_uninit<'a, T, I>(
    plan: &crate::DynamicUpdateSlicePlan,
    dest: &'a mut crate::RawStridedMut<'a, core::mem::MaybeUninit<T>>,
    operand: &crate::RawStridedRef<'_, T>,
    update: &crate::RawStridedRef<'_, T>,
    starts: &crate::RawStridedRef<'_, I>,
    exec: Exec<'_>,
) -> Result<()>
where
    T: Copy + Send + Sync,
    I: crate::GatherIndex,
{
    plan.execute_uninit(dest, operand, update, starts, exec)
}

/// Full-overwrite indexed replay for an operation-family adapter.
///
/// # Safety
/// The adapter must reject input/output overlap before creating the input
/// references. All descriptor allocation and initialization contracts must
/// remain valid for the call. Output slots may be uninitialized; the owning
/// plan retains its checked layout/index validation and private initialization
/// receipt. No initialized reference to the output backing may be formed.
/// Copy the operand before additive scatter into its initialized logical slots.
///
/// # Examples
///
/// ```
/// use strided_basic::{RawStridedRef, RawStridedMut};
/// use strided_basic::execution::*;
/// use core::mem::MaybeUninit;
/// use strided_basic::{ScatterPlan, ScatterSpec};
/// let spec = ScatterSpec { update_window_dims: vec![], inserted_window_dims: vec![0], scatter_dims_to_operand_dims: vec![0], index_vector_dim: 1 };
/// let plan = ScatterPlan::compile(&[3], &[1], &[1, 1], &[1, 1], &[1], &[1], &[3], &[1], spec).unwrap();
/// let src = RawStridedRef::new(&[1_i32, 2, 3], &[3], &[1], 0).unwrap();
/// let update = RawStridedRef::new(&[9_i32], &[1], &[1], 0).unwrap();
/// let indices = RawStridedRef::new(&[1_i64], &[1, 1], &[1, 1], 0).unwrap();
/// let mut values = [MaybeUninit::<i32>::uninit(); 3];
/// let mut dst = RawStridedMut::new(&mut values, &[3], &[1], 0).unwrap();
/// // SAFETY: all inputs are initialized and disjoint from the matching output.
/// unsafe { scatter_into_uninit(&plan, &mut dst, &src, &indices, &update, i32::wrapping_add, strided_basic::Exec::serial()) }.unwrap();
/// assert_eq!(unsafe { values[1].assume_init() }, 11);
/// ```
///
/// # Errors
///
/// Returns `StridedError::PlanLayoutMismatch` when the destination, operand,
/// scatter-index, or update view does not carry the plan's compiled extents
/// and strides, before any destination write. That is the only rejection: an
/// empty batch or window skips the replay entirely, so no offset arithmetic
/// runs, and a non-empty one carries the bounds its constructor validated.
/// Out-of-range scatter indices are clamped, not rejected.
#[inline]
pub unsafe fn scatter_into_uninit<'a, T, I>(
    plan: &crate::ScatterPlan,
    dest: &'a mut crate::RawStridedMut<'a, core::mem::MaybeUninit<T>>,
    operand: &crate::RawStridedRef<'_, T>,
    scatter_indices: &crate::RawStridedRef<'_, I>,
    updates: &crate::RawStridedRef<'_, T>,
    combine: fn(T, T) -> T,
    exec: Exec<'_>,
) -> Result<()>
where
    T: Copy + core::ops::Add<Output = T> + Send + Sync,
    I: crate::GatherIndex,
{
    plan.execute_uninit(dest, operand, scatter_indices, updates, combine, exec)
}
