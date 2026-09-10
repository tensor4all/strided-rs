//! Execution contract for operation-family implementations.
//!
//! This module is the shared kernel-extension interface, not a public mirror
//! of implementation modules. Prevalidated entries require the caller to prove
//! the exact destination geometry and every omitted shape check. A layout marker
//! alone does not prove those obligations. Ordinary callers should use the
//! checked APIs at the crate root. All families share the same execution policy.
//!
//! # Examples
//!
//! ```
//! use strided_basic::{StridedArray, execution::*};
//! let src = StridedArray::<f64>::col_major(&[2]);
//! let mut dst = StridedArray::<f64>::col_major(&[2]);
//! let marker = validate_destination_layout_without_alloc(&[2], &[1]).unwrap();
//! // SAFETY: both arrays have shape [2], distinct storage, and the checked layout.
//! unsafe { map_into_validated(&mut dst.view_mut(), &src.view(), |x| x, marker) }.unwrap();
//! ```
//!
//! A marker alone does not make the prevalidated entry safe to call:
//!
//! ```compile_fail,E0133
//! use strided_basic::{StridedArray, execution::*};
//! let src = StridedArray::<f64>::col_major(&[2]);
//! let mut dst = StridedArray::<f64>::col_major(&[2]);
//! let marker = validate_destination_layout_without_alloc(&[2], &[1]).unwrap();
//! map_into_validated(&mut dst.view_mut(), &src.view(), |x| x, marker).unwrap();
//! ```
pub use crate::erased_common::{
    check_dtype, check_static_indexing_dtype, erased_view, validate_uninit_no_overlap,
};
pub use crate::kernel::{ensure_same_shape, KernelPlan, SMALL_TENSOR_THRESHOLD};
pub use crate::layout_check::is_injective_layout;
pub use crate::map_view::{validate_destination_layout_without_alloc, ValidatedDestinationLayout};
#[cfg(feature = "parallel")]
pub use crate::threading::{
    parallel_map_reduce, parallel_threads_for_len, SendPtr, MINTHREADLENGTH,
};

use crate::{ElementOp, MaybeSendSync, MaybeSync, Result, StridedView, StridedViewMut};

/// Prevalidated build plan fused for kernel-family implementations.
///
/// # Safety
/// All rank-indexed arrays must have matching lengths and a valid destination
/// index (when present). Shape products, stride magnitudes, cost arithmetic,
/// and every intermediate offset used by planning/iteration must be representable.
/// Iteration blocks must be positive and offsets must correspond to the supplied
/// layouts; threaded partitioning additionally requires positive thread counts,
/// costs and nonnegative spacing. Callback memory accesses must be valid for each generated
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
) -> (Vec<usize>, Vec<Vec<isize>>, KernelPlan) {
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
/// layouts; threaded partitioning additionally requires positive thread counts,
/// costs and nonnegative spacing. Callback memory accesses must be valid for each generated
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
) -> (Vec<usize>, Vec<Vec<isize>>, KernelPlan) {
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
/// layouts; threaded partitioning additionally requires positive thread counts,
/// costs and nonnegative spacing. Callback memory accesses must be valid for each generated
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
/// Forwards errors from the owning kernel or callback; callers must still
/// satisfy the safety contract before invoking this prevalidated entry.
#[inline]
pub unsafe fn for_each_inner_block_preordered<F>(
    dims: &[usize],
    blocks: &[usize],
    strides: &[Vec<isize>],
    initial_offsets: &[isize],
    f: F,
) -> Result<()>
where
    F: FnMut(&[isize], usize, &[isize]) -> Result<()>,
{
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::kernel::for_each_inner_block_preordered::<F>(dims, blocks, strides, initial_offsets, f)
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
/// unsafe { map_into_validated(&mut dst.view_mut(), &src.view(), |a| a, checked) }.unwrap();
/// assert_eq!(dst.get(&[1]), 2.0);
/// ```
///
/// # Errors
/// Forwards errors from the owning kernel or callback; callers must still
/// satisfy the safety contract before invoking this prevalidated entry.
#[inline]
pub unsafe fn map_into_validated<
    D: Copy + MaybeSendSync,
    A: Copy + MaybeSendSync,
    Op: ElementOp<A>,
>(
    dest: &mut StridedViewMut<D>,
    src: &StridedView<A, Op>,
    f: impl Fn(A) -> D + MaybeSync,
    validated: ValidatedDestinationLayout,
) -> Result<()> {
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::map_view::map_into_validated::<D, A, Op>(dest, src, f, validated)
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
/// unsafe { zip_map2_into_validated(&mut dst.view_mut(), &src.view(), &src.view(), |a, b| a + b, checked) }.unwrap();
/// assert_eq!(dst.get(&[1]), 4.0);
/// ```
///
/// # Errors
/// Forwards errors from the owning kernel or callback; callers must still
/// satisfy the safety contract before invoking this prevalidated entry.
#[inline]
pub unsafe fn zip_map2_into_validated<
    D: Copy + MaybeSendSync,
    A: Copy + MaybeSendSync,
    B: Copy + MaybeSendSync,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
>(
    dest: &mut StridedViewMut<D>,
    a: &StridedView<A, OpA>,
    b: &StridedView<B, OpB>,
    f: impl Fn(A, B) -> D + MaybeSync,
    validated: ValidatedDestinationLayout,
) -> Result<()> {
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::map_view::zip_map2_into_validated::<D, A, B, OpA, OpB>(dest, a, b, f, validated)
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
/// unsafe { zip_map3_into_validated(&mut dst.view_mut(), &src.view(), &src.view(), &src.view(), |a, b, c| a + b + c, checked) }.unwrap();
/// assert_eq!(dst.get(&[1]), 6.0);
/// ```
///
/// # Errors
/// Forwards errors from the owning kernel or callback; callers must still
/// satisfy the safety contract before invoking this prevalidated entry.
#[inline]
pub unsafe fn zip_map3_into_validated<
    D: Copy + MaybeSendSync,
    A: Copy + MaybeSendSync,
    B: Copy + MaybeSendSync,
    C: Copy + MaybeSendSync,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
    OpC: ElementOp<C>,
>(
    dest: &mut StridedViewMut<D>,
    a: &StridedView<A, OpA>,
    b: &StridedView<B, OpB>,
    c: &StridedView<C, OpC>,
    f: impl Fn(A, B, C) -> D + MaybeSync,
    validated: ValidatedDestinationLayout,
) -> Result<()> {
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::map_view::zip_map3_into_validated::<D, A, B, C, OpA, OpB, OpC>(
        dest, a, b, c, f, validated,
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
/// unsafe { zip_map4_into_validated(&mut dst.view_mut(), &src.view(), &src.view(), &src.view(), &src.view(), |a, b, c, d| a + b + c + d, checked) }.unwrap();
/// assert_eq!(dst.get(&[1]), 8.0);
/// ```
///
/// # Errors
/// Forwards errors from the owning kernel or callback; callers must still
/// satisfy the safety contract before invoking this prevalidated entry.
#[inline]
pub unsafe fn zip_map4_into_validated<
    D: Copy + MaybeSendSync,
    A: Copy + MaybeSendSync,
    B: Copy + MaybeSendSync,
    C: Copy + MaybeSendSync,
    E: Copy + MaybeSendSync,
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
    f: impl Fn(A, B, C, E) -> D + MaybeSync,
    validated: ValidatedDestinationLayout,
) -> Result<()> {
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::map_view::zip_map4_into_validated::<D, A, B, C, E, OpA, OpB, OpC, OpE>(
        dest, a, b, c, e, f, validated,
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
/// unsafe { map_raw_into_validated::<f64, f64, Identity>(&mut dst, &src, |a| a, checked) }.unwrap();
/// assert_eq!(output, [2.0; 2]);
/// ```
///
/// # Errors
/// Forwards errors from the owning kernel or callback; callers must still
/// satisfy the safety contract before invoking this prevalidated entry.
#[inline]
pub unsafe fn map_raw_into_validated<
    D: Copy + MaybeSendSync,
    A: Copy + MaybeSendSync,
    Op: ElementOp<A>,
>(
    dest: &mut crate::RawStridedMut<'_, D>,
    src: &crate::RawStridedRef<'_, A>,
    f: impl Fn(A) -> D + MaybeSync,
    validated: ValidatedDestinationLayout,
) -> Result<()> {
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::map_view::map_raw_into_validated::<D, A, Op>(dest, src, f, validated)
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
/// unsafe { zip_map2_raw_into_validated::<f64, f64, f64, Identity, Identity>(&mut dst, &src, &src, |a, b| a + b, checked) }.unwrap();
/// assert_eq!(output, [4.0; 2]);
/// ```
///
/// # Errors
/// Forwards errors from the owning kernel or callback; callers must still
/// satisfy the safety contract before invoking this prevalidated entry.
#[inline]
pub unsafe fn zip_map2_raw_into_validated<
    D: Copy + MaybeSendSync,
    A: Copy + MaybeSendSync,
    B: Copy + MaybeSendSync,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
>(
    dest: &mut crate::RawStridedMut<'_, D>,
    a: &crate::RawStridedRef<'_, A>,
    b: &crate::RawStridedRef<'_, B>,
    f: impl Fn(A, B) -> D + MaybeSync,
    validated: ValidatedDestinationLayout,
) -> Result<()> {
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::map_view::zip_map2_raw_into_validated::<D, A, B, OpA, OpB>(dest, a, b, f, validated)
}

#[cfg(feature = "parallel")]
/// Prevalidated compute costs for kernel-family implementations.
///
/// # Safety
/// All rank-indexed arrays must have matching lengths and a valid destination
/// index (when present). Shape products, stride magnitudes, cost arithmetic,
/// and every intermediate offset used by planning/iteration must be representable.
/// Iteration blocks must be positive and offsets must correspond to the supplied
/// layouts; threaded partitioning additionally requires positive thread counts,
/// costs and nonnegative spacing. Callback memory accesses must be valid for each generated
/// block, with disjoint mutable regions when partitions execute concurrently.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::*;
/// // SAFETY: the example supplies matching bounded layouts and disjoint storage.
/// let costs = unsafe { compute_costs(&[vec![1, 4], vec![1, 8]]) };
/// assert_eq!(costs, [2, 8]);
/// ```
#[inline]
pub unsafe fn compute_costs<S: AsRef<[isize]>>(all_strides: &[S]) -> Vec<isize> {
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::fuse::compute_costs::<S>(all_strides)
}

#[cfg(feature = "parallel")]
/// Prevalidated mapreduce threaded for kernel-family implementations.
///
/// # Safety
/// All rank-indexed arrays must have matching lengths and a valid destination
/// index (when present). Shape products, stride magnitudes, cost arithmetic,
/// and every intermediate offset used by planning/iteration must be representable.
/// Iteration blocks must be positive and offsets must correspond to the supplied
/// layouts; threaded partitioning additionally requires positive thread counts,
/// costs and nonnegative spacing. Callback memory accesses must be valid for each generated
/// block, with disjoint mutable regions when partitions execute concurrently.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::*;
/// // SAFETY: the example supplies matching bounded layouts and disjoint storage.
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// let count = AtomicUsize::new(0);
/// unsafe { mapreduce_threaded(&[4], &[4], &[vec![1]], &[0], &[2], 1, 0, 1, &|dims, _, _, _| { count.fetch_add(dims.iter().product::<usize>(), Ordering::Relaxed); Ok(()) }) }.unwrap();
/// assert_eq!(count.load(Ordering::Relaxed), 4);
/// ```
///
/// # Errors
/// Forwards errors from the owning kernel or callback; callers must still
/// satisfy the safety contract before invoking this prevalidated entry.
#[inline]
pub unsafe fn mapreduce_threaded<F>(
    dims: &[usize],
    blocks: &[usize],
    strides_list: &[Vec<isize>],
    offsets: &[isize],
    costs: &[isize],
    nthreads: usize,
    spacing: isize,
    taskindex: usize,
    f: &F,
) -> Result<()>
where
    F: Fn(&[usize], &[usize], &[Vec<isize>], &[isize]) -> Result<()> + Sync,
{
    // SAFETY: the caller supplies the same invariants as the owning checked entry.
    crate::threading::mapreduce_threaded::<F>(
        dims,
        blocks,
        strides_list,
        offsets,
        costs,
        nthreads,
        spacing,
        taskindex,
        f,
    )
}

#[cfg(feature = "parallel")]
pub use crate::execution_policy::rayon_threads;

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
/// unsafe { gather_into_uninit(&plan, &mut dst, &src, &indices) }.unwrap();
/// assert_eq!(unsafe { values[1].assume_init() }, 3);
/// ```
///
/// # Errors
/// Forwards the plan layout/index validation errors before successful overwrite.
#[inline]
pub unsafe fn gather_into_uninit<T, I>(
    plan: &crate::GatherPlan,
    dest: &mut crate::RawStridedMut<'_, core::mem::MaybeUninit<T>>,
    operand: &crate::RawStridedRef<'_, T>,
    start_indices: &crate::RawStridedRef<'_, I>,
) -> Result<()>
where
    T: Copy + MaybeSendSync,
    I: crate::GatherIndex,
{
    plan.execute_uninit(dest, operand, start_indices)
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
/// unsafe { dynamic_slice_into_uninit(&plan, &mut dst, &src, &starts) }.unwrap();
/// assert_eq!(unsafe { values[1].assume_init() }, 3);
/// ```
///
/// # Errors
/// Forwards errors from the owning kernel or callback; callers must still
/// satisfy the safety contract before invoking this prevalidated entry.
#[inline]
pub unsafe fn dynamic_slice_into_uninit<T, I>(
    plan: &crate::DynamicSlicePlan,
    dest: &mut crate::RawStridedMut<'_, core::mem::MaybeUninit<T>>,
    operand: &crate::RawStridedRef<'_, T>,
    starts: &crate::RawStridedRef<'_, I>,
) -> Result<()>
where
    T: Copy + MaybeSendSync,
    I: crate::GatherIndex,
{
    plan.execute_uninit(dest, operand, starts)
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
/// unsafe { dynamic_update_into_uninit(&plan, &mut dst, &src, &update, &starts) }.unwrap();
/// assert_eq!(unsafe { values[1].assume_init() }, 9);
/// ```
///
/// # Errors
/// Forwards the plan layout/index validation errors.
#[inline]
pub unsafe fn dynamic_update_into_uninit<'a, T, I>(
    plan: &crate::DynamicUpdateSlicePlan,
    dest: &'a mut crate::RawStridedMut<'a, core::mem::MaybeUninit<T>>,
    operand: &crate::RawStridedRef<'_, T>,
    update: &crate::RawStridedRef<'_, T>,
    starts: &crate::RawStridedRef<'_, I>,
) -> Result<()>
where
    T: Copy + MaybeSendSync,
    I: crate::GatherIndex,
{
    plan.execute_uninit(dest, operand, update, starts)
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
/// unsafe { scatter_into_uninit(&plan, &mut dst, &src, &indices, &update, i32::wrapping_add) }.unwrap();
/// assert_eq!(unsafe { values[1].assume_init() }, 11);
/// ```
///
/// # Errors
/// Forwards the plan layout/index validation errors.
#[inline]
pub unsafe fn scatter_into_uninit<'a, T, I>(
    plan: &crate::ScatterPlan,
    dest: &'a mut crate::RawStridedMut<'a, core::mem::MaybeUninit<T>>,
    operand: &crate::RawStridedRef<'_, T>,
    scatter_indices: &crate::RawStridedRef<'_, I>,
    updates: &crate::RawStridedRef<'_, T>,
    combine: fn(T, T) -> T,
) -> Result<()>
where
    T: Copy + core::ops::Add<Output = T> + MaybeSendSync,
    I: crate::GatherIndex,
{
    plan.execute_uninit(dest, operand, scatter_indices, updates, combine)
}
