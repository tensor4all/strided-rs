//! Prepared (compile-once, execute-many) copy plans over raw strided layouts.
//!
//! [`copy_scale_raw`](crate::copy_scale_raw) and friends rebuild the fused
//! loop nest on every call; for prepared-replay consumers that issue many
//! small copies with a fixed layout, that per-call planning dominates
//! (see issue #139). [`CopyPlan`] splits the work: [`CopyPlan::compile`]
//! validates the layout pair and builds the fused traversal once,
//! [`CopyPlan::execute`]/[`CopyPlan::execute_scale`]/[`CopyPlan::execute_conj`]
//! replay it with no planning and no heap allocation for ranks at most
//! [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT).

use core::{
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Add, Mul},
};

use crate::map_view::map_raw_into;
use crate::ops_view::{copy_conj, copy_into, copy_scale};
use crate::raw_ops::{apply_fused_range, fuse_pair_layout, fused_total, FusedPairLayout};
use crate::{
    ElementOpApply, Identity, MaybeSendSync, RawStridedMut, RawStridedRef, Result, StridedError,
};

// Same pattern as map_view.rs / outer_product.rs: stack storage when the
// parallel feature pulls in smallvec, plain Vec otherwise. Only `compile`
// touches these; `execute*` never allocates either way.
#[cfg(feature = "parallel")]
type AxisVec<T> = smallvec::SmallVec<[T; crate::RAW_FUSED_RANK_LIMIT]>;
#[cfg(not(feature = "parallel"))]
type AxisVec<T> = Vec<T>;

pub(crate) trait OverwriteWriter<T> {
    fn dims(&self) -> &[usize];
    fn strides(&self) -> &[isize];
    fn offset(&self) -> isize;
    /// # Safety
    /// The caller must use the pointer only for the validated allocation and
    /// logical layout represented by this writer.
    unsafe fn data_ptr(&mut self) -> *mut T;
    /// # Safety
    /// The offset must be an in-bounds logical destination proven by layout.
    unsafe fn write_at(&mut self, offset: isize, value: T);
}

pub(crate) trait ReadModifyWrite<T>: OverwriteWriter<T> {
    /// # Safety
    /// The offset must be an in-bounds initialized slot covered by the
    /// traversal's copy and disjointness proof.
    unsafe fn add_at<F>(&mut self, offset: isize, value: T, combine: F)
    where
        F: FnOnce(T, T) -> T;
}

impl<'a, T> OverwriteWriter<T> for RawStridedMut<'a, T> {
    fn dims(&self) -> &[usize] {
        self.dims()
    }
    fn strides(&self) -> &[isize] {
        self.strides()
    }
    fn offset(&self) -> isize {
        self.offset()
    }
    unsafe fn data_ptr(&mut self) -> *mut T {
        self.data_mut().as_mut_ptr()
    }
    unsafe fn write_at(&mut self, offset: isize, value: T) {
        // SAFETY: the prepared layout validates every logical destination.
        unsafe { self.data_mut().as_mut_ptr().offset(offset).write(value) }
    }
}

impl<'a, T> ReadModifyWrite<T> for RawStridedMut<'a, T>
where
    T: Add<Output = T>,
{
    #[inline(always)]
    unsafe fn add_at<F>(&mut self, offset: isize, value: T, combine: F)
    where
        F: FnOnce(T, T) -> T,
    {
        // SAFETY: the copy or initialized caller proves this logical slot.
        unsafe {
            let ptr = self.data_mut().as_mut_ptr().offset(offset);
            ptr.write(combine(ptr.read(), value));
        }
    }
}

impl<'a, T> OverwriteWriter<T> for RawStridedMut<'a, MaybeUninit<T>> {
    fn dims(&self) -> &[usize] {
        self.dims()
    }
    fn strides(&self) -> &[isize] {
        self.strides()
    }
    fn offset(&self) -> isize {
        self.offset()
    }
    unsafe fn data_ptr(&mut self) -> *mut T {
        self.data_mut().as_mut_ptr().cast()
    }
    unsafe fn write_at(&mut self, offset: isize, value: T) {
        // SAFETY: the prepared layout validates every logical destination.
        unsafe {
            self.data_mut()
                .as_mut_ptr()
                .offset(offset)
                .write(MaybeUninit::new(value))
        }
    }
}

pub(crate) struct InitializedRawDest<'a, T> {
    ptr: *mut T,
    extent: usize,
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
    _marker: PhantomData<&'a mut [MaybeUninit<T>]>,
}

impl<'a, T> OverwriteWriter<T> for InitializedRawDest<'a, T> {
    fn dims(&self) -> &[usize] {
        self.dims
    }
    fn strides(&self) -> &[isize] {
        self.strides
    }
    fn offset(&self) -> isize {
        self.offset
    }
    unsafe fn data_ptr(&mut self) -> *mut T {
        self.ptr
    }
    unsafe fn write_at(&mut self, offset: isize, value: T) {
        debug_assert!(offset >= 0 && (offset as usize) < self.extent);
        // SAFETY: the copy proof and extent check cover this logical slot.
        unsafe { self.ptr.offset(offset).write(value) }
    }
}

impl<'a, T> ReadModifyWrite<T> for InitializedRawDest<'a, T>
where
    T: Add<Output = T>,
{
    #[inline(always)]
    unsafe fn add_at<F>(&mut self, offset: isize, value: T, combine: F)
    where
        F: FnOnce(T, T) -> T,
    {
        debug_assert!(offset >= 0 && (offset as usize) < self.extent);
        // SAFETY: the copy proof and extent check cover this logical slot.
        unsafe {
            let ptr = self.ptr.offset(offset);
            ptr.write(combine(ptr.read(), value));
        }
    }
}

/// A compiled copy traversal for one `(dims, dst_strides, src_strides)`
/// layout pair.
///
/// `compile` proves the layout facts once (rank agreement, extent overflow,
/// destination injectivity) and fuses/orders the loop nest; each `execute*`
/// call then only re-checks the per-call facts (that the supplied views carry
/// exactly the compiled layout) before replaying the prepared loops.
///
/// Overlapping `src`/`dest` memory is not supported, matching the rest of the
/// crate.
///
/// Ranks above [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT) are supported through the view-based
/// kernels; only that fallback path may allocate.
///
/// # Example
///
/// ```rust
/// use strided_basic::{CopyPlan, RawStridedMut, RawStridedRef};
///
/// let dims = [2usize, 3];
/// let src_strides = [3isize, 1];
/// let dst_strides = [1isize, 2]; // transposed destination
/// let plan = CopyPlan::compile(&dims, &dst_strides, &src_strides).unwrap();
///
/// let src = [0.0f64, 1.0, 2.0, 10.0, 11.0, 12.0];
/// let mut dst = [0.0f64; 6];
/// let src_ref = RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap();
/// let mut dst_mut = RawStridedMut::new(&mut dst, &dims, &dst_strides, 0).unwrap();
/// plan.execute(&mut dst_mut, &src_ref).unwrap();
/// assert_eq!(dst, [0.0, 10.0, 1.0, 11.0, 2.0, 12.0]);
/// ```
#[derive(Clone, Debug)]
pub struct CopyPlan {
    dims: AxisVec<usize>,
    dst_strides: AxisVec<isize>,
    src_strides: AxisVec<isize>,
    /// `None` when rank exceeds [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT); `execute*` then
    /// falls back to the view-based kernels.
    fused: Option<FusedPairLayout>,
}

impl CopyPlan {
    /// The fused traversal, or `None` above
    /// [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT).
    #[cfg(feature = "parallel")]
    #[inline]
    pub(crate) fn fused_layout(&self) -> Option<&FusedPairLayout> {
        self.fused.as_ref()
    }

    pub(crate) fn execute_uninit_then<'a, T, R>(
        &self,
        dest: &'a mut RawStridedMut<'a, MaybeUninit<T>>,
        src: &RawStridedRef<'_, T>,
        f: impl for<'b> FnOnce(InitializedRawDest<'b, T>) -> R,
    ) -> Result<R>
    where
        T: Copy + MaybeSendSync,
    {
        self.execute_uninit(dest, src)?;
        let data = dest.data_mut();
        let receipt = InitializedRawDest {
            ptr: data.as_mut_ptr().cast(),
            extent: data.len(),
            dims: dest.dims(),
            strides: dest.strides(),
            offset: dest.offset(),
            _marker: PhantomData,
        };
        Ok(f(receipt))
    }

    /// Compile a copy plan for the given layout pair.
    ///
    /// Performs the layout validation and traversal construction
    /// (fuse + order) once:
    ///
    /// - `dims`, `dst_strides`, and `src_strides` must have equal length
    ///   ([`StridedError::StrideLengthMismatch`]);
    /// - the total element count must not overflow `usize`
    ///   ([`StridedError::OffsetOverflow`]);
    /// - the destination layout must be injective, i.e. map distinct logical
    ///   indices to distinct offsets
    ///   ([`StridedError::NonInjectiveOutputLayout`]).
    pub fn compile(dims: &[usize], dst_strides: &[isize], src_strides: &[isize]) -> Result<Self> {
        if dims.len() != dst_strides.len() || dims.len() != src_strides.len() {
            return Err(StridedError::StrideLengthMismatch);
        }
        // A zero-sized layout has zero elements even when its other extents
        // overflow; only a nonempty overflowing count is rejected.
        crate::kernel::total_len(dims)?;
        if !crate::layout_check::is_injective_layout(dims, dst_strides) {
            return Err(StridedError::NonInjectiveOutputLayout);
        }
        Ok(Self {
            dims: dims.into(),
            dst_strides: dst_strides.into(),
            src_strides: src_strides.into(),
            fused: fuse_pair_layout(dims, dst_strides, src_strides),
        })
    }

    /// Check the per-call facts: the supplied views must carry exactly the
    /// compiled layout. Buffer bounds against that layout were already proven
    /// by [`RawStridedRef::new`]/[`RawStridedMut::new`] (or asserted by the
    /// caller of the `new_unchecked` constructors), so layout equality is the
    /// complete precondition for the pointer-based fused replay below.
    fn check_call<D, S>(
        &self,
        dest: &RawStridedMut<'_, D>,
        src: &RawStridedRef<'_, S>,
    ) -> Result<()> {
        if dest.dims() != &self.dims[..]
            || src.dims() != &self.dims[..]
            || dest.strides() != &self.dst_strides[..]
            || src.strides() != &self.src_strides[..]
        {
            return Err(StridedError::PlanLayoutMismatch);
        }
        Ok(())
    }

    /// `dest = src` into a potentially uninitialized destination.
    ///
    /// On success every reachable logical destination element is initialized.
    /// Non-reachable holes in the backing allocation are not written.
    pub fn execute_uninit<T>(
        &self,
        dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
        src: &RawStridedRef<'_, T>,
    ) -> Result<()>
    where
        T: Copy + MaybeSendSync,
    {
        self.check_call(dest, src)?;
        match &self.fused {
            Some(layout) => {
                replay_fused(
                    dest,
                    src,
                    layout,
                    |dst, value| {
                        dst.write(value);
                    },
                    |value| value,
                );
                Ok(())
            }
            None => map_raw_into::<MaybeUninit<T>, T, Identity>(dest, src, MaybeUninit::new),
        }
    }

    /// `dest = src`. Allocation-free for ranks at most [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT).
    pub fn execute<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        src: &RawStridedRef<'_, T>,
    ) -> Result<()>
    where
        T: Copy + MaybeSendSync,
    {
        self.check_call(dest, src)?;
        match &self.fused {
            Some(layout) => {
                replay_fused(
                    dest,
                    src,
                    layout,
                    |dst, value| *dst = value,
                    |value: T| value,
                );
                Ok(())
            }
            None => copy_into(&mut dest.as_view_mut(), &src.as_view()),
        }
    }

    /// `dest = scale * src`. Allocation-free for ranks at most
    /// [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT).
    pub fn execute_scale<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        src: &RawStridedRef<'_, T>,
        scale: T,
    ) -> Result<()>
    where
        T: Copy + Mul<T, Output = T> + MaybeSendSync,
    {
        self.check_call(dest, src)?;
        match &self.fused {
            Some(layout) => {
                replay_fused(
                    dest,
                    src,
                    layout,
                    |dst, value| *dst = value,
                    |value: T| scale * value,
                );
                Ok(())
            }
            None => copy_scale(&mut dest.as_view_mut(), &src.as_view(), scale),
        }
    }

    /// `dest = conj(src)`. Allocation-free for ranks at most
    /// [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT).
    pub fn execute_conj<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        src: &RawStridedRef<'_, T>,
    ) -> Result<()>
    where
        T: Copy + ElementOpApply + MaybeSendSync,
    {
        self.check_call(dest, src)?;
        match &self.fused {
            Some(layout) => {
                replay_fused(
                    dest,
                    src,
                    layout,
                    |dst, value| *dst = value,
                    |value: T| value.conj(),
                );
                Ok(())
            }
            None => copy_conj(&mut dest.as_view_mut(), &src.as_view()),
        }
    }
}

/// Replay a compiled fused layout, splitting it across worker threads when the
/// repository threshold and the active execution policy allow it.
///
/// Parallel replay is sound only because [`CopyPlan::compile`] proved the
/// destination layout injective: disjoint logical ranges then write disjoint
/// destination slots. The logical index space `0..total` (column-major over
/// the fused axes) is split into contiguous worker ranges, which divides the
/// outer fused axes and, when there are fewer outer positions than workers,
/// also chunks long inner runs (for example a rank-1 contiguous copy).
fn replay_fused<D, S, Apply, Op>(
    dest: &mut RawStridedMut<'_, D>,
    src: &RawStridedRef<'_, S>,
    layout: &FusedPairLayout,
    apply: Apply,
    op: Op,
) where
    D: Copy + MaybeSendSync,
    S: Copy + MaybeSendSync,
    Apply: Fn(&mut D, S) + MaybeSendSync,
    Op: Fn(S) -> S + MaybeSendSync,
{
    let src_ptr = src.data().as_ptr();
    let src_base = src.offset();
    let dst_base = dest.offset();
    let dst_ptr = dest.data_mut().as_mut_ptr();
    // SAFETY: `check_call` proved both views carry exactly the compiled
    // layout, and `RawStridedRef`/`RawStridedMut` guarantee every offset
    // reachable through that layout lies inside their data. `layout` is the
    // fusion of that layout, so every logical index in `0..total` is an
    // in-bounds slot of both. The compiled destination layout is injective
    // and the exclusive destination borrow cannot overlap the shared source
    // borrow.
    unsafe { replay_fused_raw(dst_ptr, dst_base, src_ptr, src_base, layout, apply, op) }
}

/// Pointer-level fused replay shared by [`CopyPlan`] and the dynamic slice
/// plans, which replay a fused window layout at a runtime base offset.
///
/// Splits `0..total` across workers when the repository threshold and the
/// active execution policy allow more than one thread, otherwise runs the
/// serial kernel directly.
///
/// # Safety
///
/// Every logical index of `layout`, mapped through its destination strides
/// from `dst_base` and its source strides from `src_base`, must be an
/// in-bounds slot of the destination and source allocations. The
/// destination strides must be injective over `layout`, the destination
/// must not overlap the source, and nothing else may access the destination
/// slots during the call.
pub(crate) unsafe fn replay_fused_raw<D, S, Apply, Op>(
    dst_ptr: *mut D,
    dst_base: isize,
    src_ptr: *const S,
    src_base: isize,
    layout: &FusedPairLayout,
    apply: Apply,
    op: Op,
) where
    D: Copy + MaybeSendSync,
    S: Copy + MaybeSendSync,
    Apply: Fn(&mut D, S) + MaybeSendSync,
    Op: Fn(S) -> S + MaybeSendSync,
{
    let total = fused_total(layout);
    if total == 0 {
        return;
    }
    #[cfg(feature = "parallel")]
    {
        let nthreads = crate::threading::parallel_threads_for_len(total);
        if nthreads > 1 {
            let dst_ptr = crate::threading::SendPtr(dst_ptr);
            let src_ptr = crate::threading::SendPtr(src_ptr as *mut S);
            crate::threading::parallel_for_each(0..total, nthreads, &|range| {
                // SAFETY: the caller contract makes every index in bounds;
                // the worker ranges are disjoint and the destination strides
                // injective, so no two workers write the same slot, and the
                // source is only read.
                unsafe {
                    apply_fused_range(
                        dst_ptr.as_ptr(),
                        dst_base,
                        src_ptr.as_const(),
                        src_base,
                        layout,
                        range.start,
                        range.len(),
                        &apply,
                        &op,
                    );
                }
            });
            return;
        }
    }
    // SAFETY: forwarded caller contract over the full range `0..total`.
    unsafe {
        apply_fused_range(
            dst_ptr, dst_base, src_ptr, src_base, layout, 0, total, &apply, &op,
        );
    }
}

#[cfg(test)]
#[path = "copy_plan/tests/tests.rs"]
mod tests;

#[cfg(all(test, feature = "parallel"))]
#[path = "copy_plan/tests/parallel_tests.rs"]
mod parallel_tests;
