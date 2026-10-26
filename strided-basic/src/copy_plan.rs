//! Prepared (compile-once, execute-many) copy plans over raw strided layouts.
//!
//! [`copy_scale_raw`](crate::copy_scale_raw) and friends rebuild the fused
//! loop nest on every call; for prepared-replay consumers that issue many
//! small copies with a fixed layout, that per-call planning dominates
//! (see issue #139). [`CopyPlan`] splits the work: [`CopyPlan::compile`]
//! validates the layout pair and builds the fused traversal once,
//! [`CopyPlan::execute`]/[`CopyPlan::execute_scale`]/[`CopyPlan::execute_conj`]
//! replay it with no planning and no heap allocation for fused ranks at most
//! [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT).

use core::{
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Add, Mul},
};

use num_traits::{One, Zero};

use crate::cost::KernelClass;
use crate::raw_ops::{
    apply_fused_range, apply_spilled_range, fuse_pair_traversal, fused_total, FusedAxis,
    FusedPairLayout, PairTraversal,
};
use crate::{ElementOpApply, Exec, RawStridedMut, RawStridedRef, Result, StridedError};

// Same pattern as map_view.rs / outer_product.rs: inline storage in both
// feature configurations. Only `compile` touches these; `execute*` never
// allocates for them. (A spilled fused layout needs one cursor allocation per
// replayed range, so an `execute*` can allocate once per worker range above the
// inline rank bound.)
type AxisVec<T> = crate::inline::InlineVec<T>;

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
/// Ranks above [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT) are supported by the same fused
/// traversal, which then stores its axes in one rank-bounded allocation.
///
/// # Example
///
/// ```rust
/// use strided_basic::{CopyPlan, Exec, RawStridedMut, RawStridedRef};
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
/// plan.execute(&mut dst_mut, &src_ref, Exec::serial()).unwrap();
/// assert_eq!(dst, [0.0, 10.0, 1.0, 11.0, 2.0, 12.0]);
/// ```
#[derive(Clone, Debug, Default)]
pub struct CopyPlan {
    dims: AxisVec<usize>,
    dst_strides: AxisVec<isize>,
    src_strides: AxisVec<isize>,
    fused: PairTraversal,
}

impl CopyPlan {
    /// `None` when the fused traversal is stored in the rank-bounded spill.
    #[inline]
    pub(crate) fn fused_layout(&self) -> Option<&FusedPairLayout> {
        self.fused.inline()
    }

    pub(crate) fn execute_uninit_then<'a, T, R>(
        &self,
        dest: &'a mut RawStridedMut<'a, MaybeUninit<T>>,
        src: &RawStridedRef<'_, T>,
        mut exec: Exec<'_>,
        f: impl for<'b, 'e> FnOnce(InitializedRawDest<'b, T>, Exec<'e>) -> R,
    ) -> Result<R>
    where
        T: Copy + Send + Sync,
    {
        self.execute_uninit(dest, src, exec.reborrow())?;
        let data = dest.data_mut();
        let receipt = InitializedRawDest {
            ptr: data.as_mut_ptr().cast(),
            extent: data.len(),
            dims: dest.dims(),
            strides: dest.strides(),
            offset: dest.offset(),
            _marker: PhantomData,
        };
        Ok(f(receipt, exec))
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
            fused: fuse_pair_traversal(dims, dst_strides, src_strides),
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
        check_layout_pair(&self.dims, &self.dst_strides, &self.src_strides, dest, src)
    }

    /// `dest = src` into a potentially uninitialized destination.
    ///
    /// On success every reachable logical destination element is initialized.
    /// Non-reachable holes in the backing allocation are not written.
    pub fn execute_uninit<T>(
        &self,
        dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
        src: &RawStridedRef<'_, T>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        self.check_call(dest, src)?;
        replay_fused(
            dest,
            src,
            &self.fused,
            |dst, value| {
                dst.write(value);
            },
            |value| value,
            exec,
        );
        Ok(())
    }

    /// `dest = src`. Allocation-free for fused ranks at most
    /// [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT).
    pub fn execute<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        src: &RawStridedRef<'_, T>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        self.check_call(dest, src)?;
        replay_fused(
            dest,
            src,
            &self.fused,
            |dst, value| *dst = value,
            |value: T| value,
            exec,
        );
        Ok(())
    }

    /// `dest = scale * src`. Allocation-free for fused ranks at most
    /// [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT).
    pub fn execute_scale<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        src: &RawStridedRef<'_, T>,
        scale: T,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Mul<T, Output = T> + Send + Sync,
    {
        self.check_call(dest, src)?;
        replay_fused(
            dest,
            src,
            &self.fused,
            |dst, value| *dst = value,
            |value: T| scale * value,
            exec,
        );
        Ok(())
    }

    /// `dest = conj(src)`. Allocation-free for fused ranks at most
    /// [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT).
    pub fn execute_conj<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        src: &RawStridedRef<'_, T>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + ElementOpApply + Send + Sync,
    {
        self.check_call(dest, src)?;
        replay_fused(
            dest,
            src,
            &self.fused,
            |dst, value| *dst = value,
            |value: T| value.conj(),
            exec,
        );
        Ok(())
    }
}

/// Verify that a pair of raw views carries exactly a compiled layout.
///
/// Shared by [`CopyPlan`] and [`UpdatePlan`], whose `compile` methods prove the
/// layout facts once. Buffer bounds against that layout were already proven by
/// [`RawStridedRef::new`]/[`RawStridedMut::new`] (or asserted by the caller of
/// the `new_unchecked` constructors), so layout equality is the complete
/// per-call precondition for the pointer-based fused replay.
fn check_layout_pair<D, S>(
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dest: &RawStridedMut<'_, D>,
    src: &RawStridedRef<'_, S>,
) -> Result<()> {
    if dest.dims() != dims
        || src.dims() != dims
        || dest.strides() != dst_strides
        || src.strides() != src_strides
    {
        return Err(StridedError::PlanLayoutMismatch);
    }
    Ok(())
}

/// A compiled read-modify-write traversal for one `(dims, dst_strides,
/// src_strides)` layout pair.
///
/// [`execute`](Self::execute) computes `dest = alpha * f(src) + beta * dest` by
/// replaying the same fused pair traversal as [`CopyPlan`]: `compile` proves
/// the layout facts once (rank agreement, extent overflow, destination
/// injectivity) and each call only re-checks that the supplied views carry the
/// compiled layout before running monomorphized loops. Execute is
/// allocation-free for fused ranks at most
/// [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT), and the injective
/// destination makes a large replay parallel-safe exactly as in [`CopyPlan`].
///
/// Overlapping `src`/`dest` memory is not supported, matching the rest of the
/// crate.
///
/// # Example
///
/// ```rust
/// use strided_basic::{Exec, RawStridedMut, RawStridedRef, UpdatePlan};
///
/// let dims = [2usize, 2];
/// let dst_strides = [1isize, 2];
/// let src_strides = [1isize, 2];
/// let plan = UpdatePlan::compile(&dims, &dst_strides, &src_strides).unwrap();
///
/// let src = [1.0f64, 2.0, 3.0, 4.0];
/// let mut dst = [10.0f64, 20.0, 30.0, 40.0];
/// let src_ref = RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap();
/// let mut dst_mut = RawStridedMut::new(&mut dst, &dims, &dst_strides, 0).unwrap();
/// // dst = 2 * src + 3 * dst
/// plan.execute(&mut dst_mut, &src_ref, 2.0, 3.0, |x| x, Exec::serial()).unwrap();
/// assert_eq!(dst, [32.0, 64.0, 96.0, 128.0]);
/// ```
#[derive(Clone, Debug)]
pub struct UpdatePlan {
    dims: AxisVec<usize>,
    dst_strides: AxisVec<isize>,
    src_strides: AxisVec<isize>,
    fused: PairTraversal,
}

impl UpdatePlan {
    /// Compile an update plan for the given layout pair.
    ///
    /// Performs the same layout validation as [`CopyPlan::compile`]:
    ///
    /// - `dims`, `dst_strides`, and `src_strides` must have equal length
    ///   ([`StridedError::StrideLengthMismatch`]);
    /// - the total element count must not overflow `usize`
    ///   ([`StridedError::OffsetOverflow`]);
    /// - the destination layout must be injective
    ///   ([`StridedError::NonInjectiveOutputLayout`]), so two logical indices
    ///   never name one slot and a parallel replay writes each slot once.
    pub fn compile(dims: &[usize], dst_strides: &[isize], src_strides: &[isize]) -> Result<Self> {
        if dims.len() != dst_strides.len() || dims.len() != src_strides.len() {
            return Err(StridedError::StrideLengthMismatch);
        }
        crate::kernel::total_len(dims)?;
        if !crate::layout_check::is_injective_layout(dims, dst_strides) {
            return Err(StridedError::NonInjectiveOutputLayout);
        }
        Ok(Self {
            dims: dims.into(),
            dst_strides: dst_strides.into(),
            src_strides: src_strides.into(),
            fused: fuse_pair_traversal(dims, dst_strides, src_strides),
        })
    }

    /// `dest = alpha * f(src) + beta * dest`.
    ///
    /// `f` is applied to the source element before scaling. When `beta` is
    /// zero (`-0.0` included; `NaN` is not zero) the replay is write-only: the
    /// old destination element is never read, so a NaN-filled or uninitialized
    /// destination cannot contaminate the result, matching BLAS `beta = 0`
    /// semantics. Otherwise the old destination element is read through the
    /// destination layout. Allocation-free for fused ranks at most
    /// [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT).
    ///
    /// # Errors
    ///
    /// [`StridedError::PlanLayoutMismatch`] when the supplied views do not
    /// carry the compiled layout, before any write.
    pub fn execute<T, F>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        src: &RawStridedRef<'_, T>,
        alpha: T,
        beta: T,
        f: F,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync + Add<Output = T> + Mul<Output = T> + Zero + One + PartialEq,
        F: Fn(T) -> T + Send + Sync,
    {
        check_layout_pair(&self.dims, &self.dst_strides, &self.src_strides, dest, src)?;
        if beta.is_zero() {
            // Write-only: never form a read of the old destination.
            if alpha.is_one() {
                replay_fused(dest, src, &self.fused, |dst, value| *dst = value, f, exec);
            } else {
                replay_fused(
                    dest,
                    src,
                    &self.fused,
                    |dst, value| *dst = alpha * value,
                    f,
                    exec,
                );
            }
        } else {
            replay_fused(
                dest,
                src,
                &self.fused,
                |dst, value| *dst = alpha * value + beta * *dst,
                f,
                exec,
            );
        }
        Ok(())
    }

    /// `dest = alpha * f(src)` into a potentially uninitialized destination.
    ///
    /// This is [`execute`](Self::execute) with `beta = 0`: the destination is
    /// never read. On success every reachable logical destination element is
    /// initialized; non-reachable holes in the backing allocation are not
    /// written. Allocation-free for fused ranks at most
    /// [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT).
    ///
    /// # Errors
    ///
    /// [`StridedError::PlanLayoutMismatch`] when the supplied views do not
    /// carry the compiled layout, before any write.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use core::mem::MaybeUninit;
    /// use strided_basic::{Exec, RawStridedMut, RawStridedRef, UpdatePlan};
    ///
    /// let dims = [2usize, 2];
    /// let strides = [1isize, 2];
    /// let plan = UpdatePlan::compile(&dims, &strides, &strides).unwrap();
    /// let src = [1.0f64, 2.0, 3.0, 4.0];
    /// let mut storage = [MaybeUninit::<f64>::uninit(); 4];
    /// let mut dst = RawStridedMut::new(&mut storage, &dims, &strides, 0).unwrap();
    /// let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
    /// // dst = 2 * src
    /// plan.execute_uninit(&mut dst, &source, 2.0, |x| x, Exec::serial()).unwrap();
    /// drop(dst);
    /// // SAFETY: every reachable destination slot was initialized above.
    /// let values: Vec<f64> = storage.iter().map(|v| unsafe { v.assume_init() }).collect();
    /// assert_eq!(values, [2.0, 4.0, 6.0, 8.0]);
    /// ```
    pub fn execute_uninit<T, F>(
        &self,
        dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
        src: &RawStridedRef<'_, T>,
        alpha: T,
        f: F,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync + Mul<Output = T> + One + PartialEq,
        F: Fn(T) -> T + Send + Sync,
    {
        check_layout_pair(&self.dims, &self.dst_strides, &self.src_strides, dest, src)?;
        if alpha.is_one() {
            replay_fused(
                dest,
                src,
                &self.fused,
                |dst, value| {
                    dst.write(value);
                },
                f,
                exec,
            );
        } else {
            replay_fused(
                dest,
                src,
                &self.fused,
                |dst, value| {
                    dst.write(alpha * value);
                },
                f,
                exec,
            );
        }
        Ok(())
    }
}

/// Replay a compiled fused layout, splitting it across the budget of `exec`
/// when the repository threshold allows it.
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
    traversal: &PairTraversal,
    apply: Apply,
    op: Op,
    exec: Exec<'_>,
) where
    D: Copy + Send + Sync,
    S: Copy + Send + Sync,
    Apply: Fn(&mut D, S) + Send + Sync,
    Op: Fn(S) -> S + Send + Sync,
{
    let src_ptr = src.data().as_ptr();
    let src_base = src.offset();
    let dst_base = dest.offset();
    let dst_ptr = dest.data_mut().as_mut_ptr();
    // SAFETY: `check_call` proved both views carry exactly the compiled
    // layout, and `RawStridedRef`/`RawStridedMut` guarantee every offset
    // reachable through that layout lies inside their data. `traversal` is the
    // fusion of that layout, so every logical index in `0..total` is an
    // in-bounds slot of both. The compiled destination layout is injective
    // and the exclusive destination borrow cannot overlap the shared source
    // borrow.
    unsafe {
        match traversal {
            PairTraversal::Inline(layout) => replay_fused_raw(
                dst_ptr, dst_base, src_ptr, src_base, layout, apply, op, exec,
            ),
            PairTraversal::Spilled(axes) => {
                replay_spilled_raw(dst_ptr, dst_base, src_ptr, src_base, axes, apply, op, exec)
            }
        }
    }
}

/// Pointer-level fused replay shared by [`CopyPlan`] and the dynamic slice
/// plans, which replay a fused window layout at a runtime base offset.
///
/// Splits `0..total` across the budget of `exec` when the repository threshold
/// allows more than one slot, otherwise runs the serial kernel directly.
///
/// # Safety
///
/// Every logical index of `layout`, mapped through its destination strides
/// from `dst_base` and its source strides from `src_base`, must be an
/// in-bounds slot of the destination and source allocations. The
/// destination strides must be injective over `layout`, the destination
/// must not overlap the source, and nothing else may access the destination
/// slots during the call.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn replay_fused_raw<D, S, Apply, Op>(
    dst_ptr: *mut D,
    dst_base: isize,
    src_ptr: *const S,
    src_base: isize,
    layout: &FusedPairLayout,
    apply: Apply,
    op: Op,
    exec: Exec<'_>,
) where
    D: Copy + Send + Sync,
    S: Copy + Send + Sync,
    Apply: Fn(&mut D, S) + Send + Sync,
    Op: Fn(S) -> S + Send + Sync,
{
    let total = fused_total(layout);
    if total == 0 {
        return;
    }
    let exec = crate::cost::for_len(exec, KernelClass::Copy, total);
    if !exec.is_serial() {
        let dst_ptr = crate::threading::SendPtr(dst_ptr);
        let src_ptr = crate::threading::SendPtr(src_ptr as *mut S);
        crate::threading::parallel_for_each(0..total, exec, &|range| {
            // SAFETY: the caller contract makes every index in bounds; the
            // worker ranges are disjoint and the destination strides
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
    // SAFETY: forwarded caller contract over the full range `0..total`.
    unsafe {
        apply_fused_range(
            dst_ptr, dst_base, src_ptr, src_base, layout, 0, total, &apply, &op,
        );
    }
}

/// Pointer-level replay of a spilled axis list.
///
/// Mirrors [`replay_fused_raw`] including its worker split; kept separate so
/// that the inline replay keeps its fixed-rank shape.
///
/// # Safety
///
/// Every logical index of `axes`, mapped through its destination strides from
/// `dst_base` and its source strides from `src_base`, must be an in-bounds slot
/// of the destination and source allocations. The destination strides must be
/// injective over `axes`, the destination must not overlap the source, and
/// nothing else may access the destination slots during the call.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn replay_spilled_raw<D, S, Apply, Op>(
    dst_ptr: *mut D,
    dst_base: isize,
    src_ptr: *const S,
    src_base: isize,
    axes: &[FusedAxis],
    apply: Apply,
    op: Op,
    exec: Exec<'_>,
) where
    D: Copy + Send + Sync,
    S: Copy + Send + Sync,
    Apply: Fn(&mut D, S) + Send + Sync,
    Op: Fn(S) -> S + Send + Sync,
{
    let total = axes.iter().map(|axis| axis.dim).product::<usize>();
    if total == 0 {
        return;
    }
    let exec = crate::cost::for_len(exec, KernelClass::Copy, total);
    if !exec.is_serial() {
        let dst_ptr = crate::threading::SendPtr(dst_ptr);
        let src_ptr = crate::threading::SendPtr(src_ptr as *mut S);
        crate::threading::parallel_for_each(0..total, exec, &|range| {
            // SAFETY: the caller contract makes every index in bounds; the
            // worker ranges are disjoint and the destination strides
            // injective, so no two workers write the same slot, and the
            // source is only read.
            unsafe {
                apply_spilled_range(
                    dst_ptr.as_ptr(),
                    dst_base,
                    src_ptr.as_const(),
                    src_base,
                    axes,
                    range.start,
                    range.len(),
                    &apply,
                    &op,
                );
            }
        });
        return;
    }
    // SAFETY: forwarded caller contract over the full range `0..total`.
    unsafe {
        apply_spilled_range(
            dst_ptr, dst_base, src_ptr, src_base, axes, 0, total, &apply, &op,
        );
    }
}

#[cfg(test)]
#[path = "copy_plan/tests/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "copy_plan/tests/parallel_tests.rs"]
mod parallel_tests;
