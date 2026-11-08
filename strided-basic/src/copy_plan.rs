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

    /// Copy into uninitialized storage, then lend the initialized destination to `f`.
    ///
    /// [`Self::execute_uninit`] initializes every reachable logical destination
    /// element before `f` runs, so the [`InitializedRawDest`] receipt handed to `f`
    /// may read and overwrite exactly those offsets. Slots outside the plan's
    /// reachable set are not written by that copy and it makes no claim about
    /// them: a caller that handed in a `MaybeUninit` array still owns those slots,
    /// so they keep whatever initialization the caller left. With
    /// `CopyPlan::compile(&[2], &[2], &[1])` the offset-1 hole keeps the caller's
    /// value (measured `[7.0, 99.0, 8.0]` for `src = [7, 8]`) rather than
    /// becoming uninitialized. That is why the receipt exposes
    /// `write_at`/`add_at` (copy_plan.rs:138, :150) instead of a typed slice: their
    /// contracts require an offset proven by this plan's layout. The receipt borrows
    /// the destination for `f`'s lifetime, so a read-modify-write consumer (the
    /// gather and scatter update replay in gather_plan.rs) sees the copied operand
    /// once and cannot alias the source.
    ///
    /// # Errors
    ///
    /// Returns the [`Self::execute_uninit`] errors (`PlanLayoutMismatch`, before any
    /// write); `f` runs only on success.
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
    /// (fuse + order) once.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::StrideLengthMismatch`] when `dims`,
    /// `dst_strides`, and `src_strides` do not all have the same length,
    /// [`StridedError::OffsetOverflow`] when the total element count does not
    /// fit `usize`, and [`StridedError::NonInjectiveOutputLayout`] when the
    /// destination layout cannot be proven injective (aliasing elements, or a
    /// span the check cannot represent).
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

    /// Check the per-call facts, then replay the compiled traversal.
    fn replay<D, S, Apply>(
        &self,
        dest: &mut RawStridedMut<'_, D>,
        src: &RawStridedRef<'_, S>,
        apply: Apply,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        D: Copy + Send + Sync,
        S: Copy + Send + Sync,
        Apply: Fn(&mut D, S) + Sync,
    {
        self.check_call(dest, src)?;
        // SAFETY: `check_call` proved both views carry exactly the compiled
        // layout, and `fused` is the fusion of that layout (only `compile`
        // and `Default`, the rank-zero layout, build it). `compile` checked
        // the element count and proved the destination injective.
        unsafe { replay_fused(dest, src, &self.fused, apply, exec) };
        Ok(())
    }

    /// `dest = src` into a potentially uninitialized destination.
    ///
    /// On success every reachable logical destination element is initialized.
    /// Non-reachable holes in the backing allocation are not written.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::PlanLayoutMismatch`] when the views do not
    /// carry the compiled layout, before any write.
    pub fn execute_uninit<T>(
        &self,
        dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
        src: &RawStridedRef<'_, T>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        self.replay(
            dest,
            src,
            |dst: &mut MaybeUninit<T>, value: T| {
                dst.write(value);
            },
            exec,
        )
    }

    /// `dest = src`. Allocation-free for fused ranks at most
    /// [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT).
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::PlanLayoutMismatch`] when the views do not
    /// carry the compiled layout, before any write.
    pub fn execute<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        src: &RawStridedRef<'_, T>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        self.replay(dest, src, |dst: &mut T, value: T| *dst = value, exec)
    }

    /// `dest = scale * src`. Allocation-free for fused ranks at most
    /// [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT).
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::PlanLayoutMismatch`] when the views do not
    /// carry the compiled layout, before any write.
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
        self.replay(
            dest,
            src,
            |dst: &mut T, value: T| *dst = scale * value,
            exec,
        )
    }

    /// `dest = conj(src)`. Allocation-free for fused ranks at most
    /// [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT).
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::PlanLayoutMismatch`] when the views do not
    /// carry the compiled layout, before any write.
    pub fn execute_conj<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        src: &RawStridedRef<'_, T>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + ElementOpApply + Send + Sync,
    {
        self.replay(dest, src, |dst: &mut T, value: T| *dst = value.conj(), exec)
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
    /// Performs the same layout validation as [`CopyPlan::compile`].
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::StrideLengthMismatch`] when `dims`,
    /// `dst_strides`, and `src_strides` do not all have the same length,
    /// [`StridedError::OffsetOverflow`] when the total element count does not
    /// fit `usize`, and [`StridedError::NonInjectiveOutputLayout`] when the
    /// destination layout cannot be proven injective: two logical indices name
    /// one slot (so a parallel replay would write that slot once per index), or
    /// the offset span cannot be represented, leaving injectivity undecided.
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

    /// Check the per-call facts, then replay the compiled traversal.
    fn replay<D, S, Apply>(
        &self,
        dest: &mut RawStridedMut<'_, D>,
        src: &RawStridedRef<'_, S>,
        apply: Apply,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        D: Copy + Send + Sync,
        S: Copy + Send + Sync,
        Apply: Fn(&mut D, S) + Sync,
    {
        check_layout_pair(&self.dims, &self.dst_strides, &self.src_strides, dest, src)?;
        // SAFETY: as in `CopyPlan::replay`: the views carry the compiled
        // layout, `fused` is its fusion (built only by `compile`), and
        // `compile` checked the count and destination injectivity.
        unsafe { replay_fused(dest, src, &self.fused, apply, exec) };
        Ok(())
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
        if beta.is_zero() {
            // Write-only: never form a read of the old destination.
            if alpha.is_one() {
                self.replay(dest, src, |dst: &mut T, value: T| *dst = f(value), exec)
            } else {
                self.replay(
                    dest,
                    src,
                    |dst: &mut T, value: T| *dst = alpha * f(value),
                    exec,
                )
            }
        } else {
            self.replay(
                dest,
                src,
                |dst: &mut T, value: T| *dst = alpha * f(value) + beta * *dst,
                exec,
            )
        }
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
        if alpha.is_one() {
            self.replay(
                dest,
                src,
                |dst: &mut MaybeUninit<T>, value: T| {
                    dst.write(f(value));
                },
                exec,
            )
        } else {
            self.replay(
                dest,
                src,
                |dst: &mut MaybeUninit<T>, value: T| {
                    dst.write(alpha * f(value));
                },
                exec,
            )
        }
    }
}

/// Replay `traversal` over a pair of raw views, splitting it across the budget
/// of `exec` when the repository threshold allows it.
///
/// The logical index space `0..total` (column-major over the traversal axes)
/// is split into contiguous worker ranges, which divides the outer axes and,
/// when there are fewer outer positions than workers, also chunks long inner
/// runs (for example a rank-1 contiguous copy).
///
/// # Safety
///
/// - `traversal` must replay exactly the logical index space of the two
///   views: either their dims and strides fused by `fuse_pair_traversal`, or
///   built from them by `PairTraversal::given`. The views' own bounds proofs
///   ([`RawStridedRef::new`], [`RawStridedMut::new`], or the caller of their
///   `new_unchecked`) then make every replayed offset in-bounds.
/// - The product of the traversal extents must not overflow `usize` (a
///   checked count, zero included).
/// - Unless `exec.is_serial()`, the destination strides must be injective
///   over the traversal.
/// - The destination and source must not overlap.
pub(crate) unsafe fn replay_fused<D, S, Apply>(
    dest: &mut RawStridedMut<'_, D>,
    src: &RawStridedRef<'_, S>,
    traversal: &PairTraversal,
    apply: Apply,
    exec: Exec<'_>,
) where
    D: Copy + Send + Sync,
    S: Copy + Send + Sync,
    Apply: Fn(&mut D, S) + Sync,
{
    let src_ptr = src.data().as_ptr();
    let src_base = src.offset();
    let dst_base = dest.offset();
    let dst_ptr = dest.data_mut().as_mut_ptr();
    // SAFETY: forwarded caller contract: every logical index of `traversal`
    // maps to in-bounds slots of both views, injectivity holds whenever the
    // token can fan out, and the borrows do not overlap.
    unsafe {
        match traversal {
            PairTraversal::Inline(layout) => {
                replay_fused_raw(dst_ptr, dst_base, src_ptr, src_base, layout, apply, exec)
            }
            PairTraversal::Spilled(axes) => {
                replay_spilled_raw(dst_ptr, dst_base, src_ptr, src_base, axes, apply, exec)
            }
        }
    }
}

/// Pointer-level fused replay shared by [`CopyPlan`], [`UpdatePlan`], the
/// plan-free `replay_*_raw` entries and the dynamic slice plans, which replay
/// a fused window layout at a runtime base offset.
///
/// Splits `0..total` across the budget of `exec` when the repository threshold
/// allows more than one slot, otherwise runs the serial kernel directly.
///
/// # Safety
///
/// Every logical index of `layout`, mapped through its destination strides
/// from `dst_base` and its source strides from `src_base`, must be an
/// in-bounds slot of the destination and source allocations, and the product
/// of its extents must not overflow `usize`. Unless `exec.is_serial()`, the
/// destination strides must be injective over `layout`; a serial replay
/// updates a repeated slot once per logical index, in order. The destination
/// must not overlap the source, and nothing else may access the destination
/// slots during the call.
pub(crate) unsafe fn replay_fused_raw<D, S, Apply>(
    dst_ptr: *mut D,
    dst_base: isize,
    src_ptr: *const S,
    src_base: isize,
    layout: &FusedPairLayout,
    apply: Apply,
    exec: Exec<'_>,
) where
    D: Copy + Send + Sync,
    S: Copy + Send + Sync,
    Apply: Fn(&mut D, S) + Sync,
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
            // token is not serial, so the destination strides are injective
            // and the disjoint worker ranges write disjoint slots; the source
            // is only read.
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
                );
            }
        });
        return;
    }
    // SAFETY: forwarded caller contract over the full range `0..total`; one
    // sequential walk needs no injectivity.
    unsafe {
        apply_fused_range(
            dst_ptr, dst_base, src_ptr, src_base, layout, 0, total, &apply,
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
/// The [`replay_fused_raw`] contract, for `axes`.
pub(crate) unsafe fn replay_spilled_raw<D, S, Apply>(
    dst_ptr: *mut D,
    dst_base: isize,
    src_ptr: *const S,
    src_base: isize,
    axes: &[FusedAxis],
    apply: Apply,
    exec: Exec<'_>,
) where
    D: Copy + Send + Sync,
    S: Copy + Send + Sync,
    Apply: Fn(&mut D, S) + Sync,
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
            // SAFETY: as in `replay_fused_raw`: in-bounds indices, injective
            // destination for a non-serial token, disjoint worker ranges.
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
                );
            }
        });
        return;
    }
    // SAFETY: forwarded caller contract over the full range `0..total`; one
    // sequential walk needs no injectivity.
    unsafe {
        apply_spilled_range(dst_ptr, dst_base, src_ptr, src_base, axes, 0, total, &apply);
    }
}

#[cfg(test)]
#[path = "copy_plan/tests/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "copy_plan/tests/parallel_tests.rs"]
mod parallel_tests;
