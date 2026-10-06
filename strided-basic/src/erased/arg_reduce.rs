//! Dtype-erased `argmax` / `argmin` along one axis.

use super::line::{for_each_unit, LineLayout, UnitKernel, UnitPtr, PANEL};
use super::{
    check_reduce_layout_offset_arithmetic, checked_total_len, reduce_uninit_writer, reduce_writer,
    ReduceWriter,
};
use crate::erased_common::{check_dtype, validate_uninit_no_overlap};
use crate::*;
use num_complex::{Complex32, Complex64};

/// Runtime operation of an [`ErasedArgReducePlan`].
///
/// # Examples
///
/// ```
/// use strided_basic::ArgReduceOp;
/// assert_ne!(ArgReduceOp::Max, ArgReduceOp::MaxAbs);
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArgReduceOp {
    /// Index of the largest value. Real dtypes only.
    Max,
    /// Index of the smallest value. Real dtypes only.
    Min,
    /// Index of the largest magnitude (`|x|`, the modulus for complex input).
    MaxAbs,
    /// Index of the smallest magnitude (`|x|`, the modulus for complex input).
    MinAbs,
}

impl ArgReduceOp {
    #[inline]
    fn label(self) -> &'static str {
        match self {
            Self::Max => "argmax",
            Self::Min => "argmin",
            Self::MaxAbs => "argmax_abs",
            Self::MinAbs => "argmin_abs",
        }
    }
}

/// Dtype-erased `argmax` / `argmin` along one axis.
///
/// The destination holds one index per line: its dimensions are the source
/// dimensions with `axis` removed (batch axes are preserved in order), and its
/// dtype is the index dtype, `i32` or `i64`. Indices count from the start of
/// the logical axis, whatever the sign of its stride.
///
/// Supported source dtypes are `f32`, `f64`, `i32` and `i64` for every
/// [`ArgReduceOp`], and `c32` / `c64` for the magnitude variants
/// ([`ArgReduceOp::MaxAbs`], [`ArgReduceOp::MinAbs`]) only.
///
/// # Ties, NaN and magnitudes
///
/// * Ties resolve to the **lowest index**. `-0.0` and `+0.0` compare equal,
///   so they tie.
/// * NaN propagates like [`ReduceOp::Max`]: if a line contains a NaN, the
///   result is the index of its **first** NaN, for both the max and the min
///   variants. A complex element is NaN when either component is NaN. This
///   matches NumPy, PyTorch and JAX.
/// * Infinities order normally; `+inf` beats every finite value for `Max`.
/// * The complex magnitude is the overflow-safe modulus (`hypot` of the
///   components), so values near the top of the exponent range still order
///   correctly. A component of `±inf` makes the magnitude `+inf` unless the
///   other component is NaN.
/// * The integer magnitude is the unsigned absolute value, so `i32::MIN` is
///   larger than `i32::MAX`.
///
/// The result does not depend on the layout, execution context or thread
/// count.
///
/// # Examples
///
/// ```
/// use strided_basic::{
///     ArgReduceOp, ErasedArgReducePlan, ErasedRawStridedMut, ErasedRawStridedRef, ExecContext,
///     KernelDType,
/// };
///
/// // Column-major 3 x 2 matrix; argmax down each column.
/// let src = [1.0_f64, 5.0, 5.0, -2.0, -7.0, 3.0];
/// let mut out = [0_i64; 2];
/// let plan = ErasedArgReducePlan::compile(
///     KernelDType::F64,
///     KernelDType::I64,
///     ArgReduceOp::Max,
///     &[3, 2],
///     &[1, 3],
///     &[2],
///     &[1],
///     0,
/// )
/// .unwrap();
/// let src_ref = ErasedRawStridedRef::from_slice(&src, &[3, 2], &[1, 3], 0).unwrap();
/// let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &[2], &[1], 0).unwrap();
/// plan.execute(&ExecContext::serial(), &mut dest, &src_ref).unwrap();
/// assert_eq!(out, [1, 2]); // lowest index wins the 5.0 tie
/// ```
#[derive(Clone, Debug)]
pub struct ErasedArgReducePlan {
    dtype: KernelDType,
    index_dtype: KernelDType,
    op: ArgReduceOp,
    src_dims: Vec<usize>,
    src_strides: Vec<isize>,
    dest_dims: Vec<usize>,
    dest_strides: Vec<isize>,
    layout: LineLayout,
}

impl ErasedArgReducePlan {
    /// Validate and store an arg-reduction plan for fixed layouts.
    ///
    /// `dest_dims` must equal `src_dims` with `axis` removed.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::{ArgReduceOp, ErasedArgReducePlan, KernelDType};
    /// let plan = ErasedArgReducePlan::compile(
    ///     KernelDType::C64, KernelDType::I32, ArgReduceOp::MaxAbs, &[4, 3], &[3, 1], &[3], &[1], 0,
    /// )
    /// .unwrap();
    /// assert_eq!(plan.index_dtype(), KernelDType::I32);
    /// // Complex input has no order without the magnitude.
    /// assert!(ErasedArgReducePlan::compile(
    ///     KernelDType::C64, KernelDType::I64, ArgReduceOp::Max, &[4], &[1], &[], &[], 0,
    /// )
    /// .is_err());
    /// // An empty axis has no index to return.
    /// assert!(ErasedArgReducePlan::compile(
    ///     KernelDType::F64, KernelDType::I64, ArgReduceOp::Max, &[0, 2], &[1, 1], &[2], &[1], 0,
    /// )
    /// .is_err());
    /// ```
    ///
    /// # Errors
    ///
    /// * `UnsupportedDType` for a `bool` source or an index dtype other than
    ///   `i32` / `i64`;
    /// * `UnsupportedOp` for `Max` / `Min` on complex input, and for a
    ///   zero-length `axis`, which has no index to return;
    /// * `InvalidAxis`, `StrideLengthMismatch`, `ShapeMismatch` and
    ///   `NonInjectiveOutputLayout` for inconsistent layouts;
    /// * `OffsetOverflow` when a layout's offsets are not representable, or
    ///   when the axis is too long for an `i32` index.
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        dtype: KernelDType,
        index_dtype: KernelDType,
        op: ArgReduceOp,
        src_dims: &[usize],
        src_strides: &[isize],
        dest_dims: &[usize],
        dest_strides: &[isize],
        axis: usize,
    ) -> Result<Self> {
        check_arg_dtype(dtype, op)?;
        if !matches!(index_dtype, KernelDType::I32 | KernelDType::I64) {
            return Err(StridedError::UnsupportedDType {
                dtype: index_dtype.label(),
            });
        }
        if src_dims.len() != src_strides.len() || dest_dims.len() != dest_strides.len() {
            return Err(StridedError::StrideLengthMismatch);
        }
        let rank = src_dims.len();
        if axis >= rank {
            return Err(StridedError::InvalidAxis { axis, rank });
        }
        let expected: Vec<usize> = (0..rank)
            .filter(|&a| a != axis)
            .map(|a| src_dims[a])
            .collect();
        if dest_dims != expected.as_slice() {
            return Err(StridedError::ShapeMismatch(dest_dims.to_vec(), expected));
        }
        if src_dims[axis] == 0 {
            return Err(StridedError::UnsupportedOp {
                op: "argmax/argmin over a zero-length axis",
                dtype: dtype.label(),
            });
        }
        if index_dtype == KernelDType::I32 && i32::try_from(src_dims[axis] - 1).is_err() {
            return Err(StridedError::OffsetOverflow);
        }
        checked_total_len(src_dims)?;
        check_reduce_layout_offset_arithmetic(src_dims, src_strides)?;
        check_reduce_layout_offset_arithmetic(dest_dims, dest_strides)?;
        if !crate::layout_check::is_injective_layout(dest_dims, dest_strides) {
            return Err(StridedError::NonInjectiveOutputLayout);
        }
        let layout = LineLayout::compile(src_dims, src_strides, dest_strides, 0, axis, false)?;
        Ok(Self {
            dtype,
            index_dtype,
            op,
            src_dims: src_dims.to_vec(),
            src_strides: src_strides.to_vec(),
            dest_dims: dest_dims.to_vec(),
            dest_strides: dest_strides.to_vec(),
            layout,
        })
    }

    /// Source element dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::{ArgReduceOp, ErasedArgReducePlan, KernelDType};
    /// let plan = ErasedArgReducePlan::compile(
    ///     KernelDType::F32, KernelDType::I64, ArgReduceOp::Min, &[2], &[1], &[], &[], 0,
    /// )
    /// .unwrap();
    /// assert_eq!(plan.dtype(), KernelDType::F32);
    /// ```
    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    /// Destination index dtype (`i32` or `i64`).
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::{ArgReduceOp, ErasedArgReducePlan, KernelDType};
    /// let plan = ErasedArgReducePlan::compile(
    ///     KernelDType::F32, KernelDType::I64, ArgReduceOp::Min, &[2], &[1], &[], &[], 0,
    /// )
    /// .unwrap();
    /// assert_eq!(plan.index_dtype(), KernelDType::I64);
    /// ```
    #[inline]
    pub fn index_dtype(&self) -> KernelDType {
        self.index_dtype
    }

    /// Arg-reduction operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::{ArgReduceOp, ErasedArgReducePlan, KernelDType};
    /// let plan = ErasedArgReducePlan::compile(
    ///     KernelDType::F32, KernelDType::I64, ArgReduceOp::Min, &[2], &[1], &[], &[], 0,
    /// )
    /// .unwrap();
    /// assert_eq!(plan.op(), ArgReduceOp::Min);
    /// ```
    #[inline]
    pub fn op(&self) -> ArgReduceOp {
        self.op
    }

    fn check_layouts(
        &self,
        dest_dims: &[usize],
        dest_strides: &[isize],
        src: &ErasedRawStridedRef<'_>,
    ) -> Result<()> {
        if src.dims() != self.src_dims.as_slice()
            || src.strides() != self.src_strides.as_slice()
            || dest_dims != self.dest_dims.as_slice()
            || dest_strides != self.dest_strides.as_slice()
        {
            return Err(StridedError::PlanLayoutMismatch);
        }
        Ok(())
    }

    /// Execute into an initialized index destination.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::{
    ///     ArgReduceOp, ErasedArgReducePlan, ErasedRawStridedMut, ErasedRawStridedRef,
    ///     ExecContext, KernelDType,
    /// };
    /// let src = [3_i32, -9, 9, 1];
    /// let mut out = [0_i32; 1];
    /// let plan = ErasedArgReducePlan::compile(
    ///     KernelDType::I32, KernelDType::I32, ArgReduceOp::MaxAbs, &[4], &[1], &[], &[], 0,
    /// )
    /// .unwrap();
    /// let src_ref = ErasedRawStridedRef::from_slice(&src, &[4], &[1], 0).unwrap();
    /// let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &[], &[], 0).unwrap();
    /// plan.execute(&ExecContext::serial(), &mut dest, &src_ref).unwrap();
    /// assert_eq!(out, [1]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `DTypeMismatch` or `PlanLayoutMismatch` when a descriptor does
    /// not match the plan, before any destination write.
    pub fn execute(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedMut<'_>,
        src: &ErasedRawStridedRef<'_>,
    ) -> Result<()> {
        check_dtype(self.index_dtype, dest.dtype())?;
        check_dtype(self.dtype, src.dtype())?;
        self.check_layouts(dest.dims(), dest.strides(), src)?;
        match self.index_dtype {
            KernelDType::I32 => {
                let mut writer = reduce_writer::<i32>(dest)?;
                self.dispatch_dtype(ctx, &mut writer, src)
            }
            _ => {
                let mut writer = reduce_writer::<i64>(dest)?;
                self.dispatch_dtype(ctx, &mut writer, src)
            }
        }
    }

    /// Execute into an uninitialized index destination.
    ///
    /// On success every reachable destination element is written; validation
    /// errors are returned before any write.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::mem::MaybeUninit;
    /// use num_complex::Complex64;
    /// use strided_basic::{
    ///     ArgReduceOp, ErasedArgReducePlan, ErasedRawStridedPtr, ErasedRawStridedRef,
    ///     ErasedRawStridedUninitMut, ExecContext, KernelDType,
    /// };
    /// let src = [Complex64::new(3.0, 4.0), Complex64::new(0.0, 6.0)];
    /// let mut out = [MaybeUninit::<i64>::uninit()];
    /// let plan = ErasedArgReducePlan::compile(
    ///     KernelDType::C64, KernelDType::I64, ArgReduceOp::MinAbs, &[2], &[1], &[], &[], 0,
    /// )
    /// .unwrap();
    /// let src_ref = ErasedRawStridedRef::from_slice(&src, &[2], &[1], 0).unwrap();
    /// let src_ptr = ErasedRawStridedPtr::from_ref(&src_ref);
    /// let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(&mut out, &[], &[], 0).unwrap();
    /// plan.execute_uninit(&ExecContext::serial(), &mut dest, &src_ptr).unwrap();
    /// assert_eq!(unsafe { out[0].assume_init() }, 0);
    /// ```
    ///
    /// # Errors
    ///
    /// As [`Self::execute`], plus `OverlappingInputOutput` when the source
    /// overlaps the destination allocation.
    pub fn execute_uninit(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedUninitMut<'_>,
        src: &ErasedRawStridedPtr<'_>,
    ) -> Result<()> {
        check_dtype(self.index_dtype, dest.dtype())?;
        check_dtype(self.dtype, src.dtype())?;
        validate_uninit_no_overlap(dest, src, 0)?;
        // SAFETY: the owning erased entry rejected all input/output overlap before conversion.
        let src = unsafe { src.try_as_ref_after_no_overlap() }?;
        self.check_layouts(dest.dims(), dest.strides(), &src)?;
        match self.index_dtype {
            KernelDType::I32 => {
                let mut writer = reduce_uninit_writer::<i32>(dest)?;
                self.dispatch_dtype(ctx, &mut writer, &src)
            }
            _ => {
                let mut writer = reduce_uninit_writer::<i64>(dest)?;
                self.dispatch_dtype(ctx, &mut writer, &src)
            }
        }
    }

    fn dispatch_dtype<I, W>(
        &self,
        ctx: &ExecContext,
        dest: &mut W,
        src: &ErasedRawStridedRef<'_>,
    ) -> Result<()>
    where
        I: ArgIndex,
        W: ReduceWriter<I>,
    {
        // The dtype and op are matched once per execution; each loop is
        // monomorphized for one (dtype, key, direction, index) combination.
        macro_rules! real {
            ($ty:ty) => {
                match self.op {
                    ArgReduceOp::Max => self.run::<$ty, I, W, Plain, Greater>(ctx, dest, src),
                    ArgReduceOp::Min => self.run::<$ty, I, W, Plain, Less>(ctx, dest, src),
                    ArgReduceOp::MaxAbs => {
                        self.run::<$ty, I, W, Magnitude, Greater>(ctx, dest, src)
                    }
                    ArgReduceOp::MinAbs => self.run::<$ty, I, W, Magnitude, Less>(ctx, dest, src),
                }
            };
        }
        macro_rules! complex {
            ($ty:ty) => {
                match self.op {
                    ArgReduceOp::MaxAbs => {
                        self.run::<$ty, I, W, Magnitude, Greater>(ctx, dest, src)
                    }
                    ArgReduceOp::MinAbs => self.run::<$ty, I, W, Magnitude, Less>(ctx, dest, src),
                    // INVARIANT: check_arg_dtype rejects ordered complex ops at compile.
                    ArgReduceOp::Max | ArgReduceOp::Min => Err(StridedError::UnsupportedOp {
                        op: self.op.label(),
                        dtype: self.dtype.label(),
                    }),
                }
            };
        }
        match self.dtype {
            KernelDType::F32 => real!(f32),
            KernelDType::F64 => real!(f64),
            KernelDType::I32 => real!(i32),
            KernelDType::I64 => real!(i64),
            KernelDType::C32 => complex!(Complex32),
            KernelDType::C64 => complex!(Complex64),
            _ => Err(StridedError::UnsupportedDType {
                dtype: self.dtype.label(),
            }),
        }
    }

    fn run<T, I, W, F, D>(
        &self,
        ctx: &ExecContext,
        dest: &mut W,
        src: &ErasedRawStridedRef<'_>,
    ) -> Result<()>
    where
        T: KernelStorageElement + MaybeSendSync,
        I: ArgIndex,
        W: ReduceWriter<I>,
        F: KeyFn<T>,
        D: Direction,
    {
        let layout = &self.layout;
        let source = UnitPtr(src.data_as::<T>()?.as_ptr() as *mut T);
        // SAFETY: the validated writer owns the destination allocation.
        let target = UnitPtr(unsafe { dest.ptr() });
        let n = layout.axis_len;
        let ss = layout.src_axis_stride;
        let lane = layout.dest_lane_stride;
        let kernel = ArgUnit::<T, I, F, D> {
            source,
            target,
            n,
            ss,
            lane,
            _ops: core::marker::PhantomData,
        };
        // INVARIANT: (1) compile checked the signed source/destination spans
        // and every cursor step/reset, and rejected an empty axis; (2) the raw
        // descriptors validated every reachable offset; (3) execute checked
        // exact plan-layout equality. Units own disjoint outputs.
        // SAFETY: the three-link invariant above.
        unsafe { for_each_unit(ctx, layout, src.offset(), dest.offset(), kernel) }
    }
}

/// Per-unit arg-reduction kernel of one execution.
struct ArgUnit<T, I, F, D> {
    source: UnitPtr<T>,
    target: UnitPtr<I>,
    n: usize,
    ss: isize,
    lane: isize,
    _ops: core::marker::PhantomData<fn() -> (F, D)>,
}

impl<T, I, F, D> Clone for ArgUnit<T, I, F, D> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, I, F, D> Copy for ArgUnit<T, I, F, D> {}

impl<T, I, F, D> UnitKernel for ArgUnit<T, I, F, D>
where
    T: KernelStorageElement + MaybeSendSync,
    I: ArgIndex,
    F: KeyFn<T>,
    D: Direction,
{
    #[inline(always)]
    unsafe fn unit(self, so: isize, d_o: isize, width: usize) {
        let Self {
            source,
            target,
            n,
            ss,
            lane,
            ..
        } = self;
        // SAFETY: the caller passes offsets of the validated layout.
        unsafe {
            if width == 1 {
                let index = arg_line::<T, F, D>(source.get(), so, ss, n);
                target.get().offset(d_o).write(I::from_index(index));
            } else {
                arg_panel::<T, I, F, D>(source.get(), so, ss, n, target.get(), d_o, lane, width);
            }
        }
    }
}

fn check_arg_dtype(dtype: KernelDType, op: ArgReduceOp) -> Result<()> {
    match dtype {
        KernelDType::F32 | KernelDType::F64 | KernelDType::I32 | KernelDType::I64 => Ok(()),
        KernelDType::C32 | KernelDType::C64 => match op {
            ArgReduceOp::MaxAbs | ArgReduceOp::MinAbs => Ok(()),
            ArgReduceOp::Max | ArgReduceOp::Min => Err(StridedError::UnsupportedOp {
                op: op.label(),
                dtype: dtype.label(),
            }),
        },
        _ => Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        }),
    }
}

/// Index element written to the destination.
pub(super) trait ArgIndex: KernelStorageElement + MaybeSendSync {
    fn from_index(index: usize) -> Self;
}

impl ArgIndex for i32 {
    #[inline(always)]
    fn from_index(index: usize) -> Self {
        // INVARIANT: compile rejected axes whose last index exceeds i32::MAX.
        index as i32
    }
}

impl ArgIndex for i64 {
    #[inline(always)]
    fn from_index(index: usize) -> Self {
        // INVARIANT: an addressable axis length fits in i64.
        index as i64
    }
}

/// Comparable key of an element.
pub(super) trait OrderKey: Copy + PartialEq {
    /// Whether the key is NaN (never for integer keys).
    fn key_is_nan(self) -> bool;
    /// Whether `candidate` replaces `best` for a maximum: strictly greater,
    /// or the first NaN.
    fn beats_max(candidate: Self, best: Self) -> bool;
    /// Whether `candidate` replaces `best` for a minimum: strictly less, or
    /// the first NaN.
    fn beats_min(candidate: Self, best: Self) -> bool;
}

macro_rules! impl_float_key {
    ($($ty:ty),*) => {$(
        impl OrderKey for $ty {
            #[inline(always)]
            fn key_is_nan(self) -> bool {
                self.is_nan()
            }
            #[inline(always)]
            fn beats_max(candidate: Self, best: Self) -> bool {
                // Select form without a data dependent branch; a NaN `best`
                // is never replaced, so the first NaN wins.
                (candidate > best) | (candidate.is_nan() & !best.is_nan())
            }
            #[inline(always)]
            fn beats_min(candidate: Self, best: Self) -> bool {
                (candidate < best) | (candidate.is_nan() & !best.is_nan())
            }
        }
    )*};
}
impl_float_key!(f32, f64);

macro_rules! impl_int_key {
    ($($ty:ty),*) => {$(
        impl OrderKey for $ty {
            #[inline(always)]
            fn key_is_nan(self) -> bool {
                false
            }
            #[inline(always)]
            fn beats_max(candidate: Self, best: Self) -> bool {
                candidate > best
            }
            #[inline(always)]
            fn beats_min(candidate: Self, best: Self) -> bool {
                candidate < best
            }
        }
    )*};
}
impl_int_key!(i32, i64, u32, u64);

/// Maps an element to its comparison key.
pub(super) trait KeyFn<T> {
    type Key: OrderKey;
    fn key(value: T) -> Self::Key;
}

/// The value itself (real dtypes).
pub(super) struct Plain;
/// The magnitude `|x|`.
pub(super) struct Magnitude;

macro_rules! impl_plain_key {
    ($($ty:ty),*) => {$(
        impl KeyFn<$ty> for Plain {
            type Key = $ty;
            #[inline(always)]
            fn key(value: $ty) -> $ty {
                value
            }
        }
    )*};
}
impl_plain_key!(f32, f64, i32, i64);

impl KeyFn<f32> for Magnitude {
    type Key = f32;
    #[inline(always)]
    fn key(value: f32) -> f32 {
        value.abs()
    }
}
impl KeyFn<f64> for Magnitude {
    type Key = f64;
    #[inline(always)]
    fn key(value: f64) -> f64 {
        value.abs()
    }
}
impl KeyFn<i32> for Magnitude {
    type Key = u32;
    #[inline(always)]
    fn key(value: i32) -> u32 {
        value.unsigned_abs()
    }
}
impl KeyFn<i64> for Magnitude {
    type Key = u64;
    #[inline(always)]
    fn key(value: i64) -> u64 {
        value.unsigned_abs()
    }
}
impl KeyFn<Complex32> for Magnitude {
    type Key = f32;
    #[inline(always)]
    fn key(value: Complex32) -> f32 {
        // `hypot` returns +inf for an infinite component even when the other
        // is NaN; the NaN test keeps any NaN component a NaN key.
        if value.re.is_nan() || value.im.is_nan() {
            f32::NAN
        } else {
            value.re.hypot(value.im)
        }
    }
}
impl KeyFn<Complex64> for Magnitude {
    type Key = f64;
    #[inline(always)]
    fn key(value: Complex64) -> f64 {
        if value.re.is_nan() || value.im.is_nan() {
            f64::NAN
        } else {
            value.re.hypot(value.im)
        }
    }
}

/// Maximum or minimum, fixed at compile time.
pub(super) trait Direction {
    fn beats<K: OrderKey>(candidate: K, best: K) -> bool;
}
pub(super) struct Greater;
pub(super) struct Less;
impl Direction for Greater {
    #[inline(always)]
    fn beats<K: OrderKey>(candidate: K, best: K) -> bool {
        K::beats_max(candidate, best)
    }
}
impl Direction for Less {
    #[inline(always)]
    fn beats<K: OrderKey>(candidate: K, best: K) -> bool {
        K::beats_min(candidate, best)
    }
}

/// Independent lanes of the contiguous winner search.
const ARG_LANES: usize = 8;

/// Index of the winning element of one line.
///
/// A unit-stride line runs two passes: a lane-parallel search for the winning
/// key (NaN if the line has one), then a scan for the first element whose key
/// equals it (or is NaN). Equal keys tie, so the second pass returns the
/// lowest index of the winner, exactly as the sequential scan used for
/// strided lines.
///
/// # Safety
///
/// Every `src.offset(so + k * ss)` for `k < n` must be readable, and `n > 0`.
#[inline(always)]
unsafe fn arg_line<T, F, D>(src: *const T, so: isize, ss: isize, n: usize) -> usize
where
    T: Copy,
    F: KeyFn<T>,
    D: Direction,
{
    // SAFETY: the caller guarantees every visited offset and `n > 0`; a unit
    // stride line is `n` contiguous elements.
    unsafe {
        if ss == 1 {
            let values = core::slice::from_raw_parts(src.offset(so), n);
            let mut lanes = [F::key(values[0]); ARG_LANES];
            let mut chunks = values.chunks_exact(ARG_LANES);
            for chunk in &mut chunks {
                for (lane, &value) in lanes.iter_mut().zip(chunk) {
                    let key = F::key(value);
                    *lane = if D::beats(key, *lane) { key } else { *lane };
                }
            }
            // Which NaN or which signed zero wins here does not matter: the
            // second pass matches by NaN-ness or by equality.
            let mut winner = lanes[0];
            let tail = chunks.remainder().iter().map(|&value| F::key(value));
            for key in lanes[1..].iter().copied().chain(tail) {
                if D::beats(key, winner) {
                    winner = key;
                }
            }
            let found = if winner.key_is_nan() {
                values.iter().position(|&value| F::key(value).key_is_nan())
            } else {
                values.iter().position(|&value| F::key(value) == winner)
            };
            // INVARIANT: the winner is the key of some element of the line.
            return found.unwrap_or(0);
        }
        let mut best = F::key(src.offset(so).read());
        let mut best_index = 0;
        let mut offset = so;
        for index in 1..n {
            offset += ss;
            let key = F::key(src.offset(offset).read());
            if D::beats(key, best) {
                best = key;
                best_index = index;
            }
        }
        best_index
    }
}

/// Arg-reduces `width` adjacent lines whose source elements are contiguous
/// across the lines; output `j` is written at `d_o + j * lane`.
///
/// # Safety
///
/// As [`arg_line`] for each line `j < width` with source base `so + j`, every
/// destination offset must be writable, and `width <= PANEL`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn arg_panel<T, I, F, D>(
    src: *const T,
    so: isize,
    ss: isize,
    n: usize,
    dst: *mut I,
    d_o: isize,
    lane: isize,
    width: usize,
) where
    T: Copy,
    I: ArgIndex,
    F: KeyFn<T>,
    D: Direction,
{
    debug_assert!(width <= PANEL);
    // SAFETY: the caller guarantees `width` contiguous source elements at
    // every axis position and the destination offsets.
    unsafe {
        let first = core::slice::from_raw_parts(src.offset(so), width);
        let mut best = [F::key(first[0]); PANEL];
        let mut best_index = [0usize; PANEL];
        for (best, &value) in best.iter_mut().zip(first) {
            *best = F::key(value);
        }
        let best = &mut best[..width];
        let best_index = &mut best_index[..width];
        let mut offset = so;
        for index in 1..n {
            offset += ss;
            let row = core::slice::from_raw_parts(src.offset(offset), width);
            for ((best, best_index), &value) in best.iter_mut().zip(best_index.iter_mut()).zip(row)
            {
                let key = F::key(value);
                let take = D::beats(key, *best);
                *best = if take { key } else { *best };
                *best_index = if take { index } else { *best_index };
            }
        }
        for (lane_index, &index) in best_index.iter().enumerate() {
            dst.offset(d_o + lane_index as isize * lane)
                .write(I::from_index(index));
        }
    }
}
