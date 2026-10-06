//! Dtype-erased cumulative scans along one axis.

use super::line::{for_each_unit, LineLayout, UnitPtr, PANEL};
use super::{
    check_reduce_layout_offset_arithmetic, checked_total_len, reduce_uninit_writer, reduce_writer,
    ReduceWriter,
};
use crate::erased_common::{check_dtype, validate_uninit_no_overlap};
use crate::*;
use num_complex::{Complex32, Complex64};

/// Runtime operation of an [`ErasedScanPlan`].
///
/// # Examples
///
/// ```
/// use strided_basic::ScanOp;
/// assert_ne!(ScanOp::Sum, ScanOp::Product);
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScanOp {
    /// Cumulative sum (`cumsum`). Integers wrap on overflow.
    Sum,
    /// Cumulative product (`cumprod`). Integers wrap on overflow.
    Product,
}

/// Direction and inclusivity of an [`ErasedScanPlan`].
///
/// The default is an inclusive forward scan: output `k` combines inputs
/// `0..=k`. `exclusive` combines inputs `0..k` instead, so the first output is
/// the identity (`0` for sums, `1` for products). `reverse` scans from the end
/// of the axis: output `k` combines inputs `k..n` (inclusive) or `k+1..n`
/// (exclusive).
///
/// # Examples
///
/// ```
/// use strided_basic::ScanOptions;
/// let options = ScanOptions::new().exclusive(true).reverse(true);
/// assert!(options.is_exclusive() && options.is_reverse());
/// assert_eq!(ScanOptions::default(), ScanOptions::new());
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ScanOptions {
    exclusive: bool,
    reverse: bool,
}

impl ScanOptions {
    /// An inclusive forward scan.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::ScanOptions;
    /// assert!(!ScanOptions::new().is_exclusive());
    /// ```
    #[inline]
    pub const fn new() -> Self {
        Self {
            exclusive: false,
            reverse: false,
        }
    }

    /// Select an exclusive (`true`) or inclusive (`false`) scan.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::ScanOptions;
    /// assert!(ScanOptions::new().exclusive(true).is_exclusive());
    /// ```
    #[inline]
    pub const fn exclusive(mut self, exclusive: bool) -> Self {
        self.exclusive = exclusive;
        self
    }

    /// Select a reverse (`true`) or forward (`false`) scan.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::ScanOptions;
    /// assert!(ScanOptions::new().reverse(true).is_reverse());
    /// ```
    #[inline]
    pub const fn reverse(mut self, reverse: bool) -> Self {
        self.reverse = reverse;
        self
    }

    /// Whether the scan is exclusive.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::ScanOptions;
    /// assert!(!ScanOptions::default().is_exclusive());
    /// ```
    #[inline]
    pub const fn is_exclusive(&self) -> bool {
        self.exclusive
    }

    /// Whether the scan runs from the end of the axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::ScanOptions;
    /// assert!(!ScanOptions::default().is_reverse());
    /// ```
    #[inline]
    pub const fn is_reverse(&self) -> bool {
        self.reverse
    }
}

/// Dtype-erased cumulative scan (`cumsum` / `cumprod`) along one axis.
///
/// The destination has the source dimensions; source and destination strides
/// are independent, and negative strides and offsets are accepted. Supported
/// dtypes are `f32`, `f64`, `i32`, `i64`, `c32` and `c64`; `bool` is rejected.
///
/// # Evaluation order
///
/// Every output is the sequential left fold of its inputs in scan order (from
/// the start of the axis, or from its end for a reverse scan), with no
/// reassociation. The result is therefore bitwise identical for every layout,
/// execution context and thread count. Integer scans wrap on overflow.
///
/// An empty axis or an empty set of lines writes nothing.
///
/// # Examples
///
/// ```
/// use strided_basic::{
///     ErasedRawStridedMut, ErasedRawStridedRef, ErasedScanPlan, ExecContext, KernelDType,
///     ScanOp, ScanOptions,
/// };
///
/// // A column-major 3 x 2 matrix, scanned along axis 0.
/// let src = [1.0_f64, 2.0, 3.0, 10.0, 20.0, 30.0];
/// let mut out = [0.0_f64; 6];
/// let plan = ErasedScanPlan::compile(
///     KernelDType::F64,
///     ScanOp::Sum,
///     &[3, 2],
///     &[1, 3],
///     &[1, 3],
///     0,
///     ScanOptions::new(),
/// )
/// .unwrap();
/// let src_ref = ErasedRawStridedRef::from_slice(&src, &[3, 2], &[1, 3], 0).unwrap();
/// let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &[3, 2], &[1, 3], 0).unwrap();
/// plan.execute(&ExecContext::serial(), &mut dest, &src_ref).unwrap();
/// assert_eq!(out, [1.0, 3.0, 6.0, 10.0, 30.0, 60.0]);
/// ```
#[derive(Clone, Debug)]
pub struct ErasedScanPlan {
    dtype: KernelDType,
    op: ScanOp,
    options: ScanOptions,
    dims: Vec<usize>,
    src_strides: Vec<isize>,
    dest_strides: Vec<isize>,
    layout: LineLayout,
}

impl ErasedScanPlan {
    /// Validate and store a scan plan for one dtype and fixed layouts.
    ///
    /// `dims` are the shared source and destination dimensions; `axis` is the
    /// scanned axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::{ErasedScanPlan, KernelDType, ScanOp, ScanOptions};
    /// let plan = ErasedScanPlan::compile(
    ///     KernelDType::I64, ScanOp::Product, &[4], &[1], &[-1], 0, ScanOptions::new(),
    /// )
    /// .unwrap();
    /// assert_eq!(plan.op(), ScanOp::Product);
    /// assert!(ErasedScanPlan::compile(
    ///     KernelDType::Bool, ScanOp::Sum, &[4], &[1], &[1], 0, ScanOptions::new(),
    /// )
    /// .is_err());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `UnsupportedDType` for `bool`, `InvalidAxis` for an axis out of
    /// range, `StrideLengthMismatch` for inconsistent ranks,
    /// `NonInjectiveOutputLayout` for an aliasing destination layout, and
    /// `OffsetOverflow` when a layout's offsets are not representable.
    pub fn compile(
        dtype: KernelDType,
        op: ScanOp,
        dims: &[usize],
        src_strides: &[isize],
        dest_strides: &[isize],
        axis: usize,
        options: ScanOptions,
    ) -> Result<Self> {
        check_scan_dtype(dtype)?;
        if dims.len() != src_strides.len() || dims.len() != dest_strides.len() {
            return Err(StridedError::StrideLengthMismatch);
        }
        if axis >= dims.len() {
            return Err(StridedError::InvalidAxis {
                axis,
                rank: dims.len(),
            });
        }
        checked_total_len(dims)?;
        check_reduce_layout_offset_arithmetic(dims, src_strides)?;
        check_reduce_layout_offset_arithmetic(dims, dest_strides)?;
        if !crate::layout_check::is_injective_layout(dims, dest_strides) {
            return Err(StridedError::NonInjectiveOutputLayout);
        }
        let dest_outer: Vec<isize> = (0..dims.len())
            .filter(|&a| a != axis)
            .map(|a| dest_strides[a])
            .collect();
        let layout = LineLayout::compile(
            dims,
            src_strides,
            &dest_outer,
            dest_strides[axis],
            axis,
            true,
        )?;
        Ok(Self {
            dtype,
            op,
            options,
            dims: dims.to_vec(),
            src_strides: src_strides.to_vec(),
            dest_strides: dest_strides.to_vec(),
            layout,
        })
    }

    /// Element dtype of the source and destination.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::{ErasedScanPlan, KernelDType, ScanOp, ScanOptions};
    /// let plan = ErasedScanPlan::compile(
    ///     KernelDType::F32, ScanOp::Sum, &[2], &[1], &[1], 0, ScanOptions::new(),
    /// )
    /// .unwrap();
    /// assert_eq!(plan.dtype(), KernelDType::F32);
    /// ```
    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    /// Scan operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::{ErasedScanPlan, KernelDType, ScanOp, ScanOptions};
    /// let plan = ErasedScanPlan::compile(
    ///     KernelDType::F32, ScanOp::Sum, &[2], &[1], &[1], 0, ScanOptions::new(),
    /// )
    /// .unwrap();
    /// assert_eq!(plan.op(), ScanOp::Sum);
    /// ```
    #[inline]
    pub fn op(&self) -> ScanOp {
        self.op
    }

    /// Direction and inclusivity.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::{ErasedScanPlan, KernelDType, ScanOp, ScanOptions};
    /// let options = ScanOptions::new().reverse(true);
    /// let plan =
    ///     ErasedScanPlan::compile(KernelDType::F32, ScanOp::Sum, &[2], &[1], &[1], 0, options)
    ///         .unwrap();
    /// assert_eq!(plan.options(), options);
    /// ```
    #[inline]
    pub fn options(&self) -> ScanOptions {
        self.options
    }

    fn check_layouts(
        &self,
        dest_dims: &[usize],
        dest_strides: &[isize],
        src: &ErasedRawStridedRef<'_>,
    ) -> Result<()> {
        if src.dims() != self.dims.as_slice()
            || src.strides() != self.src_strides.as_slice()
            || dest_dims != self.dims.as_slice()
            || dest_strides != self.dest_strides.as_slice()
        {
            return Err(StridedError::PlanLayoutMismatch);
        }
        Ok(())
    }

    /// Execute the scan into an initialized destination.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::{
    ///     ErasedRawStridedMut, ErasedRawStridedRef, ErasedScanPlan, ExecContext, KernelDType,
    ///     ScanOp, ScanOptions,
    /// };
    /// let src = [1_i32, 2, 3, 4];
    /// let mut out = [0_i32; 4];
    /// let options = ScanOptions::new().exclusive(true).reverse(true);
    /// let plan =
    ///     ErasedScanPlan::compile(KernelDType::I32, ScanOp::Sum, &[4], &[1], &[1], 0, options)
    ///         .unwrap();
    /// let src_ref = ErasedRawStridedRef::from_slice(&src, &[4], &[1], 0).unwrap();
    /// let mut dest = ErasedRawStridedMut::from_slice_mut(&mut out, &[4], &[1], 0).unwrap();
    /// plan.execute(&ExecContext::serial(), &mut dest, &src_ref).unwrap();
    /// assert_eq!(out, [9, 7, 4, 0]);
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
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, src.dtype())?;
        self.check_layouts(dest.dims(), dest.strides(), src)?;
        macro_rules! run {
            ($ty:ty) => {{
                let mut writer = reduce_writer::<$ty>(dest)?;
                self.dispatch::<$ty, _>(ctx, &mut writer, src)
            }};
        }
        match self.dtype {
            KernelDType::F32 => run!(f32),
            KernelDType::F64 => run!(f64),
            KernelDType::I32 => run!(i32),
            KernelDType::I64 => run!(i64),
            KernelDType::C32 => run!(Complex32),
            KernelDType::C64 => run!(Complex64),
            _ => Err(unsupported(self.dtype)),
        }
    }

    /// Execute the scan into an uninitialized destination.
    ///
    /// On success every reachable destination element is written; validation
    /// errors are returned before any write.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::mem::MaybeUninit;
    /// use strided_basic::{
    ///     ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut, ErasedScanPlan,
    ///     ExecContext, KernelDType, ScanOp, ScanOptions,
    /// };
    /// let src = [2.0_f32, 3.0, 4.0];
    /// let mut out = [MaybeUninit::<f32>::uninit(); 3];
    /// let plan = ErasedScanPlan::compile(
    ///     KernelDType::F32, ScanOp::Product, &[3], &[1], &[1], 0, ScanOptions::new(),
    /// )
    /// .unwrap();
    /// let src_ref = ErasedRawStridedRef::from_slice(&src, &[3], &[1], 0).unwrap();
    /// let src_ptr = ErasedRawStridedPtr::from_ref(&src_ref);
    /// let mut dest =
    ///     ErasedRawStridedUninitMut::from_uninit_slice(&mut out, &[3], &[1], 0).unwrap();
    /// plan.execute_uninit(&ExecContext::serial(), &mut dest, &src_ptr).unwrap();
    /// let out: Vec<f32> = out.iter().map(|v| unsafe { v.assume_init() }).collect();
    /// assert_eq!(out, [2.0, 6.0, 24.0]);
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
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, src.dtype())?;
        validate_uninit_no_overlap(dest, src, 0)?;
        // SAFETY: the owning erased entry rejected all input/output overlap before conversion.
        let src = unsafe { src.try_as_ref_after_no_overlap() }?;
        self.check_layouts(dest.dims(), dest.strides(), &src)?;
        macro_rules! run {
            ($ty:ty) => {{
                let mut writer = reduce_uninit_writer::<$ty>(dest)?;
                self.dispatch::<$ty, _>(ctx, &mut writer, &src)
            }};
        }
        match self.dtype {
            KernelDType::F32 => run!(f32),
            KernelDType::F64 => run!(f64),
            KernelDType::I32 => run!(i32),
            KernelDType::I64 => run!(i64),
            KernelDType::C32 => run!(Complex32),
            KernelDType::C64 => run!(Complex64),
            _ => Err(unsupported(self.dtype)),
        }
    }

    fn dispatch<T, W>(
        &self,
        ctx: &ExecContext,
        dest: &mut W,
        src: &ErasedRawStridedRef<'_>,
    ) -> Result<()>
    where
        T: ScanScalar,
        W: ReduceWriter<T>,
    {
        // The op and options are matched once per execution; the loops below
        // are monomorphized for each combination.
        match (self.op, self.options.exclusive, self.options.reverse) {
            (ScanOp::Sum, false, false) => self.run::<T, W, SumScan, false, false>(ctx, dest, src),
            (ScanOp::Sum, false, true) => self.run::<T, W, SumScan, false, true>(ctx, dest, src),
            (ScanOp::Sum, true, false) => self.run::<T, W, SumScan, true, false>(ctx, dest, src),
            (ScanOp::Sum, true, true) => self.run::<T, W, SumScan, true, true>(ctx, dest, src),
            (ScanOp::Product, false, false) => {
                self.run::<T, W, ProductScan, false, false>(ctx, dest, src)
            }
            (ScanOp::Product, false, true) => {
                self.run::<T, W, ProductScan, false, true>(ctx, dest, src)
            }
            (ScanOp::Product, true, false) => {
                self.run::<T, W, ProductScan, true, false>(ctx, dest, src)
            }
            (ScanOp::Product, true, true) => {
                self.run::<T, W, ProductScan, true, true>(ctx, dest, src)
            }
        }
    }

    fn run<T, W, K, const EXCLUSIVE: bool, const REVERSE: bool>(
        &self,
        ctx: &ExecContext,
        dest: &mut W,
        src: &ErasedRawStridedRef<'_>,
    ) -> Result<()>
    where
        T: ScanScalar,
        W: ReduceWriter<T>,
        K: ScanKernel<T>,
    {
        let layout = &self.layout;
        let source = UnitPtr(src.data_as::<T>()?.as_ptr() as *mut T);
        // SAFETY: the validated writer owns the destination allocation.
        let target = UnitPtr(unsafe { dest.ptr() });
        let n = layout.axis_len;
        let ss = layout.src_axis_stride;
        let ds = layout.dest_axis_stride;
        let unit = |so: isize, d_o: isize, width: usize| {
            // SAFETY: see the invariant at the `for_each_unit` call.
            unsafe {
                if width == 1 {
                    scan_line::<T, K, EXCLUSIVE, REVERSE>(
                        source.get(),
                        so,
                        ss,
                        target.get(),
                        d_o,
                        ds,
                        n,
                    )
                } else {
                    scan_panel::<T, K, EXCLUSIVE, REVERSE>(
                        source.get(),
                        so,
                        ss,
                        target.get(),
                        d_o,
                        ds,
                        n,
                        width,
                    )
                }
            }
        };
        // INVARIANT: (1) compile checked the signed source/destination spans
        // and every cursor step/reset; (2) the raw descriptors validated every
        // reachable offset; (3) execute checked exact plan-layout equality.
        // Units cover disjoint lines, so their destination writes are disjoint.
        // SAFETY: the three-link invariant above.
        unsafe { for_each_unit(ctx, layout, src.offset(), dest.offset(), &unit) }
    }
}

fn unsupported(dtype: KernelDType) -> StridedError {
    StridedError::UnsupportedDType {
        dtype: dtype.label(),
    }
}

fn check_scan_dtype(dtype: KernelDType) -> Result<()> {
    match dtype {
        KernelDType::F32
        | KernelDType::F64
        | KernelDType::I32
        | KernelDType::I64
        | KernelDType::C32
        | KernelDType::C64 => Ok(()),
        _ => Err(unsupported(dtype)),
    }
}

/// Element type supported by scans.
pub(super) trait ScanScalar: KernelStorageElement + MaybeSendSync {
    fn zero() -> Self;
    fn one() -> Self;
    fn scan_add(lhs: Self, rhs: Self) -> Self;
    fn scan_mul(lhs: Self, rhs: Self) -> Self;
}

macro_rules! impl_scan_scalar {
    ($add:ident, $mul:ident; $($ty:ty => $zero:expr, $one:expr),* $(,)?) => {$(
        impl ScanScalar for $ty {
            #[inline(always)]
            fn zero() -> Self { $zero }
            #[inline(always)]
            fn one() -> Self { $one }
            #[inline(always)]
            fn scan_add(lhs: Self, rhs: Self) -> Self { $add(lhs, rhs) }
            #[inline(always)]
            fn scan_mul(lhs: Self, rhs: Self) -> Self { $mul(lhs, rhs) }
        }
    )*};
}

#[inline(always)]
fn plain_add<T: core::ops::Add<Output = T>>(lhs: T, rhs: T) -> T {
    lhs + rhs
}
#[inline(always)]
fn plain_mul<T: core::ops::Mul<Output = T>>(lhs: T, rhs: T) -> T {
    lhs * rhs
}
trait WrappingArith {
    fn wadd(self, rhs: Self) -> Self;
    fn wmul(self, rhs: Self) -> Self;
}
impl WrappingArith for i32 {
    #[inline(always)]
    fn wadd(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
    #[inline(always)]
    fn wmul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}
impl WrappingArith for i64 {
    #[inline(always)]
    fn wadd(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
    #[inline(always)]
    fn wmul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}
#[inline(always)]
fn wrapping_add<T: WrappingArith>(lhs: T, rhs: T) -> T {
    lhs.wadd(rhs)
}
#[inline(always)]
fn wrapping_mul<T: WrappingArith>(lhs: T, rhs: T) -> T {
    lhs.wmul(rhs)
}

impl_scan_scalar!(plain_add, plain_mul;
    f32 => 0.0, 1.0,
    f64 => 0.0, 1.0,
    Complex32 => Complex32::new(0.0, 0.0), Complex32::new(1.0, 0.0),
    Complex64 => Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
);
impl_scan_scalar!(wrapping_add, wrapping_mul;
    i32 => 0, 1,
    i64 => 0, 1,
);

/// One scan operation, fixed at compile time.
pub(super) trait ScanKernel<T>: 'static {
    fn identity() -> T;
    fn combine(acc: T, value: T) -> T;
}

pub(super) struct SumScan;
pub(super) struct ProductScan;

impl<T: ScanScalar> ScanKernel<T> for SumScan {
    #[inline(always)]
    fn identity() -> T {
        T::zero()
    }
    #[inline(always)]
    fn combine(acc: T, value: T) -> T {
        T::scan_add(acc, value)
    }
}

impl<T: ScanScalar> ScanKernel<T> for ProductScan {
    #[inline(always)]
    fn identity() -> T {
        T::one()
    }
    #[inline(always)]
    fn combine(acc: T, value: T) -> T {
        T::scan_mul(acc, value)
    }
}

/// Start offset and signed step of a line visited in scan order.
#[inline(always)]
fn scan_order(base: isize, stride: isize, n: usize, reverse: bool) -> (isize, isize) {
    if reverse {
        // INVARIANT: compile checked (n - 1) * stride for this axis.
        (base + (n as isize - 1) * stride, -stride)
    } else {
        (base, stride)
    }
}

/// Scans one line.
///
/// # Safety
///
/// Every `src.offset(so + k * ss)` and `dst.offset(d_o + k * ds)` for
/// `k < n` must be valid, and `n > 0`.
#[inline(always)]
unsafe fn scan_line<T, K, const EXCLUSIVE: bool, const REVERSE: bool>(
    src: *const T,
    so: isize,
    ss: isize,
    dst: *mut T,
    d_o: isize,
    ds: isize,
    n: usize,
) where
    T: ScanScalar,
    K: ScanKernel<T>,
{
    let (mut s, ss) = scan_order(so, ss, n, REVERSE);
    let (mut d, ds) = scan_order(d_o, ds, n, REVERSE);
    let mut acc = K::identity();
    for _ in 0..n {
        // SAFETY: the caller guarantees every visited offset.
        unsafe {
            let value = src.offset(s).read();
            if EXCLUSIVE {
                dst.offset(d).write(acc);
                acc = K::combine(acc, value);
            } else {
                acc = K::combine(acc, value);
                dst.offset(d).write(acc);
            }
        }
        s += ss;
        d += ds;
    }
}

/// Scans `width` adjacent lines whose elements are contiguous across the
/// lines in both source and destination.
///
/// # Safety
///
/// As [`scan_line`] for each line `j < width`, with source base `so + j` and
/// destination base `d_o + j`; `width <= PANEL`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn scan_panel<T, K, const EXCLUSIVE: bool, const REVERSE: bool>(
    src: *const T,
    so: isize,
    ss: isize,
    dst: *mut T,
    d_o: isize,
    ds: isize,
    n: usize,
    width: usize,
) where
    T: ScanScalar,
    K: ScanKernel<T>,
{
    debug_assert!(width <= PANEL);
    let (mut s, ss) = scan_order(so, ss, n, REVERSE);
    let (mut d, ds) = scan_order(d_o, ds, n, REVERSE);
    let mut acc = [K::identity(); PANEL];
    let acc = &mut acc[..width];
    for _ in 0..n {
        // SAFETY: the caller guarantees `width` contiguous elements at both
        // offsets for every visited axis position.
        unsafe {
            let input = core::slice::from_raw_parts(src.offset(s), width);
            // Raw writes: the destination may be uninitialized.
            let output = dst.offset(d);
            for (lane, (acc, &value)) in acc.iter_mut().zip(input).enumerate() {
                if EXCLUSIVE {
                    output.add(lane).write(*acc);
                    *acc = K::combine(*acc, value);
                } else {
                    *acc = K::combine(*acc, value);
                    output.add(lane).write(*acc);
                }
            }
        }
        s += ss;
        d += ds;
    }
}
