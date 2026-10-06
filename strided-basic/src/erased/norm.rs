//! Dtype-erased fused `layer_norm` / `rms_norm` along one axis.

use super::line::{for_each_unit, LineLayout, UnitPtr, PANEL};
use super::{
    check_reduce_layout_offset_arithmetic, checked_total_len, reduce_uninit_writer, reduce_writer,
    ReduceWriter,
};
use crate::erased_common::{check_dtype, validate_uninit_no_overlap};
use crate::*;

/// Normalization computed by an [`ErasedNormPlan`].
///
/// # Examples
///
/// ```
/// use strided_basic::NormKind;
/// assert_ne!(NormKind::Layer, NormKind::Rms);
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormKind {
    /// `y = (x - mean) / sqrt(var + eps)` with the biased (population)
    /// variance `var = mean((x - mean)^2)`.
    Layer,
    /// `y = x / sqrt(mean(x^2) + eps)`.
    Rms,
}

/// Kind, `eps` and optional affine parameters of an [`ErasedNormPlan`].
///
/// The optional weight (scale) and bias (shift) are vectors along the
/// normalized axis, each with its own element stride; the result is
/// `y * weight + bias`.
///
/// # Examples
///
/// ```
/// use strided_basic::{NormKind, NormSpec};
/// let spec = NormSpec::layer_norm(1e-5).with_weight(1).with_bias(-1);
/// assert_eq!(spec.kind(), NormKind::Layer);
/// assert_eq!(spec.eps(), 1e-5);
/// assert_eq!(spec.weight_stride(), Some(1));
/// assert_eq!(spec.bias_stride(), Some(-1));
/// assert_eq!(NormSpec::rms_norm(0.0).weight_stride(), None);
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormSpec {
    kind: NormKind,
    eps: f64,
    weight_stride: Option<isize>,
    bias_stride: Option<isize>,
}

impl NormSpec {
    /// Layer normalization with the given `eps` and no affine parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::{NormKind, NormSpec};
    /// assert_eq!(NormSpec::layer_norm(1e-6).kind(), NormKind::Layer);
    /// ```
    #[inline]
    pub const fn layer_norm(eps: f64) -> Self {
        Self {
            kind: NormKind::Layer,
            eps,
            weight_stride: None,
            bias_stride: None,
        }
    }

    /// RMS normalization with the given `eps` and no affine parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::{NormKind, NormSpec};
    /// assert_eq!(NormSpec::rms_norm(1e-6).kind(), NormKind::Rms);
    /// ```
    #[inline]
    pub const fn rms_norm(eps: f64) -> Self {
        Self {
            kind: NormKind::Rms,
            eps,
            weight_stride: None,
            bias_stride: None,
        }
    }

    /// Multiply by a weight vector with the given element stride.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::NormSpec;
    /// assert_eq!(NormSpec::rms_norm(0.0).with_weight(2).weight_stride(), Some(2));
    /// ```
    #[inline]
    pub const fn with_weight(mut self, stride: isize) -> Self {
        self.weight_stride = Some(stride);
        self
    }

    /// Add a bias vector with the given element stride.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::NormSpec;
    /// assert_eq!(NormSpec::rms_norm(0.0).with_bias(1).bias_stride(), Some(1));
    /// ```
    #[inline]
    pub const fn with_bias(mut self, stride: isize) -> Self {
        self.bias_stride = Some(stride);
        self
    }

    /// Normalization kind.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::{NormKind, NormSpec};
    /// assert_eq!(NormSpec::rms_norm(0.0).kind(), NormKind::Rms);
    /// ```
    #[inline]
    pub const fn kind(&self) -> NormKind {
        self.kind
    }

    /// The `eps` added to the variance (or mean square) before the square root.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::NormSpec;
    /// assert_eq!(NormSpec::rms_norm(0.25).eps(), 0.25);
    /// ```
    #[inline]
    pub const fn eps(&self) -> f64 {
        self.eps
    }

    /// Element stride of the weight vector, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::NormSpec;
    /// assert_eq!(NormSpec::layer_norm(0.0).weight_stride(), None);
    /// ```
    #[inline]
    pub const fn weight_stride(&self) -> Option<isize> {
        self.weight_stride
    }

    /// Element stride of the bias vector, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::NormSpec;
    /// assert_eq!(NormSpec::layer_norm(0.0).bias_stride(), None);
    /// ```
    #[inline]
    pub const fn bias_stride(&self) -> Option<isize> {
        self.bias_stride
    }
}

/// Dtype-erased fused `layer_norm` / `rms_norm` along one axis.
///
/// Each line along `axis` is normalized independently in one call:
///
/// * [`NormKind::Layer`]: `y = (x - mean) * rsqrt(var + eps) * weight + bias`
///   with `mean = sum(x) / n` and the biased (population) variance
///   `var = sum((x - mean)^2) / n`, as in PyTorch's `layer_norm`.
/// * [`NormKind::Rms`]: `y = x * rsqrt(sum(x^2) / n + eps) * weight + bias`.
///
/// `rsqrt(v)` is evaluated as `1 / sqrt(v)` once per line, and the weight and
/// bias terms are present only when the [`NormSpec`] requests them. The
/// destination has the source dimensions with independent strides. Supported
/// dtypes are `f32` and `f64`; every accumulation is in the element dtype.
///
/// # Numerics
///
/// Layer norm uses the shifted-data two-pass algorithm: every element is first
/// shifted by the first element of its line, then the mean of the shifted
/// values and the variance about that mean are computed in two passes. This
/// avoids the cancellation of the one-pass `E[x^2] - E[x]^2` form and of a
/// large common offset. A line whose elements are all equal has a variance of
/// exactly zero and normalizes to exactly `0 * rsqrt(eps) * weight + bias`,
/// i.e. `bias` (or zero) whenever `eps > 0`. With `eps == 0` such a line
/// evaluates `0 * inf` and is `NaN`; that is the caller's choice of `eps`, not
/// an error.
///
/// Non-finite inputs follow IEEE evaluation of the formula. A NaN anywhere in
/// a line makes every output of that line NaN. An infinite element makes a
/// layer-norm line NaN (its deviation is `inf - inf`), and makes an RMS-norm
/// line zero at its finite elements and NaN at the infinite ones
/// (`rsqrt(inf) = 0`). Squares are not rescaled: when the sum of squared
/// deviations overflows to `inf` (finite deviations above about `1.8e19` for
/// `f32` or `1.3e154` for `f64`), the line's finite elements normalize to zero.
///
/// When the normalized axis has unit source stride, each line is summed with
/// eight independent partial sums. Otherwise lines that are adjacent in memory
/// are processed as a block and each line is summed sequentially. The rounding
/// can therefore differ between layouts, but the result never depends on the
/// execution context or thread count.
///
/// An empty axis or an empty set of lines writes nothing.
///
/// # Examples
///
/// ```
/// use strided_basic::{
///     ErasedNormPlan, ErasedRawStridedMut, ErasedRawStridedRef, ExecContext, KernelDType,
///     NormSpec,
/// };
///
/// // Two feature-first rows of width 2: (d = 2, len = 2), normalized over d.
/// let x = [1.0_f64, 3.0, -2.0, 2.0];
/// let weight = [2.0_f64, 2.0];
/// let mut y = [0.0_f64; 4];
/// let spec = NormSpec::layer_norm(0.0).with_weight(1);
/// let plan =
///     ErasedNormPlan::compile(KernelDType::F64, spec, &[2, 2], &[1, 2], &[1, 2], 0).unwrap();
/// let x = ErasedRawStridedRef::from_slice(&x, &[2, 2], &[1, 2], 0).unwrap();
/// let w = ErasedRawStridedRef::from_slice(&weight, &[2], &[1], 0).unwrap();
/// let mut out = ErasedRawStridedMut::from_slice_mut(&mut y, &[2, 2], &[1, 2], 0).unwrap();
/// plan.execute(&ExecContext::serial(), &mut out, &x, Some(&w), None).unwrap();
/// assert_eq!(y, [-2.0, 2.0, -2.0, 2.0]);
/// ```
#[derive(Clone, Debug)]
pub struct ErasedNormPlan {
    dtype: KernelDType,
    spec: NormSpec,
    dims: Vec<usize>,
    src_strides: Vec<isize>,
    dest_strides: Vec<isize>,
    layout: LineLayout,
}

impl ErasedNormPlan {
    /// Validate and store a normalization plan for fixed layouts.
    ///
    /// `dims` are the shared source and destination dimensions; `axis` is the
    /// normalized axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::{ErasedNormPlan, KernelDType, NormSpec};
    /// let spec = NormSpec::rms_norm(1e-6).with_weight(1);
    /// assert!(ErasedNormPlan::compile(KernelDType::F32, spec, &[8, 3], &[1, 8], &[1, 8], 0).is_ok());
    /// // Complex and integer input is rejected, as is a negative eps.
    /// assert!(ErasedNormPlan::compile(KernelDType::C64, spec, &[8], &[1], &[1], 0).is_err());
    /// let bad = NormSpec::rms_norm(-1.0);
    /// assert!(ErasedNormPlan::compile(KernelDType::F32, bad, &[8], &[1], &[1], 0).is_err());
    /// ```
    ///
    /// # Errors
    ///
    /// * `UnsupportedDType` for a dtype other than `f32` / `f64`;
    /// * `UnsupportedOp` for a negative, infinite or NaN `eps`;
    /// * `InvalidAxis`, `StrideLengthMismatch` and `NonInjectiveOutputLayout`
    ///   for inconsistent layouts;
    /// * `OffsetOverflow` when a layout's offsets are not representable.
    pub fn compile(
        dtype: KernelDType,
        spec: NormSpec,
        dims: &[usize],
        src_strides: &[isize],
        dest_strides: &[isize],
        axis: usize,
    ) -> Result<Self> {
        if !matches!(dtype, KernelDType::F32 | KernelDType::F64) {
            return Err(StridedError::UnsupportedDType {
                dtype: dtype.label(),
            });
        }
        if !(spec.eps.is_finite() && spec.eps >= 0.0) {
            return Err(StridedError::UnsupportedOp {
                op: "norm with a negative or non-finite eps",
                dtype: dtype.label(),
            });
        }
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
        for stride in [spec.weight_stride, spec.bias_stride].into_iter().flatten() {
            check_reduce_layout_offset_arithmetic(&dims[axis..=axis], &[stride])?;
        }
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
            spec,
            dims: dims.to_vec(),
            src_strides: src_strides.to_vec(),
            dest_strides: dest_strides.to_vec(),
            layout,
        })
    }

    /// Element dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::{ErasedNormPlan, KernelDType, NormSpec};
    /// let plan = ErasedNormPlan::compile(
    ///     KernelDType::F64, NormSpec::layer_norm(0.0), &[4], &[1], &[1], 0,
    /// )
    /// .unwrap();
    /// assert_eq!(plan.dtype(), KernelDType::F64);
    /// ```
    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    /// Kind, `eps` and affine parameter layout.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::{ErasedNormPlan, KernelDType, NormSpec};
    /// let spec = NormSpec::layer_norm(1e-5);
    /// let plan = ErasedNormPlan::compile(KernelDType::F64, spec, &[4], &[1], &[1], 0).unwrap();
    /// assert_eq!(plan.spec(), spec);
    /// ```
    #[inline]
    pub fn spec(&self) -> NormSpec {
        self.spec
    }

    fn check_layouts(
        &self,
        dest_dims: &[usize],
        dest_strides: &[isize],
        src: &ErasedRawStridedRef<'_>,
        weight: Option<&ErasedRawStridedRef<'_>>,
        bias: Option<&ErasedRawStridedRef<'_>>,
    ) -> Result<()> {
        if src.dims() != self.dims.as_slice()
            || src.strides() != self.src_strides.as_slice()
            || dest_dims != self.dims.as_slice()
            || dest_strides != self.dest_strides.as_slice()
        {
            return Err(StridedError::PlanLayoutMismatch);
        }
        let n = self.layout.axis_len;
        for (param, stride) in [
            (weight, self.spec.weight_stride),
            (bias, self.spec.bias_stride),
        ] {
            match (param, stride) {
                (None, None) => {}
                (Some(param), Some(stride)) => {
                    check_dtype(self.dtype, param.dtype())?;
                    if param.dims() != [n] || param.strides() != [stride] {
                        return Err(StridedError::PlanLayoutMismatch);
                    }
                }
                _ => return Err(StridedError::PlanLayoutMismatch),
            }
        }
        Ok(())
    }

    /// Execute into an initialized destination.
    ///
    /// `weight` and `bias` must be present exactly when the plan's
    /// [`NormSpec`] requests them, as rank-one descriptors of the axis length
    /// with the recorded stride.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_basic::{
    ///     ErasedNormPlan, ErasedRawStridedMut, ErasedRawStridedRef, ExecContext, KernelDType,
    ///     NormSpec,
    /// };
    /// let x = [3.0_f32, 4.0, 0.0, 0.0];
    /// let bias = [1.0_f32, 1.0];
    /// let mut y = [0.0_f32; 4];
    /// let spec = NormSpec::rms_norm(0.0).with_bias(1);
    /// let plan =
    ///     ErasedNormPlan::compile(KernelDType::F32, spec, &[2, 2], &[1, 2], &[1, 2], 0).unwrap();
    /// let x = ErasedRawStridedRef::from_slice(&x, &[2, 2], &[1, 2], 0).unwrap();
    /// let b = ErasedRawStridedRef::from_slice(&bias, &[2], &[1], 0).unwrap();
    /// let mut out = ErasedRawStridedMut::from_slice_mut(&mut y, &[2, 2], &[1, 2], 0).unwrap();
    /// plan.execute(&ExecContext::serial(), &mut out, &x, None, Some(&b)).unwrap();
    /// // Row 0: rms = sqrt(12.5); the all-zero row 1 is 0 * inf = NaN with eps = 0.
    /// assert!((y[0] - (1.0 + 3.0 / 12.5f32.sqrt())).abs() < 1e-6);
    /// assert!(y[2].is_nan() && y[3].is_nan());
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
        weight: Option<&ErasedRawStridedRef<'_>>,
        bias: Option<&ErasedRawStridedRef<'_>>,
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, src.dtype())?;
        self.check_layouts(dest.dims(), dest.strides(), src, weight, bias)?;
        match self.dtype {
            KernelDType::F32 => {
                let mut writer = reduce_writer::<f32>(dest)?;
                self.dispatch::<f32, _>(ctx, &mut writer, src, weight, bias)
            }
            _ => {
                let mut writer = reduce_writer::<f64>(dest)?;
                self.dispatch::<f64, _>(ctx, &mut writer, src, weight, bias)
            }
        }
    }

    /// Execute into an uninitialized destination.
    ///
    /// On success every reachable destination element is written; validation
    /// errors, including any overlap between an input and the destination
    /// allocation, are returned before any write.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::mem::MaybeUninit;
    /// use strided_basic::{
    ///     ErasedNormPlan, ErasedRawStridedPtr, ErasedRawStridedRef, ErasedRawStridedUninitMut,
    ///     ExecContext, KernelDType, NormSpec,
    /// };
    /// let x = [1.0_f64, 1.0, 1.0];
    /// let mut y = [MaybeUninit::<f64>::uninit(); 3];
    /// let plan = ErasedNormPlan::compile(
    ///     KernelDType::F64, NormSpec::layer_norm(1e-5), &[3], &[1], &[1], 0,
    /// )
    /// .unwrap();
    /// let x = ErasedRawStridedRef::from_slice(&x, &[3], &[1], 0).unwrap();
    /// let x = ErasedRawStridedPtr::from_ref(&x);
    /// let mut out = ErasedRawStridedUninitMut::from_uninit_slice(&mut y, &[3], &[1], 0).unwrap();
    /// plan.execute_uninit(&ExecContext::serial(), &mut out, &x, None, None).unwrap();
    /// // Zero variance with eps > 0 normalizes to exactly zero.
    /// assert!(y.iter().all(|v| unsafe { v.assume_init() } == 0.0));
    /// ```
    ///
    /// # Errors
    ///
    /// As [`Self::execute`], plus `OverlappingInputOutput` naming input `0`
    /// (source), `1` (weight) or `2` (bias).
    pub fn execute_uninit(
        &self,
        ctx: &ExecContext,
        dest: &mut ErasedRawStridedUninitMut<'_>,
        src: &ErasedRawStridedPtr<'_>,
        weight: Option<&ErasedRawStridedPtr<'_>>,
        bias: Option<&ErasedRawStridedPtr<'_>>,
    ) -> Result<()> {
        check_dtype(self.dtype, dest.dtype())?;
        check_dtype(self.dtype, src.dtype())?;
        validate_uninit_no_overlap(dest, src, 0)?;
        if let Some(weight) = weight {
            validate_uninit_no_overlap(dest, weight, 1)?;
        }
        if let Some(bias) = bias {
            validate_uninit_no_overlap(dest, bias, 2)?;
        }
        // SAFETY: the owning erased entry rejected all input/output overlap
        // before each conversion.
        let src = unsafe { src.try_as_ref_after_no_overlap() }?;
        let weight = weight
            .map(|weight| unsafe { weight.try_as_ref_after_no_overlap() })
            .transpose()?;
        let bias = bias
            .map(|bias| unsafe { bias.try_as_ref_after_no_overlap() })
            .transpose()?;
        self.check_layouts(
            dest.dims(),
            dest.strides(),
            &src,
            weight.as_ref(),
            bias.as_ref(),
        )?;
        match self.dtype {
            KernelDType::F32 => {
                let mut writer = reduce_uninit_writer::<f32>(dest)?;
                self.dispatch::<f32, _>(ctx, &mut writer, &src, weight.as_ref(), bias.as_ref())
            }
            _ => {
                let mut writer = reduce_uninit_writer::<f64>(dest)?;
                self.dispatch::<f64, _>(ctx, &mut writer, &src, weight.as_ref(), bias.as_ref())
            }
        }
    }

    fn dispatch<T, W>(
        &self,
        ctx: &ExecContext,
        dest: &mut W,
        src: &ErasedRawStridedRef<'_>,
        weight: Option<&ErasedRawStridedRef<'_>>,
        bias: Option<&ErasedRawStridedRef<'_>>,
    ) -> Result<()>
    where
        T: NormScalar,
        W: ReduceWriter<T>,
    {
        let affine = Affine {
            weight: param::<T>(weight)?,
            bias: param::<T>(bias)?,
        };
        // Kind and affine presence are matched once per execution; each loop
        // is monomorphized for one combination.
        macro_rules! go {
            ($layer:literal) => {
                match (weight.is_some(), bias.is_some()) {
                    (false, false) => {
                        self.run::<T, W, $layer, false, false>(ctx, dest, src, affine)
                    }
                    (true, false) => self.run::<T, W, $layer, true, false>(ctx, dest, src, affine),
                    (false, true) => self.run::<T, W, $layer, false, true>(ctx, dest, src, affine),
                    (true, true) => self.run::<T, W, $layer, true, true>(ctx, dest, src, affine),
                }
            };
        }
        match self.spec.kind {
            NormKind::Layer => go!(true),
            NormKind::Rms => go!(false),
        }
    }

    fn run<T, W, const LAYER: bool, const WEIGHT: bool, const BIAS: bool>(
        &self,
        ctx: &ExecContext,
        dest: &mut W,
        src: &ErasedRawStridedRef<'_>,
        affine: Affine<T>,
    ) -> Result<()>
    where
        T: NormScalar,
        W: ReduceWriter<T>,
    {
        let layout = &self.layout;
        let source = UnitPtr(src.data_as::<T>()?.as_ptr() as *mut T);
        // SAFETY: the validated writer owns the destination allocation.
        let target = UnitPtr(unsafe { dest.ptr() });
        let line = Line {
            n: layout.axis_len,
            n_t: T::from_usize(layout.axis_len),
            eps: T::from_f64(self.spec.eps),
            ss: layout.src_axis_stride,
            ds: layout.dest_axis_stride,
        };
        let unit = |so: isize, d_o: isize, width: usize| {
            // SAFETY: see the invariant at the `for_each_unit` call.
            unsafe {
                if width == 1 {
                    norm_line::<T, LAYER, WEIGHT, BIAS>(
                        source.get(),
                        so,
                        target.get(),
                        d_o,
                        line,
                        affine,
                    )
                } else {
                    norm_panel::<T, LAYER, WEIGHT, BIAS>(
                        source.get(),
                        so,
                        target.get(),
                        d_o,
                        width,
                        line,
                        affine,
                    )
                }
            }
        };
        // INVARIANT: (1) compile checked the signed source, destination,
        // weight and bias spans and every cursor step/reset; (2) the raw
        // descriptors validated every reachable offset; (3) execute checked
        // exact plan-layout equality for all four descriptors. Units cover
        // disjoint lines, so their destination writes are disjoint.
        // SAFETY: the three-link invariant above.
        unsafe { for_each_unit(ctx, layout, src.offset(), dest.offset(), &unit) }
    }
}

/// Base pointer, offset and stride of an optional affine vector.
#[derive(Clone, Copy)]
struct Param<T> {
    ptr: UnitPtr<T>,
    offset: isize,
    stride: isize,
}

#[derive(Clone, Copy)]
struct Affine<T> {
    weight: Option<Param<T>>,
    bias: Option<Param<T>>,
}

impl<T> Param<T> {
    /// Element `k` of the vector.
    ///
    /// # Safety
    ///
    /// `k` must be below the validated vector length.
    #[inline(always)]
    unsafe fn at(self, k: usize) -> T
    where
        T: Copy,
    {
        // SAFETY: the caller guarantees `k` is in range of the validated vector.
        unsafe {
            self.ptr
                .get()
                .offset(self.offset + k as isize * self.stride)
                .read()
        }
    }
}

fn param<T: NormScalar>(param: Option<&ErasedRawStridedRef<'_>>) -> Result<Option<Param<T>>> {
    param
        .map(|param| {
            Ok(Param {
                ptr: UnitPtr(param.data_as::<T>()?.as_ptr() as *mut T),
                offset: param.offset(),
                stride: param.strides()[0],
            })
        })
        .transpose()
}

/// Per-plan line constants.
#[derive(Clone, Copy)]
struct Line<T> {
    n: usize,
    n_t: T,
    eps: T,
    ss: isize,
    ds: isize,
}

/// Floating element type supported by the norm plan.
pub(super) trait NormScalar:
    KernelStorageElement
    + MaybeSendSync
    + PartialOrd
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    fn from_usize(value: usize) -> Self;
    fn from_f64(value: f64) -> Self;
    fn sqrt(self) -> Self;
}

macro_rules! impl_norm_scalar {
    ($($ty:ty),*) => {$(
        impl NormScalar for $ty {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            #[inline(always)]
            fn from_usize(value: usize) -> Self {
                value as $ty
            }
            #[inline(always)]
            fn from_f64(value: f64) -> Self {
                value as $ty
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                <$ty>::sqrt(self)
            }
        }
    )*};
}
impl_norm_scalar!(f32, f64);

/// Independent partial sums of the contiguous line kernels.
const NORM_LANES: usize = 8;

/// Sum of `map(x)` over a contiguous slice with [`NORM_LANES`] partial sums.
#[inline(always)]
fn lane_sum<T: NormScalar>(values: &[T], map: impl Fn(T) -> T) -> T {
    let mut partial = [T::ZERO; NORM_LANES];
    let mut chunks = values.chunks_exact(NORM_LANES);
    for chunk in &mut chunks {
        for (partial, &value) in partial.iter_mut().zip(chunk) {
            *partial = *partial + map(value);
        }
    }
    let mut tail = T::ZERO;
    for &value in chunks.remainder() {
        tail = tail + map(value);
    }
    // Fixed pairwise combination of the partial sums.
    let a = (partial[0] + partial[1]) + (partial[2] + partial[3]);
    let b = (partial[4] + partial[5]) + (partial[6] + partial[7]);
    (a + b) + tail
}

/// Sum of `map(x)` over a strided line, sequentially.
///
/// # Safety
///
/// Every `src.offset(so + k * ss)` for `k < n` must be readable.
#[inline(always)]
unsafe fn strided_sum<T: NormScalar>(
    src: *const T,
    so: isize,
    ss: isize,
    n: usize,
    map: impl Fn(T) -> T,
) -> T {
    let mut sum = T::ZERO;
    let mut offset = so;
    for _ in 0..n {
        // SAFETY: the caller guarantees every visited offset.
        sum = sum + map(unsafe { src.offset(offset).read() });
        offset += ss;
    }
    sum
}

/// Statistics of one line: `y = ((x - shift) - mean) * inv`.
///
/// Layer norm shifts every element by the line's first element before
/// summing (the shifted-data two-pass algorithm): `mean` is the mean of the
/// shifted values and the variance is taken about it. A line of equal
/// elements therefore has shifted values, mean and variance of exactly zero.
/// RMS norm uses `shift = mean = 0`.
#[derive(Clone, Copy)]
struct Stats<T> {
    shift: T,
    mean: T,
    inv: T,
}

impl<T: NormScalar> Stats<T> {
    #[inline(always)]
    fn apply(self, x: T) -> T {
        ((x - self.shift) - self.mean) * self.inv
    }
}

/// Statistics of one line.
///
/// # Safety
///
/// Every `src.offset(so + k * line.ss)` for `k < line.n` must be readable, and
/// `line.n > 0`.
#[inline(always)]
unsafe fn line_stats<T: NormScalar, const LAYER: bool>(
    src: *const T,
    so: isize,
    line: Line<T>,
) -> Stats<T> {
    // SAFETY: the caller guarantees every visited offset and `n > 0`; a unit
    // stride line is `n` contiguous elements.
    unsafe {
        let shift = if LAYER {
            src.offset(so).read()
        } else {
            T::ZERO
        };
        let (mean, var) = if line.ss == 1 {
            let values = core::slice::from_raw_parts(src.offset(so), line.n);
            let mean = if LAYER {
                lane_sum(values, |x| x - shift) / line.n_t
            } else {
                T::ZERO
            };
            let var = lane_sum(values, |x| {
                let d = (x - shift) - mean;
                d * d
            }) / line.n_t;
            (mean, var)
        } else {
            let mean = if LAYER {
                strided_sum(src, so, line.ss, line.n, |x| x - shift) / line.n_t
            } else {
                T::ZERO
            };
            let var = strided_sum(src, so, line.ss, line.n, |x| {
                let d = (x - shift) - mean;
                d * d
            }) / line.n_t;
            (mean, var)
        };
        Stats {
            shift,
            mean,
            inv: T::ONE / (var + line.eps).sqrt(),
        }
    }
}

/// Normalizes one line.
///
/// # Safety
///
/// Every source offset `so + k * line.ss` and destination offset
/// `d_o + k * line.ds` for `k < line.n` must be valid, and the affine vectors
/// must hold `line.n` elements; `line.n > 0`.
#[inline(always)]
unsafe fn norm_line<T: NormScalar, const LAYER: bool, const WEIGHT: bool, const BIAS: bool>(
    src: *const T,
    so: isize,
    dst: *mut T,
    d_o: isize,
    line: Line<T>,
    affine: Affine<T>,
) {
    // SAFETY: the caller guarantees every offset below.
    unsafe {
        let stats = line_stats::<T, LAYER>(src, so, line);
        let mut s = so;
        let mut d = d_o;
        for k in 0..line.n {
            let mut y = stats.apply(src.offset(s).read());
            if WEIGHT {
                y = y * affine.weight.unwrap_unchecked().at(k);
            }
            if BIAS {
                y = y + affine.bias.unwrap_unchecked().at(k);
            }
            dst.offset(d).write(y);
            s += line.ss;
            d += line.ds;
        }
    }
}

/// Normalizes `width` adjacent lines whose elements are contiguous across the
/// lines in both source and destination, with the statistics of
/// [`line_stats`] per line (each line summed sequentially).
///
/// # Safety
///
/// As [`norm_line`] for each line `j < width` with source base `so + j` and
/// destination base `d_o + j`; `width <= PANEL`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn norm_panel<T: NormScalar, const LAYER: bool, const WEIGHT: bool, const BIAS: bool>(
    src: *const T,
    so: isize,
    dst: *mut T,
    d_o: isize,
    width: usize,
    line: Line<T>,
    affine: Affine<T>,
) {
    debug_assert!(width <= PANEL);
    let mut shift = [T::ZERO; PANEL];
    let mut mean = [T::ZERO; PANEL];
    let mut scale = [T::ZERO; PANEL];
    let shift = &mut shift[..width];
    let mean = &mut mean[..width];
    let scale = &mut scale[..width];
    // SAFETY: the caller guarantees `width` contiguous elements at every axis
    // position of the source and destination, and `n > 0`.
    unsafe {
        let row =
            |k: usize| core::slice::from_raw_parts(src.offset(so + k as isize * line.ss), width);
        if LAYER {
            shift.copy_from_slice(row(0));
            for k in 0..line.n {
                for ((mean, &shift), &x) in mean.iter_mut().zip(shift.iter()).zip(row(k)) {
                    *mean = *mean + (x - shift);
                }
            }
            for mean in mean.iter_mut() {
                *mean = *mean / line.n_t;
            }
        }
        for k in 0..line.n {
            for (((acc, &shift), &mean), &x) in scale
                .iter_mut()
                .zip(shift.iter())
                .zip(mean.iter())
                .zip(row(k))
            {
                let d = (x - shift) - mean;
                *acc = *acc + d * d;
            }
        }
        for scale in scale.iter_mut() {
            *scale = T::ONE / (*scale / line.n_t + line.eps).sqrt();
        }
        for k in 0..line.n {
            let w = if WEIGHT {
                affine.weight.unwrap_unchecked().at(k)
            } else {
                T::ONE
            };
            let b = if BIAS {
                affine.bias.unwrap_unchecked().at(k)
            } else {
                T::ZERO
            };
            // Raw writes: the destination may be uninitialized.
            let out = dst.offset(d_o + k as isize * line.ds);
            for (lane, (((&x, &shift), &mean), &scale)) in row(k)
                .iter()
                .zip(shift.iter())
                .zip(mean.iter())
                .zip(scale.iter())
                .enumerate()
            {
                let mut y = ((x - shift) - mean) * scale;
                if WEIGHT {
                    y = y * w;
                }
                if BIAS {
                    y = y + b;
                }
                out.add(lane).write(y);
            }
        }
    }
}
