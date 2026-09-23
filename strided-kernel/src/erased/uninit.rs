//! Dtype-erased elementwise entry points that fully overwrite uninitialized
//! output storage.
//!
//! Every entry point validates dtype, shape, destination injectivity, and
//! input/output overlap before the first write. `Ok(())` means every reachable
//! destination element was initialized. A panic during replay may leave the
//! destination partially initialized, which remains safe to drop because the
//! storage is `MaybeUninit`.

use super::{erased_raw_ref, map_output_dtype, raw_any, OneShotScalar};
use crate::*;
use core::mem::MaybeUninit;
use core::num::Wrapping;
use num_complex::{Complex32, Complex64};
use strided_basic::execution::{
    check_dtype, ensure_same_shape, map_raw_into_validated,
    validate_destination_layout_without_alloc, validate_uninit_no_overlap,
    zip_map2_raw_into_validated, ValidatedDestinationLayout,
};

/// Apply one runtime-selected unary operation into uninitialized storage.
///
/// The dtype contract matches [`erased_map_into`](crate::erased_map_into):
/// complex absolute value writes `c32 -> f32` and `c64 -> f64`, `bool`
/// supports only [`ErasedMapOp::Conj`], and signed integers use wrapping
/// negate/abs.
///
/// # Examples
///
/// ```
/// use core::mem::MaybeUninit;
/// use strided_kernel::{
///     erased_map_into_uninit, ErasedMapOp, ErasedRawStridedPtr, ErasedRawStridedRef,
///     ErasedRawStridedUninitMut, ExecContext, KernelDType,
/// };
///
/// let input = [-1.5_f64, 2.0];
/// let input = ErasedRawStridedRef::from_slice(&input, &[2], &[1], 0).unwrap();
/// let mut out = [MaybeUninit::<f64>::uninit(); 2];
/// let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(&mut out, &[2], &[1], 0).unwrap();
/// erased_map_into_uninit(
///     KernelDType::F64,
///     ErasedMapOp::Abs,
///     &ExecContext::serial(),
///     &mut dest,
///     &ErasedRawStridedPtr::from_ref(&input),
/// )
/// .unwrap();
/// // SAFETY: a successful call initializes every reachable element.
/// assert_eq!(unsafe { [out[0].assume_init(), out[1].assume_init()] }, [1.5, 2.0]);
/// ```
///
/// # Errors
///
/// Returns a typed [`StridedError`] for dtype, shape, output-layout, overlap,
/// or unsupported dtype/op contracts. Validation completes before any write.
pub fn erased_map_into_uninit(
    input_dtype: KernelDType,
    op: ErasedMapOp,
    ctx: &ExecContext,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    input: &ErasedRawStridedPtr<'_>,
) -> Result<()> {
    check_dtype(input_dtype, input.dtype())?;
    check_dtype(map_output_dtype(input_dtype, op)?, dest.dtype())?;
    validate_uninit_no_overlap(dest, input, 0)?;
    // SAFETY: input/output overlap was rejected before forming references.
    let input = unsafe { input.try_as_ref_after_no_overlap() }?;

    ctx.run(|| match (input_dtype, op) {
        (KernelDType::C32, ErasedMapOp::Abs) => {
            uninit_map_with::<f32, Complex32>(dest, &input, |value| value.norm())
        }
        (KernelDType::C64, ErasedMapOp::Abs) => {
            uninit_map_with::<f64, Complex64>(dest, &input, |value| value.norm())
        }
        (KernelDType::F32, _) => uninit_map::<f32>(op, dest, &input),
        (KernelDType::F64, _) => uninit_map::<f64>(op, dest, &input),
        (KernelDType::I32, _) => uninit_map::<i32>(op, dest, &input),
        (KernelDType::I64, _) => uninit_map::<i64>(op, dest, &input),
        (KernelDType::Bool, _) => uninit_map::<bool>(op, dest, &input),
        (KernelDType::C32, _) => uninit_map::<Complex32>(op, dest, &input),
        (KernelDType::C64, _) => uninit_map::<Complex64>(op, dest, &input),
        _ => Err(StridedError::UnsupportedDType {
            dtype: input_dtype.label(),
        }),
    })
}

/// Apply one runtime-selected binary operation into uninitialized storage.
///
/// The dtype contract matches [`erased_zip_into`](crate::erased_zip_into).
/// Signed-integer divide and remainder scan the divisor before any write and
/// return [`StridedError::IntegerDivisionByZero`] when it contains zero; all
/// other integer arithmetic wraps. Real maximum/minimum propagate NaN.
///
/// # Examples
///
/// ```
/// use core::mem::MaybeUninit;
/// use strided_kernel::{
///     erased_zip_into_uninit, ErasedRawStridedPtr, ErasedRawStridedRef,
///     ErasedRawStridedUninitMut, ErasedZipOp, ExecContext, KernelDType,
/// };
///
/// let lhs = [i32::MAX, 7];
/// let rhs = [1_i32, 2];
/// let lhs = ErasedRawStridedRef::from_slice(&lhs, &[2], &[1], 0).unwrap();
/// let rhs = ErasedRawStridedRef::from_slice(&rhs, &[2], &[1], 0).unwrap();
/// let mut out = [MaybeUninit::<i32>::uninit(); 2];
/// let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(&mut out, &[2], &[1], 0).unwrap();
/// erased_zip_into_uninit(
///     KernelDType::I32,
///     ErasedZipOp::Add,
///     &ExecContext::serial(),
///     &mut dest,
///     &ErasedRawStridedPtr::from_ref(&lhs),
///     &ErasedRawStridedPtr::from_ref(&rhs),
/// )
/// .unwrap();
/// // SAFETY: a successful call initializes every reachable element.
/// assert_eq!(unsafe { [out[0].assume_init(), out[1].assume_init()] }, [i32::MIN, 9]);
/// ```
///
/// # Errors
///
/// Returns a typed [`StridedError`] for dtype, shape, output-layout, overlap,
/// unsupported dtype/op contracts, or an integer zero divisor. Validation
/// completes before any write.
pub fn erased_zip_into_uninit(
    dtype: KernelDType,
    op: ErasedZipOp,
    ctx: &ExecContext,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    lhs: &ErasedRawStridedPtr<'_>,
    rhs: &ErasedRawStridedPtr<'_>,
) -> Result<()> {
    check_dtype(dtype, dest.dtype())?;
    check_dtype(dtype, lhs.dtype())?;
    check_dtype(dtype, rhs.dtype())?;
    validate_uninit_no_overlap(dest, lhs, 0)?;
    validate_uninit_no_overlap(dest, rhs, 1)?;
    // SAFETY: input/output overlap was rejected before forming references.
    let lhs = unsafe { lhs.try_as_ref_after_no_overlap() }?;
    // SAFETY: input/output overlap was rejected before forming references.
    let rhs = unsafe { rhs.try_as_ref_after_no_overlap() }?;

    ctx.run(|| match dtype {
        KernelDType::F32 => uninit_zip::<f32>(op, dest, &lhs, &rhs),
        KernelDType::F64 => uninit_zip::<f64>(op, dest, &lhs, &rhs),
        KernelDType::I32 => uninit_zip::<i32>(op, dest, &lhs, &rhs),
        KernelDType::I64 => uninit_zip::<i64>(op, dest, &lhs, &rhs),
        KernelDType::Bool => uninit_zip::<bool>(op, dest, &lhs, &rhs),
        KernelDType::C32 => uninit_zip::<Complex32>(op, dest, &lhs, &rhs),
        KernelDType::C64 => uninit_zip::<Complex64>(op, dest, &lhs, &rhs),
        _ => Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        }),
    })
}

/// Compare two same-dtype operands elementwise into uninitialized `bool`
/// storage.
///
/// Real, signed-integer, and `bool` operands support every [`CompareOp`] with
/// `PartialOrd` semantics, so any comparison involving NaN is `false`.
/// Complex operands support only [`CompareOp::Eq`]; ordered comparisons return
/// [`StridedError::UnsupportedOp`]. The destination dtype must be `bool`.
///
/// # Examples
///
/// ```
/// use core::mem::MaybeUninit;
/// use strided_kernel::{
///     erased_compare_into_uninit, CompareOp, ErasedRawStridedPtr, ErasedRawStridedRef,
///     ErasedRawStridedUninitMut, ExecContext, KernelDType,
/// };
///
/// let lhs = [1.0_f64, f64::NAN];
/// let rhs = [2.0_f64, 0.0];
/// let lhs = ErasedRawStridedRef::from_slice(&lhs, &[2], &[1], 0).unwrap();
/// let rhs = ErasedRawStridedRef::from_slice(&rhs, &[2], &[1], 0).unwrap();
/// let mut out = [MaybeUninit::<bool>::uninit(); 2];
/// let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(&mut out, &[2], &[1], 0).unwrap();
/// erased_compare_into_uninit(
///     KernelDType::F64,
///     CompareOp::Lt,
///     &ExecContext::serial(),
///     &mut dest,
///     &ErasedRawStridedPtr::from_ref(&lhs),
///     &ErasedRawStridedPtr::from_ref(&rhs),
/// )
/// .unwrap();
/// // SAFETY: a successful call initializes every reachable element.
/// assert_eq!(unsafe { [out[0].assume_init(), out[1].assume_init()] }, [true, false]);
/// ```
///
/// # Errors
///
/// Returns a typed [`StridedError`] for dtype, shape, output-layout, overlap,
/// or unsupported dtype/op contracts. Validation completes before any write.
pub fn erased_compare_into_uninit(
    dtype: KernelDType,
    op: CompareOp,
    ctx: &ExecContext,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    lhs: &ErasedRawStridedPtr<'_>,
    rhs: &ErasedRawStridedPtr<'_>,
) -> Result<()> {
    check_dtype(KernelDType::Bool, dest.dtype())?;
    check_dtype(dtype, lhs.dtype())?;
    check_dtype(dtype, rhs.dtype())?;
    validate_uninit_no_overlap(dest, lhs, 0)?;
    validate_uninit_no_overlap(dest, rhs, 1)?;
    // SAFETY: input/output overlap was rejected before forming references.
    let lhs = unsafe { lhs.try_as_ref_after_no_overlap() }?;
    // SAFETY: input/output overlap was rejected before forming references.
    let rhs = unsafe { rhs.try_as_ref_after_no_overlap() }?;

    ctx.run(|| match dtype {
        KernelDType::F32 => uninit_compare_ordered::<f32>(op, dest, &lhs, &rhs),
        KernelDType::F64 => uninit_compare_ordered::<f64>(op, dest, &lhs, &rhs),
        KernelDType::I32 => uninit_compare_ordered::<i32>(op, dest, &lhs, &rhs),
        KernelDType::I64 => uninit_compare_ordered::<i64>(op, dest, &lhs, &rhs),
        KernelDType::Bool => uninit_compare_ordered::<bool>(op, dest, &lhs, &rhs),
        KernelDType::C32 => uninit_compare_eq::<Complex32>(op, dest, &lhs, &rhs),
        KernelDType::C64 => uninit_compare_eq::<Complex64>(op, dest, &lhs, &rhs),
        _ => Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        }),
    })
}

/// Select elementwise between two same-dtype operands by a `bool` predicate,
/// writing uninitialized storage: `dest[i] = if pred[i] { on_true[i] } else
/// { on_false[i] }`.
///
/// Every dtype is supported. The predicate dtype must be `bool`.
///
/// # Examples
///
/// ```
/// use core::mem::MaybeUninit;
/// use strided_kernel::{
///     erased_select_into_uninit, ErasedRawStridedPtr, ErasedRawStridedRef,
///     ErasedRawStridedUninitMut, ExecContext, KernelDType,
/// };
///
/// let pred = [true, false];
/// let on_true = [1_i64, 2];
/// let on_false = [10_i64, 20];
/// let pred = ErasedRawStridedRef::from_slice(&pred, &[2], &[1], 0).unwrap();
/// let on_true = ErasedRawStridedRef::from_slice(&on_true, &[2], &[1], 0).unwrap();
/// let on_false = ErasedRawStridedRef::from_slice(&on_false, &[2], &[1], 0).unwrap();
/// let mut out = [MaybeUninit::<i64>::uninit(); 2];
/// let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(&mut out, &[2], &[1], 0).unwrap();
/// erased_select_into_uninit(
///     KernelDType::I64,
///     &ExecContext::serial(),
///     &mut dest,
///     &ErasedRawStridedPtr::from_ref(&pred),
///     &ErasedRawStridedPtr::from_ref(&on_true),
///     &ErasedRawStridedPtr::from_ref(&on_false),
/// )
/// .unwrap();
/// // SAFETY: a successful call initializes every reachable element.
/// assert_eq!(unsafe { [out[0].assume_init(), out[1].assume_init()] }, [1, 20]);
/// ```
///
/// # Errors
///
/// Returns a typed [`StridedError`] for dtype, shape, output-layout, or
/// overlap contracts. Validation completes before any write.
pub fn erased_select_into_uninit(
    dtype: KernelDType,
    ctx: &ExecContext,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    pred: &ErasedRawStridedPtr<'_>,
    on_true: &ErasedRawStridedPtr<'_>,
    on_false: &ErasedRawStridedPtr<'_>,
) -> Result<()> {
    check_dtype(dtype, dest.dtype())?;
    check_dtype(KernelDType::Bool, pred.dtype())?;
    check_dtype(dtype, on_true.dtype())?;
    check_dtype(dtype, on_false.dtype())?;
    validate_uninit_no_overlap(dest, pred, 0)?;
    validate_uninit_no_overlap(dest, on_true, 1)?;
    validate_uninit_no_overlap(dest, on_false, 2)?;
    // SAFETY: input/output overlap was rejected before forming references.
    let pred = unsafe { pred.try_as_ref_after_no_overlap() }?;
    // SAFETY: input/output overlap was rejected before forming references.
    let on_true = unsafe { on_true.try_as_ref_after_no_overlap() }?;
    // SAFETY: input/output overlap was rejected before forming references.
    let on_false = unsafe { on_false.try_as_ref_after_no_overlap() }?;

    ctx.run(|| match dtype {
        KernelDType::F32 => uninit_select::<f32>(dest, &pred, &on_true, &on_false),
        KernelDType::F64 => uninit_select::<f64>(dest, &pred, &on_true, &on_false),
        KernelDType::I32 => uninit_select::<i32>(dest, &pred, &on_true, &on_false),
        KernelDType::I64 => uninit_select::<i64>(dest, &pred, &on_true, &on_false),
        KernelDType::Bool => uninit_select::<bool>(dest, &pred, &on_true, &on_false),
        KernelDType::C32 => uninit_select::<Complex32>(dest, &pred, &on_true, &on_false),
        KernelDType::C64 => uninit_select::<Complex64>(dest, &pred, &on_true, &on_false),
        _ => Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        }),
    })
}

/// Clamp elementwise into uninitialized storage:
/// `dest[i] = minimum(hi[i], maximum(lo[i], x[i]))`.
///
/// `maximum` and `minimum` are [`ErasedZipOp::Maximum`] and
/// [`ErasedZipOp::Minimum`], so a NaN in any real operand yields NaN and
/// `lo > hi` yields `hi`. Supported dtypes are `f32`, `f64`, `i32`, and `i64`.
///
/// # Examples
///
/// ```
/// use core::mem::MaybeUninit;
/// use strided_kernel::{
///     erased_clamp_into_uninit, ErasedRawStridedPtr, ErasedRawStridedRef,
///     ErasedRawStridedUninitMut, ExecContext, KernelDType,
/// };
///
/// let x = [-3.0_f32, 0.5, 9.0];
/// let lo = [0.0_f32; 1];
/// let hi = [1.0_f32; 1];
/// let x = ErasedRawStridedRef::from_slice(&x, &[3], &[1], 0).unwrap();
/// // Stride-0 descriptors broadcast one bound over the output.
/// let lo = ErasedRawStridedRef::from_slice(&lo, &[3], &[0], 0).unwrap();
/// let hi = ErasedRawStridedRef::from_slice(&hi, &[3], &[0], 0).unwrap();
/// let mut out = [MaybeUninit::<f32>::uninit(); 3];
/// let mut dest = ErasedRawStridedUninitMut::from_uninit_slice(&mut out, &[3], &[1], 0).unwrap();
/// erased_clamp_into_uninit(
///     KernelDType::F32,
///     &ExecContext::serial(),
///     &mut dest,
///     &ErasedRawStridedPtr::from_ref(&x),
///     &ErasedRawStridedPtr::from_ref(&lo),
///     &ErasedRawStridedPtr::from_ref(&hi),
/// )
/// .unwrap();
/// // SAFETY: a successful call initializes every reachable element.
/// let out = unsafe { out.map(|value| value.assume_init()) };
/// assert_eq!(out, [0.0, 0.5, 1.0]);
/// ```
///
/// # Errors
///
/// Returns a typed [`StridedError`] for dtype, shape, output-layout, or
/// overlap contracts. Validation completes before any write.
pub fn erased_clamp_into_uninit(
    dtype: KernelDType,
    ctx: &ExecContext,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    x: &ErasedRawStridedPtr<'_>,
    lo: &ErasedRawStridedPtr<'_>,
    hi: &ErasedRawStridedPtr<'_>,
) -> Result<()> {
    check_dtype(dtype, dest.dtype())?;
    check_dtype(dtype, x.dtype())?;
    check_dtype(dtype, lo.dtype())?;
    check_dtype(dtype, hi.dtype())?;
    validate_uninit_no_overlap(dest, x, 0)?;
    validate_uninit_no_overlap(dest, lo, 1)?;
    validate_uninit_no_overlap(dest, hi, 2)?;
    // SAFETY: input/output overlap was rejected before forming references.
    let x = unsafe { x.try_as_ref_after_no_overlap() }?;
    // SAFETY: input/output overlap was rejected before forming references.
    let lo = unsafe { lo.try_as_ref_after_no_overlap() }?;
    // SAFETY: input/output overlap was rejected before forming references.
    let hi = unsafe { hi.try_as_ref_after_no_overlap() }?;

    ctx.run(|| match dtype {
        KernelDType::F32 => uninit_clamp::<f32>(dest, &x, &lo, &hi),
        KernelDType::F64 => uninit_clamp::<f64>(dest, &x, &lo, &hi),
        KernelDType::I32 => uninit_clamp::<i32>(dest, &x, &lo, &hi),
        KernelDType::I64 => uninit_clamp::<i64>(dest, &x, &lo, &hi),
        _ => Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        }),
    })
}

/// Broadcast two same-dtype operands onto the destination axes and multiply
/// them into uninitialized storage.
///
/// `lhs_axes[k]` names the destination axis that `lhs` axis `k` maps to; each
/// mapped extent must equal the destination extent or be one, and unmapped
/// destination axes broadcast. Supported dtypes are `f32`, `f64`, `c32`,
/// `c64`, and the signed integers, whose products wrap on overflow.
///
/// # Examples
///
/// ```
/// use core::mem::MaybeUninit;
/// use strided_kernel::{
///     erased_broadcast_mul_into_uninit, ErasedRawStridedPtr, ErasedRawStridedRef,
///     ErasedRawStridedUninitMut, ExecContext, KernelDType,
/// };
///
/// // Outer product: out[i, j] = a[i] * b[j], column-major 2 x 3 output.
/// let a = [1.0_f64, 2.0];
/// let b = [10.0_f64, 20.0, 30.0];
/// let a = ErasedRawStridedRef::from_slice(&a, &[2], &[1], 0).unwrap();
/// let b = ErasedRawStridedRef::from_slice(&b, &[3], &[1], 0).unwrap();
/// let mut out = [MaybeUninit::<f64>::uninit(); 6];
/// let mut dest =
///     ErasedRawStridedUninitMut::from_uninit_slice(&mut out, &[2, 3], &[1, 2], 0).unwrap();
/// erased_broadcast_mul_into_uninit(
///     KernelDType::F64,
///     &ExecContext::serial(),
///     &mut dest,
///     &ErasedRawStridedPtr::from_ref(&a),
///     &[0],
///     &ErasedRawStridedPtr::from_ref(&b),
///     &[1],
/// )
/// .unwrap();
/// // SAFETY: a successful call initializes every reachable element.
/// let out = unsafe { out.map(|value| value.assume_init()) };
/// assert_eq!(out, [10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);
/// ```
///
/// # Errors
///
/// Returns a typed [`StridedError`] for dtype, axis mapping, shape,
/// output-layout, or overlap contracts. Validation completes before any write.
pub fn erased_broadcast_mul_into_uninit(
    dtype: KernelDType,
    ctx: &ExecContext,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    lhs: &ErasedRawStridedPtr<'_>,
    lhs_axes: &[usize],
    rhs: &ErasedRawStridedPtr<'_>,
    rhs_axes: &[usize],
) -> Result<()> {
    check_dtype(dtype, dest.dtype())?;
    check_dtype(dtype, lhs.dtype())?;
    check_dtype(dtype, rhs.dtype())?;
    validate_uninit_no_overlap(dest, lhs, 0)?;
    validate_uninit_no_overlap(dest, rhs, 1)?;
    // SAFETY: input/output overlap was rejected before forming references.
    let lhs = unsafe { lhs.try_as_ref_after_no_overlap() }?;
    // SAFETY: input/output overlap was rejected before forming references.
    let rhs = unsafe { rhs.try_as_ref_after_no_overlap() }?;

    ctx.run(|| match dtype {
        KernelDType::F32 => uninit_broadcast_mul::<f32>(dest, &lhs, lhs_axes, &rhs, rhs_axes),
        KernelDType::F64 => uninit_broadcast_mul::<f64>(dest, &lhs, lhs_axes, &rhs, rhs_axes),
        KernelDType::C32 => uninit_broadcast_mul::<Complex32>(dest, &lhs, lhs_axes, &rhs, rhs_axes),
        KernelDType::C64 => uninit_broadcast_mul::<Complex64>(dest, &lhs, lhs_axes, &rhs, rhs_axes),
        KernelDType::I32 => {
            uninit_broadcast_mul_wrapping::<i32>(dest, &lhs, lhs_axes, &rhs, rhs_axes)
        }
        KernelDType::I64 => {
            uninit_broadcast_mul_wrapping::<i64>(dest, &lhs, lhs_axes, &rhs, rhs_axes)
        }
        _ => Err(StridedError::UnsupportedDType {
            dtype: dtype.label(),
        }),
    })
}

/// Validate the destination layout and the input shapes.
///
/// Returns `None` for an empty destination, which needs no writes.
fn prepare_destination(
    dest: &ErasedRawStridedUninitMut<'_>,
    inputs: &[&[usize]],
) -> Result<Option<ValidatedDestinationLayout>> {
    let validated = validate_destination_layout_without_alloc(dest.dims(), dest.strides())?;
    for input_dims in inputs {
        ensure_same_shape(dest.dims(), input_dims)?;
    }
    if dest.dims().contains(&0) {
        Ok(None)
    } else {
        Ok(Some(validated))
    }
}

fn uninit_raw_mut<'a, T: KernelStorageElement>(
    dest: &'a mut ErasedRawStridedUninitMut<'_>,
) -> Result<RawStridedMut<'a, MaybeUninit<T>>> {
    let dims = dest.dims();
    let strides = dest.strides();
    let offset = dest.offset();
    let data = dest.data_as_uninit_mut::<T>()?;
    // SAFETY: the uninit descriptor validated every reachable offset against
    // its storage at construction.
    Ok(unsafe { RawStridedMut::new_unchecked(data, dims, strides, offset) })
}

fn uninit_view_mut<'a, T: KernelStorageElement>(
    dest: &'a mut ErasedRawStridedUninitMut<'_>,
) -> Result<StridedViewMut<'a, MaybeUninit<T>>> {
    let dims = dest.dims();
    let strides = dest.strides();
    let offset = dest.offset();
    let data = dest.data_as_uninit_mut::<T>()?;
    // SAFETY: the uninit descriptor validated every reachable offset against
    // its storage at construction.
    Ok(unsafe { StridedViewMut::new_unchecked(data, dims, strides, offset) })
}

fn erased_view<'a, T: KernelStorageElement>(
    src: &'a ErasedRawStridedRef<'a>,
) -> Result<StridedView<'a, T>> {
    let data = src.data_as::<T>()?;
    // SAFETY: the descriptor validated every reachable offset at construction.
    Ok(unsafe { StridedView::new_unchecked(data, src.dims(), src.strides(), src.offset()) })
}

fn uninit_map<T: OneShotScalar>(
    op: ErasedMapOp,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    input: &ErasedRawStridedRef<'_>,
) -> Result<()> {
    if !T::supports_map(op) {
        return Err(StridedError::UnsupportedOp {
            op: op.label(),
            dtype: T::one_shot_dtype_label(),
        });
    }
    // Select the operation once, outside the element loop, so each replay
    // closure is monomorphic and the inner loop can vectorize.
    match op {
        ErasedMapOp::Negate => {
            uninit_map_with::<T, T>(dest, input, |value| T::map(ErasedMapOp::Negate, value))
        }
        ErasedMapOp::Conj => {
            uninit_map_with::<T, T>(dest, input, |value| T::map(ErasedMapOp::Conj, value))
        }
        ErasedMapOp::Abs => {
            uninit_map_with::<T, T>(dest, input, |value| T::map(ErasedMapOp::Abs, value))
        }
        ErasedMapOp::Sign => {
            uninit_map_with::<T, T>(dest, input, |value| T::map(ErasedMapOp::Sign, value))
        }
    }
}

fn uninit_map_with<D, A>(
    dest: &mut ErasedRawStridedUninitMut<'_>,
    input: &ErasedRawStridedRef<'_>,
    map: impl Fn(A) -> D + crate::MaybeSync,
) -> Result<()>
where
    D: Copy + crate::MaybeSendSync + KernelStorageElement,
    A: Copy + crate::MaybeSendSync + KernelStorageElement,
{
    let Some(validated) = prepare_destination(dest, &[input.dims()])? else {
        return Ok(());
    };
    let input = erased_raw_ref::<A>(input)?;
    let mut dest = uninit_raw_mut::<D>(dest)?;
    // SAFETY: matching shapes and this destination layout were validated above,
    // and the caller rejected input/output overlap.
    unsafe {
        map_raw_into_validated::<MaybeUninit<D>, A, Identity>(
            &mut dest,
            &input,
            |value| MaybeUninit::new(map(value)),
            validated,
        )
    }
}

fn uninit_zip<T: OneShotScalar>(
    op: ErasedZipOp,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    lhs: &ErasedRawStridedRef<'_>,
    rhs: &ErasedRawStridedRef<'_>,
) -> Result<()> {
    if !T::supports_zip(op) {
        return Err(StridedError::UnsupportedOp {
            op: op.label(),
            dtype: T::one_shot_dtype_label(),
        });
    }
    let Some(validated) = prepare_destination(dest, &[lhs.dims(), rhs.dims()])? else {
        return Ok(());
    };
    let lhs = erased_raw_ref::<T>(lhs)?;
    let rhs = erased_raw_ref::<T>(rhs)?;
    if matches!(op, ErasedZipOp::Divide | ErasedZipOp::Remainder)
        && T::INTEGER
        && raw_any(&rhs, T::is_zero)?
    {
        return Err(StridedError::IntegerDivisionByZero { op: op.label() });
    }
    let mut dest = uninit_raw_mut::<T>(dest)?;
    // Select the operation once, outside the element loop, so each replay
    // closure is monomorphic and the inner loop can vectorize.
    macro_rules! replay {
        ($op:ident) => {
            // SAFETY: matching shapes and this destination layout were
            // validated above, and the caller rejected input/output overlap.
            unsafe {
                zip_map2_raw_into_validated::<MaybeUninit<T>, T, T, Identity, Identity>(
                    &mut dest,
                    &lhs,
                    &rhs,
                    |lhs, rhs| MaybeUninit::new(T::zip(ErasedZipOp::$op, lhs, rhs)),
                    validated,
                )
            }
        };
    }
    match op {
        ErasedZipOp::Add => replay!(Add),
        ErasedZipOp::Subtract => replay!(Subtract),
        ErasedZipOp::Multiply => replay!(Multiply),
        ErasedZipOp::Divide => replay!(Divide),
        ErasedZipOp::Remainder => replay!(Remainder),
        ErasedZipOp::Maximum => replay!(Maximum),
        ErasedZipOp::Minimum => replay!(Minimum),
    }
}

fn uninit_compare_with<T>(
    dest: &mut ErasedRawStridedUninitMut<'_>,
    lhs: &ErasedRawStridedRef<'_>,
    rhs: &ErasedRawStridedRef<'_>,
    compare: impl Fn(T, T) -> bool + crate::MaybeSync,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
{
    let Some(validated) = prepare_destination(dest, &[lhs.dims(), rhs.dims()])? else {
        return Ok(());
    };
    let lhs = erased_raw_ref::<T>(lhs)?;
    let rhs = erased_raw_ref::<T>(rhs)?;
    let mut dest = uninit_raw_mut::<bool>(dest)?;
    // SAFETY: matching shapes and this destination layout were validated above,
    // and the caller rejected input/output overlap.
    unsafe {
        zip_map2_raw_into_validated::<MaybeUninit<bool>, T, T, Identity, Identity>(
            &mut dest,
            &lhs,
            &rhs,
            |lhs, rhs| MaybeUninit::new(compare(lhs, rhs)),
            validated,
        )
    }
}

fn uninit_compare_ordered<T>(
    op: CompareOp,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    lhs: &ErasedRawStridedRef<'_>,
    rhs: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement + PartialOrd,
{
    // The comparison is selected once, outside the element loop.
    match op {
        CompareOp::Eq => uninit_compare_with::<T>(dest, lhs, rhs, |a, b| a == b),
        CompareOp::Lt => uninit_compare_with::<T>(dest, lhs, rhs, |a, b| a < b),
        CompareOp::Le => uninit_compare_with::<T>(dest, lhs, rhs, |a, b| a <= b),
        CompareOp::Gt => uninit_compare_with::<T>(dest, lhs, rhs, |a, b| a > b),
        CompareOp::Ge => uninit_compare_with::<T>(dest, lhs, rhs, |a, b| a >= b),
        _ => Err(StridedError::UnsupportedOp {
            op: compare_label(op),
            dtype: T::DTYPE.label(),
        }),
    }
}

fn uninit_compare_eq<T>(
    op: CompareOp,
    dest: &mut ErasedRawStridedUninitMut<'_>,
    lhs: &ErasedRawStridedRef<'_>,
    rhs: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement + PartialEq,
{
    match op {
        CompareOp::Eq => uninit_compare_with::<T>(dest, lhs, rhs, |a, b| a == b),
        _ => Err(StridedError::UnsupportedOp {
            op: compare_label(op),
            dtype: T::DTYPE.label(),
        }),
    }
}

fn compare_label(op: CompareOp) -> &'static str {
    match op {
        CompareOp::Eq => "eq",
        CompareOp::Lt => "lt",
        CompareOp::Le => "le",
        CompareOp::Gt => "gt",
        CompareOp::Ge => "ge",
        _ => "compare",
    }
}

fn uninit_select<T>(
    dest: &mut ErasedRawStridedUninitMut<'_>,
    pred: &ErasedRawStridedRef<'_>,
    on_true: &ErasedRawStridedRef<'_>,
    on_false: &ErasedRawStridedRef<'_>,
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
{
    let dims: [&[usize]; 3] = [pred.dims(), on_true.dims(), on_false.dims()];
    if prepare_destination(dest, &dims)?.is_none() {
        return Ok(());
    }
    let pred = erased_view::<bool>(pred)?;
    let on_true = erased_view::<T>(on_true)?;
    let on_false = erased_view::<T>(on_false)?;
    let mut dest = uninit_view_mut::<T>(dest)?;
    zip_map3_into(&mut dest, &pred, &on_true, &on_false, |pred, a, b| {
        MaybeUninit::new(if pred { a } else { b })
    })
}

fn uninit_clamp<T: OneShotScalar>(
    dest: &mut ErasedRawStridedUninitMut<'_>,
    x: &ErasedRawStridedRef<'_>,
    lo: &ErasedRawStridedRef<'_>,
    hi: &ErasedRawStridedRef<'_>,
) -> Result<()> {
    let dims: [&[usize]; 3] = [x.dims(), lo.dims(), hi.dims()];
    if prepare_destination(dest, &dims)?.is_none() {
        return Ok(());
    }
    let x = erased_view::<T>(x)?;
    let lo = erased_view::<T>(lo)?;
    let hi = erased_view::<T>(hi)?;
    let mut dest = uninit_view_mut::<T>(dest)?;
    zip_map3_into(&mut dest, &x, &lo, &hi, |x, lo, hi| {
        MaybeUninit::new(T::clamp(x, lo, hi))
    })
}

fn uninit_broadcast_mul<T>(
    dest: &mut ErasedRawStridedUninitMut<'_>,
    lhs: &ErasedRawStridedRef<'_>,
    lhs_axes: &[usize],
    rhs: &ErasedRawStridedRef<'_>,
    rhs_axes: &[usize],
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement + core::ops::Mul<Output = T>,
{
    let lhs = erased_view::<T>(lhs)?;
    let rhs = erased_view::<T>(rhs)?;
    let mut dest = uninit_view_mut::<T>(dest)?;
    broadcast_mul_into_uninit(&mut dest, &lhs, lhs_axes, &rhs, rhs_axes)
}

fn uninit_broadcast_mul_wrapping<T>(
    dest: &mut ErasedRawStridedUninitMut<'_>,
    lhs: &ErasedRawStridedRef<'_>,
    lhs_axes: &[usize],
    rhs: &ErasedRawStridedRef<'_>,
    rhs_axes: &[usize],
) -> Result<()>
where
    T: Copy + crate::MaybeSendSync + KernelStorageElement,
    Wrapping<T>: core::ops::Mul<Output = Wrapping<T>> + crate::MaybeSendSync,
{
    let lhs_data = wrapping_slice(lhs.data_as::<T>()?);
    let rhs_data = wrapping_slice(rhs.data_as::<T>()?);
    // SAFETY: the descriptors validated every reachable offset at construction,
    // and the reinterpreted slices keep their element count.
    let lhs = unsafe {
        StridedView::<Wrapping<T>, Identity>::new_unchecked(
            lhs_data,
            lhs.dims(),
            lhs.strides(),
            lhs.offset(),
        )
    };
    // SAFETY: as above.
    let rhs = unsafe {
        StridedView::<Wrapping<T>, Identity>::new_unchecked(
            rhs_data,
            rhs.dims(),
            rhs.strides(),
            rhs.offset(),
        )
    };
    let dims = dest.dims();
    let strides = dest.strides();
    let offset = dest.offset();
    let data = dest.data_as_uninit_mut::<T>()?;
    // SAFETY: `Wrapping<T>` is `repr(transparent)` over `T`, so
    // `MaybeUninit<Wrapping<T>>` has the layout of `MaybeUninit<T>`, and the
    // slice keeps its element count and exclusive borrow.
    let data = unsafe {
        core::slice::from_raw_parts_mut(
            data.as_mut_ptr().cast::<MaybeUninit<Wrapping<T>>>(),
            data.len(),
        )
    };
    // SAFETY: the uninit descriptor validated every reachable offset.
    let mut dest = unsafe { StridedViewMut::new_unchecked(data, dims, strides, offset) };
    broadcast_mul_into_uninit(&mut dest, &lhs, lhs_axes, &rhs, rhs_axes)
}

fn wrapping_slice<T>(data: &[T]) -> &[Wrapping<T>] {
    // SAFETY: `Wrapping<T>` is `repr(transparent)` over `T`; the slice keeps
    // its element count and shared borrow.
    unsafe { core::slice::from_raw_parts(data.as_ptr().cast::<Wrapping<T>>(), data.len()) }
}
