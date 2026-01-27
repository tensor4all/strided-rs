//! Lazy broadcast evaluation ported from Julia's Strided.jl/src/broadcast.jl
//!
//! This module provides the `CaptureArgs` type for deferred broadcast evaluation,
//! allowing complex broadcast expressions to be evaluated lazily during kernel execution.
//!
//! # Julia equivalent
//!
//! ```julia
//! struct CaptureArgs{F,Args<:Tuple}
//!     f::F
//!     args::Args
//! end
//! struct Arg
//! end
//! ```
//!
//! # Example
//!
//! ```ignore
//! use mdarray_strided::broadcast::{CaptureArgs, Arg, Scalar};
//!
//! // Capture: f(x, y, scalar) where x and y are arrays
//! let captured = CaptureArgs::new(
//!     |a: f64, b: f64, c: f64| a + b * c,
//!     (Arg, Arg, Scalar(2.0)),
//! );
//!
//! // During kernel iteration, call with values from arrays
//! let result = captured.call(&[1.0, 3.0]); // 1.0 + 3.0 * 2.0 = 7.0
//! ```

use std::marker::PhantomData;

use crate::element_op::{ElementOp, ElementOpApply, Identity};
use crate::kernel::{build_plan_fused, for_each_inner_block, is_contiguous, total_len};
use crate::promote::promote_strides_to_shape;
use crate::view::{StridedArrayView, StridedArrayViewMut};
use crate::Result;

/// Type alias for broadcast promotion result with three views
type PromoteShape3Result<'a, T, const N: usize, Op1, Op2, Op3> = Result<(
    StridedArrayView<'a, T, N, Op1>,
    StridedArrayView<'a, T, N, Op2>,
    StridedArrayView<'a, T, N, Op3>,
)>;


// ============================================================================
// Core types for lazy broadcast
// ============================================================================

/// Marker type representing an array argument placeholder.
///
/// When building a `CaptureArgs`, array arguments are replaced with `Arg`
/// markers. During evaluation, each `Arg` consumes one value from the
/// input value stream.
///
/// # Julia equivalent
/// ```julia
/// struct Arg
/// end
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Arg;

/// A scalar value that doesn't consume from the value stream.
///
/// Scalars are kept as-is during CaptureArgs evaluation.
#[derive(Debug, Clone, Copy)]
pub struct Scalar<T>(pub T);

// ============================================================================
// CaptureArgs traits for type-safe consumption
// ============================================================================

/// Trait for types that can consume values from an iterator during broadcast evaluation.
///
/// This is the core abstraction for lazy broadcast: each argument type knows how to
/// consume zero or more values from the value stream and produce a result.
pub trait Consume<T> {
    /// The output type after consumption.
    type Output;

    /// Consume values from the iterator and return the result.
    fn consume<I: Iterator<Item = T>>(&self, values: &mut I) -> Self::Output;
}

impl<T: Copy> Consume<T> for Arg {
    type Output = T;

    #[inline]
    fn consume<I: Iterator<Item = T>>(&self, values: &mut I) -> T {
        values
            .next()
            .expect("not enough values for Arg consumption")
    }
}

impl<T: Copy> Consume<T> for Scalar<T> {
    type Output = T;

    #[inline]
    fn consume<I: Iterator<Item = T>>(&self, _values: &mut I) -> T {
        self.0
    }
}

// ============================================================================
// CaptureArgs for different arities
// ============================================================================

/// Captured broadcast operation with lazy evaluation.
///
/// `CaptureArgs` stores a function and its arguments, where array arguments
/// are replaced with `Arg` markers. During kernel execution, values are
/// consumed from arrays and passed to the function.
///
/// # Type Parameters
/// - `F`: The function type
/// - `A`: Tuple of arguments (mix of `Arg`, `Scalar`, or nested `CaptureArgs`)
///
/// # Julia equivalent
/// ```julia
/// struct CaptureArgs{F,Args<:Tuple}
///     f::F
///     args::Args
/// end
/// ```
#[derive(Debug, Clone)]
pub struct CaptureArgs<F, A> {
    pub f: F,
    pub args: A,
}

impl<F, A> CaptureArgs<F, A> {
    /// Create a new CaptureArgs with the given function and arguments.
    pub fn new(f: F, args: A) -> Self {
        Self { f, args }
    }
}

// Implement Consume for CaptureArgs with 1 argument
impl<T, F, A1> Consume<T> for CaptureArgs<F, (A1,)>
where
    T: Copy,
    F: Fn(A1::Output) -> T,
    A1: Consume<T>,
{
    type Output = T;

    #[inline]
    fn consume<I: Iterator<Item = T>>(&self, values: &mut I) -> T {
        let a1 = self.args.0.consume(values);
        (self.f)(a1)
    }
}

// Implement Consume for CaptureArgs with 2 arguments
impl<T, F, A1, A2> Consume<T> for CaptureArgs<F, (A1, A2)>
where
    T: Copy,
    F: Fn(A1::Output, A2::Output) -> T,
    A1: Consume<T>,
    A2: Consume<T>,
{
    type Output = T;

    #[inline]
    fn consume<I: Iterator<Item = T>>(&self, values: &mut I) -> T {
        let a1 = self.args.0.consume(values);
        let a2 = self.args.1.consume(values);
        (self.f)(a1, a2)
    }
}

// Implement Consume for CaptureArgs with 3 arguments
impl<T, F, A1, A2, A3> Consume<T> for CaptureArgs<F, (A1, A2, A3)>
where
    T: Copy,
    F: Fn(A1::Output, A2::Output, A3::Output) -> T,
    A1: Consume<T>,
    A2: Consume<T>,
    A3: Consume<T>,
{
    type Output = T;

    #[inline]
    fn consume<I: Iterator<Item = T>>(&self, values: &mut I) -> T {
        let a1 = self.args.0.consume(values);
        let a2 = self.args.1.consume(values);
        let a3 = self.args.2.consume(values);
        (self.f)(a1, a2, a3)
    }
}

// Implement Consume for CaptureArgs with 4 arguments
impl<T, F, A1, A2, A3, A4> Consume<T> for CaptureArgs<F, (A1, A2, A3, A4)>
where
    T: Copy,
    F: Fn(A1::Output, A2::Output, A3::Output, A4::Output) -> T,
    A1: Consume<T>,
    A2: Consume<T>,
    A3: Consume<T>,
    A4: Consume<T>,
{
    type Output = T;

    #[inline]
    fn consume<I: Iterator<Item = T>>(&self, values: &mut I) -> T {
        let a1 = self.args.0.consume(values);
        let a2 = self.args.1.consume(values);
        let a3 = self.args.2.consume(values);
        let a4 = self.args.3.consume(values);
        (self.f)(a1, a2, a3, a4)
    }
}

// ============================================================================
// Direct call methods for convenience
// ============================================================================

impl<F> CaptureArgs<F, (Arg, Arg)> {
    /// Call the captured function with two values.
    #[inline]
    pub fn call2<T>(&self, a: T, b: T) -> T
    where
        F: Fn(T, T) -> T,
    {
        (self.f)(a, b)
    }
}

impl<F> CaptureArgs<F, (Arg, Arg, Arg)> {
    /// Call the captured function with three values.
    #[inline]
    pub fn call3<T>(&self, a: T, b: T, c: T) -> T
    where
        F: Fn(T, T, T) -> T,
    {
        (self.f)(a, b, c)
    }
}

impl<F> CaptureArgs<F, (Arg, Arg, Arg, Arg)> {
    /// Call the captured function with four values.
    #[inline]
    pub fn call4<T>(&self, a: T, b: T, c: T, d: T) -> T
    where
        F: Fn(T, T, T, T) -> T,
    {
        (self.f)(a, b, c, d)
    }
}

// ============================================================================
// promoteshape: Broadcast arrays to common shape with stride-0
// ============================================================================

/// Promote a strided view to a new shape by setting stride-0 for broadcasted dimensions.
///
/// This is the core of Julia's broadcasting strategy:
/// - Dimensions that match the target size keep their stride
/// - Dimensions of size 1 get stride 0 (broadcast)
/// - Incompatible dimensions cause an error
///
/// # Julia equivalent
/// ```julia
/// function promoteshape1(sz::Dims{N}, a::StridedView) where {N}
///     newstrides = ntuple(Val(N)) do d
///         if size(a, d) == sz[d]
///             stride(a, d)
///         elseif size(a, d) == 1
///             0  # stride-0 broadcast
///         else
///             throw(DimensionMismatch(...))
///         end
///     end
///     return StridedView(a.parent, sz, newstrides, a.offset, a.op)
/// end
/// ```
pub fn promoteshape<'a, T, const N: usize, Op: ElementOp>(
    target_size: &[usize; N],
    view: &StridedArrayView<'a, T, N, Op>,
) -> Result<StridedArrayView<'a, T, N, Op>> {
    let promoted = promote_strides_to_shape(&target_size[..], &view.size()[..], &view.strides()[..])?;
    debug_assert_eq!(promoted.len(), N);
    let mut new_strides = [0isize; N];
    new_strides.copy_from_slice(&promoted);

    // Create new view with promoted strides
    // Safety: We're creating a view with stride-0 dimensions, which is valid
    // as long as we don't mutate and size-1 dims are promoted correctly
    Ok(unsafe {
        StridedArrayView::new_unchecked(view.data(), *target_size, new_strides, view.offset())
    })
}

/// Promote multiple views to a common broadcast shape.
///
/// Returns a tuple of promoted views, all with the same shape but potentially
/// with stride-0 dimensions for broadcasting.
pub fn promoteshape2<'a, T, const N: usize, Op1: ElementOp, Op2: ElementOp>(
    target_size: &[usize; N],
    a: &StridedArrayView<'a, T, N, Op1>,
    b: &StridedArrayView<'a, T, N, Op2>,
) -> Result<(
    StridedArrayView<'a, T, N, Op1>,
    StridedArrayView<'a, T, N, Op2>,
)> {
    let a_promoted = promoteshape(target_size, a)?;
    let b_promoted = promoteshape(target_size, b)?;
    Ok((a_promoted, b_promoted))
}

/// Promote three views to a common broadcast shape.
pub fn promoteshape3<'a, T, const N: usize, Op1: ElementOp, Op2: ElementOp, Op3: ElementOp>(
    target_size: &[usize; N],
    a: &StridedArrayView<'a, T, N, Op1>,
    b: &StridedArrayView<'a, T, N, Op2>,
    c: &StridedArrayView<'a, T, N, Op3>,
) -> PromoteShape3Result<'a, T, N, Op1, Op2, Op3> {
    let a_promoted = promoteshape(target_size, a)?;
    let b_promoted = promoteshape(target_size, b)?;
    let c_promoted = promoteshape(target_size, c)?;
    Ok((a_promoted, b_promoted, c_promoted))
}

// ============================================================================
// Broadcast execution
// ============================================================================

/// Execute a broadcast operation into a destination array.
///
/// This is the main entry point for broadcast operations. It:
/// 1. Promotes all source arrays to the destination shape
/// 2. Iterates over all elements using the kernel
/// 3. Applies the captured function at each element position
///
/// # Arguments
/// - `dest`: Destination array to write results
/// - `capture`: The captured broadcast operation
/// - `sources`: Source arrays (one per `Arg` in the capture)
///
/// # Julia equivalent
/// ```julia
/// @inline function Base.copyto!(dest::StridedView{<:Any,N},
///                               bc::Broadcasted{StridedArrayStyle{N}}) where {N}
///     stridedargs = promoteshape(size(dest), capturestridedargs(bc)...)
///     c = make_capture(bc)
///     _mapreduce_fuse!(c, nothing, nothing, size(dest), (dest, stridedargs...))
///     return dest
/// end
/// ```
pub fn broadcast_into<T, const N: usize, Op1, Op2, F>(
    dest: &mut StridedArrayViewMut<'_, T, N, Identity>,
    f: F,
    a: &StridedArrayView<'_, T, N, Op1>,
    b: &StridedArrayView<'_, T, N, Op2>,
) -> Result<()>
where
    T: Copy + ElementOpApply,
    Op1: ElementOp,
    Op2: ElementOp,
    F: Fn(T, T) -> T,
{
    // Create capture for the operation
    let capture = CaptureArgs::new(&f, (Arg, Arg));

    broadcast_capture2_into(dest, &capture, a, b)
}

/// Execute a captured 2-input broadcast expression.
///
/// This enables Julia-style nested broadcast expressions and scalar arguments:
/// `capture` can be a nested `CaptureArgs` tree, and is evaluated lazily per element.
pub fn broadcast_capture2_into<T, const N: usize, Op1, Op2, C>(
    dest: &mut StridedArrayViewMut<'_, T, N, Identity>,
    capture: &C,
    a: &StridedArrayView<'_, T, N, Op1>,
    b: &StridedArrayView<'_, T, N, Op2>,
) -> Result<()>
where
    T: Copy + ElementOpApply,
    Op1: ElementOp,
    Op2: ElementOp,
    C: Consume<T, Output = T>,
{
    let dims = *dest.size();
    let dst_strides = *dest.strides();
    let total = total_len(&dims);
    if total == 0 {
        return Ok(());
    }

    let a_promoted = promoteshape(&dims, a)?;
    let b_promoted = promoteshape(&dims, b)?;

    let all_contig = is_contiguous(&dims, &dst_strides)
        && is_contiguous(&dims, a_promoted.strides())
        && is_contiguous(&dims, b_promoted.strides());
    if all_contig {
        let mut dst_ptr = dest.as_mut_ptr();
        let mut p0 = a_promoted.as_ptr();
        let mut p1 = b_promoted.as_ptr();
        for _ in 0..total {
            let v0 = unsafe { Op1::apply(*p0) };
            let v1 = unsafe { Op2::apply(*p1) };
            p0 = unsafe { p0.add(1) };
            p1 = unsafe { p1.add(1) };
            let mut it = [v0, v1].into_iter();
            let out = capture.consume(&mut it);
            unsafe {
                *dst_ptr = out;
                dst_ptr = dst_ptr.add(1);
            }
        }
        return Ok(());
    }

    let strides_list = [&dst_strides[..], &a_promoted.strides()[..], &b_promoted.strides()[..]];
    let (fused_dims, plan) =
        build_plan_fused(&dims, &strides_list, Some(0), std::mem::size_of::<T>());

    let dst_base = dest.as_mut_ptr();
    let a_base = a_promoted.as_ptr();
    let b_base = b_promoted.as_ptr();

    for_each_inner_block(&fused_dims, &plan, &strides_list, |offsets, len, strides| {
        let mut dst_ptr = unsafe { dst_base.offset(offsets[0]) };
        let mut p0 = unsafe { a_base.offset(offsets[1]) };
        let mut p1 = unsafe { b_base.offset(offsets[2]) };
        let dst_step = strides[0];
        let s0 = strides[1];
        let s1 = strides[2];
        for _ in 0..len {
            let v0 = unsafe { Op1::apply(*p0) };
            let v1 = unsafe { Op2::apply(*p1) };
            p0 = unsafe { p0.offset(s0) };
            p1 = unsafe { p1.offset(s1) };
            let mut it = [v0, v1].into_iter();
            let out = capture.consume(&mut it);
            unsafe {
                *dst_ptr = out;
                dst_ptr = dst_ptr.offset(dst_step);
            }
        }
        Ok(())
    })
}

/// Execute a 3-way broadcast operation.
pub fn broadcast3_into<T, const N: usize, Op1, Op2, Op3, F>(
    dest: &mut StridedArrayViewMut<'_, T, N, Identity>,
    f: F,
    a: &StridedArrayView<'_, T, N, Op1>,
    b: &StridedArrayView<'_, T, N, Op2>,
    c: &StridedArrayView<'_, T, N, Op3>,
) -> Result<()>
where
    T: Copy + ElementOpApply,
    Op1: ElementOp,
    Op2: ElementOp,
    Op3: ElementOp,
    F: Fn(T, T, T) -> T,
{
    let capture = CaptureArgs::new(&f, (Arg, Arg, Arg));

    broadcast_capture3_into(dest, &capture, a, b, c)
}

/// Execute a captured 3-input broadcast expression.
pub fn broadcast_capture3_into<T, const N: usize, Op1, Op2, Op3, C>(
    dest: &mut StridedArrayViewMut<'_, T, N, Identity>,
    capture: &C,
    a: &StridedArrayView<'_, T, N, Op1>,
    b: &StridedArrayView<'_, T, N, Op2>,
    c: &StridedArrayView<'_, T, N, Op3>,
) -> Result<()>
where
    T: Copy + ElementOpApply,
    Op1: ElementOp,
    Op2: ElementOp,
    Op3: ElementOp,
    C: Consume<T, Output = T>,
{
    let dims = *dest.size();
    let dst_strides = *dest.strides();
    let total = total_len(&dims);
    if total == 0 {
        return Ok(());
    }

    let a_promoted = promoteshape(&dims, a)?;
    let b_promoted = promoteshape(&dims, b)?;
    let c_promoted = promoteshape(&dims, c)?;

    let all_contig = is_contiguous(&dims, &dst_strides)
        && is_contiguous(&dims, a_promoted.strides())
        && is_contiguous(&dims, b_promoted.strides())
        && is_contiguous(&dims, c_promoted.strides());
    if all_contig {
        let mut dst_ptr = dest.as_mut_ptr();
        let mut p0 = a_promoted.as_ptr();
        let mut p1 = b_promoted.as_ptr();
        let mut p2 = c_promoted.as_ptr();
        for _ in 0..total {
            let v0 = unsafe { Op1::apply(*p0) };
            let v1 = unsafe { Op2::apply(*p1) };
            let v2 = unsafe { Op3::apply(*p2) };
            p0 = unsafe { p0.add(1) };
            p1 = unsafe { p1.add(1) };
            p2 = unsafe { p2.add(1) };
            let mut it = [v0, v1, v2].into_iter();
            let out = capture.consume(&mut it);
            unsafe {
                *dst_ptr = out;
                dst_ptr = dst_ptr.add(1);
            }
        }
        return Ok(());
    }

    let strides_list = [
        &dst_strides[..],
        &a_promoted.strides()[..],
        &b_promoted.strides()[..],
        &c_promoted.strides()[..],
    ];
    let (fused_dims, plan) =
        build_plan_fused(&dims, &strides_list, Some(0), std::mem::size_of::<T>());

    let dst_base = dest.as_mut_ptr();
    let a_base = a_promoted.as_ptr();
    let b_base = b_promoted.as_ptr();
    let c_base = c_promoted.as_ptr();

    for_each_inner_block(&fused_dims, &plan, &strides_list, |offsets, len, strides| {
        let mut dst_ptr = unsafe { dst_base.offset(offsets[0]) };
        let mut p0 = unsafe { a_base.offset(offsets[1]) };
        let mut p1 = unsafe { b_base.offset(offsets[2]) };
        let mut p2 = unsafe { c_base.offset(offsets[3]) };
        let dst_step = strides[0];
        let s0 = strides[1];
        let s1 = strides[2];
        let s2 = strides[3];
        for _ in 0..len {
            let v0 = unsafe { Op1::apply(*p0) };
            let v1 = unsafe { Op2::apply(*p1) };
            let v2 = unsafe { Op3::apply(*p2) };
            p0 = unsafe { p0.offset(s0) };
            p1 = unsafe { p1.offset(s1) };
            p2 = unsafe { p2.offset(s2) };
            let mut it = [v0, v1, v2].into_iter();
            let out = capture.consume(&mut it);
            unsafe {
                *dst_ptr = out;
                dst_ptr = dst_ptr.offset(dst_step);
            }
        }
        Ok(())
    })
}

/// Execute a 4-way broadcast operation.
pub fn broadcast4_into<T, const N: usize, Op1, Op2, Op3, Op4, F>(
    dest: &mut StridedArrayViewMut<'_, T, N, Identity>,
    f: F,
    a: &StridedArrayView<'_, T, N, Op1>,
    b: &StridedArrayView<'_, T, N, Op2>,
    c: &StridedArrayView<'_, T, N, Op3>,
    d: &StridedArrayView<'_, T, N, Op4>,
) -> Result<()>
where
    T: Copy + ElementOpApply,
    Op1: ElementOp,
    Op2: ElementOp,
    Op3: ElementOp,
    Op4: ElementOp,
    F: Fn(T, T, T, T) -> T,
{
    let capture = CaptureArgs::new(&f, (Arg, Arg, Arg, Arg));

    broadcast_capture4_into(dest, &capture, a, b, c, d)
}

/// Execute a captured 4-input broadcast expression.
pub fn broadcast_capture4_into<T, const N: usize, Op1, Op2, Op3, Op4, C>(
    dest: &mut StridedArrayViewMut<'_, T, N, Identity>,
    capture: &C,
    a: &StridedArrayView<'_, T, N, Op1>,
    b: &StridedArrayView<'_, T, N, Op2>,
    c: &StridedArrayView<'_, T, N, Op3>,
    d: &StridedArrayView<'_, T, N, Op4>,
) -> Result<()>
where
    T: Copy + ElementOpApply,
    Op1: ElementOp,
    Op2: ElementOp,
    Op3: ElementOp,
    Op4: ElementOp,
    C: Consume<T, Output = T>,
{
    let dims = *dest.size();
    let dst_strides = *dest.strides();
    let total = total_len(&dims);
    if total == 0 {
        return Ok(());
    }

    let a_promoted = promoteshape(&dims, a)?;
    let b_promoted = promoteshape(&dims, b)?;
    let c_promoted = promoteshape(&dims, c)?;
    let d_promoted = promoteshape(&dims, d)?;

    let all_contig = is_contiguous(&dims, &dst_strides)
        && is_contiguous(&dims, a_promoted.strides())
        && is_contiguous(&dims, b_promoted.strides())
        && is_contiguous(&dims, c_promoted.strides())
        && is_contiguous(&dims, d_promoted.strides());
    if all_contig {
        let mut dst_ptr = dest.as_mut_ptr();
        let mut p0 = a_promoted.as_ptr();
        let mut p1 = b_promoted.as_ptr();
        let mut p2 = c_promoted.as_ptr();
        let mut p3 = d_promoted.as_ptr();
        for _ in 0..total {
            let v0 = unsafe { Op1::apply(*p0) };
            let v1 = unsafe { Op2::apply(*p1) };
            let v2 = unsafe { Op3::apply(*p2) };
            let v3 = unsafe { Op4::apply(*p3) };
            p0 = unsafe { p0.add(1) };
            p1 = unsafe { p1.add(1) };
            p2 = unsafe { p2.add(1) };
            p3 = unsafe { p3.add(1) };
            let mut it = [v0, v1, v2, v3].into_iter();
            let out = capture.consume(&mut it);
            unsafe {
                *dst_ptr = out;
                dst_ptr = dst_ptr.add(1);
            }
        }
        return Ok(());
    }

    let strides_list = [
        &dst_strides[..],
        &a_promoted.strides()[..],
        &b_promoted.strides()[..],
        &c_promoted.strides()[..],
        &d_promoted.strides()[..],
    ];
    let (fused_dims, plan) =
        build_plan_fused(&dims, &strides_list, Some(0), std::mem::size_of::<T>());

    let dst_base = dest.as_mut_ptr();
    let a_base = a_promoted.as_ptr();
    let b_base = b_promoted.as_ptr();
    let c_base = c_promoted.as_ptr();
    let d_base = d_promoted.as_ptr();

    for_each_inner_block(&fused_dims, &plan, &strides_list, |offsets, len, strides| {
        let mut dst_ptr = unsafe { dst_base.offset(offsets[0]) };
        let mut p0 = unsafe { a_base.offset(offsets[1]) };
        let mut p1 = unsafe { b_base.offset(offsets[2]) };
        let mut p2 = unsafe { c_base.offset(offsets[3]) };
        let mut p3 = unsafe { d_base.offset(offsets[4]) };
        let dst_step = strides[0];
        let s0 = strides[1];
        let s1 = strides[2];
        let s2 = strides[3];
        let s3 = strides[4];
        for _ in 0..len {
            let v0 = unsafe { Op1::apply(*p0) };
            let v1 = unsafe { Op2::apply(*p1) };
            let v2 = unsafe { Op3::apply(*p2) };
            let v3 = unsafe { Op4::apply(*p3) };
            p0 = unsafe { p0.offset(s0) };
            p1 = unsafe { p1.offset(s1) };
            p2 = unsafe { p2.offset(s2) };
            p3 = unsafe { p3.offset(s3) };
            let mut it = [v0, v1, v2, v3].into_iter();
            let out = capture.consume(&mut it);
            unsafe {
                *dst_ptr = out;
                dst_ptr = dst_ptr.offset(dst_step);
            }
        }
        Ok(())
    })
}

/// Execute a captured broadcast expression with an arbitrary number of source arrays.
///
/// This is the most Julia-like entry point: a capture tree (possibly nested and containing
/// `Scalar`s) is evaluated lazily at each element, consuming values from `sources` in order.
///
/// Note: if the capture expects more `Arg`s than provided in `sources`, evaluation will panic
/// (same behavior as `Arg::consume`).
pub fn broadcast_capture_into<'a, T, const N: usize, Op, C>(
    dest: &mut StridedArrayViewMut<'_, T, N, Identity>,
    capture: &C,
    sources: &[&StridedArrayView<'a, T, N, Op>],
) -> Result<()>
where
    T: Copy + ElementOpApply,
    Op: ElementOp,
    C: Consume<T, Output = T>,
{
    let dims = *dest.size();
    let dst_strides = *dest.strides();
    let total = total_len(&dims);
    if total == 0 {
        return Ok(());
    }

    // Special case: scalar-only capture, no sources.
    if sources.is_empty() {
        let mut empty = std::iter::empty::<T>();
        let out = capture.consume(&mut empty);

        if is_contiguous(&dims, &dst_strides) {
            let mut dst_ptr = dest.as_mut_ptr();
            for _ in 0..total {
                unsafe {
                    *dst_ptr = out;
                    dst_ptr = dst_ptr.add(1);
                }
            }
            return Ok(());
        }

        let strides_list = [&dst_strides[..]];
        let (fused_dims, plan) =
            build_plan_fused(&dims, &strides_list, Some(0), std::mem::size_of::<T>());

        let dst_base = dest.as_mut_ptr();
        return for_each_inner_block(&fused_dims, &plan, &strides_list, |offsets, len, strides| {
            let mut dst_ptr = unsafe { dst_base.offset(offsets[0]) };
            let dst_step = strides[0];
            for _ in 0..len {
                unsafe {
                    *dst_ptr = out;
                    dst_ptr = dst_ptr.offset(dst_step);
                }
            }
            Ok(())
        });
    }

    // Promote all sources to destination shape (stride-0 broadcast).
    let mut promoted: Vec<StridedArrayView<'a, T, N, Op>> = Vec::with_capacity(sources.len());
    for &src in sources {
        promoted.push(promoteshape(&dims, src)?);
    }

    // Fast path: all arrays contiguous.
    let all_contig = is_contiguous(&dims, &dst_strides)
        && promoted
            .iter()
            .all(|v| is_contiguous(&dims, v.strides()));
    if all_contig {
        let mut dst_ptr = dest.as_mut_ptr();
        match promoted.len() {
            1 => {
                let mut p0 = promoted[0].as_ptr();
                for _ in 0..total {
                    let v0 = unsafe { Op::apply(*p0) };
                    p0 = unsafe { p0.add(1) };
                    let mut it = [v0].into_iter();
                    let out = capture.consume(&mut it);
                    unsafe {
                        *dst_ptr = out;
                        dst_ptr = dst_ptr.add(1);
                    }
                }
                return Ok(());
            }
            2 => {
                let mut p0 = promoted[0].as_ptr();
                let mut p1 = promoted[1].as_ptr();
                for _ in 0..total {
                    let v0 = unsafe { Op::apply(*p0) };
                    let v1 = unsafe { Op::apply(*p1) };
                    p0 = unsafe { p0.add(1) };
                    p1 = unsafe { p1.add(1) };
                    let mut it = [v0, v1].into_iter();
                    let out = capture.consume(&mut it);
                    unsafe {
                        *dst_ptr = out;
                        dst_ptr = dst_ptr.add(1);
                    }
                }
                return Ok(());
            }
            3 => {
                let mut p0 = promoted[0].as_ptr();
                let mut p1 = promoted[1].as_ptr();
                let mut p2 = promoted[2].as_ptr();
                for _ in 0..total {
                    let v0 = unsafe { Op::apply(*p0) };
                    let v1 = unsafe { Op::apply(*p1) };
                    let v2 = unsafe { Op::apply(*p2) };
                    p0 = unsafe { p0.add(1) };
                    p1 = unsafe { p1.add(1) };
                    p2 = unsafe { p2.add(1) };
                    let mut it = [v0, v1, v2].into_iter();
                    let out = capture.consume(&mut it);
                    unsafe {
                        *dst_ptr = out;
                        dst_ptr = dst_ptr.add(1);
                    }
                }
                return Ok(());
            }
            4 => {
                let mut p0 = promoted[0].as_ptr();
                let mut p1 = promoted[1].as_ptr();
                let mut p2 = promoted[2].as_ptr();
                let mut p3 = promoted[3].as_ptr();
                for _ in 0..total {
                    let v0 = unsafe { Op::apply(*p0) };
                    let v1 = unsafe { Op::apply(*p1) };
                    let v2 = unsafe { Op::apply(*p2) };
                    let v3 = unsafe { Op::apply(*p3) };
                    p0 = unsafe { p0.add(1) };
                    p1 = unsafe { p1.add(1) };
                    p2 = unsafe { p2.add(1) };
                    p3 = unsafe { p3.add(1) };
                    let mut it = [v0, v1, v2, v3].into_iter();
                    let out = capture.consume(&mut it);
                    unsafe {
                        *dst_ptr = out;
                        dst_ptr = dst_ptr.add(1);
                    }
                }
                return Ok(());
            }
            _ => {
                let mut src_ptrs: Vec<*const T> = promoted.iter().map(|v| v.as_ptr()).collect();
                let mut values: Vec<T> = Vec::with_capacity(src_ptrs.len());

                for _ in 0..total {
                    values.clear();
                    for p in &mut src_ptrs {
                        unsafe {
                            values.push(Op::apply(**p));
                            *p = p.add(1);
                        }
                    }
                    let mut it = values.iter().copied();
                    let out = capture.consume(&mut it);
                    unsafe {
                        *dst_ptr = out;
                        dst_ptr = dst_ptr.add(1);
                    }
                }
                return Ok(());
            }
        }
    }

    // Generic blocked kernel path.
    let mut strides_list: Vec<&[isize]> = Vec::with_capacity(1 + promoted.len());
    strides_list.push(&dst_strides[..]);
    for src in &promoted {
        strides_list.push(&src.strides()[..]);
    }

    let (fused_dims, plan) =
        build_plan_fused(&dims, &strides_list, Some(0), std::mem::size_of::<T>());

    let dst_base = dest.as_mut_ptr();
    let src_bases: Vec<*const T> = promoted.iter().map(|v| v.as_ptr()).collect();
    let num_src = src_bases.len();

    for_each_inner_block(&fused_dims, &plan, &strides_list, |offsets, len, strides| {
        let mut dst_ptr = unsafe { dst_base.offset(offsets[0]) };
        let dst_step = strides[0];

        match num_src {
            1 => {
                let mut p0 = unsafe { src_bases[0].offset(offsets[1]) };
                let s0 = strides[1];
                for _ in 0..len {
                    let v0 = unsafe { Op::apply(*p0) };
                    p0 = unsafe { p0.offset(s0) };
                    let mut it = [v0].into_iter();
                    let out = capture.consume(&mut it);
                    unsafe {
                        *dst_ptr = out;
                        dst_ptr = dst_ptr.offset(dst_step);
                    }
                }
            }
            2 => {
                let mut p0 = unsafe { src_bases[0].offset(offsets[1]) };
                let mut p1 = unsafe { src_bases[1].offset(offsets[2]) };
                let s0 = strides[1];
                let s1 = strides[2];
                for _ in 0..len {
                    let v0 = unsafe { Op::apply(*p0) };
                    let v1 = unsafe { Op::apply(*p1) };
                    p0 = unsafe { p0.offset(s0) };
                    p1 = unsafe { p1.offset(s1) };
                    let mut it = [v0, v1].into_iter();
                    let out = capture.consume(&mut it);
                    unsafe {
                        *dst_ptr = out;
                        dst_ptr = dst_ptr.offset(dst_step);
                    }
                }
            }
            3 => {
                let mut p0 = unsafe { src_bases[0].offset(offsets[1]) };
                let mut p1 = unsafe { src_bases[1].offset(offsets[2]) };
                let mut p2 = unsafe { src_bases[2].offset(offsets[3]) };
                let s0 = strides[1];
                let s1 = strides[2];
                let s2 = strides[3];
                for _ in 0..len {
                    let v0 = unsafe { Op::apply(*p0) };
                    let v1 = unsafe { Op::apply(*p1) };
                    let v2 = unsafe { Op::apply(*p2) };
                    p0 = unsafe { p0.offset(s0) };
                    p1 = unsafe { p1.offset(s1) };
                    p2 = unsafe { p2.offset(s2) };
                    let mut it = [v0, v1, v2].into_iter();
                    let out = capture.consume(&mut it);
                    unsafe {
                        *dst_ptr = out;
                        dst_ptr = dst_ptr.offset(dst_step);
                    }
                }
            }
            4 => {
                let mut p0 = unsafe { src_bases[0].offset(offsets[1]) };
                let mut p1 = unsafe { src_bases[1].offset(offsets[2]) };
                let mut p2 = unsafe { src_bases[2].offset(offsets[3]) };
                let mut p3 = unsafe { src_bases[3].offset(offsets[4]) };
                let s0 = strides[1];
                let s1 = strides[2];
                let s2 = strides[3];
                let s3 = strides[4];
                for _ in 0..len {
                    let v0 = unsafe { Op::apply(*p0) };
                    let v1 = unsafe { Op::apply(*p1) };
                    let v2 = unsafe { Op::apply(*p2) };
                    let v3 = unsafe { Op::apply(*p3) };
                    p0 = unsafe { p0.offset(s0) };
                    p1 = unsafe { p1.offset(s1) };
                    p2 = unsafe { p2.offset(s2) };
                    p3 = unsafe { p3.offset(s3) };
                    let mut it = [v0, v1, v2, v3].into_iter();
                    let out = capture.consume(&mut it);
                    unsafe {
                        *dst_ptr = out;
                        dst_ptr = dst_ptr.offset(dst_step);
                    }
                }
            }
            _ => {
                let mut values: Vec<T> = Vec::with_capacity(num_src);
                let mut src_ptrs: Vec<*const T> = Vec::with_capacity(num_src);
                let mut src_steps: Vec<isize> = Vec::with_capacity(num_src);

                src_ptrs.clear();
                src_steps.clear();
                for i in 0..num_src {
                    src_ptrs.push(unsafe { src_bases[i].offset(offsets[i + 1]) });
                    src_steps.push(strides[i + 1]);
                }

                for _ in 0..len {
                    values.clear();
                    for i in 0..num_src {
                        unsafe {
                            values.push(Op::apply(*src_ptrs[i]));
                            src_ptrs[i] = src_ptrs[i].offset(src_steps[i]);
                        }
                    }
                    let mut it = values.iter().copied();
                    let out = capture.consume(&mut it);
                    unsafe {
                        *dst_ptr = out;
                        dst_ptr = dst_ptr.offset(dst_step);
                    }
                }
            }
        }

        Ok(())
    })
}

// ============================================================================
// Builder pattern for complex broadcast expressions
// ============================================================================

/// Builder for constructing complex broadcast expressions.
///
/// This allows building nested CaptureArgs with a fluent interface.
pub struct BroadcastBuilder<T> {
    _phantom: PhantomData<T>,
}

impl<T> BroadcastBuilder<T> {
    /// Create a new broadcast builder.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Create a binary operation capture.
    pub fn binary<F: Fn(T, T) -> T>(f: F) -> CaptureArgs<F, (Arg, Arg)> {
        CaptureArgs::new(f, (Arg, Arg))
    }

    /// Create a ternary operation capture.
    pub fn ternary<F: Fn(T, T, T) -> T>(f: F) -> CaptureArgs<F, (Arg, Arg, Arg)> {
        CaptureArgs::new(f, (Arg, Arg, Arg))
    }

    /// Create a quaternary operation capture.
    pub fn quaternary<F: Fn(T, T, T, T) -> T>(f: F) -> CaptureArgs<F, (Arg, Arg, Arg, Arg)> {
        CaptureArgs::new(f, (Arg, Arg, Arg, Arg))
    }

    /// Create a unary operation with a scalar.
    pub fn unary_scalar<F: Fn(T, T) -> T>(f: F, scalar: T) -> CaptureArgs<F, (Arg, Scalar<T>)> {
        CaptureArgs::new(f, (Arg, Scalar(scalar)))
    }
}

impl<T> Default for BroadcastBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element_op::{Conj, Identity};
    use crate::view::StridedArrayView;
    use num_complex::Complex;

    #[test]
    fn test_arg_consume() {
        let arg = Arg;
        let mut values = vec![1.0f64, 2.0, 3.0].into_iter();

        let v1 = arg.consume(&mut values);
        let v2 = arg.consume(&mut values);
        let v3 = arg.consume(&mut values);

        assert_eq!(v1, 1.0);
        assert_eq!(v2, 2.0);
        assert_eq!(v3, 3.0);
    }

    #[test]
    fn test_scalar_consume() {
        let scalar = Scalar(42.0f64);
        let mut values = vec![1.0, 2.0, 3.0].into_iter();

        // Scalar doesn't consume from iterator
        let v1 = scalar.consume(&mut values);
        let v2 = scalar.consume(&mut values);

        assert_eq!(v1, 42.0);
        assert_eq!(v2, 42.0);

        // Values should still be available
        assert_eq!(values.next(), Some(1.0));
    }

    #[test]
    fn test_capture_args_binary() {
        let capture = CaptureArgs::new(|a: f64, b: f64| a + b, (Arg, Arg));
        let mut values = vec![1.0, 2.0].into_iter();

        let result: f64 = capture.consume(&mut values);
        assert_eq!(result, 3.0);
    }

    #[test]
    fn test_capture_args_with_scalar() {
        let capture = CaptureArgs::new(|a: f64, s: f64| a * s, (Arg, Scalar(2.0)));
        let mut values = vec![3.0f64].into_iter();

        let result: f64 = capture.consume(&mut values);
        assert_eq!(result, 6.0);
    }

    #[test]
    fn test_capture_args_ternary() {
        // fma: a + b * c
        let capture = CaptureArgs::new(|a: f64, b: f64, c: f64| a + b * c, (Arg, Arg, Arg));
        let mut values = vec![1.0, 2.0, 3.0].into_iter();

        let result: f64 = capture.consume(&mut values);
        assert_eq!(result, 7.0); // 1 + 2 * 3
    }

    #[test]
    fn test_nested_capture_args() {
        // Nested: f(g(a, b), c) where g(a, b) = a * b, f(x, c) = x + c
        let inner = CaptureArgs::new(|a: f64, b: f64| a * b, (Arg, Arg));
        let outer = CaptureArgs::new(|x: f64, c: f64| x + c, (inner, Arg));

        let mut values = vec![2.0, 3.0, 4.0].into_iter();
        let result: f64 = outer.consume(&mut values);
        assert_eq!(result, 10.0); // (2 * 3) + 4
    }

    #[test]
    fn test_promoteshape_same_size() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&data, [2, 3], [3, 1], 0).unwrap();

        let target = [2, 3];
        let promoted = promoteshape(&target, &view).unwrap();

        assert_eq!(promoted.size(), &[2, 3]);
        assert_eq!(promoted.strides(), &[3, 1]); // Unchanged
    }

    #[test]
    fn test_promoteshape_broadcast_row() {
        // Row vector [1, 3] broadcast to [4, 3]
        let data = vec![1.0, 2.0, 3.0];
        let view: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&data, [1, 3], [3, 1], 0).unwrap();

        let target = [4, 3];
        let promoted = promoteshape(&target, &view).unwrap();

        assert_eq!(promoted.size(), &[4, 3]);
        assert_eq!(promoted.strides(), &[0, 1]); // First dim has stride 0

        // All rows should have the same values
        for i in 0..4 {
            assert_eq!(promoted.get([i, 0]), 1.0);
            assert_eq!(promoted.get([i, 1]), 2.0);
            assert_eq!(promoted.get([i, 2]), 3.0);
        }
    }

    #[test]
    fn test_promoteshape_broadcast_col() {
        // Column vector [4, 1] broadcast to [4, 3]
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let view: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&data, [4, 1], [1, 1], 0).unwrap();

        let target = [4, 3];
        let promoted = promoteshape(&target, &view).unwrap();

        assert_eq!(promoted.size(), &[4, 3]);
        assert_eq!(promoted.strides(), &[1, 0]); // Second dim has stride 0

        // All columns should have the same values
        for j in 0..3 {
            assert_eq!(promoted.get([0, j]), 1.0);
            assert_eq!(promoted.get([1, j]), 2.0);
            assert_eq!(promoted.get([2, j]), 3.0);
            assert_eq!(promoted.get([3, j]), 4.0);
        }
    }

    #[test]
    fn test_promoteshape_incompatible() {
        let data = vec![1.0, 2.0, 3.0];
        let view: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&data, [1, 3], [3, 1], 0).unwrap();

        // Cannot broadcast [1, 3] to [4, 5] - inner dim doesn't match
        let target = [4, 5];
        let result = promoteshape(&target, &view);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_into_same_shape() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let mut dest_data = vec![0.0; 6];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [2, 3], [3, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [2, 3], [3, 1], 0).unwrap();
        let mut dest: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut dest_data, [2, 3], [3, 1], 0).unwrap();

        broadcast_into(&mut dest, |x, y| x + y, &a, &b).unwrap();

        assert_eq!(dest.get([0, 0]), 11.0);
        assert_eq!(dest.get([0, 1]), 22.0);
        assert_eq!(dest.get([1, 2]), 66.0);
    }

    #[test]
    fn test_broadcast_into_with_broadcast() {
        // A: [2, 3], B: [1, 3] -> broadcast B to [2, 3]
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![10.0, 20.0, 30.0];
        let mut dest_data = vec![0.0; 6];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [2, 3], [3, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [1, 3], [3, 1], 0).unwrap();
        let mut dest: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut dest_data, [2, 3], [3, 1], 0).unwrap();

        broadcast_into(&mut dest, |x, y| x + y, &a, &b).unwrap();

        // Row 0: [1, 2, 3] + [10, 20, 30] = [11, 22, 33]
        // Row 1: [4, 5, 6] + [10, 20, 30] = [14, 25, 36]
        assert_eq!(dest.get([0, 0]), 11.0);
        assert_eq!(dest.get([0, 1]), 22.0);
        assert_eq!(dest.get([0, 2]), 33.0);
        assert_eq!(dest.get([1, 0]), 14.0);
        assert_eq!(dest.get([1, 1]), 25.0);
        assert_eq!(dest.get([1, 2]), 36.0);
    }

    #[test]
    fn test_broadcast_builder() {
        let add = BroadcastBuilder::<f64>::binary(|a, b| a + b);
        assert_eq!(add.call2(1.0, 2.0), 3.0);

        let fma = BroadcastBuilder::<f64>::ternary(|a, b, c| a + b * c);
        assert_eq!(fma.call3(1.0, 2.0, 3.0), 7.0);

        let scale = BroadcastBuilder::unary_scalar(|a: f64, s: f64| a * s, 2.0);
        let mut values = vec![3.0f64].into_iter();
        let result: f64 = scale.consume(&mut values);
        assert_eq!(result, 6.0);
    }

    #[test]
    fn test_broadcast3_into() {
        // A: [2, 3], B: [2, 3], C: [2, 3] -> A + B * C
        let a_data: Vec<f64> = (1..=6).map(|x| x as f64).collect();
        let b_data: Vec<f64> = (10..=15).map(|x| x as f64).collect();
        let c_data = vec![2.0; 6];
        let mut dest_data = vec![0.0; 6];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [2, 3], [3, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [2, 3], [3, 1], 0).unwrap();
        let c: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&c_data, [2, 3], [3, 1], 0).unwrap();
        let mut dest: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut dest_data, [2, 3], [3, 1], 0).unwrap();

        broadcast3_into(&mut dest, |x, y, z| x + y * z, &a, &b, &c).unwrap();

        // Element [0, 0]: 1 + 10 * 2 = 21
        // Element [0, 1]: 2 + 11 * 2 = 24
        // Element [1, 2]: 6 + 15 * 2 = 36
        assert_eq!(dest.get([0, 0]), 21.0);
        assert_eq!(dest.get([0, 1]), 24.0);
        assert_eq!(dest.get([1, 2]), 36.0);
    }

    #[test]
    fn test_broadcast4_into() {
        // A,B,C,D: [2, 2] -> A + B + C + D
        let a_data: Vec<f64> = (1..=4).map(|x| x as f64).collect();
        let b_data = vec![10.0; 4];
        let c_data = vec![100.0; 4];
        let d_data = vec![1000.0; 4];
        let mut dest_data = vec![0.0; 4];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [2, 2], [2, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [2, 2], [2, 1], 0).unwrap();
        let c: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&c_data, [2, 2], [2, 1], 0).unwrap();
        let d: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&d_data, [2, 2], [2, 1], 0).unwrap();
        let mut dest: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut dest_data, [2, 2], [2, 1], 0).unwrap();

        broadcast4_into(&mut dest, |x, y, z, w| x + y + z + w, &a, &b, &c, &d).unwrap();

        // Element [0,0]: 1 + 10 + 100 + 1000
        assert_eq!(dest.get([0, 0]), 1111.0);
        // Element [1,1]: 4 + 10 + 100 + 1000
        assert_eq!(dest.get([1, 1]), 1114.0);
    }

    #[test]
    fn test_broadcast_capture2_nested_with_scalar() {
        // dest = (a * b) + 1
        let a_data: Vec<f64> = (1..=6).map(|x| x as f64).collect();
        let b_data = vec![2.0; 6];
        let mut dest_data = vec![0.0; 6];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [2, 3], [3, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [2, 3], [3, 1], 0).unwrap();
        let mut dest: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut dest_data, [2, 3], [3, 1], 0).unwrap();

        let inner = CaptureArgs::new(|x: f64, y: f64| x * y, (Arg, Arg));
        let outer = CaptureArgs::new(|p: f64, s: f64| p + s, (inner, Scalar(1.0)));

        broadcast_capture2_into(&mut dest, &outer, &a, &b).unwrap();

        assert_eq!(dest.get([0, 0]), 1.0 * 2.0 + 1.0);
        assert_eq!(dest.get([1, 2]), 6.0 * 2.0 + 1.0);
    }

    #[test]
    fn test_broadcast_capture_into_two_sources() {
        // Same as broadcast_capture2_into but through the arbitrary-N API.
        let a_data: Vec<f64> = (1..=6).map(|x| x as f64).collect();
        let b_data = vec![2.0; 6];
        let mut dest_data = vec![0.0; 6];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [2, 3], [3, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [2, 3], [3, 1], 0).unwrap();
        let mut dest: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut dest_data, [2, 3], [3, 1], 0).unwrap();

        let inner = CaptureArgs::new(|x: f64, y: f64| x * y, (Arg, Arg));
        let outer = CaptureArgs::new(|p: f64, s: f64| p + s, (inner, Scalar(1.0)));

        broadcast_capture_into(&mut dest, &outer, &[&a, &b]).unwrap();

        assert_eq!(dest.get([0, 0]), 1.0 * 2.0 + 1.0);
        assert_eq!(dest.get([1, 2]), 6.0 * 2.0 + 1.0);
    }

    #[test]
    fn test_broadcast_capture_into_scalar_only() {
        // Fill dest with a scalar through capture evaluation.
        let mut dest_data = vec![0.0; 6];
        let mut dest: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut dest_data, [2, 3], [3, 1], 0).unwrap();

        let capture = Scalar(3.5f64);
        broadcast_capture_into::<f64, 2, Identity, _>(&mut dest, &capture, &[]).unwrap();

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(dest.get([i, j]), 3.5);
            }
        }
    }

    #[test]
    fn test_broadcast_1d() {
        // Simple 1D broadcast
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![10.0, 20.0, 30.0, 40.0];
        let mut dest_data = vec![0.0; 4];

        let a: StridedArrayView<'_, f64, 1, Identity> =
            StridedArrayView::new(&a_data, [4], [1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 1, Identity> =
            StridedArrayView::new(&b_data, [4], [1], 0).unwrap();
        let mut dest: StridedArrayViewMut<'_, f64, 1, Identity> =
            StridedArrayViewMut::new(&mut dest_data, [4], [1], 0).unwrap();

        broadcast_into(&mut dest, |x, y| x * y, &a, &b).unwrap();

        assert_eq!(dest.get([0]), 10.0);
        assert_eq!(dest.get([1]), 40.0);
        assert_eq!(dest.get([2]), 90.0);
        assert_eq!(dest.get([3]), 160.0);
    }

    #[test]
    fn test_broadcast_capture_into_with_conj() {
        let data = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, -1.0),
            Complex::new(5.0, 4.0),
            Complex::new(7.0, -2.0),
        ];
        let mut out_data = vec![Complex::new(0.0, 0.0); 4];

        let src: StridedArrayView<'_, Complex<f64>, 1, Conj> =
            StridedArrayView::<Complex<f64>, 1, Identity>::new(&data, [4], [1], 0)
                .unwrap()
                .conj();
        let mut dest = StridedArrayViewMut::new(&mut out_data, [4], [1], 0).unwrap();

        let capture = CaptureArgs::new(|x: Complex<f64>| x, (Arg,));
        broadcast_capture_into(&mut dest, &capture, &[&src]).unwrap();

        assert_eq!(dest.get([0]), Complex::new(1.0, -2.0));
        assert_eq!(dest.get([1]), Complex::new(3.0, 1.0));
        assert_eq!(dest.get([2]), Complex::new(5.0, -4.0));
        assert_eq!(dest.get([3]), Complex::new(7.0, 2.0));
    }

    #[test]
    fn test_broadcast_capture2_into_mixed_ops() {
        let a_data = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, -1.0),
            Complex::new(5.0, 4.0),
            Complex::new(7.0, -2.0),
        ];
        let b_data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let mut out = vec![Complex::new(0.0, 0.0); 4];

        let a: StridedArrayView<'_, Complex<f64>, 1, Conj> =
            StridedArrayView::<Complex<f64>, 1, Identity>::new(&a_data, [4], [1], 0)
                .unwrap()
                .conj();
        let b: StridedArrayView<'_, Complex<f64>, 1, Identity> =
            StridedArrayView::new(&b_data, [4], [1], 0).unwrap();
        let mut dest: StridedArrayViewMut<'_, Complex<f64>, 1, Identity> =
            StridedArrayViewMut::new(&mut out, [4], [1], 0).unwrap();

        let capture = CaptureArgs::new(|x: Complex<f64>, y: Complex<f64>| x + y, (Arg, Arg));
        broadcast_capture2_into(&mut dest, &capture, &a, &b).unwrap();

        assert_eq!(dest.get([0]), Complex::new(2.0, -2.0));
        assert_eq!(dest.get([1]), Complex::new(5.0, 1.0));
        assert_eq!(dest.get([2]), Complex::new(8.0, -4.0));
        assert_eq!(dest.get([3]), Complex::new(11.0, 2.0));
    }

    #[test]
    fn test_broadcast_capture3_into_mixed_ops() {
        let a_data = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, -1.0),
            Complex::new(5.0, 4.0),
            Complex::new(7.0, -2.0),
        ];
        let b_data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let c_data = vec![
            Complex::new(2.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(2.0, 0.0),
        ];
        let mut out = vec![Complex::new(0.0, 0.0); 4];

        let a: StridedArrayView<'_, Complex<f64>, 1, Conj> =
            StridedArrayView::<Complex<f64>, 1, Identity>::new(&a_data, [4], [1], 0)
                .unwrap()
                .conj();
        let b: StridedArrayView<'_, Complex<f64>, 1, Identity> =
            StridedArrayView::new(&b_data, [4], [1], 0).unwrap();
        let c: StridedArrayView<'_, Complex<f64>, 1, Identity> =
            StridedArrayView::new(&c_data, [4], [1], 0).unwrap();
        let mut dest: StridedArrayViewMut<'_, Complex<f64>, 1, Identity> =
            StridedArrayViewMut::new(&mut out, [4], [1], 0).unwrap();

        let capture =
            CaptureArgs::new(|x: Complex<f64>, y: Complex<f64>, z: Complex<f64>| x + y + z, (Arg, Arg, Arg));
        broadcast_capture3_into(&mut dest, &capture, &a, &b, &c).unwrap();

        assert_eq!(dest.get([0]), Complex::new(4.0, -2.0));
        assert_eq!(dest.get([1]), Complex::new(7.0, 1.0));
        assert_eq!(dest.get([2]), Complex::new(10.0, -4.0));
        assert_eq!(dest.get([3]), Complex::new(13.0, 2.0));
    }

    #[test]
    fn test_broadcast_capture4_into_mixed_ops() {
        let a_data = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, -1.0),
            Complex::new(5.0, 4.0),
            Complex::new(7.0, -2.0),
        ];
        let b_data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let c_data = vec![
            Complex::new(2.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(2.0, 0.0),
        ];
        let d_data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
        ];
        let mut out = vec![Complex::new(0.0, 0.0); 4];

        let a: StridedArrayView<'_, Complex<f64>, 1, Conj> =
            StridedArrayView::<Complex<f64>, 1, Identity>::new(&a_data, [4], [1], 0)
                .unwrap()
                .conj();
        let b: StridedArrayView<'_, Complex<f64>, 1, Identity> =
            StridedArrayView::new(&b_data, [4], [1], 0).unwrap();
        let c: StridedArrayView<'_, Complex<f64>, 1, Identity> =
            StridedArrayView::new(&c_data, [4], [1], 0).unwrap();
        let d: StridedArrayView<'_, Complex<f64>, 1, Identity> =
            StridedArrayView::new(&d_data, [4], [1], 0).unwrap();
        let mut dest: StridedArrayViewMut<'_, Complex<f64>, 1, Identity> =
            StridedArrayViewMut::new(&mut out, [4], [1], 0).unwrap();

        let capture = CaptureArgs::new(
            |x: Complex<f64>, y: Complex<f64>, z: Complex<f64>, w: Complex<f64>| x + y + z + w,
            (Arg, Arg, Arg, Arg),
        );
        broadcast_capture4_into(&mut dest, &capture, &a, &b, &c, &d).unwrap();

        assert_eq!(dest.get([0]), Complex::new(5.0, -2.0));
        assert_eq!(dest.get([1]), Complex::new(8.0, 1.0));
        assert_eq!(dest.get([2]), Complex::new(11.0, -4.0));
        assert_eq!(dest.get([3]), Complex::new(14.0, 2.0));
    }

    #[test]
    fn test_broadcast_scalar_to_all() {
        // Broadcast [1, 1] to [3, 4]
        let a_data = vec![5.0];
        let b_data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let mut dest_data = vec![0.0; 12];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [1, 1], [1, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [3, 4], [4, 1], 0).unwrap();
        let mut dest: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut dest_data, [3, 4], [4, 1], 0).unwrap();

        broadcast_into(&mut dest, |x, y| x + y, &a, &b).unwrap();

        // All elements should be b + 5
        for i in 0..3 {
            for j in 0..4 {
                let expected = (i * 4 + j + 1) as f64 + 5.0;
                assert_eq!(dest.get([i, j]), expected);
            }
        }
    }

    #[test]
    fn test_promoteshape_3d() {
        // Test 3D broadcasting: [1, 3, 1] -> [2, 3, 4]
        let data = vec![1.0, 2.0, 3.0];
        let view: StridedArrayView<'_, f64, 3, Identity> =
            StridedArrayView::new(&data, [1, 3, 1], [3, 1, 1], 0).unwrap();

        let target = [2, 3, 4];
        let promoted = promoteshape(&target, &view).unwrap();

        assert_eq!(promoted.size(), &[2, 3, 4]);
        assert_eq!(promoted.strides(), &[0, 1, 0]); // First and third dims have stride 0

        // Verify broadcasting works correctly
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let expected = (j + 1) as f64; // Only middle dimension varies
                    assert_eq!(promoted.get([i, j, k]), expected);
                }
            }
        }
    }

    #[test]
    fn test_capture_args_quaternary() {
        // Test 4-argument capture: a + b * c - d
        let capture = CaptureArgs::new(
            |a: f64, b: f64, c: f64, d: f64| a + b * c - d,
            (Arg, Arg, Arg, Arg),
        );
        let mut values = vec![1.0, 2.0, 3.0, 4.0].into_iter();

        let result: f64 = capture.consume(&mut values);
        assert_eq!(result, 3.0); // 1 + 2 * 3 - 4 = 3
    }

    #[test]
    fn test_deeply_nested_capture_args() {
        // Three levels of nesting: outer(middle(inner(a, b), c), d)
        // inner(a, b) = a + b
        // middle(x, c) = x * c
        // outer(y, d) = y - d
        let inner = CaptureArgs::new(|a: f64, b: f64| a + b, (Arg, Arg));
        let middle = CaptureArgs::new(|x: f64, c: f64| x * c, (inner, Arg));
        let outer = CaptureArgs::new(|y: f64, d: f64| y - d, (middle, Arg));

        let mut values = vec![1.0, 2.0, 3.0, 4.0].into_iter();
        let result: f64 = outer.consume(&mut values);
        // inner(1, 2) = 3
        // middle(3, 3) = 9
        // outer(9, 4) = 5
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_broadcast_empty_array() {
        // Empty array should be handled gracefully
        let a_data: Vec<f64> = vec![];
        let b_data: Vec<f64> = vec![];
        let mut dest_data: Vec<f64> = vec![];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [0, 3], [3, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [0, 3], [3, 1], 0).unwrap();
        let mut dest: StridedArrayViewMut<'_, f64, 2, Identity> =
            StridedArrayViewMut::new(&mut dest_data, [0, 3], [3, 1], 0).unwrap();

        // Should succeed without doing anything
        broadcast_into(&mut dest, |x, y| x + y, &a, &b).unwrap();
    }

    #[test]
    fn test_promoteshape2_both_broadcast() {
        // Both arrays need broadcasting: [1, 3] and [4, 1] -> [4, 3]
        let a_data = vec![1.0, 2.0, 3.0];
        let b_data = vec![10.0, 20.0, 30.0, 40.0];

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [1, 3], [3, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [4, 1], [1, 1], 0).unwrap();

        let target = [4, 3];
        let (a_promoted, b_promoted) = promoteshape2(&target, &a, &b).unwrap();

        assert_eq!(a_promoted.size(), &[4, 3]);
        assert_eq!(a_promoted.strides(), &[0, 1]);
        assert_eq!(b_promoted.size(), &[4, 3]);
        assert_eq!(b_promoted.strides(), &[1, 0]);

        // Verify the outer product pattern
        for i in 0..4 {
            for j in 0..3 {
                assert_eq!(a_promoted.get([i, j]), (j + 1) as f64);
                assert_eq!(b_promoted.get([i, j]), ((i + 1) * 10) as f64);
            }
        }
    }
}
