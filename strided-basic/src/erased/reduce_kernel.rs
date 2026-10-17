//! Per-op reduction kernels for [`ErasedReducePlan`](super::ErasedReducePlan).
//!
//! The runtime [`ReduceOp`](super::ReduceOp) is matched once per plan
//! execution and selects one zero-sized kernel type. Every tensor-sized loop
//! below is generic over that kernel type, so the op never appears inside a
//! loop body and each loop is compiled once per `(dtype, op)` pair.
//!
//! Serial execution and every parallel worker range call the same range
//! functions ([`reduce_full_range`] and [`reduce_axes_range`]), so a
//! parallel chunk runs exactly the lane or SIMD kernel the serial path runs.
//!
//! A kernel declares both its scalar accumulator (`Self::Acc`) and the kernel
//! that must recompute an output in source order when that accumulator cannot
//! represent the reduction ([`ReduceKernel::needs_ordered_fallback`]). The
//! stable product uses the second hook for its `zero`/subnormal/`inf`/`NaN`
//! fallback; for every other kernel the hook is a compile-time `false` and the
//! fallback kernel is the kernel itself.

use super::{
    checked_reduce_reset, compress_reduce_outer_axes, ErasedReduceScalar, ReduceInnerAxis,
    ReduceInnerCursor, ReduceOuterAxis, ReduceOuterCursor,
};
use crate::{Result, StridedError};
use core::ops::Range;

/// Independent accumulators used by the non SIMD lane kernels.
pub(super) const REDUCE_LANES: usize = 8;

/// Minimum length of a unit-stride leading reduced run that switches an axis
/// reduction from the sequential fold to the contiguous run kernel.
pub(super) const AXIS_RUN_KERNEL_MIN_LEN: usize = 2 * REDUCE_LANES;

/// Minimum extent of a unit-stride leading kept axis that selects the
/// output block kernel.
pub(super) const AXIS_OUTPUT_BLOCK_MIN_LEN: usize = 16;

/// Byte budget of the output block kernel's accumulator block.
///
/// Each reduced step streams one contiguous source run as long as the block,
/// so a longer block gives the hardware prefetcher longer runs between the
/// jumps of the reduced stride; 64 KiB still stays in L1 or L2 on current
/// cores.
pub(super) const AXIS_OUTPUT_BLOCK_BYTES: usize = 64 * 1024;

/// Number of reduced steps the output block kernel folds per pass over the
/// accumulator block, which divides accumulator load and store traffic by
/// this factor without changing the per output order.
pub(super) const AXIS_OUTPUT_BLOCK_STEPS: usize = 4;

/// Number of adjacent outputs one accumulator block holds for `T`.
fn axis_output_block_len<T>() -> usize {
    (AXIS_OUTPUT_BLOCK_BYTES / core::mem::size_of::<T>().max(1)).max(AXIS_OUTPUT_BLOCK_MIN_LEN)
}

/// Runs `f` with a per-thread reused accumulator block of `len` elements.
///
/// The block kernel needs `axis_output_block_len::<T>()` accumulators (up to
/// `AXIS_OUTPUT_BLOCK_BYTES / size_of::<T>()`, several thousand), so an inline
/// stack buffer is too large. The buffer is grown once per worker thread and
/// reused by every later call on that thread, so a steady-state axis reduction
/// allocates nothing (#306 item 5).
///
/// `init` fills every slot before `f` sees the slice; `Acc` is `Copy`, so the
/// buffer needs no drop.
fn with_axis_block_scratch<Acc: Copy, R>(
    len: usize,
    init: Acc,
    f: impl FnOnce(&mut [Acc]) -> R,
) -> R {
    use std::cell::RefCell;
    debug_assert!(core::mem::align_of::<Acc>() <= core::mem::align_of::<u128>());
    thread_local! {
        static SCRATCH: RefCell<Vec<core::mem::MaybeUninit<u128>>> =
            const { RefCell::new(Vec::new()) };
    }
    SCRATCH.with(|cell| {
        let mut buffer = cell.borrow_mut();
        let bytes = len
            .checked_mul(core::mem::size_of::<Acc>())
            .unwrap_or(usize::MAX);
        let words = bytes.div_ceil(core::mem::size_of::<u128>());
        if buffer.len() < words {
            buffer.resize(words, core::mem::MaybeUninit::uninit());
        }
        // SAFETY: no element is read before `init` is written into it
        // (references to `MaybeUninit` carry no validity requirement), the
        // `u128` buffer is aligned for `Acc`, and the allocation covers
        // `len * size_of::<Acc>()` bytes.
        let uninit = unsafe {
            core::slice::from_raw_parts_mut(
                buffer.as_mut_ptr().cast::<core::mem::MaybeUninit<Acc>>(),
                len,
            )
        };
        for slot in uninit.iter_mut() {
            slot.write(init);
        }
        // SAFETY: every slot was initialized just above.
        let acc =
            unsafe { core::slice::from_raw_parts_mut(uninit.as_mut_ptr().cast::<Acc>(), len) };
        f(acc)
    })
}

/// One reduction operation, fixed at compile time.
pub(super) trait ReduceKernel<T: ErasedReduceScalar>: 'static {
    /// Accumulator carried through the range kernels.
    type Acc: Copy + crate::MaybeSendSync;

    /// Kernel used to recompute an output in source order when
    /// [`Self::needs_ordered_fallback`] reports that [`Self::Acc`] could not
    /// represent it. It must not itself request a fallback.
    type Fallback: ReduceKernel<T>;

    /// Whether `Self::Acc` is layout-identical to the scalar `T`.
    ///
    /// Only `true` allows the output-block kernel to `memcpy` an accumulator
    /// block straight into a contiguous destination.
    const SCALAR_ACC: bool;

    fn identity() -> Self::Acc;
    fn from_value(value: T) -> Self::Acc;
    fn combine(lhs: Self::Acc, rhs: Self::Acc) -> Self::Acc;

    /// Converts a fully reduced accumulator into the output scalar.
    fn finish(acc: Self::Acc) -> T;

    /// Whether `acc` must be recomputed in source order by [`Self::Fallback`].
    #[inline(always)]
    fn needs_ordered_fallback(_acc: &Self::Acc) -> bool {
        false
    }

    /// Whether the kernel may reassociate its partial results across lanes,
    /// chunks and runs.
    ///
    /// Sum-like kernels, integer products (wrapping multiplication is
    /// associative) and the stable product's exponent-tracked accumulator are
    /// safe to reassociate. A raw floating-point product is not: a lane that
    /// sees only factors above one overflows to `inf` while another that sees
    /// only factors below one underflows to `0`, and the combine is
    /// `inf * 0 = NaN` even when the mathematical product is representable
    /// over the run as a whole (tenferro-rs#2150). The reassociated product
    /// kernel therefore does not attempt a fallback and is documented as
    /// PyTorch-compatible rather than exact.
    #[inline(always)]
    fn reassociable() -> bool {
        true
    }

    /// Reduces a contiguous run.
    ///
    /// Full reductions, parallel full reduction chunks and qualifying axis
    /// runs all call this one function.
    #[inline(always)]
    fn contiguous(values: &[T]) -> Self::Acc {
        lanes_contiguous::<T, Self>(values)
    }

    /// Folds one unit-stride source run into the adjacent-output accumulator
    /// block, preserving the per-output reduced order.
    #[inline(always)]
    fn combine_block(acc: &mut [Self::Acc], values: &[T]) {
        scalar_combine_block::<T, Self>(acc, values);
    }
}

/// Per-output scalar fold used by [`ReduceKernel::combine_block`]'s default and
/// by kernels without a SIMD block path.
#[inline(always)]
fn scalar_combine_block<T: ErasedReduceScalar, K: ReduceKernel<T> + ?Sized>(
    acc: &mut [K::Acc],
    values: &[T],
) {
    for (slot, &value) in acc.iter_mut().zip(values) {
        *slot = K::combine(*slot, K::from_value(value));
    }
}

pub(super) struct SumKernel;
pub(super) struct ReassociatedProductKernel;
pub(super) struct OrderedProductKernel;
pub(super) struct StableProductKernel;
pub(super) struct SumSquaresKernel;
pub(super) struct MaxKernel;
pub(super) struct MinKernel;

impl<T: ErasedReduceScalar> ReduceKernel<T> for SumKernel {
    type Acc = T;
    type Fallback = Self;
    const SCALAR_ACC: bool = true;

    #[inline(always)]
    fn identity() -> T {
        T::zero()
    }
    #[inline(always)]
    fn from_value(value: T) -> T {
        value
    }
    #[inline(always)]
    fn combine(lhs: T, rhs: T) -> T {
        T::reduce_sum(lhs, rhs)
    }
    #[inline(always)]
    fn finish(value: T) -> T {
        value
    }
    #[inline(always)]
    fn contiguous(values: &[T]) -> T {
        T::try_simd_sum(values).unwrap_or_else(|| lanes_contiguous::<T, Self>(values))
    }
    #[inline(always)]
    fn combine_block(acc: &mut [T], values: &[T]) {
        if !T::try_simd_add_block(acc, values) {
            scalar_combine_block::<T, Self>(acc, values);
        }
    }
}

/// Reassociated floating product, exactly like [`SumKernel`] reassociates.
///
/// The result may differ from a source-order fold and, like PyTorch, an
/// alternating `2.0`/`0.5` input can reassociate to `NaN` even when the
/// mathematical product is representable. Use [`StableProductKernel`] when
/// that is unacceptable.
impl<T: ErasedReduceScalar> ReduceKernel<T> for ReassociatedProductKernel {
    type Acc = T;
    type Fallback = Self;
    const SCALAR_ACC: bool = true;

    #[inline(always)]
    fn identity() -> T {
        T::one()
    }
    #[inline(always)]
    fn from_value(value: T) -> T {
        value
    }
    #[inline(always)]
    fn combine(lhs: T, rhs: T) -> T {
        T::reduce_product(lhs, rhs)
    }
    #[inline(always)]
    fn finish(value: T) -> T {
        value
    }
    #[inline(always)]
    fn contiguous(values: &[T]) -> T {
        T::try_simd_product(values).unwrap_or_else(|| lanes_contiguous::<T, Self>(values))
    }
    #[inline(always)]
    fn combine_block(acc: &mut [T], values: &[T]) {
        if !T::try_simd_mul_block(acc, values) {
            scalar_combine_block::<T, Self>(acc, values);
        }
    }
}

/// Source-order floating product, used as the stable kernel's fallback and as
/// the complex stable product (complex products are accumulated in source
/// order because signed zeros and rounding do not reassociate).
impl<T: ErasedReduceScalar> ReduceKernel<T> for OrderedProductKernel {
    type Acc = T;
    type Fallback = Self;
    const SCALAR_ACC: bool = true;

    #[inline(always)]
    fn identity() -> T {
        T::one()
    }
    #[inline(always)]
    fn from_value(value: T) -> T {
        value
    }
    #[inline(always)]
    fn combine(lhs: T, rhs: T) -> T {
        T::reduce_product(lhs, rhs)
    }
    #[inline(always)]
    fn finish(value: T) -> T {
        value
    }
    #[inline(always)]
    fn reassociable() -> bool {
        false
    }
    #[inline(always)]
    fn contiguous(values: &[T]) -> T {
        values
            .iter()
            .fold(T::one(), |acc, &value| T::reduce_product(acc, value))
    }
    #[inline(always)]
    fn combine_block(acc: &mut [T], values: &[T]) {
        // The SIMD block multiply still combines one reduced step at a time
        // per output slot, so it preserves the per-output source order.
        if !T::try_simd_mul_block(acc, values) {
            scalar_combine_block::<T, Self>(acc, values);
        }
    }
}

/// Overflow/underflow-safe product kernel.
///
/// For `f32`/`f64` the accumulator tracks a signed mantissa and an integer
/// exponent per lane, so factors far outside the dtype range do not overflow
/// or underflow a partial. An input that is zero, subnormal, infinite or NaN
/// cannot be represented that way: the accumulator marks it and
/// [`ReduceKernel::needs_ordered_fallback`] makes the erased entry recompute
/// the output with [`OrderedProductKernel`], which keeps IEEE propagation
/// (`0 * inf = NaN`, and a zero or infinity result) identical to the sequential
/// fold. Integer dtypes reduce to the reassociated wrapping product and complex
/// dtypes to the source-order product; neither requests the fallback.
impl<T: ErasedReduceScalar> ReduceKernel<T> for StableProductKernel {
    type Acc = T::StableProductAcc;
    type Fallback = OrderedProductKernel;
    const SCALAR_ACC: bool = T::SP_ACC_IS_SCALAR;

    #[inline(always)]
    fn identity() -> T::StableProductAcc {
        T::sp_identity()
    }
    #[inline(always)]
    fn from_value(value: T) -> T::StableProductAcc {
        T::sp_from_value(value)
    }
    #[inline(always)]
    fn combine(lhs: T::StableProductAcc, rhs: T::StableProductAcc) -> T::StableProductAcc {
        T::sp_combine(lhs, rhs)
    }
    #[inline(always)]
    fn finish(acc: T::StableProductAcc) -> T {
        T::sp_finish(acc)
    }
    #[inline(always)]
    fn needs_ordered_fallback(acc: &T::StableProductAcc) -> bool {
        T::sp_needs_fallback(acc)
    }
    #[inline(always)]
    fn reassociable() -> bool {
        T::sp_reassociable()
    }
    #[inline(always)]
    fn contiguous(values: &[T]) -> T::StableProductAcc {
        T::sp_contiguous(values)
    }
    #[inline(always)]
    fn combine_block(acc: &mut [T::StableProductAcc], values: &[T]) {
        T::sp_combine_block(acc, values);
    }
}

impl<T: ErasedReduceScalar> ReduceKernel<T> for SumSquaresKernel {
    type Acc = T;
    type Fallback = Self;
    const SCALAR_ACC: bool = true;

    #[inline(always)]
    fn identity() -> T {
        T::zero()
    }
    #[inline(always)]
    fn from_value(value: T) -> T {
        T::reduce_product(value, value)
    }
    #[inline(always)]
    fn combine(lhs: T, rhs: T) -> T {
        T::reduce_sum(lhs, rhs)
    }
    #[inline(always)]
    fn finish(value: T) -> T {
        value
    }
    #[inline(always)]
    fn contiguous(values: &[T]) -> T {
        T::try_simd_sum_squares(values).unwrap_or_else(|| lanes_contiguous::<T, Self>(values))
    }
    #[inline(always)]
    fn combine_block(acc: &mut [T], values: &[T]) {
        if !T::try_simd_sum_squares_block(acc, values) {
            scalar_combine_block::<T, Self>(acc, values);
        }
    }
}

impl<T: ErasedReduceScalar> ReduceKernel<T> for MaxKernel {
    type Acc = T;
    type Fallback = Self;
    const SCALAR_ACC: bool = true;

    #[inline(always)]
    fn identity() -> T {
        T::max_identity()
    }
    #[inline(always)]
    fn from_value(value: T) -> T {
        value
    }
    #[inline(always)]
    fn combine(lhs: T, rhs: T) -> T {
        T::reduce_max(lhs, rhs)
    }
    #[inline(always)]
    fn finish(value: T) -> T {
        value
    }
}

impl<T: ErasedReduceScalar> ReduceKernel<T> for MinKernel {
    type Acc = T;
    type Fallback = Self;
    const SCALAR_ACC: bool = true;

    #[inline(always)]
    fn identity() -> T {
        T::min_identity()
    }
    #[inline(always)]
    fn from_value(value: T) -> T {
        value
    }
    #[inline(always)]
    fn combine(lhs: T, rhs: T) -> T {
        T::reduce_min(lhs, rhs)
    }
    #[inline(always)]
    fn finish(value: T) -> T {
        value
    }
}

/// Scalar support for [`StableProductKernel`].
///
/// Private implementation detail: it carries the accumulator type and the
/// per-dtype split/combine/finish primitives. `f32` and `f64` track an exact
/// signed exponent, integers use wrapping multiplication and complex values use
/// the source-order fold.
pub(super) trait StableProductScalar: Copy + crate::MaybeSendSync {
    /// Accumulator of the stable product kernel.
    type StableProductAcc: Copy + crate::MaybeSendSync;

    /// Whether `StableProductAcc` is layout-identical to `Self`.
    const SP_ACC_IS_SCALAR: bool = true;

    fn sp_identity() -> Self::StableProductAcc;
    fn sp_from_value(value: Self) -> Self::StableProductAcc;
    fn sp_combine(
        lhs: Self::StableProductAcc,
        rhs: Self::StableProductAcc,
    ) -> Self::StableProductAcc;
    fn sp_finish(acc: Self::StableProductAcc) -> Self;

    /// Whether `acc` must be recomputed in source order.
    #[inline(always)]
    fn sp_needs_fallback(_acc: &Self::StableProductAcc) -> bool {
        false
    }

    /// Whether the stable accumulator may be reassociated. Complex products
    /// are source order; everything else is reassociated.
    #[inline(always)]
    fn sp_reassociable() -> bool {
        true
    }

    /// Reduction of a contiguous run. The default is the shared
    /// multi-accumulator lane kernel; the float impls override it with the
    /// hybrid mantissa/exponent block kernel.
    #[inline(always)]
    fn sp_contiguous(values: &[Self]) -> Self::StableProductAcc {
        stable_lanes::<Self>(values)
    }

    /// Sequential per-slot combine, which keeps each output's source order.
    #[inline(always)]
    fn sp_combine_block(acc: &mut [Self::StableProductAcc], values: &[Self]) {
        for (slot, &value) in acc.iter_mut().zip(values) {
            *slot = Self::sp_combine(*slot, Self::sp_from_value(value));
        }
    }
}

/// Lane kernel shared by the non-float stable dtypes.
#[inline(always)]
fn stable_lanes<T: StableProductScalar>(values: &[T]) -> T::StableProductAcc {
    let mut lanes = [T::sp_identity(); CONTIGUOUS_REDUCE_LANES];
    let mut chunks = values.chunks_exact(CONTIGUOUS_REDUCE_LANES);
    for chunk in chunks.by_ref() {
        for (lane, &value) in lanes.iter_mut().zip(chunk) {
            *lane = T::sp_combine(*lane, T::sp_from_value(value));
        }
    }
    for (lane, &value) in lanes.iter_mut().zip(chunks.remainder()) {
        *lane = T::sp_combine(*lane, T::sp_from_value(value));
    }
    lanes.into_iter().fold(T::sp_identity(), T::sp_combine)
}

/// Exact signed mantissa/exponent accumulator for `f32`/`f64`.
///
/// `m` is in `[1, 2)` in magnitude (with the sign of the running product) and
/// `e` is the exact integer exponent. The `bad` flag records that a
/// zero/subnormal/infinite/NaN factor entered the accumulator, which forces the
/// ordered fallback.
#[derive(Clone, Copy, Debug)]
pub(super) struct StableProductFloatAcc {
    m: f64,
    e: i64,
    bad: bool,
}

/// Per-dtype bit primitives for the float stable product.
pub(super) trait StableProductFloatInput: Copy {
    /// Reduced steps folded per plain-product block.
    const K: usize;
    /// Smallest per-input magnitude a plain block may contain (`2^-B`).
    const LO: f64;
    /// Largest per-input magnitude a plain block may contain (`2^B`).
    const HI: f64;

    /// Splits a raw input into a signed mantissa in `[1, 2)` and an exact
    /// exponent; `bad` is set for zero, subnormal, infinite and NaN inputs.
    fn split(self) -> (f64, i64, bool);
    fn to_f64(self) -> f64;
    fn from_f64(value: f64) -> Self;
}

#[inline(always)]
fn split_f64(value: f64) -> (f64, i64, bool) {
    const EXP: u64 = 0x7ffu64 << 52;
    const ONE: u64 = 1023u64 << 52;
    let bits = value.to_bits();
    let raw_exp = (bits & EXP) >> 52;
    let m = f64::from_bits((bits & !EXP) | ONE);
    (m, raw_exp as i64 - 1023, raw_exp == 0 || raw_exp == 0x7ff)
}

#[inline(always)]
fn split_f32(value: f32) -> (f64, i64, bool) {
    const EXP: u32 = 0xffu32 << 23;
    const ONE: u32 = 127u32 << 23;
    let bits = value.to_bits();
    let raw_exp = (bits & EXP) >> 23;
    let m = f32::from_bits((bits & !EXP) | ONE) as f64;
    (m, raw_exp as i64 - 127, raw_exp == 0 || raw_exp == 0xff)
}

impl StableProductFloatInput for f64 {
    // K * B = 8 * 120 = 960, comfortably below the f64 exponent range.
    const K: usize = 8;
    const LO: f64 = 7.52316384526264e-37; // 2^-120
    const HI: f64 = 1.329227995784916e36; // 2^120

    #[inline(always)]
    fn split(self) -> (f64, i64, bool) {
        split_f64(self)
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn from_f64(value: f64) -> f64 {
        value
    }
}

impl StableProductFloatInput for f32 {
    // K * B = 4 * 30 = 120, safely below the f32 exponent range (126).
    const K: usize = 4;
    const LO: f64 = 9.313225746154785e-10; // 2^-30
    const HI: f64 = 1.073741824e9; // 2^30

    #[inline(always)]
    fn split(self) -> (f64, i64, bool) {
        split_f32(self)
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn from_f64(value: f64) -> f32 {
        value as f32
    }
}

/// Correctly rounded `m * 2^e` for `f64`, including overflow to `inf`,
/// underflow to a subnormal or signed zero, and an infinite exponent range
/// (the loops exit as soon as `m` saturates).
fn f64_ldexp(mut m: f64, mut e: i64) -> f64 {
    // Exponent fields 1 and 2046: 2^-1022 and 2^1023.
    const TWO_POW_M1022: u64 = 1u64 << 52;
    const TWO_POW_1023: u64 = 2046u64 << 52;
    while e > 1023 {
        m *= f64::from_bits(TWO_POW_1023);
        e -= 1023;
        if m.is_infinite() {
            return m;
        }
    }
    while e < -1022 {
        m *= f64::from_bits(TWO_POW_M1022);
        e += 1022;
        if m == 0.0 {
            return m;
        }
    }
    m * f64::from_bits(((e + 1023) as u64) << 52)
}

/// Renormalizes each lane mantissa into `[1, 2)` magnitude and accumulates the
/// extracted exponent.
#[inline(always)]
fn stable_renorm(am: &mut [f64], ae: &mut [i64]) {
    for (m, e) in am.iter_mut().zip(ae.iter_mut()) {
        let (mantissa, exponent, _) = split_f64(*m);
        *m = mantissa;
        *e += exponent;
    }
}

/// Combines two normalized float accumulators with one rounding step.
#[inline(always)]
fn stable_float_combine(
    lhs: StableProductFloatAcc,
    rhs: StableProductFloatAcc,
) -> StableProductFloatAcc {
    if lhs.bad | rhs.bad {
        return StableProductFloatAcc {
            m: lhs.m,
            e: lhs.e,
            bad: true,
        };
    }
    let mut m = lhs.m * rhs.m;
    let mut e = lhs.e + rhs.e;
    if m.abs() >= 2.0 {
        m *= 0.5;
        e += 1;
    }
    StableProductFloatAcc { m, e, bad: false }
}

/// Hybrid mantissa/exponent block kernel.
///
/// Each block of `LANES * K` inputs is first multiplied plainly per lane while
/// tracking the per-lane input magnitude range. If every magnitude is inside
/// `[LO, HI]`, the block product is provably finite (`K * B` bounded), so the
/// plain product is accepted. Otherwise the block is redone with the exact
/// per-input split path, which also records zero/subnormal/inf/NaN factors for
/// the ordered fallback. Lanes are renormalized once per block.
fn stable_float_contiguous<T: StableProductFloatInput>(values: &[T]) -> StableProductFloatAcc {
    const LANES: usize = 8;
    let identity = StableProductFloatAcc {
        m: 1.0,
        e: 0,
        bad: false,
    };
    if values.is_empty() {
        return identity;
    }
    let block = LANES * T::K;
    let mut am = [1.0f64; LANES];
    let mut ae = [0i64; LANES];
    let mut bad = false;
    let mut chunks = values.chunks_exact(block);
    for chunk in &mut chunks {
        let mut tentative = am;
        let mut max = [0.0f64; LANES];
        let mut min = [f64::INFINITY; LANES];
        for step in 0..T::K {
            for (lane, max) in max.iter_mut().enumerate() {
                let value = chunk[step * LANES + lane].to_f64();
                tentative[lane] *= value;
                let magnitude = value.abs();
                if magnitude > *max {
                    *max = magnitude;
                }
                if magnitude < min[lane] {
                    min[lane] = magnitude;
                }
            }
        }
        let in_range = max.iter().all(|&value| value <= T::HI)
            && min.iter().all(|&value| value >= T::LO)
            && !tentative.iter().any(|value| value.is_nan());
        if in_range {
            am = tentative;
        } else {
            // ponytail: this split path is scalar. It keeps `StableProduct`
            // correct on out-of-range inputs, but the extreme-exponent case runs
            // at roughly 3x the reassociated product instead of the near-parity
            // the in-range block path reaches. Vectorizing the split lane is the
            // next step only if `StableProduct` ever becomes the default.
            for step in 0..T::K {
                for lane in 0..LANES {
                    let (mantissa, exponent, is_bad) = chunk[step * LANES + lane].split();
                    am[lane] *= mantissa;
                    ae[lane] += exponent;
                    bad |= is_bad;
                }
            }
        }
        stable_renorm(&mut am, &mut ae);
    }
    for (index, &value) in chunks.remainder().iter().enumerate() {
        let (mantissa, exponent, is_bad) = value.split();
        let lane = index % LANES;
        am[lane] *= mantissa;
        ae[lane] += exponent;
        bad |= is_bad;
    }
    stable_renorm(&mut am, &mut ae);
    let mut m = 1.0f64;
    let mut e = 0i64;
    for lane in 0..LANES {
        m *= am[lane];
        e += ae[lane];
        let (mantissa, exponent, _) = split_f64(m);
        m = mantissa;
        e += exponent;
    }
    StableProductFloatAcc { m, e, bad }
}

macro_rules! impl_stable_product_float {
    ($($ty:ty),* $(,)?) => {
        $(
            impl StableProductScalar for $ty {
                type StableProductAcc = StableProductFloatAcc;
                const SP_ACC_IS_SCALAR: bool = false;

                #[inline(always)]
                fn sp_identity() -> StableProductFloatAcc {
                    StableProductFloatAcc { m: 1.0, e: 0, bad: false }
                }
                #[inline(always)]
                fn sp_from_value(value: Self) -> StableProductFloatAcc {
                    let (m, e, bad) = <Self as StableProductFloatInput>::split(value);
                    StableProductFloatAcc { m, e, bad }
                }
                #[inline(always)]
                fn sp_combine(
                    lhs: StableProductFloatAcc,
                    rhs: StableProductFloatAcc,
                ) -> StableProductFloatAcc {
                    stable_float_combine(lhs, rhs)
                }
                #[inline(always)]
                fn sp_finish(acc: StableProductFloatAcc) -> Self {
                    if acc.bad {
                        return <$ty>::NAN;
                    }
                    <$ty as StableProductFloatInput>::from_f64(f64_ldexp(acc.m, acc.e))
                }
                #[inline(always)]
                fn sp_needs_fallback(acc: &StableProductFloatAcc) -> bool {
                    acc.bad
                }
                #[inline(always)]
                fn sp_reassociable() -> bool {
                    true
                }
                #[inline(always)]
                fn sp_contiguous(values: &[Self]) -> StableProductFloatAcc {
                    crate::simd::dispatch_if_large(values.len(), || {
                        stable_float_contiguous::<$ty>(values)
                    })
                }
            }
        )*
    };
}

impl_stable_product_float!(f32, f64);

impl StableProductScalar for i32 {
    type StableProductAcc = i32;

    #[inline(always)]
    fn sp_identity() -> i32 {
        1
    }
    #[inline(always)]
    fn sp_from_value(value: i32) -> i32 {
        value
    }
    #[inline(always)]
    fn sp_combine(lhs: i32, rhs: i32) -> i32 {
        lhs.wrapping_mul(rhs)
    }
    #[inline(always)]
    fn sp_finish(acc: i32) -> i32 {
        acc
    }
}

impl StableProductScalar for i64 {
    type StableProductAcc = i64;

    #[inline(always)]
    fn sp_identity() -> i64 {
        1
    }
    #[inline(always)]
    fn sp_from_value(value: i64) -> i64 {
        value
    }
    #[inline(always)]
    fn sp_combine(lhs: i64, rhs: i64) -> i64 {
        lhs.wrapping_mul(rhs)
    }
    #[inline(always)]
    fn sp_finish(acc: i64) -> i64 {
        acc
    }
}

macro_rules! impl_stable_product_complex {
    ($($ty:ty),* $(,)?) => {
        $(
            impl StableProductScalar for $ty {
                type StableProductAcc = $ty;

                #[inline(always)]
                fn sp_identity() -> $ty {
                    <$ty>::new(1.0, 0.0)
                }
                #[inline(always)]
                fn sp_from_value(value: $ty) -> $ty {
                    value
                }
                #[inline(always)]
                fn sp_combine(lhs: $ty, rhs: $ty) -> $ty {
                    lhs * rhs
                }
                #[inline(always)]
                fn sp_finish(acc: $ty) -> $ty {
                    acc
                }
                #[inline(always)]
                fn sp_reassociable() -> bool {
                    false
                }
                #[inline(always)]
                fn sp_contiguous(values: &[$ty]) -> $ty {
                    values
                        .iter()
                        .fold(Self::sp_identity(), |acc, &value| acc * value)
                }
            }
        )*
    };
}

impl_stable_product_complex!(num_complex::Complex32, num_complex::Complex64);

/// Number of accumulators of [`lanes_contiguous`].
///
/// Sixteen lanes give eight independent 128-bit chains for `f64`, enough to
/// hide the compare and select latency of max and min on NEON and AVX2.
pub(super) const CONTIGUOUS_REDUCE_LANES: usize = 16;

/// Multi-accumulator kernel over a contiguous run.
///
/// Lane `i` folds elements `i, i + 16, ...` in order, the remainder is folded
/// into the first lanes, and the lanes are folded left to right.
#[inline(always)]
fn lanes_contiguous<T, K>(values: &[T]) -> K::Acc
where
    T: ErasedReduceScalar,
    K: ReduceKernel<T> + ?Sized,
{
    crate::simd::dispatch_if_large(values.len(), || {
        let mut lanes = [K::identity(); CONTIGUOUS_REDUCE_LANES];
        let mut chunks = values.chunks_exact(CONTIGUOUS_REDUCE_LANES);
        for chunk in chunks.by_ref() {
            for (lane, &value) in lanes.iter_mut().zip(chunk) {
                *lane = K::combine(*lane, K::from_value(value));
            }
        }
        for (lane, &value) in lanes.iter_mut().zip(chunks.remainder()) {
            *lane = K::combine(*lane, K::from_value(value));
        }
        lanes.into_iter().fold(K::identity(), K::combine)
    })
}

/// Multi-accumulator kernel over a strided run, with the lane assignment of
/// [`lanes_contiguous`].
///
/// # Safety
/// `ptr.offset(i * stride)` must be a readable element for every `i < len`.
#[inline(always)]
unsafe fn lanes_strided<T, K>(ptr: *const T, stride: isize, len: usize) -> K::Acc
where
    T: ErasedReduceScalar,
    K: ReduceKernel<T>,
{
    let mut lanes = [K::identity(); REDUCE_LANES];
    let mut cursor = ptr;
    for _ in 0..len / REDUCE_LANES {
        let mut lane_ptr = cursor;
        for lane in &mut lanes {
            // SAFETY: the caller proves every element of the run is readable.
            *lane = K::combine(*lane, K::from_value(unsafe { *lane_ptr }));
            lane_ptr = lane_ptr.wrapping_offset(stride);
        }
        cursor = lane_ptr;
    }
    for lane in lanes.iter_mut().take(len % REDUCE_LANES) {
        // SAFETY: the caller proves every element of the run is readable.
        *lane = K::combine(*lane, K::from_value(unsafe { *cursor }));
        cursor = cursor.wrapping_offset(stride);
    }
    lanes.into_iter().fold(K::identity(), K::combine)
}

/// Folds one run into `acc` in source order.
///
/// A separate run product would be a reassociation of this output's factors, so
/// kernels that must not reassociate carry one accumulator through every run
/// (tenferro-rs#2150).
///
/// # Safety
/// `ptr.offset(i * stride)` must be a readable element for every `i < len`.
#[inline(always)]
unsafe fn fold_run_ordered<T, K>(
    mut acc: K::Acc,
    ptr: *const T,
    stride: isize,
    len: usize,
) -> K::Acc
where
    T: ErasedReduceScalar,
    K: ReduceKernel<T>,
{
    let mut cursor = ptr;
    for _ in 0..len {
        acc = K::combine(acc, K::from_value(unsafe { *cursor }));
        cursor = cursor.wrapping_offset(stride);
    }
    acc
}

/// Reduces one run of `len` elements starting at `ptr`.
///
/// # Safety
/// `ptr.offset(i * stride)` must be a readable element for every `i < len`.
#[inline(always)]
unsafe fn reduce_run<T, K>(ptr: *const T, stride: isize, len: usize) -> K::Acc
where
    T: ErasedReduceScalar,
    K: ReduceKernel<T>,
{
    if stride == 1 {
        // SAFETY: a unit-stride readable run is a valid shared slice.
        K::contiguous(unsafe { core::slice::from_raw_parts(ptr, len) })
    } else if K::reassociable() {
        // SAFETY: forwarded caller contract.
        unsafe { lanes_strided::<T, K>(ptr, stride, len) }
    } else {
        // A non-reassociable kernel folds the strided run in source order rather
        // than assigning lanes (tenferro-rs#2150).
        // SAFETY: forwarded caller contract.
        unsafe { fold_run_ordered::<T, K>(K::identity(), ptr, stride, len) }
    }
}

/// Precomputed traversal of a full reduction.
///
/// Source axes of extent one are dropped, the rest are ordered by increasing
/// absolute stride (stride zero last) and fused where they are contiguous.
/// The first fused axis is the run axis; the others step between runs.
#[derive(Clone, Debug)]
pub(super) struct FullTraversal {
    total: usize,
    run_len: usize,
    run_stride: isize,
    run_axes: crate::inline::InlineVec<ReduceOuterAxis>,
}

impl FullTraversal {
    pub(super) fn compile(dims: &[usize], strides: &[isize]) -> Result<Self> {
        if dims.len() != strides.len() {
            return Err(StridedError::StrideLengthMismatch);
        }
        let total = dims
            .iter()
            .try_fold(1usize, |total, &dim| total.checked_mul(dim))
            .ok_or(StridedError::OffsetOverflow)?;
        if total == 0 {
            return Ok(Self {
                total,
                run_len: 0,
                run_stride: 1,
                run_axes: crate::inline::InlineVec::new(),
            });
        }
        let mut axes: crate::inline::InlineVec<(usize, isize)> = dims
            .iter()
            .copied()
            .zip(strides.iter().copied())
            .filter(|&(extent, _)| extent != 1)
            .collect();
        axes.sort_by_key(|&(_, stride)| {
            if stride == 0 {
                usize::MAX
            } else {
                stride.unsigned_abs()
            }
        });
        let mut fused = compress_reduce_outer_axes(
            axes.into_iter()
                .map(|(extent, stride)| {
                    Ok(ReduceOuterAxis {
                        extent,
                        source_step: stride,
                        source_reset: checked_reduce_reset(extent, stride)?,
                        dest_step: 0,
                        dest_reset: 0,
                    })
                })
                .collect::<Result<crate::inline::InlineVec<_>>>()?,
        )?;
        if fused.is_empty() {
            return Ok(Self {
                total,
                run_len: 1,
                run_stride: 1,
                run_axes: fused,
            });
        }
        let run = fused.remove(0);
        Ok(Self {
            total,
            run_len: run.extent,
            run_stride: run.source_step,
            run_axes: fused,
        })
    }

    #[inline]
    pub(super) fn total(&self) -> usize {
        self.total
    }
}

/// Reduces traversal positions `range` of a full reduction.
///
/// # Safety
/// `source.offset(source_base + o)` must be readable for every offset `o`
/// reachable by `traversal` (proven by descriptor validation plus exact plan
/// layout equality), and `range` must be a non-empty subrange of
/// `0..traversal.total()`.
pub(super) unsafe fn reduce_full_range<T, K>(
    source: *const T,
    source_base: isize,
    traversal: &FullTraversal,
    range: Range<usize>,
) -> Result<K::Acc>
where
    T: ErasedReduceScalar,
    K: ReduceKernel<T>,
{
    debug_assert!(!range.is_empty() && range.end <= traversal.total);
    let run_len = traversal.run_len;
    let run_stride = traversal.run_stride;
    let mut col = range.start % run_len;
    // Decode once per worker range, then advance incrementally.
    let mut runs =
        ReduceOuterCursor::decode(range.start / run_len, source_base, 0, &traversal.run_axes)?;
    let mut remaining = range.len();
    let mut acc = None;
    loop {
        let len = (run_len - col).min(remaining);
        // INVARIANT: `col + len <= run_len` and the run start is a reachable
        // source offset, so every element of this run is readable.
        let start = runs.source_offset + col as isize * run_stride;
        // SAFETY: see the invariant above and the function contract.
        let partial = if K::reassociable() {
            // SAFETY: see the invariant above and the function contract.
            unsafe { reduce_run::<T, K>(source.offset(start), run_stride, len) }
        } else {
            // A per-run partial is itself a reassociation of this output's
            // factors, so a non-reassociable kernel carries one accumulator
            // through every run (tenferro-rs#2150).
            // SAFETY: the run start is reachable and the run covers `len`
            // elements, by the invariant above.
            unsafe {
                fold_run_ordered::<T, K>(
                    acc.unwrap_or_else(K::identity),
                    source.offset(start),
                    run_stride,
                    len,
                )
            }
        };
        acc = Some(match acc {
            Some(acc) if K::reassociable() => K::combine(acc, partial),
            _ => partial,
        });
        remaining -= len;
        if remaining == 0 {
            break;
        }
        col = 0;
        runs.advance();
    }
    Ok(acc.unwrap_or_else(K::identity))
}

/// Strategy for one axis reduction layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AxesStrategy {
    /// The leading reduced run has unit source stride: reduce each run with
    /// [`ReduceKernel::contiguous`] and fold the runs in order.
    ContiguousRuns,
    /// The leading kept axis has unit source stride: accumulate a block of
    /// adjacent outputs from one contiguous source run per reduced element
    /// (an axpy style column sweep), each output in sequential reduction
    /// order.
    OutputBlocks,
    /// Sequential fold in caller-supplied reduced-axis order.
    Sequential,
}

fn axes_strategy(outer_axes: &[ReduceOuterAxis], inner_axes: &[ReduceInnerAxis]) -> AxesStrategy {
    if inner_axes
        .first()
        .is_some_and(|axis| axis.source_step == 1 && axis.extent >= AXIS_RUN_KERNEL_MIN_LEN)
    {
        AxesStrategy::ContiguousRuns
    } else if outer_axes
        .first()
        .is_some_and(|axis| axis.source_step == 1 && axis.extent >= AXIS_OUTPUT_BLOCK_MIN_LEN)
    {
        AxesStrategy::OutputBlocks
    } else {
        AxesStrategy::Sequential
    }
}

/// Source and destination description of one axis reduction execution.
pub(super) struct AxesRange<'a, T> {
    pub(super) source: *const T,
    pub(super) source_base: isize,
    pub(super) dest: *mut T,
    pub(super) dest_base: isize,
    pub(super) outer_axes: &'a [ReduceOuterAxis],
    pub(super) inner_axes: &'a [ReduceInnerAxis],
    pub(super) reduce_total: usize,
}

/// Recomputes one axis output in source order with [`ReduceKernel::Fallback`].
///
/// The stable product uses this when its accumulator could not represent the
/// output's factors. Outputs are independent, so only the flagged output is
/// recomputed; the caller's other outputs keep the stable result.
///
/// # Safety
/// `source.offset(source_offset)` must start a reachable reduced run of
/// `reduce_total` elements visited by `inner_axes`.
#[inline(always)]
unsafe fn fallback_axes_output<T, K>(
    source: *const T,
    inner_axes: &[ReduceInnerAxis],
    source_offset: isize,
    reduce_total: usize,
) -> T
where
    T: ErasedReduceScalar,
    K: ReduceKernel<T>,
{
    let mut inner = ReduceInnerCursor::new(0, inner_axes);
    inner.reset(source_offset);
    // SAFETY: forwarded caller contract; `K::Fallback` never requests another
    // fallback, so this fold is the source-order recomputation.
    let acc = unsafe { sequential_fold::<T, K::Fallback>(source, &mut inner, reduce_total) };
    K::Fallback::finish(acc)
}

/// Source offset of output `index` inside the current output block, one kept
/// axis step apart. Checked once per flagged fallback output, never per element.
#[inline]
fn block_output_source_offset(base: isize, step: isize, index: usize) -> Result<isize> {
    let index = isize::try_from(index).map_err(|_| StridedError::OffsetOverflow)?;
    let delta = step
        .checked_mul(index)
        .ok_or(StridedError::OffsetOverflow)?;
    base.checked_add(delta).ok_or(StridedError::OffsetOverflow)
}

/// Reduces outputs `range` of an axis reduction.
///
/// Every output is computed independently from its own reduced elements, so
/// the result for an output does not depend on how outputs are split across
/// worker ranges. An output whose stable accumulator cannot represent its
/// factors is recomputed in source order on its own; the other outputs keep
/// the stable result.
///
/// # Safety
/// `parts.source` and `parts.dest` must point to `T` storage whose reachable
/// offsets from the bases were validated against the plan layouts, the
/// destination must not overlap the source, `parts.reduce_total` must be
/// nonzero and `range` must be a subrange of the output domain.
pub(super) unsafe fn reduce_axes_range<T, K>(
    parts: AxesRange<'_, T>,
    range: Range<usize>,
) -> Result<()>
where
    T: ErasedReduceScalar,
    K: ReduceKernel<T>,
{
    if range.is_empty() {
        return Ok(());
    }
    let source = parts.source;
    let dest = parts.dest;
    let reduce_total = parts.reduce_total;
    debug_assert!(reduce_total != 0);
    // Decode once per worker range, then advance incrementally.
    let mut outer = ReduceOuterCursor::decode(
        range.start,
        parts.source_base,
        parts.dest_base,
        parts.outer_axes,
    )?;
    // INVARIANT: (1) compile_axes checked signed source/destination spans
    // and every cursor step/reset, including -(extent-1)*stride; (2) raw
    // descriptors validated every reachable offset; (3) execute checked exact
    // plan-layout equality before dispatch. Every pointer formed below is a
    // reachable cursor offset under that chain.
    match axes_strategy(parts.outer_axes, parts.inner_axes) {
        AxesStrategy::ContiguousRuns => {
            let run_len = parts.inner_axes[0].extent;
            let run_count = reduce_total / run_len;
            let mut runs = ReduceInnerCursor::new(0, &parts.inner_axes[1..]);
            for output in range.clone() {
                runs.reset(outer.source_offset);
                // SAFETY: each run is `run_len` reachable unit-stride elements.
                let mut acc = if K::reassociable() {
                    // SAFETY: each run is `run_len` reachable unit-stride elements.
                    K::contiguous(unsafe {
                        core::slice::from_raw_parts(source.offset(runs.source_offset), run_len)
                    })
                } else {
                    // SAFETY: as above, for the first reachable run.
                    unsafe {
                        fold_run_ordered::<T, K>(
                            K::identity(),
                            source.offset(runs.source_offset),
                            1,
                            run_len,
                        )
                    }
                };
                for _ in 1..run_count {
                    runs.advance();
                    if K::reassociable() {
                        // SAFETY: as above, for the next reachable run.
                        let partial = K::contiguous(unsafe {
                            core::slice::from_raw_parts(source.offset(runs.source_offset), run_len)
                        });
                        acc = K::combine(acc, partial);
                    } else {
                        // SAFETY: as above, for the next reachable run.
                        acc = unsafe {
                            fold_run_ordered::<T, K>(
                                acc,
                                source.offset(runs.source_offset),
                                1,
                                run_len,
                            )
                        };
                    }
                }
                let value = if K::needs_ordered_fallback(&acc) {
                    // SAFETY: `outer.source_offset` starts this output's
                    // reduced run.
                    unsafe {
                        fallback_axes_output::<T, K>(
                            source,
                            parts.inner_axes,
                            outer.source_offset,
                            reduce_total,
                        )
                    }
                } else {
                    K::finish(acc)
                };
                // SAFETY: the destination cursor offset is reachable.
                unsafe { dest.offset(outer.dest_offset).write(value) };
                if output + 1 < range.end {
                    outer.advance();
                }
            }
        }
        AxesStrategy::OutputBlocks => {
            let block_axis = parts.outer_axes[0];
            let mut inner = ReduceInnerCursor::new(0, parts.inner_axes);
            let block = axis_output_block_len::<T>().min(range.len());
            // Per-thread reused accumulator block: the first call on a worker
            // sizes it, later calls never allocate.
            with_axis_block_scratch::<K::Acc, _>(block, K::identity(), |acc| -> Result<()> {
                let mut output = range.start;
                while output < range.end {
                    // A block stays inside one run of the leading kept axis and
                    // inside the worker range.
                    let block_room = block_axis.extent - outer.leading_coord();
                    let len = block_room.min(range.end - output).min(block);
                    let acc = &mut acc[..len];
                    acc.fill(K::identity());
                    inner.reset(outer.source_offset);
                    let mut offsets = [0isize; AXIS_OUTPUT_BLOCK_STEPS];
                    let mut value_index = 0;
                    // Fold several reduced steps per pass; each slot still
                    // combines its values in reduced order.
                    while value_index + AXIS_OUTPUT_BLOCK_STEPS <= reduce_total {
                        for offset in &mut offsets {
                            *offset = inner.source_offset;
                            value_index += 1;
                            if value_index < reduce_total {
                                inner.advance();
                            }
                        }
                        // SAFETY: the `len` adjacent outputs of this block read
                        // `len` reachable unit-stride source elements along the
                        // leading kept axis at every reduced position.
                        let [v0, v1, v2, v3] = offsets.map(|offset| unsafe {
                            core::slice::from_raw_parts(source.offset(offset), len)
                        });
                        K::combine_block(acc, v0);
                        K::combine_block(acc, v1);
                        K::combine_block(acc, v2);
                        K::combine_block(acc, v3);
                    }
                    while value_index < reduce_total {
                        // SAFETY: as above, for one reduced position.
                        let values = unsafe {
                            core::slice::from_raw_parts(source.offset(inner.source_offset), len)
                        };
                        K::combine_block(acc, values);
                        value_index += 1;
                        if value_index < reduce_total {
                            inner.advance();
                        }
                    }
                    let block_needs_fallback = acc.iter().any(K::needs_ordered_fallback);
                    if block_axis.dest_step == 1 && K::SCALAR_ACC && !block_needs_fallback {
                        // SAFETY: with `SCALAR_ACC` the accumulator is
                        // layout-identical to `T`, and the block outputs are
                        // `len` reachable unit-stride destination elements,
                        // disjoint from the source.
                        unsafe {
                            core::ptr::copy_nonoverlapping(
                                acc.as_ptr() as *const T,
                                dest.offset(outer.dest_offset),
                                len,
                            )
                        };
                    } else {
                        for (index, &value) in acc.iter().enumerate() {
                            let offset = outer.dest_offset + index as isize * block_axis.dest_step;
                            let output_value = if K::needs_ordered_fallback(&value) {
                                // SAFETY: adjacent outputs of this block start a
                                // reachable reduced run one kept-axis step apart.
                                let source_offset = block_output_source_offset(
                                    outer.source_offset,
                                    block_axis.source_step,
                                    index,
                                )?;
                                unsafe {
                                    fallback_axes_output::<T, K>(
                                        source,
                                        parts.inner_axes,
                                        source_offset,
                                        reduce_total,
                                    )
                                }
                            } else {
                                K::finish(value)
                            };
                            // SAFETY: the block outputs are reachable destination offsets.
                            unsafe { dest.offset(offset).write(output_value) };
                        }
                    }
                    for _ in 0..len {
                        output += 1;
                        if output < range.end {
                            outer.advance();
                        }
                    }
                }
                Ok(())
            })?;
        }
        AxesStrategy::Sequential => {
            let mut inner = ReduceInnerCursor::new(0, parts.inner_axes);
            for output in range.clone() {
                inner.reset(outer.source_offset);
                // SAFETY: forwarded layout invariant.
                let acc = unsafe { sequential_fold::<T, K>(source, &mut inner, reduce_total) };
                let value = if K::needs_ordered_fallback(&acc) {
                    // SAFETY: `outer.source_offset` starts this output's
                    // reduced run.
                    unsafe {
                        fallback_axes_output::<T, K>(
                            source,
                            parts.inner_axes,
                            outer.source_offset,
                            reduce_total,
                        )
                    }
                } else {
                    K::finish(acc)
                };
                // SAFETY: the destination cursor offset is reachable.
                unsafe { dest.offset(outer.dest_offset).write(value) };
                if output + 1 < range.end {
                    outer.advance();
                }
            }
        }
    }
    Ok(())
}

/// Folds `count` elements visited by `inner` in cursor order.
///
/// # Safety
/// Every cursor offset visited must be a readable element of `source`.
#[inline(always)]
unsafe fn sequential_fold<T, K>(
    source: *const T,
    inner: &mut ReduceInnerCursor<'_>,
    count: usize,
) -> K::Acc
where
    T: ErasedReduceScalar,
    K: ReduceKernel<T>,
{
    let mut acc = K::identity();
    for value_index in 0..count {
        // SAFETY: forwarded caller contract.
        let value = unsafe { *source.offset(inner.source_offset) };
        acc = K::combine(acc, K::from_value(value));
        if value_index + 1 < count {
            inner.advance();
        }
    }
    acc
}
