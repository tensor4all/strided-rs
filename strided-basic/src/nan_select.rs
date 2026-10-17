//! NaN-propagating elementwise selects shared by the ordinary binary kernels
//! and the fused interpreter.
//!
//! The ordinary erased binary maximum/minimum
//! ([`crate::erased`]-adjacent `ErasedZipOp::Maximum`/`Minimum` in
//! `strided-kernel`) and the fused `FusedOp::Maximum`/`Minimum` must agree on
//! IEEE value propagation. Both call these helpers so the two definitions
//! cannot drift. The exact contract, pinned by
//! `strided-kernel/tests/erased_elementwise_kernels.rs` and the fused
//! differential tests:
//!
//! - any NaN operand yields the canonical NaN of the dtype;
//! - ties, including `+0.0` against `-0.0`, return `lhs`.
//!
//! The select is written without a short-circuit `||` so a chunk loop lowers
//! to compare-and-blend rather than a data-dependent branch.

/// NaN-propagating elementwise maximum for real floats.
///
/// ```
/// use strided_basic::nan_propagating_maximum;
/// assert!(nan_propagating_maximum(f64::NAN, 1.0).is_nan());
/// assert!(nan_propagating_maximum(1.0f32, f32::NAN).is_nan());
/// // Ties keep `lhs`.
/// assert_eq!(nan_propagating_maximum(0.0f64, -0.0).to_bits(), 0.0f64.to_bits());
/// ```
#[inline(always)]
pub fn nan_propagating_maximum<T: num_traits::Float>(lhs: T, rhs: T) -> T {
    let picked = if lhs >= rhs { lhs } else { rhs };
    if lhs.is_nan() | rhs.is_nan() {
        T::nan()
    } else {
        picked
    }
}

/// NaN-propagating elementwise minimum for real floats.
///
/// ```
/// use strided_basic::nan_propagating_minimum;
/// assert!(nan_propagating_minimum(f64::NAN, 1.0).is_nan());
/// assert!(nan_propagating_minimum(1.0f32, f32::NAN).is_nan());
/// // Ties keep `lhs`.
/// assert_eq!(nan_propagating_minimum(-0.0f64, 0.0).to_bits(), (-0.0f64).to_bits());
/// ```
#[inline(always)]
pub fn nan_propagating_minimum<T: num_traits::Float>(lhs: T, rhs: T) -> T {
    let picked = if lhs <= rhs { lhs } else { rhs };
    if lhs.is_nan() | rhs.is_nan() {
        T::nan()
    } else {
        picked
    }
}
