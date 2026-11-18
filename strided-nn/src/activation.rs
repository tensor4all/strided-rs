//! SiLU and sigmoid over contiguous slices, with an optionally cached derivative.
//!
//! `sigmoid(x) = 1 / (1 + exp(-x))` and `silu(x) = x * sigmoid(x)` (Elfwing, Uchibe, Doya
//! 2018; Ramachandran, Zoph, Le 2017), written from those definitions. With `s = sigmoid(x)`
//! the derivatives are `sigmoid'(x) = s * (1 - s)` and
//! `silu'(x) = s * (1 + x * (1 - s))`.
//!
//! The forward can write `f'(x)` next to `f(x)`; the VJP then only multiplies the upstream
//! gradient by that cached derivative (`dx = dy * f'(x)`), so it needs no `exp`.
//!
//! # Contract
//!
//! - The `exp` comes from `elem-simd-math`. It is staged through caller scratch
//!   ([`scratch_len`]); the forward allocates nothing and creates no pool.
//! - Elements are independent, so results are bitwise identical at every thread count.
//! - Inputs and outputs have equal lengths; a mismatch is an [`ActivationError`] before any
//!   output is written.

use core::mem::MaybeUninit;

use strided_basic::execution::{KernelClass, SendPtr};
use strided_basic::Exec;

use crate::error::ActivationError;
use crate::par::{for_unit_ranges, span_mut};
use crate::real::Real;

#[cfg(test)]
mod tests;

/// Elements one task stages through scratch at a time.
const CHUNK: usize = 1024;

fn check(name: &'static str, expected: usize, got: usize) -> Result<(), ActivationError> {
    if expected == got {
        Ok(())
    } else {
        Err(ActivationError::LengthMismatch {
            name,
            expected,
            got,
        })
    }
}

/// Scratch elements a forward over `len` elements needs when its [`Exec`] has `budget`.
///
/// # Examples
///
/// ```
/// assert_eq!(strided_nn::activation::scratch_len(10, 4), 10);
/// assert_eq!(strided_nn::activation::scratch_len(1 << 20, 4), 4 * 1024);
/// ```
pub fn scratch_len(len: usize, budget: usize) -> usize {
    CHUNK.min(len) * len.div_ceil(CHUNK).min(budget.max(1))
}

/// `silu` selects SiLU, otherwise sigmoid.
fn forward<T: Real>(
    silu: bool,
    x: &[T],
    y: &mut [T],
    deriv: Option<&mut [T]>,
    scratch: &mut [T],
    exec: Exec<'_>,
) -> Result<(), ActivationError> {
    check("y", x.len(), y.len())?;
    if let Some(d) = &deriv {
        check("deriv", x.len(), d.len())?;
    }
    if x.is_empty() {
        return Ok(());
    }
    let per = CHUNK.min(x.len());
    let capacity = scratch.len() / per;
    if capacity == 0 {
        return Err(ActivationError::ScratchTooSmall {
            need: per,
            got: scratch.len(),
        });
    }
    let chunks = x.len().div_ceil(CHUNK);
    let y_ptr = SendPtr(y.as_mut_ptr());
    let d_ptr = deriv.map(|d| SendPtr(d.as_mut_ptr()));
    let s_ptr = SendPtr(scratch.as_mut_ptr());
    // Tasks own whole chunks; the task count is bounded by the scratch regions, one per task.
    let task_chunks = chunks.div_ceil(capacity.min(chunks));
    let tasks = chunks.div_ceil(task_chunks);
    for_unit_ranges(
        tasks,
        x.len(),
        KernelClass::Transcendental,
        exec,
        capacity,
        |t0, t1| {
            for t in t0..t1 {
                // SAFETY: task `t < tasks <= capacity` (`tasks = ceil(chunks / ceil(chunks /
                // min(capacity, chunks)))`), and `capacity * per <= scratch.len()`, so the region
                // `t * per .. (t + 1) * per` is inside `scratch`; regions of distinct tasks are
                // disjoint and `scratch` is borrowed `&mut` for the call.
                let tmp = unsafe { span_mut(s_ptr, t * per, per) };
                for c in t * task_chunks..((t + 1) * task_chunks).min(chunks) {
                    let lo = c * CHUNK;
                    let n = CHUNK.min(x.len() - lo);
                    // SAFETY: `y` and `deriv` have `x.len()` elements (checked), `lo + n <= x.len()`,
                    // and chunk `c` belongs to exactly one task.
                    let yc = unsafe { span_mut(y_ptr, lo, n) };
                    let dc = d_ptr.map(|p| unsafe { span_mut(p, lo, n) });
                    chunk_forward(silu, &x[lo..lo + n], yc, dc, &mut tmp[..n]);
                }
            }
        },
    );
    Ok(())
}

fn chunk_forward<T: Real>(silu: bool, x: &[T], y: &mut [T], d: Option<&mut [T]>, tmp: &mut [T]) {
    // `y` holds exp(-x) first, then is overwritten by the result.
    for (t, &v) in tmp.iter_mut().zip(x) {
        *t = -v;
    }
    // SAFETY: `&mut [T]` and `&mut [MaybeUninit<T>]` have the same layout. `exp_into` only
    // stores finished `T` values (never uninitialized ones), so `y` stays fully initialized.
    let y_uninit = unsafe { &mut *(core::ptr::from_mut::<[T]>(y) as *mut [MaybeUninit<T>]) };
    let ok = T::exp_into(tmp, y_uninit);
    debug_assert!(ok, "chunk and scratch lengths are equal");
    let one = T::ONE;
    match (silu, d) {
        (false, None) => {
            for (o, &v) in y.iter_mut().zip(x) {
                let _ = v;
                *o = one / (one + *o);
            }
        }
        (false, Some(d)) => {
            for (o, g) in y.iter_mut().zip(d) {
                let s = one / (one + *o);
                *o = s;
                *g = s * (one - s);
            }
        }
        (true, None) => {
            for (o, &v) in y.iter_mut().zip(x) {
                *o = v * (one / (one + *o));
            }
        }
        (true, Some(d)) => {
            for ((o, g), &v) in y.iter_mut().zip(d).zip(x) {
                let s = one / (one + *o);
                *o = v * s;
                *g = s * (one + v * (one - s));
            }
        }
    }
}

fn vjp<T: Real>(
    dy: &[T],
    deriv: &[T],
    dx: &mut [T],
    exec: Exec<'_>,
) -> Result<(), ActivationError> {
    check("deriv", dy.len(), deriv.len())?;
    check("dx", dy.len(), dx.len())?;
    let dx_ptr = SendPtr(dx.as_mut_ptr());
    for_unit_ranges(
        dy.len().div_ceil(CHUNK),
        dy.len(),
        KernelClass::Map,
        exec,
        usize::MAX,
        |c0, c1| {
            let lo = c0 * CHUNK;
            let hi = (c1 * CHUNK).min(dy.len());
            // SAFETY: `dx.len() == dy.len()` (checked) and `lo..hi <= dy.len()`; chunk ranges of
            // distinct tasks are disjoint.
            let out = unsafe { span_mut(dx_ptr, lo, hi - lo) };
            for ((o, &g), &d) in out.iter_mut().zip(&dy[lo..hi]).zip(&deriv[lo..hi]) {
                *o = g * d;
            }
        },
    );
    Ok(())
}

macro_rules! activation_entries {
    ($t:ty, $silu:ident, $sigmoid:ident, $silu_vjp:ident, $sigmoid_vjp:ident) => {
        /// SiLU forward for this element type: `y = x * sigmoid(x)`.
        ///
        /// When `deriv` is `Some`, it receives `silu'(x)` for the VJP. `scratch` must hold
        /// at least one chunk; [`scratch_len`] sizes it for the budget of `exec`.
        ///
        /// # Errors
        ///
        /// [`ActivationError`] for a length mismatch or too little scratch.
        pub fn $silu(
            x: &[$t],
            y: &mut [$t],
            deriv: Option<&mut [$t]>,
            scratch: &mut [$t],
            exec: Exec<'_>,
        ) -> Result<(), ActivationError> {
            forward(true, x, y, deriv, scratch, exec)
        }

        /// Sigmoid forward for this element type: `y = 1 / (1 + exp(-x))`.
        ///
        /// When `deriv` is `Some`, it receives `sigmoid'(x)` for the VJP. `scratch` must
        /// hold at least one chunk; [`scratch_len`] sizes it for the budget of `exec`.
        ///
        /// # Errors
        ///
        /// [`ActivationError`] for a length mismatch or too little scratch.
        pub fn $sigmoid(
            x: &[$t],
            y: &mut [$t],
            deriv: Option<&mut [$t]>,
            scratch: &mut [$t],
            exec: Exec<'_>,
        ) -> Result<(), ActivationError> {
            forward(false, x, y, deriv, scratch, exec)
        }

        /// SiLU VJP: `dx = dy * deriv`, with `deriv` the cached `silu'(x)`.
        ///
        /// # Errors
        ///
        /// [`ActivationError`] for a length mismatch.
        pub fn $silu_vjp(
            dy: &[$t],
            deriv: &[$t],
            dx: &mut [$t],
            exec: Exec<'_>,
        ) -> Result<(), ActivationError> {
            vjp(dy, deriv, dx, exec)
        }

        /// Sigmoid VJP: `dx = dy * deriv`, with `deriv` the cached `sigmoid'(x)`.
        ///
        /// # Errors
        ///
        /// [`ActivationError`] for a length mismatch.
        pub fn $sigmoid_vjp(
            dy: &[$t],
            deriv: &[$t],
            dx: &mut [$t],
            exec: Exec<'_>,
        ) -> Result<(), ActivationError> {
            vjp(dy, deriv, dx, exec)
        }
    };
}

activation_entries!(
    f32,
    silu_forward_f32,
    sigmoid_forward_f32,
    silu_vjp_f32,
    sigmoid_vjp_f32
);
activation_entries!(
    f64,
    silu_forward_f64,
    sigmoid_forward_f64,
    silu_vjp_f64,
    sigmoid_vjp_f64
);
