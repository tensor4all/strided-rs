//! Fused `LayerNorm -> SiLU` and `RMSNorm -> SiLU` row kernels, forward and VJP.
//!
//! One pass over a contiguous `[rows, width]` slice: the normalized row is produced a
//! cache-sized chunk at a time in per-task scratch and handed straight to the SiLU
//! epilogue, so the intermediate `norm(x)` is never written to the output tensor and never
//! read back. The per-row statistics (`mean`, `inv_std` / `inv_rms`) and, optionally, the
//! cached `silu'(z)` are emitted for the VJP, which therefore recomputes neither.
//!
//! # Prologue and epilogue
//!
//! The kernel is split so that A5 (linear rows) reuses the epilogue unchanged:
//!
//! - **Prologue** — whatever produces the pre-activation row `z` (and `-z`) in scratch:
//!   here, the row statistics and the normalize-scale-shift of A1 (`prepare_row` /
//!   `fill_chunk`). A bias-add prologue (`z = x + bias`) or a linear prologue (`z` = a
//!   caller-GEMM output row plus bias) replaces only this part.
//! - **Epilogue** — `activation::epilogue` (crate-private): `exp(-z)`, then `silu(z)` and, if
//!   requested, `silu'(z)`. It is the *same function* the standalone A2 kernels call, so it
//!   is the only place the activation is rounded.
//!
//! The VJP mirrors that split: the epilogue's backward is the elementwise `dz = dy *
//! silu'(z)` (the *gate*), and the prologue's backward is the A1 input / parameter VJP
//! consuming `dz`.
//!
//! # Numerical contract
//!
//! The fused forward is **bit-identical** to `layer_norm_forward` followed by `silu_forward`
//! (and `rms_norm_forward` followed by `silu_forward`), and the fused VJPs are bit-identical
//! to `silu_vjp` followed by the A1 input / parameter VJPs fed the same cached statistics:
//! the prologue evaluates the A1 expressions in the same order with the same reductions, the
//! epilogue is shared code, and `exp` is elementwise. This is asserted bit for bit in the
//! tests across widths, affine variants and thread counts. Results are independent of the
//! thread count.
//!
//! # Contract
//!
//! - Explicit [`Exec`] last, no pool creation, no allocation. The forward needs scratch
//!   ([`norm_silu_scratch_len`]); the input VJP needs none; the parameter VJP uses the A1
//!   scratch ([`crate::norm::layer_norm_params_scratch_len`] /
//!   [`crate::norm::rms_norm_weight_scratch_len`]).
//! - `dweight` / `dbias` are overwritten, not accumulated into.
//! - Errors are [`NormError`], returned before any output is touched.

use strided_basic::execution::{KernelClass, SendPtr};
use strided_basic::Exec;

use crate::activation;
use crate::error::NormError;
use crate::norm::{check_len, check_opt, ln_stats, rms_stats, rows_of, vjp_params, CLASS};
use crate::par::{for_unit_ranges, span_mut};
use crate::real::{lane_sum, Real};

#[cfg(test)]
mod tests;

/// Elements of a row one epilogue call handles at a time; bounds per-task scratch.
const CHUNK: usize = 1024;

/// Scratch elements a forward over `rows` rows of `width` needs when its [`Exec`] has `budget`.
///
/// Each task uses `2 * min(width, 1024)` elements (`z` and `-z` of one chunk).
///
/// # Errors
///
/// [`NormError::SizeOverflow`] if the size overflows `usize`.
///
/// # Examples
///
/// ```
/// assert_eq!(strided_nn::norm_silu::norm_silu_scratch_len(10, 100, 4).unwrap(), 800);
/// assert_eq!(strided_nn::norm_silu::norm_silu_scratch_len(2, 4096, 8).unwrap(), 4096);
/// ```
pub fn norm_silu_scratch_len(rows: usize, width: usize, budget: usize) -> Result<usize, NormError> {
    CHUNK
        .min(width)
        .checked_mul(2)
        .and_then(|n| n.checked_mul(rows.min(budget.max(1))))
        .ok_or(NormError::SizeOverflow { rows, width })
}

// ------------------------------------------------------------------- prologue

/// Row statistics of the prologue: `(center, scale)`; `center` is zero for RMSNorm.
#[inline(always)]
fn prepare_row<T: Real>(layer: bool, x: &[T], eps: T) -> (T, T, T) {
    if layer {
        let (m, s) = ln_stats(x, eps);
        (m, m, s)
    } else {
        let s = rms_stats(x, eps);
        (T::ZERO, T::ZERO, s)
    }
}

/// Normalize-scale-shift of one chunk into `z`, and its negation into `neg_z`.
///
/// The per-element expressions are those of A1's forward, in the same order, so `z` is the
/// bit pattern `layer_norm_forward` / `rms_norm_forward` would store.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn fill_chunk<T: Real>(
    layer: bool,
    mean: T,
    scale: T,
    x: &[T],
    weight: Option<&[T]>,
    bias: Option<&[T]>,
    z: &mut [T],
    neg_z: &mut [T],
) {
    #[inline(always)]
    fn run<T: Real>(x: &[T], z: &mut [T], neg_z: &mut [T], f: impl Fn(usize, T) -> T) {
        for (i, ((o, n), &v)) in z.iter_mut().zip(neg_z.iter_mut()).zip(x).enumerate() {
            let r = f(i, v);
            *o = r;
            *n = -r;
        }
    }
    match (layer, weight, bias) {
        (true, Some(w), Some(b)) => run(x, z, neg_z, |i, v| (v - mean) * scale * w[i] + b[i]),
        (true, Some(w), None) => run(x, z, neg_z, |i, v| (v - mean) * scale * w[i]),
        (true, None, Some(b)) => run(x, z, neg_z, |i, v| (v - mean) * scale + b[i]),
        (true, None, None) => run(x, z, neg_z, |_, v| (v - mean) * scale),
        (false, Some(w), _) => run(x, z, neg_z, |i, v| v * scale * w[i]),
        (false, None, _) => run(x, z, neg_z, |_, v| v * scale),
    }
}

// -------------------------------------------------------------------- drivers

#[allow(clippy::too_many_arguments)]
fn forward<T: Real>(
    layer: bool,
    width: usize,
    eps: T,
    x: &[T],
    weight: Option<&[T]>,
    bias: Option<&[T]>,
    y: &mut [T],
    mean: Option<&mut [T]>,
    inv: Option<&mut [T]>,
    deriv: Option<&mut [T]>,
    scratch: &mut [T],
    exec: Exec<'_>,
) -> Result<(), NormError> {
    let rows = rows_of(width, x.len())?;
    check_len("y", x.len(), y.len())?;
    check_opt("weight", width, &weight)?;
    check_opt("bias", width, &bias)?;
    if let Some(m) = &mean {
        check_len("mean", rows, m.len())?;
    }
    if let Some(s) = &inv {
        check_len("inv_std", rows, s.len())?;
    }
    if let Some(d) = &deriv {
        check_len("deriv", x.len(), d.len())?;
    }
    let per = CHUNK.min(width);
    let capacity = scratch.len() / (2 * per);
    if capacity == 0 {
        return Err(NormError::ScratchTooSmall {
            need: 2 * per,
            got: scratch.len(),
        });
    }
    if rows == 0 {
        return Ok(());
    }
    let rows_per_task = rows.div_ceil(capacity.min(rows));
    let tasks = rows.div_ceil(rows_per_task);
    let y_ptr = SendPtr(y.as_mut_ptr());
    let d_ptr = deriv.map(|d| SendPtr(d.as_mut_ptr()));
    let mean_ptr = mean.map(|m| SendPtr(m.as_mut_ptr()));
    let inv_ptr = inv.map(|s| SendPtr(s.as_mut_ptr()));
    let s_ptr = SendPtr(scratch.as_mut_ptr());
    for_unit_ranges(
        tasks,
        x.len(),
        KernelClass::Transcendental,
        exec,
        capacity,
        |t0, t1| {
            for t in t0..t1 {
                // SAFETY: `t < tasks <= capacity` and `capacity * 2 * per <= scratch.len()`, so
                // the region `t * 2 * per .. (t + 1) * 2 * per` is inside `scratch`; regions of
                // distinct tasks are disjoint and `scratch` is borrowed `&mut` for the call.
                let region = unsafe { span_mut(s_ptr, t * 2 * per, 2 * per) };
                let (z_buf, nz_buf) = region.split_at_mut(per);
                for r in t * rows_per_task..((t + 1) * rows_per_task).min(rows) {
                    let xr = &x[r * width..(r + 1) * width];
                    let (m, center, s) = prepare_row(layer, xr, eps);
                    if let Some(p) = mean_ptr {
                        // SAFETY: `mean.len() == rows` was checked, `r < rows`, one task per row.
                        unsafe { span_mut(p, r, 1)[0] = m };
                    }
                    if let Some(p) = inv_ptr {
                        // SAFETY: `inv.len() == rows` was checked, `r < rows`, one task per row.
                        unsafe { span_mut(p, r, 1)[0] = s };
                    }
                    let mut lo = 0;
                    while lo < width {
                        let n = CHUNK.min(width - lo);
                        let at = r * width + lo;
                        // SAFETY: `y` and `deriv` have `rows * width` elements (checked) and
                        // `at + n <= rows * width`; row `r` belongs to exactly one task.
                        let yc = unsafe { span_mut(y_ptr, at, n) };
                        let dc = d_ptr.map(|p| unsafe { span_mut(p, at, n) });
                        let (z, nz) = (&mut z_buf[..n], &mut nz_buf[..n]);
                        fill_chunk(
                            layer,
                            center,
                            s,
                            &xr[lo..lo + n],
                            weight.map(|w| &w[lo..lo + n]),
                            bias.map(|b| &b[lo..lo + n]),
                            z,
                            nz,
                        );
                        activation::epilogue(true, z, nz, yc, dc);
                        lo += n;
                    }
                }
            }
        },
    );
    Ok(())
}

/// Input gradient of one row of the fused kernel; `center` is the LayerNorm mean.
///
/// The epilogue's backward `dz = dy * deriv` (times `weight`, the prologue's `g`) is staged
/// once into the output row, which is L1-hot and is overwritten in place by the final pass.
/// The expressions and the reduction order are those of A1's `row_vjp_input`, so the result
/// is bit-identical to `silu_vjp` followed by `layer_norm_vjp_input`.
fn row_vjp_staged<T: Real>(
    x: &[T],
    center: Option<T>,
    scale: T,
    weight: Option<&[T]>,
    deriv: &[T],
    dy: &[T],
    dx: &mut [T],
) {
    let n = x.len();
    let count = T::from_usize(n);
    let mu = center.unwrap_or(T::ZERO);
    match weight {
        Some(w) => {
            for (((o, &g), &f), &w) in dx.iter_mut().zip(dy).zip(deriv).zip(w) {
                *o = (g * f) * w;
            }
        }
        None => {
            for ((o, &g), &f) in dx.iter_mut().zip(dy).zip(deriv) {
                *o = g * f;
            }
        }
    }
    let mean_g = match center {
        Some(_) => lane_sum(n, |i| dx[i]) / count,
        None => T::ZERO,
    };
    let mean_gx = lane_sum(n, |i| dx[i] * ((x[i] - mu) * scale)) / count;
    for (o, &v) in dx.iter_mut().zip(x) {
        *o = scale * (*o - mean_g - (v - mu) * scale * mean_gx);
    }
}

#[allow(clippy::too_many_arguments)]
fn vjp_input<T: Real>(
    width: usize,
    x: &[T],
    mean: Option<&[T]>,
    scale: &[T],
    weight: Option<&[T]>,
    deriv: &[T],
    dy: &[T],
    dx: &mut [T],
    exec: Exec<'_>,
) -> Result<(), NormError> {
    let rows = rows_of(width, x.len())?;
    check_len("dy", x.len(), dy.len())?;
    check_len("dx", x.len(), dx.len())?;
    check_len("deriv", x.len(), deriv.len())?;
    check_len("inv_std", rows, scale.len())?;
    check_opt("mean", rows, &mean)?;
    check_opt("weight", width, &weight)?;
    let dx_ptr = SendPtr(dx.as_mut_ptr());
    for_unit_ranges(rows, x.len(), CLASS, exec, usize::MAX, |r0, r1| {
        for r in r0..r1 {
            let span = r * width..(r + 1) * width;
            // SAFETY: `dx.len() == rows * width` was checked and `r < rows`; one task per row.
            let out = unsafe { span_mut(dx_ptr, r * width, width) };
            row_vjp_staged(
                &x[span.clone()],
                mean.map(|m| m[r]),
                scale[r],
                weight,
                &deriv[span.clone()],
                &dy[span],
                out,
            );
        }
    });
    Ok(())
}

macro_rules! entries {
    ($t:ty, $ln_fwd:ident, $ln_in:ident, $ln_par:ident, $rms_fwd:ident, $rms_in:ident, $rms_w:ident) => {
        /// Fused LayerNorm -> SiLU forward: `y = silu(layer_norm(x))` in one pass.
        ///
        /// `weight` / `bias` (each `width` long) are optional. `stats = Some((mean, inv_std))`
        /// (each `rows` long) caches the row statistics and `deriv = Some(d)` (`rows * width`)
        /// caches `silu'(z)` for the VJP. `scratch` is sized by [`norm_silu_scratch_len`].
        ///
        /// Bit-identical to `layer_norm_forward` then `silu_forward` (see the module docs).
        ///
        /// # Errors
        ///
        /// [`NormError`] for a zero `width`, a ragged `x`, a slice of the wrong length, or too
        /// little scratch.
        #[allow(clippy::too_many_arguments)]
        pub fn $ln_fwd(
            width: usize,
            eps: $t,
            x: &[$t],
            weight: Option<&[$t]>,
            bias: Option<&[$t]>,
            y: &mut [$t],
            stats: Option<(&mut [$t], &mut [$t])>,
            deriv: Option<&mut [$t]>,
            scratch: &mut [$t],
            exec: Exec<'_>,
        ) -> Result<(), NormError> {
            let (m, s) = stats.map_or((None, None), |(m, s)| (Some(m), Some(s)));
            forward(
                true, width, eps, x, weight, bias, y, m, s, deriv, scratch, exec,
            )
        }

        /// Input VJP of the fused LayerNorm -> SiLU: `dx` from `dy`, the cached `deriv`
        /// (`silu'(z)`), and the cached `mean` / `inv_std`. Needs no `exp` and no scratch.
        ///
        /// # Errors
        ///
        /// [`NormError`] for a zero `width`, a ragged `x`, or a slice of the wrong length.
        #[allow(clippy::too_many_arguments)]
        pub fn $ln_in(
            width: usize,
            x: &[$t],
            mean: &[$t],
            inv_std: &[$t],
            weight: Option<&[$t]>,
            deriv: &[$t],
            dy: &[$t],
            dx: &mut [$t],
            exec: Exec<'_>,
        ) -> Result<(), NormError> {
            vjp_input(width, x, Some(mean), inv_std, weight, deriv, dy, dx, exec)
        }

        /// Parameter VJP of the fused LayerNorm -> SiLU: overwrites `dweight` / `dbias`
        /// (each `width` long) with the row sums of `dz = dy * deriv` against `xhat` / `1`.
        ///
        /// `scratch` is sized by [`crate::norm::layer_norm_params_scratch_len`].
        ///
        /// # Errors
        ///
        /// [`NormError`] for a zero `width`, a ragged `x`, a slice of the wrong length, or
        /// too little scratch.
        #[allow(clippy::too_many_arguments)]
        pub fn $ln_par(
            width: usize,
            x: &[$t],
            mean: &[$t],
            inv_std: &[$t],
            deriv: &[$t],
            dy: &[$t],
            dweight: Option<&mut [$t]>,
            dbias: Option<&mut [$t]>,
            scratch: &mut [$t],
            exec: Exec<'_>,
        ) -> Result<(), NormError> {
            check_len("deriv", x.len(), deriv.len())?;
            vjp_params(
                width,
                x,
                Some(mean),
                inv_std,
                Some(deriv),
                dy,
                dweight,
                dbias,
                scratch,
                exec,
            )
        }

        /// Fused RMSNorm -> SiLU forward: `y = silu(rms_norm(x))` in one pass.
        ///
        /// `inv_rms = Some(s)` (`rows` long) caches the statistic; `deriv` and `scratch` are
        /// as for the LayerNorm entry. Bit-identical to `rms_norm_forward` then `silu_forward`.
        ///
        /// # Errors
        ///
        /// [`NormError`] for a zero `width`, a ragged `x`, a slice of the wrong length, or too
        /// little scratch.
        #[allow(clippy::too_many_arguments)]
        pub fn $rms_fwd(
            width: usize,
            eps: $t,
            x: &[$t],
            weight: Option<&[$t]>,
            y: &mut [$t],
            inv_rms: Option<&mut [$t]>,
            deriv: Option<&mut [$t]>,
            scratch: &mut [$t],
            exec: Exec<'_>,
        ) -> Result<(), NormError> {
            forward(
                false, width, eps, x, weight, None, y, None, inv_rms, deriv, scratch, exec,
            )
        }

        /// Input VJP of the fused RMSNorm -> SiLU from the cached `inv_rms` and `deriv`.
        ///
        /// # Errors
        ///
        /// [`NormError`] for a zero `width`, a ragged `x`, or a slice of the wrong length.
        #[allow(clippy::too_many_arguments)]
        pub fn $rms_in(
            width: usize,
            x: &[$t],
            inv_rms: &[$t],
            weight: Option<&[$t]>,
            deriv: &[$t],
            dy: &[$t],
            dx: &mut [$t],
            exec: Exec<'_>,
        ) -> Result<(), NormError> {
            vjp_input(width, x, None, inv_rms, weight, deriv, dy, dx, exec)
        }

        /// Weight VJP of the fused RMSNorm -> SiLU: overwrites `dweight` (`width` long).
        ///
        /// `scratch` is sized by [`crate::norm::rms_norm_weight_scratch_len`].
        ///
        /// # Errors
        ///
        /// [`NormError`] for a zero `width`, a ragged `x`, a slice of the wrong length, or
        /// too little scratch.
        #[allow(clippy::too_many_arguments)]
        pub fn $rms_w(
            width: usize,
            x: &[$t],
            inv_rms: &[$t],
            deriv: &[$t],
            dy: &[$t],
            dweight: &mut [$t],
            scratch: &mut [$t],
            exec: Exec<'_>,
        ) -> Result<(), NormError> {
            check_len("deriv", x.len(), deriv.len())?;
            vjp_params(
                width,
                x,
                None,
                inv_rms,
                Some(deriv),
                dy,
                Some(dweight),
                None,
                scratch,
                exec,
            )
        }
    };
}

entries!(
    f32,
    layer_norm_silu_forward_f32,
    layer_norm_silu_vjp_input_f32,
    layer_norm_silu_vjp_params_f32,
    rms_norm_silu_forward_f32,
    rms_norm_silu_vjp_input_f32,
    rms_norm_silu_vjp_weight_f32
);
entries!(
    f64,
    layer_norm_silu_forward_f64,
    layer_norm_silu_vjp_input_f64,
    layer_norm_silu_vjp_params_f64,
    rms_norm_silu_forward_f64,
    rms_norm_silu_vjp_input_f64,
    rms_norm_silu_vjp_weight_f64
);
