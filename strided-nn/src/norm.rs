//! LayerNorm and RMSNorm over the last axis of a contiguous `[rows, width]` slice.
//!
//! A row is `width` consecutive elements; `rows = len / width`. Both norms have a
//! forward that can emit its per-row statistics, an input VJP, and a parameter VJP that
//! consume those cached statistics instead of recomputing them.
//!
//! # Definitions
//!
//! Written from the mathematical definitions: LayerNorm (Ba, Kiros, Hinton 2016),
//! `y = (x - mean) * inv_std * weight + bias` with `inv_std = 1 / sqrt(var + eps)` and the
//! biased (divide-by-`width`) variance; RMSNorm (Zhang, Sennrich 2019),
//! `y = x * inv_rms * weight` with `inv_rms = 1 / sqrt(mean(x^2) + eps)`. With
//! `xhat` the normalized row and `g = dy * weight` (`dy` without a weight), the input
//! gradient is `inv_std * (g - mean(g) - xhat * mean(g * xhat))` for LayerNorm and
//! `inv_rms * (g - xhat * mean(g * xhat))` for RMSNorm; `dweight = sum_rows dy * xhat`
//! and `dbias = sum_rows dy`.
//!
//! # Contract
//!
//! - Every entry takes the [`Exec`] token last, creates no pool and allocates nothing.
//! - Each row is reduced by one task with eight fixed accumulators, so a row's result is
//!   bitwise identical at every thread count and SIMD level.
//! - The parameter VJPs reduce over rows in fixed 128-row chunks into caller scratch and
//!   then sum the chunks in ascending order, so they are deterministic across thread
//!   counts too. Size the scratch with [`layer_norm_params_scratch_len`] /
//!   [`rms_norm_weight_scratch_len`]. Forward and input VJP need no scratch.
//! - `dweight` / `dbias` are overwritten, not accumulated into.
//! - Statistics are `rows` values: `mean` and `inv_std` (LayerNorm), `inv_rms` (RMSNorm).
//!
//! # Errors
//!
//! Each entry returns [`NormError`] for a zero width, a length that is not a whole number
//! of rows, a slice of the wrong length, or too little scratch, before touching any output.

use strided_basic::execution::{KernelClass, SendPtr};
use strided_basic::Exec;

use crate::error::NormError;
use crate::par::{for_unit_ranges, span_mut};
use crate::real::{lane_sum, Real};

#[cfg(test)]
mod tests;

const CLASS: KernelClass = KernelClass::Reduce;
/// Rows per parameter-VJP partial; fixed so the summation order never depends on the token.
const CHUNK_ROWS: usize = 128;
/// Columns per task when the chunk partials are combined.
const COLUMN_BLOCK: usize = 64;

fn check_len(name: &'static str, expected: usize, got: usize) -> Result<(), NormError> {
    if expected == got {
        Ok(())
    } else {
        Err(NormError::LengthMismatch {
            name,
            expected,
            got,
        })
    }
}

fn check_opt<T>(name: &'static str, expected: usize, v: &Option<&[T]>) -> Result<(), NormError> {
    v.map_or(Ok(()), |s| check_len(name, expected, s.len()))
}

/// Validate the row shape and return `rows`.
fn rows_of(width: usize, len: usize) -> Result<usize, NormError> {
    if width == 0 {
        return Err(NormError::EmptyWidth);
    }
    if !len.is_multiple_of(width) {
        return Err(NormError::UnevenRows { len, width });
    }
    Ok(len / width)
}

fn params_scratch(rows: usize, width: usize, planes: usize) -> Result<usize, NormError> {
    rows.div_ceil(CHUNK_ROWS)
        .checked_mul(width)
        .and_then(|n| n.checked_mul(planes))
        .ok_or(NormError::SizeOverflow { rows, width })
}

/// Scratch elements [`layer_norm_vjp_params_f32`] / `_f64` need for `rows` rows of `width`.
///
/// # Errors
///
/// [`NormError::SizeOverflow`] if the size overflows `usize`.
///
/// # Examples
///
/// ```
/// // 300 rows are three 128-row chunks; two planes (weight and bias) of width 8.
/// assert_eq!(strided_nn::norm::layer_norm_params_scratch_len(300, 8).unwrap(), 48);
/// ```
pub fn layer_norm_params_scratch_len(rows: usize, width: usize) -> Result<usize, NormError> {
    params_scratch(rows, width, 2)
}

/// Scratch elements [`rms_norm_vjp_weight_f32`] / `_f64` need for `rows` rows of `width`.
///
/// # Errors
///
/// [`NormError::SizeOverflow`] if the size overflows `usize`.
pub fn rms_norm_weight_scratch_len(rows: usize, width: usize) -> Result<usize, NormError> {
    params_scratch(rows, width, 1)
}

// ---------------------------------------------------------------- row kernels

fn ln_row_forward<T: Real>(
    x: &[T],
    weight: Option<&[T]>,
    bias: Option<&[T]>,
    y: &mut [T],
    eps: T,
) -> (T, T) {
    let n = x.len();
    let count = T::from_usize(n);
    let mean = lane_sum(n, |i| x[i]) / count;
    let var = lane_sum(n, |i| (x[i] - mean) * (x[i] - mean)) / count;
    let inv_std = T::ONE / (var + eps).sqrt();
    match (weight, bias) {
        (Some(w), Some(b)) => {
            for (((o, &v), &w), &b) in y.iter_mut().zip(x).zip(w).zip(b) {
                *o = (v - mean) * inv_std * w + b;
            }
        }
        (Some(w), None) => {
            for ((o, &v), &w) in y.iter_mut().zip(x).zip(w) {
                *o = (v - mean) * inv_std * w;
            }
        }
        (None, Some(b)) => {
            for ((o, &v), &b) in y.iter_mut().zip(x).zip(b) {
                *o = (v - mean) * inv_std + b;
            }
        }
        (None, None) => {
            for (o, &v) in y.iter_mut().zip(x) {
                *o = (v - mean) * inv_std;
            }
        }
    }
    (mean, inv_std)
}

fn rms_row_forward<T: Real>(x: &[T], weight: Option<&[T]>, y: &mut [T], eps: T) -> T {
    let n = x.len();
    let inv_rms = T::ONE / (lane_sum(n, |i| x[i] * x[i]) / T::from_usize(n) + eps).sqrt();
    match weight {
        Some(w) => {
            for ((o, &v), &w) in y.iter_mut().zip(x).zip(w) {
                *o = v * inv_rms * w;
            }
        }
        None => {
            for (o, &v) in y.iter_mut().zip(x) {
                *o = v * inv_rms;
            }
        }
    }
    inv_rms
}

/// Input gradient of one row; `center` is the LayerNorm mean (`None` for RMSNorm).
fn row_vjp_input<T: Real>(
    x: &[T],
    center: Option<T>,
    scale: T,
    weight: Option<&[T]>,
    dy: &[T],
    dx: &mut [T],
) {
    let n = x.len();
    let count = T::from_usize(n);
    let mu = center.unwrap_or(T::ZERO);
    let g = |i: usize| match weight {
        Some(w) => dy[i] * w[i],
        None => dy[i],
    };
    let mean_g = match center {
        Some(_) => lane_sum(n, g) / count,
        None => T::ZERO,
    };
    let mean_gx = lane_sum(n, |i| g(i) * ((x[i] - mu) * scale)) / count;
    for (i, (o, &v)) in dx.iter_mut().zip(x).enumerate() {
        *o = scale * (g(i) - mean_g - (v - mu) * scale * mean_gx);
    }
}

// ----------------------------------------------------------------- drivers

/// One forward over all rows; `layer` selects LayerNorm (statistics `a` = mean, `b` = inv_std)
/// or RMSNorm (statistic `b` = inv_rms).
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
    let y_ptr = SendPtr(y.as_mut_ptr());
    let mean_ptr = mean.map(|m| SendPtr(m.as_mut_ptr()));
    let inv_ptr = inv.map(|s| SendPtr(s.as_mut_ptr()));
    for_unit_ranges(rows, x.len(), CLASS, exec, usize::MAX, |r0, r1| {
        for r in r0..r1 {
            let xr = &x[r * width..(r + 1) * width];
            // SAFETY: `y.len() == x.len() == rows * width` was checked above and `r < rows`, so
            // the span is inside `y`; each row index belongs to exactly one task, and `y` is
            // borrowed `&mut` for the call.
            let yr = unsafe { span_mut(y_ptr, r * width, width) };
            let (m, s) = if layer {
                ln_row_forward(xr, weight, bias, yr, eps)
            } else {
                (T::ZERO, rms_row_forward(xr, weight, yr, eps))
            };
            if let Some(p) = mean_ptr {
                // SAFETY: `mean.len() == rows` was checked, `r < rows`, one task per row.
                unsafe { span_mut(p, r, 1)[0] = m };
            }
            if let Some(p) = inv_ptr {
                // SAFETY: `inv.len() == rows` was checked, `r < rows`, one task per row.
                unsafe { span_mut(p, r, 1)[0] = s };
            }
        }
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn vjp_input<T: Real>(
    width: usize,
    x: &[T],
    mean: Option<&[T]>,
    scale: &[T],
    weight: Option<&[T]>,
    dy: &[T],
    dx: &mut [T],
    exec: Exec<'_>,
) -> Result<(), NormError> {
    let rows = rows_of(width, x.len())?;
    check_len("dy", x.len(), dy.len())?;
    check_len("dx", x.len(), dx.len())?;
    check_len("inv_std", rows, scale.len())?;
    check_opt("mean", rows, &mean)?;
    check_opt("weight", width, &weight)?;
    let dx_ptr = SendPtr(dx.as_mut_ptr());
    for_unit_ranges(rows, x.len(), CLASS, exec, usize::MAX, |r0, r1| {
        for r in r0..r1 {
            let span = r * width..(r + 1) * width;
            // SAFETY: `dx.len() == rows * width` was checked and `r < rows`; one task per row.
            let out = unsafe { span_mut(dx_ptr, r * width, width) };
            row_vjp_input(
                &x[span.clone()],
                mean.map(|m| m[r]),
                scale[r],
                weight,
                &dy[span],
                out,
            );
        }
    });
    Ok(())
}

/// Parameter gradients: `dweight = sum_rows dy * xhat`, `dbias = sum_rows dy`.
#[allow(clippy::too_many_arguments)]
fn vjp_params<T: Real>(
    width: usize,
    x: &[T],
    mean: Option<&[T]>,
    scale: &[T],
    dy: &[T],
    dweight: Option<&mut [T]>,
    dbias: Option<&mut [T]>,
    scratch: &mut [T],
    mut exec: Exec<'_>,
) -> Result<(), NormError> {
    let rows = rows_of(width, x.len())?;
    check_len("dy", x.len(), dy.len())?;
    check_len("inv_std", rows, scale.len())?;
    check_opt("mean", rows, &mean)?;
    if let Some(d) = &dweight {
        check_len("dweight", width, d.len())?;
    }
    if let Some(d) = &dbias {
        check_len("dbias", width, d.len())?;
    }
    let planes = usize::from(dweight.is_some()) + usize::from(dbias.is_some());
    let need = params_scratch(rows, width, planes)?;
    if scratch.len() < need {
        return Err(NormError::ScratchTooSmall {
            need,
            got: scratch.len(),
        });
    }
    if rows == 0 {
        // An empty sum is zero: honor the "overwritten, not accumulated" contract.
        for d in [dweight, dbias].into_iter().flatten() {
            d.fill(T::ZERO);
        }
        return Ok(());
    }
    if planes == 0 {
        return Ok(());
    }
    let chunks = rows.div_ceil(CHUNK_ROWS);
    let plane_len = chunks * width;
    // Plane 0 holds the weight partials (if requested), the next the bias partials.
    let w_off = 0;
    let b_off = if dweight.is_some() { plane_len } else { 0 };
    let scratch_ptr = SendPtr(scratch.as_mut_ptr());
    let has_w = dweight.is_some();
    let has_b = dbias.is_some();

    // Stage 1: one partial row per chunk, rows added in order.
    for_unit_ranges(
        chunks,
        x.len(),
        CLASS,
        exec.reborrow(),
        usize::MAX,
        |c0, c1| {
            for c in c0..c1 {
                // SAFETY: `scratch.len() >= planes * chunks * width` was checked and `c < chunks`,
                // so both spans lie inside `scratch`; chunk `c` is owned by one task and the weight
                // and bias planes are disjoint, as `b_off == plane_len` when both are present.
                let mut pw =
                    has_w.then(|| unsafe { span_mut(scratch_ptr, w_off + c * width, width) });
                let mut pb =
                    has_b.then(|| unsafe { span_mut(scratch_ptr, b_off + c * width, width) });
                for p in pw.iter_mut().chain(pb.iter_mut()) {
                    p.fill(T::ZERO);
                }
                for r in c * CHUNK_ROWS..((c + 1) * CHUNK_ROWS).min(rows) {
                    let xr = &x[r * width..(r + 1) * width];
                    let dr = &dy[r * width..(r + 1) * width];
                    let mu = mean.map_or(T::ZERO, |m| m[r]);
                    let s = scale[r];
                    if let Some(p) = pw.as_deref_mut() {
                        for ((a, &v), &g) in p.iter_mut().zip(xr).zip(dr) {
                            *a += g * ((v - mu) * s);
                        }
                    }
                    if let Some(p) = pb.as_deref_mut() {
                        for (a, &g) in p.iter_mut().zip(dr) {
                            *a += g;
                        }
                    }
                }
            }
        },
    );

    // Stage 2: combine chunks in ascending order, one column block per unit.
    let blocks = width.div_ceil(COLUMN_BLOCK);
    let dw_ptr = dweight.map(|d| SendPtr(d.as_mut_ptr()));
    let db_ptr = dbias.map(|d| SendPtr(d.as_mut_ptr()));
    let scratch_ro = SendPtr(scratch_ptr.as_ptr());
    for_unit_ranges(
        blocks,
        plane_len * planes,
        CLASS,
        exec,
        usize::MAX,
        |b0, b1| {
            let j0 = b0 * COLUMN_BLOCK;
            let j1 = (b1 * COLUMN_BLOCK).min(width);
            for (out, off) in [(dw_ptr, w_off), (db_ptr, b_off)] {
                let Some(out) = out else { continue };
                // SAFETY: `dweight`/`dbias` have `width` elements (checked) and `j0..j1 <= width`;
                // column blocks are disjoint across tasks.
                let dst = unsafe { span_mut(out, j0, j1 - j0) };
                dst.fill(T::ZERO);
                for c in 0..chunks {
                    // SAFETY: the span is inside the plane (`c < chunks`, `j1 <= width`); stage 1
                    // has finished (`for_unit_ranges` joins), and stage 2 only reads scratch.
                    let part = unsafe { span_mut(scratch_ro, off + c * width + j0, j1 - j0) };
                    for (a, &v) in dst.iter_mut().zip(part.iter()) {
                        *a += v;
                    }
                }
            }
        },
    );
    Ok(())
}

// ------------------------------------------------------------- public API

macro_rules! norm_entries {
    ($t:ty, $ln_fwd:ident, $ln_in:ident, $ln_par:ident, $rms_fwd:ident, $rms_in:ident, $rms_w:ident) => {
        /// LayerNorm forward over the rows of `x`, for this element type.
        ///
        /// `weight` and `bias` (each `width` long) are optional. When `stats` is
        /// `Some((mean, inv_std))`, each `rows` long, the per-row statistics are written
        /// for the VJPs. `eps` is added to the variance.
        ///
        /// # Errors
        ///
        /// [`NormError`] for a zero `width`, a ragged `x`, or a slice of the wrong length.
        #[allow(clippy::too_many_arguments)]
        pub fn $ln_fwd(
            width: usize,
            eps: $t,
            x: &[$t],
            weight: Option<&[$t]>,
            bias: Option<&[$t]>,
            y: &mut [$t],
            stats: Option<(&mut [$t], &mut [$t])>,
            exec: Exec<'_>,
        ) -> Result<(), NormError> {
            let (m, s) = stats.map_or((None, None), |(m, s)| (Some(m), Some(s)));
            forward(true, width, eps, x, weight, bias, y, m, s, exec)
        }

        /// LayerNorm input VJP from the cached `mean` and `inv_std` of the forward.
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
            dy: &[$t],
            dx: &mut [$t],
            exec: Exec<'_>,
        ) -> Result<(), NormError> {
            vjp_input(width, x, Some(mean), inv_std, weight, dy, dx, exec)
        }

        /// LayerNorm parameter VJP: overwrites `dweight` and/or `dbias` (each `width` long)
        /// with the sums over rows, using the cached statistics.
        ///
        /// `scratch` must hold [`layer_norm_params_scratch_len`] elements (fewer are
        /// accepted when only one of the two outputs is requested).
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
            dy: &[$t],
            dweight: Option<&mut [$t]>,
            dbias: Option<&mut [$t]>,
            scratch: &mut [$t],
            exec: Exec<'_>,
        ) -> Result<(), NormError> {
            vjp_params(
                width,
                x,
                Some(mean),
                inv_std,
                dy,
                dweight,
                dbias,
                scratch,
                exec,
            )
        }

        /// RMSNorm forward over the rows of `x`, for this element type.
        ///
        /// When `inv_rms` is `Some`, it receives the `rows` per-row `1 / sqrt(mean(x^2) + eps)`.
        ///
        /// # Errors
        ///
        /// [`NormError`] for a zero `width`, a ragged `x`, or a slice of the wrong length.
        #[allow(clippy::too_many_arguments)]
        pub fn $rms_fwd(
            width: usize,
            eps: $t,
            x: &[$t],
            weight: Option<&[$t]>,
            y: &mut [$t],
            inv_rms: Option<&mut [$t]>,
            exec: Exec<'_>,
        ) -> Result<(), NormError> {
            forward(false, width, eps, x, weight, None, y, None, inv_rms, exec)
        }

        /// RMSNorm input VJP from the cached `inv_rms` of the forward.
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
            dy: &[$t],
            dx: &mut [$t],
            exec: Exec<'_>,
        ) -> Result<(), NormError> {
            vjp_input(width, x, None, inv_rms, weight, dy, dx, exec)
        }

        /// RMSNorm weight VJP: overwrites `dweight` (`width` long) with the sum over rows.
        ///
        /// `scratch` must hold [`rms_norm_weight_scratch_len`] elements.
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
            dy: &[$t],
            dweight: &mut [$t],
            scratch: &mut [$t],
            exec: Exec<'_>,
        ) -> Result<(), NormError> {
            vjp_params(
                width,
                x,
                None,
                inv_rms,
                dy,
                Some(dweight),
                None,
                scratch,
                exec,
            )
        }
    };
}

norm_entries!(
    f32,
    layer_norm_forward_f32,
    layer_norm_vjp_input_f32,
    layer_norm_vjp_params_f32,
    rms_norm_forward_f32,
    rms_norm_vjp_input_f32,
    rms_norm_vjp_weight_f32
);
norm_entries!(
    f64,
    layer_norm_forward_f64,
    layer_norm_vjp_input_f64,
    layer_norm_vjp_params_f64,
    rms_norm_forward_f64,
    rms_norm_vjp_input_f64,
    rms_norm_vjp_weight_f64
);
