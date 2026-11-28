//! Softmax over any axis of a contiguous column-major tensor, forward and VJP.
//!
//! The tensor is viewed as `[inner, axis_len, outer]` (column-major, so element
//! `(i, j, o)` sits at `i + inner * (j + axis_len * o)`) and the softmax runs over
//! the middle axis.
//!
//! * `inner == 1`: the softmax axis is the contiguous one. This is exactly
//!   [`softmax_axis0_f32`](crate::softmax_axis0_f32) over `outer` columns; the
//!   forward delegates to it, so its results are unchanged bit for bit.
//! * `inner > 1`: the axis is strided by `inner`. The kernel does **not** transpose
//!   and does **not** copy the tensor: it walks the axis one contiguous row of
//!   `inner` elements at a time and keeps a running max and a running sum per
//!   column in a block-sized scratch (the block is a run of columns whose
//!   `axis_len` rows fit a cache budget). Every pass is a vector operation
//!   across the contiguous columns, so the extra cost over the contiguous case
//!   is one more read of the block per pass, not a second full-size buffer.
//!
//! Numerics are those of the axis-0 kernel: subtract the exact axis maximum,
//! `exp` (the same `elem-simd-math` kernel), sum, divide. `max`, `shift`, `exp` and the
//! division are elementwise-identical to the contiguous kernel; only the order of
//! the summation can differ. The strided path sums the axis in source order
//! (`j = 0, 1, ...`), which is bit-identical to `softmax_axis0_f64` (a source-order
//! sum) and differs from the lane-order `softmax_axis0_f32` sum by rounding only.
//!
//! The VJP is the standard `dx = y * (dy - sum(y * dy))` over the same axis. It
//! consumes the forward output `y`; it does not change the forward numerics (the
//! stable formulation of #335 is a separate decision).

use core::mem::MaybeUninit;

use strided_basic::execution::{tree_width, KernelClass, SendPtr};
use strided_basic::Exec;

use crate::error::SoftmaxError;
use crate::par::{for_unit_ranges, span_mut};
use crate::softmax::{fan_out, scratch_len as axis0_scratch_len};

mod lanes;
use lanes::Lanes;

/// Shape `[inner, axis_len, outer]` of a column-major tensor whose middle axis is reduced.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AxisShape {
    /// Elements before the axis (the axis stride); `1` when the axis is contiguous.
    pub inner: usize,
    /// Length of the softmax axis.
    pub axis_len: usize,
    /// Elements after the axis.
    pub outer: usize,
}

/// Cache budget in bytes of one block of columns (`axis_len` rows of the block width).
const BLOCK_BYTES: usize = 256 * 1024;
/// Smallest block width in columns: one 64-byte line of `f32`.
const MIN_BLOCK: usize = 16;
/// `exp` dominates the passes, as in the contiguous kernel.
const CLASS: KernelClass = KernelClass::Transcendental;

/// Columns per block for a strided softmax.
fn block_width(inner: usize, axis_len: usize, size: usize) -> usize {
    let per_column = axis_len.saturating_mul(size).max(1);
    let by_cache = (BLOCK_BYTES / per_column) / MIN_BLOCK * MIN_BLOCK;
    by_cache.max(MIN_BLOCK).min(inner)
}

/// Shape check; returns the element count.
fn check_shape(
    inner: usize,
    axis_len: usize,
    outer: usize,
    size: usize,
) -> Result<usize, SoftmaxError> {
    if axis_len == 0 {
        return Err(SoftmaxError::EmptyAxis);
    }
    let overflow = SoftmaxError::SizeOverflow {
        axis_len,
        element_size: size,
    };
    inner
        .checked_mul(axis_len)
        .and_then(|n| n.checked_mul(outer))
        .filter(|n| n.checked_mul(size).is_some())
        .ok_or(overflow)
}

/// Units (`outer` slabs times column blocks) and block width of a strided call.
fn units(inner: usize, axis_len: usize, outer: usize, size: usize) -> (usize, usize) {
    let block = block_width(inner, axis_len, size);
    (block, inner.div_ceil(block).saturating_mul(outer))
}

/// Elements of caller scratch [`softmax_axis_f32`]/[`softmax_axis_f64`] need when the
/// call's [`Exec`] has the given `budget`.
///
/// For `inner == 1` this is [`scratch_len`](crate::scratch_len) of the contiguous
/// kernel; otherwise three block-width rows (running max, running sum, shifted row)
/// per task.
///
/// ```
/// use strided_nn::{softmax_axis_scratch_len, AxisShape};
/// let shape = |inner, axis_len, outer| AxisShape { inner, axis_len, outer };
/// // A contiguous axis needs the axis-0 scratch; a strided one three 16-column rows per task here.
/// assert!(softmax_axis_scratch_len::<f32>(shape(1, 64, 8), 1) > 0);
/// assert_eq!(softmax_axis_scratch_len::<f32>(shape(16, 4096, 1), 4), 3 * 16);
/// ```
pub fn softmax_axis_scratch_len<T>(shape: AxisShape, budget: usize) -> usize {
    strided_scratch_len::<T>(shape, budget, 3)
}

/// Elements of caller scratch [`softmax_vjp_axis_f32`]/[`softmax_vjp_axis_f64`] need:
/// zero for `inner == 1`, otherwise one block-width row per task.
///
/// ```
/// use strided_nn::{softmax_vjp_axis_scratch_len, AxisShape};
/// let shape = |inner, axis_len, outer| AxisShape { inner, axis_len, outer };
/// assert_eq!(softmax_vjp_axis_scratch_len::<f64>(shape(1, 64, 8), 4), 0);
/// assert_eq!(softmax_vjp_axis_scratch_len::<f64>(shape(16, 4096, 1), 4), 16);
/// ```
pub fn softmax_vjp_axis_scratch_len<T>(shape: AxisShape, budget: usize) -> usize {
    if shape.inner == 1 {
        return 0;
    }
    strided_scratch_len::<T>(shape, budget, 1)
}

fn strided_scratch_len<T>(shape: AxisShape, budget: usize, rows: usize) -> usize {
    let AxisShape {
        inner,
        axis_len,
        outer,
    } = shape;
    let size = core::mem::size_of::<T>();
    if inner == 1 {
        return axis0_scratch_len::<T>(axis_len, axis_len.saturating_mul(outer), budget);
    }
    if axis_len == 0 || inner == 0 || outer == 0 {
        return 0;
    }
    let (block, units) = units(inner, axis_len, outer, size);
    rows.saturating_mul(block)
        .saturating_mul(units.min(budget.max(1)))
}

/// The initialized view of an output span that `exp_into` has filled.
///
/// # Safety
/// Every element of `chunk` must have been written.
unsafe fn initialized<T>(chunk: &mut [MaybeUninit<T>]) -> &mut [T] {
    // SAFETY: `MaybeUninit<T>` has the size and alignment of `T`; the caller guarantees every
    // element is initialized.
    unsafe { core::slice::from_raw_parts_mut(chunk.as_mut_ptr().cast::<T>(), chunk.len()) }
}

/// Task split of a strided call: `tasks` tasks of `per_task` consecutive units each.
struct Split {
    block: usize,
    blocks: usize,
    units: usize,
    per_task: usize,
    tasks: usize,
}

fn split(
    inner: usize,
    axis_len: usize,
    outer: usize,
    size: usize,
    scratch_len: usize,
    rows: usize,
    width: usize,
) -> Result<Split, SoftmaxError> {
    let (block, units) = units(inner, axis_len, outer, size);
    let capacity = scratch_len / (rows * block);
    if capacity == 0 {
        return Err(SoftmaxError::ScratchTooSmall {
            need: rows * block,
            got: scratch_len,
        });
    }
    let cap = capacity.min(width).min(units).max(1);
    let per_task = units.div_ceil(cap);
    Ok(Split {
        block,
        blocks: inner.div_ceil(block),
        units,
        per_task,
        tasks: units.div_ceil(per_task),
    })
}

fn forward<T: Lanes>(
    shape_in: AxisShape,
    input: &[T],
    output: &mut [MaybeUninit<T>],
    scratch: &mut [T],
    exec: Exec<'_>,
) -> Result<(), SoftmaxError> {
    let AxisShape {
        inner,
        axis_len,
        outer,
    } = shape_in;
    let size = core::mem::size_of::<T>();
    let len = check_shape(inner, axis_len, outer, size)?;
    if input.len() != len {
        return Err(SoftmaxError::ShapeMismatch {
            expected: len,
            got: input.len(),
        });
    }
    if output.len() != len {
        return Err(SoftmaxError::LengthMismatch {
            input: len,
            output: output.len(),
        });
    }
    if len == 0 {
        return Ok(());
    }
    if inner == 1 {
        return T::softmax_axis0(axis_len, input, output, scratch, exec);
    }
    let width = tree_width(&exec, CLASS, len);
    let plan = split(inner, axis_len, outer, size, scratch.len(), 3, width.get())?;
    let exec = exec.limit(width);
    let out_ptr = SendPtr(output.as_mut_ptr());
    let scratch_ptr = SendPtr(scratch.as_mut_ptr());
    let body = |task: usize| {
        // SAFETY: `split` kept `tasks <= scratch.len() / (3 * block)`, and `task < tasks`, so
        // `task * 3 * block .. (task + 1) * 3 * block` lies inside `scratch`; tasks get disjoint
        // regions, and `scratch` is borrowed `&mut` for the call.
        let region = unsafe { span_mut(scratch_ptr, task * 3 * plan.block, 3 * plan.block) };
        let (max, rest) = region.split_at_mut(plan.block);
        let (sum, temp) = rest.split_at_mut(plan.block);
        let first = task * plan.per_task;
        for unit in first..(first + plan.per_task).min(plan.units) {
            let (slab, block_index) = (unit / plan.blocks, unit % plan.blocks);
            let col = block_index * plan.block;
            let w = plan.block.min(inner - col);
            let base = slab * inner * axis_len + col;
            let (max, sum, temp) = (&mut max[..w], &mut sum[..w], &mut temp[..w]);
            max.fill(T::NEG_INFINITY);
            for j in 0..axis_len {
                let at = base + j * inner;
                T::max_into(max, &input[at..at + w]);
            }
            T::clamp_max(max);
            sum.fill(T::ZERO);
            for j in 0..axis_len {
                let at = base + j * inner;
                T::shift_into(&input[at..at + w], max, temp);
                // SAFETY: `at + w <= base + (axis_len - 1) * inner + w <= len`, since the block
                // lies inside slab `slab` and column range `col..col + w <= inner`. Distinct
                // `(slab, block_index, j)` triples address disjoint spans, each unit is handled
                // by exactly one task (`fan_out` hands each task index out once), and `output`
                // is borrowed `&mut` for the call; this span is the only live reference to it.
                let row = unsafe { span_mut(out_ptr, at, w) };
                T::exp_into_slots(temp, row);
                // SAFETY: `exp_into_slots` wrote every element of `row`.
                let row = unsafe { initialized(row) };
                T::add_into(sum, row);
            }
            T::clamp_sum(sum);
            for j in 0..axis_len {
                let at = base + j * inner;
                // SAFETY: as above; every slot was initialized by the exp pass of this unit.
                let row = unsafe { initialized(span_mut(out_ptr, at, w)) };
                T::div_into(row, sum);
            }
        }
    };
    fan_out(plan.tasks, exec, body);
    Ok(())
}

fn vjp<T: Lanes>(
    shape_in: AxisShape,
    y: &[T],
    dy: &[T],
    dx: &mut [T],
    scratch: &mut [T],
    exec: Exec<'_>,
) -> Result<(), SoftmaxError> {
    let AxisShape {
        inner,
        axis_len,
        outer,
    } = shape_in;
    let size = core::mem::size_of::<T>();
    let len = check_shape(inner, axis_len, outer, size)?;
    for got in [y.len(), dy.len()] {
        if got != len {
            return Err(SoftmaxError::ShapeMismatch { expected: len, got });
        }
    }
    if dx.len() != len {
        return Err(SoftmaxError::LengthMismatch {
            input: len,
            output: dx.len(),
        });
    }
    if len == 0 {
        return Ok(());
    }
    let dx_ptr = SendPtr(dx.as_mut_ptr());
    if inner == 1 {
        for_unit_ranges(outer, len, CLASS, exec, usize::MAX, |r0, r1| {
            let (a, b) = (r0 * axis_len, r1 * axis_len);
            // SAFETY: `r0..r1` lies inside `0..outer`, so `a..b` lies inside `dx`; ranges of
            // distinct tasks are disjoint and `dx` is borrowed `&mut` for the call.
            let out = unsafe { span_mut(dx_ptr, a, b - a) };
            T::vjp_rows(&y[a..b], &dy[a..b], out, axis_len);
        });
        return Ok(());
    }
    let width = tree_width(&exec, KernelClass::Map, len);
    let plan = split(inner, axis_len, outer, size, scratch.len(), 1, width.get())?;
    let exec = exec.limit(width);
    let scratch_ptr = SendPtr(scratch.as_mut_ptr());
    let body = |task: usize| {
        // SAFETY: as in `forward`, with one row per task.
        let dot = unsafe { span_mut(scratch_ptr, task * plan.block, plan.block) };
        let first = task * plan.per_task;
        for unit in first..(first + plan.per_task).min(plan.units) {
            let (slab, block_index) = (unit / plan.blocks, unit % plan.blocks);
            let col = block_index * plan.block;
            let w = plan.block.min(inner - col);
            let base = slab * inner * axis_len + col;
            let dot = &mut dot[..w];
            dot.fill(T::ZERO);
            for j in 0..axis_len {
                let at = base + j * inner;
                T::dot_into(dot, &y[at..at + w], &dy[at..at + w]);
            }
            for j in 0..axis_len {
                let at = base + j * inner;
                // SAFETY: as in `forward`: in-bounds, disjoint per `(unit, j)`, `dx` borrowed `&mut`.
                let out = unsafe { span_mut(dx_ptr, at, w) };
                T::apply_into(&y[at..at + w], &dy[at..at + w], dot, out);
            }
        }
    };
    fan_out(plan.tasks, exec, body);
    Ok(())
}

macro_rules! entries {
    ($t:ty, $fwd:ident, $vjp:ident) => {
        /// Softmax over the middle axis of a column-major `[inner, axis_len, outer]` slice.
        ///
        /// `inner == 1` is the contiguous-axis case and runs the axis-0 kernel unchanged;
        /// `inner > 1` walks the strided axis with no transpose and no copy (module docs).
        /// The scratch is sized by [`softmax_axis_scratch_len`].
        ///
        /// # Errors
        ///
        /// [`SoftmaxError`] when `axis_len` is zero, a slice does not hold
        /// `shape.inner * shape.axis_len * shape.outer` elements, or the scratch is too small.
        pub fn $fwd(
            shape: AxisShape,
            input: &[$t],
            output: &mut [MaybeUninit<$t>],
            scratch: &mut [$t],
            exec: Exec<'_>,
        ) -> Result<(), SoftmaxError> {
            forward(shape, input, output, scratch, exec)
        }

        /// Softmax VJP over the middle axis of `[inner, axis_len, outer]`:
        /// `dx = y * (dy - sum_axis(y * dy))`, from the forward output `y`.
        ///
        /// The scratch is sized by [`softmax_vjp_axis_scratch_len`].
        ///
        /// # Errors
        ///
        /// [`SoftmaxError`] on a zero axis, a slice of the wrong length, or too little scratch.
        pub fn $vjp(
            shape: AxisShape,
            y: &[$t],
            dy: &[$t],
            dx: &mut [$t],
            scratch: &mut [$t],
            exec: Exec<'_>,
        ) -> Result<(), SoftmaxError> {
            vjp(shape, y, dy, dx, scratch, exec)
        }
    };
}
entries!(f32, softmax_axis_f32, softmax_vjp_axis_f32);
entries!(f64, softmax_axis_f64, softmax_vjp_axis_f64);

#[cfg(test)]
mod tests;
