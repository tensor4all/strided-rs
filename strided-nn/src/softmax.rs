//! Fused axis-0 softmax over contiguous column-major slices.
//!
//! One pass per cache-sized group of columns: max and shift, `exp` written
//! straight into the output, sum of the output, scale in place. There is no
//! full-size intermediate, and the only scratch is one group-sized buffer per
//! task, which the caller provides.
//!
//! The pipeline shape follows PyTorch's `_vec_softmax_lastdim`; the vectorized
//! `exp` comes from `elem-simd-math`, and the `f32` max/sum reductions use AVX2
//! where the host has it, because LLVM does not vectorize a scalar `+=` fold
//! without fast-math.

use core::mem::MaybeUninit;

use strided_basic::execution::SendPtr;
use strided_basic::ExecContext;

use crate::error::SoftmaxError;
#[cfg(target_arch = "x86_64")]
use crate::simd;

/// Bytes in one cache-sized group of columns.
const GROUP_BYTES: usize = 32 * 1024;

/// Minimum work, in bytes, that justifies one worker.
const MIN_BYTES_PER_TASK: usize = 32 * 1024;

/// Elements of caller scratch one softmax call needs for `len` elements and
/// `workers` workers.
///
/// One task holds one cache-sized group of columns; the number of tasks is what
/// `workers` and the work per task allow, not the thread count itself, so a wide
/// pool on a short input does not reserve scratch for workers that never run.
/// The scratch also caps the fan-out: a region per task and no more.
///
/// # Examples
///
/// ```
/// use strided_nn::scratch_len;
/// // Eight cache-sized groups, four of them tasks.
/// assert_eq!(scratch_len::<f32>(1024, 1024 * 1024, 4), 32 * 1024);
/// // One column of one million elements is one task, so one group.
/// assert_eq!(scratch_len::<f64>(1_000_000, 1_000_000, 64), 1_000_000);
/// ```
pub fn scratch_len<T>(axis_len: usize, len: usize, workers: usize) -> usize {
    let element_size = core::mem::size_of::<T>();
    let group = group_elements(axis_len, element_size);
    group.saturating_mul(tasks_for(len, group, element_size, workers))
}

/// Tasks one call runs: capped by the scratch capacity, the worker budget and
/// the work per task.
fn tasks_for(len: usize, group: usize, element_size: usize, workers: usize) -> usize {
    let groups = len.div_ceil(group).max(1);
    len.saturating_mul(element_size)
        .checked_div(MIN_BYTES_PER_TASK)
        .unwrap_or(0)
        .max(1)
        .min(workers.max(1))
        .min(groups)
}

/// Elements in one cache-sized group of columns.
///
/// A column longer than [`GROUP_BYTES`] stays whole, so a group is never
/// smaller than one column.
fn group_elements(axis_len: usize, element_size: usize) -> usize {
    match axis_len.checked_mul(element_size) {
        Some(bytes) if bytes < GROUP_BYTES => (GROUP_BYTES / bytes).max(1) * axis_len,
        _ => axis_len,
    }
}

/// Task split of one softmax call.
struct Plan {
    /// Column length.
    axis_len: usize,
    /// Elements in the input and output.
    len: usize,
    /// Elements in one cache-sized group of whole columns.
    group: usize,
    /// Elements in one task; a whole number of groups.
    task: usize,
    /// Number of tasks, never more than the caller's scratch allows.
    tasks: usize,
}

impl Plan {
    fn new(
        axis_len: usize,
        input_len: usize,
        output_len: usize,
        scratch_len: usize,
        element_size: usize,
    ) -> Result<Self, SoftmaxError> {
        if axis_len == 0 {
            return Err(SoftmaxError::EmptyAxis);
        }
        if input_len != output_len {
            return Err(SoftmaxError::LengthMismatch {
                input: input_len,
                output: output_len,
            });
        }
        if input_len % axis_len != 0 {
            return Err(SoftmaxError::UnevenColumns {
                len: input_len,
                axis_len,
            });
        }
        if axis_len.checked_mul(element_size).is_none() {
            return Err(SoftmaxError::SizeOverflow {
                axis_len,
                element_size,
            });
        }
        let group = group_elements(axis_len, element_size);
        // The scratch capacity decides how many tasks may run at once; the policy
        // in the `ExecContext` clamps that further.
        let capacity = scratch_len / group;
        if capacity == 0 {
            return Err(SoftmaxError::ScratchTooSmall {
                need: group,
                got: scratch_len,
            });
        }
        let groups = input_len.div_ceil(group);
        let workers = tasks_for(input_len, group, element_size, capacity);
        let groups_per_task = groups.div_ceil(workers).max(1);
        let task = (group * groups_per_task).min(input_len).max(1);
        Ok(Self {
            axis_len,
            len: input_len,
            group,
            task,
            tasks: input_len.div_ceil(task).max(1),
        })
    }
}

/// Hand each task index to `body` on the context's pool within its budget, or
/// inline on the calling thread under a serial context.
fn fan_out<F: Fn(usize) + Sync>(tasks: usize, body: F) {
    strided_basic::execution::parallel_map_reduce(
        0..tasks,
        tasks,
        &|range| {
            for task in range {
                body(task);
            }
        },
        &|(), ()| (),
    );
}

/// The initialized view of an output chunk that `exp_into` has filled.
///
/// # Safety
/// Every element of `chunk` must have been written.
unsafe fn initialized<T>(chunk: &mut [MaybeUninit<T>]) -> &mut [T] {
    unsafe { core::slice::from_raw_parts_mut(chunk.as_mut_ptr().cast::<T>(), chunk.len()) }
}

/// Generate one dtype's public entry, its scalar group functions and the shared
/// task loop.
macro_rules! softmax_axis0 {
    ($(#[$doc:meta])* $name:ident, $scalar:ty, $size:expr, $shift:ident, $scale:ident, $exp_into:ident, $vector_exp:path) => {
        $(#[$doc])*
        ///
        /// # Errors
        ///
        /// [`SoftmaxError`] when the slices do not describe whole `axis_len`
        /// columns, have different lengths, or the scratch holds less than one
        /// cache-sized group.
        ///
        /// The scratch must hold at least one cache-sized group of columns;
        /// [`scratch_len`] sizes it for a worker count.
        pub fn $name(
            exec: ExecContext,
            axis_len: usize,
            input: &[$scalar],
            output: &mut [MaybeUninit<$scalar>],
            scratch: &mut [$scalar],
        ) -> Result<(), SoftmaxError> {
            if input.is_empty() {
                return Ok(());
            }
            let plan = Plan::new(axis_len, input.len(), output.len(), scratch.len(), $size)?;
            let out_ptr = SendPtr(output.as_mut_ptr());
            let scratch_ptr = SendPtr(scratch.as_mut_ptr());
            let body = |task: usize| {
                let start = task * plan.task;
                let end = (start + plan.task).min(plan.len);
                let len = end - start;
                // SAFETY: the task spans partition `0..len`, so this task writes
                // only its own output elements.
                let out_task =
                    unsafe { core::slice::from_raw_parts_mut(out_ptr.as_ptr().add(start), len) };
                // SAFETY: as above, for the task's own scratch region; the plan
                // capped `tasks` by the scratch length, so `group <= temp` fits.
                let temp = unsafe {
                    core::slice::from_raw_parts_mut(
                        scratch_ptr.as_ptr().add(task * plan.group),
                        plan.group.min(len),
                    )
                };
                for (out_chunk, in_chunk) in out_task
                    .chunks_mut(plan.group)
                    .zip(input[start..end].chunks(plan.group))
                {
                    let temp = &mut temp[..in_chunk.len()];
                    $shift(in_chunk, temp, plan.axis_len);
                    $exp_into(temp, out_chunk);
                    // SAFETY: `$exp_into` wrote every element of the chunk.
                    let out_chunk = unsafe { initialized(out_chunk) };
                    $scale(out_chunk, plan.axis_len);
                }
            };
            exec.run(|| fan_out(plan.tasks, &body));
            Ok(())
        }

        /// `exp` of `temp` into `out`, every element written.
        fn $exp_into(temp: &[$scalar], out: &mut [MaybeUninit<$scalar>]) {
            if $vector_exp(temp, out).is_err() {
                for (slot, &value) in out.iter_mut().zip(temp) {
                    slot.write(<$scalar>::exp(value));
                }
            }
        }
    };
}

softmax_axis0!(
    /// Fused softmax over axis 0 of a contiguous column-major `f32` slice.
    softmax_axis0_f32,
    f32,
    4,
    shift_group_f32_best,
    scale_group_f32_best,
    exp_into_f32,
    elem_simd_math::f32::exp
);

softmax_axis0!(
    /// Fused softmax over axis 0 of a contiguous column-major `f64` slice.
    softmax_axis0_f64,
    f64,
    8,
    shift_group_f64,
    scale_group_f64,
    exp_into_f64,
    elem_simd_math::f64::exp
);

/// The AVX2 shift where the host has it, the scalar group function elsewhere.
fn shift_group_f32_best(src: &[f32], dst: &mut [f32], axis_len: usize) {
    #[cfg(target_arch = "x86_64")]
    if simd::avx2_available() {
        // SAFETY: the host reports AVX2, and the group function touches only the
        // validated slices.
        unsafe { simd::shift_group_f32_avx2(src, dst, axis_len) };
        return;
    }
    shift_group_f32(src, dst, axis_len);
}

/// The AVX2 scale where the host has it, the scalar group function elsewhere.
fn scale_group_f32_best(data: &mut [f32], axis_len: usize) {
    #[cfg(target_arch = "x86_64")]
    if simd::avx2_available() {
        // SAFETY: as `shift_group_f32_best`.
        unsafe { simd::scale_group_f32_avx2(data, axis_len) };
        return;
    }
    scale_group_f32(data, axis_len);
}

fn shift_group_f32(src: &[f32], dst: &mut [f32], axis_len: usize) {
    for (in_slice, out_slice) in src.chunks(axis_len).zip(dst.chunks_mut(axis_len)) {
        let max = in_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let safe_max = if max.is_nan() { max } else { max.max(f32::MIN) };
        for (out, &value) in out_slice.iter_mut().zip(in_slice) {
            *out = value - safe_max;
        }
    }
}

fn scale_group_f32(data: &mut [f32], axis_len: usize) {
    for slice in data.chunks_mut(axis_len) {
        let sum: f32 = slice.iter().copied().sum();
        let safe_sum = if sum.is_nan() {
            sum
        } else {
            sum.max(f32::MIN_POSITIVE)
        };
        for value in slice.iter_mut() {
            *value /= safe_sum;
        }
    }
}

fn shift_group_f64(src: &[f64], dst: &mut [f64], axis_len: usize) {
    for (in_slice, out_slice) in src.chunks(axis_len).zip(dst.chunks_mut(axis_len)) {
        let max = in_slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let safe_max = if max.is_nan() { max } else { max.max(f64::MIN) };
        for (out, &value) in out_slice.iter_mut().zip(in_slice) {
            *out = value - safe_max;
        }
    }
}

fn scale_group_f64(data: &mut [f64], axis_len: usize) {
    for slice in data.chunks_mut(axis_len) {
        let sum: f64 = slice.iter().copied().sum();
        let safe_sum = if sum.is_nan() {
            sum
        } else {
            sum.max(f64::MIN_POSITIVE)
        };
        for value in slice.iter_mut() {
            *value /= safe_sum;
        }
    }
}

#[cfg(test)]
mod tests;
