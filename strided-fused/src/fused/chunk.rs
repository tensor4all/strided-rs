//! Chunked vectorized replay of a fused elementwise plan.
//!
//! The per-element interpreter in [`super`] gathers every operand through its
//! runtime stride and evaluates one element per instruction, which is why a
//! fused region is slower than dispatching the same instructions one at a time
//! through this workspace's vector kernels (tensor4all/strided-rs#304).
//!
//! This module replays a **contiguous block** — every output unit-stride along
//! the block's innermost axis and every input unit-stride or broadcast (stride
//! 0) along it — through chunk-sized register buffers: load each input chunk
//! once, run every instruction over the chunk with a monomorphized kernel, store
//! the output chunks once. Values in the large arrays are read and written once,
//! and the intermediates stay in L1. A stride-0 input has one value for the whole
//! block, so its register chunk is filled from that value once per block.
//!
//! A block with any other stride, a rank-zero block, and any scalar that is not
//! `f32`/`f64` keep the per-element interpreter.

use core::mem::{align_of, size_of, MaybeUninit};
use std::any::TypeId;

use super::{eval_binary, eval_unary, FusedOp, FusedPlan, FusedScalar};

/// Elements one register chunk holds. Measured across 256, 1024, 2048, 4096 and
/// 8192 on EPYC 7713P: the fused-to-dispatch ratio moved by less than 4%, so
/// the size only has to keep the two registers one instruction touches in L1
/// (8 KiB of `f64` here) while keeping the kernel call count small.
pub(super) const FUSED_CHUNK_ELEMS: usize = 1024;

/// Elements a register file must hold for `plan`.
#[inline]
pub(super) fn register_file_len(plan: &FusedPlan) -> usize {
    (plan.input_count + plan.ops.len()) * FUSED_CHUNK_ELEMS
}

/// True when the block can be replayed through the register machine.
///
/// `strides` carries one entry per operand (outputs first, then inputs). Every
/// output stride must be 1, and every input stride must be 1 (a contiguous
/// load) or 0 (a broadcast value). Any other stride falls back to the
/// per-element interpreter, as does a rank-zero block, whose `strides` is empty.
#[inline]
pub(super) fn block_replay_eligible(
    strides: &[isize],
    output_count: usize,
    operands: usize,
) -> bool {
    if strides.len() != operands || strides.is_empty() {
        return false;
    }
    let (outputs, inputs) = strides.split_at(output_count);
    outputs.iter().all(|&stride| stride == 1)
        && inputs.iter().all(|&stride| stride == 0 || stride == 1)
}

/// Input registers that exactly one instruction reads. Their chunk is read
/// straight from the operand instead of being copied into the register file.
///
/// A destination may itself name an input register (an identity pass-through
/// output), and the store step reads it after the instructions have run, so a
/// register the plan outputs must stay in the file.
pub(super) fn single_use_inputs(plan: &FusedPlan) -> strided_basic::OperandVec<bool> {
    let mut reads: strided_basic::OperandVec<usize> =
        strided_basic::OperandVec::with_capacity(plan.input_count);
    reads.resize(plan.input_count, 0);
    for inst in &plan.ops {
        for &input in inst.inputs.iter() {
            if input < plan.input_count {
                reads[input] += 1;
            }
        }
    }
    reads
        .into_iter()
        .enumerate()
        .map(|(index, count)| count == 1 && !plan.outputs.contains(&index))
        .collect()
}

/// True when `T` has vectorized chunk kernels.
#[inline]
pub(super) fn is_real_chunk_scalar<T: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>() || TypeId::of::<T>() == TypeId::of::<f64>()
}

/// Register file seeded with `seed`. Every chunk overwrites the registers it
/// reads before reading them (SSA order), so the seed is never observed.
///
/// The file is kept in a per-thread buffer and grown once, so a steady-state
/// chunked replay allocates nothing (#306 item 6). A `None` seed declines the
/// chunked path and hands the callback an empty file.
pub(super) fn with_register_file<T: FusedScalar, R>(
    plan: &FusedPlan,
    seed: Option<T>,
    f: impl FnOnce(&mut [T]) -> R,
) -> R {
    use std::cell::RefCell;
    let Some(seed) = seed else {
        return f(&mut []);
    };
    debug_assert!(size_of::<T>() <= size_of::<u128>());
    thread_local! {
        static SCRATCH: RefCell<Vec<MaybeUninit<u128>>> =
            const { RefCell::new(Vec::new()) };
    }
    SCRATCH.with(|cell| {
        let mut buffer = cell.borrow_mut();
        let len = register_file_len(plan);
        let words = (len * size_of::<T>()).div_ceil(size_of::<u128>());
        if buffer.len() < words {
            buffer.resize(words, MaybeUninit::uninit());
        }
        // INITIALIZATION: initialize through `MaybeUninit` first, then form
        // the live slice. Forming `&mut [T]` over uninitialized storage (even
        // to fill it) violates `from_raw_parts_mut`'s validity requirement.
        // SAFETY: the `u128` buffer is aligned for `T` (the chunk path only
        // reaches f32/f64) and the allocation covers `len * size_of::<T>()`
        // bytes.
        let uninit = unsafe {
            core::slice::from_raw_parts_mut(buffer.as_mut_ptr().cast::<MaybeUninit<T>>(), len)
        };
        for slot in uninit.iter_mut() {
            slot.write(seed);
        }
        // SAFETY: every slot was initialized just above.
        let regs = unsafe { core::slice::from_raw_parts_mut(uninit.as_mut_ptr().cast::<T>(), len) };
        f(regs)
    })
}

/// Replay one eligible block into initialized destinations.
///
/// # Safety
/// Each entry of `dst_ptrs` and `input_ptrs` must address `offsets[i]` plus
/// `len` contiguous readable (inputs) or writable (outputs) elements, the plan
/// must be validated for these operand counts, `strides` must satisfy
/// [`block_replay_eligible`], and `regs` must hold [`register_file_len`]
/// elements.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn replay_chunked<T: FusedScalar>(
    dst_ptrs: &[*mut T],
    input_ptrs: &[*const T],
    plan: &FusedPlan,
    offsets: &[isize],
    len: usize,
    strides: &[isize],
    direct: &[bool],
    regs: &mut [T],
) {
    let output_count = dst_ptrs.len();
    // SAFETY: the caller proved every operand addresses this block's `len`
    // contiguous elements, so the broadcast value is readable.
    unsafe { fill_broadcast_inputs(input_ptrs, offsets, output_count, strides, regs) };
    let mut done = 0;
    while done < len {
        let chunk = (len - done).min(FUSED_CHUNK_ELEMS);
        // SAFETY: the caller proved every operand addresses this block's `len`
        // contiguous elements, so the chunk is inside the operand.
        unsafe {
            load_input_chunks(
                input_ptrs,
                offsets,
                output_count,
                direct,
                strides,
                done,
                chunk,
                regs,
            )
        };
        // SAFETY: as for the load.
        unsafe {
            run_chunks(
                plan,
                chunk,
                input_ptrs,
                offsets,
                output_count,
                direct,
                done,
                regs,
            )
        };
        // SAFETY: as for the load.
        unsafe { store_output_chunks(dst_ptrs, plan, offsets, done, chunk, regs) };
        done += chunk;
    }
}

/// Replay one eligible block into uninitialized destinations.
///
/// # Safety
/// As for [`replay_chunked`], with each entry of `dst_ptrs` addressing
/// `offsets[i]` plus `len` writable `MaybeUninit<T>` elements. Every reachable
/// element is written.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn replay_chunked_uninit<T: FusedScalar>(
    dst_ptrs: &[*mut MaybeUninit<T>],
    input_ptrs: &[*const T],
    plan: &FusedPlan,
    offsets: &[isize],
    len: usize,
    strides: &[isize],
    direct: &[bool],
    regs: &mut [T],
) {
    let output_count = dst_ptrs.len();
    // SAFETY: the caller proved every operand addresses this block's `len`
    // contiguous elements, so the broadcast value is readable.
    unsafe { fill_broadcast_inputs(input_ptrs, offsets, output_count, strides, regs) };
    let mut done = 0;
    while done < len {
        let chunk = (len - done).min(FUSED_CHUNK_ELEMS);
        // SAFETY: the caller proved every operand addresses this block's `len`
        // contiguous elements, so the chunk is inside the operand.
        unsafe {
            load_input_chunks(
                input_ptrs,
                offsets,
                output_count,
                direct,
                strides,
                done,
                chunk,
                regs,
            )
        };
        // SAFETY: as for the load.
        unsafe {
            run_chunks(
                plan,
                chunk,
                input_ptrs,
                offsets,
                output_count,
                direct,
                done,
                regs,
            )
        };
        // SAFETY: as for the load, and the destination chunks are writable.
        unsafe { store_output_chunks_uninit(dst_ptrs, plan, offsets, done, chunk, regs) };
        done += chunk;
    }
}

/// Fill the register chunk of every stride-0 input with its single value.
///
/// A stride-0 operand is constant across the block, so its value is read once
/// per block instead of once per chunk. Instructions only write registers
/// `input_count..`, so the filled input register survives every later chunk.
///
/// # Safety
/// Every operand must address a readable element at its block offset.
#[inline]
unsafe fn fill_broadcast_inputs<T: FusedScalar>(
    input_ptrs: &[*const T],
    offsets: &[isize],
    first_operand: usize,
    strides: &[isize],
    regs: &mut [T],
) {
    for (index, &ptr) in input_ptrs.iter().enumerate() {
        if strides[first_operand + index] != 0 {
            continue;
        }
        // SAFETY: the caller proved the operand addresses this block.
        let value = unsafe { *ptr.offset(offsets[first_operand + index]) };
        regs[index * FUSED_CHUNK_ELEMS..][..FUSED_CHUNK_ELEMS].fill(value);
    }
}

/// # Safety
/// See [`replay_chunked`]; `first_operand` is the index of the first input in
/// `offsets` (the output count). Only stride-1 inputs that are not `direct` are
/// copied here; stride-0 inputs were already filled by
/// [`fill_broadcast_inputs`].
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn load_input_chunks<T: FusedScalar>(
    input_ptrs: &[*const T],
    offsets: &[isize],
    first_operand: usize,
    direct: &[bool],
    strides: &[isize],
    done: usize,
    chunk: usize,
    regs: &mut [T],
) {
    for (index, &ptr) in input_ptrs.iter().enumerate() {
        if direct[index] || strides[first_operand + index] == 0 {
            continue;
        }
        // SAFETY: the caller proved the block is contiguous for this operand.
        let src = unsafe {
            std::slice::from_raw_parts(
                ptr.offset(offsets[first_operand + index] + done as isize),
                chunk,
            )
        };
        regs[index * FUSED_CHUNK_ELEMS..][..chunk].copy_from_slice(src);
    }
}

/// # Safety
/// See [`replay_chunked`].
#[inline]
unsafe fn store_output_chunks<T: FusedScalar>(
    dst_ptrs: &[*mut T],
    plan: &FusedPlan,
    offsets: &[isize],
    done: usize,
    chunk: usize,
    regs: &[T],
) {
    for (index, &ptr) in dst_ptrs.iter().enumerate() {
        // SAFETY: the caller proved the block is contiguous for this operand.
        let dst = unsafe {
            std::slice::from_raw_parts_mut(ptr.offset(offsets[index] + done as isize), chunk)
        };
        dst.copy_from_slice(&regs[plan.outputs[index] * FUSED_CHUNK_ELEMS..][..chunk]);
    }
}

/// Store every destination's chunk from its output register.
///
/// # Safety
/// See [`replay_chunked_uninit`].
#[inline]
unsafe fn store_output_chunks_uninit<T: FusedScalar>(
    dst_ptrs: &[*mut MaybeUninit<T>],
    plan: &FusedPlan,
    offsets: &[isize],
    done: usize,
    chunk: usize,
    regs: &[T],
) {
    for (index, &ptr) in dst_ptrs.iter().enumerate() {
        // SAFETY: the caller proved the block is contiguous for this operand.
        let dst = unsafe {
            std::slice::from_raw_parts_mut(ptr.offset(offsets[index] + done as isize), chunk)
        };
        let values = &regs[plan.outputs[index] * FUSED_CHUNK_ELEMS..][..chunk];
        for (slot, &value) in dst.iter_mut().zip(values) {
            slot.write(value);
        }
    }
}

/// Evaluate every instruction over one chunk of the register file.
///
/// Instruction `k` reads registers below `input_count + k` and writes exactly
/// `input_count + k`, which validation guarantees, so splitting the file at the
/// output register keeps the borrows disjoint without raw pointers.
#[allow(clippy::too_many_arguments)]
unsafe fn run_chunks<T: FusedScalar>(
    plan: &FusedPlan,
    chunk: usize,
    input_ptrs: &[*const T],
    offsets: &[isize],
    first_operand: usize,
    direct: &[bool],
    done: usize,
    regs: &mut [T],
) {
    for (instruction, inst) in plan.ops.iter().enumerate() {
        let output = plan.input_count + instruction;
        let (earlier, rest) = regs.split_at_mut(output * FUSED_CHUNK_ELEMS);
        let dest = &mut rest[..chunk];
        let read = |index: usize| -> &[T] {
            if index < plan.input_count && direct[index] {
                // SAFETY: the caller proved the block is contiguous for this
                // operand, and a single-use input has not been written over.
                return unsafe {
                    std::slice::from_raw_parts(
                        input_ptrs[index].offset(offsets[first_operand + index] + done as isize),
                        chunk,
                    )
                };
            }
            &earlier[index * FUSED_CHUNK_ELEMS..][..chunk]
        };
        match inst.inputs.as_slice() {
            [a] => chunk_unary(inst.op, read(*a), dest),
            [a, b] => chunk_binary(inst.op, read(*a), read(*b), dest),
            [a, b, c] => chunk_ternary(inst.op, read(*a), read(*b), read(*c), dest),
            _ => unreachable!("a validated fused plan supplies each op's arity"),
        }
    }
}

/// One unary instruction over a chunk.
fn chunk_unary<T: FusedScalar>(op: FusedOp, src: &[T], dst: &mut [T]) {
    if try_analytic_unary_chunk(op, src, dst) {
        return;
    }
    match op {
        FusedOp::Negate => {
            for (slot, &x) in dst.iter_mut().zip(src) {
                *slot = x.fused_negate();
            }
        }
        FusedOp::Conj => {
            for (slot, &x) in dst.iter_mut().zip(src) {
                *slot = x.fused_conj();
            }
        }
        FusedOp::Abs => {
            for (slot, &x) in dst.iter_mut().zip(src) {
                *slot = x.fused_abs();
            }
        }
        FusedOp::Sqrt => {
            for (slot, &x) in dst.iter_mut().zip(src) {
                *slot = x.fused_sqrt();
            }
        }
        FusedOp::Rsqrt => {
            for (slot, &x) in dst.iter_mut().zip(src) {
                *slot = x.fused_rsqrt();
            }
        }
        // Analytic ops without a chunk kernel for this scalar keep the scalar
        // methods; the plan and dtype combination that reaches them is already
        // the interpreter's.
        _ => {
            for (slot, &x) in dst.iter_mut().zip(src) {
                *slot = eval_unary(op, x);
            }
        }
    }
}

/// One binary instruction over a chunk.
fn chunk_binary<T: FusedScalar>(op: FusedOp, a: &[T], b: &[T], dst: &mut [T]) {
    match op {
        FusedOp::Add => {
            for ((slot, &x), &y) in dst.iter_mut().zip(a).zip(b) {
                *slot = x.fused_add(y);
            }
        }
        FusedOp::Multiply => {
            for ((slot, &x), &y) in dst.iter_mut().zip(a).zip(b) {
                *slot = x.fused_multiply(y);
            }
        }
        FusedOp::Divide => {
            for ((slot, &x), &y) in dst.iter_mut().zip(a).zip(b) {
                *slot = x.fused_divide(y);
            }
        }
        FusedOp::Maximum => {
            for ((slot, &x), &y) in dst.iter_mut().zip(a).zip(b) {
                *slot = x.fused_maximum(y);
            }
        }
        FusedOp::Minimum => {
            for ((slot, &x), &y) in dst.iter_mut().zip(a).zip(b) {
                *slot = x.fused_minimum(y);
            }
        }
        FusedOp::Pow => {
            for ((slot, &x), &y) in dst.iter_mut().zip(a).zip(b) {
                *slot = x.fused_pow(y);
            }
        }
        _ => {
            for ((slot, &x), &y) in dst.iter_mut().zip(a).zip(b) {
                *slot = eval_binary(op, x, y);
            }
        }
    }
}

/// The single ternary instruction over a chunk.
fn chunk_ternary<T: FusedScalar>(op: FusedOp, a: &[T], b: &[T], c: &[T], dst: &mut [T]) {
    if op == FusedOp::Clamp {
        for (((slot, &x), &lo), &hi) in dst.iter_mut().zip(a).zip(b).zip(c) {
            *slot = x.fused_clamp(lo, hi);
        }
        return;
    }
    for (index, slot) in dst.iter_mut().enumerate() {
        *slot = super::eval_ternary(op, a[index], b[index], c[index]);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::{FusedInst, FusedOp};

    #[test]
    fn register_file_is_seeded_before_the_live_slice_is_formed() {
        // The per-thread buffer starts uninitialized; the seed must be written
        // through `MaybeUninit` before `&mut [T]` is formed. Miri flags reading
        // or referencing the uninitialized buffer otherwise.
        let plan = FusedPlan::new(1, vec![2], vec![FusedInst::new(FusedOp::Negate, vec![0])]);
        with_register_file::<f64, _>(&plan, Some(7.0), |regs| {
            assert_eq!(regs.len(), register_file_len(&plan));
            assert!(regs.iter().all(|&value| value == 7.0));
        });
        // A second call reuses the grown buffer and re-seeds it.
        with_register_file::<f64, _>(&plan, Some(-3.5), |regs| {
            assert!(regs.iter().all(|&value| value == -3.5));
        });
    }
}

/// Apply one analytic unary kernel over a chunk, or return `false` when `op`
/// has no vector kernel for `T`.
///
/// `T` is guarded by `TypeId`, and the chunk pointers are reinterpreted, exactly
/// as `strided_basic`'s SIMD dispatch does. The kernels are documented at 1.0
/// ULP; they are the same ones a per-instruction dispatch through
/// `map_into_uninit_with_kernel` uses, so a fused region and the non-fused path
/// it replaces agree with each other.
///
/// `Log1p` deliberately stays on the scalar chunk loop: `elem-simd-math 0.1.0`'s
/// slice kernel overflows to `inf` at `f64::MAX`, where `f64::ln_1p` returns
/// `709.78`. That is a wrong answer, not a one-ULP difference, and it is worth
/// reporting upstream rather than inheriting here.
fn try_analytic_unary_chunk<T: FusedScalar>(op: FusedOp, src: &[T], dst: &mut [T]) -> bool {
    debug_assert_eq!(src.len(), dst.len());

    fn apply<U>(
        op: FusedOp,
        src: &[U],
        dst: &mut [U],
        kernels: impl Fn(FusedOp, &[U], &mut [MaybeUninit<U>]) -> bool,
    ) -> bool {
        // SAFETY: `MaybeUninit<U>` and `U` have the same layout, and every
        // element is written exactly once before it is read back as `U`.
        let uninit = unsafe {
            std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<MaybeUninit<U>>(), dst.len())
        };
        kernels(op, src, uninit)
    }

    if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: the `TypeId` guard proves `T` is `f32`.
        let (src, dst) = unsafe { cast_chunk::<T, f32>(src, dst) };
        return apply(op, src, dst, |op, src, dst| match op {
            FusedOp::Exp => elem_simd_math::f32::exp(src, dst).is_ok(),
            FusedOp::Log => elem_simd_math::f32::log(src, dst).is_ok(),
            FusedOp::Sin => elem_simd_math::f32::sin(src, dst).is_ok(),
            FusedOp::Cos => elem_simd_math::f32::cos(src, dst).is_ok(),
            FusedOp::Tanh => elem_simd_math::f32::tanh(src, dst).is_ok(),
            FusedOp::Expm1 => elem_simd_math::f32::expm1(src, dst).is_ok(),
            _ => false,
        });
    }
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: the `TypeId` guard proves `T` is `f64`.
        let (src, dst) = unsafe { cast_chunk::<T, f64>(src, dst) };
        return apply(op, src, dst, |op, src, dst| match op {
            FusedOp::Exp => elem_simd_math::f64::exp(src, dst).is_ok(),
            FusedOp::Log => elem_simd_math::f64::log(src, dst).is_ok(),
            FusedOp::Sin => elem_simd_math::f64::sin(src, dst).is_ok(),
            FusedOp::Cos => elem_simd_math::f64::cos(src, dst).is_ok(),
            FusedOp::Tanh => elem_simd_math::f64::tanh(src, dst).is_ok(),
            FusedOp::Expm1 => elem_simd_math::f64::expm1(src, dst).is_ok(),
            _ => false,
        });
    }
    false
}

/// # Safety
/// `T` and `U` must be the same type.
#[inline]
unsafe fn cast_chunk<'a, T: 'static, U: 'static>(
    src: &'a [T],
    dst: &'a mut [T],
) -> (&'a [U], &'a mut [U]) {
    debug_assert_eq!(TypeId::of::<T>(), TypeId::of::<U>());
    debug_assert_eq!(size_of::<T>(), size_of::<U>());
    debug_assert_eq!(align_of::<T>(), align_of::<U>());
    let src = unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<U>(), src.len()) };
    let dst = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<U>(), dst.len()) };
    (src, dst)
}
