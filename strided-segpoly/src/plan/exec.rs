//! Replay of a compiled plan: the two entry points, the task kernel and the
//! block combine.
//!
//! One fork level, one task list. A task owns a contiguous range of the row
//! space of one unit (one destination block), and the children partition the
//! task list by row count, so a run holding every row spreads over the workers
//! instead of pinning one. Units keep destination rows disjoint (one task family
//! per destination row, and a scattered output's runs are distinct rows), so no
//! task writes another's slots and nothing is atomic.

use core::ops::Range;

use strided_basic::execution::{tree_width, KernelClass, SendPtr};
use strided_basic::{Exec, RawStridedMut, RawStridedRef, StridedError};

use crate::error::SegPolyError;
use crate::spec::IndexMode;

use super::{
    check_gather, check_gather_length, SegPolyElem, SegPolyPlan, TaskSlot, MAX_INPUTS, MAX_OUTPUTS,
};

/// How one input buffer finds its row for a batch point.
#[derive(Clone, Copy)]
enum RowIdx<'a> {
    Batched,
    Shared,
    I32(&'a [i32]),
    I64(&'a [i64]),
}

/// One call's raw bases and topology, built once per call on the stack.
struct Call<'a, T> {
    nin: usize,
    in_base: [SendPtr<T>; MAX_INPUTS],
    in_batch: [isize; MAX_INPUTS],
    in_step: [isize; MAX_INPUTS],
    in_idx: [RowIdx<'a>; MAX_INPUTS],
    out_base: [SendPtr<T>; MAX_OUTPUTS],
    out_batch: [isize; MAX_OUTPUTS],
    out_perm: [Option<&'a [usize]>; MAX_OUTPUTS],
}

impl<T: SegPolyElem> SegPolyPlan<T> {
    /// Execute the overwriting form: every destination block is zeroed, then the
    /// contributions are added in the plan's fixed order (D2). A run with no
    /// rows writes zeros (D7).
    ///
    /// # Errors
    ///
    /// [`SegPolyError::Strided`] carrying
    /// [`StridedError::PlanLayoutMismatch`] when a view does not carry its
    /// compiled extents and strides, [`SegPolyError::IndexOutOfRange`] when a
    /// gather index is negative or past its buffer's batch extent (D3),
    /// [`SegPolyError::Strided`] carrying [`StridedError::ShapeMismatch`] when a
    /// run index describes a different topology, and
    /// [`SegPolyError::ScratchTooSmall`] when the scratch cannot hold the block
    /// partials.
    ///
    /// Inputs may alias one another; an output must not overlap any input or
    /// another output, which is the caller's obligation as for every plan taking
    /// `RawStridedRef` / `RawStridedMut`.
    pub fn execute_into(
        &self,
        outputs: &mut [RawStridedMut<'_, T>],
        inputs: &[RawStridedRef<'_, T>],
        indexes: &[&[IndexMode<'_>]],
        scratch: &mut [T],
        exec: Exec<'_>,
    ) -> Result<(), SegPolyError> {
        self.execute(outputs, inputs, indexes, scratch, exec, true)
    }

    /// Execute the accumulating form: every destination block keeps its value,
    /// which is the first term of the sum, and the contributions are added in
    /// the plan's fixed order (D2). A run with no rows adds nothing (D7).
    ///
    /// # Errors
    ///
    /// As [`execute_into`](Self::execute_into).
    pub fn execute_acc(
        &self,
        outputs: &mut [RawStridedMut<'_, T>],
        inputs: &[RawStridedRef<'_, T>],
        indexes: &[&[IndexMode<'_>]],
        scratch: &mut [T],
        exec: Exec<'_>,
    ) -> Result<(), SegPolyError> {
        self.execute(outputs, inputs, indexes, scratch, exec, false)
    }

    fn execute(
        &self,
        outputs: &mut [RawStridedMut<'_, T>],
        inputs: &[RawStridedRef<'_, T>],
        indexes: &[&[IndexMode<'_>]],
        scratch: &mut [T],
        exec: Exec<'_>,
        into: bool,
    ) -> Result<(), SegPolyError> {
        self.check_call(outputs, inputs, indexes)?;
        if self.tasks.is_empty() || self.tile_elems == 0 {
            return Ok(());
        }
        // The block partials must all exist at once; the child tiles can be
        // capped by what the caller supplied.
        let floor = (self.block_slots + 1).saturating_mul(self.tile_elems);
        if scratch.len() < floor {
            return Err(SegPolyError::ScratchTooSmall {
                need: floor,
                got: scratch.len(),
            });
        }
        let call = self.call(outputs, inputs, indexes);
        let child_tiles = scratch.len() / self.tile_elems - self.block_slots;
        let width = tree_width(&exec, KernelClass::Map, self.macs)
            .get()
            .min(child_tiles)
            .min(self.tasks.len())
            .max(1);
        let scratch_ptr = SendPtr(scratch.as_mut_ptr());
        if width <= 1 || exec.is_serial() {
            self.run_range(&call, 0..self.tasks.len(), 0, scratch_ptr, into);
        } else {
            exec.fork(width, |child, _token| {
                let range = self.task_range(child, width);
                self.run_range(&call, range, child, scratch_ptr, into);
            });
        }
        self.combine(&call, scratch);
        Ok(())
    }

    /// One entry per buffer: its base pointer and how it finds its row.
    fn call<'c>(
        &self,
        outputs: &'c mut [RawStridedMut<'_, T>],
        inputs: &'c [RawStridedRef<'_, T>],
        indexes: &'c [&[IndexMode<'_>]],
    ) -> Call<'c, T> {
        let mut call = Call {
            nin: self.num_inputs,
            in_base: [SendPtr(core::ptr::null_mut()); MAX_INPUTS],
            in_batch: [0; MAX_INPUTS],
            in_step: [0; MAX_INPUTS],
            in_idx: [RowIdx::Batched; MAX_INPUTS],
            out_base: [SendPtr(core::ptr::null_mut()); MAX_OUTPUTS],
            out_batch: [0; MAX_OUTPUTS],
            out_perm: [None; MAX_OUTPUTS],
        };
        for (buffer, modes) in indexes.iter().enumerate().take(self.num_inputs) {
            call.in_base[buffer] = SendPtr(inputs[buffer].ptr() as *mut T);
            call.in_batch[buffer] = self.strides[buffer][0];
            call.in_step[buffer] = self.in_lane[buffer];
            call.in_idx[buffer] = match &modes[0] {
                IndexMode::Batched => RowIdx::Batched,
                IndexMode::Shared => RowIdx::Shared,
                IndexMode::Gather(index) => RowIdx::I32(index),
                IndexMode::Gather64(index) => RowIdx::I64(index),
                IndexMode::Scatter(_) => unreachable!("role checked at compile"),
            };
        }
        for (dense, modes) in indexes.iter().enumerate().skip(self.num_inputs) {
            let dense = dense - self.num_inputs;
            call.out_base[dense] = SendPtr(outputs[dense].as_mut_ptr());
            call.out_batch[dense] = self.strides[self.num_inputs + dense][0];
            call.out_perm[dense] = match &modes[0] {
                IndexMode::Scatter(runs) => runs.perm(),
                _ => None,
            };
        }
        call
    }

    /// Validate the call's views and topology against the compiled plan.
    fn check_call(
        &self,
        outputs: &[RawStridedMut<'_, T>],
        inputs: &[RawStridedRef<'_, T>],
        indexes: &[&[IndexMode<'_>]],
    ) -> Result<(), SegPolyError> {
        if inputs.len() != self.num_inputs || outputs.len() + inputs.len() != self.dims.len() {
            return Err(SegPolyError::BufferModeCount {
                buffer: inputs.len() + outputs.len(),
                modes: self.dims.len(),
                axes: 1,
            });
        }
        for (buffer, (dims, strides)) in inputs
            .iter()
            .map(|view| (view.dims(), view.strides()))
            .chain(outputs.iter().map(|view| (view.dims(), view.strides())))
            .enumerate()
        {
            if dims != self.dims[buffer].as_slice() || strides != self.strides[buffer].as_slice() {
                return Err(StridedError::PlanLayoutMismatch.into());
            }
        }
        if indexes.len() != self.dims.len() {
            return Err(SegPolyError::BufferModeCount {
                buffer: indexes.len(),
                modes: self.dims.len(),
                axes: 1,
            });
        }
        for (buffer, modes) in indexes.iter().enumerate() {
            if modes.len() != 1 {
                return Err(SegPolyError::BufferModeCount {
                    buffer,
                    modes: modes.len(),
                    axes: 1,
                });
            }
            // The row mapping decides the result, so a call must index a buffer
            // the way the plan was compiled for.
            if modes[0].kind() != self.mode_kinds[buffer] {
                return Err(SegPolyError::IndexModeNotAllowed { buffer });
            }
            match &modes[0] {
                IndexMode::Batched | IndexMode::Shared => {}
                IndexMode::Gather(index) => {
                    check_gather_length(buffer, index.len(), self.batch)?;
                    check_gather(buffer, index, self.dims[buffer][0])?;
                }
                IndexMode::Gather64(index) => {
                    check_gather_length(buffer, index.len(), self.batch)?;
                    check_gather(buffer, index, self.dims[buffer][0])?;
                }
                IndexMode::Scatter(runs) => {
                    let dense = buffer - self.num_inputs;
                    if runs.ptr().len().saturating_sub(1) != self.out_rows[dense] {
                        return Err(StridedError::ShapeMismatch(
                            vec![runs.ptr().len() - 1],
                            vec![self.out_rows[dense]],
                        )
                        .into());
                    }
                    for (run, unit) in self
                        .units
                        .iter()
                        .filter(|unit| unit.out == dense)
                        .enumerate()
                    {
                        if runs.ptr()[run] != unit.rows.start
                            || runs.ptr()[run + 1] != unit.rows.end
                        {
                            return Err(StridedError::ShapeMismatch(
                                vec![runs.ptr()[run], runs.ptr()[run + 1]],
                                vec![unit.rows.start, unit.rows.end],
                            )
                            .into());
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The tasks of one child: a contiguous range, balanced by row count.
    fn task_range(&self, child: usize, width: usize) -> Range<usize> {
        let total = self.tasks.len();
        let start = self.task_boundary(child, width, total);
        let end = self.task_boundary(child + 1, width, total);
        start..end
    }

    fn task_boundary(&self, index: usize, width: usize, total: usize) -> usize {
        if index == 0 {
            return 0;
        }
        if index >= width {
            return total;
        }
        let rows = self.cum_rows[total];
        let target = (rows as u128 * index as u128 / width as u128) as usize;
        // Strictly below: a task that starts exactly at the target belongs to
        // the next child, or a balanced split would hand every task to the first.
        self.cum_rows
            .partition_point(|&before| before < target)
            .min(total)
    }

    fn run_range(
        &self,
        call: &Call<'_, T>,
        tasks: Range<usize>,
        child: usize,
        scratch: SendPtr<T>,
        into: bool,
    ) {
        for index in tasks {
            // SAFETY: the child owns its own tile, and the task carries its slot.
            unsafe { self.run_task(call, index, child, scratch, into) };
        }
    }

    /// Sum one task's rows into its accumulator.
    ///
    /// # Safety
    ///
    /// `check_call` matched every view to the compiled layouts; the scratch is at
    /// least `(block_slots + 1) * tile_elems` elements and `child` is below the
    /// number of children; every destination row this task writes belongs to
    /// this task alone.
    unsafe fn run_task(
        &self,
        call: &Call<'_, T>,
        index: usize,
        child: usize,
        scratch: SendPtr<T>,
        into: bool,
    ) {
        let task = &self.tasks[index];
        let unit = &self.units[task.unit as usize];
        let dense = unit.out;
        let buffer = self.num_inputs + dense;
        let segments = self.dims[buffer][1];
        let u = self.u;
        let len = segments * u;
        let dest_seg = self.strides[buffer][1];
        let dest_lane = self.strides[buffer][2];
        let direct = self.out_direct[dense];
        // An empty output has nothing to write, and its base pointer is a
        // dangling empty view's, so it must not be offset at all.
        if len == 0 {
            return;
        }
        let dest_base = unsafe {
            call.out_base[dense]
                .as_ptr()
                .offset(unit.dest_row as isize * call.out_batch[dense])
        };
        let acc_ptr = match task.slot {
            TaskSlot::Direct => dest_base,
            TaskSlot::Block(slot) => unsafe {
                scratch.as_ptr().add(slot as usize * self.tile_elems)
            },
            TaskSlot::Child => unsafe {
                scratch
                    .as_ptr()
                    .add((self.block_slots + child) * self.tile_elems)
            },
        };
        // SAFETY: the accumulator slot is `tile_elems >= len` elements and owned
        // by this task.
        let acc = unsafe { core::slice::from_raw_parts_mut(acc_ptr, len) };
        if into {
            acc.fill(T::zero());
        } else {
            match task.slot {
                // A `Direct` accumulator *is* the destination block, so seeding it
                // with its own value would alias the copy.
                TaskSlot::Direct => {}
                // Only block zero of a unit continues from the destination value;
                // the other partials are contributions and start at zero. Seeding
                // every partial would add the destination once per block.
                TaskSlot::Block(slot) if slot != unit.slot_base => acc.fill(T::zero()),
                TaskSlot::Block(_) | TaskSlot::Child => {
                    // SAFETY: the destination block belongs to this task's unit.
                    unsafe { read_block(dest_base, dest_seg, dest_lane, segments, u, direct, acc) };
                }
            }
        }

        let perm = call.out_perm[dense];
        let mut rows = [0isize; MAX_INPUTS];
        for row in task.rows.clone() {
            let space = unit.rows.start + row;
            let point = match perm {
                // SAFETY: `check_call` matched the run topology, so `space` is a
                // row of this output's `RunIndex`.
                Some(perm) => unsafe { *perm.get_unchecked(space) },
                None => space,
            };
            for (slot, (mode, stride)) in rows
                .iter_mut()
                .zip(call.in_idx.iter().zip(call.in_batch.iter()))
                .take(call.nin)
            {
                let row = match *mode {
                    RowIdx::Batched => point as isize,
                    RowIdx::Shared => 0,
                    // SAFETY: index ranges are validated before the fork, so
                    // every read below the index length is in bounds.
                    RowIdx::I32(index) => (unsafe { *index.get_unchecked(point) }) as isize,
                    RowIdx::I64(index) => (unsafe { *index.get_unchecked(point) }) as isize,
                };
                *slot = row * *stride;
            }
            for &op_index in self.out_ops[dense].iter() {
                let op = &self.ops[op_index];
                let arity = op.inputs.len() + 1;
                // Contiguous lanes at arity one to three: the row bases are
                // resolved once here and the paths walk slices, which is the
                // same arithmetic in the same order as `accumulate` below.
                if op.unit_lanes && (1..=3).contains(&op.inputs.len()) {
                    let paths = op.base..op.base + op.num_paths();
                    let offsets = op.offsets_base..op.offsets_base + op.num_paths() * arity;
                    macro_rules! contiguous {
                        ($n:literal) => {{
                            let bases: [*const T; $n] = core::array::from_fn(|t| {
                                let input = op.inputs[t] as usize;
                                // SAFETY: `rows` holds this row's base for that
                                // buffer, as in the general path below.
                                unsafe { call.in_base[input].as_const().offset(rows[input]) }
                            });
                            // SAFETY: as for `accumulate`; every input has lane
                            // step 1, so each path reads `u` contiguous lanes.
                            unsafe {
                                op_row_contiguous::<T, $n>(
                                    acc,
                                    u,
                                    &op.seg_starts,
                                    &op.seg_ends,
                                    &self.path_coeffs[paths],
                                    &self.path_offsets[offsets],
                                    &bases,
                                )
                            }
                        }};
                    }
                    match op.inputs.len() {
                        1 => contiguous!(1),
                        2 => contiguous!(2),
                        _ => contiguous!(3),
                    }
                    continue;
                }
                let mut steps = [0isize; MAX_INPUTS];
                for (slot, &input) in steps.iter_mut().zip(op.inputs.iter()) {
                    *slot = self.in_lane[input as usize];
                }
                for segment in 0..op.seg_starts.len() {
                    let lane_base = segment * u;
                    for local in op.seg_starts[segment]..op.seg_ends[segment] {
                        let path = op.base + local as usize;
                        let coefficient = self.path_coeffs[path];
                        let offsets_base = op.offsets_base + local as usize * arity;
                        let offsets = &self.path_offsets[offsets_base..offsets_base + arity];
                        let mut pointers = [core::ptr::null::<T>(); MAX_INPUTS];
                        for (slot, (&input, &offset)) in pointers
                            .iter_mut()
                            .zip(op.inputs.iter().zip(offsets.iter()))
                        {
                            let input = input as usize;
                            // SAFETY: `rows` holds this row's base for that buffer,
                            // and the path's segment offset was validated at
                            // compile against the buffer's segment count.
                            *slot = unsafe {
                                call.in_base[input].as_const().offset(rows[input] + offset)
                            };
                        }
                        // SAFETY: every pointer reaches this buffer's `u` lanes of
                        // the current row (or one lane, stepped by zero), the
                        // accumulator holds `segments * u` elements, and
                        // `lane_base + u <= segments * u`.
                        unsafe {
                            accumulate(
                                acc,
                                lane_base,
                                u,
                                coefficient,
                                &pointers,
                                op.inputs.len(),
                                &steps,
                            )
                        };
                    }
                }
            }
        }
        if task.slot == TaskSlot::Child {
            // SAFETY: the destination block belongs to this task's unit.
            unsafe { write_block(dest_base, dest_seg, dest_lane, segments, u, direct, acc) };
        }
    }

    /// Combine the block partials of every multi-block unit, left to right, and
    /// write the result out.
    ///
    /// Sequential on the calling thread: it runs after the fork joins, and the
    /// work is one add per partial element, which no pool primitive indexes.
    fn combine(&self, call: &Call<'_, T>, scratch: &mut [T]) {
        for unit in &self.units {
            if unit.blocks <= 1 {
                continue;
            }
            let buffer = self.num_inputs + unit.out;
            let segments = self.dims[buffer][1];
            let u = self.u;
            let len = segments * u;
            if len == 0 {
                continue;
            }
            let base = unit.slot_base as usize * self.tile_elems;
            let (head, tail) = scratch.split_at_mut(base + self.tile_elems);
            let acc = &mut head[base..base + len];
            for block in 1..unit.blocks {
                let other = (block - 1) * self.tile_elems;
                for (slot, &value) in acc.iter_mut().zip(tail[other..other + len].iter()) {
                    *slot = *slot + value;
                }
            }
            let dest_base = unsafe {
                call.out_base[unit.out]
                    .as_ptr()
                    .offset(unit.dest_row as isize * call.out_batch[unit.out])
            };
            // SAFETY: the destination block belongs to this unit alone.
            unsafe {
                write_block(
                    dest_base,
                    self.strides[buffer][1],
                    self.strides[buffer][2],
                    segments,
                    u,
                    self.out_direct[unit.out],
                    acc,
                )
            };
        }
    }
}

/// `acc[lane_base + l] += coefficient * Π_t pointers[t][l * steps[t]]`.
///
/// # Safety
///
/// `lane_base + u <= acc.len()`, and every pointer must reach `u` lanes of `T`
/// at stride `steps[t]` (a step of zero reads one lane `u` times, which is the
/// scalar broadcast).
#[inline]
unsafe fn accumulate<T: SegPolyElem>(
    acc: &mut [T],
    lane_base: usize,
    u: usize,
    coefficient: T,
    pointers: &[*const T; MAX_INPUTS],
    nin: usize,
    steps: &[isize; MAX_INPUTS],
) {
    /// Lanes materialised into a stack tile, so the read phase and the add phase
    /// are separate loops over `u`: the read phase writes only the local tile and
    /// the add phase reads only it, which is what lets the compiler vectorise
    /// both. Past this width the lane loop accumulates directly.
    ///
    /// `shortcut: the lane loop is not SIMD-dispatched (no pulp kernel), and this
    /// general path (a lane step other than 1, a broadcast operand, or more than
    /// three operands) indexes lanes by a runtime step, so it vectorises poorly;
    /// lane-contiguous operations of arity one to three take
    /// `op_row_contiguous` instead. Specialise the broadcast case when it is the
    /// measured bottleneck` (strided-rs#479).
    const LANE_TILE: usize = 64;

    // Sliced to the operation's arity, so the inner loop is a zip over two slices
    // instead of two indexed reads against a fixed capacity (which costs a bounds
    // check per operand per lane).
    let pointers = &pointers[..nin];
    let steps = &steps[..nin];

    // The arities a lowered polynomial actually uses: one, two or three operands.
    // Unrolling the lane loop for them removes the runtime arity from the inner
    // most code, which is most of the per-path cost at a narrow lane axis.
    macro_rules! unrolled {
        ($($pointer:expr => $step:expr),*) => {{
            for lane in 0..u {
                let mut product = coefficient;
                $( product = product * unsafe { *$pointer.offset(lane as isize * $step) }; )*
                let slot = lane_base + lane;
                acc[slot] = acc[slot] + product;
            }
            return;
        }};
    }
    match pointers.len() {
        0 => {
            for slot in &mut acc[lane_base..lane_base + u] {
                *slot = *slot + coefficient;
            }
            return;
        }
        1 => unrolled!(pointers[0] => steps[0]),
        2 => unrolled!(pointers[0] => steps[0], pointers[1] => steps[1]),
        3 => unrolled!(pointers[0] => steps[0], pointers[1] => steps[1], pointers[2] => steps[2]),
        _ => {}
    }
    if u <= LANE_TILE {
        // Uninitialised, because the read loop writes every slot the add loop
        // reads: zero-filling the whole tile costs more than the arithmetic of a
        // narrow lane axis (measured on `chain-8k`/`small-128`, see the bench).
        let mut lanes = [core::mem::MaybeUninit::<T>::uninit(); LANE_TILE];
        let tile = lanes.as_mut_ptr().cast::<T>();
        for lane in 0..u {
            let mut product = coefficient;
            for (&pointer, &step) in pointers.iter().zip(steps.iter()) {
                // SAFETY: the caller's pointer contract.
                product = product * unsafe { *pointer.offset(lane as isize * step) };
            }
            // SAFETY: `lane < u <= LANE_TILE`.
            unsafe { tile.add(lane).write(product) };
        }
        for lane in 0..u {
            // SAFETY: written by the loop above.
            let value = unsafe { tile.add(lane).read() };
            let slot = lane_base + lane;
            acc[slot] = acc[slot] + value;
        }
        return;
    }
    for lane in 0..u {
        let mut product = coefficient;
        for (&pointer, &step) in pointers.iter().zip(steps.iter()) {
            // SAFETY: the caller's pointer contract.
            product = product * unsafe { *pointer.offset(lane as isize * step) };
        }
        let slot = lane_base + lane;
        acc[slot] = acc[slot] + product;
    }
}

/// One operation's paths for one batch point, every input lane-contiguous.
///
/// The same sum as calling [`accumulate`] once per path in plan order: for each
/// output segment, for each of its paths, `acc[lane] = acc[lane] + ((c · x_0) ·
/// x_1) · x_2`, the product associated left to right from the coefficient. Only
/// the bookkeeping moves: the row bases come in resolved, the arity is a
/// constant, and each path's operands are slices of `u` lanes, so the lane loop
/// has no index arithmetic or bounds checks and the compiler can vectorise it.
///
/// # Safety
///
/// `acc` holds `seg_starts.len() * u` elements; for every path `p` of
/// `coeffs` and every operand `t < N`, `bases[t].offset(offsets[p * (N + 1) +
/// t])` reaches `u` contiguous, initialised elements.
#[inline(always)]
unsafe fn op_row_contiguous<T: SegPolyElem, const N: usize>(
    acc: &mut [T],
    u: usize,
    seg_starts: &[u32],
    seg_ends: &[u32],
    coeffs: &[T],
    offsets: &[isize],
    bases: &[*const T; N],
) {
    let arity = N + 1;
    for (segment, (&start, &end)) in seg_starts.iter().zip(seg_ends.iter()).enumerate() {
        if start == end {
            continue;
        }
        let lane_base = segment * u;
        let slots = &mut acc[lane_base..lane_base + u];
        for local in start as usize..end as usize {
            let coefficient = coeffs[local];
            let path_offsets = &offsets[local * arity..local * arity + N];
            // SAFETY: the caller's contract; each operand is `u` contiguous lanes.
            let operand = |t: usize| unsafe {
                core::slice::from_raw_parts(bases[t].offset(path_offsets[t]), u)
            };
            match N {
                1 => {
                    let x0 = operand(0);
                    for (slot, &a) in slots.iter_mut().zip(x0) {
                        *slot = *slot + coefficient * a;
                    }
                }
                2 => {
                    let (x0, x1) = (operand(0), operand(1));
                    for ((slot, &a), &b) in slots.iter_mut().zip(x0).zip(x1) {
                        *slot = *slot + coefficient * a * b;
                    }
                }
                _ => {
                    let (x0, x1, x2) = (operand(0), operand(1), operand(2));
                    for (((slot, &a), &b), &c) in slots.iter_mut().zip(x0).zip(x1).zip(x2) {
                        *slot = *slot + coefficient * a * b * c;
                    }
                }
            }
        }
    }
}

/// Copy a destination block into a packed accumulator.
///
/// # Safety
///
/// `dest` must reach the whole `segments x u` block, and `out` must hold
/// `segments * u` elements.
unsafe fn read_block<T: Copy>(
    dest: *const T,
    seg: isize,
    lane: isize,
    segments: usize,
    u: usize,
    direct: bool,
    out: &mut [T],
) {
    let len = segments * u;
    if direct {
        // SAFETY: a contiguous block is one run of `len` slots.
        let block = unsafe { core::slice::from_raw_parts(dest, len) };
        out[..len].copy_from_slice(block);
        return;
    }
    for segment in 0..segments {
        for lane_index in 0..u {
            // SAFETY: the block's own layout, per the caller's contract.
            out[segment * u + lane_index] =
                unsafe { *dest.offset(segment as isize * seg + lane_index as isize * lane) };
        }
    }
}

/// Write a packed accumulator out to a destination block.
///
/// # Safety
///
/// As [`read_block`], in the other direction.
unsafe fn write_block<T: Copy>(
    dest: *mut T,
    seg: isize,
    lane: isize,
    segments: usize,
    u: usize,
    direct: bool,
    values: &[T],
) {
    let len = segments * u;
    if direct {
        // SAFETY: a contiguous block is one run of `len` slots.
        let block = unsafe { core::slice::from_raw_parts_mut(dest, len) };
        block.copy_from_slice(&values[..len]);
        return;
    }
    for segment in 0..segments {
        for lane_index in 0..u {
            // SAFETY: the block's own layout, per the caller's contract.
            unsafe {
                *dest.offset(segment as isize * seg + lane_index as isize * lane) =
                    values[segment * u + lane_index];
            }
        }
    }
}
