//! Replay of a compiled plan: the two entry points, the task kernel and the
//! block combine.
//!
//! One fork level, one task list. A task owns a contiguous range of the row
//! space of one unit, and the children partition the task list by row count, so
//! a run holding every row spreads over the workers instead of pinning one. A
//! unit carries one destination row per accumulating output of its group (a
//! scattered group may write several in the one pass; a batched output is
//! streamed row by row), and units keep destination rows disjoint, so no task
//! writes another's slots and nothing is atomic.

use core::ops::Range;

use strided_basic::execution::{tree_width, KernelClass, SendPtr};
use strided_basic::{Exec, RawStridedMut, RawStridedRef, StridedError};

use crate::error::SegPolyError;
use crate::spec::IndexMode;

use super::{
    check_gather, check_gather_length, AccOut, SegPolyElem, SegPolyPlan, TaskSlot, Unit,
    MAX_INPUTS, MAX_OUTPUTS,
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
    /// compiled extents and strides, or a scatter output's `RunIndex` does not
    /// carry the compiled run order, [`SegPolyError::IndexOutOfRange`] when a
    /// gather index is negative or past its buffer's batch extent (D3),
    /// [`SegPolyError::Strided`] carrying [`StridedError::ShapeMismatch`] when a
    /// run index describes a different topology, and
    /// [`SegPolyError::ScratchTooSmall`] when the scratch cannot hold the block
    /// partials plus one child's accumulators, streamed row tile and pack area.
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
        // The block partials must all exist at once; the child regions can be
        // capped by what the caller supplied.
        let partials = self.block_slots.saturating_mul(self.acc_stride);
        let child_elems = self.child_elems();
        let floor = partials.saturating_add(child_elems);
        if scratch.len() < floor {
            return Err(SegPolyError::ScratchTooSmall {
                need: floor,
                got: scratch.len(),
            });
        }
        let call = self.call(outputs, inputs, indexes);
        let child_tiles = (scratch.len() - partials) / child_elems;
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
        for (dense, _) in indexes.iter().enumerate().skip(self.num_inputs) {
            let dense = dense - self.num_inputs;
            call.out_base[dense] = SendPtr(outputs[dense].as_mut_ptr());
            call.out_batch[dense] = self.strides[self.num_inputs + dense][0];
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
                        .filter(|unit| unit.group == self.out_group[dense])
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
                    // The plan's traversal follows the compiled run order: a call
                    // with the same windows in a different order is not the
                    // topology the plan was compiled against.
                    let group = &self.groups[self.out_group[dense]];
                    if group.perm.as_deref() != runs.perm() {
                        return Err(StridedError::PlanLayoutMismatch.into());
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

    /// Sum one task's rows into its group's accumulators.
    ///
    /// # Safety
    ///
    /// `check_call` matched every view to the compiled layouts; the scratch holds
    /// `block_slots * acc_stride` block partials followed by at least one child
    /// region of `child_elems()` elements -- the floor `execute` checks and
    /// returns [`SegPolyError::ScratchTooSmall`] below -- and `child` is below
    /// the number of children; every destination row this task writes belongs to
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
        let group = &self.groups[unit.group];
        let child_base = unsafe {
            scratch
                .as_ptr()
                .add(self.block_slots * self.acc_stride + child * self.child_elems())
        };
        let row_base = unsafe { child_base.add(self.acc_stride) };
        let pack_base = unsafe { row_base.add(self.row_elems) };
        // Seed the accumulating outputs once per task: `into` starts every block
        // at zero, `acc` continues from the destination. Only block zero of a
        // multi-block unit continues from the destination value; the other
        // partials are contributions and start at zero.
        for acc_out in &group.acc {
            let Some((ptr, len)) =
                (unsafe { self.acc_slot(call, unit, task.slot, acc_out, child_base, scratch) })
            else {
                continue;
            };
            // SAFETY: the slot holds `len` elements and belongs to this task.
            let acc = unsafe { core::slice::from_raw_parts_mut(ptr, len) };
            unsafe {
                self.seed_block(
                    call,
                    acc,
                    acc_out.out,
                    unit.dest_row,
                    task.slot,
                    unit.slot_base,
                    into,
                )
            };
        }

        let mut rows = [0isize; MAX_INPUTS];
        for row in task.rows.clone() {
            let space = unit.rows.start + row;
            // SAFETY: `check_call` matched the run topology, so `space` is a row
            // of the group's compiled run permutation when there is one.
            let point = match &group.perm {
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
            for acc_out in &group.acc {
                let Some((ptr, len)) =
                    (unsafe { self.acc_slot(call, unit, task.slot, acc_out, child_base, scratch) })
                else {
                    continue;
                };
                // SAFETY: the slot belongs to this task; `rows` was resolved for
                // this batch point above.
                let acc = unsafe { core::slice::from_raw_parts_mut(ptr, len) };
                unsafe { self.accumulate_output(call, acc_out.out, acc, &rows, pack_base) };
            }
            for &dense in &group.stream {
                let buffer = self.num_inputs + dense;
                let len = self.dims[buffer][1].saturating_mul(self.u);
                // An empty output has nothing to write, and its base pointer is a
                // dangling empty view's, so it must not be offset at all.
                if len == 0 {
                    continue;
                }
                let dest_base = unsafe {
                    call.out_base[dense]
                        .as_ptr()
                        .offset(point as isize * call.out_batch[dense])
                };
                // A contiguous row is its own accumulator; otherwise the child's
                // row tile carries it for this batch point and is written back.
                let direct = self.out_direct[dense];
                let acc_ptr = if direct { dest_base } else { row_base };
                // SAFETY: a direct row is `len` elements of the destination; the
                // row tile holds `row_elems >= len` and belongs to this child.
                let acc = unsafe { core::slice::from_raw_parts_mut(acc_ptr, len) };
                let slot = if direct {
                    TaskSlot::Direct
                } else {
                    TaskSlot::Child
                };
                unsafe { self.seed_block(call, acc, dense, point, slot, 0, into) };
                unsafe { self.accumulate_output(call, dense, acc, &rows, pack_base) };
                if !direct {
                    // SAFETY: the destination row is disjoint from every other
                    // task's; the accumulator was written above.
                    unsafe {
                        write_block(
                            dest_base,
                            self.strides[buffer][1],
                            self.strides[buffer][2],
                            self.dims[buffer][1],
                            self.u,
                            self.block_layout(dense),
                            acc,
                        )
                    };
                }
            }
        }
        if task.slot == TaskSlot::Child {
            for acc_out in &group.acc {
                let Some((ptr, len)) =
                    (unsafe { self.acc_slot(call, unit, task.slot, acc_out, child_base, scratch) })
                else {
                    continue;
                };
                let dense = acc_out.out;
                let buffer = self.num_inputs + dense;
                let dest_base = unsafe {
                    call.out_base[dense]
                        .as_ptr()
                        .offset(unit.dest_row as isize * call.out_batch[dense])
                };
                // SAFETY: the destination block belongs to this task's unit.
                unsafe {
                    write_block(
                        dest_base,
                        self.strides[buffer][1],
                        self.strides[buffer][2],
                        self.dims[buffer][1],
                        self.u,
                        self.block_layout(dense),
                        core::slice::from_raw_parts(ptr, len),
                    )
                };
            }
        }
    }

    /// The accumulator of one accumulating output in a task: its own destination
    /// block for a direct slot, or `len` elements of the task's slot region.
    ///
    /// Returns `None` for an empty output: its view's pointer is a dangling empty
    /// view's and must not be offset.
    ///
    /// # Safety
    ///
    /// `slot` and `unit` come from the task; the slot region exists in `scratch`
    /// or `child_base`, and `child_base` is this child's region.
    unsafe fn acc_slot(
        &self,
        call: &Call<'_, T>,
        unit: &Unit,
        slot: TaskSlot,
        acc_out: &AccOut,
        child_base: *mut T,
        scratch: SendPtr<T>,
    ) -> Option<(*mut T, usize)> {
        let dense = acc_out.out;
        let len = self.dims[self.num_inputs + dense][1].saturating_mul(self.u);
        if len == 0 {
            return None;
        }
        let ptr = match slot {
            TaskSlot::Direct => unsafe {
                call.out_base[dense]
                    .as_ptr()
                    .offset(unit.dest_row as isize * call.out_batch[dense])
            },
            TaskSlot::Block(slot) => unsafe {
                scratch
                    .as_ptr()
                    .add(slot as usize * self.acc_stride + acc_out.offset)
            },
            TaskSlot::Child => unsafe { child_base.add(acc_out.offset) },
        };
        Some((ptr, len))
    }

    /// Seed one accumulator: zero it for `into`, continue from the destination
    /// for `acc`. A direct accumulator *is* the destination block, so it is never
    /// copied; only block zero of a multi-block unit continues from it.
    ///
    /// # Safety
    ///
    /// The destination row belongs to this task's unit; `acc` holds the output's
    /// `segments × u` elements.
    // The parameters are the task's own live values; a struct would only move the
    // same list one line up the call.
    #[allow(clippy::too_many_arguments)]
    unsafe fn seed_block(
        &self,
        call: &Call<'_, T>,
        acc: &mut [T],
        dense: usize,
        dest_row: usize,
        slot: TaskSlot,
        unit_slot_base: u32,
        into: bool,
    ) {
        if into {
            acc.fill(T::zero());
            return;
        }
        match slot {
            // A `Direct` accumulator *is* the destination block, so seeding it
            // with its own value would alias the copy.
            TaskSlot::Direct => {}
            // Only block zero of a unit continues from the destination value; the
            // other partials are contributions and start at zero.
            TaskSlot::Block(slot) if slot != unit_slot_base => acc.fill(T::zero()),
            TaskSlot::Block(_) | TaskSlot::Child => {
                let buffer = self.num_inputs + dense;
                let dest_base = unsafe {
                    call.out_base[dense]
                        .as_ptr()
                        .offset(dest_row as isize * call.out_batch[dense])
                };
                // SAFETY: the destination block belongs to this task's unit.
                unsafe {
                    read_block(
                        dest_base,
                        self.strides[buffer][1],
                        self.strides[buffer][2],
                        self.dims[buffer][1],
                        self.u,
                        self.block_layout(dense),
                        acc,
                    )
                };
            }
        }
    }

    /// Run the operations of one output for one batch point into `acc`.
    ///
    /// # Safety
    ///
    /// `rows` holds this batch point's base offset for every input, `acc` holds
    /// the output's `segments × u` elements, and `pack_base` is the child's pack
    /// region, large enough for the widest operation.
    unsafe fn accumulate_output(
        &self,
        call: &Call<'_, T>,
        dense: usize,
        acc: &mut [T],
        rows: &[isize; MAX_INPUTS],
        pack_base: *mut T,
    ) {
        // One resolution for the whole output row; the path loop runs inside the
        // dispatch below, one dispatch per operation row.
        let arch = pulp::Arch::new();
        let u = self.u;
        for &op_index in self.out_ops[dense].iter() {
            let op = &self.ops[op_index];
            // A pathless operation contributes nothing. Skipping it here also
            // keeps its inputs' bases unresolved: an empty input's pointer is
            // dangling, and offsetting it by a row base is undefined even when
            // no path would read it.
            if op.num_paths() == 0 {
                continue;
            }
            let arity = op.inputs.len() + 1;
            // Arity one to three: the lanes of one path go through the
            // dispatched kernel as whole vectors. Operand bases are resolved
            // once per batch point -- a lane-contiguous buffer is read where it
            // lives, a lane-strided one is copied into this child's pack tile
            // (and the plan's offsets for it are already in tile space), and a
            // unit-lane one is read as the splat it is.
            if op.lanes {
                let paths = op.base..op.base + op.num_paths();
                let offsets = op.offsets_base..op.offsets_base + op.num_paths() * arity;
                let mut tile = pack_base;
                macro_rules! lanes {
                    ($n:literal) => {{
                        let bases: [*const T; $n] = core::array::from_fn(|t| {
                            let input = op.inputs[t] as usize;
                            let step = self.in_lane[input];
                            // SAFETY: `rows` holds this row's base for that
                            // buffer, and a packed tile is inside this child's
                            // own pack region.
                            let base =
                                unsafe { call.in_base[input].as_const().offset(rows[input]) };
                            if step == 0 || step == 1 {
                                return base;
                            }
                            let segments = self.dims[input][1];
                            unsafe {
                                pack_into(base, self.strides[input][1], step, segments, u, tile)
                            };
                            let packed = tile;
                            tile = unsafe { tile.add(segments * u) };
                            packed
                        });
                        // SAFETY: every base reaches the whole segment block
                        // this op reads, the accumulator holds `segments * u`,
                        // and a lane-strided operand was packed above.
                        unsafe {
                            op_row_lanes::<T, $n>(
                                arch,
                                acc,
                                u,
                                &op.seg_starts,
                                &op.seg_ends,
                                &self.path_coeffs[paths],
                                &self.path_offsets[offsets],
                                &bases,
                                op.splat_mask,
                            )
                        }
                    }};
                }
                match op.inputs.len() {
                    1 => lanes!(1),
                    2 => lanes!(2),
                    _ => lanes!(3),
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
                        *slot =
                            unsafe { call.in_base[input].as_const().offset(rows[input] + offset) };
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

    /// How one output's destination block is laid out.
    fn block_layout(&self, dense: usize) -> BlockLayout {
        if self.reduce_lane[dense] {
            BlockLayout::ReduceLane
        } else if self.out_direct[dense] {
            BlockLayout::Direct
        } else {
            BlockLayout::Strided
        }
    }

    /// Combine the block partials of every multi-block unit, left to right, and
    /// write each accumulating output out.
    ///
    /// Sequential on the calling thread: it runs after the fork joins, and the
    /// work is one add per partial element, which no pool primitive indexes.
    fn combine(&self, call: &Call<'_, T>, scratch: &mut [T]) {
        for unit in &self.units {
            if unit.blocks <= 1 {
                continue;
            }
            let group = &self.groups[unit.group];
            for acc_out in &group.acc {
                let dense = acc_out.out;
                let buffer = self.num_inputs + dense;
                let segments = self.dims[buffer][1];
                let len = segments * self.u;
                if len == 0 {
                    continue;
                }
                let slot_base = unit.slot_base as usize * self.acc_stride;
                let (head, tail) = scratch.split_at_mut(slot_base + self.acc_stride);
                let acc = &mut head[slot_base + acc_out.offset..slot_base + acc_out.offset + len];
                for block in 1..unit.blocks {
                    let other = (block - 1) * self.acc_stride + acc_out.offset;
                    for (slot, &value) in acc.iter_mut().zip(tail[other..other + len].iter()) {
                        *slot = *slot + value;
                    }
                }
                let dest_base = unsafe {
                    call.out_base[dense]
                        .as_ptr()
                        .offset(unit.dest_row as isize * call.out_batch[dense])
                };
                // SAFETY: the destination block belongs to this unit alone.
                unsafe {
                    write_block(
                        dest_base,
                        self.strides[buffer][1],
                        self.strides[buffer][2],
                        segments,
                        self.u,
                        self.block_layout(dense),
                        acc,
                    )
                };
            }
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
    /// `shortcut: this path runs only for an operation with more than three
    /// operands or with none at all (`op.lanes` needs at least one input), so it
    /// still indexes lanes by a runtime step and vectorises poorly. Give the deeper arities a packed lane tile and the
    /// dispatched kernel when they are the measured bottleneck` (strided-rs#479).
    const LANE_TILE: usize = 64;

    // Sliced to the operation's arity, so the inner loop is a zip over two slices
    // instead of two indexed reads against a fixed capacity (which costs a bounds
    // check per operand per lane).
    let pointers = &pointers[..nin];
    let steps = &steps[..nin];

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

/// One operation's paths for one batch point, through the dispatched lane kernel.
///
/// The same sum as calling [`accumulate`] once per path in plan order: for each
/// output segment, for each of its paths, `acc[lane] = acc[lane] + ((c · x_0) ·
/// x_1) · x_2` with the product associated left to right from the coefficient.
/// Only the bookkeeping moves: the row bases come in resolved, the arity is a
/// constant, a lane-strided operand was packed into contiguous lanes, and each
/// path's operand is either a slice of `u` lanes or one splatted scalar. The
/// whole row goes through one dispatch, not one per path.
///
/// # Safety
///
/// `acc` holds `seg_starts.len() * u` elements; for every path `p` of `coeffs`
/// and every operand `t < N`, `bases[t].offset(offsets[p * (N + 1) + t])` reaches
/// either `u` contiguous, initialised elements (bit `t` of `!splat_mask`) or one
/// initialised element (bit `t` of `splat_mask`).
// The parameters are the loop's own live values; a struct would only move the
// same list one line up the call.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn op_row_lanes<T: SegPolyElem, const N: usize>(
    arch: pulp::Arch,
    acc: &mut [T],
    u: usize,
    seg_starts: &[u32],
    seg_ends: &[u32],
    coeffs: &[T],
    offsets: &[isize],
    bases: &[*const T; N],
    splat_mask: u32,
) {
    let mut row_bases = [core::ptr::null::<T>(); MAX_INPUTS];
    row_bases[..N].copy_from_slice(bases);
    let row = crate::simd::OpRow {
        acc,
        u,
        seg_starts,
        seg_ends,
        coeffs,
        offsets,
        bases: row_bases,
        arity: N,
        splat_mask,
    };
    // SAFETY: the caller's contract, handed through unchanged.
    unsafe { T::op_row_lanes(arch, row) };
}

/// Copy one input's `[segments, u]` block into a contiguous tile, so its lanes
/// can be read as whole vectors.
///
/// # Safety
///
/// `src.offset(segment * seg).offset(lane * lane_step)` must reach an initialised
/// element for every `segment < segments` and `lane < u`, and `tile` must hold
/// `segments * u` elements.
unsafe fn pack_into<T: Copy>(
    src: *const T,
    seg: isize,
    lane: isize,
    segments: usize,
    u: usize,
    tile: *mut T,
) {
    for segment in 0..segments {
        // SAFETY: the caller's contract.
        let from = unsafe { src.offset(segment as isize * seg) };
        let to = unsafe { tile.add(segment * u) };
        for index in 0..u {
            // SAFETY: the caller's contract.
            unsafe {
                *to.add(index) = *from.offset(index as isize * lane);
            }
        }
    }
}

/// How a destination block is laid out, which decides how an accumulator tile is
/// seeded from it and written back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockLayout {
    /// One contiguous run of `segments * u` elements, the accumulator itself.
    Direct,
    /// `segments` runs of `u` elements at the compiled strides.
    Strided,
    /// `segments` scalars: the accumulator holds `u` lanes per segment and the
    /// destination sums them.
    ReduceLane,
}

/// Copy a destination block into a packed accumulator.
///
/// # Safety
///
/// `dest` must reach the whole `segments x u` block (or, for
/// [`BlockLayout::ReduceLane`], `segments` scalars at stride `seg`), and `out`
/// must hold `segments * u` elements.
unsafe fn read_block<T: SegPolyElem>(
    dest: *const T,
    seg: isize,
    lane: isize,
    segments: usize,
    u: usize,
    layout: BlockLayout,
    out: &mut [T],
) {
    let len = segments * u;
    match layout {
        BlockLayout::Direct => {
            // SAFETY: a contiguous block is one run of `len` slots.
            let block = unsafe { core::slice::from_raw_parts(dest, len) };
            out[..len].copy_from_slice(block);
        }
        BlockLayout::Strided => {
            for segment in 0..segments {
                for lane_index in 0..u {
                    // SAFETY: the block's own layout, per the caller's contract.
                    out[segment * u + lane_index] = unsafe {
                        *dest.offset(segment as isize * seg + lane_index as isize * lane)
                    };
                }
            }
        }
        BlockLayout::ReduceLane => {
            // The destination value is the accumulator's first term, so it seeds
            // lane 0 -- the first lane the reduction reads -- and the remaining
            // lanes start from zero rather than repeating it once per lane.
            out[..len].fill(T::zero());
            for segment in 0..segments {
                // SAFETY: the block's own layout, per the caller's contract.
                out[segment * u] = unsafe { *dest.offset(segment as isize * seg) };
            }
        }
    }
}

/// Write a packed accumulator out to a destination block.
///
/// # Safety
///
/// As [`read_block`], in the other direction.
unsafe fn write_block<T: SegPolyElem>(
    dest: *mut T,
    seg: isize,
    lane: isize,
    segments: usize,
    u: usize,
    layout: BlockLayout,
    values: &[T],
) {
    let len = segments * u;
    match layout {
        BlockLayout::Direct => {
            // SAFETY: a contiguous block is one run of `len` slots.
            let block = unsafe { core::slice::from_raw_parts_mut(dest, len) };
            block.copy_from_slice(&values[..len]);
        }
        BlockLayout::Strided => {
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
        BlockLayout::ReduceLane => {
            // The lane reduction: one sum per destination segment, in increasing
            // lane order, so the bits do not depend on the `Exec` budget.
            for segment in 0..segments {
                let row = &values[segment * u..segment * u + u];
                let mut sum = row[0];
                for &value in &row[1..] {
                    sum = sum + value;
                }
                // SAFETY: the block's own layout, per the caller's contract.
                unsafe { *dest.offset(segment as isize * seg) = sum };
            }
        }
    }
}
