//! Compiling a lowered segmented polynomial into a replayable plan.
//!
//! `SegPolyPlan::compile` does the work the hot path must not do: it checks the
//! contract, rounds the path coefficients to the element type, sorts each
//! operation's paths the way the reference implementation does
//! (`sort_paths(-1)`: a stable sort by the output segment id, then
//! `compressed_path_segment(-1)`'s per-segment ranges), precomputes every path's
//! segment offsets against the compiled strides, and lays out the *units* and
//! *tasks* the replay walks.
//!
//! # Summation order
//!
//! One output element is a sum whose association is fixed by the plan, not by
//! the caller, the destination layout or the [`Exec`] budget. The destination
//! block is seeded with its current value ([`SegPolyPlan::execute_acc`]) or zero
//! ([`SegPolyPlan::execute_into`]), then the contributions are added in this
//! order:
//!
//! 1. batch points in the unit's row order (for a scattered output that is the
//!    [`RunIndex`](strided_basic::RunIndex)'s stable permutation order);
//! 2. operations writing that output buffer, in descriptor order;
//! 3. the operation's paths for that output segment, in the stable
//!    output-segment sort order above.
//!
//! A unit with more rows than [`BLOCK_ROWS`] is split into fixed `BLOCK_ROWS`-row
//! blocks; the block partials are combined left to right, and block zero is the
//! one seeded with the destination value. The block boundaries are a function of
//! the row count alone, so every `Exec` budget, worker count and task assignment
//! produces the same bits. The one exception is the sign of zero: a later block's
//! partial starts from `0`, so a block whose value is exactly `-0.0` contributes
//! `+0.0`.

use core::ops::{Add, Mul, Range};

use num_traits::Zero;
use strided_basic::execution::is_injective_layout;
use strided_basic::{Result as StridedResult, StridedError};

use crate::error::SegPolyError;
use crate::spec::{IndexMode, Layout, ModeKind, Spec};

/// Rows in one fixed reduction block.
///
/// This value is part of the result, and it is deliberately not tunable: a unit
/// with more rows than this is summed as `BLOCK_ROWS`-row partials combined left
/// to right, which is a different floating-point association from a single long
/// accumulation. Changing the value changes the bits of every call whose units
/// exceed it, so it changes this crate's results.
pub const BLOCK_ROWS: usize = 256;

/// Inputs one plan may have. The replay keeps one entry per input in a stack
/// array, so the count is bounded rather than heap-allocated.
pub const MAX_INPUTS: usize = 16;

/// Outputs one plan may have; see [`MAX_INPUTS`].
pub const MAX_OUTPUTS: usize = 8;

mod sealed {
    /// The sealing bound: only the element types below implement it, so a
    /// downstream crate cannot add a `SegPolyElem` of its own.
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// The scalar types a plan executes.
///
/// Sealed to `f32` and `f64` (decision D5): mixed buffer dtypes and the
/// half-precision formats need a promotion and rounding policy this increment
/// does not define.
pub trait SegPolyElem:
    sealed::Sealed
    + Copy
    + Zero
    + Add<Output = Self>
    + Mul<Output = Self>
    + Send
    + Sync
    + core::fmt::Debug
{
    /// Round a descriptor coefficient to this type, once, at compile time.
    fn from_coefficient(value: f64) -> Self;
}

impl SegPolyElem for f32 {
    #[inline]
    fn from_coefficient(value: f64) -> Self {
        value as f32
    }
}

impl SegPolyElem for f64 {
    #[inline]
    fn from_coefficient(value: f64) -> Self {
        value
    }
}

/// One operation, prepared for replay.
#[derive(Clone, Debug)]
struct OpPlan {
    /// Dense output buffer index (`buffer - num_inputs`).
    out: usize,
    /// Buffer ids of the inputs, in operand order.
    inputs: Vec<u32>,
    /// First path of the plan's path arrays.
    base: usize,
    /// Offset of this operation's first path in `path_offsets`: the path stride
    /// is the operation's own arity, which differs between degrees.
    offsets_base: usize,
    /// Per output segment: the first local path index.
    seg_starts: Vec<u32>,
    /// Per output segment: one past the last local path index.
    seg_ends: Vec<u32>,
}

/// A destination block and the batch points that contribute to it.
#[derive(Clone, Debug)]
struct Unit {
    /// Dense output buffer index.
    out: usize,
    /// Destination row of the output buffer.
    dest_row: usize,
    /// Rows of the unit's row space, in order. For a scattered output these are
    /// the run's row window, so the plan is tied to the run topology.
    rows: Range<usize>,
    /// Number of fixed blocks the rows split into.
    blocks: usize,
    /// First block slot when `blocks > 1`.
    slot_base: u32,
}

/// Where one task accumulates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TaskSlot {
    /// Straight into the destination block, which is contiguous.
    Direct,
    /// Into a block partial's own slot.
    Block(u32),
    /// Into the running child's tile, then out to a strided destination.
    Child,
}

/// One schedulable piece of work: a block of one unit's rows.
#[derive(Clone, Debug)]
struct Task {
    unit: u32,
    /// Rows of the unit's row space this task sums, relative to the unit.
    rows: Range<usize>,
    slot: TaskSlot,
}

/// A compiled segmented-polynomial plan.
#[derive(Clone, Debug)]
pub struct SegPolyPlan<T> {
    u: usize,
    batch: usize,
    num_inputs: usize,
    /// Per buffer (inputs then outputs): compiled extents and strides.
    dims: Vec<[usize; 3]>,
    strides: Vec<[isize; 3]>,
    /// Per input: lane step of the pointer walk (0 for a unit-lane buffer).
    in_lane: Vec<isize>,
    /// Per output: the operations that write it, in descriptor order. An output
    /// may be written by several operations, so one range per output would sweep
    /// up the operations of another output interleaved with them.
    out_ops: Vec<Vec<usize>>,
    /// Per output: number of destination rows.
    out_rows: Vec<usize>,
    /// Per output: whether its `[segments, u]` block is contiguous.
    out_direct: Vec<bool>,
    /// Per buffer: the index mode the plan was compiled against.
    mode_kinds: Vec<ModeKind>,
    ops: Vec<OpPlan>,
    path_coeffs: Vec<T>,
    path_offsets: Vec<isize>,
    units: Vec<Unit>,
    tasks: Vec<Task>,
    /// Prefix sums of task row counts, for the row-balanced split.
    cum_rows: Vec<usize>,
    block_slots: usize,
    tile_elems: usize,
    macs: usize,
}

impl<T: SegPolyElem> SegPolyPlan<T> {
    /// Compile a plan for one descriptor and one set of buffer layouts.
    ///
    /// The layouts are the `[batch, num_segments, u]` geometry of every buffer,
    /// inputs first and outputs last. `compile` validates them against the
    /// descriptor and stores them: [`execute_into`](Self::execute_into) rejects a
    /// view whose extents or strides differ
    /// ([`StridedError::PlanLayoutMismatch`]), so nothing is planned per call.
    ///
    /// # Errors
    ///
    /// Every rule of the lowered contract, as the [`SegPolyError`] variants
    /// document: more than one batch axis, a lane extent that is neither one nor
    /// the plan's uniform extent, an unused buffer, an operation whose last
    /// operand is not an output, a path range outside the descriptor, a segment
    /// id outside its buffer, disagreeing batch extents, more buffers than
    /// [`MAX_INPUTS`] / [`MAX_OUTPUTS`], and the layout, collation and overflow
    /// errors of `strided-basic`.
    pub fn compile(
        spec: &Spec<'_>,
        inputs: &[Layout<'_>],
        outputs: &[Layout<'_>],
    ) -> Result<Self, SegPolyError> {
        Self::compile_with_block_rows(spec, inputs, outputs, BLOCK_ROWS)
    }

    fn compile_with_block_rows(
        spec: &Spec<'_>,
        inputs: &[Layout<'_>],
        outputs: &[Layout<'_>],
        block_rows: usize,
    ) -> Result<Self, SegPolyError> {
        if spec.num_batch_axes != 1 {
            return Err(SegPolyError::MultiBatchAxis {
                axes: spec.num_batch_axes,
            });
        }
        if inputs.len() > MAX_INPUTS || outputs.len() > MAX_OUTPUTS {
            return Err(SegPolyError::TooManyBuffers {
                inputs: inputs.len(),
                outputs: outputs.len(),
            });
        }
        let num_inputs = inputs.len();
        let num_buffers = num_inputs + outputs.len();
        if spec.indexes.len() != num_buffers {
            return Err(SegPolyError::BufferModeCount {
                buffer: spec.indexes.len(),
                modes: num_buffers,
                axes: spec.num_batch_axes,
            });
        }
        for (buffer, modes) in spec.indexes.iter().enumerate() {
            if modes.len() != spec.num_batch_axes {
                return Err(SegPolyError::BufferModeCount {
                    buffer,
                    modes: modes.len(),
                    axes: spec.num_batch_axes,
                });
            }
        }

        // Layouts: rank, reach and element type of every buffer.
        let mut dims: Vec<[usize; 3]> = Vec::with_capacity(num_buffers);
        let mut strides: Vec<[isize; 3]> = Vec::with_capacity(num_buffers);
        for layout in inputs.iter().chain(outputs.iter()) {
            check_layout(layout)?;
            dims.push([layout.dims[0], layout.dims[1], layout.dims[2]]);
            strides.push([layout.strides[0], layout.strides[1], layout.strides[2]]);
        }

        // The uniform lane extent: the largest last-axis extent, with every
        // buffer at either that extent or one (a scalar broadcast).
        let u = dims.iter().map(|d| d[2]).max().unwrap_or(1);
        for (buffer, d) in dims.iter().enumerate() {
            let is_output = buffer >= num_inputs;
            if is_output && d[2] != u {
                return Err(SegPolyError::OutputLaneBroadcast {
                    buffer,
                    lane: d[2],
                    u,
                });
            }
            if d[2] != u && d[2] != 1 {
                return Err(SegPolyError::LaneExtent {
                    buffer,
                    lane: d[2],
                    u,
                });
            }
        }

        // Destination rows are owned by one task each, so an aliasing output
        // layout would be a data race between workers.
        for layout in outputs.iter() {
            if !is_injective_layout(layout.dims, layout.strides) {
                return Err(StridedError::NonInjectiveOutputLayout.into());
            }
        }

        // Index modes: role, and the batch extent they imply.
        let mut batch = 1usize;
        let mut mode_kinds: Vec<ModeKind> = Vec::with_capacity(num_buffers);
        for (buffer, modes) in spec.indexes.iter().enumerate() {
            mode_kinds.push(modes[0].kind());
            let mode = &modes[0];
            let is_output = buffer >= num_inputs;
            // `Batched` and `Shared` suit both roles; a gather reads an input and
            // a scatter writes an output.
            if (is_output && mode.is_gather()) || (!is_output && mode.is_output_mode()) {
                return Err(SegPolyError::IndexModeNotAllowed { buffer });
            }
            let extent = match mode {
                IndexMode::Batched => dims[buffer][0],
                IndexMode::Shared => {
                    if dims[buffer][0] != 1 {
                        return Err(SegPolyError::BatchExtentMismatch {
                            buffer,
                            extent: dims[buffer][0],
                            batch: 1,
                        });
                    }
                    1
                }
                IndexMode::Gather(index) => index.len(),
                IndexMode::Gather64(index) => index.len(),
                // A scatter's batch extent is its row count; its *run* count is
                // the output's row axis, checked below.
                IndexMode::Scatter(runs) => *runs.ptr().last().unwrap_or(&0),
            };
            if extent != 1 && extent != batch {
                if batch != 1 {
                    return Err(SegPolyError::BatchExtentMismatch {
                        buffer,
                        extent,
                        batch,
                    });
                }
                batch = extent;
            }
        }
        // A `Batched` buffer holds one row per batch point; a scattered output's
        // runs hold one row per batch point between them (D7).
        for (buffer, modes) in spec.indexes.iter().enumerate() {
            match &modes[0] {
                IndexMode::Shared if dims[buffer][0] != 1 => {
                    return Err(SegPolyError::BatchExtentMismatch {
                        buffer,
                        extent: dims[buffer][0],
                        batch: 1,
                    });
                }
                IndexMode::Batched if dims[buffer][0] != batch => {
                    return Err(SegPolyError::BatchExtentMismatch {
                        buffer,
                        extent: dims[buffer][0],
                        batch,
                    });
                }
                IndexMode::Scatter(runs) => {
                    // The view's rows axis *is* the run axis, so it must match the
                    // topology: the executor writes row `run` of the view.
                    let runs_count = runs.ptr().len() - 1;
                    if dims[buffer][0] != runs_count {
                        return Err(SegPolyError::BatchExtentMismatch {
                            buffer,
                            extent: dims[buffer][0],
                            batch: runs_count,
                        });
                    }
                    let rows = runs.ptr()[runs_count];
                    if rows != batch {
                        return Err(SegPolyError::BatchExtentMismatch {
                            buffer,
                            extent: rows,
                            batch,
                        });
                    }
                }
                // A gather maps one batch point to one row, so its length is the
                // batch extent. Allowing a shorter list would read past it.
                IndexMode::Gather(index) if index.len() != batch => {
                    return Err(SegPolyError::BatchExtentMismatch {
                        buffer,
                        extent: index.len(),
                        batch,
                    });
                }
                IndexMode::Gather64(index) if index.len() != batch => {
                    return Err(SegPolyError::BatchExtentMismatch {
                        buffer,
                        extent: index.len(),
                        batch,
                    });
                }
                _ => {}
            }
        }

        // Path tables: sort each operation's paths by output segment, then cut
        // per-segment ranges.
        let mut ops: Vec<OpPlan> = Vec::with_capacity(spec.operations.len());
        let mut path_coeffs: Vec<T> = Vec::with_capacity(spec.paths.len());
        let mut path_offsets: Vec<isize> = Vec::new();
        let mut used = vec![false; num_buffers];
        let mut out_ops: Vec<Vec<usize>> = vec![Vec::new(); outputs.len()];
        for (index, operation) in spec.operations.iter().enumerate() {
            if operation.buffers.is_empty() {
                return Err(SegPolyError::ArityMismatch {
                    operation: index,
                    operands: 0,
                    segments: 0,
                });
            }
            let arity = operation.buffers.len();
            let mut op_inputs = Vec::with_capacity(arity - 1);
            for (position, &buffer) in operation.buffers.iter().enumerate() {
                let buffer_usize = buffer as usize;
                if buffer_usize >= num_buffers {
                    return Err(SegPolyError::UnknownBuffer { buffer });
                }
                let is_last = position + 1 == arity;
                if is_last != (buffer_usize >= num_inputs) {
                    return Err(SegPolyError::BufferRole {
                        operation: index,
                        buffer,
                    });
                }
                used[buffer_usize] = true;
                if !is_last {
                    op_inputs.push(buffer);
                }
            }
            let out_buffer = *operation.buffers.last().expect("non-empty") as usize;
            let out = out_buffer - num_inputs;
            let segments = dims[out_buffer][1];

            let start = operation.path_start as usize;
            let count = operation.num_paths as usize;
            if arity - 1 > MAX_INPUTS {
                return Err(SegPolyError::TooManyOperands {
                    operation: index,
                    operands: arity - 1,
                    max: MAX_INPUTS,
                });
            }
            let end = start
                .checked_add(count)
                .ok_or(StridedError::OffsetOverflow)?;
            if end > spec.paths.len() {
                return Err(SegPolyError::PathRange {
                    operation: index,
                    path_start: operation.path_start,
                    num_paths: operation.num_paths,
                    paths: spec.paths.len(),
                });
            }
            let op_paths = &spec.paths[start..end];
            for path in op_paths {
                if path.segments.len() != arity {
                    return Err(SegPolyError::ArityMismatch {
                        operation: index,
                        operands: arity,
                        segments: path.segments.len(),
                    });
                }
                for (&buffer, &segment) in operation.buffers.iter().zip(path.segments.iter()) {
                    let buffer_usize = buffer as usize;
                    if segment as usize >= dims[buffer_usize][1] {
                        return Err(SegPolyError::SegmentOutOfRange {
                            buffer,
                            segment,
                            segments: dims[buffer_usize][1],
                        });
                    }
                }
            }

            // The per-segment ranges below are `u32` path indices.
            u32::try_from(op_paths.len()).map_err(|_| StridedError::OffsetOverflow)?;

            // `sort_paths(-1)`: stable, by the output segment id.
            let mut order: Vec<usize> = (0..op_paths.len()).collect();
            order.sort_by_key(|&local| op_paths[local].segments[arity - 1]);

            let base = path_coeffs.len();
            let offsets_base = path_offsets.len();
            let mut seg_starts = zero_u32(segments)?;
            let mut seg_ends = zero_u32(segments)?;
            let mut cursor = 0u32;
            for segment in 0..segments {
                seg_starts[segment] = cursor;
                while (cursor as usize) < order.len() {
                    let path = &op_paths[order[cursor as usize]];
                    if path.segments[arity - 1] as usize != segment {
                        break;
                    }
                    path_coeffs.push(T::from_coefficient(path.coefficient));
                    for (buffer, &segment_id) in operation.buffers.iter().zip(path.segments.iter())
                    {
                        let buffer_usize = *buffer as usize;
                        path_offsets.push(segment_id as isize * strides[buffer_usize][1]);
                    }
                    cursor += 1;
                }
                seg_ends[segment] = cursor;
            }

            ops.push(OpPlan {
                out,
                inputs: op_inputs,
                base,
                offsets_base,
                seg_starts,
                seg_ends,
            });
        }
        for (op_index, op) in ops.iter().enumerate() {
            out_ops[op.out].push(op_index);
        }
        for (buffer, was_used) in used.iter().enumerate() {
            if !was_used {
                return Err(SegPolyError::UnusedBuffer { buffer });
            }
        }

        // Destination rows and the multiply-adds one row costs.
        let mut out_rows = vec![0usize; outputs.len()];
        let mut macs_per_row = vec![0usize; outputs.len()];
        // Local: the executor re-derives nothing from it.
        for (dense, modes) in spec.indexes.iter().enumerate().skip(num_inputs) {
            let dense = dense - num_inputs;
            out_rows[dense] = match &modes[0] {
                IndexMode::Batched => batch,
                IndexMode::Shared => 1,
                IndexMode::Scatter(runs) => runs.ptr().len().saturating_sub(1),
                IndexMode::Gather(_) | IndexMode::Gather64(_) => {
                    unreachable!("role checked above")
                }
            };
        }
        for op in &ops {
            let paths = op.seg_ends.last().copied().unwrap_or(0) as usize;
            macs_per_row[op.out] = macs_per_row[op.out].saturating_add(paths.saturating_mul(u));
        }

        // A destination block is contiguous when its lanes are unit-stride and
        // its segments follow the lanes.
        let out_direct: Vec<bool> = (0..outputs.len())
            .map(|dense| {
                let buffer = num_inputs + dense;
                strides[buffer][2] == 1
                    && (dims[buffer][1] <= 1 || strides[buffer][1] == dims[buffer][2] as isize)
            })
            .collect();

        // Units: one destination block per unit, with the rows that feed it.
        let mut units: Vec<Unit> = Vec::new();
        for (dense, modes) in spec.indexes.iter().enumerate().skip(num_inputs) {
            let dense = dense - num_inputs;
            match &modes[0] {
                IndexMode::Batched => {
                    for row in 0..batch {
                        units.push(Unit {
                            out: dense,
                            dest_row: row,
                            rows: row..row + 1,
                            blocks: 1,
                            slot_base: 0,
                        });
                    }
                }
                IndexMode::Shared => units.push(Unit {
                    out: dense,
                    dest_row: 0,
                    rows: 0..batch,
                    blocks: block_count(batch, block_rows),
                    slot_base: 0,
                }),
                IndexMode::Scatter(runs) => {
                    for (run, window) in runs.ptr().windows(2).enumerate() {
                        units.push(Unit {
                            out: dense,
                            dest_row: run,
                            rows: window[0]..window[1],
                            blocks: block_count(window[1] - window[0], block_rows),
                            slot_base: 0,
                        });
                    }
                }
                IndexMode::Gather(_) | IndexMode::Gather64(_) => {
                    unreachable!("role checked above")
                }
            }
        }
        // Slots: only a multi-block unit's partials need one, and they must all
        // coexist until the combine runs after the fork.
        let mut block_slots = 0usize;
        for unit in units.iter_mut() {
            if unit.blocks > 1 {
                unit.slot_base =
                    u32::try_from(block_slots).map_err(|_| StridedError::OffsetOverflow)?;
                block_slots = block_slots
                    .checked_add(unit.blocks)
                    .ok_or(StridedError::OffsetOverflow)?;
            }
        }
        let block_slots = u32::try_from(block_slots).map_err(|_| StridedError::OffsetOverflow)?;

        // Tasks: one per block, in unit order.
        let mut tasks: Vec<Task> = Vec::new();
        let mut cum_rows: Vec<usize> = vec![0];
        let mut macs = 0usize;
        for (index, unit) in units.iter().enumerate() {
            let unit_id = u32::try_from(index).map_err(|_| StridedError::OffsetOverflow)?;
            let rows = unit.rows.len();
            let unit_macs = macs_per_row[unit.out];
            if unit.blocks == 1 {
                let slot = if out_direct[unit.out] {
                    TaskSlot::Direct
                } else {
                    TaskSlot::Child
                };
                tasks.push(Task {
                    unit: unit_id,
                    rows: 0..rows,
                    slot,
                });
                macs = macs.saturating_add(rows.saturating_mul(unit_macs));
                cum_rows.push(cum_rows[cum_rows.len() - 1].saturating_add(rows));
                continue;
            }
            for block in 0..unit.blocks {
                let start = block * block_rows;
                let end = (start + block_rows).min(rows);
                let slot = TaskSlot::Block(
                    unit.slot_base
                        + u32::try_from(block).map_err(|_| StridedError::OffsetOverflow)?,
                );
                tasks.push(Task {
                    unit: unit_id,
                    rows: start..end,
                    slot,
                });
                macs = macs.saturating_add((end - start).saturating_mul(unit_macs));
                cum_rows.push(cum_rows[cum_rows.len() - 1].saturating_add(end - start));
            }
        }

        let tile_elems = (0..outputs.len())
            .map(|dense| dims[num_inputs + dense][1].saturating_mul(u))
            .max()
            .unwrap_or(0);
        // The lane step is the compiled lane stride, not a hardcoded 1: a caller
        // may hand a view whose lanes are separated. A unit-lane buffer (lane
        // extent 1) broadcasts over the `u` lanes, so its step is 0 whatever its
        // stride: the stride of a length-1 axis is arbitrary in a valid view, and
        // stepping by it would read `(u - 1) * stride` elements past the buffer.
        let in_lane = (0..num_inputs)
            .map(|buffer| {
                if dims[buffer][2] == 1 {
                    0
                } else {
                    strides[buffer][2]
                }
            })
            .collect();

        Ok(Self {
            u,
            batch,
            num_inputs,
            dims,
            strides,
            in_lane,
            out_ops,
            out_rows,
            out_direct,
            mode_kinds,
            ops,
            path_coeffs,
            path_offsets,
            units,
            tasks,
            cum_rows,
            block_slots: block_slots as usize,
            tile_elems,
            macs,
        })
    }

    /// The uniform lane extent, `1` when every buffer is a scalar.
    pub fn u(&self) -> usize {
        self.u
    }

    /// The batch extent the plan was compiled for.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Elements in one scratch slot: the largest `num_segments × u` over the
    /// output buffers.
    pub fn tile_elems(&self) -> usize {
        self.tile_elems
    }

    /// Elements of `T` that [`execute_into`](Self::execute_into) /
    /// [`execute_acc`](Self::execute_acc) need for a call whose
    /// [`Exec`](strided_basic::Exec) has the given budget.
    ///
    /// Two parts. The block partials of every unit with more rows than
    /// [`BLOCK_ROWS`] are compile-time constants and must all exist at once,
    /// because the combine runs after the fork joins. The child tiles are one per
    /// concurrently running child, so this part grows with the budget: pass the
    /// budget of the token the call will use. A narrower token simply runs fewer
    /// children, and a call whose scratch cannot hold the block partials fails
    /// with [`SegPolyError::ScratchTooSmall`] rather than truncating the sum.
    pub fn scratch_len(&self, budget: usize) -> usize {
        self.block_slots
            .saturating_add(budget.max(1))
            .saturating_mul(self.tile_elems)
    }

    /// Multiply-adds one call performs: the plan's cost for the token's gate.
    pub fn macs(&self) -> usize {
        self.macs
    }
}

/// A zeroed `u32` table of a descriptor-derived length.
///
/// `vec![0; segments]` aborts on a capacity overflow, and `segments` comes from a
/// layout whose element count can be zero while one axis is huge, so this
/// allocation is fallible.
fn zero_u32(len: usize) -> StridedResult<Vec<u32>> {
    let mut table: Vec<u32> = Vec::new();
    table
        .try_reserve_exact(len)
        .map_err(|_| StridedError::OffsetOverflow)?;
    table.resize(len, 0);
    Ok(table)
}

fn block_count(rows: usize, block_rows: usize) -> usize {
    if rows <= block_rows {
        1
    } else {
        rows.div_ceil(block_rows.max(1))
    }
}

/// A gather's length is the batch extent: the replay reads one entry per batch
/// point and does not bound that read.
fn check_gather_length(buffer: usize, length: usize, batch: usize) -> Result<(), SegPolyError> {
    if length != batch {
        return Err(SegPolyError::BatchExtentMismatch {
            buffer,
            extent: length,
            batch,
        });
    }
    Ok(())
}

/// A gather index array's range: negative or `>= extent` is an error (D3).
fn check_gather<I: strided_basic::GatherIndex>(
    buffer: usize,
    index: &[I],
    extent: usize,
) -> Result<(), SegPolyError> {
    // A fold without an early exit: both bounds in one branch-free pass, so the
    // check vectorises and stays below the cost of the kernel it guards.
    let mut low = i64::MAX;
    let mut high = i64::MIN;
    for &value in index {
        let value = value.to_i64();
        low = low.min(value);
        high = high.max(value);
    }
    if index.is_empty() {
        return Ok(());
    }
    if low < 0 {
        return Err(SegPolyError::IndexOutOfRange {
            buffer,
            index: low,
            extent,
        });
    }
    if high as u64 >= extent as u64 {
        return Err(SegPolyError::IndexOutOfRange {
            buffer,
            index: high,
            extent,
        });
    }
    Ok(())
}

/// Rank, reach and offset representability of one buffer layout.
fn check_layout(layout: &Layout<'_>) -> StridedResult<()> {
    if layout.dims.len() != 3 || layout.strides.len() != 3 {
        return Err(StridedError::RankMismatch(layout.dims.len(), 3));
    }
    // Reach in absolute stride: a negative stride reaches as far in the other
    // direction, and the view's own offset is what places it, so a negative
    // stride is not an overflow. The views validate their own reach; this only
    // rejects a layout whose arithmetic cannot be represented at all.
    let mut reach = 0isize;
    for (&dim, &stride) in layout.dims.iter().zip(layout.strides.iter()) {
        let last = dim.saturating_sub(1) as isize;
        let stride = stride.checked_abs().ok_or(StridedError::OffsetOverflow)?;
        let span = last
            .checked_mul(stride)
            .ok_or(StridedError::OffsetOverflow)?;
        reach = reach
            .checked_add(span)
            .ok_or(StridedError::OffsetOverflow)?;
    }
    let _ = reach;
    Ok(())
}

mod exec;

#[cfg(test)]
#[path = "plan/tests.rs"]
mod tests;
