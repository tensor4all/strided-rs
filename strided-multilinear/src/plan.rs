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
//! 4. an output whose lane extent is 1 while the plan's uniform extent is `u`
//!    (a *lane reduction*: the shape a gradient takes when the lane axis was
//!    contracted, e.g. `bar_edge_attrs` in MACE layer-1) also sums its lanes, in
//!    increasing lane order, after the block partials are combined. This is the
//!    lane-axis counterpart of a [`Shared`](crate::IndexMode::Shared) output,
//!    which is a batch reduction (D4).
//!
//! A unit with more rows than [`BLOCK_ROWS`] is split into fixed `BLOCK_ROWS`-row
//! blocks; the block partials are combined left to right, and block zero is the
//! one seeded with the destination value. The block boundaries are a function of
//! the row count alone, so every `Exec` budget, worker count and task assignment
//! produces the same bits. The one exception is the sign of zero: a later block's
//! partial starts from `0`, so a block whose value is exactly `-0.0` contributes
//! `+0.0`.
//!
//! # Several outputs in one plan
//!
//! A plan may lower several outputs, and it evaluates them in one pass over the
//! batch points when their row orders are compatible: scattered outputs with the
//! same compiled run topology form one group and keep that topology's
//! permutation order, batched outputs join that group (each of their rows is one
//! batch point, so the visit order does not enter its sum), and a shared output
//! keeps its own natural-order `0..batch` reduction. Nothing in the per-output
//! order above changes: the traversal interleaves outputs, and outputs write
//! disjoint destinations. The only cross-output requirement is that a group's
//! scattered outputs were compiled from the same run topology; a call whose
//! `RunIndex` describes the same windows in a different order is rejected
//! ([`StridedError::PlanLayoutMismatch`]).
//!
//! # Lane access
//!
//! The lane loop runs `u` lanes of one path at a time through the dispatched
//! kernel in [`crate::simd`]. An input's lanes must be unit-stride for that; an
//! input whose lane extent is `u` but whose lane stride is not 1 is *packed*:
//! per batch point the replay copies its `[num_segments, u]` row block into a
//! contiguous tile in the caller's scratch, and the plan stores its path offsets
//! in tile space (`segment_id * u`) instead of buffer space. An input with lane
//! extent 1 is a splat: one value broadcast over the `u` lanes, the same read the
//! scalar replay did.

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
    use crate::simd;

    /// The sealing bound: only the element types below implement it, so a
    /// downstream crate cannot add a `SegPolyElem` of its own.
    ///
    /// The lane kernel's dispatch seam lives here so [`SegPolyElem`] carries it
    /// without publishing a second trait. Sealing is about *implementations*: a
    /// caller generic over [`SegPolyElem`] can still reach this method through the
    /// supertrait bound, so its contract is the caller's to keep.
    pub trait Sealed {
        /// One operation row through the dispatched lane kernel: for each output
        /// segment, for each of its paths, `acc[lane] = acc[lane] + ((c · x_0) ·
        /// x_1) · x_2`, in plan order. The ISA is resolved once per replay and the
        /// path loop runs inside the dispatched body.
        ///
        /// # Safety
        ///
        /// The caller of the replay guarantees that every base and offset in `row`
        /// reaches the operands [`crate::simd::OpRow`] documents, that `row.acc`
        /// holds `seg_starts.len() * u` elements, and that `row.u` equals the lane
        /// extent the operands carry.
        unsafe fn op_row_lanes(arch: pulp::Arch, row: crate::simd::OpRow<'_, Self>)
        where
            Self: Sized;

        /// The lane count the run-time ISA's vector holds for this type, as the
        /// kernel's dispatch selects it (`1` on a scalar target).
        fn vector_lanes() -> usize;
    }

    impl Sealed for f32 {
        #[inline]
        unsafe fn op_row_lanes(arch: pulp::Arch, row: simd::OpRow<'_, f32>) {
            // SAFETY: the trait's contract is the caller's.
            unsafe { simd::op_row_f32(arch, row) }
        }

        #[inline]
        fn vector_lanes() -> usize {
            simd::lanes_f32()
        }
    }

    impl Sealed for f64 {
        #[inline]
        unsafe fn op_row_lanes(arch: pulp::Arch, row: simd::OpRow<'_, f64>) {
            // SAFETY: the trait's contract is the caller's.
            unsafe { simd::op_row_f64(arch, row) }
        }

        #[inline]
        fn vector_lanes() -> usize {
            simd::lanes_f64()
        }
    }
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

/// The lane count the run-time ISA's vector holds for `T`, as selected by the
/// dispatch the lane kernel uses: (2, 4) on NEON, (4, 8) on AVX2, (8, 16) on
/// AVX-512, (1, 1) on a scalar target. Reported, not configured.
#[must_use]
pub fn lane_width<T: SegPolyElem>() -> usize {
    <T as sealed::Sealed>::vector_lanes()
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
    /// Whether the replay runs this operation's lanes through the dispatched
    /// kernel: arity one to three. Deeper operations take the general scalar
    /// path in `accumulate`.
    lanes: bool,
    /// Bit `t` set when input `t` has lane extent 1, so its value for a path is
    /// one scalar broadcast over the lane axis rather than `u` lanes.
    splat_mask: u32,
    /// Scratch elements this operation's packed input tiles need at one batch
    /// point (0 when every input is lane-contiguous or a splat).
    pack_elems: usize,
}

impl OpPlan {
    /// Paths of this operation: `seg_ends` is the running path cursor, so its
    /// last entry counts every path.
    fn num_paths(&self) -> usize {
        self.seg_ends.last().map_or(0, |&end| end as usize)
    }
}

/// One output whose destination block accumulates over a unit's rows, and
/// where its accumulator lives in the group's accumulator region.
#[derive(Clone, Debug)]
struct AccOut {
    /// Dense output buffer index.
    out: usize,
    /// First element of this output's `[segments, u]` accumulator inside the
    /// group's `acc_elems` region.
    offset: usize,
}

/// The outputs one batch-point traversal evaluates together.
///
/// A group exists so that several outputs of one plan share the gather, the
/// input-row resolution and the cache footprint of one pass over the batch
/// points instead of one pass per output. Every output keeps the summation
/// order it has on its own (see the module documentation).
#[derive(Clone, Debug)]
struct Group {
    /// Outputs whose destination block accumulates over the unit's rows: the
    /// group's scattered outputs, or the single shared output of a shared group.
    acc: Vec<AccOut>,
    /// Batched outputs written once per batch point; they never accumulate
    /// across rows, so they need no block partials or unit slot.
    stream: Vec<usize>,
    /// Compiled row order of a scattered group: traversal step to batch point.
    /// `None` for natural row order (a batched-only or shared group).
    perm: Option<Vec<usize>>,
    /// Multiply-adds one row of this group performs over all of its outputs.
    macs_per_row: usize,
}

/// A destination block and the batch points that contribute to it.
#[derive(Clone, Debug)]
struct Unit {
    /// The group this unit belongs to.
    group: usize,
    /// Destination row of the group's accumulating outputs. For a scattered
    /// output this is the run index; for a shared one it is row 0.
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
    /// Per output: the group whose traversal writes it.
    out_group: Vec<usize>,
    /// The output groups, in descriptor order.
    groups: Vec<Group>,
    /// Per output: number of destination rows.
    out_rows: Vec<usize>,
    /// Per output: whether its `[segments, u]` block is contiguous.
    out_direct: Vec<bool>,
    /// Per output: whether its lane extent is 1 while `u > 1`, i.e. the block is
    /// accumulated as `[segments, u]` and summed over the lanes on the way out.
    reduce_lane: Vec<bool>,
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
    /// Elements per block slot: the largest sum of the accumulating outputs'
    /// `segments × u` over the groups.
    acc_stride: usize,
    /// Elements one child reserves for a streamed batched row; 0 when no group
    /// has a batched output.
    row_elems: usize,
    tile_elems: usize,
    /// Per-child scratch elements for packed lane tiles; 0 when nothing packs.
    pack_elems: usize,
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
            // An output may have the uniform lane extent or 1. A lane extent of 1
            // under `u > 1` is a *lane reduction*: the block is accumulated as
            // `[segments, u]` and summed over the lanes on the way out, the lane
            // axis's counterpart of a `Shared` output's batch reduction (D4).
            if is_output && d[2] != u && d[2] != 1 {
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

        // A lane operand is read one of three ways, and the three must agree
        // exactly between here and the replay: a lane stride of 1 is a slice of
        // `u` lanes read in place; a lane extent of 1 or a lane stride of 0 (every
        // lane reads the same element) is a *splat*; any other lane stride is
        // packed into a contiguous tile per batch point, and this plan's offsets
        // for it live in tile space.
        let splat: Vec<bool> = (0..num_inputs)
            .map(|buffer| dims[buffer][2] == 1 || strides[buffer][2] == 0)
            .collect();
        let packed: Vec<bool> = (0..num_inputs)
            .map(|buffer| !splat[buffer] && strides[buffer][2] != 1)
            .collect();

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
            // One accumulator vector per path for arity one to three; deeper
            // operations keep the general scalar loop, and they are also the ones
            // that must not have their offsets rewritten (see `tiled` below).
            let lanes = !op_inputs.is_empty() && op_inputs.len() <= 3;

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
                        // A packed input's tile has lane stride 1 and segment
                        // stride `u`, so its offsets live in tile space. Only an
                        // operation the replay runs through the lane kernel packs,
                        // so only its offsets may be rewritten; a deeper operation
                        // falls back to the scalar loop and reads the buffer it
                        // names, with the buffer's own segment stride.
                        let tiled = lanes && packed.get(buffer_usize).copied().unwrap_or(false);
                        let stride = if tiled {
                            u as isize
                        } else {
                            strides[buffer_usize][1]
                        };
                        let offset = (segment_id as isize)
                            .checked_mul(stride)
                            .ok_or(StridedError::OffsetOverflow)?;
                        path_offsets.push(offset);
                    }
                    cursor += 1;
                }
                seg_ends[segment] = cursor;
            }

            let mut splat_mask = 0u32;
            let mut pack_elems = 0usize;
            for (position, &input) in op_inputs.iter().enumerate() {
                if splat[input as usize] {
                    splat_mask |= 1 << position;
                }
                if lanes && packed[input as usize] {
                    let tile = dims[input as usize][1]
                        .checked_mul(u)
                        .ok_or(StridedError::OffsetOverflow)?;
                    pack_elems = pack_elems
                        .checked_add(tile)
                        .ok_or(StridedError::OffsetOverflow)?;
                }
            }
            ops.push(OpPlan {
                out,
                inputs: op_inputs,
                base,
                offsets_base,
                seg_starts,
                seg_ends,
                lanes,
                splat_mask,
                pack_elems,
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
        // its segments follow the lanes. A lane-reducing output is never direct:
        // its accumulator is `[segments, u]` while the destination row is
        // `[segments]`, so it always goes through a scratch tile.
        let reduce_lane: Vec<bool> = (0..outputs.len())
            .map(|dense| dims[num_inputs + dense][2] == 1 && u > 1)
            .collect();
        let out_direct: Vec<bool> = (0..outputs.len())
            .map(|dense| {
                if reduce_lane[dense] {
                    return false;
                }
                let buffer = num_inputs + dense;
                strides[buffer][2] == 1
                    && (dims[buffer][1] <= 1 || strides[buffer][1] == dims[buffer][2] as isize)
            })
            .collect();

        // Groups: outputs evaluated in one pass over the batch points. Scattered
        // outputs group by their compiled run topology and pin the traversal
        // order; batched outputs join a scattered group because each of their
        // rows is one batch point and independent of the visit order; a shared
        // output keeps its own natural-order batch reduction.
        let mut groups: Vec<Group> = Vec::new();
        let mut out_group = vec![usize::MAX; outputs.len()];
        // Compile-time run offsets of each group, used only while grouping; the
        // group itself already owns the permutation.
        let mut topologies: Vec<Option<Vec<usize>>> = Vec::new();
        fn scatter_group(
            groups: &mut Vec<Group>,
            topologies: &mut Vec<Option<Vec<usize>>>,
            runs: &strided_basic::RunIndex,
        ) -> usize {
            // The two vectors are pushed together, so their indices agree.
            if let Some((group, _)) = topologies.iter().enumerate().find(|(group, offsets)| {
                offsets.as_deref() == Some(runs.ptr())
                    && groups[*group].perm.as_deref() == runs.perm()
            }) {
                return group;
            }
            groups.push(Group {
                acc: Vec::new(),
                stream: Vec::new(),
                perm: runs.perm().map(|perm| perm.to_vec()),
                macs_per_row: 0,
            });
            topologies.push(Some(runs.ptr().to_vec()));
            groups.len() - 1
        }
        for (dense, modes) in spec.indexes.iter().enumerate().skip(num_inputs) {
            let dense = dense - num_inputs;
            if let IndexMode::Scatter(runs) = &modes[0] {
                let group = scatter_group(&mut groups, &mut topologies, runs);
                let offset = groups[group]
                    .acc
                    .iter()
                    .map(|acc| dims[num_inputs + acc.out][1].saturating_mul(u))
                    .fold(0usize, usize::saturating_add);
                groups[group].acc.push(AccOut { out: dense, offset });
                out_group[dense] = group;
            }
        }
        for (dense, modes) in spec.indexes.iter().enumerate().skip(num_inputs) {
            let dense = dense - num_inputs;
            match &modes[0] {
                IndexMode::Scatter(_) => {}
                IndexMode::Batched => {
                    if groups.is_empty() {
                        groups.push(Group {
                            acc: Vec::new(),
                            stream: Vec::new(),
                            perm: None,
                            macs_per_row: 0,
                        });
                        topologies.push(None);
                    }
                    groups[0].stream.push(dense);
                    out_group[dense] = 0;
                }
                IndexMode::Shared => {
                    groups.push(Group {
                        acc: vec![AccOut {
                            out: dense,
                            offset: 0,
                        }],
                        stream: Vec::new(),
                        perm: None,
                        macs_per_row: 0,
                    });
                    topologies.push(None);
                    out_group[dense] = groups.len() - 1;
                }
                IndexMode::Gather(_) | IndexMode::Gather64(_) => {
                    unreachable!("role checked above")
                }
            }
        }
        for group in groups.iter_mut() {
            group.macs_per_row = group
                .acc
                .iter()
                .map(|acc| acc.out)
                .chain(group.stream.iter().copied())
                .map(|dense| macs_per_row[dense])
                .fold(0usize, usize::saturating_add);
        }

        // Units: one destination block per unit, with the rows that feed it. A
        // scattered group takes its run windows from the compiled topology;
        // batched outputs inside it have no per-row unit.
        let mut units: Vec<Unit> = Vec::new();
        for (group_index, group) in groups.iter().enumerate() {
            if let Some(ptr) = &topologies[group_index] {
                for (run, window) in ptr.windows(2).enumerate() {
                    units.push(Unit {
                        group: group_index,
                        dest_row: run,
                        rows: window[0]..window[1],
                        blocks: block_count(window[1] - window[0], block_rows),
                        slot_base: 0,
                    });
                }
            } else if group.acc.is_empty() {
                for row in 0..batch {
                    units.push(Unit {
                        group: group_index,
                        dest_row: row,
                        rows: row..row + 1,
                        blocks: 1,
                        slot_base: 0,
                    });
                }
            } else {
                units.push(Unit {
                    group: group_index,
                    dest_row: 0,
                    rows: 0..batch,
                    blocks: block_count(batch, block_rows),
                    slot_base: 0,
                });
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
            let group = &groups[unit.group];
            let unit_macs = group.macs_per_row;
            if unit.blocks == 1 {
                let slot = if group.acc.iter().all(|acc| out_direct[acc.out]) {
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
        // One block slot holds every accumulating output of one group; one child
        // holds a slot's worth of accumulators, the streamed row tile, and the
        // pack area of the widest operation.
        let acc_stride = groups
            .iter()
            .map(|group| {
                group
                    .acc
                    .iter()
                    .map(|acc| dims[num_inputs + acc.out][1].saturating_mul(u))
                    .fold(0usize, usize::saturating_add)
            })
            .max()
            .unwrap_or(0);
        let row_elems = if groups.iter().any(|group| !group.stream.is_empty()) {
            tile_elems
        } else {
            0
        };
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
            .collect::<Vec<isize>>();
        // Each child runs its tasks one at a time, so one pack region per child
        // serves every task slot; the widest operation sizes it.
        let pack_elems = ops.iter().map(|op| op.pack_elems).max().unwrap_or(0);

        Ok(Self {
            u,
            batch,
            num_inputs,
            dims,
            strides,
            in_lane,
            out_ops,
            out_group,
            groups,
            out_rows,
            out_direct,
            mode_kinds,
            ops,
            path_coeffs,
            path_offsets,
            reduce_lane,
            units,
            tasks,
            cum_rows,
            block_slots: block_slots as usize,
            acc_stride,
            row_elems,
            tile_elems,
            pack_elems,
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
    /// because the combine runs after the fork joins; a slot holds the
    /// accumulators of every output the unit's group accumulates (a scattered
    /// group may write several outputs in one pass). The child regions are one
    /// per concurrently running child, so this part grows with the budget: each
    /// holds the group's accumulators, one streamed batched row and one pack
    /// area. Pass the budget of the token the call will use. A narrower token
    /// simply runs fewer children, and a call whose scratch cannot hold the
    /// block partials fails with [`SegPolyError::ScratchTooSmall`] rather than
    /// truncating the sum.
    pub fn scratch_len(&self, budget: usize) -> usize {
        // `[ block partials ][ child 0: acc | row | pack ][ child 1: ... ]`; the
        // block partials stay first because `combine` indexes them by slot.
        self.block_slots
            .saturating_mul(self.acc_stride)
            .saturating_add(budget.max(1).saturating_mul(self.child_elems()))
    }

    /// Scratch elements one child region holds: the group's accumulators, the
    /// streamed row tile, and the pack area of the widest operation.
    fn child_elems(&self) -> usize {
        self.acc_stride
            .saturating_add(self.row_elems)
            .saturating_add(self.pack_elems)
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
