//! Prepared indexed plans over raw strided value and index layouts.
//!
//! This module owns the generic gather, dynamic-slice/update, and scatter
//! traversals used by the erased replay layer. It models the XLA/tenferro
//! indexed shape vocabulary, but keeps tensor allocation, dtype promotion, and
//! frontend error policy outside `strided-kernel`.

use core::{mem::MaybeUninit, ops::Add};

use crate::copy_plan::{replay_fused_raw, CopyPlan, OverwriteWriter, ReadModifyWrite};
use crate::cost::KernelClass;
use crate::inline::InlineVec;
use crate::raw_ops::pair_tiles;
use crate::raw_ops::{fuse_pair_layout, FusedPairLayout};
use crate::{Exec, RawStridedMut, RawStridedRef, Result, StridedError, RAW_FUSED_RANK_LIMIT};

type AxisVec<T> = InlineVec<T>;

/// Gather configuration shared by generic and erased replay.
///
/// The fields follow the usual gather vocabulary:
///
/// - `start_index_map[component]` names the operand axis controlled by a
///   component in the index vector;
/// - `collapsed_slice_dims` names operand axes whose slice size is one and
///   which do not appear as output window axes;
/// - `offset_dims` names output axes that represent window offsets;
/// - output axes not in `offset_dims` are batch axes from `start_indices`;
/// - `index_vector_dim == start_indices_rank` represents scalar index vectors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GatherSpec {
    pub offset_dims: Vec<usize>,
    pub collapsed_slice_dims: Vec<usize>,
    pub start_index_map: Vec<usize>,
    pub index_vector_dim: usize,
    pub slice_sizes: Vec<usize>,
}

/// Index scalar types accepted by [`GatherPlan`].
pub trait GatherIndex: Copy + Send + Sync {
    fn to_i64(self) -> i64;
}

impl GatherIndex for i32 {
    #[inline]
    fn to_i64(self) -> i64 {
        i64::from(self)
    }
}

impl GatherIndex for i64 {
    #[inline]
    fn to_i64(self) -> i64 {
        self
    }
}

/// A compiled gather traversal for one value layout, index layout, and output layout.
#[derive(Clone, Debug)]
pub struct GatherPlan {
    operand_dims: AxisVec<usize>,
    operand_strides: AxisVec<isize>,
    index_dims: AxisVec<usize>,
    index_strides: AxisVec<isize>,
    dest_dims: AxisVec<usize>,
    dest_strides: AxisVec<isize>,
    spec: GatherSpec,
    replay: GatherReplay,
    total: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct GatherReplayAxis {
    dest_step: isize,
    dest_reset: isize,
    window_step: isize,
    window_reset: isize,
    index_batch_step: isize,
    index_batch_reset: isize,
}

#[derive(Clone, Debug)]
struct GatherReplay {
    axes: AxisVec<GatherReplayAxis>,
    index_component_offsets: AxisVec<isize>,
    index_operand_strides: AxisVec<isize>,
}

struct GatherReplayState {
    coords: AxisVec<usize>,
    dest_offset: isize,
    window_offset: isize,
    index_batch_offset: isize,
}

#[derive(Clone, Copy, Debug, Default)]
struct WindowReplayAxis {
    source_step: isize,
    source_reset: isize,
    dest_step: isize,
    dest_reset: isize,
}

/// The offsets a strided window reaches, relative to its base offset.
#[derive(Clone, Copy, Debug, Default)]
struct OffsetSpan {
    min: isize,
    max: isize,
}

impl OffsetSpan {
    /// Every offset the window reaches from `base` fits `isize`.
    fn covers(&self, base: isize) -> Result<()> {
        base.checked_add(self.min)
            .ok_or(StridedError::OffsetOverflow)?;
        base.checked_add(self.max)
            .ok_or(StridedError::OffsetOverflow)?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct WindowReplay {
    shape: AxisVec<usize>,
    axes: AxisVec<WindowReplayAxis>,
    /// Offsets the window reaches in the update and destination layouts,
    /// relative to the runtime base. Fusing keeps the reachable offset set, so
    /// these spans also bound every fused replay step.
    source_span: OffsetSpan,
    dest_span: OffsetSpan,
}

struct WindowReplayState {
    coords: CoordScratch,
    source_offset: isize,
    dest_offset: isize,
}

#[derive(Clone, Debug)]
struct ScatterReplay {
    batch: WindowReplay,
    window: WindowReplay,
    index_component_offsets: AxisVec<isize>,
}

impl ScatterReplay {
    // Scatter compile: batch and window shapes, the index layout with its vector
    // stride and component count, and the collapsed-slice set.
    #[allow(clippy::too_many_arguments)]
    fn compile(
        batch_shape: &[usize],
        index_dims: &[usize],
        index_strides: &[isize],
        index_vector_dim: usize,
        index_component_count: usize,
        update_dims: &[usize],
        update_strides: &[isize],
        update_window_dims: &[usize],
        is_update_window_dim: &[bool],
        window_shape_updates: &[usize],
        window_dims: &[usize],
        dest_strides: &[isize],
    ) -> Result<Self> {
        let (
            batch_source_strides,
            batch_dest_strides,
            window_source_strides,
            window_dest_strides,
            vector_stride,
        ) = scatter_replay_strides(
            index_dims,
            index_strides,
            index_vector_dim,
            update_dims,
            update_strides,
            update_window_dims,
            is_update_window_dim,
            window_dims,
            dest_strides,
        );
        let batch = WindowReplay::compile(batch_shape, &batch_source_strides, &batch_dest_strides)?;
        let window = WindowReplay::compile(
            window_shape_updates,
            &window_source_strides,
            &window_dest_strides,
        )?;
        let mut index_component_offsets = AxisVec::with_capacity(index_component_count);
        for component in 0..index_component_count {
            index_component_offsets.push(checked_offset_add(0, vector_stride, component)?);
        }
        Ok(Self {
            batch,
            window,
            index_component_offsets,
        })
    }
}

// Replay strides: both shapes, the index strides and vector axis, the update
// window set and the operand shape.
#[allow(clippy::too_many_arguments)]
fn scatter_replay_strides(
    index_dims: &[usize],
    index_strides: &[isize],
    index_vector_dim: usize,
    update_dims: &[usize],
    update_strides: &[isize],
    update_window_dims: &[usize],
    is_update_window_dim: &[bool],
    window_dims: &[usize],
    dest_strides: &[isize],
) -> (
    AxisVec<isize>,
    AxisVec<isize>,
    AxisVec<isize>,
    AxisVec<isize>,
    isize,
) {
    let batch_source_strides = index_dims
        .iter()
        .zip(index_strides.iter())
        .enumerate()
        .filter_map(|(axis, (_, &stride))| (axis != index_vector_dim).then_some(stride))
        .collect();
    let batch_dest_strides = update_dims
        .iter()
        .zip(update_strides.iter())
        .enumerate()
        .filter_map(|(axis, (_, &stride))| (!is_update_window_dim[axis]).then_some(stride))
        .collect();
    let window_source_strides = update_window_dims
        .iter()
        .map(|&axis| update_strides[axis])
        .collect();
    let window_dest_strides = window_dims.iter().map(|&axis| dest_strides[axis]).collect();
    let vector_stride = if index_vector_dim < index_dims.len() {
        index_strides[index_vector_dim]
    } else {
        0
    };
    (
        batch_source_strides,
        batch_dest_strides,
        window_source_strides,
        window_dest_strides,
        vector_stride,
    )
}

impl WindowReplay {
    fn compile(shape: &[usize], source_strides: &[isize], dest_strides: &[isize]) -> Result<Self> {
        let source_span = validate_layout_span(shape, source_strides)?;
        let dest_span = validate_layout_span(shape, dest_strides)?;
        let mut fused_shape: AxisVec<usize> = AxisVec::with_capacity(shape.len());
        let mut axes: AxisVec<WindowReplayAxis> = AxisVec::with_capacity(shape.len());
        for (axis, &dim) in shape.iter().enumerate() {
            let source_step = source_strides[axis];
            let dest_step = dest_strides[axis];
            if let Some(previous_axis) = fused_shape.len().checked_sub(1) {
                let previous_extent = fused_shape[previous_axis];
                let previous_extent =
                    isize::try_from(previous_extent).map_err(|_| StridedError::OffsetOverflow)?;
                let expected_source = axes[previous_axis]
                    .source_step
                    .checked_mul(previous_extent)
                    .ok_or(StridedError::OffsetOverflow)?;
                let expected_dest = axes[previous_axis]
                    .dest_step
                    .checked_mul(previous_extent)
                    .ok_or(StridedError::OffsetOverflow)?;
                if source_step == expected_source && dest_step == expected_dest {
                    let fused_extent = fused_shape[previous_axis]
                        .checked_mul(dim)
                        .ok_or(StridedError::OffsetOverflow)?;
                    fused_shape[previous_axis] = fused_extent;
                    axes[previous_axis].source_reset =
                        checked_replay_reset(fused_extent, axes[previous_axis].source_step)?;
                    axes[previous_axis].dest_reset =
                        checked_replay_reset(fused_extent, axes[previous_axis].dest_step)?;
                    continue;
                }
            }
            fused_shape.push(dim);
            axes.push(WindowReplayAxis {
                source_step,
                source_reset: checked_replay_reset(dim, source_step)?,
                dest_step,
                dest_reset: checked_replay_reset(dim, dest_step)?,
            });
        }
        Ok(Self {
            shape: fused_shape,
            axes,
            source_span,
            dest_span,
        })
    }

    /// The replay state at linear index zero. Every coordinate is zero, so
    /// this needs no offset arithmetic and cannot fail after the destination
    /// copy.
    fn start(&self, source_base: isize, dest_base: isize) -> WindowReplayState {
        WindowReplayState {
            coords: CoordScratch::new(self.shape.len()),
            source_offset: source_base,
            dest_offset: dest_base,
        }
    }

    fn decode(
        &self,
        mut linear: usize,
        source_base: isize,
        dest_base: isize,
    ) -> Result<WindowReplayState> {
        let mut coords = CoordScratch::new(self.shape.len());
        let mut source_offset = source_base;
        let mut dest_offset = dest_base;
        for (axis, (&dim, coord)) in self
            .shape
            .iter()
            .zip(coords.as_mut_slice().iter_mut())
            .enumerate()
        {
            *coord = linear % dim;
            linear /= dim;
            let replay_axis = self.axes[axis];
            source_offset = checked_offset_add(source_offset, replay_axis.source_step, *coord)?;
            dest_offset = checked_offset_add(dest_offset, replay_axis.dest_step, *coord)?;
        }
        Ok(WindowReplayState {
            coords,
            source_offset,
            dest_offset,
        })
    }

    #[inline]
    fn advance(&self, state: &mut WindowReplayState) {
        for ((coord, &dim), replay_axis) in state
            .coords
            .as_mut_slice()
            .iter_mut()
            .zip(self.shape.iter())
            .zip(self.axes.iter())
        {
            let next = *coord + 1;
            if next < dim {
                *coord = next;
                state.source_offset += replay_axis.source_step;
                state.dest_offset += replay_axis.dest_step;
                return;
            }
            *coord = 0;
            state.source_offset += replay_axis.source_reset;
            state.dest_offset += replay_axis.dest_reset;
        }
    }
}

/// Scatter configuration shared by generic and erased replay.
///
/// `ScatterPlan` implements tenferro's current additive scatter semantics:
/// every update value is added to the selected output slot, so overlapping
/// windows accumulate in deterministic column-major replay order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScatterSpec {
    pub update_window_dims: Vec<usize>,
    pub inserted_window_dims: Vec<usize>,
    pub scatter_dims_to_operand_dims: Vec<usize>,
    pub index_vector_dim: usize,
}

/// A compiled fixed-window dynamic-slice traversal.
#[derive(Clone, Debug)]
pub struct DynamicSlicePlan {
    operand_dims: AxisVec<usize>,
    operand_strides: AxisVec<isize>,
    start_dims: AxisVec<usize>,
    start_strides: AxisVec<isize>,
    dest_dims: AxisVec<usize>,
    dest_strides: AxisVec<isize>,
    slice_sizes: AxisVec<usize>,
    total: usize,
    /// Fused window traversal (dest strides, operand strides) replayed at the
    /// runtime source base; `None` when its fused rank exceeds
    /// [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT), in which case
    /// `replay`/`execute_parallel` walk the window instead.
    window: Option<FusedPairLayout>,
    replay: WindowReplay,
}

/// A compiled dynamic-update-slice traversal.
///
/// Execution first copies `operand` into `dest`, then overwrites the clamped
/// update window. The plan performs no allocation for ranks at most
/// [`RAW_FUSED_RANK_LIMIT`].
#[derive(Clone, Debug)]
pub struct DynamicUpdateSlicePlan {
    operand_dims: AxisVec<usize>,
    operand_strides: AxisVec<isize>,
    start_dims: AxisVec<usize>,
    start_strides: AxisVec<isize>,
    update_dims: AxisVec<usize>,
    update_strides: AxisVec<isize>,
    dest_dims: AxisVec<usize>,
    dest_strides: AxisVec<isize>,
    total: usize,
    copy_plan: CopyPlan,
    /// Fused update window traversal (dest strides, update strides) replayed
    /// at the runtime destination base; `None` when its fused rank exceeds
    /// [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT), in which case
    /// `replay`/`execute_update_parallel` walk the window instead.
    window: Option<FusedPairLayout>,
    replay: WindowReplay,
}

/// Update-window metadata resolved before the operand copy.
///
/// Reading the clamped start vector, mapping it onto the destination layout,
/// bounding the window against both runtime bases, and (on the rank-one
/// contiguous path) converting the run bounds are the only fallible steps of
/// the update phase. Preparing them up front keeps a rejected call from
/// leaving `dest` copied but un-updated.
enum UpdateWindow<'a, T> {
    /// The window is empty: the copy alone is the result.
    Empty,
    /// Rank-one contiguous run, as the checked update run and destination
    /// index.
    RankOne { update: &'a [T], dest_start: usize },
    /// Replay base in the destination layout, shared by the fused and generic
    /// paths.
    Replay { dest_base: isize },
}

/// A compiled additive scatter traversal.
///
/// Execution first copies `operand` into `dest`, then applies additive updates
/// in deterministic column-major order. Boolean values are intentionally not
/// supported because additive scatter has no bool semantics.
#[derive(Clone, Debug)]
pub struct ScatterPlan {
    operand_dims: AxisVec<usize>,
    operand_strides: AxisVec<isize>,
    index_dims: AxisVec<usize>,
    index_strides: AxisVec<isize>,
    update_dims: AxisVec<usize>,
    update_strides: AxisVec<isize>,
    dest_dims: AxisVec<usize>,
    dest_strides: AxisVec<isize>,
    spec: ScatterSpec,
    window_shape: AxisVec<usize>,
    batch_elems: usize,
    window_elems: usize,
    copy_plan: CopyPlan,
    replay: ScatterReplay,
}

impl GatherPlan {
    /// Compile a gather plan for fixed operand, index, and destination layouts.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::StrideLengthMismatch`] when a stride list does
    /// not match its dimension list, [`StridedError::OffsetOverflow`] when an
    /// element count or a layout span does not fit `usize`/`isize`,
    /// [`StridedError::NonInjectiveOutputLayout`] when the destination layout
    /// maps two logical indices onto one offset.
    /// [`StridedError::RankMismatch`] reports `spec.slice_sizes` or
    /// `spec.offset_dims` of the wrong length, or an index vector that does not
    /// match `spec.start_index_map`;
    /// [`StridedError::InvalidAxis`] reports `collapsed_slice_dims`,
    /// `start_index_map`, or `offset_dims` that repeat or exceed their rank,
    /// `index_vector_dim` beyond the index rank, a slice size beyond its
    /// operand extent, or a collapsed axis whose slice size is not one; and
    /// [`StridedError::ShapeMismatch`] reports a `dest_dims` that is not the
    /// gathered window/batch shape.
    pub fn compile(
        operand_dims: &[usize],
        operand_strides: &[isize],
        index_dims: &[usize],
        index_strides: &[isize],
        dest_dims: &[usize],
        dest_strides: &[isize],
        spec: GatherSpec,
    ) -> Result<Self> {
        if operand_dims.len() != operand_strides.len()
            || index_dims.len() != index_strides.len()
            || dest_dims.len() != dest_strides.len()
        {
            return Err(StridedError::StrideLengthMismatch);
        }
        checked_total_len(operand_dims)?;
        checked_total_len(index_dims)?;
        let total = checked_total_len(dest_dims)?;
        validate_layout_span(operand_dims, operand_strides)?;
        validate_layout_span(index_dims, index_strides)?;
        validate_layout_span(dest_dims, dest_strides)?;
        if !crate::layout_check::is_injective_layout(dest_dims, dest_strides) {
            return Err(StridedError::NonInjectiveOutputLayout);
        }

        let operand_rank = operand_dims.len();
        if spec.slice_sizes.len() != operand_rank {
            return Err(StridedError::RankMismatch(
                spec.slice_sizes.len(),
                operand_rank,
            ));
        }
        validate_unique_axes(&spec.collapsed_slice_dims, operand_rank)?;
        validate_unique_axes(&spec.start_index_map, operand_rank)?;
        if spec.index_vector_dim > index_dims.len() {
            return Err(StridedError::InvalidAxis {
                axis: spec.index_vector_dim,
                rank: index_dims.len() + 1,
            });
        }

        for (axis, (&window, &dim)) in spec.slice_sizes.iter().zip(operand_dims.iter()).enumerate()
        {
            if window > dim {
                return Err(StridedError::InvalidAxis {
                    axis,
                    rank: operand_rank,
                });
            }
        }
        for &axis in &spec.collapsed_slice_dims {
            if spec.slice_sizes[axis] != 1 {
                return Err(StridedError::InvalidAxis {
                    axis,
                    rank: operand_rank,
                });
            }
        }

        let index_vector_size = if spec.index_vector_dim == index_dims.len() {
            1
        } else {
            index_dims[spec.index_vector_dim]
        };
        if index_vector_size != spec.start_index_map.len() {
            return Err(StridedError::RankMismatch(
                index_vector_size,
                spec.start_index_map.len(),
            ));
        }

        let window_dims = operand_window_dims(operand_rank, &spec.collapsed_slice_dims);
        if spec.offset_dims.len() != window_dims.len() {
            return Err(StridedError::RankMismatch(
                spec.offset_dims.len(),
                window_dims.len(),
            ));
        }

        let batch_shape = index_batch_shape(index_dims, spec.index_vector_dim);
        let out_rank = batch_shape.len() + spec.offset_dims.len();
        validate_unique_axes(&spec.offset_dims, out_rank)?;

        let mut out_axis_to_operand_dim: AxisVec<Option<usize>> =
            (0..out_rank).map(|_| None).collect();
        for (offset_axis, &out_axis) in spec.offset_dims.iter().enumerate() {
            out_axis_to_operand_dim[out_axis] = Some(window_dims[offset_axis]);
        }

        let mut expected_dest_dims: AxisVec<usize> = AxisVec::with_capacity(out_rank);
        let mut batch_axis = 0usize;
        for &operand_dim in &out_axis_to_operand_dim {
            match operand_dim {
                Some(axis) => expected_dest_dims.push(spec.slice_sizes[axis]),
                None => {
                    expected_dest_dims.push(batch_shape[batch_axis]);
                    batch_axis += 1;
                }
            }
        }
        if dest_dims != &expected_dest_dims[..] {
            return Err(StridedError::ShapeMismatch(
                dest_dims.to_vec(),
                expected_dest_dims.to_vec(),
            ));
        }

        let batch_index_strides: AxisVec<isize> = index_dims
            .iter()
            .zip(index_strides.iter())
            .enumerate()
            .filter_map(|(axis, (_, &stride))| (axis != spec.index_vector_dim).then_some(stride))
            .collect();
        let mut batch_axis = 0usize;
        let mut replay_axes = AxisVec::with_capacity(out_rank);
        for (out_axis, &operand_dim) in out_axis_to_operand_dim.iter().enumerate() {
            let (window_step, index_batch_step) = match operand_dim {
                Some(axis) => (operand_strides[axis], 0),
                None => {
                    let step = batch_index_strides[batch_axis];
                    batch_axis += 1;
                    (0, step)
                }
            };
            replay_axes.push(GatherReplayAxis {
                dest_step: dest_strides[out_axis],
                dest_reset: checked_replay_reset(dest_dims[out_axis], dest_strides[out_axis])?,
                window_step,
                window_reset: checked_replay_reset(dest_dims[out_axis], window_step)?,
                index_batch_step,
                index_batch_reset: checked_replay_reset(dest_dims[out_axis], index_batch_step)?,
            });
        }

        let vector_stride = if spec.index_vector_dim < index_dims.len() {
            index_strides[spec.index_vector_dim]
        } else {
            0
        };
        let mut index_component_offsets = AxisVec::with_capacity(spec.start_index_map.len());
        for component in 0..spec.start_index_map.len() {
            index_component_offsets.push(checked_offset_add(0, vector_stride, component)?);
        }

        let index_operand_strides: AxisVec<isize> = spec
            .start_index_map
            .iter()
            .map(|&axis| operand_strides[axis])
            .collect();

        Ok(Self {
            operand_dims: operand_dims.into(),
            operand_strides: operand_strides.into(),
            index_dims: index_dims.into(),
            index_strides: index_strides.into(),
            dest_dims: dest_dims.into(),
            dest_strides: dest_strides.into(),
            spec,
            replay: GatherReplay {
                axes: replay_axes,
                index_component_offsets,
                index_operand_strides,
            },
            total,
        })
    }

    #[inline]
    pub fn spec(&self) -> &GatherSpec {
        &self.spec
    }

    #[inline]
    pub fn dest_dims(&self) -> &[usize] {
        &self.dest_dims
    }

    /// Execute the prepared gather traversal.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::PlanLayoutMismatch`] when the operand, index,
    /// or destination view does not carry its compiled extents and strides,
    /// before any destination write, and
    /// [`StridedError::OffsetOverflow`] when a replayed chunk's offset
    /// arithmetic cannot be represented. Start indices are clamped into the
    /// operand window, so an out-of-range index is not an error. The offset
    /// arithmetic runs per worker above the threading threshold, so a
    /// destination split across workers may be partially written when that
    /// error surfaces.
    pub fn execute<T, I>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        operand: &RawStridedRef<'_, T>,
        start_indices: &RawStridedRef<'_, I>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        I: GatherIndex,
    {
        self.execute_with_writer(dest, operand, start_indices, exec)
    }

    /// Execute the prepared gather into a destination whose reachable slots
    /// may be uninitialized. Every logical destination slot is written.
    pub(crate) fn execute_uninit<T, I>(
        &self,
        dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
        operand: &RawStridedRef<'_, T>,
        start_indices: &RawStridedRef<'_, I>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        I: GatherIndex,
    {
        self.execute_with_writer(dest, operand, start_indices, exec)
    }

    fn execute_with_writer<T, I, W>(
        &self,
        dest: &mut W,
        operand: &RawStridedRef<'_, T>,
        start_indices: &RawStridedRef<'_, I>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        I: GatherIndex,
        W: OverwriteWriter<T>,
    {
        self.check_call(dest, operand, start_indices)?;
        if self.total == 0 {
            return Ok(());
        }
        if self.uses_rank_one_scalar_take_path() {
            let exec = crate::cost::for_len_tree(exec, KernelClass::Copy, self.total);
            if !exec.is_serial() {
                return self.execute_rank_one_scalar_take_parallel(
                    dest,
                    operand,
                    start_indices,
                    exec,
                );
            }
            return self.execute_rank_one_scalar_take(dest, operand, start_indices);
        }
        let exec = crate::cost::for_len_tree(exec, KernelClass::Copy, self.total);
        if !exec.is_serial() {
            return self.execute_parallel(dest, operand, start_indices, exec);
        }

        let mut state =
            self.decode_replay_state(0, dest.offset(), operand.offset(), start_indices.offset())?;
        let operand_data = operand.data();
        let index_data = start_indices.data();

        // INVARIANT: the checked decode starts at a valid logical output. Each
        // replay axis has checked step/reset deltas, so this state remains the
        // corresponding destination, window, and batch-index offsets.
        for _ in 0..self.total {
            // INVARIANT: window_offset plus every clamped mapped contribution
            // is an operand offset for coordinates inside operand_dims; the
            // validated operand span and RawStridedRef reachability therefore
            // keep the fresh source offset inside the allocation.
            let mut source_offset = state.window_offset;
            for ((&index_component_offset, &operand_stride), &operand_dim) in self
                .replay
                .index_component_offsets
                .iter()
                .zip(self.replay.index_operand_strides.iter())
                .zip(self.spec.start_index_map.iter())
            {
                let index_offset = state.index_batch_offset + index_component_offset;
                // SAFETY: checked index layout replay and RawStridedRef
                // reachability prove this component read is in bounds.
                let start = unsafe { *index_data.as_ptr().offset(index_offset) }.to_i64();
                let clamped = self.clamp_window_start(start, operand_dim);
                source_offset += operand_stride * clamped as isize;
            }

            // SAFETY: checked replay metadata and the validated writer prove
            // both the source and the distinct destination offset are valid.
            let value = unsafe { *operand_data.as_ptr().offset(source_offset) };
            unsafe { dest.write_at(state.dest_offset, value) };
            self.advance_replay_state(&mut state);
        }
        Ok(())
    }

    fn uses_rank_one_scalar_take_path(&self) -> bool {
        self.operand_dims.len() == 1
            && self.index_dims.len() == 1
            && self.dest_dims.len() == 1
            && self.operand_strides[0] == 1
            && self.index_strides[0] == 1
            && self.dest_strides[0] == 1
            && self.spec.offset_dims.is_empty()
            && self.spec.collapsed_slice_dims.as_slice() == [0]
            && self.spec.start_index_map.as_slice() == [0]
            && self.spec.index_vector_dim == 1
            && self.spec.slice_sizes.as_slice() == [1]
    }

    fn execute_rank_one_scalar_take<T, I, W>(
        &self,
        dest: &mut W,
        operand: &RawStridedRef<'_, T>,
        start_indices: &RawStridedRef<'_, I>,
    ) -> Result<()>
    where
        T: Copy,
        I: GatherIndex,
        W: OverwriteWriter<T>,
    {
        let mut dest_offset = dest.offset();
        let mut index_offset = start_indices.offset();
        let operand_offset = operand.offset();
        let operand_data = operand.data();
        let index_data = start_indices.data();

        // INVARIANT: compile and check_call validated compact rank-one layouts,
        // so incrementing these offsets once per logical element stays within
        // the validated allocations and cannot overflow isize.
        // `index_offset` is the pointer walk of this gather line, not a counter.
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..self.total {
            // SAFETY: the invariant above proves all three offsets are in bounds.
            unsafe {
                let start = (*index_data.as_ptr().offset(index_offset)).to_i64();
                let source_offset = operand_offset + self.clamp_window_start(start, 0) as isize;
                dest.write_at(dest_offset, *operand_data.as_ptr().offset(source_offset));
            }
            dest_offset += 1;
            index_offset += 1;
        }
        Ok(())
    }

    fn execute_rank_one_scalar_take_parallel<T, I, W>(
        &self,
        dest: &mut W,
        operand: &RawStridedRef<'_, T>,
        start_indices: &RawStridedRef<'_, I>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        I: GatherIndex,
        W: OverwriteWriter<T>,
    {
        let dest_offset = dest.offset();
        let operand_offset = operand.offset();
        let index_offset = start_indices.offset();
        // SAFETY: check_call validated the writer's complete destination layout.
        let dest_ptr = crate::threading::SendPtr(unsafe { dest.data_ptr() });
        let operand_ptr = crate::threading::SendPtr(operand.data().as_ptr() as *mut T);
        let index_ptr = crate::threading::SendPtr(start_indices.data().as_ptr() as *mut I);

        crate::threading::parallel_map_reduce(
            0..self.total,
            exec,
            &|range| {
                let dest_ptr = dest_ptr.as_ptr();
                let operand_ptr = operand_ptr.as_const();
                let index_ptr = index_ptr.as_const();
                // INVARIANT: the validated compact rank-one layouts map each
                // output position to one distinct destination offset.
                for position in range {
                    let position = position as isize;
                    // SAFETY: the invariant above proves the source and distinct
                    // destination offsets are in their validated allocations.
                    unsafe {
                        let start = (*index_ptr.offset(index_offset + position)).to_i64();
                        let source_offset =
                            operand_offset + self.clamp_window_start(start, 0) as isize;
                        dest_ptr
                            .offset(dest_offset + position)
                            .write(operand_ptr.offset(source_offset).read());
                    }
                }
                Ok(())
            },
            &|left, right| left.and(right),
        )
    }

    fn execute_parallel<T, I, W>(
        &self,
        dest: &mut W,
        operand: &RawStridedRef<'_, T>,
        start_indices: &RawStridedRef<'_, I>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        I: GatherIndex,
        W: OverwriteWriter<T>,
    {
        let dest_offset_base = dest.offset();
        let operand_offset_base = operand.offset();
        let index_offset_base = start_indices.offset();
        // SAFETY: the validated writer owns the destination allocation.
        let dest_ptr = crate::threading::SendPtr(unsafe { dest.data_ptr() });
        let operand_ptr = crate::threading::SendPtr(operand.data().as_ptr() as *mut T);
        let index_ptr = crate::threading::SendPtr(start_indices.data().as_ptr() as *mut I);

        crate::threading::parallel_map_reduce(
            0..self.total,
            exec,
            &|range| {
                let mut state = self.decode_replay_state(
                    range.start,
                    dest_offset_base,
                    operand_offset_base,
                    index_offset_base,
                )?;
                let dest_ptr = dest_ptr.as_ptr();
                let operand_ptr = operand_ptr.as_const();
                let index_ptr = index_ptr.as_const();

                // INVARIANT: this worker owns a disjoint logical range. Checked
                // range-start decode plus checked replay deltas keeps every
                // state at its corresponding output, and injectivity makes all
                // writes across workers distinct.
                for _ in range {
                    // INVARIANT: source_offset is freshly based on the current
                    // window offset; clamped mapped coordinates remain inside
                    // the validated operand span and are never accumulated.
                    let mut source_offset = state.window_offset;
                    for ((&index_component_offset, &operand_stride), &operand_dim) in self
                        .replay
                        .index_component_offsets
                        .iter()
                        .zip(self.replay.index_operand_strides.iter())
                        .zip(self.spec.start_index_map.iter())
                    {
                        let index_offset = state.index_batch_offset + index_component_offset;
                        // SAFETY: the checked index span and this worker's
                        // validated range-start state prove this read is valid.
                        let start = unsafe { (*index_ptr.offset(index_offset)).to_i64() };
                        let clamped = self.clamp_window_start(start, operand_dim);
                        source_offset += operand_stride * clamped as isize;
                    }

                    // SAFETY: the source proof above and distinct destination
                    // invariant justify these raw accesses.
                    unsafe {
                        dest_ptr
                            .offset(state.dest_offset)
                            .write(operand_ptr.offset(source_offset).read());
                    }
                    self.advance_replay_state(&mut state);
                }
                Ok(())
            },
            &|left, right| left.and(right),
        )
    }

    fn check_call<T, I, W>(
        &self,
        dest: &W,
        operand: &RawStridedRef<'_, T>,
        start_indices: &RawStridedRef<'_, I>,
    ) -> Result<()>
    where
        W: OverwriteWriter<T>,
    {
        if dest.dims() != &self.dest_dims[..]
            || dest.strides() != &self.dest_strides[..]
            || operand.dims() != &self.operand_dims[..]
            || operand.strides() != &self.operand_strides[..]
            || start_indices.dims() != &self.index_dims[..]
            || start_indices.strides() != &self.index_strides[..]
        {
            return Err(StridedError::PlanLayoutMismatch);
        }
        Ok(())
    }

    fn decode_replay_state(
        &self,
        mut linear: usize,
        dest_offset_base: isize,
        operand_offset_base: isize,
        index_offset_base: isize,
    ) -> Result<GatherReplayState> {
        let mut coords = AxisVec::with_capacity(self.dest_dims.len());
        let mut dest_offset = dest_offset_base;
        let mut window_offset = operand_offset_base;
        let mut index_batch_offset = index_offset_base;
        for (axis, &dim) in self.dest_dims.iter().enumerate() {
            let coord = linear % dim;
            linear /= dim;
            let replay_axis = self.replay.axes[axis];
            coords.push(coord);
            dest_offset = checked_offset_add(dest_offset, replay_axis.dest_step, coord)?;
            window_offset = checked_offset_add(window_offset, replay_axis.window_step, coord)?;
            index_batch_offset =
                checked_offset_add(index_batch_offset, replay_axis.index_batch_step, coord)?;
        }
        Ok(GatherReplayState {
            coords,
            dest_offset,
            window_offset,
            index_batch_offset,
        })
    }

    #[inline]
    fn advance_replay_state(&self, state: &mut GatherReplayState) {
        for (axis, replay_axis) in self.replay.axes.iter().enumerate() {
            let coord = state.coords[axis];
            if coord + 1 < self.dest_dims[axis] {
                state.coords[axis] = coord + 1;
                state.dest_offset += replay_axis.dest_step;
                state.window_offset += replay_axis.window_step;
                state.index_batch_offset += replay_axis.index_batch_step;
                return;
            }
            state.coords[axis] = 0;
            state.dest_offset += replay_axis.dest_reset;
            state.window_offset += replay_axis.window_reset;
            state.index_batch_offset += replay_axis.index_batch_reset;
        }
    }

    #[inline]
    fn clamp_window_start(&self, start: i64, operand_dim: usize) -> usize {
        let dim_size = self.operand_dims[operand_dim];
        let window_size = self.spec.slice_sizes[operand_dim];
        let max_start = dim_size.saturating_sub(window_size) as i64;
        start.clamp(0, max_start) as usize
    }
}

impl DynamicSlicePlan {
    /// Compile a fixed-window dynamic-slice traversal.
    ///
    /// `start_*` describes a rank-1 index vector whose length equals the
    /// operand rank. `dest_dims` must equal `slice_sizes`.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::StrideLengthMismatch`] when a stride list does
    /// not match its dimension list, [`StridedError::RankMismatch`] when
    /// `slice_sizes` differs from the operand rank or the start layout is not a
    /// rank-1 vector of that same length,
    /// [`StridedError::OffsetOverflow`] when an element count or layout span
    /// does not fit `usize`/`isize`, [`StridedError::ShapeMismatch`] when
    /// `dest_dims` differs from `slice_sizes` in extent or rank,
    /// [`StridedError::NonInjectiveOutputLayout`] when the destination layout
    /// maps two logical indices onto one offset, and
    /// [`StridedError::InvalidAxis`] when a slice size exceeds its operand
    /// extent.
    pub fn compile(
        operand_dims: &[usize],
        operand_strides: &[isize],
        start_dims: &[usize],
        start_strides: &[isize],
        dest_dims: &[usize],
        dest_strides: &[isize],
        slice_sizes: &[usize],
    ) -> Result<Self> {
        if operand_dims.len() != operand_strides.len()
            || start_dims.len() != start_strides.len()
            || dest_dims.len() != dest_strides.len()
        {
            return Err(StridedError::StrideLengthMismatch);
        }
        if slice_sizes.len() != operand_dims.len() {
            return Err(StridedError::RankMismatch(
                slice_sizes.len(),
                operand_dims.len(),
            ));
        }
        validate_start_vector(start_dims, operand_dims.len())?;
        checked_total_len(operand_dims)?;
        checked_total_len(start_dims)?;
        let total = checked_total_len(dest_dims)?;
        validate_layout_span(operand_dims, operand_strides)?;
        validate_layout_span(start_dims, start_strides)?;
        validate_layout_span(dest_dims, dest_strides)?;
        if dest_dims != slice_sizes {
            return Err(StridedError::ShapeMismatch(
                dest_dims.to_vec(),
                slice_sizes.to_vec(),
            ));
        }
        if !crate::layout_check::is_injective_layout(dest_dims, dest_strides) {
            return Err(StridedError::NonInjectiveOutputLayout);
        }
        validate_window_sizes(operand_dims, slice_sizes)?;
        let replay = WindowReplay::compile(slice_sizes, operand_strides, dest_strides)?;

        Ok(Self {
            operand_dims: operand_dims.into(),
            operand_strides: operand_strides.into(),
            start_dims: start_dims.into(),
            start_strides: start_strides.into(),
            dest_dims: dest_dims.into(),
            dest_strides: dest_strides.into(),
            slice_sizes: slice_sizes.into(),
            total,
            window: fuse_pair_layout(slice_sizes, dest_strides, operand_strides),
            replay,
        })
    }

    /// Execute the prepared dynamic-slice traversal.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::PlanLayoutMismatch`] when the operand, start,
    /// or destination view does not carry its compiled extents and strides,
    /// before any destination write. Start values are clamped to
    /// `operand_dims - slice_sizes`, so an out-of-range start is not an
    /// error.
    pub fn execute<T, I>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        operand: &RawStridedRef<'_, T>,
        starts: &RawStridedRef<'_, I>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        I: GatherIndex,
    {
        self.execute_with_writer(dest, operand, starts, exec)
    }

    pub(crate) fn execute_uninit<T, I>(
        &self,
        dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
        operand: &RawStridedRef<'_, T>,
        starts: &RawStridedRef<'_, I>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        I: GatherIndex,
    {
        self.execute_with_writer(dest, operand, starts, exec)
    }

    fn execute_with_writer<T, I, W>(
        &self,
        dest: &mut W,
        operand: &RawStridedRef<'_, T>,
        starts: &RawStridedRef<'_, I>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        I: GatherIndex,
        W: OverwriteWriter<T>,
    {
        self.check_call(dest, operand, starts)?;
        if self.total == 0 {
            return Ok(());
        }
        if self.uses_rank_one_contiguous_path() {
            return self.execute_rank_one_contiguous(dest, operand, starts, exec);
        }
        if let Some(window) = &self.window {
            return self.execute_fused_window(dest, operand, starts, window, exec);
        }
        let replay = &self.replay;
        let exec = crate::cost::for_len_tree(exec, KernelClass::Copy, self.total);
        if !exec.is_serial() {
            return self.execute_parallel(dest, operand, starts, exec, replay);
        }

        let mut starts_storage = CoordScratch::new(self.operand_dims.len());
        let clamped_starts = starts_storage.as_mut_slice();
        read_clamped_starts(
            starts,
            &self.operand_dims,
            &self.slice_sizes,
            clamped_starts,
        )?;
        let source_base =
            checked_strided_offset(operand.offset(), &self.operand_strides, clamped_starts)?;
        let mut state = replay.decode(0, source_base, dest.offset())?;
        let operand_data = operand.data();

        // INVARIANT: compile validated the full operand/destination spans and the
        // replay window spans; checked bases plus checked reset deltas therefore
        // keep every current offset reachable without per-element offset scans.
        for _ in 0..self.total {
            // SAFETY: the invariant above proves both current offsets are valid.
            let value = unsafe { *operand_data.as_ptr().offset(state.source_offset) };
            // SAFETY: the invariant above proves this logical destination offset
            // is in-bounds, and the output layout is injective.
            unsafe { dest.write_at(state.dest_offset, value) };
            replay.advance(&mut state);
        }
        Ok(())
    }

    /// Replay the compiled fused window from the clamped runtime source base.
    ///
    /// The window is a CopyPlan-style fused layout, so full inner runs are
    /// replayed with a pointer loop (slice copies for unit strides) and the
    /// work is split across workers above the repository threshold.
    fn execute_fused_window<T, I, W>(
        &self,
        dest: &mut W,
        operand: &RawStridedRef<'_, T>,
        starts: &RawStridedRef<'_, I>,
        window: &FusedPairLayout,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        I: GatherIndex,
        W: OverwriteWriter<T>,
    {
        let mut starts_storage = CoordScratch::new(self.operand_dims.len());
        let clamped_starts = starts_storage.as_mut_slice();
        read_clamped_starts(
            starts,
            &self.operand_dims,
            &self.slice_sizes,
            clamped_starts,
        )?;
        let source_base =
            checked_strided_offset(operand.offset(), &self.operand_strides, clamped_starts)?;
        let dest_base = dest.offset();
        // SAFETY: the validated writer owns the destination allocation.
        let dest_ptr = unsafe { dest.data_ptr() }.cast::<MaybeUninit<T>>();
        let tiles = pair_tiles::<MaybeUninit<T>, T>(window);
        // SAFETY: `window` fuses `slice_sizes` with the checked destination
        // and operand strides. The clamped window `start + slice_sizes` lies
        // inside `operand_dims`, so every source offset from `source_base` is
        // an offset the validated operand view can reach; the destination has
        // exactly `slice_sizes` dims and its validated layout reaches every
        // destination offset. Compile proved the destination injective, and
        // the exclusive writer borrow does not overlap the shared operand.
        // Writing `MaybeUninit::new` is valid for both initialized and
        // uninitialized destination slots. The injective destination lets the
        // window replay its tiles.
        unsafe {
            replay_fused_raw(
                dest_ptr,
                dest_base,
                operand.data().as_ptr(),
                source_base,
                window,
                tiles.as_ref(),
                |slot: &mut MaybeUninit<T>, value: T| *slot = MaybeUninit::new(value),
                KernelClass::Copy.ns_per_element(),
                exec,
            );
        }
        Ok(())
    }

    #[inline]
    fn uses_rank_one_contiguous_path(&self) -> bool {
        self.operand_dims.len() == 1 && self.operand_strides[0] == 1 && self.dest_strides[0] == 1
    }

    fn execute_rank_one_contiguous<T, I, W>(
        &self,
        dest: &mut W,
        operand: &RawStridedRef<'_, T>,
        starts: &RawStridedRef<'_, I>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        I: GatherIndex,
        W: OverwriteWriter<T>,
    {
        let mut clamped_starts = [0usize; 1];
        read_clamped_starts(
            starts,
            &self.operand_dims,
            &self.slice_sizes,
            &mut clamped_starts,
        )?;
        let source_start = checked_offset_add(operand.offset(), 1, clamped_starts[0])?;
        let source_start =
            usize::try_from(source_start).map_err(|_| StridedError::OffsetOverflow)?;
        let dest_start =
            usize::try_from(dest.offset()).map_err(|_| StridedError::OffsetOverflow)?;
        let source_end = source_start
            .checked_add(self.total)
            .ok_or(StridedError::OffsetOverflow)?;
        let source = operand
            .data()
            .get(source_start..source_end)
            .ok_or(StridedError::OffsetOverflow)?;
        // SAFETY: the validated writer owns the destination allocation.
        let dest_ptr = unsafe { dest.data_ptr() };
        // SAFETY: the source run was resolved above (`operand.data().get`,
        // :1260) and the destination run is the plan's own layout: compile set
        // `total = checked_total_len(dest_dims)` with `dest_dims == slice_sizes`
        // (:1050-:1054), required `dest_strides[0] == 1` on this rank-one path
        // (:1228), and `check_call` (:1341) proved the writer carries exactly the
        // compiled destination layout, whose reachable offsets `RawStridedMut::new`
        // validated (strided-view/src/raw.rs:1036, `validate_bounds` at :1042;
        // strided-view/src/view.rs:187). `data_ptr()` (copy_plan.rs:41) is the
        // allocation base, so `dest.offset() .. dest.offset() + total` is inside the
        // writer's allocation and the chunked workers write disjoint slices of it.
        unsafe {
            // Chunked across workers above the repository threshold; each
            // worker writes a disjoint chunk of the checked destination run.
            crate::threading::copy_contiguous(
                source.as_ptr(),
                dest_ptr.add(dest_start),
                self.total,
                exec,
            );
        }
        Ok(())
    }

    fn execute_parallel<T, I, W>(
        &self,
        dest: &mut W,
        operand: &RawStridedRef<'_, T>,
        starts: &RawStridedRef<'_, I>,
        exec: Exec<'_>,
        replay: &WindowReplay,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        I: GatherIndex,
        W: OverwriteWriter<T>,
    {
        let mut clamped_starts: AxisVec<usize> = (0..self.operand_dims.len()).map(|_| 0).collect();
        read_clamped_starts(
            starts,
            &self.operand_dims,
            &self.slice_sizes,
            &mut clamped_starts,
        )?;
        let source_base =
            checked_strided_offset(operand.offset(), &self.operand_strides, &clamped_starts)?;
        let dest_base = dest.offset();
        let operand_ptr = crate::threading::SendPtr(operand.data().as_ptr() as *mut T);
        // SAFETY: the validated writer owns the destination allocation.
        let dest_ptr = crate::threading::SendPtr(unsafe { dest.data_ptr() });

        crate::threading::parallel_map_reduce(
            0..self.total,
            exec,
            &|range| {
                let mut state = replay.decode(range.start, source_base, dest_base)?;
                let operand_ptr = operand_ptr.as_const();
                let dest_ptr = dest_ptr.as_ptr();

                // INVARIANT: each worker decodes one checked range start. Its
                // replay range is disjoint, and the injective output layout makes
                // all writes distinct across workers.
                for _ in range {
                    // SAFETY: checked replay bases/resets keep both offsets in
                    // their validated allocations.
                    unsafe {
                        dest_ptr
                            .offset(state.dest_offset)
                            .write(operand_ptr.offset(state.source_offset).read());
                    }
                    replay.advance(&mut state);
                }
                Ok(())
            },
            &|left, right| left.and(right),
        )
    }

    fn check_call<T, I, W>(
        &self,
        dest: &W,
        operand: &RawStridedRef<'_, T>,
        starts: &RawStridedRef<'_, I>,
    ) -> Result<()>
    where
        W: OverwriteWriter<T>,
    {
        if dest.dims() != &self.dest_dims[..]
            || dest.strides() != &self.dest_strides[..]
            || operand.dims() != &self.operand_dims[..]
            || operand.strides() != &self.operand_strides[..]
            || starts.dims() != &self.start_dims[..]
            || starts.strides() != &self.start_strides[..]
        {
            return Err(StridedError::PlanLayoutMismatch);
        }
        Ok(())
    }
}

impl DynamicUpdateSlicePlan {
    /// Compile a dynamic-update-slice traversal.
    ///
    /// `dest_dims` must match `operand_dims`; execution materializes
    /// `dest = operand` and then overwrites the clamped update window.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::StrideLengthMismatch`] when a stride list does
    /// not match its dimension list, [`StridedError::RankMismatch`] when
    /// `update_dims` differs from the operand rank or the start layout is not a
    /// rank-1 vector of that same length,
    /// [`StridedError::OffsetOverflow`] when an element count or layout span
    /// does not fit `usize`/`isize`, [`StridedError::ShapeMismatch`] when
    /// `dest_dims` differs from `operand_dims` in extent or rank,
    /// [`StridedError::NonInjectiveOutputLayout`] when the destination layout
    /// maps two logical indices onto one offset, and
    /// [`StridedError::InvalidAxis`] when an update extent exceeds its
    /// operand extent.
    // Compile key: the operand, start, update and destination layouts, as four
    // dim/stride pairs; a wrapper type would only move the same slices.
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        operand_dims: &[usize],
        operand_strides: &[isize],
        start_dims: &[usize],
        start_strides: &[isize],
        update_dims: &[usize],
        update_strides: &[isize],
        dest_dims: &[usize],
        dest_strides: &[isize],
    ) -> Result<Self> {
        if operand_dims.len() != operand_strides.len()
            || start_dims.len() != start_strides.len()
            || update_dims.len() != update_strides.len()
            || dest_dims.len() != dest_strides.len()
        {
            return Err(StridedError::StrideLengthMismatch);
        }
        if update_dims.len() != operand_dims.len() {
            return Err(StridedError::RankMismatch(
                update_dims.len(),
                operand_dims.len(),
            ));
        }
        validate_start_vector(start_dims, operand_dims.len())?;
        checked_total_len(operand_dims)?;
        checked_total_len(start_dims)?;
        let total = checked_total_len(update_dims)?;
        validate_layout_span(operand_dims, operand_strides)?;
        validate_layout_span(start_dims, start_strides)?;
        validate_layout_span(update_dims, update_strides)?;
        validate_layout_span(dest_dims, dest_strides)?;
        if dest_dims != operand_dims {
            return Err(StridedError::ShapeMismatch(
                dest_dims.to_vec(),
                operand_dims.to_vec(),
            ));
        }
        if !crate::layout_check::is_injective_layout(dest_dims, dest_strides) {
            return Err(StridedError::NonInjectiveOutputLayout);
        }
        validate_window_sizes(operand_dims, update_dims)?;
        let copy_plan = CopyPlan::compile(operand_dims, dest_strides, operand_strides)?;
        let replay = WindowReplay::compile(update_dims, update_strides, dest_strides)?;

        Ok(Self {
            operand_dims: operand_dims.into(),
            operand_strides: operand_strides.into(),
            start_dims: start_dims.into(),
            start_strides: start_strides.into(),
            update_dims: update_dims.into(),
            update_strides: update_strides.into(),
            dest_dims: dest_dims.into(),
            dest_strides: dest_strides.into(),
            total,
            copy_plan,
            window: fuse_pair_layout(update_dims, dest_strides, update_strides),
            replay,
        })
    }

    /// Execute the prepared dynamic-update-slice traversal.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::PlanLayoutMismatch`] when the operand, update,
    /// start, or destination view does not carry its compiled extents and
    /// strides, and [`StridedError::OffsetOverflow`] when the checked update
    /// window does not fit `isize`. Both are decided before the operand copy
    /// runs, so a failed call leaves `dest` unmodified. Start values are
    /// clamped to `operand_dims - update_dims`, so an out-of-range start is
    /// not an error.
    pub fn execute<T, I>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        operand: &RawStridedRef<'_, T>,
        update: &RawStridedRef<'_, T>,
        starts: &RawStridedRef<'_, I>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        I: GatherIndex,
    {
        let mut exec = exec;
        self.check_call(dest, operand, update, starts)?;
        let window = self.preflight_update_window(dest, update, starts)?;
        self.copy_plan.execute(dest, operand, exec.reborrow())?;
        self.execute_update_with_writer(dest, update, &window, exec)
    }

    /// Execute dynamic update into a destination whose reachable slots may be
    /// uninitialized. The copy completes before any read-modify-write access,
    /// and a rejected update window is reported before the copy.
    pub(crate) fn execute_uninit<'a, T, I>(
        &self,
        dest: &'a mut RawStridedMut<'a, MaybeUninit<T>>,
        operand: &RawStridedRef<'_, T>,
        update: &RawStridedRef<'_, T>,
        starts: &RawStridedRef<'_, I>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        I: GatherIndex,
    {
        self.check_call(dest, operand, update, starts)?;
        let window = self.preflight_update_window(dest, update, starts)?;
        self.copy_plan
            .execute_uninit_then(dest, operand, exec, |mut receipt, exec| {
                self.execute_update_with_writer(&mut receipt, update, &window, exec)
            })?
    }

    /// Resolve every fallible step of the update phase before the operand copy.
    fn preflight_update_window<'a, T, I, W>(
        &self,
        dest: &W,
        update: &'a RawStridedRef<'_, T>,
        starts: &RawStridedRef<'_, I>,
    ) -> Result<UpdateWindow<'a, T>>
    where
        T: Copy,
        I: GatherIndex,
        W: OverwriteWriter<T>,
    {
        if self.total == 0 {
            return Ok(UpdateWindow::Empty);
        }
        let mut starts_storage = CoordScratch::new(self.operand_dims.len());
        let clamped_starts = starts_storage.as_mut_slice();
        read_clamped_starts(
            starts,
            &self.operand_dims,
            &self.update_dims,
            clamped_starts,
        )?;
        let dest_base = checked_strided_offset(dest.offset(), &self.dest_strides, clamped_starts)?;
        // The replay adds per-axis steps to these bases, on every worker, so
        // the whole window and not only its base has to fit `isize` first.
        self.replay.source_span.covers(update.offset())?;
        self.replay.dest_span.covers(dest_base)?;
        if self.uses_rank_one_contiguous_path() {
            let update_start =
                usize::try_from(update.offset()).map_err(|_| StridedError::OffsetOverflow)?;
            let dest_start =
                usize::try_from(dest_base).map_err(|_| StridedError::OffsetOverflow)?;
            let update_end = update_start
                .checked_add(self.total)
                .ok_or(StridedError::OffsetOverflow)?;
            let run = update
                .data()
                .get(update_start..update_end)
                .ok_or(StridedError::OffsetOverflow)?;
            return Ok(UpdateWindow::RankOne {
                update: run,
                dest_start,
            });
        }
        Ok(UpdateWindow::Replay { dest_base })
    }

    fn execute_update_with_writer<T, W>(
        &self,
        dest: &mut W,
        update: &RawStridedRef<'_, T>,
        window: &UpdateWindow<'_, T>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        W: OverwriteWriter<T>,
    {
        let dest_base = match *window {
            UpdateWindow::Empty => return Ok(()),
            UpdateWindow::RankOne { update, dest_start } => {
                return self.execute_rank_one_contiguous(dest, update, dest_start, exec)
            }
            UpdateWindow::Replay { dest_base } => dest_base,
        };
        if let Some(fused) = &self.window {
            return self.execute_fused_window(dest, update, dest_base, fused, exec);
        }
        let replay = &self.replay;
        let exec = crate::cost::for_len_tree(exec, KernelClass::Copy, self.total);
        if !exec.is_serial() {
            return self.execute_update_parallel(dest, update, dest_base, exec);
        }

        let mut state = replay.start(update.offset(), dest_base);
        let update_data = update.data();

        // INVARIANT: the initial CopyPlan has completed before this replay;
        // compile validated both window spans and the preflight proved every
        // replay offset fits from the checked bases.
        for _ in 0..self.total {
            // SAFETY: checked replay metadata proves the current update read.
            let value = unsafe { *update_data.as_ptr().offset(state.source_offset) };
            // SAFETY: the copied, injective destination layout proves this write
            // is initialized and in-bounds.
            unsafe { dest.write_at(state.dest_offset, value) };
            replay.advance(&mut state);
        }
        Ok(())
    }

    /// Replay the compiled fused update window at the preflighted runtime
    /// destination base (see [`DynamicSlicePlan`]'s fused window replay).
    fn execute_fused_window<T, W>(
        &self,
        dest: &mut W,
        update: &RawStridedRef<'_, T>,
        dest_base: isize,
        window: &FusedPairLayout,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        W: OverwriteWriter<T>,
    {
        // SAFETY: the validated writer owns the destination allocation.
        let dest_ptr = unsafe { dest.data_ptr() }.cast::<MaybeUninit<T>>();
        let tiles = pair_tiles::<MaybeUninit<T>, T>(window);
        // SAFETY: `window` fuses `update_dims` with the checked destination
        // and update strides. The update view has exactly `update_dims` and
        // its validated layout reaches every source offset; the clamped window
        // lies inside `dest_dims`, so every destination offset from
        // `dest_base` is reachable through the validated destination layout.
        // Compile proved the destination injective, and the exclusive writer
        // borrow does not overlap the shared update. The injective destination
        // lets the window replay its tiles.
        unsafe {
            replay_fused_raw(
                dest_ptr,
                dest_base,
                update.data().as_ptr(),
                update.offset(),
                window,
                tiles.as_ref(),
                |slot: &mut MaybeUninit<T>, value: T| *slot = MaybeUninit::new(value),
                KernelClass::Copy.ns_per_element(),
                exec,
            );
        }
        Ok(())
    }

    #[inline]
    fn uses_rank_one_contiguous_path(&self) -> bool {
        self.operand_dims.len() == 1
            && self.operand_strides[0] == 1
            && self.update_strides[0] == 1
            && self.dest_strides[0] == 1
    }

    fn execute_rank_one_contiguous<T, W>(
        &self,
        dest: &mut W,
        update: &[T],
        dest_start: usize,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        W: OverwriteWriter<T>,
    {
        // SAFETY: the validated writer owns the destination allocation.
        let dest_ptr = unsafe { dest.data_ptr() };
        // SAFETY: the preflight resolved the update run (`update.data().get`,
        // :1534) and proved the window span fits `isize` (`dest_span.covers`,
        // :1523); the destination run is the plan's own layout: compile set
        // `total = checked_total_len(update_dims)` (:1409) with
        // `dest_dims == operand_dims` (:1414) and `validate_window_sizes`
        // (:1423), this rank-one path requires `dest_strides[0] == 1` (:1630),
        // and `check_call` (:1718) proved the writer carries exactly that
        // layout. The last offset this run writes is `dest_start + total - 1`
        // (`total >= 1` here: an empty window returns as `UpdateWindow::Empty`
        // at :1508 and leaves at :1556), and that offset cannot pass the
        // highest one `RawStridedMut::new` validated
        // (strided-view/src/raw.rs:1036, `validate_bounds` at :1042). The
        // exclusive end `dest_start + total` is one past the run's last
        // element and may equal `max + 1`.
        unsafe {
            // Chunked across workers above the repository threshold; each
            // worker writes a disjoint chunk of the checked destination run.
            crate::threading::copy_contiguous(
                update.as_ptr(),
                dest_ptr.add(dest_start),
                self.total,
                exec,
            );
        }
        Ok(())
    }

    fn execute_update_parallel<T, W>(
        &self,
        dest: &mut W,
        update: &RawStridedRef<'_, T>,
        dest_base: isize,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        W: OverwriteWriter<T>,
    {
        let replay = &self.replay;
        let source_base = update.offset();
        let update_ptr = crate::threading::SendPtr(update.data().as_ptr() as *mut T);
        // SAFETY: the validated writer owns the destination allocation.
        let dest_ptr = crate::threading::SendPtr(unsafe { dest.data_ptr() });

        crate::threading::parallel_map_reduce(
            0..self.total,
            exec,
            &|range| {
                let mut state = replay.decode(range.start, source_base, dest_base)?;
                let update_ptr = update_ptr.as_const();
                let dest_ptr = dest_ptr.as_ptr();

                // INVARIANT: the initial operand copy completed before this
                // replay; workers own disjoint ranges and injective output
                // layouts make their writes distinct. Preflight bounded both
                // window spans, so a worker decode cannot overflow.
                for _ in range {
                    // SAFETY: checked replay bases/resets prove both offsets are
                    // within the initialized update and destination allocations.
                    unsafe {
                        dest_ptr
                            .offset(state.dest_offset)
                            .write(update_ptr.offset(state.source_offset).read());
                    }
                    replay.advance(&mut state);
                }
                Ok(())
            },
            &|left, right| left.and(right),
        )
    }

    fn check_call<T, I, W>(
        &self,
        dest: &W,
        operand: &RawStridedRef<'_, T>,
        update: &RawStridedRef<'_, T>,
        starts: &RawStridedRef<'_, I>,
    ) -> Result<()>
    where
        W: OverwriteWriter<T>,
    {
        if dest.dims() != &self.dest_dims[..]
            || dest.strides() != &self.dest_strides[..]
            || operand.dims() != &self.operand_dims[..]
            || operand.strides() != &self.operand_strides[..]
            || update.dims() != &self.update_dims[..]
            || update.strides() != &self.update_strides[..]
            || starts.dims() != &self.start_dims[..]
            || starts.strides() != &self.start_strides[..]
        {
            return Err(StridedError::PlanLayoutMismatch);
        }
        Ok(())
    }
}

impl ScatterPlan {
    /// Compile an additive scatter traversal.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::StrideLengthMismatch`] when a stride list does
    /// not match its dimension list, [`StridedError::OffsetOverflow`] when an
    /// element count or layout span does not fit `usize`/`isize`,
    /// [`StridedError::ShapeMismatch`] when `dest_dims` differs from
    /// `operand_dims` or an update batch extent differs from the index batch
    /// shape, and [`StridedError::NonInjectiveOutputLayout`] when the
    /// destination layout cannot be proven injective (aliasing elements, or a
    /// span the check cannot represent).
    /// [`StridedError::InvalidAxis`] reports `inserted_window_dims`,
    /// `scatter_dims_to_operand_dims`, or `update_window_dims` that repeat or
    /// exceed their rank, `index_vector_dim` beyond the index rank, or an
    /// update window extent beyond its operand extent; and
    /// [`StridedError::RankMismatch`] reports an index vector that does not
    /// match `scatter_dims_to_operand_dims`, a wrong-length
    /// `update_window_dims`, or an update batch rank that disagrees with the
    /// index batch shape.
    // Compile key: the operand, index, update and destination layouts, as four
    // dim/stride pairs, plus the scatter spec.
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        operand_dims: &[usize],
        operand_strides: &[isize],
        index_dims: &[usize],
        index_strides: &[isize],
        update_dims: &[usize],
        update_strides: &[isize],
        dest_dims: &[usize],
        dest_strides: &[isize],
        spec: ScatterSpec,
    ) -> Result<Self> {
        if operand_dims.len() != operand_strides.len()
            || index_dims.len() != index_strides.len()
            || update_dims.len() != update_strides.len()
            || dest_dims.len() != dest_strides.len()
        {
            return Err(StridedError::StrideLengthMismatch);
        }
        checked_total_len(operand_dims)?;
        checked_total_len(index_dims)?;
        checked_total_len(update_dims)?;
        if dest_dims != operand_dims {
            return Err(StridedError::ShapeMismatch(
                dest_dims.to_vec(),
                operand_dims.to_vec(),
            ));
        }
        if !crate::layout_check::is_injective_layout(dest_dims, dest_strides) {
            return Err(StridedError::NonInjectiveOutputLayout);
        }

        let operand_rank = operand_dims.len();
        validate_unique_axes(&spec.inserted_window_dims, operand_rank)?;
        validate_unique_axes(&spec.scatter_dims_to_operand_dims, operand_rank)?;
        if spec.index_vector_dim > index_dims.len() {
            return Err(StridedError::InvalidAxis {
                axis: spec.index_vector_dim,
                rank: index_dims.len() + 1,
            });
        }
        let index_vector_size = if spec.index_vector_dim == index_dims.len() {
            1
        } else {
            index_dims[spec.index_vector_dim]
        };
        if index_vector_size != spec.scatter_dims_to_operand_dims.len() {
            return Err(StridedError::RankMismatch(
                index_vector_size,
                spec.scatter_dims_to_operand_dims.len(),
            ));
        }

        let batch_shape = index_batch_shape(index_dims, spec.index_vector_dim);
        let window_dims = operand_window_dims(operand_rank, &spec.inserted_window_dims);
        if spec.update_window_dims.len() != window_dims.len() {
            return Err(StridedError::RankMismatch(
                spec.update_window_dims.len(),
                window_dims.len(),
            ));
        }

        let update_rank = update_dims.len();
        let expected_batch_rank = update_rank
            .checked_sub(spec.update_window_dims.len())
            .ok_or(StridedError::RankMismatch(
                spec.update_window_dims.len(),
                update_rank,
            ))?;
        if expected_batch_rank != batch_shape.len() {
            return Err(StridedError::RankMismatch(
                expected_batch_rank,
                batch_shape.len(),
            ));
        }
        validate_unique_axes(&spec.update_window_dims, update_rank)?;

        let mut is_update_window_dim: AxisVec<bool> = (0..update_rank).map(|_| false).collect();
        for &axis in &spec.update_window_dims {
            is_update_window_dim[axis] = true;
        }

        let mut batch_axis = 0usize;
        for axis in 0..update_rank {
            if !is_update_window_dim[axis] {
                if update_dims[axis] != batch_shape[batch_axis] {
                    return Err(StridedError::ShapeMismatch(
                        update_dims.to_vec(),
                        expected_scatter_update_shape(&batch_shape, &spec, update_dims).to_vec(),
                    ));
                }
                batch_axis += 1;
            }
        }

        let mut window_shape: AxisVec<usize> = (0..operand_rank).map(|_| 1).collect();
        let mut window_shape_updates: AxisVec<usize> =
            AxisVec::with_capacity(spec.update_window_dims.len());
        for (pos, &update_axis) in spec.update_window_dims.iter().enumerate() {
            let dim = update_dims[update_axis];
            window_shape_updates.push(dim);
            window_shape[window_dims[pos]] = dim;
        }
        validate_window_sizes(operand_dims, &window_shape)?;

        let batch_elems = checked_total_len(&batch_shape)?;
        let window_elems = checked_total_len(&window_shape_updates)?;
        let copy_plan = CopyPlan::compile(operand_dims, dest_strides, operand_strides)?;
        let replay = ScatterReplay::compile(
            &batch_shape,
            index_dims,
            index_strides,
            spec.index_vector_dim,
            spec.scatter_dims_to_operand_dims.len(),
            update_dims,
            update_strides,
            &spec.update_window_dims,
            &is_update_window_dim,
            &window_shape_updates,
            &window_dims,
            dest_strides,
        )?;

        Ok(Self {
            operand_dims: operand_dims.into(),
            operand_strides: operand_strides.into(),
            index_dims: index_dims.into(),
            index_strides: index_strides.into(),
            update_dims: update_dims.into(),
            update_strides: update_strides.into(),
            dest_dims: dest_dims.into(),
            dest_strides: dest_strides.into(),
            spec,
            window_shape,
            batch_elems,
            window_elems,
            copy_plan,
            replay,
        })
    }

    /// Execute the prepared additive scatter traversal.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::PlanLayoutMismatch`] when the operand, index,
    /// update, or destination view does not carry its compiled extents and
    /// strides, before the operand copy runs. Start values are clamped into
    /// the operand window, so an out-of-range scatter index is not an error.
    #[inline(always)]
    pub fn execute<T, I>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        operand: &RawStridedRef<'_, T>,
        scatter_indices: &RawStridedRef<'_, I>,
        updates: &RawStridedRef<'_, T>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Add<Output = T> + Send + Sync,
        I: GatherIndex,
    {
        self.check_call(dest, operand, scatter_indices, updates)?;
        self.copy_plan.execute(dest, operand, exec)?;
        self.execute_updates(dest, scatter_indices, updates, |a, b| a + b)
    }

    /// Execute additive scatter into a destination whose reachable slots may
    /// be uninitialized. The operand copy completes before any RMW access.
    pub(crate) fn execute_uninit<'a, T, I>(
        &self,
        dest: &'a mut RawStridedMut<'a, MaybeUninit<T>>,
        operand: &RawStridedRef<'_, T>,
        scatter_indices: &RawStridedRef<'_, I>,
        updates: &RawStridedRef<'_, T>,
        combine: fn(T, T) -> T,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Add<Output = T> + Send + Sync,
        I: GatherIndex,
    {
        self.check_call(dest, operand, scatter_indices, updates)?;
        self.copy_plan
            .execute_uninit_then(dest, operand, exec, |mut receipt, _| {
                self.execute_updates(&mut receipt, scatter_indices, updates, combine)
            })?
    }

    #[inline(always)]
    fn execute_updates<T, I, W, F>(
        &self,
        dest: &mut W,
        scatter_indices: &RawStridedRef<'_, I>,
        updates: &RawStridedRef<'_, T>,
        combine: F,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        I: GatherIndex,
        W: ReadModifyWrite<T>,
        F: Fn(T, T) -> T + Copy,
    {
        if self.batch_elems == 0 || self.window_elems == 0 {
            return Ok(());
        }
        if self.uses_rank_one_scalar_update_path() {
            return self.execute_rank_one_scalar_updates(dest, scatter_indices, updates, combine);
        }
        self.execute_generic_updates(dest, scatter_indices, updates, combine)
    }

    #[inline(never)]
    fn execute_generic_updates<T, I, W, F>(
        &self,
        dest: &mut W,
        scatter_indices: &RawStridedRef<'_, I>,
        updates: &RawStridedRef<'_, T>,
        combine: F,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
        I: GatherIndex,
        W: ReadModifyWrite<T>,
        F: Fn(T, T) -> T + Copy,
    {
        // Overlapping additive updates are order-sensitive, so this remains a
        // deterministic serial replay until a combine-aware parallel plan exists.
        let replay = &self.replay;

        let mut operand_base_storage = CoordScratch::new(self.operand_dims.len());
        let operand_base = operand_base_storage.as_mut_slice();
        let mut batch_state = replay
            .batch
            .decode(0, scatter_indices.offset(), updates.offset())?;
        let index_data = scatter_indices.data();
        let update_data = updates.data();

        // INVARIANT: batch and window replay metadata was checked at compile;
        // the indirect component loop below is the only data-dependent lookup.
        for _ in 0..self.batch_elems {
            operand_base.fill(0);
            for (&component_offset, &operand_axis) in replay
                .index_component_offsets
                .iter()
                .zip(self.spec.scatter_dims_to_operand_dims.iter())
            {
                let index_offset = batch_state
                    .source_offset
                    .checked_add(component_offset)
                    .ok_or(StridedError::OffsetOverflow)?;
                // SAFETY: checked batch replay and component offsets keep this
                // indirect index read inside the validated index allocation.
                let start = unsafe { *index_data.as_ptr().offset(index_offset) }.to_i64();
                operand_base[operand_axis] = clamp_window_start(
                    start,
                    self.operand_dims[operand_axis],
                    self.window_shape[operand_axis],
                );
            }

            let dest_base = checked_strided_offset(dest.offset(), dest.strides(), operand_base)?;
            let mut window_state = replay
                .window
                .decode(0, batch_state.dest_offset, dest_base)?;
            for _ in 0..self.window_elems {
                // SAFETY: copy completion and checked replay state prove both
                // the update read and initialized destination RMW are valid.
                let value = unsafe { *update_data.as_ptr().offset(window_state.source_offset) };
                unsafe { dest.add_at(window_state.dest_offset, value, combine) };
                replay.window.advance(&mut window_state);
            }
            replay.batch.advance(&mut batch_state);
        }
        Ok(())
    }

    fn uses_rank_one_scalar_update_path(&self) -> bool {
        self.operand_dims.len() == 1
            && self.index_dims.len() == 2
            && self.update_dims.len() == 1
            && self.dest_dims.len() == 1
            && self.operand_strides[0] == 1
            && self.index_strides[0] == 1
            && self.update_strides[0] == 1
            && self.dest_strides[0] == 1
            && self.index_dims[1] == 1
            && self.spec.update_window_dims.is_empty()
            && self.spec.inserted_window_dims.as_slice() == [0]
            && self.spec.scatter_dims_to_operand_dims.as_slice() == [0]
            && self.spec.index_vector_dim == 1
            && self.window_elems == 1
    }

    #[inline(always)]
    fn execute_rank_one_scalar_updates<T, I, W, F>(
        &self,
        dest: &mut W,
        scatter_indices: &RawStridedRef<'_, I>,
        updates: &RawStridedRef<'_, T>,
        combine: F,
    ) -> Result<()>
    where
        T: Copy,
        I: GatherIndex,
        W: ReadModifyWrite<T>,
        F: Fn(T, T) -> T + Copy,
    {
        let mut index_offset = scatter_indices.offset();
        let mut update_offset = updates.offset();
        let dest_offset = dest.offset();
        let index_data = scatter_indices.data();
        let update_data = updates.data();
        // SAFETY: the validated writer owns the destination allocation.
        let dest_ptr = unsafe { dest.data_ptr() };

        // INVARIANT: compile and check_call validated compact rank-one index
        // and update layouts. Ordered replay preserves repeated-index semantics.
        // `index_offset`/`update_offset` are the pointer walk, not counters.
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..self.batch_elems {
            // SAFETY: the invariant above proves index/update reads and the
            // clamped destination offset are in their validated allocations.
            unsafe {
                let start = (*index_data.as_ptr().offset(index_offset)).to_i64();
                let output_offset =
                    dest_offset + clamp_window_start(start, self.operand_dims[0], 1) as isize;
                let output = dest_ptr.offset(output_offset);
                output.write(combine(
                    output.read(),
                    *update_data.as_ptr().offset(update_offset),
                ));
            }
            index_offset += 1;
            update_offset += 1;
        }
        Ok(())
    }

    fn check_call<T, I, W>(
        &self,
        dest: &W,
        operand: &RawStridedRef<'_, T>,
        scatter_indices: &RawStridedRef<'_, I>,
        updates: &RawStridedRef<'_, T>,
    ) -> Result<()>
    where
        W: OverwriteWriter<T>,
    {
        if dest.dims() != &self.dest_dims[..]
            || dest.strides() != &self.dest_strides[..]
            || operand.dims() != &self.operand_dims[..]
            || operand.strides() != &self.operand_strides[..]
            || scatter_indices.dims() != &self.index_dims[..]
            || scatter_indices.strides() != &self.index_strides[..]
            || updates.dims() != &self.update_dims[..]
            || updates.strides() != &self.update_strides[..]
        {
            return Err(StridedError::PlanLayoutMismatch);
        }
        Ok(())
    }
}

fn validate_unique_axes(axes: &[usize], rank: usize) -> Result<()> {
    // Inline boolean metadata: rank <= INLINE_RANK validation allocates
    // nothing, and it spills above the inline bound.
    let mut seen: InlineVec<bool> = InlineVec::new();
    seen.resize(rank, false);
    for &axis in axes {
        if axis >= rank {
            return Err(StridedError::InvalidAxis { axis, rank });
        }
        if seen[axis] {
            return Err(StridedError::InvalidAxis { axis, rank });
        }
        seen[axis] = true;
    }
    Ok(())
}

fn validate_start_vector(start_dims: &[usize], operand_rank: usize) -> Result<()> {
    if start_dims.len() != 1 {
        return Err(StridedError::RankMismatch(start_dims.len(), 1));
    }
    if start_dims[0] != operand_rank {
        return Err(StridedError::RankMismatch(start_dims[0], operand_rank));
    }
    Ok(())
}

fn validate_window_sizes(operand_dims: &[usize], window_sizes: &[usize]) -> Result<()> {
    if operand_dims.len() != window_sizes.len() {
        return Err(StridedError::RankMismatch(
            window_sizes.len(),
            operand_dims.len(),
        ));
    }
    for (axis, (&window, &dim)) in window_sizes.iter().zip(operand_dims.iter()).enumerate() {
        if window > dim {
            return Err(StridedError::InvalidAxis {
                axis,
                rank: operand_dims.len(),
            });
        }
    }
    Ok(())
}

fn read_clamped_starts<I>(
    starts: &RawStridedRef<'_, I>,
    operand_dims: &[usize],
    window_sizes: &[usize],
    out: &mut [usize],
) -> Result<()>
where
    I: GatherIndex,
{
    debug_assert_eq!(operand_dims.len(), window_sizes.len());
    debug_assert_eq!(operand_dims.len(), out.len());
    for axis in 0..operand_dims.len() {
        let offset = checked_offset_add(starts.offset(), starts.strides()[0], axis)?;
        let start = unsafe { *starts.data().as_ptr().offset(offset) }.to_i64();
        out[axis] = clamp_window_start(start, operand_dims[axis], window_sizes[axis]);
    }
    Ok(())
}

#[inline]
fn clamp_window_start(start: i64, dim_size: usize, window_size: usize) -> usize {
    let max_start = dim_size.saturating_sub(window_size) as i64;
    start.clamp(0, max_start) as usize
}

fn expected_scatter_update_shape(
    batch_shape: &[usize],
    spec: &ScatterSpec,
    update_dims: &[usize],
) -> AxisVec<usize> {
    let mut expected: AxisVec<usize> = AxisVec::with_capacity(update_dims.len());
    let mut batch_axis = 0usize;
    for (axis, &dim) in update_dims.iter().enumerate() {
        if spec.update_window_dims.contains(&axis) {
            expected.push(dim);
        } else {
            expected.push(batch_shape[batch_axis]);
            batch_axis += 1;
        }
    }
    expected
}

fn operand_window_dims(rank: usize, collapsed_slice_dims: &[usize]) -> AxisVec<usize> {
    (0..rank)
        .filter(|axis| !collapsed_slice_dims.contains(axis))
        .collect()
}

fn index_batch_shape(index_dims: &[usize], index_vector_dim: usize) -> AxisVec<usize> {
    if index_vector_dim == index_dims.len() {
        return index_dims.into();
    }
    index_dims
        .iter()
        .enumerate()
        .filter_map(|(axis, &dim)| (axis != index_vector_dim).then_some(dim))
        .collect()
}

/// Validate a layout's reachable offsets and return them relative to its base.
fn validate_layout_span(dims: &[usize], strides: &[isize]) -> Result<OffsetSpan> {
    if dims.len() != strides.len() {
        return Err(StridedError::StrideLengthMismatch);
    }
    let mut span = OffsetSpan::default();
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        let last =
            isize::try_from(dim.saturating_sub(1)).map_err(|_| StridedError::OffsetOverflow)?;
        let extent = stride
            .checked_mul(last)
            .ok_or(StridedError::OffsetOverflow)?;
        if extent < 0 {
            span.min = span
                .min
                .checked_add(extent)
                .ok_or(StridedError::OffsetOverflow)?;
        } else {
            span.max = span
                .max
                .checked_add(extent)
                .ok_or(StridedError::OffsetOverflow)?;
        }
    }
    Ok(span)
}

fn checked_replay_reset(dim: usize, step: isize) -> Result<isize> {
    if dim == 0 {
        return Ok(0);
    }
    let last = isize::try_from(dim - 1).map_err(|_| StridedError::OffsetOverflow)?;
    step.checked_mul(last)
        .and_then(isize::checked_neg)
        .ok_or(StridedError::OffsetOverflow)
}

fn checked_total_len(dims: &[usize]) -> Result<usize> {
    if dims.is_empty() {
        return Ok(1);
    }
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or(StridedError::OffsetOverflow)
}

fn checked_strided_offset(base: isize, strides: &[isize], index: &[usize]) -> Result<isize> {
    let mut offset = base;
    for (&stride, &coord) in strides.iter().zip(index.iter()) {
        offset = checked_offset_add(offset, stride, coord)?;
    }
    Ok(offset)
}

fn checked_offset_add(base: isize, stride: isize, coord: usize) -> Result<isize> {
    let coord = isize::try_from(coord).map_err(|_| StridedError::OffsetOverflow)?;
    let scaled = stride
        .checked_mul(coord)
        .ok_or(StridedError::OffsetOverflow)?;
    base.checked_add(scaled).ok_or(StridedError::OffsetOverflow)
}

struct CoordScratch {
    inline: [usize; RAW_FUSED_RANK_LIMIT],
    heap: Option<Vec<usize>>,
    len: usize,
}

impl CoordScratch {
    fn new(len: usize) -> Self {
        if len <= RAW_FUSED_RANK_LIMIT {
            Self {
                inline: [0; RAW_FUSED_RANK_LIMIT],
                heap: None,
                len,
            }
        } else {
            Self {
                inline: [0; RAW_FUSED_RANK_LIMIT],
                heap: Some(vec![0; len]),
                len,
            }
        }
    }

    fn as_mut_slice(&mut self) -> &mut [usize] {
        match &mut self.heap {
            Some(heap) => heap,
            None => &mut self.inline[..self.len],
        }
    }
}

#[cfg(test)]
#[path = "gather_plan/tests/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "gather_plan/tests/parallel_tests.rs"]
mod parallel_tests;
