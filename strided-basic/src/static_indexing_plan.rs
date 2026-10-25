//! Prepared static-indexing plans over raw strided value layouts.
//!
//! This module owns reusable static indexing traversal for downstream tensor
//! runtimes. It keeps allocation, dtype policy, and tensor-level validation out
//! of `strided-kernel`; callers provide already-owned output buffers and fixed
//! raw descriptors.

use core::mem::MaybeUninit;

use crate::{CopyPlan, Exec, RawStridedMut, RawStridedRef, Result, StridedError};

type AxisVec<T> = crate::inline::InlineVec<T>;

/// A compiled static-slice traversal.
///
/// `compile` validates the fixed `starts`/`limits`/`slice_strides` contract and
/// lowers replay to a strided copy from the corresponding source view.
#[derive(Clone, Debug)]
pub struct SlicePlan {
    operand_dims: AxisVec<usize>,
    operand_strides: AxisVec<isize>,
    dest_dims: AxisVec<usize>,
    dest_strides: AxisVec<isize>,
    source_strides: AxisVec<isize>,
    source_offset_delta: isize,
    copy_plan: CopyPlan,
}

/// A compiled reverse traversal over selected axes.
///
/// Replay is a strided copy from a negative-stride source view.
#[derive(Clone, Debug)]
pub struct ReversePlan {
    operand_dims: AxisVec<usize>,
    operand_strides: AxisVec<isize>,
    dest_strides: AxisVec<isize>,
    source_strides: AxisVec<isize>,
    source_offset_delta: isize,
    copy_plan: CopyPlan,
}

/// A compiled pad traversal.
///
/// Replay fills every destination element with the caller-provided fill
/// scalar, then copies reachable input positions into the padded output.
/// A dense column-major destination with a unit-stride axis-0 operand and
/// non-negative edges without interior spacing instead writes disjoint spans
/// (low pad, operand run, high pad) per column so each element is written once.
#[derive(Clone, Debug)]
pub struct PadPlan {
    operand_dims: AxisVec<usize>,
    operand_strides: AxisVec<isize>,
    dest_dims: AxisVec<usize>,
    dest_strides: AxisVec<isize>,
    edge_padding_low: AxisVec<i64>,
    interior_step: AxisVec<i64>,
    operand_total: usize,
    dest_total: usize,
    contiguous_dest_fill: bool,
    contiguous_axis0_run: Option<ContiguousPadAxis0Run>,
    contiguous_axis0_columns: Option<ContiguousPadAxis0Columns>,
    generic_fill: PadFillCursor,
    generic_copy: PadCopyCursor,
}

#[derive(Clone, Debug)]
struct PadFillCursor {
    steps: AxisVec<isize>,
    resets: AxisVec<isize>,
}

#[derive(Clone, Debug)]
struct PadCopyCursor {
    shape: AxisVec<usize>,
    source_base_delta: isize,
    dest_base_delta: isize,
    source_steps: AxisVec<isize>,
    source_resets: AxisVec<isize>,
    dest_steps: AxisVec<isize>,
    dest_resets: AxisVec<isize>,
    total: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ContiguousPadAxis0Run {
    operand_start: usize,
    dest_start: usize,
    len: usize,
}

/// Prepared single-pass write for a dense column-major destination whose
/// axis-0 operand stride is one and whose edges are non-negative with no
/// interior spacing on any axis.
///
/// Replay walks destination columns. Each column writes its low pad, copies
/// the operand axis-0 run, and writes its high pad; columns with no reachable
/// operand coordinate write the whole column as padding. Every reachable
/// destination slot is therefore written exactly once, and the write spans of
/// distinct columns are disjoint because the destination layout is injective.
#[derive(Clone, Debug)]
struct ContiguousPadAxis0Columns {
    /// Number of destination columns, `dest_dims[0]` elements each.
    column_total: usize,
    dest_axis0_len: usize,
    dest_axis0_step: isize,
    low0: usize,
    high0: usize,
    run_len: usize,
    operand_outer_dims: AxisVec<usize>,
    operand_outer_strides: AxisVec<isize>,
    dest_outer_dims: AxisVec<usize>,
    edge_low_outer: AxisVec<usize>,
}

/// A compiled multi-input concatenate traversal.
///
/// Each input segment is lowered to a prepared strided copy into the
/// corresponding destination window. The per-input collections are inline up
/// to [`OPERAND_CAP`](crate::inline::OPERAND_CAP) slots (destination plus four
/// inputs), so a bounded concatenate compiles and executes without touching the
/// heap; above that bound they spill gracefully. `segment_starts` carries one
/// more entry than there are inputs (the `input_count + 1` prefix-sum sentinel
/// whose last entry is the destination total), which is why its bound is the
/// same operand cap.
#[derive(Clone, Debug)]
pub struct ConcatenatePlan {
    input_dims: crate::inline::OperandVec<AxisVec<usize>>,
    input_strides: crate::inline::OperandVec<AxisVec<isize>>,
    dest_dims: AxisVec<usize>,
    dest_strides: AxisVec<isize>,
    dest_offset_deltas: crate::inline::OperandVec<isize>,
    /// Logical start of each segment in the concatenated index space
    /// (segment totals prefix sums); the last entry is the destination total.
    segment_starts: crate::inline::OperandVec<usize>,
    copy_plans: crate::inline::OperandVec<CopyPlan>,
}

impl SlicePlan {
    /// Compile a static slice plan for one operand layout and destination layout.
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        operand_dims: &[usize],
        operand_strides: &[isize],
        dest_dims: &[usize],
        dest_strides: &[isize],
        starts: &[usize],
        limits: &[usize],
        slice_strides: &[usize],
    ) -> Result<Self> {
        let rank = operand_dims.len();
        if operand_strides.len() != rank || dest_dims.len() != rank || dest_strides.len() != rank {
            return Err(StridedError::StrideLengthMismatch);
        }
        if starts.len() != rank {
            return Err(StridedError::RankMismatch(starts.len(), rank));
        }
        if limits.len() != rank {
            return Err(StridedError::RankMismatch(limits.len(), rank));
        }
        if slice_strides.len() != rank {
            return Err(StridedError::RankMismatch(slice_strides.len(), rank));
        }
        checked_total_len(operand_dims)?;
        checked_total_len(dest_dims)?;

        let mut expected_dest_dims: AxisVec<usize> = AxisVec::with_capacity(rank);
        let mut source_strides: AxisVec<isize> = AxisVec::with_capacity(rank);
        let mut source_offset_delta = 0isize;
        for axis in 0..rank {
            let start = starts[axis];
            let limit = limits[axis];
            let stride = slice_strides[axis];
            if start > limit || limit > operand_dims[axis] || stride == 0 {
                return Err(StridedError::InvalidAxis { axis, rank });
            }
            let span = limit - start;
            expected_dest_dims.push(span.div_ceil(stride));
            source_strides.push(checked_stride_mul(operand_strides[axis], stride)?);
            source_offset_delta =
                checked_offset_add(source_offset_delta, operand_strides[axis], start)?;
        }
        if dest_dims != &expected_dest_dims[..] {
            return Err(StridedError::ShapeMismatch(
                dest_dims.to_vec(),
                expected_dest_dims.to_vec(),
            ));
        }
        let copy_plan = CopyPlan::compile(dest_dims, dest_strides, &source_strides)?;

        Ok(Self {
            operand_dims: operand_dims.into(),
            operand_strides: operand_strides.into(),
            dest_dims: dest_dims.into(),
            dest_strides: dest_strides.into(),
            source_strides,
            source_offset_delta,
            copy_plan,
        })
    }

    /// Execute the prepared static slice traversal.
    pub fn execute<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        operand: &RawStridedRef<'_, T>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        self.check_call(dest, operand)?;
        let source_offset = operand
            .offset()
            .checked_add(self.source_offset_delta)
            .ok_or(StridedError::OffsetOverflow)?;
        let source = unsafe {
            RawStridedRef::new_unchecked(
                operand.data(),
                &self.dest_dims,
                &self.source_strides,
                source_offset,
            )
        };
        self.copy_plan.execute(dest, &source, exec)
    }

    /// Execute into storage whose reachable destination elements are uninitialized.
    pub fn execute_uninit<T>(
        &self,
        dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
        operand: &RawStridedRef<'_, T>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        self.check_call(dest, operand)?;
        let source_offset = operand
            .offset()
            .checked_add(self.source_offset_delta)
            .ok_or(StridedError::OffsetOverflow)?;
        let source = unsafe {
            RawStridedRef::new_unchecked(
                operand.data(),
                &self.dest_dims,
                &self.source_strides,
                source_offset,
            )
        };
        self.copy_plan.execute_uninit(dest, &source, exec)
    }

    fn check_call<D, T>(
        &self,
        dest: &RawStridedMut<'_, D>,
        operand: &RawStridedRef<'_, T>,
    ) -> Result<()> {
        if operand.dims() != &self.operand_dims[..]
            || operand.strides() != &self.operand_strides[..]
            || dest.dims() != &self.dest_dims[..]
            || dest.strides() != &self.dest_strides[..]
        {
            return Err(StridedError::PlanLayoutMismatch);
        }
        Ok(())
    }
}

impl PadPlan {
    /// Compile a pad plan for one operand layout and destination layout.
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        operand_dims: &[usize],
        operand_strides: &[isize],
        dest_dims: &[usize],
        dest_strides: &[isize],
        edge_padding_low: &[i64],
        edge_padding_high: &[i64],
        interior_padding: &[i64],
    ) -> Result<Self> {
        let rank = operand_dims.len();
        if operand_strides.len() != rank || dest_dims.len() != rank || dest_strides.len() != rank {
            return Err(StridedError::StrideLengthMismatch);
        }
        if edge_padding_low.len() != rank {
            return Err(StridedError::RankMismatch(edge_padding_low.len(), rank));
        }
        if edge_padding_high.len() != rank {
            return Err(StridedError::RankMismatch(edge_padding_high.len(), rank));
        }
        if interior_padding.len() != rank {
            return Err(StridedError::RankMismatch(interior_padding.len(), rank));
        }

        let operand_total = checked_total_len(operand_dims)?;
        let dest_total = checked_total_len(dest_dims)?;
        if !crate::layout_check::is_injective_layout(dest_dims, dest_strides) {
            return Err(StridedError::NonInjectiveOutputLayout);
        }

        let mut expected_dest_dims: AxisVec<usize> = AxisVec::with_capacity(rank);
        let mut interior_step: AxisVec<i64> = AxisVec::with_capacity(rank);
        for axis in 0..rank {
            if interior_padding[axis] < 0 {
                return Err(StridedError::InvalidAxis { axis, rank });
            }
            let step = interior_padding[axis]
                .checked_add(1)
                .ok_or(StridedError::OffsetOverflow)?;
            interior_step.push(step);
            expected_dest_dims.push(checked_pad_output_dim(
                operand_dims[axis],
                edge_padding_low[axis],
                edge_padding_high[axis],
                step,
                axis,
                rank,
            )?);
        }
        if dest_dims != &expected_dest_dims[..] {
            return Err(StridedError::ShapeMismatch(
                dest_dims.to_vec(),
                expected_dest_dims.to_vec(),
            ));
        }
        let contiguous_dest_fill = is_dense_col_major(dest_dims, dest_strides);
        let contiguous_axis0_run = compile_contiguous_pad_axis0_run(
            operand_dims,
            operand_strides,
            dest_dims,
            dest_strides,
            edge_padding_low,
            &interior_step,
        );
        let contiguous_axis0_columns = contiguous_axis0_run.and_then(|run| {
            compile_contiguous_pad_axis0_columns(
                operand_dims,
                operand_strides,
                dest_dims,
                edge_padding_low,
                edge_padding_high,
                &interior_step,
                contiguous_dest_fill,
                run,
            )
        });
        let generic_fill = compile_pad_fill_cursor(dest_dims, dest_strides)?;
        let generic_copy = compile_pad_copy_cursor(
            operand_dims,
            operand_strides,
            dest_dims,
            dest_strides,
            edge_padding_low,
            &interior_step,
        )?;

        Ok(Self {
            operand_dims: operand_dims.into(),
            operand_strides: operand_strides.into(),
            dest_dims: dest_dims.into(),
            dest_strides: dest_strides.into(),
            edge_padding_low: edge_padding_low.into(),
            interior_step,
            operand_total,
            dest_total,
            contiguous_dest_fill,
            contiguous_axis0_run,
            contiguous_axis0_columns,
            generic_fill,
            generic_copy,
        })
    }

    /// Execute the prepared pad traversal.
    pub fn execute<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        operand: &RawStridedRef<'_, T>,
        fill: T,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        let mut exec = exec;
        self.check_call(dest, operand)?;
        if self.operand_total == 0 {
            return self.fill_dest(dest, fill, exec);
        }
        if let Some(columns) = &self.contiguous_axis0_columns {
            return self.execute_axis0_columns(dest, operand, fill, columns, exec);
        }
        self.fill_dest(dest, fill, exec.reborrow())?;
        if let Some(run) = self.contiguous_axis0_run {
            return self.copy_operand_axis0_runs(dest, operand, run, exec);
        }
        self.copy_operand(dest, operand, exec)
    }

    /// Execute pad into storage whose reachable destination elements are uninitialized.
    pub fn execute_uninit<T>(
        &self,
        dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
        operand: &RawStridedRef<'_, T>,
        fill: T,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        let mut exec = exec;
        self.check_call(dest, operand)?;
        if self.operand_total == 0 {
            return self.fill_dest(dest, MaybeUninit::new(fill), exec);
        }
        if let Some(columns) = &self.contiguous_axis0_columns {
            return self.execute_axis0_columns_uninit(dest, operand, fill, columns, exec);
        }
        self.fill_dest(dest, MaybeUninit::new(fill), exec.reborrow())?;
        if let Some(run) = self.contiguous_axis0_run {
            return self.copy_operand_axis0_runs_uninit(dest, operand, run, exec);
        }
        self.copy_operand_uninit(dest, operand, exec)
    }

    fn fill_dest<T>(&self, dest: &mut RawStridedMut<'_, T>, fill: T, exec: Exec<'_>) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        if self.dest_total == 0 {
            return Ok(());
        }
        if self.contiguous_dest_fill {
            let dest_offset =
                usize::try_from(dest.offset()).map_err(|_| StridedError::OffsetOverflow)?;
            let dest_end = dest_offset
                .checked_add(self.dest_total)
                .ok_or(StridedError::OffsetOverflow)?;
            let dest_data = dest.data_mut();
            let dest_slice = dest_data
                .get_mut(dest_offset..dest_end)
                .ok_or(StridedError::OffsetOverflow)?;
            crate::threading::fill_contiguous(dest_slice, fill, exec);
            return Ok(());
        }
        let exec = crate::threading::for_len(exec, self.dest_total);
        if !exec.is_serial() {
            return self.fill_dest_parallel(dest, fill, exec);
        }
        self.fill_dest_serial(dest, fill)
    }

    fn copy_operand_axis0_runs<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        operand: &RawStridedRef<'_, T>,
        run: ContiguousPadAxis0Run,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        let dest_offset = dest.offset();
        let dest_ptr = dest.data_mut().as_mut_ptr();
        // SAFETY: `check_call` proved `dest` carries the compiled destination
        // layout, so its validated data covers every reachable destination
        // slot, and the exclusive borrow is the only access during the call.
        unsafe { self.copy_axis0_runs_raw(dest_ptr, dest_offset, operand, run, exec) }
    }

    fn copy_operand_axis0_runs_uninit<T>(
        &self,
        dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
        operand: &RawStridedRef<'_, T>,
        run: ContiguousPadAxis0Run,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        let dest_offset = dest.offset();
        let dest_ptr = dest.data_mut().as_mut_ptr().cast::<T>();
        // SAFETY: as in `copy_operand_axis0_runs`; `MaybeUninit<T>` has the
        // layout of `T`, and only writes go through the cast pointer.
        unsafe { self.copy_axis0_runs_raw(dest_ptr, dest_offset, operand, run, exec) }
    }

    fn execute_axis0_columns<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        operand: &RawStridedRef<'_, T>,
        fill: T,
        columns: &ContiguousPadAxis0Columns,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        let dest_offset = dest.offset();
        let dest_ptr = dest.data_mut().as_mut_ptr();
        let operand_offset = operand.offset();
        let operand_ptr = operand.data().as_ptr();
        // SAFETY: `check_call` proved `dest` carries the compiled destination
        // layout, so its validated data covers every reachable destination
        // slot, and the exclusive borrow is the only access during the call.
        unsafe {
            self.write_axis0_columns_raw::<T, false>(
                dest_ptr,
                dest_offset,
                operand_ptr,
                operand_offset,
                fill,
                columns,
                exec,
            )
        }
    }

    fn execute_axis0_columns_uninit<T>(
        &self,
        dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
        operand: &RawStridedRef<'_, T>,
        fill: T,
        columns: &ContiguousPadAxis0Columns,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        let dest_offset = dest.offset();
        let dest_ptr = dest.data_mut().as_mut_ptr().cast::<T>();
        let operand_offset = operand.offset();
        let operand_ptr = operand.data().as_ptr();
        // SAFETY: as in `execute_axis0_columns`; `MaybeUninit<T>` has the
        // layout of `T`, and only writes go through the cast pointer.
        unsafe {
            self.write_axis0_columns_raw::<T, true>(
                dest_ptr,
                dest_offset,
                operand_ptr,
                operand_offset,
                fill,
                columns,
                exec,
            )
        }
    }

    /// Write every destination column as disjoint low-pad, operand-run, and
    /// high-pad spans.
    ///
    /// With at least as many columns as workers (above the repository
    /// threshold) the columns are split across workers; otherwise columns run
    /// in order and each long span is chunked across workers by
    /// [`crate::threading::copy_contiguous`] and
    /// [`crate::threading::fill_contiguous`].
    ///
    /// # Safety
    ///
    /// `dest_ptr` and `operand_ptr` must be the data pointers of descriptors
    /// whose layouts are exactly the compiled layouts from `dest_offset` and
    /// `operand_offset`. The destination must be writable at every reachable
    /// slot and no other access may overlap the call; the operand may only be
    /// read at coordinates that are copied, but no read is performed for a
    /// padding-only column.
    #[allow(clippy::too_many_arguments)]
    unsafe fn write_axis0_columns_raw<T, const UNINIT: bool>(
        &self,
        dest_ptr: *mut T,
        dest_offset: isize,
        operand_ptr: *const T,
        operand_offset: isize,
        fill: T,
        columns: &ContiguousPadAxis0Columns,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        let column_total = columns.column_total;
        if column_total == 0 {
            return Ok(());
        }
        let exec = crate::threading::for_len(exec, self.dest_total);
        {
            if exec.budget().get() > 1 && column_total >= exec.budget().get() {
                let dest_ptr = crate::threading::SendPtr(dest_ptr);
                let operand_ptr = crate::threading::SendPtr(operand_ptr as *mut T);
                return crate::threading::parallel_map_reduce(
                    0..column_total,
                    exec,
                    &|range| {
                        let mut outer_idx_storage =
                            CoordScratch::new(columns.dest_outer_dims.len());
                        let outer_idx = outer_idx_storage.as_mut_slice();
                        fill_col_major_index(range.start, &columns.dest_outer_dims, outer_idx);
                        // SAFETY: worker ranges are disjoint; the destination
                        // is dense column-major, so distinct columns occupy
                        // disjoint contiguous slots, and the caller's contract
                        // covers every reachable slot. Padding-only columns
                        // never read the operand.
                        unsafe {
                            self.write_axis0_columns_range::<T, UNINIT>(
                                dest_ptr.as_ptr(),
                                dest_offset,
                                operand_ptr.as_const(),
                                operand_offset,
                                fill,
                                columns,
                                outer_idx,
                                range.start,
                                range.len(),
                                Exec::serial(),
                            )
                        }
                    },
                    &|left, right| left.and(right),
                );
            }
        }
        let mut outer_idx_storage = CoordScratch::new(columns.dest_outer_dims.len());
        let outer_idx = outer_idx_storage.as_mut_slice();
        // SAFETY: forwarded from this function's contract.
        unsafe {
            self.write_axis0_columns_range::<T, UNINIT>(
                dest_ptr,
                dest_offset,
                operand_ptr,
                operand_offset,
                fill,
                columns,
                outer_idx,
                0,
                column_total,
                exec,
            )
        }
    }

    /// Write `count` destination columns starting at linear column
    /// `column_start`, advancing `outer_idx` in place.
    ///
    /// # Safety
    ///
    /// As for [`Self::write_axis0_columns_raw`], and no other thread may write
    /// the destination columns visited by this call.
    #[allow(clippy::too_many_arguments)]
    unsafe fn write_axis0_columns_range<T, const UNINIT: bool>(
        &self,
        dest_ptr: *mut T,
        dest_offset: isize,
        operand_ptr: *const T,
        operand_offset: isize,
        fill: T,
        columns: &ContiguousPadAxis0Columns,
        outer_idx: &mut [usize],
        column_start: usize,
        count: usize,
        mut exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        if count == 0 {
            return Ok(());
        }
        let mut dest_column_offset =
            checked_offset_add(dest_offset, columns.dest_axis0_step, column_start)?;
        // Carry the operand column offset and in-bounds count through the outer
        // odometer instead of rescanning every outer axis per column. Only axes
        // currently in bounds contribute a `stride * q` term, so every product
        // is one the validated operand layout already produced.
        let mut operand_column_offset = operand_offset;
        let mut out_of_bounds = 0usize;
        for (axis, &coord) in outer_idx.iter().enumerate() {
            match in_bounds_outer_coord(columns, axis, coord) {
                Some(q) => {
                    operand_column_offset = checked_offset_add(
                        operand_column_offset,
                        columns.operand_outer_strides[axis],
                        q,
                    )?;
                }
                None => out_of_bounds += 1,
            }
        }
        for _ in 0..count {
            if out_of_bounds == 0 {
                let low_len = columns.low0;
                let run_len = columns.run_len;
                let high_len = columns.high0;
                // SAFETY: the compiled run span and the dense column layout
                // place the low pad, operand run, and high pad at disjoint
                // offsets inside this column; the operand run was validated by
                // the descriptor bounds and `run_len` copies only in-bounds
                // axis-0 elements. The destination layout is injective and
                // this call owns the whole column.
                unsafe {
                    if UNINIT {
                        fill_raw_span_uninit(
                            dest_ptr.offset(dest_column_offset),
                            low_len,
                            fill,
                            exec.reborrow(),
                        );
                    } else {
                        fill_raw_span(
                            dest_ptr.offset(dest_column_offset),
                            low_len,
                            fill,
                            exec.reborrow(),
                        );
                    }
                    crate::threading::copy_contiguous(
                        operand_ptr.offset(operand_column_offset),
                        dest_ptr.offset(dest_column_offset + low_len as isize),
                        run_len,
                        exec.reborrow(),
                    );
                    if UNINIT {
                        fill_raw_span_uninit(
                            dest_ptr
                                .offset(dest_column_offset + low_len as isize + run_len as isize),
                            high_len,
                            fill,
                            exec.reborrow(),
                        );
                    } else {
                        fill_raw_span(
                            dest_ptr
                                .offset(dest_column_offset + low_len as isize + run_len as isize),
                            high_len,
                            fill,
                            exec.reborrow(),
                        );
                    }
                }
            } else {
                // SAFETY: as for the in-bounds branch; a padding-only column
                // never reads the operand.
                unsafe {
                    if UNINIT {
                        fill_raw_span_uninit(
                            dest_ptr.offset(dest_column_offset),
                            columns.dest_axis0_len,
                            fill,
                            exec.reborrow(),
                        );
                    } else {
                        fill_raw_span(
                            dest_ptr.offset(dest_column_offset),
                            columns.dest_axis0_len,
                            fill,
                            exec.reborrow(),
                        );
                    }
                }
            }
            // Advance the destination column, the outer odometer, and the
            // incremental operand offset / in-bounds count together.
            dest_column_offset = dest_column_offset
                .checked_add(columns.dest_axis0_step)
                .ok_or(StridedError::OffsetOverflow)?;
            for (axis, coord) in outer_idx.iter_mut().enumerate() {
                let was_in_bounds = in_bounds_outer_coord(columns, axis, *coord);
                if let Some(q) = was_in_bounds {
                    let q = isize::try_from(q).map_err(|_| StridedError::OffsetOverflow)?;
                    let scaled = columns.operand_outer_strides[axis]
                        .checked_mul(q)
                        .ok_or(StridedError::OffsetOverflow)?;
                    operand_column_offset = operand_column_offset
                        .checked_sub(scaled)
                        .ok_or(StridedError::OffsetOverflow)?;
                }
                *coord += 1;
                let wrapped = *coord >= columns.dest_outer_dims[axis];
                if wrapped {
                    *coord = 0;
                }
                let is_in_bounds = in_bounds_outer_coord(columns, axis, *coord);
                if let Some(q) = is_in_bounds {
                    operand_column_offset = checked_offset_add(
                        operand_column_offset,
                        columns.operand_outer_strides[axis],
                        q,
                    )?;
                }
                match (was_in_bounds.is_some(), is_in_bounds.is_some()) {
                    (false, true) => out_of_bounds -= 1,
                    (true, false) => out_of_bounds += 1,
                    _ => {}
                }
                if !wrapped {
                    break;
                }
            }
        }
        Ok(())
    }

    /// Copy every in-bounds axis-0 run of the operand into the destination.
    ///
    /// With at least as many runs as workers (above the repository
    /// threshold) the runs are split across workers; otherwise runs are
    /// copied in order and each long run is chunked across workers by
    /// [`crate::threading::copy_contiguous`].
    ///
    /// # Safety
    ///
    /// `dest_ptr` must be the data pointer of a destination whose layout is
    /// exactly the compiled destination layout from `dest_offset`, valid for
    /// writes to every reachable slot, with no concurrent access.
    unsafe fn copy_axis0_runs_raw<T>(
        &self,
        dest_ptr: *mut T,
        dest_offset: isize,
        operand: &RawStridedRef<'_, T>,
        run: ContiguousPadAxis0Run,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        if run.len == 0 {
            return Ok(());
        }
        let outer_dims = &self.operand_dims[1..];
        let outer_total = checked_total_len(outer_dims)?;
        let copied = outer_total.saturating_mul(run.len);
        let exec = crate::threading::for_len(exec, copied);
        {
            if exec.budget().get() > 1 && outer_total >= exec.budget().get() {
                let dest_ptr = crate::threading::SendPtr(dest_ptr);
                return crate::threading::parallel_map_reduce(
                    0..outer_total,
                    exec,
                    &|range| {
                        let mut outer_idx_storage = CoordScratch::new(outer_dims.len());
                        let outer_idx = outer_idx_storage.as_mut_slice();
                        fill_col_major_index(range.start, outer_dims, outer_idx);
                        // SAFETY: the caller's contract covers every run, and
                        // disjoint outer ranges write disjoint runs because the
                        // compiled destination layout is injective.
                        unsafe {
                            self.copy_axis0_run_range(
                                dest_ptr.as_ptr(),
                                dest_offset,
                                operand,
                                run,
                                outer_idx,
                                range.len(),
                                Exec::serial(),
                            )
                        }
                    },
                    &|left, right| left.and(right),
                );
            }
        }
        let mut outer_idx_storage = CoordScratch::new(outer_dims.len());
        let outer_idx = outer_idx_storage.as_mut_slice();
        // SAFETY: forwarded from this function's contract.
        unsafe {
            self.copy_axis0_run_range(
                dest_ptr,
                dest_offset,
                operand,
                run,
                outer_idx,
                outer_total,
                exec,
            )
        }
    }

    /// Copy `count` consecutive axis-0 runs starting at outer coordinate
    /// `outer_idx` (column-major over axes `1..`), advancing it in place.
    ///
    /// # Safety
    ///
    /// As for [`Self::copy_axis0_runs_raw`], and no other thread may write the
    /// destination runs visited by this call.
    #[allow(clippy::too_many_arguments)]
    unsafe fn copy_axis0_run_range<T>(
        &self,
        dest_ptr: *mut T,
        dest_offset: isize,
        operand: &RawStridedRef<'_, T>,
        run: ContiguousPadAxis0Run,
        outer_idx: &mut [usize],
        count: usize,
        mut exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        let outer_dims = &self.operand_dims[1..];
        let operand_ptr = operand.data().as_ptr();
        for _ in 0..count {
            let mut operand_offset =
                checked_offset_add(operand.offset(), self.operand_strides[0], run.operand_start)?;
            let mut dest_run_offset =
                checked_offset_add(dest_offset, self.dest_strides[0], run.dest_start)?;
            let mut in_bounds = true;
            for (outer_axis, &coord) in outer_idx.iter().enumerate() {
                let axis = outer_axis + 1;
                let out_pos = i128::from(self.edge_padding_low[axis])
                    + coord as i128 * i128::from(self.interior_step[axis]);
                if out_pos < 0 || out_pos >= self.dest_dims[axis] as i128 {
                    in_bounds = false;
                    break;
                }
                operand_offset =
                    checked_offset_add(operand_offset, self.operand_strides[axis], coord)?;
                dest_run_offset =
                    checked_offset_add(dest_run_offset, self.dest_strides[axis], out_pos as usize)?;
            }
            if in_bounds {
                // SAFETY: compile requires unit axis-0 strides, the clipped
                // run is in bounds of the validated operand and destination,
                // and the destination layout is injective. RawStridedMut's
                // exclusive borrow cannot overlap the RawStridedRef's shared
                // borrow in safe Rust.
                unsafe {
                    crate::threading::copy_contiguous(
                        operand_ptr.offset(operand_offset),
                        dest_ptr.offset(dest_run_offset),
                        run.len,
                        exec.reborrow(),
                    );
                }
            }
            advance_col_major_index(outer_idx, outer_dims);
        }
        Ok(())
    }

    #[cfg(test)]
    fn contiguous_axis0_run(&self) -> Option<(usize, usize, usize)> {
        self.contiguous_axis0_run
            .map(|run| (run.operand_start, run.dest_start, run.len))
    }

    #[cfg(test)]
    fn has_contiguous_dest_fill(&self) -> bool {
        self.contiguous_dest_fill
    }

    #[cfg(test)]
    fn has_contiguous_axis0_columns(&self) -> bool {
        self.contiguous_axis0_columns.is_some()
    }

    fn fill_dest_serial<T>(&self, dest: &mut RawStridedMut<'_, T>, fill: T) -> Result<()>
    where
        T: Copy,
    {
        let dest_ptr = dest.data_mut().as_mut_ptr();
        let mut cursor = PadFillState::new(dest.offset(), &self.dest_dims, &self.generic_fill);
        for _ in 0..self.dest_total {
            unsafe {
                // INVARIANT: compile-time checked fill spans, validated raw
                // descriptors, and exact plan-layout equality prove this is a
                // reachable destination offset.
                *dest_ptr.offset(cursor.offset) = fill;
            }
            cursor.advance(&self.dest_dims, &self.generic_fill);
        }
        Ok(())
    }

    fn fill_dest_parallel<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        fill: T,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        let dest_offset_base = dest.offset();
        let dest_ptr = crate::threading::SendPtr(dest.data_mut().as_mut_ptr());
        crate::threading::parallel_map_reduce(
            0..self.dest_total,
            exec,
            &|range| {
                let mut cursor = PadFillState::decode(
                    range.start,
                    dest_offset_base,
                    &self.dest_dims,
                    &self.generic_fill,
                )?;
                let dest_ptr = dest_ptr.as_ptr();
                for _ in range {
                    unsafe {
                        // INVARIANT: compile-time checked fill spans, validated
                        // raw descriptors, exact layout equality, and disjoint
                        // positive logical partition mapping prove this offset
                        // is reachable and uniquely owned by this range.
                        *dest_ptr.offset(cursor.offset) = fill;
                    }
                    cursor.advance(&self.dest_dims, &self.generic_fill);
                }
                Ok(())
            },
            &|left, right| left.and(right),
        )
    }

    fn copy_operand<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        operand: &RawStridedRef<'_, T>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        let exec = crate::threading::for_len(exec, self.generic_copy.total);
        if !exec.is_serial() {
            return self.copy_operand_parallel(dest, operand, exec);
        }
        self.copy_operand_serial(dest, operand)
    }

    fn copy_operand_uninit<T>(
        &self,
        dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
        operand: &RawStridedRef<'_, T>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        let exec = crate::threading::for_len(exec, self.generic_copy.total);
        if !exec.is_serial() {
            return self.copy_operand_uninit_parallel(dest, operand, exec);
        }
        self.copy_operand_uninit_serial(dest, operand)
    }

    fn copy_operand_uninit_serial<T>(
        &self,
        dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
        operand: &RawStridedRef<'_, T>,
    ) -> Result<()>
    where
        T: Copy,
    {
        if self.generic_copy.total == 0 {
            return Ok(());
        }
        let operand_ptr = operand.data().as_ptr();
        let dest_ptr = dest.data_mut().as_mut_ptr();
        let source_base = operand
            .offset()
            .checked_add(self.generic_copy.source_base_delta)
            .ok_or(StridedError::OffsetOverflow)?;
        let dest_base = dest
            .offset()
            .checked_add(self.generic_copy.dest_base_delta)
            .ok_or(StridedError::OffsetOverflow)?;
        let mut cursor = PadCopyState::new(source_base, dest_base, &self.generic_copy);
        for _ in 0..self.generic_copy.total {
            unsafe {
                // INVARIANT: compile-time checked copy spans/base deltas,
                // validated raw descriptors, and exact plan-layout equality
                // prove both offsets reachable; positive steps make destinations
                // injective, so no copy aliases another cursor position.
                (*dest_ptr.offset(cursor.dest_offset))
                    .write(*operand_ptr.offset(cursor.source_offset));
            }
            cursor.advance(&self.generic_copy);
        }
        Ok(())
    }

    fn copy_operand_uninit_parallel<T>(
        &self,
        dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
        operand: &RawStridedRef<'_, T>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        let operand_base = operand
            .offset()
            .checked_add(self.generic_copy.source_base_delta)
            .ok_or(StridedError::OffsetOverflow)?;
        let operand_ptr = crate::threading::SendPtr(operand.data().as_ptr() as *mut T);
        let dest_base = dest
            .offset()
            .checked_add(self.generic_copy.dest_base_delta)
            .ok_or(StridedError::OffsetOverflow)?;
        let dest_ptr = crate::threading::SendPtr(dest.data_mut().as_mut_ptr());
        crate::threading::parallel_map_reduce(
            0..self.generic_copy.total,
            exec,
            &|range| {
                let mut cursor =
                    PadCopyState::decode(range.start, operand_base, dest_base, &self.generic_copy)?;
                let operand_ptr = operand_ptr.as_const();
                let dest_ptr = dest_ptr.as_ptr();

                for _ in range {
                    unsafe {
                        // INVARIANT: compile-time checked copy spans/base
                        // deltas, validated descriptors, exact layout equality,
                        // and positive injective mapping prove disjoint,
                        // reachable source/destination offsets per range.
                        (*dest_ptr.offset(cursor.dest_offset))
                            .write(*operand_ptr.offset(cursor.source_offset));
                    }
                    cursor.advance(&self.generic_copy);
                }
                Ok(())
            },
            &|left, right| left.and(right),
        )
    }

    fn copy_operand_serial<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        operand: &RawStridedRef<'_, T>,
    ) -> Result<()>
    where
        T: Copy,
    {
        if self.generic_copy.total == 0 {
            return Ok(());
        }
        let operand_ptr = operand.data().as_ptr();
        let dest_ptr = dest.data_mut().as_mut_ptr();
        let source_base = operand
            .offset()
            .checked_add(self.generic_copy.source_base_delta)
            .ok_or(StridedError::OffsetOverflow)?;
        let dest_base = dest
            .offset()
            .checked_add(self.generic_copy.dest_base_delta)
            .ok_or(StridedError::OffsetOverflow)?;
        let mut cursor = PadCopyState::new(source_base, dest_base, &self.generic_copy);
        for _ in 0..self.generic_copy.total {
            unsafe {
                // INVARIANT: compile-time checked copy spans/base deltas,
                // validated raw descriptors, and exact plan-layout equality
                // prove both offsets reachable; positive steps make destinations
                // injective, so no copy aliases another cursor position.
                *dest_ptr.offset(cursor.dest_offset) = *operand_ptr.offset(cursor.source_offset);
            }
            cursor.advance(&self.generic_copy);
        }
        Ok(())
    }

    fn copy_operand_parallel<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        operand: &RawStridedRef<'_, T>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        let operand_base = operand
            .offset()
            .checked_add(self.generic_copy.source_base_delta)
            .ok_or(StridedError::OffsetOverflow)?;
        let operand_ptr = crate::threading::SendPtr(operand.data().as_ptr() as *mut T);
        let dest_base = dest
            .offset()
            .checked_add(self.generic_copy.dest_base_delta)
            .ok_or(StridedError::OffsetOverflow)?;
        let dest_ptr = crate::threading::SendPtr(dest.data_mut().as_mut_ptr());
        crate::threading::parallel_map_reduce(
            0..self.generic_copy.total,
            exec,
            &|range| {
                let mut cursor =
                    PadCopyState::decode(range.start, operand_base, dest_base, &self.generic_copy)?;
                let operand_ptr = operand_ptr.as_const();
                let dest_ptr = dest_ptr.as_ptr();

                for _ in range {
                    unsafe {
                        // INVARIANT: compile-time checked copy spans/base
                        // deltas, validated descriptors, exact layout equality,
                        // and positive injective mapping prove disjoint,
                        // reachable source/destination offsets per range.
                        *dest_ptr.offset(cursor.dest_offset) =
                            *operand_ptr.offset(cursor.source_offset);
                    }
                    cursor.advance(&self.generic_copy);
                }
                Ok(())
            },
            &|left, right| left.and(right),
        )
    }

    fn check_call<D, T>(
        &self,
        dest: &RawStridedMut<'_, D>,
        operand: &RawStridedRef<'_, T>,
    ) -> Result<()> {
        if operand.dims() != &self.operand_dims[..]
            || operand.strides() != &self.operand_strides[..]
            || dest.dims() != &self.dest_dims[..]
            || dest.strides() != &self.dest_strides[..]
        {
            return Err(StridedError::PlanLayoutMismatch);
        }
        Ok(())
    }
}

impl ConcatenatePlan {
    /// Compile a multi-input concatenate plan for fixed input and destination layouts.
    pub fn compile(
        input_dims: &[&[usize]],
        input_strides: &[&[isize]],
        dest_dims: &[usize],
        dest_strides: &[isize],
        axis: usize,
    ) -> Result<Self> {
        if input_dims.is_empty() {
            return Err(StridedError::UnsupportedArity {
                arity: 0,
                max: usize::MAX,
            });
        }
        if input_dims.len() != input_strides.len() {
            return Err(StridedError::RankMismatch(
                input_strides.len(),
                input_dims.len(),
            ));
        }

        let rank = input_dims[0].len();
        if dest_dims.len() != rank || dest_strides.len() != rank {
            return Err(StridedError::StrideLengthMismatch);
        }
        if axis >= rank {
            return Err(StridedError::InvalidAxis { axis, rank });
        }
        checked_total_len(dest_dims)?;
        if !crate::layout_check::is_injective_layout(dest_dims, dest_strides) {
            return Err(StridedError::NonInjectiveOutputLayout);
        }

        let mut expected_dest_dims: AxisVec<usize> = input_dims[0].into();
        expected_dest_dims[axis] = 0;
        let mut stored_input_dims = crate::inline::OperandVec::with_capacity(input_dims.len());
        let mut stored_input_strides = crate::inline::OperandVec::with_capacity(input_dims.len());
        let mut dest_offset_deltas = crate::inline::OperandVec::with_capacity(input_dims.len());
        let mut copy_plans = crate::inline::OperandVec::with_capacity(input_dims.len());
        let mut segment_starts = crate::inline::OperandVec::with_capacity(input_dims.len() + 1);
        segment_starts.push(0usize);
        let mut axis_base = 0usize;

        for (dims, strides) in input_dims.iter().zip(input_strides.iter()) {
            if dims.len() != rank {
                return Err(StridedError::RankMismatch(dims.len(), rank));
            }
            if strides.len() != rank {
                return Err(StridedError::StrideLengthMismatch);
            }
            let segment_total = checked_total_len(dims)?;
            let segment_end = segment_starts[segment_starts.len() - 1]
                .checked_add(segment_total)
                .ok_or(StridedError::OffsetOverflow)?;
            segment_starts.push(segment_end);
            for dim in 0..rank {
                if dim == axis {
                    expected_dest_dims[axis] = expected_dest_dims[axis]
                        .checked_add(dims[axis])
                        .ok_or(StridedError::OffsetOverflow)?;
                } else if dims[dim] != input_dims[0][dim] {
                    return Err(StridedError::ShapeMismatch(
                        dims.to_vec(),
                        input_dims[0].to_vec(),
                    ));
                }
            }
            dest_offset_deltas.push(checked_offset_add(0, dest_strides[axis], axis_base)?);
            axis_base = axis_base
                .checked_add(dims[axis])
                .ok_or(StridedError::OffsetOverflow)?;
            copy_plans.push(CopyPlan::compile(dims, dest_strides, strides)?);
            stored_input_dims.push((*dims).into());
            stored_input_strides.push((*strides).into());
        }

        if dest_dims != &expected_dest_dims[..] {
            return Err(StridedError::ShapeMismatch(
                dest_dims.to_vec(),
                expected_dest_dims.to_vec(),
            ));
        }

        Ok(Self {
            input_dims: stored_input_dims,
            input_strides: stored_input_strides,
            dest_dims: dest_dims.into(),
            dest_strides: dest_strides.into(),
            dest_offset_deltas,
            segment_starts,
            copy_plans,
        })
    }

    /// Execute the prepared concatenate traversal.
    pub fn execute<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        inputs: &[RawStridedRef<'_, T>],
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        let mut exec = exec;
        self.check_dest_layout(dest)?;
        if inputs.len() != self.input_dims.len() {
            return Err(StridedError::RankMismatch(
                inputs.len(),
                self.input_dims.len(),
            ));
        }
        for (position, input) in inputs.iter().enumerate() {
            self.check_input_layout(position, input)?;
            self.segment_offset(position, dest.offset())?;
        }
        if self.try_execute_parallel(
            dest,
            inputs,
            |dst: &mut T, value| *dst = value,
            exec.reborrow(),
        )? {
            return Ok(());
        }
        for (position, input) in inputs.iter().enumerate() {
            self.execute_segment(position, dest, input, exec.reborrow())?;
        }
        Ok(())
    }

    /// Execute concatenate into storage whose reachable destination elements are uninitialized.
    pub fn execute_uninit<T>(
        &self,
        dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
        inputs: &[RawStridedRef<'_, T>],
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        let mut exec = exec;
        self.check_dest_layout(dest)?;
        if inputs.len() != self.input_dims.len() {
            return Err(StridedError::RankMismatch(
                inputs.len(),
                self.input_dims.len(),
            ));
        }
        for (position, input) in inputs.iter().enumerate() {
            self.check_input_layout(position, input)?;
            self.segment_offset(position, dest.offset())?;
        }
        if self.try_execute_parallel(
            dest,
            inputs,
            |dst: &mut MaybeUninit<T>, value| {
                dst.write(value);
            },
            exec.reborrow(),
        )? {
            return Ok(());
        }
        for (position, input) in inputs.iter().enumerate() {
            self.execute_segment_uninit(position, dest, input, exec.reborrow())?;
        }
        Ok(())
    }

    /// Replay all segments as one logical index space split across workers.
    ///
    /// Splitting the concatenated space (instead of fanning out per segment)
    /// keeps many small segments parallel as a group and splits a large
    /// segment across workers, with one fork-join for the whole call. Returns
    /// `Ok(false)` without writing when the serial per-segment path must run:
    /// the destination is at most the repository threshold, the budget
    /// allows one thread, or a segment exceeds the fused-rank limit.
    ///
    /// The caller must have validated every input layout and segment offset.
    fn try_execute_parallel<D, T, Apply>(
        &self,
        dest: &mut RawStridedMut<'_, D>,
        inputs: &[RawStridedRef<'_, T>],
        apply: Apply,
        exec: Exec<'_>,
    ) -> Result<bool>
    where
        D: Copy + Send + Sync,
        T: Copy + Send + Sync,
        Apply: Fn(&mut D, T) + Send + Sync,
    {
        let total = self.segment_starts[self.segment_starts.len() - 1];
        let exec = crate::threading::for_len(exec, total);
        if exec.is_serial() {
            return Ok(false);
        }
        // Precompute each segment's destination base offset once into an
        // inline operand vector (allocation-free within the operand bound);
        // the per-segment descriptors are read straight from `self`/`inputs`
        // inside the loop, so the parallel replay allocates nothing.
        let mut dest_offsets: crate::inline::OperandVec<isize> =
            crate::inline::OperandVec::with_capacity(inputs.len());
        for position in 0..inputs.len() {
            if self.copy_plans[position].fused_layout().is_none() {
                return Ok(false);
            }
            dest_offsets.push(self.segment_offset(position, dest.offset())?);
        }
        let dest_ptr = crate::threading::SendPtr(dest.data_mut().as_mut_ptr());
        let starts = &self.segment_starts;
        let dest_offsets = &dest_offsets;
        crate::threading::parallel_for_each(0..total, exec, &|range| {
            // First segment whose end lies past the range start.
            let mut position = starts[1..].partition_point(|&end| end <= range.start);
            while position < inputs.len() && starts[position] < range.end {
                let local_start = range.start.max(starts[position]) - starts[position];
                let local_end = range.end.min(starts[position + 1]) - starts[position];
                // The whole-plan guard above proved every segment has an
                // inline fused layout; skip one only if a future caller
                // relaxes that check.
                let Some(layout) = self.copy_plans[position].fused_layout() else {
                    position += 1;
                    continue;
                };
                let input = &inputs[position];
                // SAFETY: `check_dest_layout`/`check_input_layout` proved the
                // views carry the compiled layouts, so each segment window
                // (input dims, destination strides, checked segment offset) is
                // a sub-layout of the validated destination and each input
                // layout is in-bounds of its data. The destination layout is
                // injective and the segments occupy disjoint ranges of the
                // concatenation axis, so disjoint logical ranges write
                // disjoint slots; inputs are shared borrows that cannot alias
                // the exclusive destination borrow.
                unsafe {
                    crate::raw_ops::apply_fused_range(
                        dest_ptr.as_ptr(),
                        dest_offsets[position],
                        input.data().as_ptr(),
                        input.offset(),
                        layout,
                        local_start,
                        local_end - local_start,
                        &apply,
                        &|value| value,
                    );
                }
                position += 1;
            }
        });
        Ok(true)
    }

    pub(crate) fn check_dest_layout<T>(&self, dest: &RawStridedMut<'_, T>) -> Result<()> {
        if dest.dims() != &self.dest_dims[..] || dest.strides() != &self.dest_strides[..] {
            return Err(StridedError::PlanLayoutMismatch);
        }
        Ok(())
    }

    pub(crate) fn check_input_layout<T>(
        &self,
        position: usize,
        input: &RawStridedRef<'_, T>,
    ) -> Result<()> {
        if position >= self.input_dims.len()
            || input.dims() != &self.input_dims[position][..]
            || input.strides() != &self.input_strides[position][..]
        {
            return Err(StridedError::PlanLayoutMismatch);
        }
        Ok(())
    }

    pub(crate) fn input_count(&self) -> usize {
        self.input_dims.len()
    }

    /// Whether the whole-plan entry would replay in parallel within the
    /// budget of `exec`. Erased callers use it to keep the allocation-free
    /// per-segment loop for serial execution.
    pub(crate) fn prefers_whole_plan(&self, exec: &Exec<'_>) -> bool {
        let total = self.segment_starts[self.segment_starts.len() - 1];
        crate::threading::is_parallel(total, exec)
    }

    pub(crate) fn segment_offset(&self, position: usize, dest_offset: isize) -> Result<isize> {
        dest_offset
            .checked_add(self.dest_offset_deltas[position])
            .ok_or(StridedError::OffsetOverflow)
    }

    pub(crate) fn execute_segment<T>(
        &self,
        position: usize,
        dest: &mut RawStridedMut<'_, T>,
        input: &RawStridedRef<'_, T>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        let segment_offset = self.segment_offset(position, dest.offset())?;
        let dest_data = dest.data_mut();
        let mut segment = unsafe {
            RawStridedMut::new_unchecked(
                dest_data,
                &self.input_dims[position],
                &self.dest_strides,
                segment_offset,
            )
        };
        self.copy_plans[position].execute(&mut segment, input, exec)
    }

    pub(crate) fn execute_segment_uninit<T>(
        &self,
        position: usize,
        dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
        input: &RawStridedRef<'_, T>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        let segment_offset = self.segment_offset(position, dest.offset())?;
        let dest_data = dest.data_mut();
        let mut segment = unsafe {
            RawStridedMut::new_unchecked(
                dest_data,
                &self.input_dims[position],
                &self.dest_strides,
                segment_offset,
            )
        };
        self.copy_plans[position].execute_uninit(&mut segment, input, exec)
    }
}

impl ReversePlan {
    /// Compile a reverse plan for one operand layout, destination layout, and axis set.
    pub fn compile(
        operand_dims: &[usize],
        operand_strides: &[isize],
        dest_strides: &[isize],
        axes: &[usize],
    ) -> Result<Self> {
        let rank = operand_dims.len();
        if operand_strides.len() != rank || dest_strides.len() != rank {
            return Err(StridedError::StrideLengthMismatch);
        }
        checked_total_len(operand_dims)?;

        let mut reverse_axis: AxisVec<bool> = (0..rank).map(|_| false).collect();
        for &axis in axes {
            if axis >= rank {
                return Err(StridedError::InvalidAxis { axis, rank });
            }
            reverse_axis[axis] = true;
        }

        let mut source_strides: AxisVec<isize> = AxisVec::with_capacity(rank);
        let mut source_offset_delta = 0isize;
        for axis in 0..rank {
            if reverse_axis[axis] {
                source_strides.push(
                    operand_strides[axis]
                        .checked_neg()
                        .ok_or(StridedError::OffsetOverflow)?,
                );
                if operand_dims[axis] > 0 {
                    source_offset_delta = checked_offset_add(
                        source_offset_delta,
                        operand_strides[axis],
                        operand_dims[axis] - 1,
                    )?;
                }
            } else {
                source_strides.push(operand_strides[axis]);
            }
        }
        let copy_plan = CopyPlan::compile(operand_dims, dest_strides, &source_strides)?;

        Ok(Self {
            operand_dims: operand_dims.into(),
            operand_strides: operand_strides.into(),
            dest_strides: dest_strides.into(),
            source_strides,
            source_offset_delta,
            copy_plan,
        })
    }

    /// Execute the prepared reverse traversal.
    pub fn execute<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        operand: &RawStridedRef<'_, T>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        self.check_call(dest, operand)?;
        let source_offset = operand
            .offset()
            .checked_add(self.source_offset_delta)
            .ok_or(StridedError::OffsetOverflow)?;
        let source = unsafe {
            RawStridedRef::new_unchecked(
                operand.data(),
                &self.operand_dims,
                &self.source_strides,
                source_offset,
            )
        };
        self.copy_plan.execute(dest, &source, exec)
    }

    /// Execute reverse into storage whose reachable destination elements are uninitialized.
    pub fn execute_uninit<T>(
        &self,
        dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
        operand: &RawStridedRef<'_, T>,
        exec: Exec<'_>,
    ) -> Result<()>
    where
        T: Copy + Send + Sync,
    {
        self.check_call(dest, operand)?;
        let source_offset = operand
            .offset()
            .checked_add(self.source_offset_delta)
            .ok_or(StridedError::OffsetOverflow)?;
        let source = unsafe {
            RawStridedRef::new_unchecked(
                operand.data(),
                &self.operand_dims,
                &self.source_strides,
                source_offset,
            )
        };
        self.copy_plan.execute_uninit(dest, &source, exec)
    }

    fn check_call<D, T>(
        &self,
        dest: &RawStridedMut<'_, D>,
        operand: &RawStridedRef<'_, T>,
    ) -> Result<()> {
        if operand.dims() != &self.operand_dims[..]
            || operand.strides() != &self.operand_strides[..]
            || dest.dims() != &self.operand_dims[..]
            || dest.strides() != &self.dest_strides[..]
        {
            return Err(StridedError::PlanLayoutMismatch);
        }
        Ok(())
    }
}

fn checked_total_len(dims: &[usize]) -> Result<usize> {
    if dims.is_empty() {
        return Ok(1);
    }
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or(StridedError::OffsetOverflow)
}

fn checked_stride_mul(stride: isize, factor: usize) -> Result<isize> {
    let factor = isize::try_from(factor).map_err(|_| StridedError::OffsetOverflow)?;
    stride
        .checked_mul(factor)
        .ok_or(StridedError::OffsetOverflow)
}

fn checked_pad_output_dim(
    input_extent: usize,
    edge_low: i64,
    edge_high: i64,
    interior_step: i64,
    axis: usize,
    rank: usize,
) -> Result<usize> {
    let base = if input_extent == 0 {
        0i128
    } else {
        (input_extent as i128 - 1)
            .checked_mul(i128::from(interior_step))
            .and_then(|value| value.checked_add(1))
            .ok_or(StridedError::OffsetOverflow)?
    };
    let dim = i128::from(edge_low)
        .checked_add(i128::from(edge_high))
        .and_then(|value| value.checked_add(base))
        .ok_or(StridedError::OffsetOverflow)?;
    usize::try_from(dim).map_err(|_| StridedError::InvalidAxis { axis, rank })
}

fn compile_pad_fill_cursor(dims: &[usize], strides: &[isize]) -> Result<PadFillCursor> {
    let mut steps = AxisVec::with_capacity(dims.len());
    let mut resets = AxisVec::with_capacity(dims.len());
    for (&dim, &stride) in dims.iter().zip(strides) {
        steps.push(stride);
        resets.push(checked_cursor_reset(stride, dim)?);
    }
    check_offset_span(dims, &steps)?;
    Ok(PadFillCursor { steps, resets })
}

fn compile_pad_copy_cursor(
    operand_dims: &[usize],
    operand_strides: &[isize],
    dest_dims: &[usize],
    dest_strides: &[isize],
    edge_padding_low: &[i64],
    interior_step: &[i64],
) -> Result<PadCopyCursor> {
    let mut shape = AxisVec::with_capacity(operand_dims.len());
    let mut source_steps = AxisVec::with_capacity(operand_dims.len());
    let mut source_resets = AxisVec::with_capacity(operand_dims.len());
    let mut dest_steps = AxisVec::with_capacity(operand_dims.len());
    let mut dest_resets = AxisVec::with_capacity(operand_dims.len());
    let mut source_base_delta = 0isize;
    let mut dest_base_delta = 0isize;
    let mut copy_empty = false;

    for axis in 0..operand_dims.len() {
        let (start, end) = checked_pad_valid_interval(
            operand_dims[axis],
            dest_dims[axis],
            edge_padding_low[axis],
            interior_step[axis],
        )?;
        let extent = end - start;
        shape.push(extent);
        if !copy_empty && extent != 0 {
            source_base_delta =
                checked_offset_add(source_base_delta, operand_strides[axis], start)?;
            let output_start = i128::from(edge_padding_low[axis])
                .checked_add(
                    i128::try_from(start)
                        .map_err(|_| StridedError::OffsetOverflow)?
                        .checked_mul(i128::from(interior_step[axis]))
                        .ok_or(StridedError::OffsetOverflow)?,
                )
                .ok_or(StridedError::OffsetOverflow)?;
            let output_start =
                usize::try_from(output_start).map_err(|_| StridedError::OffsetOverflow)?;
            dest_base_delta =
                checked_offset_add(dest_base_delta, dest_strides[axis], output_start)?;
        }
        copy_empty |= extent == 0;

        let source_step = operand_strides[axis];
        let dest_step = checked_stride_mul_i64(dest_strides[axis], interior_step[axis])?;
        source_steps.push(source_step);
        source_resets.push(checked_cursor_reset(source_step, extent)?);
        dest_steps.push(dest_step);
        dest_resets.push(checked_cursor_reset(dest_step, extent)?);
    }

    check_offset_span(&shape, &source_steps)?;
    check_offset_span(&shape, &dest_steps)?;
    let total = if shape.iter().any(|&extent| extent == 0) {
        0
    } else {
        checked_total_len(&shape)?
    };

    Ok(PadCopyCursor {
        shape,
        source_base_delta,
        dest_base_delta,
        source_steps,
        source_resets,
        dest_steps,
        dest_resets,
        total,
    })
}

fn checked_pad_valid_interval(
    input_extent: usize,
    dest_extent: usize,
    edge_low: i64,
    step: i64,
) -> Result<(usize, usize)> {
    if input_extent == 0 || dest_extent == 0 {
        return Ok((0, 0));
    }
    let step = i128::from(step);
    let lower = ceil_div_positive(-i128::from(edge_low), step)?;
    let dest_last = i128::try_from(dest_extent)
        .map_err(|_| StridedError::OffsetOverflow)?
        .checked_sub(1)
        .ok_or(StridedError::OffsetOverflow)?;
    let upper = floor_div_positive(
        dest_last
            .checked_sub(i128::from(edge_low))
            .ok_or(StridedError::OffsetOverflow)?,
        step,
    )?
    .checked_add(1)
    .ok_or(StridedError::OffsetOverflow)?;
    let input_extent = i128::try_from(input_extent).map_err(|_| StridedError::OffsetOverflow)?;
    let lower = lower.clamp(0, input_extent);
    let upper = upper.clamp(0, input_extent);
    if lower >= upper {
        return Ok((0, 0));
    }
    Ok((
        usize::try_from(lower).map_err(|_| StridedError::OffsetOverflow)?,
        usize::try_from(upper).map_err(|_| StridedError::OffsetOverflow)?,
    ))
}

fn ceil_div_positive(numerator: i128, denominator: i128) -> Result<i128> {
    if denominator <= 0 {
        return Err(StridedError::OffsetOverflow);
    }
    let quotient = numerator.div_euclid(denominator);
    let remainder = numerator.rem_euclid(denominator);
    quotient
        .checked_add(i128::from(remainder != 0))
        .ok_or(StridedError::OffsetOverflow)
}

fn floor_div_positive(numerator: i128, denominator: i128) -> Result<i128> {
    if denominator <= 0 {
        return Err(StridedError::OffsetOverflow);
    }
    Ok(numerator.div_euclid(denominator))
}

fn checked_stride_mul_i64(stride: isize, factor: i64) -> Result<isize> {
    let factor = isize::try_from(factor).map_err(|_| StridedError::OffsetOverflow)?;
    stride
        .checked_mul(factor)
        .ok_or(StridedError::OffsetOverflow)
}

fn checked_cursor_reset(step: isize, extent: usize) -> Result<isize> {
    if extent == 0 {
        return Ok(0);
    }
    let last = isize::try_from(extent - 1).map_err(|_| StridedError::OffsetOverflow)?;
    step.checked_mul(last)
        .and_then(isize::checked_neg)
        .ok_or(StridedError::OffsetOverflow)
}

fn check_offset_span(shape: &[usize], strides: &[isize]) -> Result<()> {
    let mut min = 0isize;
    let mut max = 0isize;
    for (&extent, &stride) in shape.iter().zip(strides) {
        if extent <= 1 {
            continue;
        }
        let last = isize::try_from(extent - 1).map_err(|_| StridedError::OffsetOverflow)?;
        let delta = stride
            .checked_mul(last)
            .ok_or(StridedError::OffsetOverflow)?;
        if delta < 0 {
            min = min.checked_add(delta).ok_or(StridedError::OffsetOverflow)?;
        } else {
            max = max.checked_add(delta).ok_or(StridedError::OffsetOverflow)?;
        }
    }
    let _ = (min, max);
    Ok(())
}

fn is_dense_col_major(dims: &[usize], strides: &[isize]) -> bool {
    let mut expected = 1isize;
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        if stride != expected {
            return false;
        }
        let Ok(dim) = isize::try_from(dim) else {
            return false;
        };
        let Some(next) = expected.checked_mul(dim) else {
            return false;
        };
        expected = next;
    }
    true
}

fn compile_contiguous_pad_axis0_run(
    operand_dims: &[usize],
    operand_strides: &[isize],
    dest_dims: &[usize],
    dest_strides: &[isize],
    edge_padding_low: &[i64],
    interior_step: &[i64],
) -> Option<ContiguousPadAxis0Run> {
    if operand_dims.is_empty()
        || operand_strides[0] != 1
        || dest_strides[0] != 1
        || interior_step[0] != 1
    {
        return None;
    }

    let operand_extent = operand_dims[0] as i128;
    let dest_extent = dest_dims[0] as i128;
    let edge_low = i128::from(edge_padding_low[0]);
    let operand_start = (-edge_low).clamp(0, operand_extent);
    let dest_start = edge_low.clamp(0, dest_extent);
    let len = (operand_extent - operand_start).min(dest_extent - dest_start);
    Some(ContiguousPadAxis0Run {
        operand_start: usize::try_from(operand_start).ok()?,
        dest_start: usize::try_from(dest_start).ok()?,
        len: usize::try_from(len).ok()?,
    })
}

/// Compile the disjoint-span column write for the dense, non-cropping,
/// no-interior-padding class, or `None` to keep the fill-then-copy fallback.
#[allow(clippy::too_many_arguments)]
fn compile_contiguous_pad_axis0_columns(
    operand_dims: &[usize],
    operand_strides: &[isize],
    dest_dims: &[usize],
    edge_padding_low: &[i64],
    edge_padding_high: &[i64],
    interior_step: &[i64],
    contiguous_dest_fill: bool,
    run: ContiguousPadAxis0Run,
) -> Option<ContiguousPadAxis0Columns> {
    if run.len == 0 || !contiguous_dest_fill || dest_dims.is_empty() {
        return None;
    }
    if edge_padding_low.iter().any(|&low| low < 0)
        || edge_padding_high.iter().any(|&high| high < 0)
        || interior_step.iter().any(|&step| step != 1)
    {
        return None;
    }

    let rank = operand_dims.len();
    let dest_axis0_len = dest_dims[0];
    let dest_axis0_step = isize::try_from(dest_axis0_len).ok()?;
    let column_total = checked_total_len(&dest_dims[1..]).ok()?;
    let mut operand_outer_dims = AxisVec::with_capacity(rank - 1);
    let mut operand_outer_strides = AxisVec::with_capacity(rank - 1);
    let mut dest_outer_dims = AxisVec::with_capacity(rank - 1);
    let mut edge_low_outer = AxisVec::with_capacity(rank - 1);
    for axis in 1..rank {
        operand_outer_dims.push(operand_dims[axis]);
        operand_outer_strides.push(operand_strides[axis]);
        dest_outer_dims.push(dest_dims[axis]);
        edge_low_outer.push(usize::try_from(edge_padding_low[axis]).ok()?);
    }

    let low0 = run.dest_start;
    let high0 = dest_axis0_len.checked_sub(low0)?.checked_sub(run.len)?;
    Some(ContiguousPadAxis0Columns {
        column_total,
        dest_axis0_len,
        dest_axis0_step,
        low0,
        high0,
        run_len: run.len,
        operand_outer_dims,
        operand_outer_strides,
        dest_outer_dims,
        edge_low_outer,
    })
}

/// Fill a disjoint raw destination span, chunked across the budget of `exec`.
///
/// # Safety
///
/// `ptr..ptr + len` must be a writable destination span with no concurrent
/// access from another worker.
unsafe fn fill_raw_span<T: Copy + Send + Sync>(ptr: *mut T, len: usize, fill: T, exec: Exec<'_>) {
    if len == 0 {
        return;
    }
    // SAFETY: the caller guarantees the exact writable span; the slice is not
    // retained past `fill_contiguous`.
    let span = unsafe { core::slice::from_raw_parts_mut(ptr, len) };
    crate::threading::fill_contiguous(span, fill, exec);
}

/// Fill a disjoint raw destination span that may be uninitialized.
///
/// # Safety
///
/// `ptr..ptr + len` must be a writable destination span with no concurrent
/// access from another worker.
unsafe fn fill_raw_span_uninit<T: Copy + Send + Sync>(
    ptr: *mut T,
    len: usize,
    fill: T,
    exec: Exec<'_>,
) {
    if len == 0 {
        return;
    }
    // SAFETY: the caller guarantees the exact writable span; the
    // `MaybeUninit` view never forms a `&mut [T]` over uninitialized memory.
    unsafe { crate::threading::fill_contiguous_uninit(ptr, len, fill, exec) };
}

fn checked_offset_add(base: isize, stride: isize, coord: usize) -> Result<isize> {
    let coord = isize::try_from(coord).map_err(|_| StridedError::OffsetOverflow)?;
    let scaled = stride
        .checked_mul(coord)
        .ok_or(StridedError::OffsetOverflow)?;
    base.checked_add(scaled).ok_or(StridedError::OffsetOverflow)
}

/// The operand coordinate contributed by one outer axis, or `None` when the
/// axis falls outside the operand's in-bounds interval.
fn in_bounds_outer_coord(
    columns: &ContiguousPadAxis0Columns,
    axis: usize,
    coord: usize,
) -> Option<usize> {
    let q = coord.checked_sub(columns.edge_low_outer[axis])?;
    (q < columns.operand_outer_dims[axis]).then_some(q)
}

fn advance_col_major_index(index: &mut [usize], shape: &[usize]) {
    for axis in 0..index.len() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            return;
        }
        index[axis] = 0;
    }
}

fn fill_col_major_index(mut linear: usize, shape: &[usize], out: &mut [usize]) {
    for (axis, coord) in out.iter_mut().enumerate() {
        let dim = shape[axis];
        *coord = linear % dim;
        linear /= dim;
    }
}

struct PadFillState {
    coords: CoordScratch,
    offset: isize,
}

impl PadFillState {
    fn new(base: isize, shape: &[usize], _cursor: &PadFillCursor) -> Self {
        Self {
            coords: CoordScratch::new(shape.len()),
            offset: base,
        }
    }

    fn decode(linear: usize, base: isize, shape: &[usize], cursor: &PadFillCursor) -> Result<Self> {
        let mut state = Self::new(base, shape, cursor);
        fill_col_major_index(linear, shape, state.coords.as_mut_slice());
        for (&coord, &step) in state.coords.as_mut_slice().iter().zip(&cursor.steps) {
            state.offset = checked_offset_add(state.offset, step, coord)?;
        }
        Ok(state)
    }

    #[inline]
    fn advance(&mut self, shape: &[usize], cursor: &PadFillCursor) {
        // INVARIANT: precomputed checked offset arithmetic, the validated
        // descriptor, and exact layout equality prove each incremental offset
        // remains reachable.
        for axis in 0..shape.len() {
            let next = self.coords.as_mut_slice()[axis] + 1;
            if next < shape[axis] {
                self.coords.as_mut_slice()[axis] = next;
                self.offset += cursor.steps[axis];
                return;
            }
            self.coords.as_mut_slice()[axis] = 0;
            self.offset += cursor.resets[axis];
        }
    }
}

struct PadCopyState {
    coords: CoordScratch,
    source_offset: isize,
    dest_offset: isize,
}

impl PadCopyState {
    fn new(source_base: isize, dest_base: isize, cursor: &PadCopyCursor) -> Self {
        Self {
            coords: CoordScratch::new(cursor.shape.len()),
            source_offset: source_base,
            dest_offset: dest_base,
        }
    }

    fn decode(
        linear: usize,
        source_base: isize,
        dest_base: isize,
        cursor: &PadCopyCursor,
    ) -> Result<Self> {
        let mut state = Self::new(source_base, dest_base, cursor);
        fill_col_major_index(linear, &cursor.shape, state.coords.as_mut_slice());
        for axis in 0..cursor.shape.len() {
            let coord = state.coords.as_mut_slice()[axis];
            state.source_offset =
                checked_offset_add(state.source_offset, cursor.source_steps[axis], coord)?;
            state.dest_offset =
                checked_offset_add(state.dest_offset, cursor.dest_steps[axis], coord)?;
        }
        Ok(state)
    }

    #[inline]
    fn advance(&mut self, cursor: &PadCopyCursor) {
        // INVARIANT: compile-time checked steps/resets/offset arithmetic,
        // validated descriptors, and exact plan-layout equality prove every update below
        // stays in the reachable source/destination domains.
        for axis in 0..cursor.shape.len() {
            let next = self.coords.as_mut_slice()[axis] + 1;
            if next < cursor.shape[axis] {
                self.coords.as_mut_slice()[axis] = next;
                self.source_offset += cursor.source_steps[axis];
                self.dest_offset += cursor.dest_steps[axis];
                return;
            }
            self.coords.as_mut_slice()[axis] = 0;
            self.source_offset += cursor.source_resets[axis];
            self.dest_offset += cursor.dest_resets[axis];
        }
    }
}

struct CoordScratch {
    inline: [usize; crate::RAW_FUSED_RANK_LIMIT],
    heap: Option<Vec<usize>>,
    len: usize,
}

impl CoordScratch {
    fn new(len: usize) -> Self {
        if len <= crate::RAW_FUSED_RANK_LIMIT {
            Self {
                inline: [0; crate::RAW_FUSED_RANK_LIMIT],
                heap: None,
                len,
            }
        } else {
            Self {
                inline: [0; crate::RAW_FUSED_RANK_LIMIT],
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
#[path = "static_indexing_plan/tests/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "static_indexing_plan/tests/parallel_tests.rs"]
mod parallel_tests;
