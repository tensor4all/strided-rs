//! Prepared static-indexing plans over raw strided value layouts.
//!
//! This module owns reusable static indexing traversal for downstream tensor
//! runtimes. It keeps allocation, dtype policy, and tensor-level validation out
//! of `strided-kernel`; callers provide already-owned output buffers and fixed
//! raw descriptors.

use crate::{CopyPlan, MaybeSendSync, RawStridedMut, RawStridedRef, Result, StridedError};

#[cfg(feature = "parallel")]
type AxisVec<T> = smallvec::SmallVec<[T; crate::RAW_FUSED_RANK_LIMIT]>;
#[cfg(not(feature = "parallel"))]
type AxisVec<T> = Vec<T>;

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
/// Replay first fills every destination element with the caller-provided fill
/// scalar, then copies reachable input positions into the padded output.
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
}

/// A compiled multi-input concatenate traversal.
///
/// Each input segment is lowered to a prepared strided copy into the
/// corresponding destination window.
#[derive(Clone, Debug)]
pub struct ConcatenatePlan {
    input_dims: Vec<AxisVec<usize>>,
    input_strides: Vec<AxisVec<isize>>,
    dest_dims: AxisVec<usize>,
    dest_strides: AxisVec<isize>,
    dest_offset_deltas: Vec<isize>,
    copy_plans: Vec<CopyPlan>,
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
    ) -> Result<()>
    where
        T: Copy + MaybeSendSync,
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
        self.copy_plan.execute(dest, &source)
    }

    fn check_call<T>(
        &self,
        dest: &RawStridedMut<'_, T>,
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
        if !crate::fused::is_injective_layout(dest_dims, dest_strides) {
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

        Ok(Self {
            operand_dims: operand_dims.into(),
            operand_strides: operand_strides.into(),
            dest_dims: dest_dims.into(),
            dest_strides: dest_strides.into(),
            edge_padding_low: edge_padding_low.into(),
            interior_step,
            operand_total,
            dest_total,
        })
    }

    /// Execute the prepared pad traversal.
    pub fn execute<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        operand: &RawStridedRef<'_, T>,
        fill: T,
    ) -> Result<()>
    where
        T: Copy + MaybeSendSync,
    {
        self.check_call(dest, operand)?;
        self.fill_dest(dest, fill)?;

        if self.operand_total == 0 {
            return Ok(());
        }
        self.copy_operand(dest, operand)
    }

    fn fill_dest<T>(&self, dest: &mut RawStridedMut<'_, T>, fill: T) -> Result<()>
    where
        T: Copy + MaybeSendSync,
    {
        if self.dest_total == 0 {
            return Ok(());
        }
        #[cfg(feature = "parallel")]
        {
            let nthreads = crate::threading::parallel_threads_for_len(self.dest_total);
            if nthreads > 1 {
                return self.fill_dest_parallel(dest, fill, nthreads);
            }
        }
        self.fill_dest_serial(dest, fill)
    }

    fn fill_dest_serial<T>(&self, dest: &mut RawStridedMut<'_, T>, fill: T) -> Result<()>
    where
        T: Copy,
    {
        let dest_offset_base = dest.offset();
        let dest_strides = dest.strides();
        let dest_data = dest.data_mut();
        let mut dest_idx_storage = CoordScratch::new(self.dest_dims.len());
        let dest_idx = dest_idx_storage.as_mut_slice();
        for _ in 0..self.dest_total {
            let dest_offset = checked_strided_offset(dest_offset_base, dest_strides, dest_idx)?;
            unsafe {
                *dest_data.as_mut_ptr().offset(dest_offset) = fill;
            }
            advance_col_major_index(dest_idx, &self.dest_dims);
        }
        Ok(())
    }

    #[cfg(feature = "parallel")]
    fn fill_dest_parallel<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        fill: T,
        nthreads: usize,
    ) -> Result<()>
    where
        T: Copy + MaybeSendSync,
    {
        let dest_offset_base = dest.offset();
        let dest_ptr = crate::threading::SendPtr(dest.data_mut().as_mut_ptr());
        crate::threading::parallel_map_reduce(
            0..self.dest_total,
            nthreads,
            &|range| {
                let mut dest_idx_storage = CoordScratch::new(self.dest_dims.len());
                let dest_idx = dest_idx_storage.as_mut_slice();
                fill_col_major_index(range.start, &self.dest_dims, dest_idx);
                let dest_ptr = dest_ptr.as_ptr();
                for _ in range {
                    let dest_offset =
                        checked_strided_offset(dest_offset_base, &self.dest_strides, dest_idx)?;
                    unsafe {
                        // SAFETY: `compile` rejected non-injective destination
                        // layouts, and each logical destination index is visited
                        // by exactly one range partition.
                        *dest_ptr.offset(dest_offset) = fill;
                    }
                    advance_col_major_index(dest_idx, &self.dest_dims);
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
    ) -> Result<()>
    where
        T: Copy + MaybeSendSync,
    {
        #[cfg(feature = "parallel")]
        {
            let nthreads = crate::threading::parallel_threads_for_len(self.operand_total);
            if nthreads > 1 {
                return self.copy_operand_parallel(dest, operand, nthreads);
            }
        }
        self.copy_operand_serial(dest, operand)
    }

    fn copy_operand_serial<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        operand: &RawStridedRef<'_, T>,
    ) -> Result<()>
    where
        T: Copy,
    {
        let operand_offset_base = operand.offset();
        let operand_strides = operand.strides();
        let operand_data = operand.data();
        let dest_offset_base = dest.offset();
        let dest_strides = dest.strides();
        let dest_data = dest.data_mut();
        let mut input_idx_storage = CoordScratch::new(self.operand_dims.len());
        let mut out_idx_storage = CoordScratch::new(self.dest_dims.len());
        let input_idx = input_idx_storage.as_mut_slice();
        let out_idx = out_idx_storage.as_mut_slice();

        for _ in 0..self.operand_total {
            let mut in_bounds = true;
            for axis in 0..self.operand_dims.len() {
                let out_pos = i128::from(self.edge_padding_low[axis])
                    + input_idx[axis] as i128 * i128::from(self.interior_step[axis]);
                if out_pos < 0 || out_pos >= self.dest_dims[axis] as i128 {
                    in_bounds = false;
                    break;
                }
                out_idx[axis] = out_pos as usize;
            }
            if in_bounds {
                let operand_offset =
                    checked_strided_offset(operand_offset_base, operand_strides, input_idx)?;
                let dest_offset = checked_strided_offset(dest_offset_base, dest_strides, out_idx)?;
                unsafe {
                    *dest_data.as_mut_ptr().offset(dest_offset) =
                        *operand_data.as_ptr().offset(operand_offset);
                }
            }
            advance_col_major_index(input_idx, &self.operand_dims);
        }
        Ok(())
    }

    #[cfg(feature = "parallel")]
    fn copy_operand_parallel<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        operand: &RawStridedRef<'_, T>,
        nthreads: usize,
    ) -> Result<()>
    where
        T: Copy + MaybeSendSync,
    {
        let operand_offset_base = operand.offset();
        let operand_ptr = crate::threading::SendPtr(operand.data().as_ptr() as *mut T);
        let dest_offset_base = dest.offset();
        let dest_ptr = crate::threading::SendPtr(dest.data_mut().as_mut_ptr());
        crate::threading::parallel_map_reduce(
            0..self.operand_total,
            nthreads,
            &|range| {
                let mut input_idx_storage = CoordScratch::new(self.operand_dims.len());
                let mut out_idx_storage = CoordScratch::new(self.dest_dims.len());
                let input_idx = input_idx_storage.as_mut_slice();
                let out_idx = out_idx_storage.as_mut_slice();
                fill_col_major_index(range.start, &self.operand_dims, input_idx);
                let operand_ptr = operand_ptr.as_const();
                let dest_ptr = dest_ptr.as_ptr();

                for _ in range {
                    let mut in_bounds = true;
                    for axis in 0..self.operand_dims.len() {
                        let out_pos = i128::from(self.edge_padding_low[axis])
                            + input_idx[axis] as i128 * i128::from(self.interior_step[axis]);
                        if out_pos < 0 || out_pos >= self.dest_dims[axis] as i128 {
                            in_bounds = false;
                            break;
                        }
                        out_idx[axis] = out_pos as usize;
                    }
                    if in_bounds {
                        let operand_offset = checked_strided_offset(
                            operand_offset_base,
                            &self.operand_strides,
                            input_idx,
                        )?;
                        let dest_offset =
                            checked_strided_offset(dest_offset_base, &self.dest_strides, out_idx)?;
                        unsafe {
                            // SAFETY: positive interior steps make the
                            // input-to-output mapping injective for in-bounds
                            // positions; the destination layout is also
                            // injective.
                            *dest_ptr.offset(dest_offset) = *operand_ptr.offset(operand_offset);
                        }
                    }
                    advance_col_major_index(input_idx, &self.operand_dims);
                }
                Ok(())
            },
            &|left, right| left.and(right),
        )
    }

    fn check_call<T>(
        &self,
        dest: &RawStridedMut<'_, T>,
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
        if !crate::fused::is_injective_layout(dest_dims, dest_strides) {
            return Err(StridedError::NonInjectiveOutputLayout);
        }

        let mut expected_dest_dims: AxisVec<usize> = input_dims[0].into();
        expected_dest_dims[axis] = 0;
        let mut stored_input_dims = Vec::with_capacity(input_dims.len());
        let mut stored_input_strides = Vec::with_capacity(input_dims.len());
        let mut dest_offset_deltas = Vec::with_capacity(input_dims.len());
        let mut copy_plans = Vec::with_capacity(input_dims.len());
        let mut axis_base = 0usize;

        for (dims, strides) in input_dims.iter().zip(input_strides.iter()) {
            if dims.len() != rank {
                return Err(StridedError::RankMismatch(dims.len(), rank));
            }
            if strides.len() != rank {
                return Err(StridedError::StrideLengthMismatch);
            }
            checked_total_len(dims)?;
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
            copy_plans,
        })
    }

    /// Execute the prepared concatenate traversal.
    pub fn execute<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        inputs: &[RawStridedRef<'_, T>],
    ) -> Result<()>
    where
        T: Copy + MaybeSendSync,
    {
        self.check_dest_layout(dest)?;
        if inputs.len() != self.input_dims.len() {
            return Err(StridedError::RankMismatch(
                inputs.len(),
                self.input_dims.len(),
            ));
        }
        for (position, input) in inputs.iter().enumerate() {
            self.check_input_layout(position, input)?;
            self.execute_segment(position, dest, input)?;
        }
        Ok(())
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

    pub(crate) fn execute_segment<T>(
        &self,
        position: usize,
        dest: &mut RawStridedMut<'_, T>,
        input: &RawStridedRef<'_, T>,
    ) -> Result<()>
    where
        T: Copy + MaybeSendSync,
    {
        let segment_offset = dest
            .offset()
            .checked_add(self.dest_offset_deltas[position])
            .ok_or(StridedError::OffsetOverflow)?;
        let dest_data = dest.data_mut();
        let mut segment = unsafe {
            RawStridedMut::new_unchecked(
                dest_data,
                &self.input_dims[position],
                &self.dest_strides,
                segment_offset,
            )
        };
        self.copy_plans[position].execute(&mut segment, input)
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
    ) -> Result<()>
    where
        T: Copy + MaybeSendSync,
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
        self.copy_plan.execute(dest, &source)
    }

    fn check_call<T>(
        &self,
        dest: &RawStridedMut<'_, T>,
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

fn advance_col_major_index(index: &mut [usize], shape: &[usize]) {
    for axis in 0..index.len() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            return;
        }
        index[axis] = 0;
    }
}

#[cfg(feature = "parallel")]
fn fill_col_major_index(mut linear: usize, shape: &[usize], out: &mut [usize]) {
    for (axis, coord) in out.iter_mut().enumerate() {
        let dim = shape[axis];
        *coord = linear % dim;
        linear /= dim;
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
