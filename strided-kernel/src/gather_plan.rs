//! Prepared gather plans over raw strided value and index layouts.
//!
//! This module owns the generic indexed-read traversal used by the erased
//! replay layer. It models the XLA/tenferro gather shape vocabulary, but keeps
//! tensor allocation, dtype promotion, and frontend error policy outside
//! `strided-kernel`.

use crate::{
    MaybeSendSync, RawStridedMut, RawStridedRef, Result, StridedError, RAW_FUSED_RANK_LIMIT,
};

#[cfg(feature = "parallel")]
type AxisVec<T> = smallvec::SmallVec<[T; RAW_FUSED_RANK_LIMIT]>;
#[cfg(not(feature = "parallel"))]
type AxisVec<T> = Vec<T>;

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
pub trait GatherIndex: Copy + MaybeSendSync {
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
    batch_shape: AxisVec<usize>,
    out_axis_to_operand_dim: AxisVec<Option<usize>>,
}

impl GatherPlan {
    /// Compile a gather plan for fixed operand, index, and destination layouts.
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
        checked_total_len(dest_dims)?;
        if !crate::fused::is_injective_layout(dest_dims, dest_strides) {
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

        Ok(Self {
            operand_dims: operand_dims.into(),
            operand_strides: operand_strides.into(),
            index_dims: index_dims.into(),
            index_strides: index_strides.into(),
            dest_dims: dest_dims.into(),
            dest_strides: dest_strides.into(),
            spec,
            batch_shape,
            out_axis_to_operand_dim,
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
    pub fn execute<T, I>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        operand: &RawStridedRef<'_, T>,
        start_indices: &RawStridedRef<'_, I>,
    ) -> Result<()>
    where
        T: Copy + MaybeSendSync,
        I: GatherIndex,
    {
        self.check_call(dest, operand, start_indices)?;
        let total = checked_total_len(&self.dest_dims)?;
        if total == 0 {
            return Ok(());
        }

        let mut out_idx_storage = CoordScratch::new(self.dest_dims.len());
        let mut batch_idx_storage = CoordScratch::new(self.batch_shape.len());
        let mut operand_idx_storage = CoordScratch::new(self.operand_dims.len());
        let mut window_offsets_storage = CoordScratch::new(self.operand_dims.len());
        let out_idx = out_idx_storage.as_mut_slice();
        let batch_idx = batch_idx_storage.as_mut_slice();
        let operand_idx = operand_idx_storage.as_mut_slice();
        let window_offsets = window_offsets_storage.as_mut_slice();

        let dest_offset_base = dest.offset();
        let dest_strides = dest.strides();
        let operand_offset_base = operand.offset();
        let operand_strides = operand.strides();
        let index_offset_base = start_indices.offset();
        let index_strides = start_indices.strides();
        let operand_data = operand.data();
        let index_data = start_indices.data();
        let dest_data = dest.data_mut();

        for _ in 0..total {
            window_offsets.fill(0);
            let mut batch_axis = 0usize;
            for (out_axis, &operand_dim) in self.out_axis_to_operand_dim.iter().enumerate() {
                match operand_dim {
                    Some(axis) => window_offsets[axis] = out_idx[out_axis],
                    None => {
                        batch_idx[batch_axis] = out_idx[out_axis];
                        batch_axis += 1;
                    }
                }
            }

            operand_idx.fill(0);
            for (component, &operand_dim) in self.spec.start_index_map.iter().enumerate() {
                let start = self.index_component(
                    start_indices.dims(),
                    index_strides,
                    index_offset_base,
                    index_data,
                    &batch_idx,
                    component,
                )?;
                operand_idx[operand_dim] = self.clamp_window_start(start, operand_dim);
            }
            for axis in 0..operand_idx.len() {
                operand_idx[axis] += window_offsets[axis];
            }

            let dest_offset = checked_strided_offset(dest_offset_base, dest_strides, &out_idx)?;
            let operand_offset =
                checked_strided_offset(operand_offset_base, operand_strides, &operand_idx)?;
            unsafe {
                *dest_data.as_mut_ptr().offset(dest_offset) =
                    *operand_data.as_ptr().offset(operand_offset);
            }
            advance_col_major_index(out_idx, &self.dest_dims);
        }
        Ok(())
    }

    fn check_call<T, I>(
        &self,
        dest: &RawStridedMut<'_, T>,
        operand: &RawStridedRef<'_, T>,
        start_indices: &RawStridedRef<'_, I>,
    ) -> Result<()> {
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

    fn index_component<I>(
        &self,
        index_dims: &[usize],
        index_strides: &[isize],
        index_offset_base: isize,
        index_data: &[I],
        batch_idx: &[usize],
        component: usize,
    ) -> Result<i64>
    where
        I: GatherIndex,
    {
        let mut offset = index_offset_base;
        let mut batch_axis = 0usize;
        for axis in 0..index_dims.len() {
            let coord = if axis == self.spec.index_vector_dim {
                component
            } else {
                let coord = batch_idx[batch_axis];
                batch_axis += 1;
                coord
            };
            offset = checked_offset_add(offset, index_strides[axis], coord)?;
        }
        Ok(unsafe { *index_data.as_ptr().offset(offset) }.to_i64())
    }

    #[inline]
    fn clamp_window_start(&self, start: i64, operand_dim: usize) -> usize {
        let dim_size = self.operand_dims[operand_dim];
        let window_size = self.spec.slice_sizes[operand_dim];
        let max_start = dim_size.saturating_sub(window_size) as i64;
        start.clamp(0, max_start) as usize
    }
}

fn validate_unique_axes(axes: &[usize], rank: usize) -> Result<()> {
    let mut seen = vec![false; rank];
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

fn advance_col_major_index(index: &mut [usize], shape: &[usize]) {
    for axis in 0..index.len() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            return;
        }
        index[axis] = 0;
    }
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
