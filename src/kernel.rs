use crate::fuse::fuse_dims;
use crate::{block, order, Result, StridedError};
use mdarray::{Layout, Shape, Slice};

pub(crate) struct StridedView<T> {
    pub(crate) ptr: *const T,
    pub(crate) dims: Vec<usize>,
    pub(crate) strides: Vec<isize>,
}

pub(crate) struct StridedViewMut<T> {
    pub(crate) ptr: *mut T,
    pub(crate) dims: Vec<usize>,
    pub(crate) strides: Vec<isize>,
}

impl<T> StridedView<T> {
    pub(crate) fn from_slice<S: Shape, L: Layout>(slice: &Slice<T, S, L>) -> Result<Self> {
        let (dims, strides) = dims_and_strides(slice);
        validate_layout(&dims, &strides)?;
        Ok(Self {
            ptr: slice.as_ptr(),
            dims,
            strides,
        })
    }

    pub(crate) fn from_parts(ptr: *const T, dims: &[usize], strides: &[isize]) -> Result<Self> {
        validate_layout(dims, strides)?;
        Ok(Self {
            ptr,
            dims: dims.to_vec(),
            strides: strides.to_vec(),
        })
    }
}

impl<T> StridedViewMut<T> {
    pub(crate) fn from_slice<S: Shape, L: Layout>(slice: &mut Slice<T, S, L>) -> Result<Self> {
        let (dims, strides) = dims_and_strides(&*slice);
        validate_layout(&dims, &strides)?;
        Ok(Self {
            ptr: slice.as_mut_ptr(),
            dims,
            strides,
        })
    }
}

pub(crate) struct KernelPlan {
    pub(crate) order: Vec<usize>, // outer -> inner
    pub(crate) block: Vec<usize>,
}

/// Build an execution plan for strided iteration.
///
/// This follows Julia's `_mapreduce_fuse!` -> `_mapreduce_order!` -> `_mapreduce_block!` pipeline:
/// 1. Fuse contiguous dimensions
/// 2. Compute optimal iteration order
/// 3. Compute block sizes for cache efficiency
pub(crate) fn build_plan(
    dims: &[usize],
    strides_list: &[&[isize]],
    dest_index: Option<usize>,
    elem_size: usize,
) -> KernelPlan {
    let order = order::compute_order(dims, strides_list, dest_index);
    let block = block::compute_block_sizes(dims, &order, strides_list, elem_size);
    KernelPlan { order, block }
}

/// Build an execution plan with dimension fusion.
///
/// This is the Julia-faithful version that fuses contiguous dimensions
/// before computing the iteration order.
pub(crate) fn build_plan_fused(
    dims: &[usize],
    strides_list: &[&[isize]],
    dest_index: Option<usize>,
    elem_size: usize,
) -> (Vec<usize>, KernelPlan) {
    // Fuse contiguous dimensions
    let fused_dims = fuse_dims(dims, strides_list);

    // Compute order and blocks on fused dimensions
    let order = order::compute_order(&fused_dims, strides_list, dest_index);
    let block = block::compute_block_sizes(&fused_dims, &order, strides_list, elem_size);

    (fused_dims, KernelPlan { order, block })
}

/// Iterate over all elements, calling f with the current offsets.
///
/// For small arrays or simple cases, considers using specialized fast paths.
#[inline]
pub(crate) fn for_each_offset<F>(
    dims: &[usize],
    plan: &KernelPlan,
    strides_list: &[&[isize]],
    mut f: F,
) -> Result<()>
where
    F: FnMut(&[isize]) -> Result<()>,
{
    let rank = dims.len();
    if rank == 0 {
        let offsets = vec![0isize; strides_list.len()];
        return f(&offsets);
    }

    let mut offsets = vec![0isize; strides_list.len()];
    let mut strides_by_level = Vec::with_capacity(plan.order.len());
    for &dim in &plan.order {
        let mut per_array = Vec::with_capacity(strides_list.len());
        for strides in strides_list {
            per_array.push(strides[dim]);
        }
        strides_by_level.push(per_array);
    }

    loop_level(
        0,
        dims,
        &plan.order,
        &plan.block,
        &strides_by_level,
        &mut offsets,
        &mut f,
    )
}

/// Iterate over blocks, calling f with (offsets, block_len, inner_strides).
///
/// This is useful for operations that can vectorize the innermost loop.
#[inline]
pub(crate) fn for_each_inner_block<F>(
    dims: &[usize],
    plan: &KernelPlan,
    strides_list: &[&[isize]],
    mut f: F,
) -> Result<()>
where
    F: FnMut(&[isize], usize, &[isize]) -> Result<()>,
{
    let rank = dims.len();
    if rank == 0 {
        let offsets = vec![0isize; strides_list.len()];
        return f(&offsets, 1, &[]);
    }

    let mut offsets = vec![0isize; strides_list.len()];
    let mut strides_by_level = Vec::with_capacity(plan.order.len());
    for &dim in &plan.order {
        let mut per_array = Vec::with_capacity(strides_list.len());
        for strides in strides_list {
            per_array.push(strides[dim]);
        }
        strides_by_level.push(per_array);
    }

    let inner_level = plan.order.len().saturating_sub(1);
    loop_outer(
        0,
        dims,
        &plan.order,
        &plan.block,
        &strides_by_level,
        inner_level,
        &mut offsets,
        &mut f,
    )
}

#[inline]
fn loop_level<F>(
    level: usize,
    dims: &[usize],
    order: &[usize],
    block: &[usize],
    strides_by_level: &[Vec<isize>],
    offsets: &mut [isize],
    f: &mut F,
) -> Result<()>
where
    F: FnMut(&[isize]) -> Result<()>,
{
    if level == order.len() {
        return f(offsets);
    }

    let dim = order[level];
    let dim_len = dims[dim];
    if dim_len == 0 {
        return Ok(());
    }

    let step = block[level].max(1).min(dim_len);
    let stride_vec = &strides_by_level[level];
    let base_offsets = offsets.to_vec();

    let mut start = 0usize;
    while start < dim_len {
        let block_len = step.min(dim_len - start);

        for ((offset, base), stride) in offsets
            .iter_mut()
            .zip(base_offsets.iter())
            .zip(stride_vec.iter())
        {
            *offset = checked_offset(*base, *stride, start)?;
        }

        for _ in 0..block_len {
            loop_level(level + 1, dims, order, block, strides_by_level, offsets, f)?;
            for (offset, stride) in offsets.iter_mut().zip(stride_vec.iter()) {
                *offset = checked_add(*offset, *stride)?;
            }
        }

        start += block_len;
    }

    offsets.copy_from_slice(&base_offsets);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn loop_outer<F>(
    level: usize,
    dims: &[usize],
    order: &[usize],
    block: &[usize],
    strides_by_level: &[Vec<isize>],
    inner_level: usize,
    offsets: &mut [isize],
    f: &mut F,
) -> Result<()>
where
    F: FnMut(&[isize], usize, &[isize]) -> Result<()>,
{
    if level == order.len() {
        return Ok(());
    }

    let dim = order[level];
    let dim_len = dims[dim];
    if dim_len == 0 {
        return Ok(());
    }

    let step = block[level].max(1).min(dim_len);
    let stride_vec = &strides_by_level[level];
    let base_offsets = offsets.to_vec();

    let mut start = 0usize;
    while start < dim_len {
        let block_len = step.min(dim_len - start);

        for ((offset, base), stride) in offsets
            .iter_mut()
            .zip(base_offsets.iter())
            .zip(stride_vec.iter())
        {
            *offset = checked_offset(*base, *stride, start)?;
        }

        if level == inner_level {
            checked_inner_block(offsets, stride_vec, block_len)?;
            f(offsets, block_len, stride_vec)?;
        } else {
            for _ in 0..block_len {
                loop_outer(
                    level + 1,
                    dims,
                    order,
                    block,
                    strides_by_level,
                    inner_level,
                    offsets,
                    f,
                )?;
                for (offset, stride) in offsets.iter_mut().zip(stride_vec.iter()) {
                    *offset = checked_add(*offset, *stride)?;
                }
            }
        }

        start += block_len;
    }

    offsets.copy_from_slice(&base_offsets);
    Ok(())
}

pub(crate) fn validate_layout(dims: &[usize], strides: &[isize]) -> Result<()> {
    if dims.len() != strides.len() {
        return Err(StridedError::StrideLengthMismatch);
    }
    for (i, (&dim, &stride)) in dims.iter().zip(strides.iter()).enumerate() {
        if dim > 1 && stride == 0 {
            return Err(StridedError::ZeroStride { dim: i });
        }
    }
    Ok(())
}

fn checked_inner_block(offsets: &[isize], strides: &[isize], len: usize) -> Result<()> {
    if len <= 1 {
        return Ok(());
    }
    let last = isize::try_from(len - 1).map_err(|_| StridedError::OffsetOverflow)?;
    for (&base, &stride) in offsets.iter().zip(strides.iter()) {
        let delta = stride
            .checked_mul(last)
            .ok_or(StridedError::OffsetOverflow)?;
        base.checked_add(delta)
            .ok_or(StridedError::OffsetOverflow)?;
    }
    Ok(())
}

fn dims_and_strides<T, S: Shape, L: Layout>(slice: &Slice<T, S, L>) -> (Vec<usize>, Vec<isize>) {
    let rank = slice.rank();
    let mut dims = Vec::with_capacity(rank);
    let mut strides = Vec::with_capacity(rank);
    for i in 0..rank {
        dims.push(slice.dim(i));
        strides.push(slice.stride(i));
    }
    (dims, strides)
}

fn checked_offset(base: isize, stride: isize, index: usize) -> Result<isize> {
    let idx = isize::try_from(index).map_err(|_| StridedError::OffsetOverflow)?;
    let delta = stride
        .checked_mul(idx)
        .ok_or(StridedError::OffsetOverflow)?;
    base.checked_add(delta).ok_or(StridedError::OffsetOverflow)
}

fn checked_add(a: isize, b: isize) -> Result<isize> {
    a.checked_add(b).ok_or(StridedError::OffsetOverflow)
}

pub(crate) fn ensure_same_shape(a: &[usize], b: &[usize]) -> Result<()> {
    if a.len() != b.len() {
        return Err(StridedError::RankMismatch(a.len(), b.len()));
    }
    if a != b {
        return Err(StridedError::ShapeMismatch(a.to_vec(), b.to_vec()));
    }
    Ok(())
}

pub(crate) fn is_contiguous(dims: &[usize], strides: &[isize]) -> bool {
    if dims.len() != strides.len() {
        return false;
    }
    if dims.is_empty() {
        return true;
    }
    let mut expected = 1isize;
    for (&dim, &stride) in dims.iter().rev().zip(strides.iter().rev()) {
        if dim <= 1 {
            continue;
        }
        if stride != expected {
            return false;
        }
        expected = expected.saturating_mul(dim as isize);
    }
    true
}

pub(crate) fn total_len(dims: &[usize]) -> usize {
    if dims.is_empty() {
        return 1;
    }
    dims.iter().product()
}
