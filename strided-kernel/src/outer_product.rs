//! Outer-product kernels on dynamic-rank strided views.

use std::ops::Mul;

use crate::view::{StridedView, StridedViewMut};
use crate::{ElementOp, Result, StridedError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BatchedOuterShape {
    lhs_free_ndim: usize,
    rhs_free_ndim: usize,
    batch_ndim: usize,
    rows: usize,
    cols: usize,
    batches: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompactBatchedOuter {
    dst_row_stride: isize,
    dst_col_stride: isize,
    dst_batch_stride: isize,
    lhs_row_stride: isize,
    lhs_batch_stride: isize,
    rhs_col_stride: isize,
    rhs_batch_stride: isize,
}

/// Compute `dest[lhs_free..., rhs_free..., batch...] =
/// lhs[lhs_free..., batch...] * rhs[rhs_free..., batch...]`.
///
/// The dimension groups are determined by `lhs_free_ndim` and `rhs_free_ndim`.
/// A pure outer product has zero batch dimensions.
pub fn batched_outer_product_into<T, OpA, OpB>(
    dest: &mut StridedViewMut<T>,
    lhs: &StridedView<T, OpA>,
    rhs: &StridedView<T, OpB>,
    lhs_free_ndim: usize,
    rhs_free_ndim: usize,
) -> Result<()>
where
    T: Copy + Mul<Output = T>,
    OpA: ElementOp<T>,
    OpB: ElementOp<T>,
{
    batched_outer_product_into_seq(dest, lhs, rhs, lhs_free_ndim, rhs_free_ndim)
}

/// Sequential batched outer-product kernel.
pub fn batched_outer_product_into_seq<T, OpA, OpB>(
    dest: &mut StridedViewMut<T>,
    lhs: &StridedView<T, OpA>,
    rhs: &StridedView<T, OpB>,
    lhs_free_ndim: usize,
    rhs_free_ndim: usize,
) -> Result<()>
where
    T: Copy + Mul<Output = T>,
    OpA: ElementOp<T>,
    OpB: ElementOp<T>,
{
    let shape = validate_batched_outer_shape(dest, lhs, rhs, lhs_free_ndim, rhs_free_ndim)?;
    if let Some(compact) = compact_batched_outer(dest, lhs, rhs, &shape) {
        unsafe {
            execute_compact_seq(dest, lhs, rhs, &shape, &compact);
        }
    } else {
        unsafe {
            execute_strided_seq(dest, lhs, rhs, &shape)?;
        }
    }
    Ok(())
}

/// Parallel batched outer-product kernel using Rayon.
#[cfg(feature = "parallel")]
pub fn batched_outer_product_into_par<T, OpA, OpB>(
    dest: &mut StridedViewMut<T>,
    lhs: &StridedView<T, OpA>,
    rhs: &StridedView<T, OpB>,
    lhs_free_ndim: usize,
    rhs_free_ndim: usize,
) -> Result<()>
where
    T: Copy + Send + Sync + Mul<Output = T>,
    OpA: ElementOp<T> + Send + Sync,
    OpB: ElementOp<T> + Send + Sync,
{
    let shape = validate_batched_outer_shape(dest, lhs, rhs, lhs_free_ndim, rhs_free_ndim)?;
    if let Some(compact) = compact_batched_outer(dest, lhs, rhs, &shape) {
        unsafe {
            execute_compact_par(dest, lhs, rhs, &shape, &compact);
        }
    } else {
        unsafe {
            execute_strided_par(dest, lhs, rhs, &shape)?;
        }
    }
    Ok(())
}

/// Choose the sequential or Rayon implementation by output element count.
#[cfg(feature = "parallel")]
pub fn batched_outer_product_into_auto<T, OpA, OpB>(
    dest: &mut StridedViewMut<T>,
    lhs: &StridedView<T, OpA>,
    rhs: &StridedView<T, OpB>,
    lhs_free_ndim: usize,
    rhs_free_ndim: usize,
    min_parallel_elements: usize,
) -> Result<()>
where
    T: Copy + Send + Sync + Mul<Output = T>,
    OpA: ElementOp<T> + Send + Sync,
    OpB: ElementOp<T> + Send + Sync,
{
    let shape = validate_batched_outer_shape(dest, lhs, rhs, lhs_free_ndim, rhs_free_ndim)?;
    let total = checked_mul3(shape.rows, shape.cols, shape.batches)?;
    if total >= min_parallel_elements && rayon::current_num_threads() > 1 {
        batched_outer_product_into_par(dest, lhs, rhs, lhs_free_ndim, rhs_free_ndim)
    } else {
        batched_outer_product_into_seq(dest, lhs, rhs, lhs_free_ndim, rhs_free_ndim)
    }
}

fn validate_batched_outer_shape<T, OpA, OpB>(
    dest: &StridedViewMut<T>,
    lhs: &StridedView<T, OpA>,
    rhs: &StridedView<T, OpB>,
    lhs_free_ndim: usize,
    rhs_free_ndim: usize,
) -> Result<BatchedOuterShape> {
    if lhs_free_ndim > lhs.ndim() {
        return Err(StridedError::RankMismatch(lhs_free_ndim, lhs.ndim()));
    }
    if rhs_free_ndim > rhs.ndim() {
        return Err(StridedError::RankMismatch(rhs_free_ndim, rhs.ndim()));
    }

    let lhs_batch_ndim = lhs.ndim() - lhs_free_ndim;
    let rhs_batch_ndim = rhs.ndim() - rhs_free_ndim;
    if lhs_batch_ndim != rhs_batch_ndim {
        return Err(StridedError::RankMismatch(lhs_batch_ndim, rhs_batch_ndim));
    }
    let expected_dest_rank = lhs_free_ndim + rhs_free_ndim + lhs_batch_ndim;
    if dest.ndim() != expected_dest_rank {
        return Err(StridedError::RankMismatch(dest.ndim(), expected_dest_rank));
    }

    ensure_dims(&dest.dims()[..lhs_free_ndim], &lhs.dims()[..lhs_free_ndim])?;
    ensure_dims(
        &dest.dims()[lhs_free_ndim..lhs_free_ndim + rhs_free_ndim],
        &rhs.dims()[..rhs_free_ndim],
    )?;
    ensure_dims(
        &dest.dims()[lhs_free_ndim + rhs_free_ndim..],
        &lhs.dims()[lhs_free_ndim..],
    )?;
    ensure_dims(
        &dest.dims()[lhs_free_ndim + rhs_free_ndim..],
        &rhs.dims()[rhs_free_ndim..],
    )?;

    Ok(BatchedOuterShape {
        lhs_free_ndim,
        rhs_free_ndim,
        batch_ndim: lhs_batch_ndim,
        rows: product(&lhs.dims()[..lhs_free_ndim])?,
        cols: product(&rhs.dims()[..rhs_free_ndim])?,
        batches: product(&lhs.dims()[lhs_free_ndim..])?,
    })
}

fn ensure_dims(a: &[usize], b: &[usize]) -> Result<()> {
    if a != b {
        return Err(StridedError::ShapeMismatch(a.to_vec(), b.to_vec()));
    }
    Ok(())
}

fn product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or(StridedError::OffsetOverflow)
    })
}

#[cfg(feature = "parallel")]
fn checked_mul3(a: usize, b: usize, c: usize) -> Result<usize> {
    a.checked_mul(b)
        .and_then(|x| x.checked_mul(c))
        .ok_or(StridedError::OffsetOverflow)
}

fn compact_batched_outer<T, OpA, OpB>(
    dest: &StridedViewMut<T>,
    lhs: &StridedView<T, OpA>,
    rhs: &StridedView<T, OpB>,
    shape: &BatchedOuterShape,
) -> Option<CompactBatchedOuter> {
    Some(CompactBatchedOuter {
        dst_row_stride: compact_group_stride(dest.dims(), dest.strides(), 0, shape.lhs_free_ndim)?,
        dst_col_stride: compact_group_stride(
            dest.dims(),
            dest.strides(),
            shape.lhs_free_ndim,
            shape.lhs_free_ndim + shape.rhs_free_ndim,
        )?,
        dst_batch_stride: compact_group_stride(
            dest.dims(),
            dest.strides(),
            shape.lhs_free_ndim + shape.rhs_free_ndim,
            dest.ndim(),
        )?,
        lhs_row_stride: compact_group_stride(lhs.dims(), lhs.strides(), 0, shape.lhs_free_ndim)?,
        lhs_batch_stride: compact_group_stride(
            lhs.dims(),
            lhs.strides(),
            shape.lhs_free_ndim,
            lhs.ndim(),
        )?,
        rhs_col_stride: compact_group_stride(rhs.dims(), rhs.strides(), 0, shape.rhs_free_ndim)?,
        rhs_batch_stride: compact_group_stride(
            rhs.dims(),
            rhs.strides(),
            shape.rhs_free_ndim,
            rhs.ndim(),
        )?,
    })
}

fn compact_group_stride(
    dims: &[usize],
    strides: &[isize],
    start: usize,
    end: usize,
) -> Option<isize> {
    let mut base = None;
    let mut expected = 0isize;
    for axis in start..end {
        if dims[axis] <= 1 {
            continue;
        }
        if let Some(_) = base {
            if strides[axis] != expected {
                return None;
            }
        } else {
            base = Some(strides[axis]);
            expected = strides[axis];
        }
        expected = expected.checked_mul(dims[axis] as isize)?;
    }
    Some(base.unwrap_or(0))
}

unsafe fn execute_compact_seq<T, OpA, OpB>(
    dest: &mut StridedViewMut<T>,
    lhs: &StridedView<T, OpA>,
    rhs: &StridedView<T, OpB>,
    shape: &BatchedOuterShape,
    compact: &CompactBatchedOuter,
) where
    T: Copy + Mul<Output = T>,
    OpA: ElementOp<T>,
    OpB: ElementOp<T>,
{
    let dst = dest.as_mut_ptr();
    let lhs_ptr = lhs.ptr();
    let rhs_ptr = rhs.ptr();
    for batch in 0..shape.batches {
        let batch = batch as isize;
        let dst_batch = batch * compact.dst_batch_stride;
        let lhs_batch = batch * compact.lhs_batch_stride;
        let rhs_batch = batch * compact.rhs_batch_stride;
        for col in 0..shape.cols {
            let col = col as isize;
            let rhs_value = OpB::apply(*rhs_ptr.offset(rhs_batch + col * compact.rhs_col_stride));
            let dst_col = dst_batch + col * compact.dst_col_stride;
            for row in 0..shape.rows {
                let row = row as isize;
                let lhs_value =
                    OpA::apply(*lhs_ptr.offset(lhs_batch + row * compact.lhs_row_stride));
                *dst.offset(dst_col + row * compact.dst_row_stride) = lhs_value * rhs_value;
            }
        }
    }
}

unsafe fn execute_strided_seq<T, OpA, OpB>(
    dest: &mut StridedViewMut<T>,
    lhs: &StridedView<T, OpA>,
    rhs: &StridedView<T, OpB>,
    shape: &BatchedOuterShape,
) -> Result<()>
where
    T: Copy + Mul<Output = T>,
    OpA: ElementOp<T>,
    OpB: ElementOp<T>,
{
    let offsets = StridedOuterOffsets::new(dest, lhs, rhs, shape)?;
    let dst = dest.as_mut_ptr();
    let lhs_ptr = lhs.ptr();
    let rhs_ptr = rhs.ptr();
    for batch in 0..shape.batches {
        for col in 0..shape.cols {
            let rhs_value =
                OpB::apply(*rhs_ptr.offset(offsets.rhs_batch[batch] + offsets.rhs_col[col]));
            for row in 0..shape.rows {
                let lhs_value =
                    OpA::apply(*lhs_ptr.offset(offsets.lhs_batch[batch] + offsets.lhs_row[row]));
                *dst.offset(
                    offsets.dst_batch[batch] + offsets.dst_col[col] + offsets.dst_row[row],
                ) = lhs_value * rhs_value;
            }
        }
    }
    Ok(())
}

#[cfg(feature = "parallel")]
unsafe fn execute_compact_par<T, OpA, OpB>(
    dest: &mut StridedViewMut<T>,
    lhs: &StridedView<T, OpA>,
    rhs: &StridedView<T, OpB>,
    shape: &BatchedOuterShape,
    compact: &CompactBatchedOuter,
) where
    T: Copy + Send + Sync + Mul<Output = T>,
    OpA: ElementOp<T> + Send + Sync,
    OpB: ElementOp<T> + Send + Sync,
{
    use rayon::prelude::*;

    let dst = SendPtr(dest.as_mut_ptr());
    let lhs_ptr = SendPtr(lhs.ptr() as *mut T);
    let rhs_ptr = SendPtr(rhs.ptr() as *mut T);
    let cols = shape.cols;
    let rows = shape.rows;
    (0..shape.batches * shape.cols)
        .into_par_iter()
        .for_each(|bc| {
            let batch = (bc / cols) as isize;
            let col = (bc % cols) as isize;
            let dst_batch = batch * compact.dst_batch_stride;
            let lhs_batch = batch * compact.lhs_batch_stride;
            let rhs_batch = batch * compact.rhs_batch_stride;
            let rhs_value = unsafe {
                OpB::apply(
                    *rhs_ptr
                        .as_const()
                        .offset(rhs_batch + col * compact.rhs_col_stride),
                )
            };
            let dst_col = dst_batch + col * compact.dst_col_stride;
            for row in 0..rows {
                let row = row as isize;
                let lhs_value = unsafe {
                    OpA::apply(
                        *lhs_ptr
                            .as_const()
                            .offset(lhs_batch + row * compact.lhs_row_stride),
                    )
                };
                unsafe {
                    *dst.as_ptr().offset(dst_col + row * compact.dst_row_stride) =
                        lhs_value * rhs_value;
                }
            }
        });
}

#[cfg(feature = "parallel")]
unsafe fn execute_strided_par<T, OpA, OpB>(
    dest: &mut StridedViewMut<T>,
    lhs: &StridedView<T, OpA>,
    rhs: &StridedView<T, OpB>,
    shape: &BatchedOuterShape,
) -> Result<()>
where
    T: Copy + Send + Sync + Mul<Output = T>,
    OpA: ElementOp<T> + Send + Sync,
    OpB: ElementOp<T> + Send + Sync,
{
    use rayon::prelude::*;

    let offsets = StridedOuterOffsets::new(dest, lhs, rhs, shape)?;
    let dst = SendPtr(dest.as_mut_ptr());
    let lhs_ptr = SendPtr(lhs.ptr() as *mut T);
    let rhs_ptr = SendPtr(rhs.ptr() as *mut T);
    let cols = shape.cols;
    let rows = shape.rows;
    (0..shape.batches * shape.cols)
        .into_par_iter()
        .for_each(|bc| {
            let batch = bc / cols;
            let col = bc % cols;
            let rhs_value = unsafe {
                OpB::apply(
                    *rhs_ptr
                        .as_const()
                        .offset(offsets.rhs_batch[batch] + offsets.rhs_col[col]),
                )
            };
            for row in 0..rows {
                let lhs_value = unsafe {
                    OpA::apply(
                        *lhs_ptr
                            .as_const()
                            .offset(offsets.lhs_batch[batch] + offsets.lhs_row[row]),
                    )
                };
                unsafe {
                    *dst.as_ptr().offset(
                        offsets.dst_batch[batch] + offsets.dst_col[col] + offsets.dst_row[row],
                    ) = lhs_value * rhs_value;
                }
            }
        });
    Ok(())
}

struct StridedOuterOffsets {
    dst_row: Vec<isize>,
    dst_col: Vec<isize>,
    dst_batch: Vec<isize>,
    lhs_row: Vec<isize>,
    lhs_batch: Vec<isize>,
    rhs_col: Vec<isize>,
    rhs_batch: Vec<isize>,
}

impl StridedOuterOffsets {
    fn new<T, OpA, OpB>(
        dest: &StridedViewMut<T>,
        lhs: &StridedView<T, OpA>,
        rhs: &StridedView<T, OpB>,
        shape: &BatchedOuterShape,
    ) -> Result<Self> {
        Ok(Self {
            dst_row: group_offsets(dest.dims(), dest.strides(), 0, shape.lhs_free_ndim)?,
            dst_col: group_offsets(
                dest.dims(),
                dest.strides(),
                shape.lhs_free_ndim,
                shape.lhs_free_ndim + shape.rhs_free_ndim,
            )?,
            dst_batch: group_offsets(
                dest.dims(),
                dest.strides(),
                shape.lhs_free_ndim + shape.rhs_free_ndim,
                dest.ndim(),
            )?,
            lhs_row: group_offsets(lhs.dims(), lhs.strides(), 0, shape.lhs_free_ndim)?,
            lhs_batch: group_offsets(lhs.dims(), lhs.strides(), shape.lhs_free_ndim, lhs.ndim())?,
            rhs_col: group_offsets(rhs.dims(), rhs.strides(), 0, shape.rhs_free_ndim)?,
            rhs_batch: group_offsets(rhs.dims(), rhs.strides(), shape.rhs_free_ndim, rhs.ndim())?,
        })
    }
}

fn group_offsets(
    dims: &[usize],
    strides: &[isize],
    start: usize,
    end: usize,
) -> Result<Vec<isize>> {
    let group_dims = &dims[start..end];
    let group_strides = &strides[start..end];
    let total = product(group_dims)?;
    let mut offsets = Vec::with_capacity(total);
    if group_dims.is_empty() {
        offsets.push(0);
        return Ok(offsets);
    }
    for mut linear in 0..total {
        let mut offset = 0isize;
        for (&dim, &stride) in group_dims.iter().zip(group_strides.iter()) {
            let idx = linear % dim;
            linear /= dim;
            offset = offset
                .checked_add(
                    (idx as isize)
                        .checked_mul(stride)
                        .ok_or(StridedError::OffsetOverflow)?,
                )
                .ok_or(StridedError::OffsetOverflow)?;
        }
        offsets.push(offset);
    }
    Ok(offsets)
}

#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);

#[cfg(feature = "parallel")]
unsafe impl<T> Send for SendPtr<T> {}
#[cfg(feature = "parallel")]
unsafe impl<T> Sync for SendPtr<T> {}

#[cfg(feature = "parallel")]
impl<T> SendPtr<T> {
    fn as_ptr(self) -> *mut T {
        self.0
    }

    fn as_const(self) -> *const T {
        self.0 as *const T
    }
}
