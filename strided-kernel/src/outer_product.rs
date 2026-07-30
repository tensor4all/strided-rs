//! Semantic outer-product API on dynamic-rank strided views.

use core::mem::MaybeUninit;
use std::ops::Mul;

#[cfg(feature = "parallel")]
use smallvec::SmallVec;

use crate::map_view::{broadcast_mul_into, broadcast_mul_into_uninit};
use crate::maybe_sync::MaybeSendSync;
use crate::view::{StridedView, StridedViewMut};
use crate::{ElementOp, Result, StridedError};

#[cfg(feature = "parallel")]
type AxisVec<T> = SmallVec<[T; 8]>;
#[cfg(not(feature = "parallel"))]
type AxisVec<T> = Vec<T>;

/// Compute `dest[lhs_free..., rhs_free..., batch...] =
/// lhs[lhs_free..., batch...] * rhs[rhs_free..., batch...]`.
///
/// This is a semantic convenience wrapper over [`broadcast_mul_into`]. The
/// broadcast/mul planner owns kernel selection, so explicit outer-product calls
/// and equivalent broadcasted multiplication use the same implementation path.
pub fn batched_outer_product_into<D, A, B, OpA, OpB>(
    dest: &mut StridedViewMut<D>,
    lhs: &StridedView<A, OpA>,
    rhs: &StridedView<B, OpB>,
    lhs_free_ndim: usize,
    rhs_free_ndim: usize,
) -> Result<()>
where
    D: Copy + MaybeSendSync + 'static,
    A: Copy + MaybeSendSync + Mul<B, Output = D> + 'static,
    B: Copy + MaybeSendSync + 'static,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
{
    validate_batched_outer_shape(dest, lhs, rhs, lhs_free_ndim, rhs_free_ndim)?;

    let batch_ndim = lhs.ndim() - lhs_free_ndim;
    let mut lhs_axes = AxisVec::<usize>::with_capacity(lhs.ndim());
    let mut rhs_axes = AxisVec::<usize>::with_capacity(rhs.ndim());

    lhs_axes.extend(0..lhs_free_ndim);
    rhs_axes.extend(lhs_free_ndim..lhs_free_ndim + rhs_free_ndim);

    let batch_axis_start = lhs_free_ndim + rhs_free_ndim;
    lhs_axes.extend(batch_axis_start..batch_axis_start + batch_ndim);
    rhs_axes.extend(batch_axis_start..batch_axis_start + batch_ndim);

    broadcast_mul_into(dest, lhs, &lhs_axes, rhs, &rhs_axes)
}

/// Compute a batched outer product into a fully overwritten uninitialized output.
///
/// Rank, shape, destination-injectivity, and reachable-byte overlap validation
/// completes before the first write. Safe Rust borrows already prevent
/// input/output aliasing; the explicit overlap check in the shared broadcast
/// kernel preserves the contract for views produced through unsafe constructors.
///
/// `Ok(())` means every logical destination element is initialized. An error
/// occurs before writes. A panic during replay may leave a partially initialized
/// destination, which remains safe to drop as `MaybeUninit<D>`.
///
/// # Errors
///
/// Returns a typed rank or shape error for incompatible free/batch dimensions,
/// [`StridedError::NonInjectiveOutputLayout`] for an overlapping output layout,
/// [`StridedError::OverlappingInputOutput`] for aliased storage, or
/// [`StridedError::OffsetOverflow`] when a reachable byte range is not
/// representable.
pub fn batched_outer_product_into_uninit<D, A, B, OpA, OpB>(
    dest: &mut StridedViewMut<MaybeUninit<D>>,
    lhs: &StridedView<A, OpA>,
    rhs: &StridedView<B, OpB>,
    lhs_free_ndim: usize,
    rhs_free_ndim: usize,
) -> Result<()>
where
    D: Copy + MaybeSendSync + 'static,
    A: Copy + MaybeSendSync + Mul<B, Output = D> + 'static,
    B: Copy + MaybeSendSync + 'static,
    OpA: ElementOp<A>,
    OpB: ElementOp<B>,
{
    validate_batched_outer_shape(dest, lhs, rhs, lhs_free_ndim, rhs_free_ndim)?;
    let batch_ndim = lhs.ndim() - lhs_free_ndim;
    let mut lhs_axes = AxisVec::<usize>::with_capacity(lhs.ndim());
    let mut rhs_axes = AxisVec::<usize>::with_capacity(rhs.ndim());
    lhs_axes.extend(0..lhs_free_ndim);
    rhs_axes.extend(lhs_free_ndim..lhs_free_ndim + rhs_free_ndim);
    let batch_axis_start = lhs_free_ndim + rhs_free_ndim;
    lhs_axes.extend(batch_axis_start..batch_axis_start + batch_ndim);
    rhs_axes.extend(batch_axis_start..batch_axis_start + batch_ndim);
    broadcast_mul_into_uninit(dest, lhs, &lhs_axes, rhs, &rhs_axes)
}

fn validate_batched_outer_shape<D, A, OpA, B, OpB>(
    dest: &StridedViewMut<D>,
    lhs: &StridedView<A, OpA>,
    rhs: &StridedView<B, OpB>,
    lhs_free_ndim: usize,
    rhs_free_ndim: usize,
) -> Result<()> {
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

    Ok(())
}

fn ensure_dims(actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(StridedError::ShapeMismatch(
            actual.to_vec(),
            expected.to_vec(),
        ))
    }
}
