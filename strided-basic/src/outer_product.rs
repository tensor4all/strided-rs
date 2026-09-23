//! Semantic outer-product API on dynamic-rank strided views.

use core::mem::MaybeUninit;
use std::ops::Mul;

#[cfg(feature = "parallel")]
use smallvec::SmallVec;

use crate::view::{StridedView, StridedViewMut};
use crate::MaybeSendSync;
use crate::{broadcast_mul_into, broadcast_mul_into_uninit};
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

/// Storage layout for a lazily ordered outer product.
///
/// Returned by [`plan_lazy_outer_product`]. The caller allocates a dense
/// column-major base of [`base_dims`](Self::base_dims), fills it through a
/// destination descriptor over that base with the logical output dims and
/// [`output_strides`](Self::output_strides) at offset zero (for example with
/// [`broadcast_mul_into_uninit`] and the original axis maps), and then exposes
/// the result as a strided view with those same dims and strides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LazyOuterProductLayout {
    /// Column-major extents of the dense base allocation.
    pub base_dims: Vec<usize>,
    /// Strides, in base elements, of the logical output axes over the base.
    pub output_strides: Vec<isize>,
}

/// Plan an outer-product output whose memory order follows the inputs'
/// physical stride order instead of the logical output order.
///
/// `lhs_axes[k]` (respectively `rhs_axes[k]`) names the output axis that
/// operand axis `k` maps to, with the operand extent equal to the output
/// extent. Output axes mapped only by `lhs` are its free axes, axes mapped
/// only by `rhs` are its free axes, and axes mapped by both are batch axes.
///
/// A layout is returned only when the product splits into free and batch
/// groups, both free groups have more than one element, the logical output
/// order is `[lhs_free, rhs_free, batch]` or `[rhs_free, lhs_free, batch]`,
/// all strides are non-negative, and sorting either operand's free axes by
/// `(stride, axis)` changes their order. The base then holds the leading
/// group's free axes in the leading operand's physical order, then the
/// trailing group's free axes in the trailing operand's physical order, then
/// the batch axes in output order. Traversing inputs in their physical order
/// while writing the base contiguously is what makes the layout useful.
///
/// # Examples
///
/// ```
/// use strided_basic::plan_lazy_outer_product;
///
/// // lhs is a transposed 2 x 3 matrix (row-major strides), rhs is a vector.
/// let layout = plan_lazy_outer_product(&[2, 3, 4], &[2, 3], &[3, 1], &[0, 1], &[4], &[1], &[2])
///     .unwrap()
///     .unwrap();
/// assert_eq!(layout.base_dims, [3, 2, 4]);
/// assert_eq!(layout.output_strides, [3, 1, 6]);
/// ```
///
/// # Errors
///
/// Returns [`StridedError::RankMismatch`] when an operand's dims, strides,
/// and axis map lengths disagree, and [`StridedError::OffsetOverflow`] when a
/// base extent product or stride is not representable. Inputs that are valid
/// but ineligible return `Ok(None)`.
pub fn plan_lazy_outer_product(
    output_dims: &[usize],
    lhs_dims: &[usize],
    lhs_strides: &[isize],
    lhs_axes: &[usize],
    rhs_dims: &[usize],
    rhs_strides: &[isize],
    rhs_axes: &[usize],
) -> Result<Option<LazyOuterProductLayout>> {
    for (dims, strides, axes) in [
        (lhs_dims, lhs_strides, lhs_axes),
        (rhs_dims, rhs_strides, rhs_axes),
    ] {
        if dims.len() != strides.len() {
            return Err(StridedError::RankMismatch(dims.len(), strides.len()));
        }
        if dims.len() != axes.len() {
            return Err(StridedError::RankMismatch(dims.len(), axes.len()));
        }
    }
    if !extents_match_output(output_dims, lhs_dims, lhs_axes)
        || !extents_match_output(output_dims, rhs_dims, rhs_axes)
        || lhs_strides
            .iter()
            .chain(rhs_strides)
            .any(|&stride| stride < 0)
    {
        return Ok(None);
    }
    let Some(partition) = classify_outer_axes(output_dims.len(), lhs_axes, rhs_axes) else {
        return Ok(None);
    };
    if checked_axes_product(lhs_dims, &partition.lhs_free)? <= 1
        || checked_axes_product(rhs_dims, &partition.rhs_free)? <= 1
    {
        return Ok(None);
    }

    let output_rank = output_dims.len();
    let lhs_prefix = partition
        .lhs_free_out
        .iter()
        .chain(&partition.rhs_free_out)
        .chain(&partition.batch_out)
        .copied()
        .eq(0..output_rank);
    let rhs_prefix = !lhs_prefix
        && partition
            .rhs_free_out
            .iter()
            .chain(&partition.lhs_free_out)
            .chain(&partition.batch_out)
            .copied()
            .eq(0..output_rank);
    if !lhs_prefix && !rhs_prefix {
        return Ok(None);
    }

    let lhs_physical = axes_by_physical_stride(lhs_strides, &partition.lhs_free);
    let rhs_physical = axes_by_physical_stride(rhs_strides, &partition.rhs_free);
    if lhs_physical == partition.lhs_free && rhs_physical == partition.rhs_free {
        return Ok(None);
    }

    // Base axis order: leading free, trailing free, then batch in output order.
    let mut base_out_axes = Vec::with_capacity(output_rank);
    let (leading, leading_axes, trailing, trailing_axes) = if lhs_prefix {
        (&lhs_physical, lhs_axes, &rhs_physical, rhs_axes)
    } else {
        (&rhs_physical, rhs_axes, &lhs_physical, lhs_axes)
    };
    base_out_axes.extend(leading.iter().map(|&axis| leading_axes[axis]));
    base_out_axes.extend(trailing.iter().map(|&axis| trailing_axes[axis]));
    base_out_axes.extend(partition.batch_out.iter().copied());

    let base_dims: Vec<usize> = base_out_axes
        .iter()
        .map(|&axis| output_dims[axis])
        .collect();
    let mut output_strides = vec![0isize; output_rank];
    let mut stride = 1isize;
    for (&out_axis, &extent) in base_out_axes.iter().zip(&base_dims) {
        // INVARIANT: `classify_outer_axes` proved every output axis appears
        // exactly once in the free/batch groups, so each slot is written once.
        output_strides[out_axis] = stride;
        let extent = isize::try_from(extent).map_err(|_| StridedError::OffsetOverflow)?;
        stride = stride
            .checked_mul(extent)
            .ok_or(StridedError::OffsetOverflow)?;
    }
    Ok(Some(LazyOuterProductLayout {
        base_dims,
        output_strides,
    }))
}

struct OuterAxisPartition {
    lhs_free_out: Vec<usize>,
    rhs_free_out: Vec<usize>,
    batch_out: Vec<usize>,
    lhs_free: Vec<usize>,
    rhs_free: Vec<usize>,
}

fn extents_match_output(output_dims: &[usize], dims: &[usize], axes: &[usize]) -> bool {
    dims.iter()
        .zip(axes)
        .all(|(&dim, &axis)| output_dims.get(axis) == Some(&dim))
}

fn operand_axes_by_output(axes: &[usize], output_rank: usize) -> Option<Vec<Option<usize>>> {
    let mut by_output = vec![None; output_rank];
    for (operand_axis, &output_axis) in axes.iter().enumerate() {
        if by_output
            .get_mut(output_axis)?
            .replace(operand_axis)
            .is_some()
        {
            return None;
        }
    }
    Some(by_output)
}

fn classify_outer_axes(
    output_rank: usize,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
) -> Option<OuterAxisPartition> {
    let lhs_by_output = operand_axes_by_output(lhs_axes, output_rank)?;
    let rhs_by_output = operand_axes_by_output(rhs_axes, output_rank)?;
    let mut partition = OuterAxisPartition {
        lhs_free_out: Vec::new(),
        rhs_free_out: Vec::new(),
        batch_out: Vec::new(),
        lhs_free: Vec::new(),
        rhs_free: Vec::new(),
    };
    for output_axis in 0..output_rank {
        match (lhs_by_output[output_axis], rhs_by_output[output_axis]) {
            (Some(_), Some(_)) => partition.batch_out.push(output_axis),
            (Some(lhs_axis), None) => {
                partition.lhs_free_out.push(output_axis);
                partition.lhs_free.push(lhs_axis);
            }
            (None, Some(rhs_axis)) => {
                partition.rhs_free_out.push(output_axis);
                partition.rhs_free.push(rhs_axis);
            }
            (None, None) => return None,
        }
    }
    Some(partition)
}

fn checked_axes_product(dims: &[usize], axes: &[usize]) -> Result<usize> {
    axes.iter().try_fold(1usize, |acc, &axis| {
        acc.checked_mul(dims[axis])
            .ok_or(StridedError::OffsetOverflow)
    })
}

fn axes_by_physical_stride(strides: &[isize], axes: &[usize]) -> Vec<usize> {
    let mut sorted = axes.to_vec();
    sorted.sort_by(|&lhs, &rhs| strides[lhs].cmp(&strides[rhs]).then(lhs.cmp(&rhs)));
    sorted
}
