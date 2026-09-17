//! Dense column-major update and structural kernels.
//!
//! AXPBY and triangular masking are adapted from tenferro-rs d8759f43,
//! `crates/tenferro-cpu/src/{blas1,structural}.rs` (MIT OR Apache-2.0).
//! Tensor allocation/placement stay with the caller. Diagonal embedding uses
//! the existing strided map traversal rather than tenferro's coordinate loop.

use core::mem::MaybeUninit;
use core::ops::{Add, Mul};

use crate::{MaybeSendSync, Result, StridedError, StridedView, StridedViewMut};

fn same_len(actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(StridedError::ShapeMismatch(vec![actual], vec![expected]))
    }
}

fn product(shape: &[usize]) -> Result<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d).ok_or(StridedError::OffsetOverflow)
    })
}

fn for_each_chunk<T: MaybeSendSync, F: Fn(usize, &mut [T]) + crate::MaybeSync>(
    output: &mut [T],
    operation: F,
) {
    #[cfg(feature = "parallel")]
    {
        let threads = crate::threading::parallel_threads_for_len(output.len());
        if threads > 1 {
            let ptr = crate::threading::SendPtr(output.as_mut_ptr());
            crate::threading::parallel_for_each(0..output.len(), threads, &|range| {
                // SAFETY: the scoped scheduler partitions the exclusive output
                // borrow into disjoint ranges; all tasks finish before return.
                let chunk = unsafe {
                    core::slice::from_raw_parts_mut(ptr.as_ptr().add(range.start), range.len())
                };
                operation(range.start, chunk);
            });
            return;
        }
    }
    operation(0, output);
}

/// Update caller-owned contiguous storage: `y = alpha * x + beta * y`.
///
/// This is an accumulation operation: it reads the old `y`, even for beta=0
/// (including ordinary IEEE NaN propagation). No temporary tensor is allocated.
/// Execution obeys the active execution policy and shared parallel threshold.
///
/// # Examples
/// ```
/// let mut y = [3.0, 4.0];
/// strided_basic::axpby_accum(&mut y, &[1.0, 2.0], 2.0, 3.0).unwrap();
/// assert_eq!(y, [11.0, 16.0]);
/// ```
/// # Errors
/// Returns `ShapeMismatch` for unequal slice lengths, before modifying `y`.
pub fn axpby_accum<T>(y: &mut [T], x: &[T], alpha: T, beta: T) -> Result<()>
where
    T: Copy + Send + Sync + Add<Output = T> + Mul<Output = T>,
{
    same_len(x.len(), y.len())?;
    for_each_chunk(y, |start, dst| {
        let len = dst.len();
        for (out, &src) in dst.iter_mut().zip(&x[start..start + len]) {
            *out = alpha * src + beta * *out;
        }
    });
    Ok(())
}

/// Copy dense column-major matrices, replacing the masked triangle with `fill`.
///
/// The first two dimensions are rows and columns; remaining dimensions are
/// batches. `upper=true` keeps `row <= column-k`, otherwise the lower triangle.
/// Every output element is initialized on success; old output is never read.
///
/// # Examples
/// ```
/// use core::mem::MaybeUninit;
/// let mut out = [MaybeUninit::uninit(); 4];
/// strided_basic::triangular_mask_into_uninit(
///     &mut out, &[1, 2, 3, 4], &[2, 2], 0, false, 0).unwrap();
/// // SAFETY: successful full-overwrite kernel initialized every element.
/// assert_eq!(out.map(|v| unsafe { v.assume_init() }), [1, 2, 0, 4]);
/// ```
/// # Errors
/// Returns `RankMismatch`, `ShapeMismatch`, or `OffsetOverflow` for invalid
/// shape/buffer lengths, before writing output.
pub fn triangular_mask_into_uninit<T: Copy + MaybeSendSync>(
    output: &mut [MaybeUninit<T>],
    input: &[T],
    shape: &[usize],
    k: i64,
    upper: bool,
    fill: T,
) -> Result<()> {
    if shape.len() < 2 {
        return Err(StridedError::RankMismatch(shape.len(), 2));
    }
    let len = product(shape)?;
    same_len(input.len(), len)?;
    same_len(output.len(), len)?;
    if len == 0 {
        return Ok(());
    }
    let rows = shape[0];
    let cols = shape[1];
    for_each_chunk(output, |start, chunk| {
        // Preserve the original copy-then-mask algorithm for the migration
        // baseline; the benchmark suite compares subsequent loop changes.
        let len = chunk.len();
        for (dst, &src) in chunk.iter_mut().zip(&input[start..start + len]) {
            dst.write(src);
        }
        let mut pos = 0;
        while pos < chunk.len() {
            let flat = start + pos;
            let row = flat % rows;
            let col = (flat / rows) % cols;
            let count = (rows - row).min(chunk.len() - pos);
            let boundary = col as i128 - k as i128;
            let (lo, hi) = if upper {
                (
                    boundary.saturating_add(1).clamp(0, rows as i128) as usize,
                    rows,
                )
            } else {
                (0, boundary.clamp(0, rows as i128) as usize)
            };
            let begin = lo.max(row);
            let end = hi.min(row + count);
            if begin < end {
                chunk[pos + begin - row..pos + end - row].fill(MaybeUninit::new(fill));
            }
            pos += count;
        }
    });
    Ok(())
}

/// Embed a dense column-major input on a diagonal in a new axis.
///
/// Insert an axis of size `shape[axis]` at `insert_axis`. Its coordinate must
/// equal the original `axis` coordinate; other entries receive `zero`.
/// Output is fully initialized without reading its previous contents. The
/// active execution policy governs fill and strided diagonal-copy traversal.
///
/// # Examples
/// ```
/// use core::mem::MaybeUninit;
/// let mut out = [MaybeUninit::uninit(); 4];
/// strided_basic::embed_diagonal_into_uninit(&mut out, &[3, 5], &[2], 0, 1, 0).unwrap();
/// // SAFETY: successful full-overwrite kernel initialized every element.
/// assert_eq!(out.map(|v| unsafe { v.assume_init() }), [3, 0, 0, 5]);
/// ```
/// # Errors
/// Returns `InvalidAxis`, `ShapeMismatch`, or `OffsetOverflow` for invalid
/// axes, lengths or unrepresentable layouts. Destination layout validation
/// may return `NonInjectiveOutputLayout` if injectivity cannot be established.
pub fn embed_diagonal_into_uninit<T: Copy + MaybeSendSync + 'static>(
    output: &mut [MaybeUninit<T>],
    input: &[T],
    shape: &[usize],
    axis: usize,
    insert_axis: usize,
    zero: T,
) -> Result<()> {
    if axis >= shape.len() {
        return Err(StridedError::InvalidAxis {
            axis,
            rank: shape.len(),
        });
    }
    if insert_axis > shape.len() {
        return Err(StridedError::InvalidAxis {
            axis: insert_axis,
            rank: shape.len() + 1,
        });
    }
    let len = product(shape)?;
    same_len(input.len(), len)?;
    let out_len = len
        .checked_mul(shape[axis])
        .ok_or(StridedError::OffsetOverflow)?;
    same_len(output.len(), out_len)?;
    if out_len == 0 {
        return Ok(());
    }
    isize::try_from(out_len).map_err(|_| StridedError::OffsetOverflow)?;
    let mut out_shape = shape.to_vec();
    out_shape.insert(insert_axis, shape[axis]);
    // INVARIANT: nonempty products fit isize, so every prefix stride fits too.
    let src_strides = crate::col_major_strides(shape);
    let mut dst_strides = crate::col_major_strides(&out_shape);
    let inserted_stride = dst_strides.remove(insert_axis);
    dst_strides[axis] = dst_strides[axis]
        .checked_add(inserted_stride)
        .ok_or(StridedError::OffsetOverflow)?;
    let src = StridedView::<T>::new(input, shape, &src_strides, 0)?;
    let mut dst = StridedViewMut::new(output, shape, &dst_strides, 0)?;
    crate::map_view::validate_destination_layout_without_alloc(shape, &dst_strides)?;
    // INVARIANT: off-diagonal zeros are part of the mathematical result, not
    // scratch initialization. Only the diagonal subset is overwritten below.
    for_each_chunk(dst.data_mut(), |_, chunk| {
        chunk.fill(MaybeUninit::new(zero))
    });
    crate::map_into(&mut dst, &src, MaybeUninit::new)
}
