//! Dense column-major update and structural kernels.
//!
//! AXPBY and triangular masking are adapted from tenferro-rs d8759f43,
//! `crates/tenferro-cpu/src/{blas1,structural}.rs` (MIT OR Apache-2.0).
//! Tensor allocation/placement stay with the caller. Diagonal embedding uses
//! the existing strided map traversal rather than tenferro's coordinate loop.

use core::mem::MaybeUninit;
use core::ops::{Add, Mul};

use crate::cost::KernelClass;
use crate::{Exec, Result, StridedError, StridedView, StridedViewMut};

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

fn for_each_chunk<T: Send + Sync, F: Fn(usize, &mut [T]) + Sync>(
    output: &mut [T],
    operation: F,
    exec: Exec<'_>,
) {
    let exec = crate::cost::for_len(exec, KernelClass::Map, output.len());
    if !exec.is_serial() {
        let ptr = crate::threading::SendPtr(output.as_mut_ptr());
        crate::threading::parallel_for_each(0..output.len(), exec, &|range| {
            // SAFETY: the scoped scheduler partitions the exclusive output
            // borrow into disjoint ranges; all tasks finish before return.
            let chunk = unsafe {
                core::slice::from_raw_parts_mut(ptr.as_ptr().add(range.start), range.len())
            };
            operation(range.start, chunk);
        });
        return;
    }
    operation(0, output);
}

/// Source offset of a linear batch index within `shape[2..]`.
fn batch_source_offset(mut batch: usize, shape: &[usize], strides: &[isize]) -> isize {
    let mut offset = 0isize;
    for (&dim, &stride) in shape.iter().zip(strides) {
        let coord = batch % dim;
        batch /= dim;
        offset += coord as isize * stride;
    }
    offset
}

/// Borrow `data` as a dense column-major view of `shape`.
fn dense_view<'a, T>(data: &'a [T], shape: &[usize]) -> Result<StridedView<'a, T>> {
    let len = product(shape)?;
    same_len(data.len(), len)?;
    let strides = crate::col_major_strides(shape);
    StridedView::new(data, shape, &strides, 0)
}

/// Copy `dst.len()` source elements starting at reachable byte-free offset
/// `base` with element step `stride`.
fn copy_strided_run<T: Copy>(dst: &mut [MaybeUninit<T>], data: &[T], base: isize, stride: isize) {
    if stride == 1 {
        // INVARIANT: `base` is reachable inside a view-validated buffer, so
        // the run start plus its length stays in bounds.
        let start = base as usize;
        let src = &data[start..start + dst.len()];
        for (slot, &value) in dst.iter_mut().zip(src) {
            slot.write(value);
        }
    } else {
        let mut offset = base;
        for slot in dst.iter_mut() {
            slot.write(data[offset as usize]);
            offset += stride;
        }
    }
}

/// Update caller-owned contiguous storage: `y = alpha * x + beta * y`.
///
/// This is an accumulation operation: it reads the old `y`, even for beta=0
/// (including ordinary IEEE NaN propagation). No temporary tensor is allocated.
/// The update splits within the budget of `exec` above the shared parallel
/// threshold.
///
/// # Examples
/// ```
/// let mut y = [3.0, 4.0];
/// strided_basic::axpby_accum(&mut y, &[1.0, 2.0], 2.0, 3.0, strided_basic::Exec::serial()).unwrap();
/// assert_eq!(y, [11.0, 16.0]);
/// ```
/// # Errors
/// Returns `ShapeMismatch` for unequal slice lengths, before modifying `y`.
pub fn axpby_accum<T>(y: &mut [T], x: &[T], alpha: T, beta: T, exec: Exec<'_>) -> Result<()>
where
    T: Copy + Send + Sync + Add<Output = T> + Mul<Output = T>,
{
    same_len(x.len(), y.len())?;
    for_each_chunk(
        y,
        |start, dst| {
            let len = dst.len();
            for (out, &src) in dst.iter_mut().zip(&x[start..start + len]) {
                *out = alpha * src + beta * *out;
            }
        },
        exec,
    );
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
///     &mut out, &[1, 2, 3, 4], &[2, 2], 0, false, 0, strided_basic::Exec::serial()).unwrap();
/// // SAFETY: successful full-overwrite kernel initialized every element.
/// assert_eq!(out.map(|v| unsafe { v.assume_init() }), [1, 2, 0, 4]);
/// ```
/// # Errors
/// Returns `RankMismatch`, `ShapeMismatch`, or `OffsetOverflow` for invalid
/// shape/buffer lengths, before writing output.
pub fn triangular_mask_into_uninit<T: Copy + Send + Sync>(
    output: &mut [MaybeUninit<T>],
    input: &[T],
    shape: &[usize],
    k: i64,
    upper: bool,
    fill: T,
    exec: Exec<'_>,
) -> Result<()> {
    let src = dense_view(input, shape)?;
    triangular_mask_view_into_uninit(output, &src, k, upper, fill, exec)
}

/// Mask a triangular region of a strided input into caller-owned contiguous
/// storage.
///
/// The source layout is given by `src`; reads follow its strides and offset, so
/// a reversed, transposed or otherwise strided source needs no staging copy.
/// Kept elements come from `src`; masked elements receive `fill`. Every output
/// element is written exactly once and the masked region of the source is
/// never read.
///
/// # Errors
/// Returns `RankMismatch` or `ShapeMismatch` for invalid shapes and buffer
/// lengths, and `OffsetOverflow` for unrepresentable layouts, before writing.
pub fn triangular_mask_view_into_uninit<T: Copy + Send + Sync>(
    output: &mut [MaybeUninit<T>],
    src: &StridedView<'_, T>,
    k: i64,
    upper: bool,
    fill: T,
    exec: Exec<'_>,
) -> Result<()> {
    let shape = src.dims();
    if shape.len() < 2 {
        return Err(StridedError::RankMismatch(shape.len(), 2));
    }
    let len = product(shape)?;
    same_len(output.len(), len)?;
    if len == 0 {
        return Ok(());
    }
    let rows = shape[0];
    let cols = shape[1];
    let data = src.data();
    let strides = src.strides();
    let offset = src.offset();
    for_each_chunk(
        output,
        |start, chunk| {
            let mut pos = 0;
            let mut batch = usize::MAX;
            let mut batch_offset = 0isize;
            while pos < chunk.len() {
                let flat = start + pos;
                let row = flat % rows;
                let col = (flat / rows) % cols;
                // INVARIANT: `product(shape)` succeeded, so `rows * cols` fits.
                let run_batch = flat / (rows * cols);
                if run_batch != batch {
                    batch = run_batch;
                    batch_offset = batch_source_offset(run_batch, &shape[2..], &strides[2..]);
                }
                let count = (rows - row).min(chunk.len() - pos);
                let boundary = col as i128 - k as i128;
                let split = if upper {
                    boundary.saturating_add(1)
                } else {
                    boundary
                }
                .clamp(row as i128, (row + count) as i128) as usize
                    - row;
                let (low, high) = chunk[pos..pos + count].split_at_mut(split);
                let (kept, first_row, masked) = if upper {
                    (low, row, high)
                } else {
                    (high, row + split, low)
                };
                let base = offset
                    + batch_offset
                    + (col as isize) * strides[1]
                    + (first_row as isize) * strides[0];
                // Full overwrite: never read masked input or copy it only to
                // overwrite it a second time. Each output element is written once.
                copy_strided_run(kept, data, base, strides[0]);
                masked.fill(MaybeUninit::new(fill));
                pos += count;
            }
        },
        exec,
    );
    Ok(())
}

/// Embed a dense column-major input on a diagonal in a new axis.
///
/// Insert an axis of size `shape[axis]` at `insert_axis`. Its coordinate must
/// equal the original `axis` coordinate; other entries receive `zero`.
/// Output is fully initialized without reading its previous contents. The
/// budget of `exec` governs fill and strided diagonal-copy traversal.
///
/// # Examples
/// ```
/// use core::mem::MaybeUninit;
/// let mut out = [MaybeUninit::uninit(); 4];
/// strided_basic::embed_diagonal_into_uninit(&mut out, &[3, 5], &[2], 0, 1, 0, strided_basic::Exec::serial()).unwrap();
/// // SAFETY: successful full-overwrite kernel initialized every element.
/// assert_eq!(out.map(|v| unsafe { v.assume_init() }), [3, 0, 0, 5]);
/// ```
/// # Errors
/// Returns `InvalidAxis`, `ShapeMismatch`, or `OffsetOverflow` for invalid
/// axes, lengths or unrepresentable layouts. Destination layout validation
/// may return `NonInjectiveOutputLayout` if injectivity cannot be established.
pub fn embed_diagonal_into_uninit<T: Copy + Send + Sync + 'static>(
    output: &mut [MaybeUninit<T>],
    input: &[T],
    shape: &[usize],
    axis: usize,
    insert_axis: usize,
    zero: T,
    exec: Exec<'_>,
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
    let src = dense_view(input, shape)?;
    embed_diagonal_view_into_uninit(output, &src, axis, insert_axis, zero, exec)
}

/// Embed a strided input on a diagonal in a new axis of caller-owned
/// contiguous storage.
///
/// The source layout is given by `src`; the diagonal copy follows its strides
/// and offset, so a strided source needs no staging copy. Off-diagonal entries
/// receive `zero`, which is part of the result rather than scratch
/// initialization.
///
/// # Errors
/// Returns `InvalidAxis`, `ShapeMismatch`, or `OffsetOverflow` for invalid
/// axes, lengths or unrepresentable layouts. Destination layout validation may
/// return `NonInjectiveOutputLayout` if injectivity cannot be established.
pub fn embed_diagonal_view_into_uninit<T: Copy + Send + Sync + 'static>(
    output: &mut [MaybeUninit<T>],
    src: &StridedView<'_, T>,
    axis: usize,
    insert_axis: usize,
    zero: T,
    exec: Exec<'_>,
) -> Result<()> {
    let mut exec = exec;
    let shape = src.dims();
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
    let out_len = len
        .checked_mul(shape[axis])
        .ok_or(StridedError::OffsetOverflow)?;
    same_len(output.len(), out_len)?;
    if out_len == 0 {
        return Ok(());
    }
    isize::try_from(out_len).map_err(|_| StridedError::OffsetOverflow)?;
    let mut out_shape: crate::InlineVec<usize> = shape.iter().copied().collect();
    out_shape.insert(insert_axis, shape[axis]);
    let mut dst_strides = crate::col_major_strides(&out_shape);
    let inserted_stride = dst_strides.remove(insert_axis);
    dst_strides[axis] = dst_strides[axis]
        .checked_add(inserted_stride)
        .ok_or(StridedError::OffsetOverflow)?;
    let mut dst = StridedViewMut::new(output, shape, &dst_strides, 0)?;
    crate::map_view::validate_destination_layout_without_alloc(shape, &dst_strides)?;
    // INVARIANT: off-diagonal zeros are part of the mathematical result, not
    // scratch initialization. Only the diagonal subset is overwritten below.
    for_each_chunk(
        dst.data_mut(),
        |_, chunk| chunk.fill(MaybeUninit::new(zero)),
        exec.reborrow(),
    );
    crate::map_into(&mut dst, src, MaybeUninit::new, exec)
}
