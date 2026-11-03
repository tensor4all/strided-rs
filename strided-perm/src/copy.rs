//! Copy/permutation operations on strided views.

use crate::fuse::FusionPlanError;
use crate::hptt::{build_permute_plan, execute_permute_blocked};
use strided_view::{Result, StridedError, StridedView, StridedViewMut};
use threadteam::Exec;

/// Map a permutation planning failure onto the shared view error vocabulary.
///
/// Rank and shape are validated before planning, so a length mismatch cannot
/// reach this point; it still maps to a typed error instead of a panic. An
/// oversized (typically stride-0 broadcast) extent whose fused product cannot
/// be represented is reported as an offset overflow (issue #263).
fn plan_error(err: FusionPlanError) -> StridedError {
    match err {
        FusionPlanError::LengthMismatch { .. } => StridedError::StrideLengthMismatch,
        FusionPlanError::DimensionOverflow => StridedError::OffsetOverflow,
    }
}

fn total_len(dims: &[usize]) -> usize {
    dims.iter().product()
}

/// Check if all strides indicate contiguous column-major or row-major layout.
fn is_both_contiguous(dims: &[usize], dst_strides: &[isize], src_strides: &[isize]) -> bool {
    if dims.is_empty() {
        return true;
    }

    // Check col-major for both
    let mut expected = 1isize;
    let mut col_ok = true;
    for (&d, (&ds, &ss)) in dims.iter().zip(dst_strides.iter().zip(src_strides.iter())) {
        if d <= 1 {
            continue;
        }
        if ds != expected || ss != expected {
            col_ok = false;
            break;
        }
        expected = expected.saturating_mul(d as isize);
    }
    if col_ok {
        return true;
    }

    // Check row-major for both
    let mut expected = 1isize;
    let mut row_ok = true;
    for (&d, (&ds, &ss)) in dims
        .iter()
        .rev()
        .zip(dst_strides.iter().rev().zip(src_strides.iter().rev()))
    {
        if d <= 1 {
            continue;
        }
        if ds != expected || ss != expected {
            row_ok = false;
            break;
        }
        expected = expected.saturating_mul(d as isize);
    }
    row_ok
}

/// The validated raw geometry of one copy: the shared dims, both stride
/// runs and the two origins. `U` is the destination element type: `T` for
/// an initialized destination, `MaybeUninit<T>` for a fresh one.
struct RawCopy<'a, T, U> {
    dims: &'a [usize],
    dst_strides: &'a [isize],
    src_strides: &'a [isize],
    dst: *mut U,
    src: *const T,
}

/// Validates rank and shape and extracts the raw geometry; `None` for a
/// zero-sized copy, which has nothing to write.
///
/// A zero-sized view validates without its other extents or strides forming
/// an in-range product, so neither the contiguous fast path nor the
/// permutation planner may see it.
fn prepare<'a, T, U>(
    dest: &'a mut StridedViewMut<U>,
    src: &'a StridedView<T>,
) -> Result<Option<RawCopy<'a, T, U>>> {
    let dst_dims = dest.dims();
    let src_dims = src.dims();
    if dst_dims.len() != src_dims.len() {
        return Err(StridedError::RankMismatch(dst_dims.len(), src_dims.len()));
    }
    if dst_dims != src_dims {
        return Err(StridedError::ShapeMismatch(
            dst_dims.to_vec(),
            src_dims.to_vec(),
        ));
    }
    if dst_dims.contains(&0) {
        return Ok(None);
    }
    Ok(Some(RawCopy {
        dims: dst_dims,
        dst_strides: dest.strides(),
        src_strides: src.strides(),
        dst: dest.as_mut_ptr(),
        src: src.ptr(),
    }))
}

/// The copy over validated raw geometry, run within the budget of `exec`.
///
/// # Safety
///
/// `raw.dst` must be valid for writes of `T` at every destination offset and
/// `raw.src` valid for reads of initialized `T` at every source offset; the
/// two ranges must not overlap. Every store goes through raw pointers, so the
/// destination may be uninitialized storage.
unsafe fn copy_raw<T: Copy + Send + Sync + 'static, U>(
    raw: RawCopy<'_, T, U>,
    exec: Exec<'_>,
) -> Result<()> {
    let dst_ptr = raw.dst.cast::<T>();
    // The width is the token's decision: the serial time of the copy against the
    // dispatch cost of the one section the engine opens (`for_ranges`).
    let exec = match raw.dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) {
        Some(n) => {
            let estimate =
                std::time::Duration::from_secs_f64(n as f64 * crate::copy_ns_per_element() * 1e-9);
            let width = exec.width_for(estimate, 1);
            exec.limit(width)
        }
        // An overflowing count is reported by planning below; run it serially.
        None => exec.limit(std::num::NonZeroUsize::MIN),
    };
    // Fast path: both contiguous. A contiguous layout is injective by
    // construction, so it skips the validation below.
    if is_both_contiguous(raw.dims, raw.dst_strides, raw.src_strides) {
        let len = total_len(raw.dims);
        if exec.is_serial() {
            // SAFETY: the validated contiguous ranges are live and non-overlapping.
            unsafe { std::ptr::copy_nonoverlapping(raw.src, dst_ptr, len) };
        } else {
            let src_addr = raw.src as usize;
            let dst_addr = dst_ptr as usize;
            exec.for_ranges(len, std::num::NonZeroUsize::MIN, |range| {
                // SAFETY: `for_ranges` joins disjoint subranges of the validated
                // contiguous buffers; raw stores also support uninitialized output.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (src_addr as *const T).add(range.start),
                        (dst_addr as *mut T).add(range.start),
                        range.len(),
                    );
                }
            });
        }
        return Ok(());
    }

    // HPTT-inspired blocked permutation. Planning runs before the injectivity
    // check so an unrepresentable layout keeps reporting `OffsetOverflow`
    // (issue #263) even when its destination is also non-injective; planning
    // only builds metadata and writes nothing.
    let elem_size = std::mem::size_of::<T>();
    let plan = build_permute_plan(raw.dims, raw.src_strides, raw.dst_strides, elem_size)
        .map_err(plan_error)?;
    // A non-injective destination maps distinct logical elements onto one
    // offset, so the unsafe write loop would produce overlapping writes (and a
    // data race when `exec` fans out). This matches the sibling output-writing
    // APIs (issue #281). The check is global over the destination, so two
    // worker partitions (disjoint logical positions) can never write one
    // offset; no extra per-partition separation proof is needed.
    if !strided_view::is_injective_layout(raw.dims, raw.dst_strides) {
        return Err(StridedError::NonInjectiveOutputLayout);
    }
    execute_permute_blocked(raw.src, dst_ptr, &plan, exec);
    Ok(())
}

/// Copy elements from source to destination: `dest[i] = src[i]`.
///
/// Uses HPTT-inspired blocked permutation with bilateral dimension fusion,
/// cache-aware blocking, and optimal loop ordering. The permutation is split
/// across the budget of `exec`, at the width the token's dispatch cost
/// justifies for the copy ([`copy_ns_per_element`](crate::copy_ns_per_element)
/// times its elements); a small copy stays on the calling thread. A caller that wants a large copy on one thread passes
/// [`Exec::serial`] or narrows the token with [`Exec::limit`].
/// A contiguous copy uses one `memcpy` per range of the narrowed token, or a
/// single `memcpy` on the calling thread when that token is serial.
///
/// This is a simple copy without ElementOp support. For copies with
/// element operations (conj, transpose, etc.), use `strided_kernel::copy_into`.
///
/// # Examples
///
/// ```
/// use threadteam::Exec;
/// use strided_perm::copy_into;
/// use strided_view::StridedArray;
///
/// let src = StridedArray::<f64>::from_fn_col_major(&[2, 3], |idx| (idx[0] * 10 + idx[1]) as f64);
/// let mut dst = StridedArray::<f64>::col_major(&[2, 3]);
/// copy_into(&mut dst.view_mut(), &src.view(), Exec::serial()).unwrap();
/// assert_eq!(dst.get(&[1, 2]), 12.0);
/// ```
pub fn copy_into<T: Copy + Send + Sync + 'static>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T>,
    exec: Exec<'_>,
) -> Result<()> {
    match prepare(dest, src)? {
        // SAFETY: both views are validated layouts over live, non-overlapping
        // storage (`StridedViewMut` is exclusive), and `prepare` matched them.
        Some(raw) => unsafe { copy_raw(raw, exec) },
        None => Ok(()),
    }
}

/// Copy elements into an uninitialized destination: `dest[i] = src[i]`.
///
/// The same permutation engine as [`copy_into`], selected on `T` rather than
/// on `MaybeUninit<T>`: the engine's native `f32`/`f64` kernels are chosen by
/// element type, so a destination reinterpreted as `MaybeUninit<T>` would
/// otherwise fall to the generic element path (about 2.5x slower on a
/// 4096x4096 `f64` transpose). Every destination element the views describe
/// is written exactly once; the stores go through raw pointers only, so no
/// reference to uninitialized storage is formed.
pub fn copy_into_uninit<T: Copy + Send + Sync + 'static>(
    dest: &mut StridedViewMut<core::mem::MaybeUninit<T>>,
    src: &StridedView<T>,
    exec: Exec<'_>,
) -> Result<()> {
    match prepare(dest, src)? {
        // SAFETY: as `copy_into`; `MaybeUninit<T>` has `T`'s layout and the
        // engine only stores into the destination.
        Some(raw) => unsafe { copy_raw(raw, exec) },
        None => Ok(()),
    }
}

/// Copy elements to a col-major destination.
///
/// This delegates to the same HPTT-inspired blocked permutation as
/// [`copy_into`]. The HPTT planner handles the col-major destination case
/// optimally.
pub fn copy_into_col_major<T: Copy + Send + Sync + 'static>(
    dst: &mut StridedViewMut<T>,
    src: &StridedView<T>,
    exec: Exec<'_>,
) -> Result<()> {
    copy_into(dst, src, exec)
}

#[cfg(test)]
fn try_fuse_group(dims: &[usize], strides: &[isize]) -> Option<(usize, isize)> {
    match dims.len() {
        0 => Some((1, 0)),
        1 => Some((dims[0], strides[0])),
        _ => {
            if dims.len() != strides.len() {
                return None;
            }
            for (&d, &s) in dims.iter().zip(strides.iter()) {
                if d > 1 && s == 0 {
                    return None;
                }
            }
            let mut base_idx: Option<usize> = None;
            let mut base_abs = usize::MAX;
            for (i, (&d, &s)) in dims.iter().zip(strides.iter()).enumerate() {
                if d <= 1 {
                    continue;
                }
                let abs = s.unsigned_abs();
                if abs < base_abs {
                    base_abs = abs;
                    base_idx = Some(i);
                }
            }

            let Some(base) = base_idx else {
                let stride = *strides
                    .iter()
                    .min_by_key(|s| s.unsigned_abs())
                    .unwrap_or(&0);
                return Some((dims.iter().product(), stride));
            };

            let mut used = vec![false; dims.len()];
            used[base] = true;
            let mut expected_abs = base_abs.checked_mul(dims[base])?;

            let non_singleton = dims.iter().filter(|&&d| d > 1).count();
            for _ in 1..non_singleton {
                let mut next = None;
                for i in 0..dims.len() {
                    if used[i] || dims[i] <= 1 {
                        continue;
                    }
                    if strides[i].unsigned_abs() == expected_abs {
                        next = Some(i);
                        break;
                    }
                }
                let i = next?;
                used[i] = true;
                expected_abs = expected_abs.checked_mul(dims[i])?;
            }

            Some((dims.iter().product(), strides[base]))
        }
    }
}

#[cfg(test)]
mod tests;
