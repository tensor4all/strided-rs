//! Copy/permutation operations on strided views.

#[cfg(feature = "parallel")]
use crate::hptt::execute_permute_blocked_par;
use crate::hptt::{build_permute_plan, execute_permute_blocked};
use crate::kernel::total_len;
use strided_view::{Result, StridedError, StridedView, StridedViewMut};

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

/// Copy elements from source to destination: `dest[i] = src[i]`.
///
/// Uses HPTT-inspired blocked permutation with bilateral dimension fusion,
/// cache-aware blocking, and optimal loop ordering.
///
/// This is a simple copy without ElementOp support. For copies with
/// element operations (conj, transpose, etc.), use `strided_kernel::copy_into`.
pub fn copy_into<T: Copy>(dest: &mut StridedViewMut<T>, src: &StridedView<T>) -> Result<()> {
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

    let dst_ptr = dest.as_mut_ptr();
    let src_ptr = src.ptr();
    let dst_strides = dest.strides();
    let src_strides = src.strides();

    // Fast path: both contiguous
    if is_both_contiguous(dst_dims, dst_strides, src_strides) {
        let len = total_len(dst_dims);
        unsafe { std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, len) };
        return Ok(());
    }

    // HPTT-inspired blocked permutation
    let elem_size = std::mem::size_of::<T>();
    let plan = build_permute_plan(dst_dims, src_strides, dst_strides, elem_size);
    unsafe {
        execute_permute_blocked(src_ptr, dst_ptr, &plan);
    }
    Ok(())
}

/// Copy elements to a col-major destination.
///
/// This now delegates to the same HPTT-inspired blocked permutation as
/// `copy_into`. The HPTT planner automatically handles the col-major
/// destination case optimally.
pub fn copy_into_col_major<T: Copy>(
    dst: &mut StridedViewMut<T>,
    src: &StridedView<T>,
) -> Result<()> {
    copy_into(dst, src)
}

/// Copy elements from source to destination with Rayon parallelism.
///
/// Parallelizes the outermost block loop of the HPTT-inspired permutation.
/// Falls back to single-threaded for small tensors.
#[cfg(feature = "parallel")]
pub fn copy_into_par<T: Copy + Send + Sync>(
    dest: &mut StridedViewMut<T>,
    src: &StridedView<T>,
) -> Result<()> {
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

    let dst_ptr = dest.as_mut_ptr();
    let src_ptr = src.ptr();
    let dst_strides = dest.strides();
    let src_strides = src.strides();

    // Fast path: both contiguous (memcpy is already bandwidth-limited)
    if is_both_contiguous(dst_dims, dst_strides, src_strides) {
        let len = total_len(dst_dims);
        unsafe { std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, len) };
        return Ok(());
    }

    let elem_size = std::mem::size_of::<T>();
    let plan = build_permute_plan(dst_dims, src_strides, dst_strides, elem_size);
    unsafe {
        execute_permute_blocked_par(src_ptr, dst_ptr, &plan);
    }
    Ok(())
}

/// Copy elements to a col-major destination with Rayon parallelism.
#[cfg(feature = "parallel")]
pub fn copy_into_col_major_par<T: Copy + Send + Sync>(
    dst: &mut StridedViewMut<T>,
    src: &StridedView<T>,
) -> Result<()> {
    copy_into_par(dst, src)
}

/// Try to fuse a contiguous dimension group into a single (total_size, innermost_stride).
///
/// For the group to be fusable, consecutive dimensions must have contiguous strides.
/// Returns `None` if strides are not contiguous within the group.
pub fn try_fuse_group(dims: &[usize], strides: &[isize]) -> Option<(usize, isize)> {
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
mod tests {
    use super::*;
    use strided_view::StridedArray;

    #[test]
    fn test_copy_into_contiguous() {
        let src =
            StridedArray::<f64>::from_fn_col_major(&[2, 3], |idx| (idx[0] * 10 + idx[1]) as f64);
        let mut dst = StridedArray::<f64>::col_major(&[2, 3]);
        copy_into(&mut dst.view_mut(), &src.view()).unwrap();
        assert_eq!(dst.get(&[0, 0]), 0.0);
        assert_eq!(dst.get(&[1, 2]), 12.0);
    }

    #[test]
    fn test_copy_into_transposed() {
        // src is row-major [3,2], dst is col-major [3,2]
        let src = StridedArray::<f64>::from_parts(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            &[3, 2],
            &[2, 1], // row-major
            0,
        )
        .unwrap();
        let mut dst = StridedArray::<f64>::col_major(&[3, 2]);
        copy_into(&mut dst.view_mut(), &src.view()).unwrap();
        assert_eq!(dst.get(&[0, 0]), 0.0);
        assert_eq!(dst.get(&[0, 1]), 1.0);
        assert_eq!(dst.get(&[1, 0]), 2.0);
        assert_eq!(dst.get(&[2, 1]), 5.0);
    }

    #[test]
    fn test_copy_into_col_major_basic() {
        let src =
            StridedArray::<f64>::from_fn_col_major(&[4, 3], |idx| (idx[0] * 10 + idx[1]) as f64);
        let mut dst = StridedArray::<f64>::col_major(&[4, 3]);
        copy_into_col_major(&mut dst.view_mut(), &src.view()).unwrap();
        assert_eq!(dst.get(&[0, 0]), 0.0);
        assert_eq!(dst.get(&[3, 2]), 32.0);
    }

    #[test]
    fn test_copy_into_col_major_permuted_src() {
        // src has scattered strides, dst is col-major
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let src = StridedArray::<f64>::from_parts(
            data,
            &[2, 3, 4],
            &[12, 4, 1], // row-major
            0,
        )
        .unwrap();
        let mut dst = StridedArray::<f64>::col_major(&[2, 3, 4]);
        copy_into_col_major(&mut dst.view_mut(), &src.view()).unwrap();
        // Verify element-by-element
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(
                        dst.get(&[i, j, k]),
                        src.get(&[i, j, k]),
                        "mismatch at [{},{},{}]",
                        i,
                        j,
                        k
                    );
                }
            }
        }
    }

    #[test]
    fn test_try_fuse_group_empty() {
        assert_eq!(try_fuse_group(&[], &[]), Some((1, 0)));
    }

    #[test]
    fn test_try_fuse_group_single() {
        assert_eq!(try_fuse_group(&[5], &[2]), Some((5, 2)));
    }

    #[test]
    fn test_try_fuse_group_contiguous_row_major() {
        assert_eq!(try_fuse_group(&[3, 4], &[4, 1]), Some((12, 1)));
    }

    #[test]
    fn test_try_fuse_group_contiguous_col_major() {
        assert_eq!(try_fuse_group(&[3, 4], &[1, 3]), Some((12, 1)));
    }

    #[test]
    fn test_try_fuse_group_non_contiguous() {
        assert_eq!(try_fuse_group(&[3, 4], &[8, 1]), None);
    }

    #[test]
    fn test_copy_shape_mismatch() {
        let src = StridedArray::<f64>::col_major(&[2, 3]);
        let mut dst = StridedArray::<f64>::col_major(&[3, 2]);
        let result = copy_into(&mut dst.view_mut(), &src.view());
        assert!(result.is_err());
    }
}
