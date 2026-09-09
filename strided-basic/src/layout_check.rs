//! Output layout validation shared by kernel families.

/// Conservatively establish non-overlapping logical output positions.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::is_injective_layout;
/// assert!(is_injective_layout(&[2, 3], &[1, 2]));
/// assert!(!is_injective_layout(&[2, 3], &[0, 1]));
/// ```
pub fn is_injective_layout(dims: &[usize], strides: &[isize]) -> bool {
    let Some(total) = validate_injective_layout_inputs(dims, strides) else {
        return false;
    };
    if total <= 1 || has_disjoint_stride_spans(dims, strides) {
        return true;
    }

    const EXACT_CHECK_LIMIT: usize = 4096;
    if total <= EXACT_CHECK_LIMIT {
        return has_unique_offsets_exact(dims, strides, total);
    }

    false
}

pub(crate) fn is_injective_layout_without_alloc(dims: &[usize], strides: &[isize]) -> bool {
    let Some(total) = validate_injective_layout_inputs(dims, strides) else {
        return false;
    };
    if total <= 1 || has_disjoint_stride_spans(dims, strides) {
        return true;
    }

    const EXACT_CHECK_LIMIT: usize = 4096;
    total <= EXACT_CHECK_LIMIT && has_unique_offsets_pairwise(dims, strides, total)
}

fn offset_for_linear_index(dims: &[usize], strides: &[isize], mut linear: usize) -> Option<isize> {
    let mut offset = 0isize;
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        let index = linear % dim;
        linear /= dim;
        offset = offset.checked_add(stride.checked_mul(index as isize)?)?;
    }
    Some(offset)
}

fn has_unique_offsets_pairwise(dims: &[usize], strides: &[isize], total: usize) -> bool {
    for lhs in 0..total {
        let Some(lhs_offset) = offset_for_linear_index(dims, strides, lhs) else {
            return false;
        };
        for rhs in (lhs + 1)..total {
            if offset_for_linear_index(dims, strides, rhs) == Some(lhs_offset) {
                return false;
            }
        }
    }
    true
}

fn validate_injective_layout_inputs(dims: &[usize], strides: &[isize]) -> Option<usize> {
    if dims.len() != strides.len() {
        return None;
    }

    let total = dims
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))?;
    if total <= 1 {
        return Some(total);
    }
    if dims
        .iter()
        .zip(strides.iter())
        .any(|(&dim, &stride)| dim > 1 && stride == 0)
    {
        return None;
    }

    let mut min_offset = 0isize;
    let mut max_offset = 0isize;
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        if dim <= 1 {
            continue;
        }
        let extent = isize::try_from(dim - 1).ok()?;
        let span = stride.checked_mul(extent)?;
        if span >= 0 {
            max_offset = max_offset.checked_add(span)?;
        } else {
            min_offset = min_offset.checked_add(span)?;
        }
    }
    Some(total)
}

fn has_unique_offsets_exact(dims: &[usize], strides: &[isize], total: usize) -> bool {
    let mut seen = std::collections::HashSet::with_capacity(total);
    let mut indices = vec![0usize; dims.len()];
    let mut offset = 0isize;

    for _ in 0..total {
        if !seen.insert(offset) {
            return false;
        }

        for axis in 0..dims.len() {
            indices[axis] += 1;
            offset = match offset.checked_add(strides[axis]) {
                Some(offset) => offset,
                None => return false,
            };
            if indices[axis] < dims[axis] {
                break;
            }

            let rewind = match strides[axis].checked_mul(indices[axis] as isize) {
                Some(rewind) => rewind,
                None => return false,
            };
            offset = match offset.checked_sub(rewind) {
                Some(offset) => offset,
                None => return false,
            };
            indices[axis] = 0;
        }
    }

    true
}

fn has_disjoint_stride_spans(dims: &[usize], strides: &[isize]) -> bool {
    let mut covered_span = 0u128;
    let mut previous_axis = None;
    let active_axes = dims.iter().filter(|&&dim| dim > 1).count();
    for _ in 0..active_axes {
        let mut next = None;
        for (axis, (&dim, &stride)) in dims.iter().zip(strides.iter()).enumerate() {
            if dim <= 1 {
                continue;
            }
            let stride = match stride.checked_abs() {
                Some(stride) => stride as u128,
                None => return false,
            };
            let key = (stride, axis);
            if previous_axis.is_some_and(|previous| key <= previous) {
                continue;
            }
            if next.is_none_or(|(best, _)| key < best) {
                next = Some((key, dim as u128 - 1));
            }
        }
        let Some(((stride, axis), extent)) = next else {
            return false;
        };
        if stride <= covered_span {
            return false;
        }
        covered_span = match stride
            .checked_mul(extent)
            .and_then(|span| covered_span.checked_add(span))
        {
            Some(covered_span) => covered_span,
            None => return false,
        };
        previous_axis = Some((stride, axis));
    }

    true
}
