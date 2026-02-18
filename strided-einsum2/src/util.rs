//! Shared helpers for strided-einsum2.

/// Invert a permutation: if `perm[i] = j`, then `result[j] = i`.
pub fn invert_perm(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

/// Iterator over multi-dimensional index tuples within given dimensions.
///
/// Iterates in row-major order (last index varies fastest).
pub struct MultiIndex {
    dims: Vec<usize>,
    current: Vec<usize>,
    total: usize,
    count: usize,
}

impl MultiIndex {
    /// Create a new iterator over all index tuples for the given dimensions.
    pub fn new(dims: &[usize]) -> Self {
        let total: usize = dims.iter().product();
        Self {
            dims: dims.to_vec(),
            current: vec![0; dims.len()],
            total,
            count: 0,
        }
    }

    /// Compute byte offset from current indices and given strides.
    pub fn offset(&self, strides: &[isize]) -> isize {
        self.current
            .iter()
            .zip(strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum()
    }

    /// Reset the iterator to the beginning.
    pub fn reset(&mut self) {
        self.current.fill(0);
        self.count = 0;
    }
}

impl Iterator for MultiIndex {
    type Item = ();

    fn next(&mut self) -> Option<()> {
        if self.count >= self.total {
            return None;
        }
        if self.count > 0 {
            // Increment: last index varies fastest (row-major)
            let mut carry = true;
            for i in (0..self.dims.len()).rev() {
                if carry {
                    self.current[i] += 1;
                    if self.current[i] >= self.dims[i] {
                        self.current[i] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
        }
        self.count += 1;
        Some(())
    }
}

/// Try to fuse a contiguous dimension group into a single (total_size, innermost_stride).
///
/// For the group to be fusable, consecutive dimensions must have contiguous strides:
/// `stride[i] == stride[i+1] * dim[i+1]` for all i.
///
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
            // Ignore size-1 axes for contiguity checks; they do not constrain layout.
            // Find the first non-singleton axis with minimum absolute stride.
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

            // All singleton axes: trivially fusable.
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

            // Reconstruct increasing-stride contiguous chain without sorting.
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

    #[test]
    fn test_invert_perm() {
        assert_eq!(invert_perm(&[2, 0, 1]), vec![1, 2, 0]);
        assert_eq!(invert_perm(&[0, 1, 2]), vec![0, 1, 2]);
    }

    #[test]
    fn test_multi_index_2d() {
        let mut iter = MultiIndex::new(&[2, 3]);
        let mut indices = vec![];
        while iter.next().is_some() {
            indices.push(iter.current.clone());
        }
        assert_eq!(
            indices,
            vec![
                vec![0, 0],
                vec![0, 1],
                vec![0, 2],
                vec![1, 0],
                vec![1, 1],
                vec![1, 2],
            ]
        );
    }

    #[test]
    fn test_multi_index_offset() {
        let mut iter = MultiIndex::new(&[2, 3]);
        let strides = [3, 1]; // row-major strides
        let mut offsets = vec![];
        while iter.next().is_some() {
            offsets.push(iter.offset(&strides));
        }
        assert_eq!(offsets, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_multi_index_empty() {
        let mut iter = MultiIndex::new(&[]);
        assert!(iter.next().is_some()); // single scalar iteration
        assert!(iter.next().is_none());
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
        // 3x4 row-major: strides [4, 1]
        assert_eq!(try_fuse_group(&[3, 4], &[4, 1]), Some((12, 1)));
    }

    #[test]
    fn test_try_fuse_group_contiguous_col_major() {
        // 3x4 col-major: strides [1, 3]
        assert_eq!(try_fuse_group(&[3, 4], &[1, 3]), Some((12, 1)));
    }

    #[test]
    fn test_try_fuse_group_contiguous_3d() {
        // 2x3x4 row-major: strides [12, 4, 1]
        assert_eq!(try_fuse_group(&[2, 3, 4], &[12, 4, 1]), Some((24, 1)));
    }

    #[test]
    fn test_try_fuse_group_non_contiguous() {
        // strides don't follow contiguity rule
        assert_eq!(try_fuse_group(&[3, 4], &[8, 1]), None);
    }
}
