use crate::{Result, StridedError};

/// Compute a common broadcast shape across multiple arrays.
///
/// This mirrors Julia's broadcasting rule per dimension:
/// - sizes must be equal, or one of them must be 1
/// - the resulting size is the maximum of the non-1 sizes
pub(crate) fn broadcast_shape(dims_list: &[&[usize]]) -> Result<Vec<usize>> {
    if dims_list.is_empty() {
        return Ok(vec![]);
    }

    let rank = dims_list[0].len();
    for dims in dims_list.iter().skip(1) {
        if dims.len() != rank {
            return Err(StridedError::RankMismatch(rank, dims.len()));
        }
    }

    let mut out = vec![1usize; rank];
    for d in 0..rank {
        let mut target = 1usize;
        for dims in dims_list {
            let n = dims[d];
            if n == 1 {
                continue;
            }
            if target == 1 {
                target = n;
            } else if target != n {
                return Err(StridedError::ShapeMismatch(
                    dims_list[0].to_vec(),
                    dims.to_vec(),
                ));
            }
        }
        out[d] = target;
    }

    Ok(out)
}

/// Promote strides to a broadcast target shape by setting stride-0 for broadcasted dimensions.
///
/// For each dimension `d`:
/// - if `src_dims[d] == target_dims[d]`, keep `src_strides[d]`
/// - else if `src_dims[d] == 1 && target_dims[d] > 1`, set stride to 0
/// - otherwise, shapes are incompatible
pub(crate) fn promote_strides_to_shape(
    target_dims: &[usize],
    src_dims: &[usize],
    src_strides: &[isize],
) -> Result<Vec<isize>> {
    if src_dims.len() != target_dims.len() {
        return Err(StridedError::RankMismatch(src_dims.len(), target_dims.len()));
    }
    if src_strides.len() != src_dims.len() {
        return Err(StridedError::StrideLengthMismatch);
    }

    let mut out = Vec::with_capacity(src_strides.len());
    for i in 0..target_dims.len() {
        let sdim = src_dims[i];
        let tdim = target_dims[i];
        if sdim == tdim {
            out.push(src_strides[i]);
        } else if sdim == 1 {
            out.push(0);
        } else {
            return Err(StridedError::ShapeMismatch(src_dims.to_vec(), target_dims.to_vec()));
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shape_basic() {
        let a = [2usize, 3];
        let b = [1usize, 3];
        let out = broadcast_shape(&[&a, &b]).unwrap();
        assert_eq!(out, vec![2, 3]);
    }

    #[test]
    fn test_broadcast_shape_incompatible() {
        let a = [2usize, 3];
        let b = [4usize, 3];
        let err = broadcast_shape(&[&a, &b]).unwrap_err();
        match err {
            StridedError::ShapeMismatch(_, _) => {}
            _ => panic!("unexpected error: {err:?}"),
        }
    }

    #[test]
    fn test_promote_strides_to_shape() {
        let target = [2usize, 3];
        let src_dims = [1usize, 3];
        let src_strides = [3isize, 1];
        let promoted = promote_strides_to_shape(&target, &src_dims, &src_strides).unwrap();
        assert_eq!(promoted, vec![0, 1]);
    }
}
