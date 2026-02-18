//! Dimension fusion logic ported from Strided.jl/src/mapreduce.jl
//!
//! This module implements the core dimension fusion algorithm that merges
//! contiguous dimensions to reduce iteration complexity.

/// Fuse contiguous dimensions across multiple arrays.
///
/// This function fuses subsequent dimensions that are contiguous in memory
/// for all arrays. If `strides[k][i] == dims[i-1] * strides[k][i-1]` for all k,
/// dimensions i-1 and i can be merged.
pub fn fuse_dims(dims: &[usize], all_strides: &[&[isize]]) -> Vec<usize> {
    let n = dims.len();
    if n <= 1 || all_strides.is_empty() {
        return dims.to_vec();
    }

    let mut result = dims.to_vec();

    // Work from the end towards the beginning (Julia: for i in length(dims):-1:2)
    for i in (1..n).rev() {
        let mut can_merge = true;

        // Check all arrays for contiguity
        for strides in all_strides {
            // s[i] should equal dims[i-1] * s[i-1] for fusion
            let expected = result[i - 1] as isize * strides[i - 1];
            if strides[i] != expected {
                can_merge = false;
                break;
            }
        }

        if can_merge {
            // Fuse dimensions: merge dimension i into i-1
            result[i - 1] *= result[i];
            result[i] = 1;
        }
    }

    result
}

/// Remove size-1 dimensions from fused dims and all corresponding strides.
///
/// After `fuse_dims()`, many dimensions may be 1 (either originally size-1
/// or merged into a neighbor). These contribute nothing to iteration but
/// increase loop depth. This function strips them out.
///
/// If ALL dimensions are 1 (scalar-like), a single dimension of size 1
/// is preserved so the kernel has something to iterate over.
pub fn compress_dims(dims: &[usize], all_strides: &[Vec<isize>]) -> (Vec<usize>, Vec<Vec<isize>>) {
    let kept: Vec<usize> = (0..dims.len()).filter(|&i| dims[i] != 1).collect();

    if kept.is_empty() {
        // All dims are 1 (or empty). Preserve a single trivial dimension.
        if dims.is_empty() {
            return (vec![], all_strides.to_vec());
        }
        let new_strides = all_strides.iter().map(|s| vec![s[0]]).collect();
        return (vec![1], new_strides);
    }

    let new_dims: Vec<usize> = kept.iter().map(|&i| dims[i]).collect();
    let new_strides: Vec<Vec<isize>> = all_strides
        .iter()
        .map(|s| kept.iter().map(|&i| s[i]).collect())
        .collect();

    (new_dims, new_strides)
}

/// Compute the "importance" of each dimension for loop ordering.
///
/// This encodes stride order information into importance scores that determine
/// the optimal iteration order. The output array's strides are weighted 2x.
pub fn compute_importance(
    dims: &[usize],
    all_strides: &[&[isize]],
    index_orders: &[Vec<usize>],
) -> Vec<u64> {
    let n = dims.len();
    let m = all_strides.len();

    if n == 0 || m == 0 {
        return vec![];
    }

    // g = ceil(log2(M + 2)) = number of bits needed to encode array count
    let g = (64 - (m as u64 + 1).leading_zeros()) as u64;

    let mut importance = vec![0u64; n];

    // First array (output) is weighted 2x
    for i in 0..n {
        let shift = g * (n - index_orders[0][i]) as u64;
        importance[i] = 2 * (1u64 << shift);
    }

    // Add contributions from remaining arrays
    #[allow(clippy::needless_range_loop)]
    for k in 1..m {
        for i in 0..n {
            let shift = g * (n - index_orders[k][i]) as u64;
            importance[i] += 1u64 << shift;
        }
    }

    // Zero importance for size-1 dimensions (put them at the back)
    for i in 0..n {
        if dims[i] <= 1 {
            importance[i] = 0;
        }
    }

    importance
}

/// Get the permutation that sorts by importance (descending).
pub fn sort_by_importance(importance: &[u64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..importance.len()).collect();
    indices.sort_by(|&a, &b| importance[b].cmp(&importance[a]));
    indices
}

/// Compute the minimum stride cost for each dimension.
pub fn compute_costs<S: AsRef<[isize]>>(all_strides: &[S]) -> Vec<isize> {
    if all_strides.is_empty() {
        return vec![];
    }

    let n = all_strides[0].as_ref().len();
    let mut costs = vec![isize::MAX; n];

    for strides in all_strides {
        let strides = strides.as_ref();
        for i in 0..n {
            costs[i] = costs[i].min(strides[i].abs());
        }
    }

    // Transform: zero -> 1, nonzero -> 2*abs
    for cost in &mut costs {
        if *cost == 0 {
            *cost = 1;
        } else {
            *cost *= 2;
        }
    }

    costs
}

/// Bilateral dimension fusion for src + dst stride patterns.
///
/// Two dimensions `i` and `i+1` can be fused if BOTH src and dst strides
/// are contiguous for those dimensions. Returns the fused (dims, src_strides, dst_strides).
pub fn fuse_dims_bilateral(
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
) -> (Vec<usize>, Vec<isize>, Vec<isize>) {
    let n = dims.len();
    if n <= 1 {
        return (dims.to_vec(), src_strides.to_vec(), dst_strides.to_vec());
    }

    let mut fused_dims = Vec::with_capacity(n);
    let mut fused_src = Vec::with_capacity(n);
    let mut fused_dst = Vec::with_capacity(n);

    fused_dims.push(dims[0]);
    fused_src.push(src_strides[0]);
    fused_dst.push(dst_strides[0]);

    for i in 1..n {
        let last = fused_dims.len() - 1;
        let d_prev = fused_dims[last];
        let ss_prev = fused_src[last];
        let ds_prev = fused_dst[last];

        // Check if dim i is contiguous with the previous fused dim in BOTH src and dst
        let src_contiguous = src_strides[i] == ss_prev * d_prev as isize;
        let dst_contiguous = dst_strides[i] == ds_prev * d_prev as isize;

        if src_contiguous && dst_contiguous {
            // Fuse: multiply the last fused dim
            fused_dims[last] *= dims[i];
        } else {
            fused_dims.push(dims[i]);
            fused_src.push(src_strides[i]);
            fused_dst.push(dst_strides[i]);
        }
    }

    (fused_dims, fused_src, fused_dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuse_dims_contiguous() {
        let dims = [3, 4];
        let strides1 = [1isize, 3];
        let strides2 = [1isize, 3];
        let all_strides: Vec<&[isize]> = vec![&strides1, &strides2];
        let fused = fuse_dims(&dims, &all_strides);
        assert_eq!(fused, vec![12, 1]);
    }

    #[test]
    fn test_fuse_dims_non_contiguous() {
        let dims = [3, 4];
        let strides1 = [1isize, 10];
        let all_strides: Vec<&[isize]> = vec![&strides1];
        let fused = fuse_dims(&dims, &all_strides);
        assert_eq!(fused, vec![3, 4]);
    }

    #[test]
    fn test_fuse_dims_bilateral_all_contiguous() {
        // 24-dim all-size-2 col-major: all contiguous -> fuses to single dim
        let dims = vec![2, 2, 2, 2];
        let src_strides = vec![1, 2, 4, 8];
        let dst_strides = vec![1, 2, 4, 8];
        let (fd, fs, fds) = fuse_dims_bilateral(&dims, &src_strides, &dst_strides);
        assert_eq!(fd, vec![16]);
        assert_eq!(fs, vec![1]);
        assert_eq!(fds, vec![1]);
    }

    #[test]
    fn test_fuse_dims_bilateral_partial() {
        // src: contiguous 0-1, not 1-2
        // dst: contiguous 0-1-2
        let dims = vec![2, 3, 4];
        let src_strides = vec![1, 2, 100]; // 0-1 contiguous, 1-2 not
        let dst_strides = vec![1, 2, 6]; // all contiguous
        let (fd, fs, fds) = fuse_dims_bilateral(&dims, &src_strides, &dst_strides);
        assert_eq!(fd, vec![6, 4]); // first two fuse
        assert_eq!(fs, vec![1, 100]);
        assert_eq!(fds, vec![1, 6]);
    }

    #[test]
    fn test_fuse_dims_bilateral_scattered() {
        // The benchmark case: scattered strides, nothing fuses
        let dims = vec![2, 2, 2];
        let src_strides = vec![1, 4194304, 2]; // scattered
        let dst_strides = vec![1, 2, 4]; // contiguous
        let (fd, fs, fds) = fuse_dims_bilateral(&dims, &src_strides, &dst_strides);
        assert_eq!(fd, vec![2, 2, 2]); // nothing fuses
        assert_eq!(fs, vec![1, 4194304, 2]);
        assert_eq!(fds, vec![1, 2, 4]);
    }

    #[test]
    fn test_compress_dims_removes_fused() {
        let dims = vec![12usize, 1];
        let strides = vec![vec![1isize, 3]];
        let (cd, cs) = compress_dims(&dims, &strides);
        assert_eq!(cd, vec![12]);
        assert_eq!(cs, vec![vec![1]]);
    }

    #[test]
    fn test_compute_costs() {
        let strides1 = [1isize, 4, 0];
        let strides2 = [2isize, 1, 0];
        let all_strides: Vec<&[isize]> = vec![&strides1, &strides2];
        let costs = compute_costs(&all_strides);
        assert_eq!(costs, vec![2, 2, 1]);
    }
}
