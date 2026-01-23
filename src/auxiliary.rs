//! Auxiliary routines ported from StridedViews.jl/src/auxiliary.jl
//!
//! These helper functions are used for dimension simplification and stride normalization.

// Some functions are ported for completeness but not yet used in the main code paths.
#![allow(dead_code)]

/// Simplify the dimensions of a strided view by fusing contiguous dimensions.
///
/// This function fuses subsequent dimensions that are contiguous in memory,
/// without changing the order of elements. For type stability, dimensions
/// are not removed but replaced with size 1 and moved to the end.
///
/// # Arguments
/// * `size` - The dimensions of the array
/// * `strides` - The strides of the array
///
/// # Returns
/// A tuple of (simplified_size, simplified_strides)
///
/// # Julia equivalent
/// ```julia
/// function _simplifydims(size::Dims{N}, strides::Dims{N}) where {N}
///     # Fuses dimensions where size[i] * strides[i] == strides[i+1]
/// end
/// ```
pub fn simplify_dims(size: &[usize], strides: &[isize]) -> (Vec<usize>, Vec<isize>) {
    let n = size.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (size.to_vec(), strides.to_vec());
    }

    // Work recursively from the front, matching Julia's implementation
    simplify_dims_recursive(size, strides)
}

fn simplify_dims_recursive(size: &[usize], strides: &[isize]) -> (Vec<usize>, Vec<isize>) {
    let n = size.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (size.to_vec(), strides.to_vec());
    }

    // Recurse on tail first
    let (mut tail_size, mut tail_strides) = simplify_dims_recursive(&size[1..], &strides[1..]);

    if size[0] == 1 {
        // Move size-1 dimension to end
        tail_size.push(1);
        tail_strides.push(1);
        return (tail_size, tail_strides);
    } else if !tail_size.is_empty()
        && size[0] as isize * strides[0] == tail_strides[0]
    {
        // Fuse with next dimension: size[0] * tail_size[0], then move a 1 to end
        let fused_size = size[0] * tail_size[0];
        let mut new_size = vec![fused_size];
        new_size.extend_from_slice(&tail_size[1..]);
        new_size.push(1);

        let mut new_strides = vec![strides[0]];
        new_strides.extend_from_slice(&tail_strides[1..]);
        new_strides.push(1);

        return (new_size, new_strides);
    } else {
        // No fusion possible
        let mut new_size = vec![size[0]];
        new_size.extend_from_slice(&tail_size);

        let mut new_strides = vec![strides[0]];
        new_strides.extend_from_slice(&tail_strides);

        return (new_size, new_strides);
    }
}

/// Normalize the strides of a strided view.
///
/// Strides associated with dimensions of size 1 have no intrinsic meaning
/// and can be changed arbitrarily. If one of the dimensions has size zero,
/// then the whole array has length zero, and all strides are ambiguous.
/// All ambiguous strides are set to produce a consistent layout.
///
/// # Arguments
/// * `size` - The dimensions of the array
/// * `strides` - The strides of the array
///
/// # Returns
/// Normalized strides
///
/// # Julia equivalent
/// ```julia
/// function _normalizestrides(size::Dims{N}, strides::Dims{N}) where {N}
///     for i in 1:N
///         if size[i] == 1
///             newstride = i == 1 ? 1 : strides[i - 1] * size[i - 1]
///             strides = Base.setindex(strides, newstride, i)
///         elseif size[i] == 0
///             return (1, Base.front(cumprod(size))...)
///         end
///     end
///     return strides
/// end
/// ```
pub fn normalize_strides(size: &[usize], strides: &[isize]) -> Vec<isize> {
    let n = size.len();
    if n == 0 {
        return vec![];
    }

    let mut result = strides.to_vec();

    for i in 0..n {
        if size[i] == 1 {
            // Stride for size-1 dimension is arbitrary, make it consistent
            let new_stride = if i == 0 {
                1
            } else {
                result[i - 1] * size[i - 1] as isize
            };
            result[i] = new_stride;
        } else if size[i] == 0 {
            // Zero-size dimension: return cumulative product strides
            let mut cumprod = vec![1isize; n];
            for j in 1..n {
                cumprod[j] = cumprod[j - 1] * size[j - 1] as isize;
            }
            return cumprod;
        }
    }

    result
}

/// Compute the linear memory index given multi-dimensional indices and strides.
///
/// # Arguments
/// * `indices` - The multi-dimensional indices (0-based)
/// * `strides` - The strides
///
/// # Returns
/// The linear offset from the base pointer
#[inline]
pub fn compute_linear_index(indices: &[usize], strides: &[isize]) -> isize {
    indices
        .iter()
        .zip(strides.iter())
        .map(|(&idx, &stride)| idx as isize * stride)
        .sum()
}

/// Compute the relative order of strides.
///
/// Returns a vector where `result[i]` is the rank of `strides[i]` among all non-zero strides.
/// Zero strides have order 1.
///
/// # Julia equivalent
/// ```julia
/// function indexorder(strides::NTuple{N,Int}) where {N}
///     return ntuple(Val(N)) do i
///         si = abs(strides[i])
///         si == 0 && return 1
///         k = 1
///         for s in strides
///             if s != 0 && abs(s) < si
///                 k += 1
///             end
///         end
///         return k
///     end
/// end
/// ```
pub fn index_order(strides: &[isize]) -> Vec<usize> {
    let n = strides.len();
    let mut result = vec![1usize; n];

    for i in 0..n {
        let si = strides[i].unsigned_abs();
        if si == 0 {
            result[i] = 1;
            continue;
        }
        let mut k = 1usize;
        for &s in strides {
            if s != 0 && s.unsigned_abs() < si {
                k += 1;
            }
        }
        result[i] = k;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_dims_no_fusion() {
        // Non-contiguous dimensions: no fusion
        let (size, strides) = simplify_dims(&[3, 4], &[1, 10]);
        assert_eq!(size, vec![3, 4]);
        assert_eq!(strides, vec![1, 10]);
    }

    #[test]
    fn test_simplify_dims_fuse_contiguous() {
        // Contiguous dimensions: 3x4 with strides [1, 3] -> fused to 12
        let (size, strides) = simplify_dims(&[3, 4], &[1, 3]);
        assert_eq!(size, vec![12, 1]);
        assert_eq!(strides, vec![1, 1]);
    }

    #[test]
    fn test_simplify_dims_size_one() {
        // Size-1 dimension moves to end
        let (size, strides) = simplify_dims(&[1, 4], &[1, 1]);
        assert_eq!(size, vec![4, 1]);
        assert_eq!(strides, vec![1, 1]);
    }

    #[test]
    fn test_simplify_dims_3d_partial() {
        // 3D case: fuse first two, not third
        let (size, strides) = simplify_dims(&[2, 3, 4], &[1, 2, 100]);
        assert_eq!(size, vec![6, 4, 1]);
        assert_eq!(strides, vec![1, 100, 1]);
    }

    #[test]
    fn test_normalize_strides_size_one() {
        let strides = normalize_strides(&[1, 4, 3], &[999, 1, 4]);
        assert_eq!(strides, vec![1, 1, 4]); // First stride normalized to 1
    }

    #[test]
    fn test_normalize_strides_zero_size() {
        let strides = normalize_strides(&[3, 0, 4], &[1, 3, 0]);
        assert_eq!(strides, vec![1, 3, 0]); // Cumulative product: [1, 3, 0]
    }

    #[test]
    fn test_index_order() {
        // strides [4, 1, 2]: order is [3, 1, 2] (4 is largest, 1 is smallest, 2 is middle)
        let order = index_order(&[4, 1, 2]);
        assert_eq!(order, vec![3, 1, 2]);
    }

    #[test]
    fn test_index_order_with_zero() {
        // Zero strides have order 1
        let order = index_order(&[4, 0, 2]);
        assert_eq!(order, vec![2, 1, 1]);
    }

    #[test]
    fn test_index_order_negative_strides() {
        // Negative strides should use absolute value
        // Julia: si = abs(strides[i])
        let order = index_order(&[-4, 1, -2]);
        // abs: [4, 1, 2] -> order: [3, 1, 2]
        assert_eq!(order, vec![3, 1, 2]);
    }

    #[test]
    fn test_index_order_tied_strides() {
        // Tied strides (same absolute value) - both get same order
        // Julia behavior: if s != 0 && abs(s) < si then k += 1
        // For tied strides, neither is < the other, so they get same k
        let order = index_order(&[2, 2, 1]);
        // For index 0: stride=2, k=1, then check [2,2,1]: 2<2? no, 2<2? no, 1<2? yes -> k=2
        // For index 1: stride=2, k=1, then check [2,2,1]: 2<2? no, 2<2? no, 1<2? yes -> k=2
        // For index 2: stride=1, k=1, no other stride < 1 -> k=1
        assert_eq!(order, vec![2, 2, 1]);
    }

    #[test]
    fn test_index_order_all_same() {
        // All strides are the same
        let order = index_order(&[3, 3, 3]);
        // All get order 1 (no stride is smaller than another)
        assert_eq!(order, vec![1, 1, 1]);
    }

    #[test]
    fn test_index_order_mixed_signs() {
        // Mixed positive and negative strides with same absolute value
        let order = index_order(&[2, -2, 1]);
        // abs: [2, 2, 1]
        // For both 2s: only 1 is smaller -> order 2
        // For 1: nothing smaller -> order 1
        assert_eq!(order, vec![2, 2, 1]);
    }

    #[test]
    fn test_compute_linear_index() {
        let idx = compute_linear_index(&[2, 3], &[1, 4]);
        assert_eq!(idx, 2 * 1 + 3 * 4); // 14
    }
}
