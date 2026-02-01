//! Auxiliary routines ported from StridedViews.jl/src/auxiliary.jl

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
        let order = index_order(&[-4, 1, -2]);
        assert_eq!(order, vec![3, 1, 2]);
    }

    #[test]
    fn test_index_order_tied_strides() {
        let order = index_order(&[2, 2, 1]);
        assert_eq!(order, vec![2, 2, 1]);
    }

    #[test]
    fn test_index_order_all_same() {
        let order = index_order(&[3, 3, 3]);
        assert_eq!(order, vec![1, 1, 1]);
    }
}
