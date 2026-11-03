//! Auxiliary routines ported from StridedViews.jl/src/auxiliary.jl

/// Compute the relative order of strides.
///
/// Returns a vector where `result[i]` is the rank of `strides[i]` among all non-zero strides.
/// Zero strides have order 1.
///
/// Returns an inline [`crate::AxisVec`]: the result does not allocate at rank
/// `<=` [`crate::INLINE_RANK`] and spills above it.
///
/// # Examples
///
/// ```
/// use strided_view::auxiliary::index_order;
///
/// let order = index_order(&[4, 1, 2]);
/// assert_eq!(order.as_slice(), &[3, 1, 2]);
/// ```
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
pub fn index_order(strides: &[isize]) -> crate::AxisVec {
    let n = strides.len();
    let mut result = crate::AxisVec::new();
    result.resize(n, 1usize);

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
mod tests;
