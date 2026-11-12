//! Auxiliary routines ported from StridedViews.jl/src/auxiliary.jl

/// Compute the relative order of strides.
///
/// Returns a vector where `result[i]` is the rank of `strides[i]` among all non-zero strides.
/// Zero strides have order 1.
///
/// Returns an inline [`crate::AxisVec`]: the result does not allocate at rank
/// `<=` [`crate::INLINE_RANK`] and spills above it. This is the allocating form
/// of [`index_order_into`].
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
    let mut result = crate::AxisVec::new();
    result.resize(strides.len(), 1usize);
    // The lengths agree by construction, so the only error cannot occur.
    let _ = index_order_into(strides, &mut result);
    result
}

/// [`index_order`] written into `out` instead of a new vector, so a caller
/// with its own storage computes the order without allocating at any rank.
///
/// # Errors
///
/// [`crate::StridedError::StrideLengthMismatch`] when `out.len()` differs from
/// `strides.len()`; nothing is written then.
///
/// # Examples
///
/// ```
/// use strided_view::auxiliary::index_order_into;
///
/// let mut order = [0usize; 3];
/// index_order_into(&[4, 1, 2], &mut order).unwrap();
/// assert_eq!(order, [3, 1, 2]);
/// ```
pub fn index_order_into(strides: &[isize], out: &mut [usize]) -> crate::Result<()> {
    if out.len() != strides.len() {
        return Err(crate::StridedError::StrideLengthMismatch);
    }
    for (slot, &stride) in out.iter_mut().zip(strides) {
        let si = stride.unsigned_abs();
        *slot = if si == 0 {
            1
        } else {
            1 + strides
                .iter()
                .filter(|&&s| s != 0 && s.unsigned_abs() < si)
                .count()
        };
    }
    Ok(())
}

#[cfg(test)]
mod tests;
