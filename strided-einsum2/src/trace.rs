//! Trace-axis reduction for einsum operands.
//!
//! Trace axes are axes that appear in only one operand and not in the output.
//! They must be summed out (reduced) before the main contraction.

use stridedview::{ElementOp, ElementOpApply, StridedArray, StridedView};

/// Reduce all trace axes from a view by summing them out.
///
/// `trace_axes` are indices into the original view's dimensions, given in
/// ascending order. Each axis is reduced (summed) in sequence; later indices
/// are adjusted because each reduction removes one dimension.
///
/// Returns a new `StridedArray` with the trace axes removed, and a
/// `StridedView` over that array.
pub fn reduce_trace_axes<T, Op>(
    src: &StridedView<T, Op>,
    trace_axes: &[usize],
) -> stridedview::Result<StridedArray<T>>
where
    T: Copy + ElementOpApply + Send + Sync + std::ops::Add<Output = T> + num_traits::Zero,
    Op: ElementOp,
{
    if trace_axes.is_empty() {
        // No trace axes — just do a single dummy reduce on no axes; return a copy.
        // This shouldn't happen in practice since the caller checks.
        panic!("reduce_trace_axes called with empty trace_axes");
    }

    // Use strided::reduce_axis iteratively
    let mut current = strided::reduce_axis(src, trace_axes[0], Op::apply, |a, b| a + b, T::zero())?;

    for (k, &axis) in trace_axes.iter().enumerate().skip(1) {
        // Each previous reduction removed one axis, so adjust the index
        let adjusted_axis = axis - (k);
        current = strided::reduce_axis(
            &current.view(),
            adjusted_axis,
            |x| x,
            |a, b| a + b,
            T::zero(),
        )?;
    }

    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use stridedview::Identity;

    #[test]
    #[should_panic(expected = "reduce_trace_axes called with empty trace_axes")]
    fn test_reduce_no_trace() {
        let a =
            StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        let _ = reduce_trace_axes::<f64, Identity>(&a.view(), &[]);
    }

    #[test]
    fn test_reduce_single_trace() {
        // A: 2x3, reduce axis 1 => [2] with sums [6, 15]
        let a =
            StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        // A = [[1,2,3],[4,5,6]]
        let result = reduce_trace_axes::<f64, Identity>(&a.view(), &[1]).unwrap();
        assert_eq!(result.dims(), &[2]);
        assert_eq!(result.get(&[0]), 6.0); // 1+2+3
        assert_eq!(result.get(&[1]), 15.0); // 4+5+6
    }

    #[test]
    fn test_reduce_two_traces() {
        // A: 2x3x4, reduce axes [0, 2] => [3]
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 3, 4], |idx| {
            (idx[0] * 12 + idx[1] * 4 + idx[2]) as f64
        });
        let result = reduce_trace_axes::<f64, Identity>(&a.view(), &[0, 2]).unwrap();
        assert_eq!(result.dims(), &[3]);
        // After reducing axis 0: [3, 4] where result[j,k] = a[0,j,k] + a[1,j,k]
        //   = (j*4+k) + (12+j*4+k) = 2*j*4 + 2*k + 12
        // After reducing axis 1 (was axis 2, now adjusted to 1): [3]
        //   result[j] = sum_k (2*j*4 + 2*k + 12) for k=0..3
        //             = 4*(8*j + 12) + 2*(0+1+2+3) = 32*j + 48 + 12 = 32*j + 60
        // Hmm, let me recalculate...
        // a[i,j,k] = i*12 + j*4 + k
        // After reducing axis 0 (sum over i=0,1): b[j,k] = (0*12+j*4+k) + (1*12+j*4+k) = 12 + 2*j*4 + 2*k
        // After reducing axis 1 (originally axis 2, adjusted to 1 -> sum over k=0..3):
        //   c[j] = sum_{k=0}^{3} (12 + 8*j + 2*k) = 4*12 + 4*8*j + 2*(0+1+2+3) = 48 + 32*j + 12 = 60 + 32*j
        assert_eq!(result.get(&[0]), 60.0); // 60 + 32*0
        assert_eq!(result.get(&[1]), 92.0); // 60 + 32*1
        assert_eq!(result.get(&[2]), 124.0); // 60 + 32*2
    }
}
