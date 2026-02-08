use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use strided_kernel::copy_into;
use strided_view::{row_major_strides, StridedArray, StridedView, StridedViewMut};

/// Compute the min and max element-offset reachable from index [0,0,...,0].
///
/// For non-negative strides the min is 0; for negative strides (reversed views)
/// the min can be negative relative to `as_ptr()`.
fn compute_offset_range(shape: &[usize], strides: &[isize]) -> (isize, isize) {
    let mut min_off: isize = 0;
    let mut max_off: isize = 0;
    for (&d, &s) in shape.iter().zip(strides.iter()) {
        if d == 0 {
            continue;
        }
        let end = s * (d as isize - 1);
        if end < 0 {
            min_off += end;
        } else {
            max_off += end;
        }
    }
    (min_off, max_off)
}

/// Wrap an ndarray `ArrayD<T>` as a `StridedView` (zero-copy).
///
/// Dims and strides are passed through directly (no reversal).
pub fn array_to_strided_view<T>(arr: &ArrayD<T>) -> StridedView<'_, T> {
    let shape = arr.shape();
    let strides = arr.strides();
    let (min_off, max_off) = compute_offset_range(shape, strides);
    let base_ptr = unsafe { arr.as_ptr().offset(min_off) };
    let data_len = (max_off - min_off + 1) as usize;
    let data = unsafe { std::slice::from_raw_parts(base_ptr, data_len) };
    let offset = -min_off;
    StridedView::new(data, shape, strides, offset).expect("valid bounds")
}

/// Wrap an ndarray `ArrayViewD<T>` as a `StridedView` (zero-copy).
pub fn view_to_strided_view<'a, T>(view: &ArrayViewD<'a, T>) -> StridedView<'a, T> {
    let shape = view.shape();
    let strides = view.strides();
    let (min_off, max_off) = compute_offset_range(shape, strides);
    let base_ptr = unsafe { view.as_ptr().offset(min_off) };
    let data_len = (max_off - min_off + 1) as usize;
    let data = unsafe { std::slice::from_raw_parts(base_ptr, data_len) };
    let offset = -min_off;
    StridedView::new(data, shape, strides, offset).expect("valid bounds")
}

/// Wrap an ndarray `ArrayViewMutD<T>` as a `StridedViewMut` (zero-copy).
pub fn view_mut_to_strided_view_mut<'a, T>(
    view: &'a mut ArrayViewMutD<'a, T>,
) -> StridedViewMut<'a, T> {
    let shape: Vec<usize> = view.shape().to_vec();
    let strides: Vec<isize> = view.strides().to_vec();
    let (min_off, max_off) = compute_offset_range(&shape, &strides);
    let base_ptr = unsafe { view.as_mut_ptr().offset(min_off) };
    let data_len = (max_off - min_off + 1) as usize;
    let data = unsafe { std::slice::from_raw_parts_mut(base_ptr, data_len) };
    let offset = -min_off;
    StridedViewMut::new(data, &shape, &strides, offset).expect("valid bounds")
}

/// Convert a `StridedArray` result into an ndarray `ArrayD<T>`.
///
/// A copy is performed to materialize data into dense row-major order,
/// since the `StridedArray` from einsum may have arbitrary strides.
pub fn strided_array_to_ndarray<T>(arr: StridedArray<T>) -> ArrayD<T>
where
    T: Copy + strided_view::ElementOpApply + Send + Sync + num_traits::Zero + Default,
{
    let dims = arr.dims().to_vec();
    let total: usize = dims.iter().product();
    let mut buf = vec![T::default(); total];
    let rm_strides = row_major_strides(&dims);
    {
        let mut dest = StridedViewMut::new(&mut buf, &dims, &rm_strides, 0).expect("valid dest");
        copy_into(&mut dest, &arr.view()).expect("copy_into failed");
    }
    ArrayD::from_shape_vec(dims, buf).expect("shape matches")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    #[test]
    fn test_array_to_strided_view_2d() {
        // Row-major 2x3: [[1,2,3],[4,5,6]]
        let arr = ArrayD::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let sv = array_to_strided_view(&arr);

        // Dims passed through directly (no reversal)
        assert_eq!(sv.dims(), &[2, 3]);
        // Row-major strides for [2,3]: [3, 1]
        assert_eq!(sv.strides(), &[3, 1]);

        // Verify element access
        let ptr = sv.ptr();
        unsafe {
            assert_eq!(*ptr, 1.0); // [0,0]
            assert_eq!(*ptr.offset(1), 2.0); // [0,1]
            assert_eq!(*ptr.offset(3), 4.0); // [1,0]
            assert_eq!(*ptr.offset(5), 6.0); // [1,2]
        }
    }

    #[test]
    fn test_strided_array_to_ndarray_roundtrip() {
        // Column-major StridedArray [2, 3] with strides [1, 2]
        let data = vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0];
        let arr = StridedArray::from_parts(data, &[2, 3], &[1, 2], 0).expect("valid strided array");

        let nd = strided_array_to_ndarray(arr);
        assert_eq!(nd.shape(), &[2, 3]);
        // Row-major: [[1,2,3],[4,5,6]]
        assert_eq!(nd[[0, 0]], 1.0);
        assert_eq!(nd[[0, 1]], 2.0);
        assert_eq!(nd[[0, 2]], 3.0);
        assert_eq!(nd[[1, 0]], 4.0);
        assert_eq!(nd[[1, 1]], 5.0);
        assert_eq!(nd[[1, 2]], 6.0);
    }

    #[test]
    fn test_negative_stride_view() {
        // Create 3-element array [10, 20, 30], then reverse it
        let arr = ArrayD::from_shape_vec(vec![3], vec![10.0, 20.0, 30.0]).unwrap();
        use ndarray::s;
        let reversed = arr.slice(s![..;-1]);
        // reversed = [30, 20, 10], stride = -1

        // Convert the dynamic view
        let dyn_view: ArrayViewD<'_, f64> = reversed.into_dimensionality().unwrap();
        let sv = view_to_strided_view(&dyn_view);

        assert_eq!(sv.dims(), &[3]);
        assert_eq!(sv.strides(), &[-1]);

        // sv.ptr() points to element [0] = 30 (data[offset])
        let ptr = sv.ptr();
        unsafe {
            assert_eq!(*ptr, 30.0); // [0]
            assert_eq!(*ptr.offset(-1), 20.0); // [1]: offset + 1*(-1)
            assert_eq!(*ptr.offset(-2), 10.0); // [2]: offset + 2*(-1)
        }
    }
}
