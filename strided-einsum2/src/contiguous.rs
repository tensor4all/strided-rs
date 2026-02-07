//! GEMM-ready operand types and preparation functions for contiguous data.
//!
//! These types encapsulate the logic for preparing strided operands for GEMM:
//! checking fusability, copying to col-major buffers when needed, and managing
//! the writeback for borrowed output operands.

use crate::util::try_fuse_group;
use crate::Scalar;
use strided_view::{StridedArray, StridedView, StridedViewMut};

/// GEMM-ready input operand with contiguous data.
pub struct ContiguousOperand<T> {
    ptr: *const T,
    row_stride: isize,
    col_stride: isize,
    batch_strides: Vec<isize>,
    conj: bool,
    /// Owns the buffer if a copy was made or input was consumed.
    pub(crate) _buf: Option<StridedArray<T>>,
}

/// GEMM-ready output operand with contiguous data.
pub struct ContiguousOperandMut<T> {
    ptr: *mut T,
    row_stride: isize,
    col_stride: isize,
    batch_strides: Vec<isize>,
    /// Whether the caller must copy the buffer back to the original destination
    /// after GEMM completes (true only for borrowed non-contiguous C).
    needs_writeback: bool,
    /// Owns the buffer if a copy was made.
    pub(crate) _buf: Option<StridedArray<T>>,
}

impl<T> ContiguousOperand<T> {
    /// Raw const pointer to the operand data at the base offset.
    #[inline]
    pub fn ptr(&self) -> *const T {
        self.ptr
    }

    /// Row (lo-group) stride for the fused 2D matrix.
    #[inline]
    pub fn row_stride(&self) -> isize {
        self.row_stride
    }

    /// Column (sum/ro-group) stride for the fused 2D matrix.
    #[inline]
    pub fn col_stride(&self) -> isize {
        self.col_stride
    }

    /// Batch dimension strides.
    #[inline]
    pub fn batch_strides(&self) -> &[isize] {
        &self.batch_strides
    }

    /// Whether this operand requires conjugation.
    #[inline]
    pub fn conj(&self) -> bool {
        self.conj
    }

    /// Returns `true` if this operand owns a buffer (copy was made or ownership transferred).
    #[cfg(test)]
    #[inline]
    pub(crate) fn has_buf(&self) -> bool {
        self._buf.is_some()
    }
}

impl<T> ContiguousOperandMut<T> {
    /// Raw mutable pointer to the operand data at the base offset.
    #[inline]
    pub fn ptr(&self) -> *mut T {
        self.ptr
    }

    /// Row (lo-group) stride for the fused 2D matrix.
    #[inline]
    pub fn row_stride(&self) -> isize {
        self.row_stride
    }

    /// Column (ro-group) stride for the fused 2D matrix.
    #[inline]
    pub fn col_stride(&self) -> isize {
        self.col_stride
    }

    /// Batch dimension strides.
    #[inline]
    pub fn batch_strides(&self) -> &[isize] {
        &self.batch_strides
    }

    /// Returns `true` if this operand owns a buffer (copy was made).
    #[cfg(test)]
    #[inline]
    pub(crate) fn has_buf(&self) -> bool {
        self._buf.is_some()
    }

    /// Returns `true` if the caller must copy the buffer back to the original
    /// destination after GEMM completes.
    #[cfg(test)]
    #[inline]
    pub(crate) fn needs_writeback(&self) -> bool {
        self.needs_writeback
    }
}

impl<T: Scalar> ContiguousOperandMut<T> {
    /// After GEMM: copy the internal buffer back to `dest` if needed.
    ///
    /// This is a no-op when the GEMM wrote directly to the destination
    /// (contiguous case or owned output).
    pub fn finalize_into(self, dest: &mut StridedViewMut<T>) -> crate::Result<()> {
        if self.needs_writeback {
            if let Some(ref buf) = self._buf {
                strided_kernel::copy_into(dest, &buf.view())?;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "faer")]
use crate::bgemm_faer::alloc_batched_col_major;

#[cfg(any(feature = "blas", feature = "blas-inject"))]
use crate::bgemm_blas::alloc_batched_col_major;

/// Prepare a borrowed input view for GEMM.
///
/// Checks if the two inner dimension groups are fusable.
/// If not, copies to a contiguous col-major buffer.
#[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
pub fn prepare_input_view<T: Scalar>(
    view: &StridedView<T>,
    n_batch: usize,
    n_group1: usize,
    n_group2: usize,
    conj: bool,
) -> crate::Result<ContiguousOperand<T>> {
    let dims = view.dims();
    let strides = view.strides();

    // Extract dimension/stride groups
    let group1_dims = &dims[n_batch..n_batch + n_group1];
    let group1_strides = &strides[n_batch..n_batch + n_group1];
    let group2_dims = &dims[n_batch + n_group1..n_batch + n_group1 + n_group2];
    let group2_strides = &strides[n_batch + n_group1..n_batch + n_group1 + n_group2];

    // For BLAS backends, resolve conjugation by copying with conj applied.
    // BLAS does not support conjugation flags natively (unlike faer), so we
    // materialize conj into the data before the GEMM call.
    #[cfg(any(feature = "blas", feature = "blas-inject"))]
    if conj {
        use strided_view::Conj as ConjOp;
        use strided_view::ElementOp;

        let m: usize = group1_dims.iter().product::<usize>().max(1);
        let mut buf = alloc_batched_col_major(dims, n_batch);
        strided_kernel::map_into(&mut buf.view_mut(), view, |x| ConjOp::apply(x))?;
        let ptr = buf.view().ptr();
        let batch_strides = buf.strides()[..n_batch].to_vec();
        let row_stride = if m == 0 { 0 } else { 1isize };
        let col_stride = m as isize;
        return Ok(ContiguousOperand {
            ptr,
            row_stride,
            col_stride,
            batch_strides,
            conj: false,
            _buf: Some(buf),
        });
    }

    let fused_g1 = try_fuse_group(group1_dims, group1_strides);
    let fused_g2 = try_fuse_group(group2_dims, group2_strides);

    let needs_copy = fused_g1.is_none() || fused_g2.is_none();

    if needs_copy {
        let m: usize = group1_dims.iter().product::<usize>().max(1);
        let mut buf = alloc_batched_col_major(dims, n_batch);
        strided_kernel::copy_into(&mut buf.view_mut(), view)?;
        let ptr = buf.view().ptr();
        let batch_strides = buf.strides()[..n_batch].to_vec();
        let row_stride = if m == 0 { 0 } else { 1isize };
        let col_stride = m as isize;
        Ok(ContiguousOperand {
            ptr,
            row_stride,
            col_stride,
            batch_strides,
            conj,
            _buf: Some(buf),
        })
    } else {
        let (_, rs) = fused_g1.unwrap();
        let (_, cs) = fused_g2.unwrap();
        let batch_strides = strides[..n_batch].to_vec();
        Ok(ContiguousOperand {
            ptr: view.ptr(),
            row_stride: rs,
            col_stride: cs,
            batch_strides,
            conj,
            _buf: None,
        })
    }
}

/// Prepare an owned input array for GEMM.
///
/// If already contiguous after dimension grouping, transfers ownership without copying.
/// Otherwise, copies to a new col-major buffer.
#[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
pub fn prepare_input_owned<T: Scalar>(
    arr: StridedArray<T>,
    n_batch: usize,
    n_group1: usize,
    n_group2: usize,
    conj: bool,
) -> crate::Result<ContiguousOperand<T>> {
    let dims = arr.dims().to_vec();
    let strides = arr.strides().to_vec();

    // Extract dimension/stride groups
    let group1_dims = &dims[n_batch..n_batch + n_group1];
    let group1_strides = &strides[n_batch..n_batch + n_group1];
    let group2_dims = &dims[n_batch + n_group1..n_batch + n_group1 + n_group2];
    let group2_strides = &strides[n_batch + n_group1..n_batch + n_group1 + n_group2];

    // For BLAS backends, resolve conjugation by copying with conj applied.
    #[cfg(any(feature = "blas", feature = "blas-inject"))]
    if conj {
        use strided_view::Conj as ConjOp;
        use strided_view::ElementOp;

        let m: usize = group1_dims.iter().product::<usize>().max(1);
        let mut buf = alloc_batched_col_major(&dims, n_batch);
        strided_kernel::map_into(&mut buf.view_mut(), &arr.view(), |x| ConjOp::apply(x))?;
        let ptr = buf.view().ptr();
        let batch_strides = buf.strides()[..n_batch].to_vec();
        let row_stride = if m == 0 { 0 } else { 1isize };
        let col_stride = m as isize;
        return Ok(ContiguousOperand {
            ptr,
            row_stride,
            col_stride,
            batch_strides,
            conj: false,
            _buf: Some(buf),
        });
    }

    let fused_g1 = try_fuse_group(group1_dims, group1_strides);
    let fused_g2 = try_fuse_group(group2_dims, group2_strides);

    let needs_copy = fused_g1.is_none() || fused_g2.is_none();

    if needs_copy {
        let m: usize = group1_dims.iter().product::<usize>().max(1);
        let mut buf = alloc_batched_col_major(&dims, n_batch);
        strided_kernel::copy_into(&mut buf.view_mut(), &arr.view())?;
        let ptr = buf.view().ptr();
        let batch_strides = buf.strides()[..n_batch].to_vec();
        let row_stride = if m == 0 { 0 } else { 1isize };
        let col_stride = m as isize;
        Ok(ContiguousOperand {
            ptr,
            row_stride,
            col_stride,
            batch_strides,
            conj,
            _buf: Some(buf),
        })
    } else {
        let (_, rs) = fused_g1.unwrap();
        let (_, cs) = fused_g2.unwrap();
        let batch_strides = strides[..n_batch].to_vec();
        let ptr = arr.view().ptr();
        Ok(ContiguousOperand {
            ptr,
            row_stride: rs,
            col_stride: cs,
            batch_strides,
            conj,
            _buf: Some(arr),
        })
    }
}

/// Prepare a borrowed mutable output view for GEMM.
///
/// Checks if the two inner dimension groups (lo, ro) are fusable.
/// If not, allocates a col-major buffer and copies the existing data into it
/// when `beta` is non-zero (so the GEMM accumulation is correct).
///
/// After GEMM, call [`ContiguousOperandMut::finalize_into`] with the original
/// view to copy results back if needed.
///
/// # Safety contract
///
/// When inner dims are fusable (no copy needed), the returned `ContiguousOperandMut`
/// holds a raw pointer into `view`'s data. The caller must ensure `view` outlives
/// the returned operand and that no aliasing mutable references exist during GEMM.
#[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
pub fn prepare_output_view<T: Scalar>(
    view: &mut StridedViewMut<T>,
    n_batch: usize,
    n_group1: usize,
    n_group2: usize,
    beta: T,
) -> crate::Result<ContiguousOperandMut<T>> {
    let dims = view.dims().to_vec();
    let strides = view.strides().to_vec();

    let group1_dims = &dims[n_batch..n_batch + n_group1];
    let group1_strides = &strides[n_batch..n_batch + n_group1];
    let group2_dims = &dims[n_batch + n_group1..n_batch + n_group1 + n_group2];
    let group2_strides = &strides[n_batch + n_group1..n_batch + n_group1 + n_group2];

    let fused_g1 = try_fuse_group(group1_dims, group1_strides);
    let fused_g2 = try_fuse_group(group2_dims, group2_strides);

    let needs_copy = fused_g1.is_none() || fused_g2.is_none();

    if needs_copy {
        let m: usize = group1_dims.iter().product::<usize>().max(1);
        let mut buf = alloc_batched_col_major(&dims, n_batch);
        if beta != T::zero() {
            // Need to preserve existing values for accumulation
            strided_kernel::copy_into(&mut buf.view_mut(), &view.as_view())?;
        }
        let ptr = buf.view_mut().as_mut_ptr();
        let batch_strides = buf.strides()[..n_batch].to_vec();
        let row_stride = if m == 0 { 0 } else { 1isize };
        let col_stride = m as isize;
        Ok(ContiguousOperandMut {
            ptr,
            row_stride,
            col_stride,
            batch_strides,
            needs_writeback: true,
            _buf: Some(buf),
        })
    } else {
        let (_, rs) = fused_g1.unwrap();
        let (_, cs) = fused_g2.unwrap();
        let batch_strides = strides[..n_batch].to_vec();
        Ok(ContiguousOperandMut {
            ptr: view.as_mut_ptr(),
            row_stride: rs,
            col_stride: cs,
            batch_strides,
            needs_writeback: false,
            _buf: None,
        })
    }
}

/// Prepare an owned mutable output array for GEMM.
///
/// If already contiguous after dimension grouping, uses the array in-place.
/// Otherwise, allocates a col-major buffer and copies existing data when
/// `beta` is non-zero.
///
/// Unlike [`prepare_output_view`], `needs_writeback` is always `false` for owned
/// arrays because the caller owns the buffer and can use it directly.
///
/// Currently unused in production (C is always a `StridedViewMut` from the caller).
/// Kept for future use when `einsum2_into` accepts owned output arrays.
#[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
#[allow(dead_code)]
pub fn prepare_output_owned<T: Scalar>(
    arr: &mut StridedArray<T>,
    n_batch: usize,
    n_group1: usize,
    n_group2: usize,
    beta: T,
) -> crate::Result<ContiguousOperandMut<T>> {
    let dims = arr.dims().to_vec();
    let strides = arr.strides().to_vec();

    let group1_dims = &dims[n_batch..n_batch + n_group1];
    let group1_strides = &strides[n_batch..n_batch + n_group1];
    let group2_dims = &dims[n_batch + n_group1..n_batch + n_group1 + n_group2];
    let group2_strides = &strides[n_batch + n_group1..n_batch + n_group1 + n_group2];

    let fused_g1 = try_fuse_group(group1_dims, group1_strides);
    let fused_g2 = try_fuse_group(group2_dims, group2_strides);

    let needs_copy = fused_g1.is_none() || fused_g2.is_none();

    if needs_copy {
        let m: usize = group1_dims.iter().product::<usize>().max(1);
        let mut buf = alloc_batched_col_major(&dims, n_batch);
        if beta != T::zero() {
            strided_kernel::copy_into(&mut buf.view_mut(), &arr.view())?;
        }
        let ptr = buf.view_mut().as_mut_ptr();
        let batch_strides = buf.strides()[..n_batch].to_vec();
        let row_stride = if m == 0 { 0 } else { 1isize };
        let col_stride = m as isize;
        Ok(ContiguousOperandMut {
            ptr,
            row_stride,
            col_stride,
            batch_strides,
            needs_writeback: false,
            _buf: Some(buf),
        })
    } else {
        let (_, rs) = fused_g1.unwrap();
        let (_, cs) = fused_g2.unwrap();
        let batch_strides = strides[..n_batch].to_vec();
        Ok(ContiguousOperandMut {
            ptr: arr.view_mut().as_mut_ptr(),
            row_stride: rs,
            col_stride: cs,
            batch_strides,
            needs_writeback: false,
            _buf: None,
        })
    }
}

#[cfg(test)]
#[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
mod tests {
    use super::*;

    #[test]
    fn test_borrowed_contiguous_no_copy() {
        // Col-major [2,3]: strides [1,2]. n_batch=0, n_group1=1, n_group2=1.
        // Group1 = dim [2], stride [1] -> fuses to (2, 1).
        // Group2 = dim [3], stride [2] -> fuses to (3, 2).
        // Both fuse -> no copy needed.
        let a = StridedArray::<f64>::col_major(&[2, 3]);
        let view = a.view();

        let op = prepare_input_view(&view, 0, 1, 1, false).unwrap();

        assert!(!op.has_buf());
        assert_eq!(op.row_stride(), 1);
        assert_eq!(op.col_stride(), 2);
        assert!(!op.conj());
    }

    #[test]
    fn test_borrowed_non_contiguous_copies() {
        // dims [2,3,4] with strides [20,4,1], n_batch=0, n_group1=2, n_group2=1.
        // Group1 = dims [2,3], strides [20,4]. Try fuse: sorted by |stride| -> [(3,4),(2,20)].
        // Check: 4*3=12 != 20, so fusion fails -> needs copy.
        let data = vec![0.0f64; 100];
        let a = StridedArray::<f64>::from_parts(data, &[2, 3, 4], &[20, 4, 1], 0).unwrap();
        let view = a.view();

        let op = prepare_input_view(&view, 0, 2, 1, false).unwrap();

        assert!(op.has_buf());
        // After copy to col-major: row_stride=1, col_stride = m = 2*3 = 6
        assert_eq!(op.row_stride(), 1);
        assert_eq!(op.col_stride(), 6);
    }

    #[test]
    fn test_owned_contiguous_no_copy() {
        // Col-major [2,3]: strides [1,2]. n_batch=0, n_group1=1, n_group2=1.
        // Fusable -> ownership transferred, no copy.
        let a = StridedArray::<f64>::col_major(&[2, 3]);

        let op = prepare_input_owned(a, 0, 1, 1, false).unwrap();

        // Ownership transferred: _buf = Some (the original array).
        assert!(op.has_buf());
        assert_eq!(op.row_stride(), 1);
        assert_eq!(op.col_stride(), 2);
    }

    #[test]
    fn test_owned_non_contiguous_copies() {
        // dims [2,3,4] with strides [20,4,1], n_batch=0, n_group1=2, n_group2=1.
        // Non-fusable -> copies to new buffer.
        let data = vec![0.0f64; 100];
        let a = StridedArray::<f64>::from_parts(data, &[2, 3, 4], &[20, 4, 1], 0).unwrap();

        let op = prepare_input_owned(a, 0, 2, 1, false).unwrap();

        assert!(op.has_buf());
        // After copy: row_stride=1, col_stride = m = 2*3 = 6
        assert_eq!(op.row_stride(), 1);
        assert_eq!(op.col_stride(), 6);
    }

    // ---- Output preparation tests ----

    #[test]
    fn test_output_view_contiguous() {
        // Col-major [2,3]: strides [1,2]. n_batch=0, n_group1=1, n_group2=1.
        // Both groups fuse -> no copy, no writeback.
        let mut c = StridedArray::<f64>::col_major(&[2, 3]);
        let mut view = c.view_mut();

        let op = prepare_output_view(&mut view, 0, 1, 1, 0.0).unwrap();

        assert!(!op.needs_writeback());
        assert!(!op.has_buf());
        assert_eq!(op.row_stride(), 1);
        assert_eq!(op.col_stride(), 2);
    }

    #[test]
    fn test_output_view_non_contiguous_beta_zero() {
        // dims [2,3,4] with strides [20,4,1], n_batch=0, n_group1=2, n_group2=1.
        // Group1 = dims [2,3], strides [20,4] -> non-fusable -> needs copy.
        // beta=0 -> no copy-in of existing data, but writeback needed.
        let data = vec![0.0f64; 100];
        let mut c = StridedArray::<f64>::from_parts(data, &[2, 3, 4], &[20, 4, 1], 0).unwrap();
        let mut view = c.view_mut();

        let op = prepare_output_view(&mut view, 0, 2, 1, 0.0).unwrap();

        assert!(op.needs_writeback());
        assert!(op.has_buf());
        // After alloc col-major: row_stride=1, col_stride = m = 2*3 = 6
        assert_eq!(op.row_stride(), 1);
        assert_eq!(op.col_stride(), 6);
    }

    #[test]
    fn test_output_view_non_contiguous_beta_nonzero_and_finalize() {
        // Use 3D: dims [2,3,1] with strides [10,1,1], group1=2 dims, group2=1 dim.
        // group1 = dims [2,3], strides [10,1]. Sorted: [(3,1),(2,10)]. 1*3=3 != 10 -> non-fusable!

        // Pre-populate a data buffer with known values at the right offsets.
        // With strides [10,1,1], element [i,j,0] is at offset i*10 + j*1.
        let mut data = vec![0.0f64; 30];
        data[0] = 10.0; // [0,0,0]
        data[1] = 20.0; // [0,1,0]
        data[2] = 30.0; // [0,2,0]
        data[10] = 40.0; // [1,0,0]
        data[11] = 50.0; // [1,1,0]
        data[12] = 60.0; // [1,2,0]
        let mut c = StridedArray::<f64>::from_parts(data, &[2, 3, 1], &[10, 1, 1], 0).unwrap();

        // Verify the known values
        assert_eq!(c.get(&[0, 0, 0]), 10.0);
        assert_eq!(c.get(&[1, 1, 0]), 50.0);

        let mut view = c.view_mut();

        // group1 dims [2,3], group2 dims [1] -> group1 is non-fusable.
        let mut op = prepare_output_view(&mut view, 0, 2, 1, 1.0).unwrap();

        assert!(op.needs_writeback());
        assert!(op.has_buf());

        // beta=1.0 -> existing data should have been copied into the buffer.
        // Verify by reading from the buffer via the internal _buf.
        let buf = op._buf.as_ref().unwrap();
        assert_eq!(buf.get(&[0, 0, 0]), 10.0);
        assert_eq!(buf.get(&[1, 1, 0]), 50.0);

        // Simulate GEMM by writing to the buffer through copy_into.
        {
            let result_data = vec![100.0f64; 6];
            let result =
                StridedArray::<f64>::from_parts(result_data, &[2, 3, 1], &[3, 1, 1], 0).unwrap();
            strided_kernel::copy_into(&mut op._buf.as_mut().unwrap().view_mut(), &result.view())
                .unwrap();
            // Update ptr to the buffer (in case of reallocation)
            op.ptr = op._buf.as_mut().unwrap().view_mut().as_mut_ptr();
        }

        // finalize_into should copy the buffer back to the original view.
        op.finalize_into(&mut view).unwrap();

        // All elements should now be 100.0 in the original non-contiguous array.
        assert_eq!(c.get(&[0, 0, 0]), 100.0);
        assert_eq!(c.get(&[0, 1, 0]), 100.0);
        assert_eq!(c.get(&[0, 2, 0]), 100.0);
        assert_eq!(c.get(&[1, 0, 0]), 100.0);
        assert_eq!(c.get(&[1, 1, 0]), 100.0);
        assert_eq!(c.get(&[1, 2, 0]), 100.0);
    }

    #[test]
    fn test_output_owned_no_writeback() {
        // Non-fusable owned array: needs_writeback should be false.
        // dims [2,3,4] with strides [20,4,1], n_batch=0, n_group1=2, n_group2=1.
        let data = vec![0.0f64; 100];
        let mut c = StridedArray::<f64>::from_parts(data, &[2, 3, 4], &[20, 4, 1], 0).unwrap();

        let op = prepare_output_owned(&mut c, 0, 2, 1, 0.0).unwrap();

        // Non-fusable -> has buffer, but owned -> no writeback.
        assert!(!op.needs_writeback());
        assert!(op.has_buf());
        assert_eq!(op.row_stride(), 1);
        assert_eq!(op.col_stride(), 6);
    }
}
