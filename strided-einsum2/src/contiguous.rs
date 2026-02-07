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
pub struct ContiguousOperandMut<'a, T> {
    ptr: *mut T,
    row_stride: isize,
    col_stride: isize,
    batch_strides: Vec<isize>,
    /// Write-back destination for borrowed non-contiguous C.
    writeback: Option<StridedViewMut<'a, T>>,
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
    #[inline]
    pub(crate) fn has_buf(&self) -> bool {
        self._buf.is_some()
    }
}

impl<'a, T> ContiguousOperandMut<'a, T> {
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
    #[inline]
    pub(crate) fn has_buf(&self) -> bool {
        self._buf.is_some()
    }
}

#[cfg(feature = "faer")]
use crate::bgemm_faer::alloc_batched_col_major;

/// Prepare a borrowed input view for GEMM.
///
/// Checks if the two inner dimension groups are fusable.
/// If not, copies to a contiguous col-major buffer.
#[cfg(feature = "faer")]
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
#[cfg(feature = "faer")]
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

#[cfg(test)]
#[cfg(feature = "faer")]
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
}
