//! GEMM-ready operand types and preparation functions for contiguous data.
//!
//! These types encapsulate the logic for preparing strided operands for GEMM:
//! checking fusability, copying to col-major buffers when needed, and managing
//! the writeback for borrowed output operands.

use crate::ScalarBase;
use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;
use strided_perm::try_fuse_group;
use strided_view::{StridedArray, StridedView, StridedViewMut};

/// GEMM-ready input operand with contiguous data.
pub struct ContiguousOperand<T: Copy + 'static> {
    ptr: *const T,
    row_stride: isize,
    col_stride: isize,
    batch_strides: Vec<isize>,
    conj: bool,
    /// Owns the buffer if a copy was made or input was consumed.
    pub(crate) _buf: Option<StridedArray<T>>,
    buf_is_pooled: bool,
}

/// GEMM-ready output operand with contiguous data.
pub struct ContiguousOperandMut<T: Copy + 'static> {
    ptr: *mut T,
    row_stride: isize,
    col_stride: isize,
    batch_strides: Vec<isize>,
    /// Whether the caller must copy the buffer back to the original destination
    /// after GEMM completes (true only for borrowed non-contiguous C).
    needs_writeback: bool,
    /// Owns the buffer if a copy was made.
    pub(crate) _buf: Option<StridedArray<T>>,
    buf_is_pooled: bool,
}

thread_local! {
    static BUFFER_POOL: RefCell<HashMap<TypeId, Box<dyn Any>>> = RefCell::new(HashMap::new());
}

const MAX_POOL_PER_TYPE: usize = 16;
const MAX_POOLED_BYTES: usize = 64 * 1024 * 1024;

fn take_pooled_vec_uninit<T: Copy + 'static>(len: usize) -> Vec<T> {
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let entry = pool
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(Vec::<Vec<T>>::new()));
        let vecs = entry
            .downcast_mut::<Vec<Vec<T>>>()
            .expect("buffer pool type mismatch");

        let mut best_idx = None;
        let mut best_cap = usize::MAX;
        for (idx, v) in vecs.iter().enumerate() {
            let cap = v.capacity();
            if cap >= len && cap < best_cap {
                best_idx = Some(idx);
                best_cap = cap;
            }
        }

        let mut data = best_idx
            .map(|idx| vecs.swap_remove(idx))
            .unwrap_or_else(|| Vec::with_capacity(len));
        if data.capacity() < len {
            data.reserve(len - data.capacity());
        }
        unsafe { data.set_len(len) };
        data
    })
}

fn return_pooled_vec<T: Copy + 'static>(mut data: Vec<T>) {
    let bytes = data.capacity().saturating_mul(std::mem::size_of::<T>());
    if bytes == 0 || bytes > MAX_POOLED_BYTES {
        return;
    }
    data.clear();
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let entry = pool
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(Vec::<Vec<T>>::new()));
        let vecs = entry
            .downcast_mut::<Vec<Vec<T>>>()
            .expect("buffer pool type mismatch");
        if vecs.len() >= MAX_POOL_PER_TYPE {
            if let Some((min_idx, min_cap)) = vecs
                .iter()
                .enumerate()
                .map(|(i, v)| (i, v.capacity()))
                .min_by_key(|(_, cap)| *cap)
            {
                if min_cap < data.capacity() {
                    vecs.swap_remove(min_idx);
                    vecs.push(data);
                }
            }
        } else {
            vecs.push(data);
        }
    });
}

fn alloc_col_major_uninit_with_pool<T: Copy + 'static>(dims: &[usize]) -> (StridedArray<T>, bool) {
    let total: usize = dims.iter().product::<usize>().max(1);
    let bytes = total.saturating_mul(std::mem::size_of::<T>());
    if bytes == 0 || bytes > MAX_POOLED_BYTES {
        return (alloc_col_major_uninit(dims), false);
    }
    let data = take_pooled_vec_uninit::<T>(total);
    let arr = unsafe { StridedArray::col_major_from_buffer_uninit(data, dims) };
    (arr, true)
}

/// Allocate a col-major buffer, optionally reusing from the thread-local pool.
fn alloc_maybe_pooled<T: Copy + 'static>(dims: &[usize], use_pool: bool) -> (StridedArray<T>, bool) {
    if use_pool {
        alloc_col_major_uninit_with_pool(dims)
    } else {
        (alloc_col_major_uninit(dims), false)
    }
}

#[cfg(test)]
fn pooled_count_for_type<T: 'static>() -> usize {
    BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let Some(entry) = pool.get_mut(&TypeId::of::<T>()) else {
            return 0;
        };
        entry
            .downcast_mut::<Vec<Vec<T>>>()
            .map_or(0, |vecs| vecs.len())
    })
}

impl<T: Copy + 'static> ContiguousOperand<T> {
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

impl<T: Copy + 'static> ContiguousOperandMut<T> {
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

impl<T: Copy + Send + Sync> ContiguousOperandMut<T> {
    /// After GEMM: copy the internal buffer back to `dest` if needed.
    ///
    /// This is a no-op when the GEMM wrote directly to the destination
    /// (contiguous case or owned output).
    pub fn finalize_into(self, dest: &mut StridedViewMut<T>) -> crate::Result<()> {
        if self.needs_writeback {
            if let Some(ref buf) = self._buf {
                strided_perm::copy_into(dest, &buf.view())?;
            }
        }
        Ok(())
    }
}

impl<T: Copy + 'static> Drop for ContiguousOperand<T> {
    fn drop(&mut self) {
        if self.buf_is_pooled {
            if let Some(arr) = self._buf.take() {
                return_pooled_vec(arr.into_data());
            }
        }
    }
}

impl<T: Copy + 'static> Drop for ContiguousOperandMut<T> {
    fn drop(&mut self) {
        if self.buf_is_pooled {
            if let Some(arr) = self._buf.take() {
                return_pooled_vec(arr.into_data());
            }
        }
    }
}

/// Result of checking whether dimension groups are contiguous enough for GEMM.
struct ContiguityCheck {
    fused_g1: Option<(usize, isize)>,
    fused_g2: Option<(usize, isize)>,
    needs_copy: bool,
}

/// Check if two dimension groups are fusable (contiguous) for GEMM.
///
/// When `requires_unit_stride` is true (e.g., CBLAS backend), also checks that
/// at least one of the fused strides is 0 or 1.
fn check_contiguity(
    group1_dims: &[usize],
    group1_strides: &[isize],
    group2_dims: &[usize],
    group2_strides: &[isize],
    requires_unit_stride: bool,
) -> ContiguityCheck {
    let fused_g1 = try_fuse_group(group1_dims, group1_strides);
    let fused_g2 = try_fuse_group(group2_dims, group2_strides);

    let mut needs_copy = fused_g1.is_none() || fused_g2.is_none();

    if requires_unit_stride && !needs_copy {
        let (_, rs) = fused_g1.unwrap();
        let (_, cs) = fused_g2.unwrap();
        if rs != 0 && rs != 1 && cs != 0 && cs != 1 {
            needs_copy = true;
        }
    }

    ContiguityCheck {
        fused_g1,
        fused_g2,
        needs_copy,
    }
}

/// Compute col-major layout parameters from a freshly-allocated col-major buffer.
///
/// Returns `(row_stride, col_stride, batch_strides)`.
fn col_major_layout(buf: &StridedArray<impl Copy>, n_group1: usize, n_inner: usize) -> (isize, isize, Vec<isize>) {
    let m: usize = buf.dims()[..n_group1].iter().product::<usize>().max(1);
    let row_stride = if m == 0 { 0 } else { 1isize };
    let col_stride = m as isize;
    let batch_strides = buf.strides()[n_inner..].to_vec();
    (row_stride, col_stride, batch_strides)
}

/// Allocate a column-major StridedArray with uninitialized data.
///
/// With batch-last canonical order `[inner..., batch...]`, pure column-major
/// naturally gives batch dims the largest strides — each batch slice is a
/// contiguous column-major matrix.
pub(crate) fn alloc_col_major_uninit<T: Copy>(dims: &[usize]) -> StridedArray<T> {
    let total: usize = dims.iter().product::<usize>().max(1);
    // SAFETY: `T: Copy` guarantees no drop glue, so leaving elements
    // uninitialised is safe. Every call-site writes all elements before
    // reading: A and B via `copy_into`, C via `copy_into` (beta != 0)
    // or GEMM with replace semantics (beta == 0).
    let mut data = Vec::with_capacity(total);
    unsafe { data.set_len(total) };

    // Pure column-major: stride 1 for first dim, each subsequent dim
    // has stride = previous stride * previous dim size.
    let mut strides = vec![0isize; dims.len()];
    if !dims.is_empty() {
        strides[0] = 1;
        for i in 1..dims.len() {
            strides[i] = strides[i - 1] * dims[i - 1] as isize;
        }
    }

    let arr = StridedArray::from_parts(data, dims, &strides, 0).expect("col-major allocation");
    arr
}

/// Prepare a borrowed input view for GEMM.
///
/// Expects batch-last canonical order: `[group1..., group2..., batch...]`.
/// Checks if the two inner dimension groups are fusable.
/// If not, copies to a contiguous col-major buffer.
///
/// - `requires_unit_stride`: backend needs at least one unit stride (e.g. CBLAS).
/// - `use_pool`: reuse thread-local buffers to avoid repeated allocation.
/// - `materialize_conj_fn`: when `Some(f)` and `conj == true`, applies `f` to each
///   element during copy (for backends that cannot pass conj flags to GEMM).
pub fn prepare_input_view<T: ScalarBase + 'static>(
    view: &StridedView<T>,
    n_group1: usize,
    n_group2: usize,
    conj: bool,
    requires_unit_stride: bool,
    use_pool: bool,
    materialize_conj_fn: Option<fn(T) -> T>,
) -> crate::Result<ContiguousOperand<T>> {
    let dims = view.dims();
    let strides = view.strides();
    let n_inner = n_group1 + n_group2;

    // For backends that cannot pass conjugation flags to GEMM (e.g., CBLAS),
    // materialize conj into the data before the GEMM call.
    if let Some(conj_fn) = materialize_conj_fn {
        if conj {
            let (mut buf, buf_is_pooled) = alloc_maybe_pooled(dims, use_pool);
            strided_kernel::map_into(&mut buf.view_mut(), view, conj_fn)?;
            let ptr = buf.view().ptr();
            let (row_stride, col_stride, batch_strides) =
                col_major_layout(&buf, n_group1, n_inner);
            return Ok(ContiguousOperand {
                ptr, row_stride, col_stride, batch_strides,
                conj: false, _buf: Some(buf), buf_is_pooled,
            });
        }
    }

    let check = check_contiguity(
        &dims[..n_group1], &strides[..n_group1],
        &dims[n_group1..n_inner], &strides[n_group1..n_inner],
        requires_unit_stride,
    );

    if check.needs_copy {
        let (mut buf, buf_is_pooled) = alloc_maybe_pooled(dims, use_pool);
        strided_kernel::copy_into_col_major(&mut buf.view_mut(), view)?;
        let ptr = buf.view().ptr();
        let (row_stride, col_stride, batch_strides) = col_major_layout(&buf, n_group1, n_inner);
        Ok(ContiguousOperand {
            ptr, row_stride, col_stride, batch_strides, conj,
            _buf: Some(buf), buf_is_pooled,
        })
    } else {
        let (_, rs) = check.fused_g1.unwrap();
        let (_, cs) = check.fused_g2.unwrap();
        Ok(ContiguousOperand {
            ptr: view.ptr(),
            row_stride: rs, col_stride: cs,
            batch_strides: strides[n_inner..].to_vec(),
            conj, _buf: None, buf_is_pooled: false,
        })
    }
}

/// Prepare an owned input array for GEMM.
///
/// Expects batch-last canonical order: `[group1..., group2..., batch...]`.
/// If already contiguous after dimension grouping, transfers ownership without copying.
/// Otherwise, copies to a new col-major buffer.
///
/// Parameters are the same as [`prepare_input_view`].
pub fn prepare_input_owned<T: ScalarBase + 'static>(
    arr: StridedArray<T>,
    n_group1: usize,
    n_group2: usize,
    conj: bool,
    requires_unit_stride: bool,
    use_pool: bool,
    materialize_conj_fn: Option<fn(T) -> T>,
) -> crate::Result<ContiguousOperand<T>> {
    let dims = arr.dims().to_vec();
    let strides = arr.strides().to_vec();
    let n_inner = n_group1 + n_group2;

    // For backends that cannot pass conjugation flags to GEMM,
    // materialize conj into the data before the GEMM call.
    if let Some(conj_fn) = materialize_conj_fn {
        if conj {
            let (mut buf, buf_is_pooled) = alloc_maybe_pooled(&dims, use_pool);
            strided_kernel::map_into(&mut buf.view_mut(), &arr.view(), conj_fn)?;
            let ptr = buf.view().ptr();
            let (row_stride, col_stride, batch_strides) =
                col_major_layout(&buf, n_group1, n_inner);
            return Ok(ContiguousOperand {
                ptr, row_stride, col_stride, batch_strides,
                conj: false, _buf: Some(buf), buf_is_pooled,
            });
        }
    }

    let check = check_contiguity(
        &dims[..n_group1], &strides[..n_group1],
        &dims[n_group1..n_inner], &strides[n_group1..n_inner],
        requires_unit_stride,
    );

    if check.needs_copy {
        let (mut buf, buf_is_pooled) = alloc_maybe_pooled(&dims, use_pool);
        strided_kernel::copy_into_col_major(&mut buf.view_mut(), &arr.view())?;
        let ptr = buf.view().ptr();
        let (row_stride, col_stride, batch_strides) = col_major_layout(&buf, n_group1, n_inner);
        Ok(ContiguousOperand {
            ptr, row_stride, col_stride, batch_strides, conj,
            _buf: Some(buf), buf_is_pooled,
        })
    } else {
        let (_, rs) = check.fused_g1.unwrap();
        let (_, cs) = check.fused_g2.unwrap();
        let ptr = arr.view().ptr();
        Ok(ContiguousOperand {
            ptr, row_stride: rs, col_stride: cs,
            batch_strides: strides[n_inner..].to_vec(),
            conj,
            _buf: Some(arr), buf_is_pooled: false,
        })
    }
}

/// Prepare a borrowed mutable output view for GEMM.
///
/// Expects batch-last canonical order: `[group1..., group2..., batch...]`.
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
pub fn prepare_output_view<T: ScalarBase + 'static>(
    view: &mut StridedViewMut<T>,
    n_group1: usize,
    n_group2: usize,
    beta: T,
    requires_unit_stride: bool,
    use_pool: bool,
) -> crate::Result<ContiguousOperandMut<T>> {
    let dims = view.dims().to_vec();
    let strides = view.strides().to_vec();
    let n_inner = n_group1 + n_group2;

    let check = check_contiguity(
        &dims[..n_group1], &strides[..n_group1],
        &dims[n_group1..n_inner], &strides[n_group1..n_inner],
        requires_unit_stride,
    );

    if check.needs_copy {
        let (mut buf, buf_is_pooled) = alloc_maybe_pooled(&dims, use_pool);
        if beta != T::zero() {
            strided_kernel::copy_into_col_major(&mut buf.view_mut(), &view.as_view())?;
        }
        let ptr = buf.view_mut().as_mut_ptr();
        let (row_stride, col_stride, batch_strides) = col_major_layout(&buf, n_group1, n_inner);
        Ok(ContiguousOperandMut {
            ptr, row_stride, col_stride, batch_strides,
            needs_writeback: true,
            _buf: Some(buf), buf_is_pooled,
        })
    } else {
        let (_, rs) = check.fused_g1.unwrap();
        let (_, cs) = check.fused_g2.unwrap();
        Ok(ContiguousOperandMut {
            ptr: view.as_mut_ptr(),
            row_stride: rs, col_stride: cs,
            batch_strides: strides[n_inner..].to_vec(),
            needs_writeback: false,
            _buf: None, buf_is_pooled: false,
        })
    }
}


#[cfg(test)]
mod tests_generic_backend {
    use super::*;
    use crate::backend::{Backend, NaiveBackend};

    #[test]
    fn test_input_for_backend_contiguous() {
        let a = StridedArray::<f64>::col_major(&[2, 3]);
        let view = a.view();
        let op = prepare_input_view(
            &view, 1, 1, false, <NaiveBackend as Backend<f64>>::REQUIRES_UNIT_STRIDE, false, None,
        ).unwrap();
        assert!(op._buf.is_none());
        assert_eq!(op.row_stride(), 1);
        assert_eq!(op.col_stride(), 2);
        assert!(!op.conj());
    }

    #[test]
    fn test_input_for_backend_non_contiguous() {
        let data = vec![0.0f64; 100];
        let a = StridedArray::<f64>::from_parts(data, &[2, 3, 4], &[20, 4, 1], 0).unwrap();
        let view = a.view();
        let op = prepare_input_view(
            &view, 2, 1, false, <NaiveBackend as Backend<f64>>::REQUIRES_UNIT_STRIDE, false, None,
        ).unwrap();
        assert!(op._buf.is_some());
        assert_eq!(op.row_stride(), 1);
        assert_eq!(op.col_stride(), 6);
    }

    #[test]
    fn test_output_for_backend_contiguous() {
        let mut c = StridedArray::<f64>::col_major(&[2, 3]);
        let mut view = c.view_mut();
        let op = prepare_output_view(
            &mut view, 1, 1, 0.0, <NaiveBackend as Backend<f64>>::REQUIRES_UNIT_STRIDE, false,
        ).unwrap();
        assert!(!op.needs_writeback);
        assert!(op._buf.is_none());
        assert_eq!(op.row_stride(), 1);
        assert_eq!(op.col_stride(), 2);
    }

    #[test]
    fn test_output_for_backend_non_contiguous_beta_zero() {
        let data = vec![0.0f64; 100];
        let mut c = StridedArray::<f64>::from_parts(data, &[2, 3, 4], &[20, 4, 1], 0).unwrap();
        let mut view = c.view_mut();
        let op = prepare_output_view(
            &mut view, 2, 1, 0.0, <NaiveBackend as Backend<f64>>::REQUIRES_UNIT_STRIDE, false,
        ).unwrap();
        assert!(op.needs_writeback);
        assert!(op._buf.is_some());
        assert_eq!(op.row_stride(), 1);
        assert_eq!(op.col_stride(), 6);
    }

    #[test]
    fn test_output_for_backend_non_contiguous_beta_nonzero_and_finalize() {
        let mut data = vec![0.0f64; 30];
        data[0] = 10.0;
        data[1] = 20.0;
        data[10] = 40.0;
        let mut c = StridedArray::<f64>::from_parts(data, &[2, 3, 1], &[10, 1, 1], 0).unwrap();
        let mut view = c.view_mut();
        let op = prepare_output_view(
            &mut view, 2, 1, 1.0, <NaiveBackend as Backend<f64>>::REQUIRES_UNIT_STRIDE, false,
        ).unwrap();
        assert!(op.needs_writeback);
        let buf = op._buf.as_ref().unwrap();
        assert_eq!(buf.get(&[0, 0, 0]), 10.0);
        assert_eq!(buf.get(&[0, 1, 0]), 20.0);
        assert_eq!(buf.get(&[1, 0, 0]), 40.0);
        op.finalize_into(&mut view).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{ActiveBackend, Backend};

    // Helper to construct prepare_input_view params matching the active backend.
    const UNIT_STRIDE: bool = <ActiveBackend as Backend<f64>>::REQUIRES_UNIT_STRIDE;

    #[test]
    fn test_borrowed_contiguous_no_copy() {
        let a = StridedArray::<f64>::col_major(&[2, 3]);
        let view = a.view();

        let op = prepare_input_view(&view, 1, 1, false, UNIT_STRIDE, true, None).unwrap();

        assert!(!op.has_buf());
        assert_eq!(op.row_stride(), 1);
        assert_eq!(op.col_stride(), 2);
        assert!(!op.conj());
    }

    #[test]
    fn test_borrowed_non_contiguous_copies() {
        let data = vec![0.0f64; 100];
        let a = StridedArray::<f64>::from_parts(data, &[2, 3, 4], &[20, 4, 1], 0).unwrap();
        let view = a.view();

        let op = prepare_input_view(&view, 2, 1, false, UNIT_STRIDE, true, None).unwrap();

        assert!(op.has_buf());
        assert_eq!(op.row_stride(), 1);
        assert_eq!(op.col_stride(), 6);
    }

    #[test]
    fn test_owned_contiguous_no_copy() {
        let a = StridedArray::<f64>::col_major(&[2, 3]);

        let op = prepare_input_owned(a, 1, 1, false, UNIT_STRIDE, true, None).unwrap();

        assert!(op.has_buf());
        assert_eq!(op.row_stride(), 1);
        assert_eq!(op.col_stride(), 2);
    }

    #[test]
    fn test_owned_non_contiguous_copies() {
        let data = vec![0.0f64; 100];
        let a = StridedArray::<f64>::from_parts(data, &[2, 3, 4], &[20, 4, 1], 0).unwrap();

        let op = prepare_input_owned(a, 2, 1, false, UNIT_STRIDE, true, None).unwrap();

        assert!(op.has_buf());
        assert_eq!(op.row_stride(), 1);
        assert_eq!(op.col_stride(), 6);
    }

    #[test]
    fn test_output_view_contiguous() {
        let mut c = StridedArray::<f64>::col_major(&[2, 3]);
        let mut view = c.view_mut();

        let op = prepare_output_view(&mut view, 1, 1, 0.0, UNIT_STRIDE, true).unwrap();

        assert!(!op.needs_writeback());
        assert!(!op.has_buf());
        assert_eq!(op.row_stride(), 1);
        assert_eq!(op.col_stride(), 2);
    }

    #[test]
    fn test_output_view_non_contiguous_beta_zero() {
        let data = vec![0.0f64; 100];
        let mut c = StridedArray::<f64>::from_parts(data, &[2, 3, 4], &[20, 4, 1], 0).unwrap();
        let mut view = c.view_mut();

        let op = prepare_output_view(&mut view, 2, 1, 0.0, UNIT_STRIDE, true).unwrap();

        assert!(op.needs_writeback());
        assert!(op.has_buf());
        assert_eq!(op.row_stride(), 1);
        assert_eq!(op.col_stride(), 6);
    }

    #[test]
    fn test_output_view_non_contiguous_beta_nonzero_and_finalize() {
        let mut data = vec![0.0f64; 30];
        data[0] = 10.0;
        data[1] = 20.0;
        data[2] = 30.0;
        data[10] = 40.0;
        data[11] = 50.0;
        data[12] = 60.0;
        let mut c = StridedArray::<f64>::from_parts(data, &[2, 3, 1], &[10, 1, 1], 0).unwrap();

        assert_eq!(c.get(&[0, 0, 0]), 10.0);
        assert_eq!(c.get(&[1, 1, 0]), 50.0);

        let mut view = c.view_mut();

        let mut op = prepare_output_view(&mut view, 2, 1, 1.0, UNIT_STRIDE, true).unwrap();

        assert!(op.needs_writeback());
        assert!(op.has_buf());

        let buf = op._buf.as_ref().unwrap();
        assert_eq!(buf.get(&[0, 0, 0]), 10.0);
        assert_eq!(buf.get(&[1, 1, 0]), 50.0);

        {
            let result_data = vec![100.0f64; 6];
            let result =
                StridedArray::<f64>::from_parts(result_data, &[2, 3, 1], &[3, 1, 1], 0).unwrap();
            strided_kernel::copy_into(&mut op._buf.as_mut().unwrap().view_mut(), &result.view())
                .unwrap();
            op.ptr = op._buf.as_mut().unwrap().view_mut().as_mut_ptr();
        }

        op.finalize_into(&mut view).unwrap();

        assert_eq!(c.get(&[0, 0, 0]), 100.0);
        assert_eq!(c.get(&[0, 1, 0]), 100.0);
        assert_eq!(c.get(&[0, 2, 0]), 100.0);
        assert_eq!(c.get(&[1, 0, 0]), 100.0);
        assert_eq!(c.get(&[1, 1, 0]), 100.0);
        assert_eq!(c.get(&[1, 2, 0]), 100.0);
    }

    #[test]
    fn test_prepare_input_view_temp_buffer_is_recycled() {
        let before = pooled_count_for_type::<f64>();
        let data = vec![0.0f64; 100];
        let a = StridedArray::<f64>::from_parts(data, &[2, 3, 4], &[20, 4, 1], 0).unwrap();
        let view = a.view();

        {
            let op = prepare_input_view(&view, 2, 1, false, UNIT_STRIDE, true, None).unwrap();
            assert!(op.has_buf());
        }

        let after = pooled_count_for_type::<f64>();
        assert!(after >= before.saturating_add(1));
    }
}
