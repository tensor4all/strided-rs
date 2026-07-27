//! Borrowed raw strided layout types.
//!
//! These are the prepared-replay counterparts to [`crate::StridedView`] and
//! [`crate::StridedViewMut`]. They borrow shape/stride metadata instead of
//! owning it, so compiled kernels can reuse already-validated layout
//! descriptors without rebuilding dynamic-rank view wrappers.

use crate::element_op::Identity;
use crate::view::validate_bounds;
use num_complex::{Complex32, Complex64};

use crate::{Result, StridedError, StridedView, StridedViewMut};

/// Dtypes supported by dtype-erased kernel entry points.
///
/// The enum is intentionally limited to the scalar set currently used by the
/// tensor runtime callers. Later FFI layers should map their ABI dtype tags to
/// this enum before dispatching into prepared kernels.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum KernelDType {
    F32 = 1,
    F64 = 2,
    I32 = 3,
    I64 = 4,
    Bool = 5,
    C32 = 6,
    C64 = 7,
}

impl KernelDType {
    #[inline]
    pub const fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::Bool => "bool",
            Self::C32 => "c32",
            Self::C64 => "c64",
        }
    }

    #[inline]
    pub const fn size_of(self) -> usize {
        match self {
            Self::F32 => core::mem::size_of::<f32>(),
            Self::F64 => core::mem::size_of::<f64>(),
            Self::I32 => core::mem::size_of::<i32>(),
            Self::I64 => core::mem::size_of::<i64>(),
            Self::Bool => core::mem::size_of::<bool>(),
            Self::C32 => core::mem::size_of::<Complex32>(),
            Self::C64 => core::mem::size_of::<Complex64>(),
        }
    }

    #[inline]
    pub const fn alignment(self) -> usize {
        match self {
            Self::F32 => core::mem::align_of::<f32>(),
            Self::F64 => core::mem::align_of::<f64>(),
            Self::I32 => core::mem::align_of::<i32>(),
            Self::I64 => core::mem::align_of::<i64>(),
            Self::Bool => core::mem::align_of::<bool>(),
            Self::C32 => core::mem::align_of::<Complex32>(),
            Self::C64 => core::mem::align_of::<Complex64>(),
        }
    }

    #[inline]
    pub const fn requires_valid_byte_values(self) -> bool {
        matches!(self, Self::Bool)
    }
}

fn validate_erased_buffer(dtype: KernelDType, data: &[u8]) -> Result<usize> {
    let element_size = dtype.size_of();
    if data.len() % element_size != 0 {
        return Err(StridedError::ByteLengthMismatch {
            dtype: dtype.label(),
            byte_len: data.len(),
            element_size,
        });
    }

    let element_count = data.len() / element_size;
    if element_count == 0 {
        return Ok(0);
    }

    let alignment = dtype.alignment();
    if data.as_ptr() as usize % alignment != 0 {
        return Err(StridedError::DataAlignmentMismatch {
            dtype: dtype.label(),
            alignment,
        });
    }
    if dtype.requires_valid_byte_values() {
        if let Some(&value) = data.iter().find(|&&value| value > 1) {
            return Err(StridedError::InvalidBoolByte { value });
        }
    }
    Ok(element_count)
}

/// Borrowed dtype-erased raw strided input layout.
///
/// `dims`, `strides`, and `offset` are expressed in dtype elements, not bytes.
#[derive(Clone, Copy, Debug)]
pub struct ErasedRawStridedRef<'a> {
    dtype: KernelDType,
    data: &'a [u8],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
}

impl<'a> ErasedRawStridedRef<'a> {
    /// Create a byte-backed erased input descriptor after validating dtype byte
    /// length, alignment, rank/stride agreement, and reachable element bounds.
    pub fn new(
        dtype: KernelDType,
        data: &'a [u8],
        dims: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Result<Self> {
        let element_count = validate_erased_buffer(dtype, data)?;
        validate_bounds(element_count, dims, strides, offset)?;
        Ok(Self {
            dtype,
            data,
            dims,
            strides,
            offset,
        })
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    #[inline]
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    #[inline]
    pub fn dims(&self) -> &'a [usize] {
        self.dims
    }

    #[inline]
    pub fn strides(&self) -> &'a [isize] {
        self.strides
    }

    #[inline]
    pub fn offset(&self) -> isize {
        self.offset
    }

    /// Re-check the current byte buffer against the descriptor dtype.
    ///
    /// This is normally redundant after construction, but callers that can
    /// mutate a sibling output descriptor's bytes may need a cheap way to
    /// re-establish dtype byte validity before typed replay.
    pub fn validate_data(&self) -> Result<()> {
        validate_erased_buffer(self.dtype, self.data).map(|_| ())
    }
}

/// Borrowed dtype-erased raw strided output layout.
///
/// `dims`, `strides`, and `offset` are expressed in dtype elements, not bytes.
#[derive(Debug)]
pub struct ErasedRawStridedMut<'a> {
    dtype: KernelDType,
    data: &'a mut [u8],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
    needs_data_revalidation: bool,
}

impl<'a> ErasedRawStridedMut<'a> {
    /// Create a byte-backed erased output descriptor after validating dtype
    /// byte length, alignment, rank/stride agreement, and reachable element
    /// bounds.
    pub fn new(
        dtype: KernelDType,
        data: &'a mut [u8],
        dims: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Result<Self> {
        let element_count = validate_erased_buffer(dtype, data)?;
        validate_bounds(element_count, dims, strides, offset)?;
        Ok(Self {
            dtype,
            data,
            dims,
            strides,
            offset,
            needs_data_revalidation: false,
        })
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    #[inline]
    pub fn data(&self) -> &[u8] {
        self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [u8] {
        if self.dtype.requires_valid_byte_values() {
            self.needs_data_revalidation = true;
        }
        self.data
    }

    #[inline]
    pub fn dims(&self) -> &'a [usize] {
        self.dims
    }

    #[inline]
    pub fn strides(&self) -> &'a [isize] {
        self.strides
    }

    #[inline]
    pub fn offset(&self) -> isize {
        self.offset
    }

    /// Re-check the current byte buffer against the descriptor dtype.
    ///
    /// This guards safe replay when callers mutate bytes through
    /// [`ErasedRawStridedMut::data_mut`] after construction.
    pub fn validate_data(&self) -> Result<()> {
        validate_erased_buffer(self.dtype, self.data).map(|_| ())
    }

    /// Re-check dtype byte validity only when mutable bytes escaped since the
    /// last validation.
    pub fn validate_data_if_needed(&mut self) -> Result<()> {
        if self.needs_data_revalidation {
            self.validate_data()?;
            self.needs_data_revalidation = false;
        }
        Ok(())
    }

    /// Mark the current byte buffer as satisfying this descriptor's dtype
    /// value invariants.
    ///
    /// # Safety
    ///
    /// The caller must ensure every byte sequence in `data` is valid for
    /// `dtype`. This matters for `bool`, where Rust requires stored values to
    /// be exactly `0` or `1` before a typed `bool` slice is formed.
    pub unsafe fn assume_data_valid(&mut self) {
        self.needs_data_revalidation = false;
    }
}

/// Borrowed raw strided input layout.
///
/// Use [`RawStridedRef::new`] for checked construction, or
/// [`RawStridedRef::new_unchecked`] when a higher-level compiled plan has
/// already validated bounds.
#[derive(Clone, Copy, Debug)]
pub struct RawStridedRef<'a, T> {
    data: &'a [T],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
}

impl<'a, T> RawStridedRef<'a, T> {
    /// Create a raw strided input after validating reachable offsets.
    pub fn new(
        data: &'a [T],
        dims: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Result<Self> {
        validate_bounds(data.len(), dims, strides, offset)?;
        Ok(Self {
            data,
            dims,
            strides,
            offset,
        })
    }

    /// Create a raw strided input without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure every index reachable by `dims`/`strides` from
    /// `offset` lies inside `data`.
    pub unsafe fn new_unchecked(
        data: &'a [T],
        dims: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Self {
        Self {
            data,
            dims,
            strides,
            offset,
        }
    }

    #[inline]
    pub fn data(&self) -> &'a [T] {
        self.data
    }

    #[inline]
    pub fn dims(&self) -> &'a [usize] {
        self.dims
    }

    #[inline]
    pub fn strides(&self) -> &'a [isize] {
        self.strides
    }

    #[inline]
    pub fn offset(&self) -> isize {
        self.offset
    }

    #[inline]
    pub fn ptr(&self) -> *const T {
        unsafe { self.data.as_ptr().offset(self.offset) }
    }

    /// Convert to an immutable owning-metadata view.
    ///
    /// This is for compatibility paths. Hot prepared paths should use the raw
    /// accessors directly and avoid this conversion.
    #[inline]
    pub fn as_view(&self) -> StridedView<'a, T, Identity> {
        unsafe { StridedView::new_unchecked(self.data, self.dims, self.strides, self.offset) }
    }
}

/// Borrowed raw strided output layout.
///
/// This is the mutable counterpart to [`RawStridedRef`]. It avoids allocating
/// owned shape/stride metadata in prepared replay paths.
#[derive(Debug)]
pub struct RawStridedMut<'a, T> {
    data: &'a mut [T],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
}

impl<'a, T> RawStridedMut<'a, T> {
    /// Create a raw strided output after validating reachable offsets.
    pub fn new(
        data: &'a mut [T],
        dims: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Result<Self> {
        validate_bounds(data.len(), dims, strides, offset)?;
        Ok(Self {
            data,
            dims,
            strides,
            offset,
        })
    }

    /// Create a raw strided output without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure every index reachable by `dims`/`strides` from
    /// `offset` lies inside `data`, and no aliases violate mutable access.
    pub unsafe fn new_unchecked(
        data: &'a mut [T],
        dims: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Self {
        Self {
            data,
            dims,
            strides,
            offset,
        }
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    #[inline]
    pub fn dims(&self) -> &'a [usize] {
        self.dims
    }

    #[inline]
    pub fn strides(&self) -> &'a [isize] {
        self.strides
    }

    #[inline]
    pub fn offset(&self) -> isize {
        self.offset
    }

    #[inline]
    pub fn ptr(&self) -> *const T {
        unsafe { self.data.as_ptr().offset(self.offset) }
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        unsafe { self.data.as_mut_ptr().offset(self.offset) }
    }

    /// Convert to an immutable owning-metadata view.
    ///
    /// This is for compatibility paths. Hot prepared paths should use the raw
    /// accessors directly and avoid this conversion.
    #[inline]
    pub fn as_view(&self) -> StridedView<'_, T, Identity> {
        unsafe { StridedView::new_unchecked(self.data, self.dims, self.strides, self.offset) }
    }

    /// Convert to a mutable owning-metadata view.
    ///
    /// This is for compatibility paths. Hot prepared paths should use the raw
    /// accessors directly and avoid this conversion.
    #[inline]
    pub fn as_view_mut(&mut self) -> StridedViewMut<'_, T> {
        unsafe { StridedViewMut::new_unchecked(self.data, self.dims, self.strides, self.offset) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_ref_rejects_out_of_bounds_layout() {
        let data = [0.0f64; 4];
        let err = RawStridedRef::new(&data, &[2, 3], &[3, 1], 0).unwrap_err();
        assert!(matches!(err, crate::StridedError::OffsetOverflow));
    }

    #[test]
    fn raw_mut_can_reborrow_as_view() {
        let mut data = [1, 2, 3, 4];
        let mut raw = RawStridedMut::new(&mut data, &[2, 2], &[2, 1], 0).unwrap();
        {
            let view = raw.as_view();
            assert_eq!(view.dims(), &[2, 2]);
        }
        let view_mut = raw.as_view_mut();
        assert_eq!(view_mut.get(&[1, 1]), 4);
    }
}
