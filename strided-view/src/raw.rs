//! Borrowed raw strided layout types.
//!
//! These are the prepared-replay counterparts to [`crate::StridedView`] and
//! [`crate::StridedViewMut`]. They borrow shape/stride metadata instead of
//! owning it, so compiled kernels can reuse already-validated layout
//! descriptors without rebuilding dynamic-rank view wrappers.

use crate::element_op::Identity;
use crate::view::validate_bounds;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr::NonNull;
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

fn validate_erased_buffer_layout(
    dtype: KernelDType,
    data: NonNull<u8>,
    byte_len: usize,
) -> Result<usize> {
    let element_size = dtype.size_of();
    if byte_len % element_size != 0 {
        return Err(StridedError::ByteLengthMismatch {
            dtype: dtype.label(),
            byte_len,
            element_size,
        });
    }
    if byte_len != 0 && data.as_ptr() as usize % dtype.alignment() != 0 {
        return Err(StridedError::DataAlignmentMismatch {
            dtype: dtype.label(),
            alignment: dtype.alignment(),
        });
    }
    Ok(byte_len / element_size)
}

/// Pointer-backed dtype-erased input used by one-shot write entry points.
///
/// Unlike [`ErasedRawStridedRef`], this descriptor does not create a shared
/// Rust reference at construction time. That lets the entry point reject an
/// input/output overlap before forming references to either side.
#[derive(Clone, Copy, Debug)]
pub struct ErasedRawStridedPtr<'a> {
    dtype: KernelDType,
    data: NonNull<u8>,
    byte_len: usize,
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
    marker: PhantomData<&'a [u8]>,
}

impl<'a> ErasedRawStridedPtr<'a> {
    /// Create a pointer-backed descriptor.
    ///
    /// # Safety
    ///
    /// `data..data + byte_len` must remain readable and allocated for `'a`.
    /// Its bytes must contain valid values for `dtype`. The allocation may
    /// overlap a destination passed to a one-shot entry point; overlap is
    /// checked before the pointer is dereferenced.
    pub unsafe fn new(
        dtype: KernelDType,
        data: NonNull<u8>,
        byte_len: usize,
        dims: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Result<Self> {
        let element_count = validate_erased_buffer_layout(dtype, data, byte_len)?;
        validate_bounds(element_count, dims, strides, offset)?;
        Ok(Self {
            dtype,
            data,
            byte_len,
            dims,
            strides,
            offset,
            marker: PhantomData,
        })
    }

    /// Borrow a safe erased input as a pointer descriptor.
    pub fn from_ref(input: &ErasedRawStridedRef<'a>) -> Self {
        Self {
            dtype: input.dtype,
            data: NonNull::new(input.data.as_ptr().cast_mut()).unwrap_or_else(NonNull::dangling),
            byte_len: input.data.len(),
            dims: input.dims,
            strides: input.strides,
            offset: input.offset,
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    #[inline]
    pub fn data_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    #[inline]
    pub fn byte_len(&self) -> usize {
        self.byte_len
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

    /// Materialize the shared byte slice after the caller has ruled out a
    /// concurrent mutable alias.
    ///
    /// # Safety
    ///
    /// No live mutable reference may overlap the returned slice.
    pub unsafe fn data_unchecked(&self) -> &'a [u8] {
        core::slice::from_raw_parts(self.data.as_ptr(), self.byte_len)
    }
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

/// Borrowed dtype-erased raw strided output whose reachable elements may be
/// uninitialized.
///
/// This descriptor is only accepted by operations that prove and perform a
/// full overwrite of every reachable logical destination element. The backing
/// allocation may contain non-reachable holes, which remain uninitialized.
#[derive(Debug)]
pub struct ErasedRawStridedUninitMut<'a> {
    dtype: KernelDType,
    data: &'a mut [MaybeUninit<u8>],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
}

impl<'a> ErasedRawStridedUninitMut<'a> {
    /// Create an uninitialized erased output descriptor after validating byte
    /// length, alignment, rank/stride agreement, and reachable bounds.
    pub fn new(
        dtype: KernelDType,
        data: &'a mut [MaybeUninit<u8>],
        dims: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Result<Self> {
        let data_ptr =
            NonNull::new(data.as_mut_ptr().cast::<u8>()).unwrap_or_else(NonNull::dangling);
        let element_count = validate_erased_buffer_layout(dtype, data_ptr, data.len())?;
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
    pub fn data_ptr(&self) -> *const u8 {
        self.data.as_ptr().cast::<u8>()
    }

    #[inline]
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [MaybeUninit<u8>] {
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
        if self.dims.iter().any(|&dim| dim == 0) {
            NonNull::<T>::dangling().as_ptr()
        } else {
            unsafe { self.data.as_ptr().offset(self.offset) }
        }
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
        if self.dims.iter().any(|&dim| dim == 0) {
            NonNull::<T>::dangling().as_ptr()
        } else {
            unsafe { self.data.as_ptr().offset(self.offset) }
        }
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        if self.dims.iter().any(|&dim| dim == 0) {
            NonNull::<T>::dangling().as_ptr()
        } else {
            unsafe { self.data.as_mut_ptr().offset(self.offset) }
        }
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
    fn rejects_usize_max_dimension_before_isize_truncation() {
        let data = [0u8; 1];
        let dims = [usize::MAX];
        let strides = [-1isize];
        assert!(matches!(
            RawStridedRef::new(&data, &dims, &strides, 0),
            Err(crate::StridedError::OffsetOverflow)
        ));
        let mut data = [0u8; 1];
        assert!(matches!(
            RawStridedMut::new(&mut data, &dims, &strides, 0),
            Err(crate::StridedError::OffsetOverflow)
        ));
        assert!(matches!(
            ErasedRawStridedRef::new(KernelDType::Bool, &data, &dims, &strides, 0),
            Err(crate::StridedError::OffsetOverflow)
        ));
        let mut data = [0u8; 1];
        assert!(matches!(
            ErasedRawStridedMut::new(KernelDType::Bool, &mut data, &dims, &strides, 0),
            Err(crate::StridedError::OffsetOverflow)
        ));
        let ptr = NonNull::new(data.as_mut_ptr()).unwrap();
        assert!(matches!(
            unsafe {
                ErasedRawStridedPtr::new(KernelDType::Bool, ptr, data.len(), &dims, &strides, 0)
            },
            Err(crate::StridedError::OffsetOverflow)
        ));
        let mut data = vec![MaybeUninit::<u8>::uninit(); 1];
        assert!(matches!(
            ErasedRawStridedUninitMut::new(KernelDType::Bool, &mut data, &dims, &strides, 0),
            Err(crate::StridedError::OffsetOverflow)
        ));
    }

    #[test]
    fn empty_raw_views_use_dangling_base_pointers_for_extreme_offsets() {
        let dims = [0usize];
        let strides = [1isize];
        let data: [f64; 0] = [];
        let raw = RawStridedRef::new(&data, &dims, &strides, isize::MAX).unwrap();
        assert_eq!(raw.ptr(), NonNull::<f64>::dangling().as_ptr());

        let mut data: [f64; 0] = [];
        let mut raw = RawStridedMut::new(&mut data, &dims, &strides, isize::MAX).unwrap();
        assert_eq!(raw.ptr(), NonNull::<f64>::dangling().as_ptr());
        assert_eq!(raw.as_mut_ptr(), NonNull::<f64>::dangling().as_ptr());
    }

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
