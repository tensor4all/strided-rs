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

mod private {
    pub trait Sealed {}
}

/// A scalar type that can safely witness storage for an erased kernel view.
pub trait KernelStorageElement: private::Sealed + Copy + 'static {
    const DTYPE: KernelDType;
}

macro_rules! kernel_storage_element {
    ($($ty:ty => $dtype:ident),* $(,)?) => {$ (
        impl private::Sealed for $ty {}
        impl KernelStorageElement for $ty {
            const DTYPE: KernelDType = KernelDType::$dtype;
        }
    )* };
}

kernel_storage_element! {
    f32 => F32,
    f64 => F64,
    i32 => I32,
    i64 => I64,
    bool => Bool,
    Complex32 => C32,
    Complex64 => C64,
}

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
    marker: PhantomData<&'a [MaybeUninit<u8>]>,
}

impl<'a> ErasedRawStridedPtr<'a> {
    /// Create a pointer-backed descriptor from erased storage.
    ///
    /// # Safety
    ///
    /// `data` must point into an allocation whose alignment is suitable for
    /// `dtype`, independently of the observed address. The allocation must
    /// provide `byte_len` initialized, readable bytes with valid provenance
    /// for `'a`, and remain alive for `'a`. For `bool`, the bytes may contain
    /// invalid values temporarily. They must not be read or used to form a
    /// typed reference until a caller has rejected overlap and
    /// [`Self::try_as_ref_after_no_overlap`] has validated the complete extent.
    /// The allocation may overlap a destination.
    /// The original owner may perform synchronized sequential mutation through
    /// the same-provenance raw pointer before conversion. No concurrent
    /// mutation or conflicting access is permitted during overlap checking,
    /// conversion, or consumer access.
    pub unsafe fn from_raw_parts(
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

    /// Convert this pointer to an initialized descriptor after overlap checks.
    ///
    /// # Safety
    ///
    /// The caller must have proved that no mutable allocation overlaps this
    /// pointer's complete byte extent.
    /// The allocation contract from [`Self::from_raw_parts`] must still hold.
    /// No mutation or other conflicting access may occur during this conversion
    /// or for the duration of the returned descriptor's consumer access.
    /// Bool bytes are validated here, immediately before typed access.
    pub unsafe fn try_as_ref_after_no_overlap(&self) -> Result<ErasedRawStridedRef<'_>> {
        let bytes = core::slice::from_raw_parts(self.data.as_ptr(), self.byte_len);
        validate_erased_buffer(self.dtype, bytes)?;
        ErasedRawStridedRef::from_raw_parts(
            self.dtype,
            self.data,
            self.byte_len,
            self.dims,
            self.strides,
            self.offset,
        )
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
    }

    /// Check overlap with initialized mutable storage without reading it.
    pub fn overlaps_mut(&self, dest: &ErasedRawStridedMut<'_>) -> Result<bool> {
        ranges_overlap(
            self.data.as_ptr() as usize,
            self.byte_len,
            dest.data.as_ptr() as usize,
            dest.data.len(),
        )
    }

    /// Check overlap with uninitialized mutable storage without reading it.
    pub fn overlaps_uninit_mut(&self, dest: &ErasedRawStridedUninitMut<'_>) -> Result<bool> {
        ranges_overlap(
            self.data.as_ptr() as usize,
            self.byte_len,
            dest.data.as_ptr() as usize,
            dest.data.len(),
        )
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
    /// Create an erased input descriptor from typed, initialized storage.
    pub fn from_slice<T: KernelStorageElement>(
        data: &'a [T],
        dims: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Result<Self> {
        let dtype = T::DTYPE;
        let byte_len = data
            .len()
            .checked_mul(core::mem::size_of::<T>())
            .ok_or(StridedError::OffsetOverflow)?;
        let bytes = unsafe { core::slice::from_raw_parts(data.as_ptr().cast::<u8>(), byte_len) };
        let element_count = validate_erased_buffer(dtype, bytes)?;
        validate_bounds(element_count, dims, strides, offset)?;
        Ok(Self {
            dtype,
            data: bytes,
            dims,
            strides,
            offset,
        })
    }

    /// Construct from erased storage.
    ///
    /// # Safety
    /// `data` must be the start of an allocation aligned for `dtype`; the
    /// allocation must contain `byte_len` initialized, readable bytes for
    /// `'a`, and every byte extent must represent valid values for `dtype`.
    /// The allocation and metadata must outlive `'a`; no mutable alias may
    /// exist while this descriptor is used. Alignment is an allocation
    /// property and must not be inferred from the observed address alone.
    pub unsafe fn from_raw_parts(
        dtype: KernelDType,
        data: NonNull<u8>,
        byte_len: usize,
        dims: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Result<Self> {
        let count = validate_erased_buffer_layout(dtype, data, byte_len)?;
        let bytes = core::slice::from_raw_parts(data.as_ptr(), byte_len);
        validate_bounds(count, dims, strides, offset)?;
        Ok(Self {
            dtype,
            data: bytes,
            dims,
            strides,
            offset,
        })
    }

    /// Borrow the initialized storage as its concrete scalar type.
    pub fn data_as<T: KernelStorageElement>(&self) -> Result<&[T]> {
        if self.dtype != T::DTYPE {
            return Err(StridedError::DTypeMismatch {
                expected: T::DTYPE.label(),
                actual: self.dtype.label(),
            });
        }
        Ok(unsafe {
            core::slice::from_raw_parts(
                self.data.as_ptr().cast::<T>(),
                self.data.len() / core::mem::size_of::<T>(),
            )
        })
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
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
}

impl<'a> ErasedRawStridedMut<'a> {
    /// Create an erased output descriptor from typed, initialized storage.
    pub fn from_slice_mut<T: KernelStorageElement>(
        data: &'a mut [T],
        dims: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Result<Self> {
        let dtype = T::DTYPE;
        let byte_len = data
            .len()
            .checked_mul(core::mem::size_of::<T>())
            .ok_or(StridedError::OffsetOverflow)?;
        let data =
            unsafe { core::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u8>(), byte_len) };
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

    /// Construct from erased storage.
    ///
    /// # Safety
    /// `data..data + byte_len` must be an aligned, writable allocation valid
    /// for `'a`, with valid initialized values for `dtype` throughout the
    /// complete byte extent; its provenance, extent, lifetime, and exclusive
    /// aliasing must be upheld by the caller.
    /// The alignment requirement is on the allocation, not merely the observed
    /// address. The caller must retain exclusive access: no mutable alias or
    /// concurrent mutation may exist while this descriptor is used.
    pub unsafe fn from_raw_parts(
        dtype: KernelDType,
        data: NonNull<u8>,
        byte_len: usize,
        dims: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Result<Self> {
        let count = validate_erased_buffer_layout(dtype, data, byte_len)?;
        let data = core::slice::from_raw_parts_mut(data.as_ptr(), byte_len);
        validate_erased_buffer(dtype, data)?;
        validate_bounds(count, dims, strides, offset)?;
        Ok(Self {
            dtype,
            data,
            dims,
            strides,
            offset,
        })
    }

    /// Borrow initialized storage as its concrete scalar type.
    pub fn data_as<T: KernelStorageElement>(&self) -> Result<&[T]> {
        if self.dtype != T::DTYPE {
            return Err(StridedError::DTypeMismatch {
                expected: T::DTYPE.label(),
                actual: self.dtype.label(),
            });
        }
        Ok(unsafe {
            core::slice::from_raw_parts(
                self.data.as_ptr().cast::<T>(),
                self.data.len() / core::mem::size_of::<T>(),
            )
        })
    }

    /// Borrow initialized storage mutably as its concrete scalar type.
    pub fn data_as_mut<T: KernelStorageElement>(&mut self) -> Result<&mut [T]> {
        if self.dtype != T::DTYPE {
            return Err(StridedError::DTypeMismatch {
                expected: T::DTYPE.label(),
                actual: self.dtype.label(),
            });
        }
        Ok(unsafe {
            core::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().cast::<T>(),
                self.data.len() / core::mem::size_of::<T>(),
            )
        })
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
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
    /// Create an uninitialized erased output descriptor from typed storage.
    pub fn from_uninit_slice<T: KernelStorageElement>(
        data: &'a mut [MaybeUninit<T>],
        dims: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Result<Self> {
        let dtype = T::DTYPE;
        let byte_len = data
            .len()
            .checked_mul(core::mem::size_of::<T>())
            .ok_or(StridedError::OffsetOverflow)?;
        let data = unsafe {
            core::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<MaybeUninit<u8>>(), byte_len)
        };
        let data_ptr =
            NonNull::new(data.as_mut_ptr().cast::<u8>()).unwrap_or_else(NonNull::dangling);
        let element_count = validate_erased_buffer_layout(dtype, data_ptr, byte_len)?;
        validate_bounds(element_count, dims, strides, offset)?;
        Ok(Self {
            dtype,
            data,
            dims,
            strides,
            offset,
        })
    }

    /// Construct from erased uninitialized storage.
    ///
    /// # Safety
    /// `data..data + byte_len` must be an aligned, writable allocation valid
    /// for `'a`, with provenance and extent sufficient for all reachable
    /// elements. The caller must ensure every reachable element is completely
    /// overwritten before it is read or exposed as initialized storage. The
    /// allocation's alignment is an allocation property independent of the
    /// observed address, and the descriptor must have exclusive access with no
    /// mutable alias or concurrent mutation during use.
    pub unsafe fn from_raw_parts(
        dtype: KernelDType,
        data: NonNull<u8>,
        byte_len: usize,
        dims: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Result<Self> {
        let count = validate_erased_buffer_layout(dtype, data, byte_len)?;
        let data =
            core::slice::from_raw_parts_mut(data.as_ptr().cast::<MaybeUninit<u8>>(), byte_len);
        validate_bounds(count, dims, strides, offset)?;
        Ok(Self {
            dtype,
            data,
            dims,
            strides,
            offset,
        })
    }

    /// Borrow the uninitialized storage as its concrete scalar type.
    pub fn data_as_uninit_mut<T: KernelStorageElement>(&mut self) -> Result<&mut [MaybeUninit<T>]> {
        if self.dtype != T::DTYPE {
            return Err(StridedError::DTypeMismatch {
                expected: T::DTYPE.label(),
                actual: self.dtype.label(),
            });
        }
        Ok(unsafe {
            core::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().cast::<MaybeUninit<T>>(),
                self.data.len() / core::mem::size_of::<T>(),
            )
        })
    }

    #[inline]
    pub fn dtype(&self) -> KernelDType {
        self.dtype
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

fn ranges_overlap(a_start: usize, a_len: usize, b_start: usize, b_len: usize) -> Result<bool> {
    let a_end = a_start
        .checked_add(a_len)
        .ok_or(StridedError::OffsetOverflow)?;
    let b_end = b_start
        .checked_add(b_len)
        .ok_or(StridedError::OffsetOverflow)?;
    Ok(a_start < b_end && b_start < a_end)
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
    fn typed_erased_storage_covers_all_kernel_dtypes() {
        macro_rules! check {
            ($ty:ty, $value:expr) => {{
                let mut initialized = [$value, $value];
                let dims = [2usize];
                let strides = [1isize];
                let reference =
                    ErasedRawStridedRef::from_slice(&initialized, &dims, &strides, 0).unwrap();
                assert_eq!(reference.data_as::<$ty>().unwrap().len(), 2);
                let mut mutable =
                    ErasedRawStridedMut::from_slice_mut(&mut initialized, &dims, &strides, 0)
                        .unwrap();
                assert_eq!(mutable.data_as::<$ty>().unwrap().len(), 2);
                assert_eq!(mutable.data_as_mut::<$ty>().unwrap().len(), 2);
                let mut uninitialized = [MaybeUninit::<$ty>::uninit(); 2];
                let mut destination = ErasedRawStridedUninitMut::from_uninit_slice(
                    &mut uninitialized,
                    &dims,
                    &strides,
                    0,
                )
                .unwrap();
                assert_eq!(destination.data_as_uninit_mut::<$ty>().unwrap().len(), 2);
            }};
        }

        check!(f32, 1.0f32);
        check!(f64, 1.0f64);
        check!(i32, 1i32);
        check!(i64, 1i64);
        check!(bool, true);
        check!(Complex32, Complex32::new(1.0, 0.0));
        check!(Complex64, Complex64::new(1.0, 0.0));
    }

    #[test]
    fn rejects_usize_max_dimension_before_isize_truncation() {
        let data = [false; 1];
        let dims = [usize::MAX];
        let strides = [-1isize];
        assert!(matches!(
            RawStridedRef::new(&data, &dims, &strides, 0),
            Err(crate::StridedError::OffsetOverflow)
        ));
        let mut data = [false; 1];
        assert!(matches!(
            RawStridedMut::new(&mut data, &dims, &strides, 0),
            Err(crate::StridedError::OffsetOverflow)
        ));
        assert!(matches!(
            ErasedRawStridedRef::from_slice(&data, &dims, &strides, 0),
            Err(crate::StridedError::OffsetOverflow)
        ));
        let mut data = [false; 1];
        assert!(matches!(
            ErasedRawStridedMut::from_slice_mut(&mut data, &dims, &strides, 0),
            Err(crate::StridedError::OffsetOverflow)
        ));
        let ptr = NonNull::new(data.as_mut_ptr()).unwrap();
        assert!(matches!(
            unsafe {
                ErasedRawStridedPtr::from_raw_parts(
                    KernelDType::Bool,
                    ptr.cast(),
                    data.len(),
                    &dims,
                    &strides,
                    0,
                )
            },
            Err(crate::StridedError::OffsetOverflow)
        ));
        let mut data = vec![MaybeUninit::<bool>::uninit(); 1];
        assert!(matches!(
            ErasedRawStridedUninitMut::from_uninit_slice(&mut data, &dims, &strides, 0),
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
