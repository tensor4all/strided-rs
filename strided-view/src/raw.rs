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
    let element_count = validate_erased_buffer_extent(dtype, data)?;
    if element_count != 0 && dtype.requires_valid_byte_values() {
        validate_bool_bytes(data)?;
    }
    Ok(element_count)
}

/// Validate the length and alignment of erased storage, without reading it.
///
/// Typed constructors call this alone: `KernelStorageElement` is sealed, so a
/// `&[T]` already holds valid values of `T::DTYPE`.
fn validate_erased_buffer_extent(dtype: KernelDType, data: &[u8]) -> Result<usize> {
    let element_size = dtype.size_of();
    if !data.len().is_multiple_of(element_size) {
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
    if !(data.as_ptr() as usize).is_multiple_of(alignment) {
        return Err(StridedError::DataAlignmentMismatch {
            dtype: dtype.label(),
            alignment,
        });
    }
    Ok(element_count)
}

/// Reject any byte other than 0 or 1.
///
/// The OR fold has no early exit, so it vectorizes and runs at memory
/// bandwidth; a byte by byte `find` does not, and cost more than the select
/// kernel it guarded. The slow scan runs only to report the offending byte.
fn validate_bool_bytes(data: &[u8]) -> Result<()> {
    if data.iter().fold(0u8, |acc, &value| acc | value) <= 1 {
        return Ok(());
    }
    match data.iter().find(|&&value| value > 1) {
        Some(&value) => Err(StridedError::InvalidBoolByte { value }),
        None => Ok(()),
    }
}

fn validate_erased_buffer_layout(
    dtype: KernelDType,
    data: NonNull<u8>,
    byte_len: usize,
) -> Result<usize> {
    let element_size = dtype.size_of();
    if !byte_len.is_multiple_of(element_size) {
        return Err(StridedError::ByteLengthMismatch {
            dtype: dtype.label(),
            byte_len,
            element_size,
        });
    }
    if byte_len != 0 && !(data.as_ptr() as usize).is_multiple_of(dtype.alignment()) {
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
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::ByteLengthMismatch`] when `byte_len` is not
    /// a whole number of `dtype` elements, and
    /// [`StridedError::DataAlignmentMismatch`] when a non-empty `data` is
    /// not aligned for `dtype`. Returns
    /// [`StridedError::StrideLengthMismatch`] when `dims` and `strides`
    /// have different lengths, and [`StridedError::OffsetOverflow`] when
    /// a reachable offset is not representable: `dim - 1` of a non-unit
    /// extent does not fit `isize`, a stride times that last index or the
    /// accumulated offsets overflow `isize` or go negative, or the highest
    /// reachable offset is not below the storage length. A layout with a zero
    /// extent skips this check.
    /// Bool bytes are not checked here;
    /// [`Self::try_as_ref_after_no_overlap`] does that.
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
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::InvalidBoolByte`] when `dtype` is `Bool` and
    /// a byte of the extent is neither 0 nor 1. It repeats the extent
    /// length, alignment, stride length and bounds checks
    /// ([`StridedError::ByteLengthMismatch`],
    /// [`StridedError::DataAlignmentMismatch`],
    /// [`StridedError::StrideLengthMismatch`],
    /// [`StridedError::OffsetOverflow`]) that [`Self::from_raw_parts`] and
    /// [`Self::from_ref`] already applied to the same values, so those
    /// cannot fire again here.
    pub unsafe fn try_as_ref_after_no_overlap(&self) -> Result<ErasedRawStridedRef<'_>> {
        // SAFETY: this function's contract keeps the `from_raw_parts` contract
        // (raw.rs:204) in force: `data` (a `NonNull`) heads `byte_len`
        // initialized, readable bytes alive for `'a`, and the caller has ruled
        // out a mutable overlap and any concurrent access, so a shared `u8`
        // slice over them is not aliased by a writer. `u8` accepts every byte
        // value.
        let bytes = core::slice::from_raw_parts(self.data.as_ptr(), self.byte_len);
        validate_erased_buffer(self.dtype, bytes)?;
        // SAFETY: `ErasedRawStridedRef::from_raw_parts` (raw.rs:447) needs the
        // start of an allocation aligned for `dtype`, `byte_len` initialized
        // bytes for `'a`, valid `dtype` values, and no mutable alias. Extent and
        // lifetime are this pointer's own `from_raw_parts` contract
        // (raw.rs:204); bool values were just checked by `validate_erased_buffer`
        // (raw.rs:109) via `validate_bool_bytes` (raw.rs:151); the absence of a
        // mutable alias is this function's contract. The result borrows `self`,
        // so it cannot outlive `'a`. Alignment is proven only for a non-empty
        // buffer, whose address `validate_erased_buffer_extent` (raw.rs:121)
        // checked. NOT PROVEN for `byte_len == 0`: the pointer contract
        // (raw.rs:204) asks only that `data` point *into* an aligned
        // allocation, and both validators skip alignment for an empty buffer,
        // so a misaligned interior pointer reaches this call and a later safe
        // `data_as` forms a misaligned slice (tensor4all/strided-rs#434).
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
    ///
    /// # Errors
    ///
    /// Never returns `Err` in practice: both byte extents come from valid
    /// descriptors, whose contracts already guarantee an allocation of that
    /// size, so the end-address arithmetic cannot wrap. The `Result` only
    /// carries that arithmetic's [`StridedError::OffsetOverflow`].
    pub fn overlaps_mut(&self, dest: &ErasedRawStridedMut<'_>) -> Result<bool> {
        ranges_overlap(
            self.data.as_ptr() as usize,
            self.byte_len,
            dest.data.as_ptr() as usize,
            dest.data.len(),
        )
    }

    /// Check overlap with uninitialized mutable storage without reading it.
    ///
    /// # Errors
    ///
    /// Never returns `Err` in practice: both byte extents come from valid
    /// descriptors, whose contracts already guarantee an allocation of that
    /// size, so the end-address arithmetic cannot wrap. The `Result` only
    /// carries that arithmetic's [`StridedError::OffsetOverflow`].
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
#[derive(Clone, Copy)]
pub struct ErasedRawStridedRef<'a> {
    dtype: KernelDType,
    data: &'a [u8],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
}

// A derived `Debug` would print every byte of the backing buffer; the layout
// metadata is what a reader wants.
impl core::fmt::Debug for ErasedRawStridedRef<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ErasedRawStridedRef")
            .field("dtype", &self.dtype)
            .field("byte_len", &self.data.len())
            .field("dims", &self.dims)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .finish()
    }
}

impl<'a> ErasedRawStridedRef<'a> {
    /// Create an erased input descriptor from typed, initialized storage.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::StrideLengthMismatch`] when `dims` and
    /// `strides` have different lengths, and
    /// [`StridedError::OffsetOverflow`] when a reachable offset is not
    /// representable: `dim - 1` of a non-unit extent does not fit `isize`, a
    /// stride times that last index or the accumulated offsets overflow
    /// `isize` or go negative, or the highest reachable offset is not below
    /// the buffer length. A layout with a zero extent skips this check. The
    /// byte-length and alignment checks
    /// cannot fail for a typed `&[T]`.
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
        // SAFETY: `data` is a live `&[T]` and `byte_len` is
        // `data.len() * size_of::<T>()` from the `checked_mul` above, so the
        // `u8` view spans exactly that initialized allocation for `data`'s
        // lifetime. `u8` has no invalid bit patterns, and the resulting length
        // is re-checked by `validate_erased_buffer_extent` (raw.rs:121).
        let bytes = unsafe { core::slice::from_raw_parts(data.as_ptr().cast::<u8>(), byte_len) };
        let element_count = validate_erased_buffer_extent(dtype, bytes)?;
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
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::ByteLengthMismatch`] when `byte_len` is not
    /// a whole number of `dtype` elements, and
    /// [`StridedError::DataAlignmentMismatch`] when a non-empty `data` is
    /// not aligned for `dtype`. Returns
    /// [`StridedError::StrideLengthMismatch`] when `dims` and `strides`
    /// have different lengths, and [`StridedError::OffsetOverflow`] when
    /// a reachable offset is not representable: `dim - 1` of a non-unit
    /// extent does not fit `isize`, a stride times that last index or the
    /// accumulated offsets overflow `isize` or go negative, or the highest
    /// reachable offset is not below the storage length. A layout with a zero
    /// extent skips this check.
    /// Bool bytes are the caller's responsibility here, per the safety
    /// contract.
    pub unsafe fn from_raw_parts(
        dtype: KernelDType,
        data: NonNull<u8>,
        byte_len: usize,
        dims: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Result<Self> {
        let count = validate_erased_buffer_layout(dtype, data, byte_len)?;
        // SAFETY: `data` is a `NonNull`, and this function's contract
        // (raw.rs:447) makes `data..data + byte_len` one live allocation of
        // initialized bytes for `'a` with no mutable alias, which also bounds
        // `byte_len` by `isize::MAX`. `u8` needs no alignment beyond 1.
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
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::DTypeMismatch`] when `T` is not the
    /// descriptor's dtype.
    pub fn data_as<T: KernelStorageElement>(&self) -> Result<&[T]> {
        if self.dtype != T::DTYPE {
            return Err(StridedError::DTypeMismatch {
                expected: T::DTYPE.label(),
                actual: self.dtype.label(),
            });
        }
        // SAFETY: the dtype check above proves the stored dtype is `T`, and the
        // constructor ran `validate_erased_buffer_extent` (raw.rs:121) /
        // `validate_erased_buffer_layout` (raw.rs:161): the byte length is a
        // whole multiple of `size_of::<T>()` and the base is `T`-aligned, so
        // `len / size_of::<T>()` initialized `T` values are in bounds. Both
        // checks skip alignment for an empty buffer. There the base is
        // `T`-aligned when it came from a `&[T]` (`from_slice`, raw.rs:416) or
        // from a direct `from_raw_parts` call, whose contract (raw.rs:447)
        // requires the start of an aligned allocation. NOT PROVEN for a
        // descriptor from `ErasedRawStridedPtr::try_as_ref_after_no_overlap`
        // (raw.rs:288): its pointer may be a misaligned interior pointer,
        // and with `byte_len == 0` this forms a misaligned empty slice, which
        // is UB (tensor4all/strided-rs#434).
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
pub struct ErasedRawStridedMut<'a> {
    dtype: KernelDType,
    data: &'a mut [u8],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
}

// A derived `Debug` would print every byte of the backing buffer; the layout
// metadata is what a reader wants.
impl core::fmt::Debug for ErasedRawStridedMut<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ErasedRawStridedMut")
            .field("dtype", &self.dtype)
            .field("byte_len", &self.data.len())
            .field("dims", &self.dims)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .finish()
    }
}

impl<'a> ErasedRawStridedMut<'a> {
    /// Create an erased output descriptor from typed, initialized storage.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::StrideLengthMismatch`] when `dims` and
    /// `strides` have different lengths, and
    /// [`StridedError::OffsetOverflow`] when a reachable offset is not
    /// representable: `dim - 1` of a non-unit extent does not fit `isize`, a
    /// stride times that last index or the accumulated offsets overflow
    /// `isize` or go negative, or the highest reachable offset is not below
    /// the buffer length. A layout with a zero extent skips this check. The
    /// byte-length and alignment checks
    /// cannot fail for a typed `&mut [T]`.
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
        // SAFETY: `data` is a live, exclusively borrowed `&mut [T]` and
        // `byte_len` is `data.len() * size_of::<T>()` from the `checked_mul`
        // above, so the `u8` view spans exactly that initialized allocation for
        // `'a`. `u8` has no invalid bit patterns, and the resulting length is
        // re-checked by `validate_erased_buffer_extent` (raw.rs:121).
        let data =
            unsafe { core::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u8>(), byte_len) };
        let element_count = validate_erased_buffer_extent(dtype, data)?;
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
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::ByteLengthMismatch`] when `byte_len` is not
    /// a whole number of `dtype` elements, and
    /// [`StridedError::DataAlignmentMismatch`] when a non-empty `data` is
    /// not aligned for `dtype`. Returns
    /// [`StridedError::InvalidBoolByte`] when `dtype` is `Bool` and a byte
    /// of the extent is neither 0 nor 1. Returns
    /// [`StridedError::StrideLengthMismatch`] when `dims` and `strides`
    /// have different lengths, and [`StridedError::OffsetOverflow`] when
    /// a reachable offset is not representable: `dim - 1` of a non-unit
    /// extent does not fit `isize`, a stride times that last index or the
    /// accumulated offsets overflow `isize` or go negative, or the highest
    /// reachable offset is not below the storage length. A layout with a zero
    /// extent skips this check.
    pub unsafe fn from_raw_parts(
        dtype: KernelDType,
        data: NonNull<u8>,
        byte_len: usize,
        dims: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Result<Self> {
        let count = validate_erased_buffer_layout(dtype, data, byte_len)?;
        // SAFETY: `data` is a `NonNull`, and this function's contract
        // (raw.rs:619) makes `data..data + byte_len` one live, writable,
        // initialized allocation for `'a` held exclusively by the caller, which
        // also bounds `byte_len` by `isize::MAX`. `u8` needs no alignment
        // beyond 1.
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
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::DTypeMismatch`] when `T` is not the
    /// descriptor's dtype.
    pub fn data_as<T: KernelStorageElement>(&self) -> Result<&[T]> {
        if self.dtype != T::DTYPE {
            return Err(StridedError::DTypeMismatch {
                expected: T::DTYPE.label(),
                actual: self.dtype.label(),
            });
        }
        // SAFETY: the dtype check above proves the stored dtype is `T`, and the
        // constructor ran `validate_erased_buffer_extent` (raw.rs:121) /
        // `validate_erased_buffer_layout` (raw.rs:161): the byte length is a
        // whole multiple of `size_of::<T>()` and the base is `T`-aligned, so
        // `len / size_of::<T>()` initialized `T` values are in bounds. Both
        // checks skip alignment for an empty buffer. There the base came from a
        // `&mut [T]` (`from_slice_mut`, raw.rs:587) or from `from_raw_parts`, whose
        // contract (raw.rs:619) makes `data..data + byte_len` itself the
        // aligned allocation, so `data` is its aligned start. Unlike the `Ref`
        // (tensor4all/strided-rs#434), nothing in this crate builds this
        // descriptor from an interior pointer.
        Ok(unsafe {
            core::slice::from_raw_parts(
                self.data.as_ptr().cast::<T>(),
                self.data.len() / core::mem::size_of::<T>(),
            )
        })
    }

    /// Borrow initialized storage mutably as its concrete scalar type.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::DTypeMismatch`] when `T` is not the
    /// descriptor's dtype.
    pub fn data_as_mut<T: KernelStorageElement>(&mut self) -> Result<&mut [T]> {
        if self.dtype != T::DTYPE {
            return Err(StridedError::DTypeMismatch {
                expected: T::DTYPE.label(),
                actual: self.dtype.label(),
            });
        }
        // SAFETY: as `data_as` (raw.rs:674) — the dtype check proves `T`, and the
        // constructor's extent/layout check (raw.rs:161), or for an empty
        // buffer the `&mut [T]` origin / `from_raw_parts` contract (aligned
        // start, no interior-pointer path), makes the base `T`-aligned with a
        // whole multiple of `size_of::<T>()` bytes.
        // `&mut self` keeps this reborrow exclusive for its own lifetime.
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
pub struct ErasedRawStridedUninitMut<'a> {
    dtype: KernelDType,
    data: &'a mut [MaybeUninit<u8>],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
}

// A derived `Debug` would print one entry per byte of the backing buffer; the
// layout metadata is what a reader wants.
impl core::fmt::Debug for ErasedRawStridedUninitMut<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ErasedRawStridedUninitMut")
            .field("dtype", &self.dtype)
            .field("byte_len", &self.data.len())
            .field("dims", &self.dims)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .finish()
    }
}

impl<'a> ErasedRawStridedUninitMut<'a> {
    /// Create an uninitialized erased output descriptor from typed storage.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::StrideLengthMismatch`] when `dims` and
    /// `strides` have different lengths, and
    /// [`StridedError::OffsetOverflow`] when a reachable offset is not
    /// representable: `dim - 1` of a non-unit extent does not fit `isize`, a
    /// stride times that last index or the accumulated offsets overflow
    /// `isize` or go negative, or the highest reachable offset is not below
    /// the buffer length. A layout with a zero extent skips this check. The
    /// byte-length and alignment checks
    /// cannot fail for a typed `&mut [MaybeUninit<T>]`.
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
        // SAFETY: `data` is a live, exclusively borrowed `&mut
        // [MaybeUninit<T>]` and `byte_len` is `data.len() * size_of::<T>()` from
        // the `checked_mul` above, so the view spans exactly that allocation for
        // `'a`. `MaybeUninit<u8>` accepts every bit pattern, so no
        // initialization is asserted here.
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
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::ByteLengthMismatch`] when `byte_len` is not
    /// a whole number of `dtype` elements, and
    /// [`StridedError::DataAlignmentMismatch`] when a non-empty `data` is
    /// not aligned for `dtype`. Returns
    /// [`StridedError::StrideLengthMismatch`] when `dims` and `strides`
    /// have different lengths, and [`StridedError::OffsetOverflow`] when
    /// a reachable offset is not representable: `dim - 1` of a non-unit
    /// extent does not fit `isize`, a stride times that last index or the
    /// accumulated offsets overflow `isize` or go negative, or the highest
    /// reachable offset is not below the storage length. A layout with a zero
    /// extent skips this check.
    pub unsafe fn from_raw_parts(
        dtype: KernelDType,
        data: NonNull<u8>,
        byte_len: usize,
        dims: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Result<Self> {
        let count = validate_erased_buffer_layout(dtype, data, byte_len)?;
        // SAFETY: `data` is a `NonNull`, and this function's contract
        // (raw.rs:822) makes `data..data + byte_len` one live, writable
        // allocation for `'a` held exclusively by the caller, which also bounds
        // `byte_len` by `isize::MAX`. `MaybeUninit<u8>` accepts every bit
        // pattern, so no initialization is asserted.
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
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::DTypeMismatch`] when `T` is not the
    /// descriptor's dtype.
    pub fn data_as_uninit_mut<T: KernelStorageElement>(&mut self) -> Result<&mut [MaybeUninit<T>]> {
        if self.dtype != T::DTYPE {
            return Err(StridedError::DTypeMismatch {
                expected: T::DTYPE.label(),
                actual: self.dtype.label(),
            });
        }
        // SAFETY: the dtype check proves `T`, and
        // `validate_erased_buffer_layout` (raw.rs:161) proved the base is
        // `T`-aligned with a whole multiple of `size_of::<T>()` bytes. It skips
        // alignment for an empty buffer; there the base came from a
        // `&mut [MaybeUninit<T>]` (`from_uninit_slice`, raw.rs:790) or from
        // `from_raw_parts`, whose contract (raw.rs:822) makes `data..data + byte_len`
        // the aligned allocation itself. As for `ErasedRawStridedMut`, no path
        // in this crate builds it from an interior pointer (contrast #434).
        // `MaybeUninit<T>` accepts every bit pattern, so the partially
        // initialized extent is not read as `T` here.
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
#[derive(Clone, Copy)]
pub struct RawStridedRef<'a, T> {
    data: &'a [T],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
}

// A derived `Debug` would print every element of the backing buffer; the layout
// metadata is what a reader wants.
impl<T: core::fmt::Debug> core::fmt::Debug for RawStridedRef<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RawStridedRef")
            .field("data_len", &self.data.len())
            .field("dims", &self.dims)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .finish()
    }
}

impl<'a, T> RawStridedRef<'a, T> {
    /// Create a raw strided input after validating reachable offsets.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::StrideLengthMismatch`] when `dims` and
    /// `strides` have different lengths, and
    /// [`StridedError::OffsetOverflow`] when a reachable offset is not
    /// representable: `dim - 1` of a non-unit extent does not fit `isize`, a
    /// stride times that last index or the accumulated offsets overflow
    /// `isize` or go negative, or the highest reachable offset is not below
    /// the buffer length. A layout with a zero extent skips this check.
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
        if self.dims.contains(&0) {
            NonNull::<T>::dangling().as_ptr()
        } else {
            // SAFETY: `offset` was accepted by `validate_bounds`
            // (view.rs:187) in `RawStridedRef::new`, or by the
            // caller contract of `new_unchecked`, so `data.as_ptr() + offset`
            // stays inside `data`. The empty-layout branch above returns a
            // dangling pointer instead of offsetting a pointer no element
            // reaches.
            unsafe { self.data.as_ptr().offset(self.offset) }
        }
    }

    /// Convert to an immutable view, borrowing this descriptor's metadata.
    ///
    /// The conversion allocates nothing; hot prepared paths use the raw
    /// accessors directly all the same.
    #[inline]
    pub fn as_view(&self) -> StridedView<'a, T, Identity> {
        // SAFETY: `new_unchecked` requires every reachable index to stay inside
        // `data`; this descriptor's `offset`/`dims`/`strides` came from
        // `validate_bounds` (view.rs:187) in `RawStridedRef::new`,
        // or from the same contract on `new_unchecked`.
        unsafe { StridedView::new_unchecked(self.data, self.dims, self.strides, self.offset) }
    }
}

/// Borrowed raw strided output layout.
///
/// This is the mutable counterpart to [`RawStridedRef`]. It avoids allocating
/// owned shape/stride metadata in prepared replay paths.
pub struct RawStridedMut<'a, T> {
    data: &'a mut [T],
    dims: &'a [usize],
    strides: &'a [isize],
    offset: isize,
}

// A derived `Debug` would print every element of the backing buffer; the layout
// metadata is what a reader wants.
impl<T: core::fmt::Debug> core::fmt::Debug for RawStridedMut<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RawStridedMut")
            .field("data_len", &self.data.len())
            .field("dims", &self.dims)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .finish()
    }
}

impl<'a, T> RawStridedMut<'a, T> {
    /// Create a raw strided output after validating reachable offsets.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::StrideLengthMismatch`] when `dims` and
    /// `strides` have different lengths, and
    /// [`StridedError::OffsetOverflow`] when a reachable offset is not
    /// representable: `dim - 1` of a non-unit extent does not fit `isize`, a
    /// stride times that last index or the accumulated offsets overflow
    /// `isize` or go negative, or the highest reachable offset is not below
    /// the buffer length. A layout with a zero extent skips this check.
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
        if self.dims.contains(&0) {
            NonNull::<T>::dangling().as_ptr()
        } else {
            // SAFETY: `offset` was accepted by `validate_bounds`
            // (view.rs:187) in `RawStridedMut::new`, or by the
            // caller contract of `new_unchecked`, so `data.as_ptr() + offset`
            // stays inside `data`. The empty-layout branch above returns a
            // dangling pointer instead of offsetting a pointer no element
            // reaches.
            unsafe { self.data.as_ptr().offset(self.offset) }
        }
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        if self.dims.contains(&0) {
            NonNull::<T>::dangling().as_ptr()
        } else {
            // SAFETY: as `ptr` above — `offset` was accepted by
            // `validate_bounds` (view.rs:187) in
            // `RawStridedMut::new` or by the `new_unchecked` contract, and
            // `&mut self` keeps the write provenance exclusive.
            unsafe { self.data.as_mut_ptr().offset(self.offset) }
        }
    }

    /// Convert to an immutable view, borrowing this descriptor's metadata.
    ///
    /// The conversion allocates nothing; hot prepared paths use the raw
    /// accessors directly all the same.
    #[inline]
    pub fn as_view(&self) -> StridedView<'_, T, Identity> {
        // SAFETY: `new_unchecked` requires every reachable index to stay inside
        // `data`; this descriptor's metadata came from `validate_bounds`
        // (view.rs:187) in `RawStridedMut::new`, or from the
        // same contract on `new_unchecked`.
        unsafe { StridedView::new_unchecked(self.data, self.dims, self.strides, self.offset) }
    }

    /// Convert to a mutable view, borrowing this descriptor's metadata.
    ///
    /// The conversion allocates nothing; hot prepared paths use the raw
    /// accessors directly all the same.
    #[inline]
    pub fn as_view_mut(&mut self) -> StridedViewMut<'_, T> {
        // SAFETY: as `as_view` above, and `&mut self` keeps the mutable borrow
        // exclusive to the returned view, which `new_unchecked` requires.
        unsafe { StridedViewMut::new_unchecked(self.data, self.dims, self.strides, self.offset) }
    }
}

#[cfg(test)]
mod tests;
