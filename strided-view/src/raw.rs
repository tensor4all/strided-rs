//! Borrowed raw strided layout types.
//!
//! These are the prepared-replay counterparts to [`crate::StridedView`] and
//! [`crate::StridedViewMut`]. They borrow shape/stride metadata instead of
//! owning it, so compiled kernels can reuse already-validated layout
//! descriptors without rebuilding dynamic-rank view wrappers.

use crate::element_op::Identity;
use crate::view::validate_bounds;
use crate::{Result, StridedView, StridedViewMut};

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
