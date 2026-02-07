//! Julia-like dynamic-rank strided view types.
//!
//! This module provides the canonical view types for strided operations,
//! matching Julia's StridedViews.jl data model:
//!
//! - [`StridedView`]: Immutable dynamic-rank strided view with lazy element operations
//! - [`StridedViewMut`]: Mutable dynamic-rank strided view (Identity op only)
//! - [`StridedArray`]: Owned strided multidimensional array

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::sync::Arc;

use crate::element_op::{ElementOp, ElementOpApply, Identity};
use crate::{Result, StridedError};

// ============================================================================
// Validation helpers
// ============================================================================

/// Validate that all accessed offsets stay within `[0, len)`.
fn validate_bounds(len: usize, dims: &[usize], strides: &[isize], offset: isize) -> Result<()> {
    if dims.len() != strides.len() {
        return Err(StridedError::StrideLengthMismatch);
    }
    // Empty array - no access needed
    if dims.iter().any(|&d| d == 0) {
        return Ok(());
    }
    // Compute min and max offsets
    let mut min_offset = offset;
    let mut max_offset = offset;
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        if dim > 1 {
            let end = stride
                .checked_mul(dim as isize - 1)
                .ok_or(StridedError::OffsetOverflow)?;
            if end >= 0 {
                max_offset = max_offset
                    .checked_add(end)
                    .ok_or(StridedError::OffsetOverflow)?;
            } else {
                min_offset = min_offset
                    .checked_add(end)
                    .ok_or(StridedError::OffsetOverflow)?;
            }
        }
    }
    if min_offset < 0 || max_offset < 0 {
        return Err(StridedError::OffsetOverflow);
    }
    if max_offset as usize >= len {
        return Err(StridedError::OffsetOverflow);
    }
    Ok(())
}

/// Compute column-major strides (Julia default: first index varies fastest).
pub fn col_major_strides(dims: &[usize]) -> Vec<isize> {
    let rank = dims.len();
    if rank == 0 {
        return vec![];
    }
    let mut strides = vec![1isize; rank];
    for i in 1..rank {
        strides[i] = strides[i - 1] * dims[i - 1] as isize;
    }
    strides
}

/// Compute row-major strides (C default: last index varies fastest).
pub fn row_major_strides(dims: &[usize]) -> Vec<isize> {
    let rank = dims.len();
    if rank == 0 {
        return vec![];
    }
    let mut strides = vec![1isize; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1] as isize;
    }
    strides
}

// ============================================================================
// StridedView
// ============================================================================

/// Dynamic-rank immutable strided view with lazy element operations.
///
/// This is the Julia-equivalent `StridedView` type with:
/// - Dynamic rank (dims/strides are heap-allocated)
/// - Lazy element operations via the `Op` type parameter
/// - Zero-copy transformations (permute, transpose, adjoint, conj)
///
/// # Type Parameters
/// - `'a`: Lifetime of the underlying data
/// - `T`: Element type
/// - `Op`: Element operation applied lazily on access (default: `Identity`)
pub struct StridedView<'a, T, Op: ElementOp = Identity> {
    ptr: *const T,
    data: &'a [T],
    dims: Arc<[usize]>,
    strides: Arc<[isize]>,
    offset: isize,
    _op: PhantomData<Op>,
}

unsafe impl<T: Send, Op: ElementOp> Send for StridedView<'_, T, Op> {}
unsafe impl<T: Sync, Op: ElementOp> Sync for StridedView<'_, T, Op> {}

impl<T, Op: ElementOp> Clone for StridedView<'_, T, Op> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            data: self.data,
            dims: self.dims.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            _op: PhantomData,
        }
    }
}

impl<T: std::fmt::Debug, Op: ElementOp> std::fmt::Debug for StridedView<'_, T, Op> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StridedView")
            .field("dims", &self.dims)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .finish()
    }
}

impl<'a, T, Op: ElementOp> StridedView<'a, T, Op> {
    /// Create a new immutable strided view from a borrowed slice.
    pub fn new(data: &'a [T], dims: &[usize], strides: &[isize], offset: isize) -> Result<Self> {
        validate_bounds(data.len(), dims, strides, offset)?;
        let ptr = unsafe { data.as_ptr().offset(offset) };
        Ok(Self {
            ptr,
            data,
            dims: Arc::from(dims),
            strides: Arc::from(strides),
            offset,
            _op: PhantomData,
        })
    }

    /// Create a view without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure all index combinations stay within bounds.
    pub unsafe fn new_unchecked(
        data: &'a [T],
        dims: &[usize],
        strides: &[isize],
        offset: isize,
    ) -> Self {
        let ptr = data.as_ptr().offset(offset);
        Self {
            ptr,
            data,
            dims: Arc::from(dims),
            strides: Arc::from(strides),
            offset,
            _op: PhantomData,
        }
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    #[inline]
    pub fn offset(&self) -> isize {
        self.offset
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dims.iter().any(|&d| d == 0)
    }

    #[inline]
    pub fn data(&self) -> &'a [T] {
        self.data
    }

    /// Raw const pointer to element at the view's base offset.
    #[inline]
    pub fn ptr(&self) -> *const T {
        self.ptr
    }

    /// Permute dimensions.
    pub fn permute(&self, perm: &[usize]) -> Result<StridedView<'a, T, Op>> {
        let rank = self.dims.len();
        if perm.len() != rank {
            return Err(StridedError::RankMismatch(perm.len(), rank));
        }
        let mut seen = vec![false; rank];
        for &p in perm {
            if p >= rank {
                return Err(StridedError::InvalidAxis { axis: p, rank });
            }
            if seen[p] {
                return Err(StridedError::InvalidAxis { axis: p, rank });
            }
            seen[p] = true;
        }
        let new_dims: Vec<usize> = perm.iter().map(|&p| self.dims[p]).collect();
        let new_strides: Vec<isize> = perm.iter().map(|&p| self.strides[p]).collect();
        Ok(StridedView {
            ptr: self.ptr,
            data: self.data,
            dims: Arc::from(new_dims),
            strides: Arc::from(new_strides),
            offset: self.offset,
            _op: PhantomData,
        })
    }

    /// Transpose a 2D view: reverses dimensions and composes the Transpose element op.
    ///
    /// Julia equivalent: `Base.transpose(a::AbstractStridedView{<:Any, 2})`
    pub fn transpose_2d(&self) -> Result<StridedView<'a, T, Op::ComposeTranspose>>
    where
        Op::ComposeTranspose: ElementOp,
    {
        if self.dims.len() != 2 {
            return Err(StridedError::RankMismatch(self.dims.len(), 2));
        }
        Ok(StridedView {
            ptr: self.ptr,
            data: self.data,
            dims: Arc::new([self.dims[1], self.dims[0]]),
            strides: Arc::new([self.strides[1], self.strides[0]]),
            offset: self.offset,
            _op: PhantomData,
        })
    }

    /// Adjoint (conjugate transpose) of a 2D view.
    ///
    /// Julia equivalent: `Base.adjoint(a::AbstractStridedView{<:Any, 2})`
    pub fn adjoint_2d(&self) -> Result<StridedView<'a, T, Op::ComposeAdjoint>>
    where
        Op::ComposeAdjoint: ElementOp,
    {
        if self.dims.len() != 2 {
            return Err(StridedError::RankMismatch(self.dims.len(), 2));
        }
        Ok(StridedView {
            ptr: self.ptr,
            data: self.data,
            dims: Arc::new([self.dims[1], self.dims[0]]),
            strides: Arc::new([self.strides[1], self.strides[0]]),
            offset: self.offset,
            _op: PhantomData,
        })
    }

    /// Complex conjugate (compose Conj without changing dims/strides).
    ///
    /// Julia equivalent: `Base.conj(a::AbstractStridedView)`
    pub fn conj(&self) -> StridedView<'a, T, Op::ComposeConj>
    where
        Op::ComposeConj: ElementOp,
    {
        StridedView {
            ptr: self.ptr,
            data: self.data,
            dims: self.dims.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            _op: PhantomData,
        }
    }

    /// Create a diagonal view by fusing repeated axis pairs via stride trick (zero-copy).
    ///
    /// For each pair `(a, b)`:
    /// - New stride = `strides[a] + strides[b]`
    /// - New dim = `min(dims[a], dims[b])`
    /// - The higher-numbered axis is removed
    /// - Pairs use **original** axis numbering
    ///
    /// # Example
    /// `A[i,i,j]` shape=`[n,n,m]` strides=`[s0,s1,s2]` -> shape=`[n,m]` strides=`[s0+s1, s2]`
    pub fn diagonal_view(&self, axis_pairs: &[(usize, usize)]) -> Result<StridedView<'a, T, Op>> {
        let ndim = self.ndim();
        let mut dims: Vec<usize> = self.dims().to_vec();
        let mut strides: Vec<isize> = self.strides().to_vec();

        let mut axes_to_remove = Vec::new();
        for &(a, b) in axis_pairs {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            if lo >= ndim || hi >= ndim {
                return Err(StridedError::InvalidAxis {
                    axis: hi,
                    rank: ndim,
                });
            }
            if lo == hi {
                return Err(StridedError::InvalidAxis {
                    axis: lo,
                    rank: ndim,
                });
            }
            strides[lo] = strides[lo] + strides[hi];
            dims[lo] = dims[lo].min(dims[hi]);
            axes_to_remove.push(hi);
        }

        axes_to_remove.sort_unstable();
        axes_to_remove.dedup();
        for &ax in axes_to_remove.iter().rev() {
            dims.remove(ax);
            strides.remove(ax);
        }

        unsafe {
            Ok(StridedView::new_unchecked(
                self.data(),
                &dims,
                &strides,
                self.offset(),
            ))
        }
    }

    /// Broadcast this view to a target shape.
    ///
    /// Size-1 dimensions are expanded (stride set to 0) to match target.
    pub fn broadcast(&self, target_dims: &[usize]) -> Result<StridedView<'a, T, Op>> {
        if self.dims.len() != target_dims.len() {
            return Err(StridedError::RankMismatch(
                self.dims.len(),
                target_dims.len(),
            ));
        }
        let mut new_strides = Vec::with_capacity(self.dims.len());
        for i in 0..self.dims.len() {
            if self.dims[i] == target_dims[i] {
                new_strides.push(self.strides[i]);
            } else if self.dims[i] == 1 {
                new_strides.push(0);
            } else {
                return Err(StridedError::ShapeMismatch(
                    self.dims.to_vec(),
                    target_dims.to_vec(),
                ));
            }
        }
        Ok(StridedView {
            ptr: self.ptr,
            data: self.data,
            dims: Arc::from(target_dims),
            strides: Arc::from(new_strides),
            offset: self.offset,
            _op: PhantomData,
        })
    }
}

impl<'a, T: Copy + ElementOpApply, Op: ElementOp> StridedView<'a, T, Op> {
    /// Get an element with the element operation applied.
    pub fn get(&self, indices: &[usize]) -> T {
        assert_eq!(indices.len(), self.dims.len(), "wrong number of indices");
        let mut idx = 0isize;
        for (i, &index) in indices.iter().enumerate() {
            assert!(
                index < self.dims[i],
                "index {} out of bounds for dim {}",
                index,
                self.dims[i]
            );
            idx += index as isize * self.strides[i];
        }
        Op::apply(unsafe { *self.ptr.offset(idx) })
    }

    /// Get an element without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure indices are within bounds.
    #[inline]
    pub unsafe fn get_unchecked(&self, indices: &[usize]) -> T {
        let mut idx = 0isize;
        for (i, &index) in indices.iter().enumerate() {
            idx += index as isize * self.strides[i];
        }
        Op::apply(*self.ptr.offset(idx))
    }
}

// ============================================================================
// StridedViewMut
// ============================================================================

/// Dynamic-rank mutable strided view.
///
/// Always uses `Identity` element operation for write simplicity.
/// Julia typically applies ops on the read side.
pub struct StridedViewMut<'a, T> {
    ptr: *mut T,
    data: &'a mut [T],
    dims: Arc<[usize]>,
    strides: Arc<[isize]>,
    offset: isize,
}

unsafe impl<T: Send> Send for StridedViewMut<'_, T> {}

impl<T: std::fmt::Debug> std::fmt::Debug for StridedViewMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StridedViewMut")
            .field("dims", &self.dims)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .finish()
    }
}

impl<'a, T> StridedViewMut<'a, T> {
    /// Create a new mutable strided view.
    pub fn new(
        data: &'a mut [T],
        dims: &[usize],
        strides: &[isize],
        offset: isize,
    ) -> Result<Self> {
        validate_bounds(data.len(), dims, strides, offset)?;
        let ptr = unsafe { data.as_mut_ptr().offset(offset) };
        Ok(Self {
            ptr,
            data,
            dims: Arc::from(dims),
            strides: Arc::from(strides),
            offset,
        })
    }

    /// Create without bounds checking.
    ///
    /// # Safety
    /// Caller must ensure all index combinations stay within bounds.
    pub unsafe fn new_unchecked(
        data: &'a mut [T],
        dims: &[usize],
        strides: &[isize],
        offset: isize,
    ) -> Self {
        let ptr = data.as_mut_ptr().offset(offset);
        Self {
            ptr,
            data,
            dims: Arc::from(dims),
            strides: Arc::from(strides),
            offset,
        }
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    #[inline]
    pub fn offset(&self) -> isize {
        self.offset
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dims.iter().any(|&d| d == 0)
    }

    /// Raw const pointer to element at the view's base offset.
    #[inline]
    pub fn ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// Raw mutable pointer to element at the view's base offset.
    #[inline]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Permute dimensions, consuming the mutable view.
    ///
    /// Returns a new mutable view with reordered dimensions and strides.
    /// Takes `self` by value to prevent aliasing of mutable views.
    pub fn permute(self, perm: &[usize]) -> Result<StridedViewMut<'a, T>> {
        let rank = self.dims.len();
        if perm.len() != rank {
            return Err(StridedError::RankMismatch(perm.len(), rank));
        }
        let mut seen = vec![false; rank];
        for &p in perm {
            if p >= rank {
                return Err(StridedError::InvalidAxis { axis: p, rank });
            }
            if seen[p] {
                return Err(StridedError::InvalidAxis { axis: p, rank });
            }
            seen[p] = true;
        }
        let new_dims: Vec<usize> = perm.iter().map(|&p| self.dims[p]).collect();
        let new_strides: Vec<isize> = perm.iter().map(|&p| self.strides[p]).collect();
        Ok(StridedViewMut {
            ptr: self.ptr,
            data: self.data,
            dims: Arc::from(new_dims),
            strides: Arc::from(new_strides),
            offset: self.offset,
        })
    }

    /// Reborrow as an immutable view.
    pub fn as_view(&self) -> StridedView<'_, T, Identity> {
        StridedView {
            ptr: self.ptr as *const T,
            data: unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.data.len()) },
            dims: self.dims.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            _op: PhantomData,
        }
    }
}

impl<'a, T: Copy> StridedViewMut<'a, T> {
    /// Get an element.
    pub fn get(&self, indices: &[usize]) -> T {
        assert_eq!(indices.len(), self.dims.len());
        let mut idx = 0isize;
        for (i, &index) in indices.iter().enumerate() {
            assert!(index < self.dims[i]);
            idx += index as isize * self.strides[i];
        }
        unsafe { *self.ptr.offset(idx) }
    }

    /// Set an element.
    pub fn set(&mut self, indices: &[usize], value: T) {
        assert_eq!(indices.len(), self.dims.len());
        let mut idx = 0isize;
        for (i, &index) in indices.iter().enumerate() {
            assert!(index < self.dims[i]);
            idx += index as isize * self.strides[i];
        }
        unsafe {
            *self.ptr.offset(idx) = value;
        }
    }
}

// ============================================================================
// StridedArray
// ============================================================================

/// Owned strided multidimensional array.
///
/// Supports both column-major (Julia default) and row-major (C default) layouts.
pub struct StridedArray<T> {
    data: Vec<T>,
    dims: Arc<[usize]>,
    strides: Arc<[isize]>,
    offset: isize,
}

impl<T: std::fmt::Debug> std::fmt::Debug for StridedArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StridedArray")
            .field("dims", &self.dims)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .finish()
    }
}

impl<T: Clone> Clone for StridedArray<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            dims: self.dims.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }
}

impl<T: Clone + Default> StridedArray<T> {
    /// Create a column-major (Julia default) tensor filled with Default values.
    pub fn col_major(dims: &[usize]) -> Self {
        let total: usize = dims.iter().product();
        let data = vec![T::default(); total];
        let strides = col_major_strides(dims);
        Self {
            data,
            dims: Arc::from(dims),
            strides: Arc::from(strides),
            offset: 0,
        }
    }

    /// Create a row-major (C default) tensor filled with Default values.
    pub fn row_major(dims: &[usize]) -> Self {
        let total: usize = dims.iter().product();
        let data = vec![T::default(); total];
        let strides = row_major_strides(dims);
        Self {
            data,
            dims: Arc::from(dims),
            strides: Arc::from(strides),
            offset: 0,
        }
    }

    /// Create a column-major tensor with values produced by a function.
    ///
    /// The function is called with indices in column-major iteration order.
    pub fn from_fn_col_major(dims: &[usize], mut f: impl FnMut(&[usize]) -> T) -> Self {
        let total: usize = dims.iter().product();
        let strides = col_major_strides(dims);
        let rank = dims.len();
        let mut data = Vec::with_capacity(total);
        let mut idx = vec![0usize; rank];
        for _ in 0..total {
            data.push(f(&idx));
            for d in 0..rank {
                idx[d] += 1;
                if idx[d] < dims[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Self {
            data,
            dims: Arc::from(dims),
            strides: Arc::from(strides),
            offset: 0,
        }
    }

    /// Create a row-major tensor with values produced by a function.
    ///
    /// The function is called with indices in row-major iteration order.
    pub fn from_fn_row_major(dims: &[usize], mut f: impl FnMut(&[usize]) -> T) -> Self {
        let total: usize = dims.iter().product();
        let strides = row_major_strides(dims);
        let rank = dims.len();
        let mut data = Vec::with_capacity(total);
        let mut idx = vec![0usize; rank];
        for _ in 0..total {
            data.push(f(&idx));
            for d in (0..rank).rev() {
                idx[d] += 1;
                if idx[d] < dims[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Self {
            data,
            dims: Arc::from(dims),
            strides: Arc::from(strides),
            offset: 0,
        }
    }
}

impl<T> StridedArray<T> {
    /// Create from raw parts.
    pub fn from_parts(
        data: Vec<T>,
        dims: &[usize],
        strides: &[isize],
        offset: isize,
    ) -> Result<Self> {
        validate_bounds(data.len(), dims, strides, offset)?;
        Ok(Self {
            data,
            dims: Arc::from(dims),
            strides: Arc::from(strides),
            offset,
        })
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dims.iter().any(|&d| d == 0)
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Create an immutable view over this tensor.
    pub fn view(&self) -> StridedView<'_, T> {
        let ptr = unsafe { self.data.as_ptr().offset(self.offset) };
        StridedView {
            ptr,
            data: &self.data,
            dims: self.dims.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            _op: PhantomData,
        }
    }

    /// Create a mutable view over this tensor.
    pub fn view_mut(&mut self) -> StridedViewMut<'_, T> {
        let ptr = unsafe { self.data.as_mut_ptr().offset(self.offset) };
        StridedViewMut {
            ptr,
            data: &mut self.data,
            dims: self.dims.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }

    /// Iterate over all elements in memory order.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Mutable iteration over all elements in memory order.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }
}

impl<T: Copy + ElementOpApply> StridedArray<T> {
    /// Get an element by multi-dimensional index.
    pub fn get(&self, indices: &[usize]) -> T {
        self.view().get(indices)
    }

    /// Set an element by multi-dimensional index.
    pub fn set(&mut self, indices: &[usize], value: T) {
        assert_eq!(indices.len(), self.dims.len());
        let mut idx = self.offset;
        for (i, &index) in indices.iter().enumerate() {
            assert!(index < self.dims[i]);
            idx += index as isize * self.strides[i];
        }
        self.data[idx as usize] = value;
    }
}

impl<T: Copy + ElementOpApply> Index<&[usize]> for StridedArray<T> {
    type Output = T;

    fn index(&self, indices: &[usize]) -> &T {
        let mut idx = self.offset;
        for (i, &index) in indices.iter().enumerate() {
            assert!(index < self.dims[i]);
            idx += index as isize * self.strides[i];
        }
        &self.data[idx as usize]
    }
}

impl<T: Copy + ElementOpApply> IndexMut<&[usize]> for StridedArray<T> {
    fn index_mut(&mut self, indices: &[usize]) -> &mut T {
        let mut idx = self.offset;
        for (i, &index) in indices.iter().enumerate() {
            assert!(index < self.dims[i]);
            idx += index as isize * self.strides[i];
        }
        &mut self.data[idx as usize]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_col_major_strides() {
        assert_eq!(col_major_strides(&[3, 4]), vec![1, 3]);
        assert_eq!(col_major_strides(&[2, 3, 4]), vec![1, 2, 6]);
    }

    #[test]
    fn test_row_major_strides() {
        assert_eq!(row_major_strides(&[3, 4]), vec![4, 1]);
        assert_eq!(row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn test_strided_view_new() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = StridedView::<f64>::new(&data, &[2, 3], &[3, 1], 0).unwrap();
        assert_eq!(view.ndim(), 2);
        assert_eq!(view.dims(), &[2, 3]);
        assert_eq!(view.strides(), &[3, 1]);
        assert_eq!(view.len(), 6);
    }

    #[test]
    fn test_strided_view_get() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = StridedView::<f64>::new(&data, &[2, 3], &[3, 1], 0).unwrap();
        assert_eq!(view.get(&[0, 0]), 1.0);
        assert_eq!(view.get(&[0, 1]), 2.0);
        assert_eq!(view.get(&[0, 2]), 3.0);
        assert_eq!(view.get(&[1, 0]), 4.0);
        assert_eq!(view.get(&[1, 2]), 6.0);
    }

    #[test]
    fn test_strided_view_col_major() {
        // Column-major: strides [1, 2] for 2x3
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = StridedView::<f64>::new(&data, &[2, 3], &[1, 2], 0).unwrap();
        assert_eq!(view.get(&[0, 0]), 1.0); // data[0]
        assert_eq!(view.get(&[1, 0]), 2.0); // data[1]
        assert_eq!(view.get(&[0, 1]), 3.0); // data[2]
        assert_eq!(view.get(&[1, 1]), 4.0); // data[3]
        assert_eq!(view.get(&[0, 2]), 5.0); // data[4]
        assert_eq!(view.get(&[1, 2]), 6.0); // data[5]
    }

    #[test]
    fn test_strided_view_permute() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = StridedView::<f64>::new(&data, &[2, 3], &[3, 1], 0).unwrap();
        let perm = view.permute(&[1, 0]).unwrap();
        assert_eq!(perm.dims(), &[3, 2]);
        assert_eq!(perm.strides(), &[1, 3]);
        assert_eq!(perm.get(&[0, 0]), 1.0);
        assert_eq!(perm.get(&[1, 0]), 2.0);
        assert_eq!(perm.get(&[0, 1]), 4.0);
    }

    #[test]
    fn test_strided_view_transpose_2d() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = StridedView::<f64>::new(&data, &[2, 3], &[3, 1], 0).unwrap();
        let t = view.transpose_2d().unwrap();
        assert_eq!(t.dims(), &[3, 2]);
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[1, 0]), 2.0);
        assert_eq!(t.get(&[0, 1]), 4.0);
    }

    #[test]
    fn test_strided_view_conj() {
        let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let view = StridedView::<Complex64>::new(&data, &[2], &[1], 0).unwrap();
        let c = view.conj();
        assert_eq!(c.get(&[0]), Complex64::new(1.0, -2.0));
        assert_eq!(c.get(&[1]), Complex64::new(3.0, -4.0));
    }

    #[test]
    fn test_strided_view_adjoint_2d() {
        let data = vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
            Complex64::new(7.0, 8.0),
        ];
        // 2x2 row-major
        let view = StridedView::<Complex64>::new(&data, &[2, 2], &[2, 1], 0).unwrap();
        let adj = view.adjoint_2d().unwrap();
        assert_eq!(adj.dims(), &[2, 2]);
        // Adjoint: conj + transpose
        assert_eq!(adj.get(&[0, 0]), Complex64::new(1.0, -2.0));
        assert_eq!(adj.get(&[1, 0]), Complex64::new(3.0, -4.0));
        assert_eq!(adj.get(&[0, 1]), Complex64::new(5.0, -6.0));
    }

    #[test]
    fn test_strided_view_broadcast() {
        let data = vec![1.0, 2.0, 3.0];
        let view = StridedView::<f64>::new(&data, &[1, 3], &[3, 1], 0).unwrap();
        let broad = view.broadcast(&[4, 3]).unwrap();
        assert_eq!(broad.dims(), &[4, 3]);
        for i in 0..4 {
            assert_eq!(broad.get(&[i, 0]), 1.0);
            assert_eq!(broad.get(&[i, 1]), 2.0);
            assert_eq!(broad.get(&[i, 2]), 3.0);
        }
    }

    #[test]
    fn test_strided_view_mut() {
        let mut data = vec![0.0; 6];
        {
            let mut view = StridedViewMut::<f64>::new(&mut data, &[2, 3], &[3, 1], 0).unwrap();
            view.set(&[0, 0], 1.0);
            view.set(&[1, 2], 6.0);
        }
        assert_eq!(data[0], 1.0);
        assert_eq!(data[5], 6.0);
    }

    #[test]
    fn test_strided_view_mut_as_view() {
        let mut data = vec![1.0, 2.0, 3.0];
        let vm = StridedViewMut::<f64>::new(&mut data, &[3], &[1], 0).unwrap();
        let v = vm.as_view();
        assert_eq!(v.get(&[0]), 1.0);
        assert_eq!(v.get(&[2]), 3.0);
    }

    #[test]
    fn test_strided_tensor_col_major() {
        let t = StridedArray::<f64>::from_fn_col_major(&[2, 3], |idx| (idx[0] * 3 + idx[1]) as f64);
        assert_eq!(t.dims(), &[2, 3]);
        assert_eq!(t.strides(), &[1, 2]); // column-major
        assert_eq!(t.get(&[0, 0]), 0.0);
        assert_eq!(t.get(&[1, 0]), 3.0);
        assert_eq!(t.get(&[0, 1]), 1.0);
        assert_eq!(t.get(&[1, 2]), 5.0);
    }

    #[test]
    fn test_strided_tensor_row_major() {
        let t = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1]) as f64);
        assert_eq!(t.dims(), &[2, 3]);
        assert_eq!(t.strides(), &[3, 1]); // row-major
        assert_eq!(t.get(&[0, 0]), 0.0);
        assert_eq!(t.get(&[0, 1]), 1.0);
        assert_eq!(t.get(&[1, 0]), 3.0);
        assert_eq!(t.get(&[1, 2]), 5.0);
    }

    #[test]
    fn test_strided_tensor_view() {
        let t =
            StridedArray::<f64>::from_fn_col_major(&[2, 3], |idx| (idx[0] * 10 + idx[1]) as f64);
        let v = t.view();
        assert_eq!(v.get(&[0, 0]), 0.0);
        assert_eq!(v.get(&[1, 0]), 10.0);
        assert_eq!(v.get(&[0, 2]), 2.0);
    }

    #[test]
    fn test_strided_tensor_view_mut() {
        let mut t = StridedArray::<f64>::col_major(&[2, 3]);
        {
            let mut vm = t.view_mut();
            vm.set(&[1, 2], 42.0);
        }
        assert_eq!(t.get(&[1, 2]), 42.0);
    }

    #[test]
    fn test_strided_tensor_index() {
        let t =
            StridedArray::<f64>::from_fn_row_major(&[3, 4], |idx| (idx[0] * 10 + idx[1]) as f64);
        assert_eq!(t[&[0usize, 0] as &[usize]], 0.0);
        assert_eq!(t[&[2usize, 3] as &[usize]], 23.0);
    }

    #[test]
    fn test_strided_tensor_index_mut() {
        let mut t = StridedArray::<f64>::row_major(&[2, 3]);
        t[&[1usize, 2] as &[usize]] = 99.0;
        assert_eq!(t.get(&[1, 2]), 99.0);
    }

    #[test]
    fn test_validate_bounds_ok() {
        assert!(validate_bounds(6, &[2, 3], &[3, 1], 0).is_ok());
        assert!(validate_bounds(6, &[2, 3], &[1, 2], 0).is_ok());
    }

    #[test]
    fn test_validate_bounds_out_of_range() {
        assert!(validate_bounds(5, &[2, 3], &[3, 1], 0).is_err());
    }

    #[test]
    fn test_validate_bounds_empty() {
        assert!(validate_bounds(0, &[0, 3], &[3, 1], 0).is_ok());
    }

    #[test]
    fn test_validate_bounds_with_offset() {
        assert!(validate_bounds(7, &[2, 3], &[3, 1], 1).is_ok());
        assert!(validate_bounds(6, &[2, 3], &[3, 1], 1).is_err());
    }

    #[test]
    fn test_strided_tensor_3d() {
        let t = StridedArray::<f64>::from_fn_col_major(&[2, 3, 4], |idx| {
            (idx[0] * 100 + idx[1] * 10 + idx[2]) as f64
        });
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.strides(), &[1, 2, 6]); // column-major
        assert_eq!(t.get(&[0, 0, 0]), 0.0);
        assert_eq!(t.get(&[1, 0, 0]), 100.0);
        assert_eq!(t.get(&[0, 1, 0]), 10.0);
        assert_eq!(t.get(&[0, 0, 1]), 1.0);
        assert_eq!(t.get(&[1, 2, 3]), 123.0);
    }

    #[test]
    fn test_diagonal_view_2d() {
        // A[i,i] shape=[3,3] row-major strides=[3,1]
        // diagonal: shape=[3] strides=[4] (3+1)
        let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
        let view = StridedView::<f64>::new(&data, &[3, 3], &[3, 1], 0).unwrap();
        let diag = view.diagonal_view(&[(0, 1)]).unwrap();
        assert_eq!(diag.dims(), &[3]);
        assert_eq!(diag.strides(), &[4]);
        assert_eq!(diag.get(&[0]), 0.0); // A[0,0]
        assert_eq!(diag.get(&[1]), 4.0); // A[1,1]
        assert_eq!(diag.get(&[2]), 8.0); // A[2,2]
    }

    #[test]
    fn test_diagonal_view_3d_adjacent() {
        // A[i,i,j] shape=[2,2,3] row-major strides=[6,3,1]
        // diagonal over (0,1): shape=[2,3] strides=[9,1] (6+3)
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let view = StridedView::<f64>::new(&data, &[2, 2, 3], &[6, 3, 1], 0).unwrap();
        let diag = view.diagonal_view(&[(0, 1)]).unwrap();
        assert_eq!(diag.dims(), &[2, 3]);
        assert_eq!(diag.strides(), &[9, 1]);
        assert_eq!(diag.get(&[0, 0]), 0.0);
        assert_eq!(diag.get(&[0, 2]), 2.0);
        assert_eq!(diag.get(&[1, 0]), 9.0);
    }

    #[test]
    fn test_diagonal_view_3d_non_adjacent() {
        // A[i,j,i] shape=[2,3,2] row-major strides=[6,2,1]
        // diagonal over (0,2): shape=[2,3] strides=[7,2] (6+1)
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let view = StridedView::<f64>::new(&data, &[2, 3, 2], &[6, 2, 1], 0).unwrap();
        let diag = view.diagonal_view(&[(0, 2)]).unwrap();
        assert_eq!(diag.dims(), &[2, 3]);
        assert_eq!(diag.strides(), &[7, 2]);
        assert_eq!(diag.get(&[0, 0]), 0.0);
        assert_eq!(diag.get(&[0, 1]), 2.0);
        assert_eq!(diag.get(&[1, 0]), 7.0);
        assert_eq!(diag.get(&[1, 2]), 11.0);
    }

    #[test]
    fn test_diagonal_view_two_pairs() {
        // A[i,j,i,j] shape=[2,3,2,3] -> A_diag[i,j] shape=[2,3]
        let data: Vec<f64> = (0..36).map(|x| x as f64).collect();
        let view = StridedView::<f64>::new(&data, &[2, 3, 2, 3], &[18, 6, 3, 1], 0).unwrap();
        let diag = view.diagonal_view(&[(0, 2), (1, 3)]).unwrap();
        assert_eq!(diag.dims(), &[2, 3]);
        assert_eq!(diag.strides(), &[21, 7]);
        assert_eq!(diag.get(&[0, 0]), 0.0);
        assert_eq!(diag.get(&[1, 1]), 28.0);
    }
}
