//! Julia-like dynamic-rank strided view types.
//!
//! This module provides the canonical view types for strided operations,
//! matching Julia's StridedViews.jl data model:
//!
//! - [`StridedView`]: Immutable dynamic-rank strided view with lazy element operations
//! - [`StridedViewMut`]: Mutable dynamic-rank strided view (Identity op only)
//! - [`StridedArray`]: Owned strided multidimensional array
//!
//! A view's shape and strides are a small inline-or-heap container with
//! capacity [`crate::INLINE_RANK`]: every constructor, clone, and metadata
//! transform allocates nothing up to that rank and spills gracefully above
//! it. The view itself only borrows the element data.
//!
//! The metadata container is `tinyvec::ArrayVec`, unified across tensor4all
//! repositories (tensor4all-agent-rules#22, strided-rs#312). `ArrayVec` is
//! `Copy` for `Copy` items, so a clone is one fixed-size bitwise copy, and the
//! heap fallback for rank above [`INLINE_RANK`] lives in a separate
//! `Option<Box<SpilledMeta>>` so the inline value has no drop glue.

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut};

use tinyvec::ArrayVec;

use crate::element_op::{ComposableElementOp, ElementOp, ElementOpApply, Identity};
use crate::{AxisVec, Result, StrideVec, StridedError, INLINE_RANK};

/// Build an `ArrayVec` from at most [`INLINE_RANK`] elements.
///
/// The value is written with literal-count arms: a runtime-length
/// `copy_from_slice` or loop is recognized by LLVM and lowered to a `memcpy`
/// PLT call, whose overhead dominates a handful of words.
#[inline(always)]
fn arrayvec_from_slice<T: Copy + Default>(src: &[T]) -> ArrayVec<[T; INLINE_RANK]> {
    debug_assert!(src.len() <= INLINE_RANK);
    let n = src.len();
    let mut buf = [T::default(); INLINE_RANK];
    match n {
        0 => {}
        1 => buf[0] = src[0],
        2 => {
            buf[0] = src[0];
            buf[1] = src[1];
        }
        3 => {
            buf[0] = src[0];
            buf[1] = src[1];
            buf[2] = src[2];
        }
        4 => {
            buf[0] = src[0];
            buf[1] = src[1];
            buf[2] = src[2];
            buf[3] = src[3];
        }
        5 => {
            buf[0] = src[0];
            buf[1] = src[1];
            buf[2] = src[2];
            buf[3] = src[3];
            buf[4] = src[4];
        }
        6 => {
            buf[0] = src[0];
            buf[1] = src[1];
            buf[2] = src[2];
            buf[3] = src[3];
            buf[4] = src[4];
            buf[5] = src[5];
        }
        7 => {
            buf[0] = src[0];
            buf[1] = src[1];
            buf[2] = src[2];
            buf[3] = src[3];
            buf[4] = src[4];
            buf[5] = src[5];
            buf[6] = src[6];
        }
        _ => {
            buf[0] = src[0];
            buf[1] = src[1];
            buf[2] = src[2];
            buf[3] = src[3];
            buf[4] = src[4];
            buf[5] = src[5];
            buf[6] = src[6];
            buf[7] = src[7];
        }
    }
    ArrayVec::from_array_len(buf, n)
}

/// Heap-owned metadata used only above [`INLINE_RANK`].
#[derive(Clone, Debug)]
struct SpilledMeta {
    dims: Box<[usize]>,
    strides: Box<[isize]>,
}

/// Shape and strides of a view or array.
///
/// Rank `<= INLINE_RANK` is stored inline in two `ArrayVec`s and the
/// `spilled` slot is `None`, so cloning the whole metadata is a fixed-size
/// bitwise copy with one predictable `None` check. A higher rank keeps the
/// inline buffers empty and boxes both vectors.
#[derive(Debug)]
struct ViewMeta {
    inline_dims: ArrayVec<[usize; INLINE_RANK]>,
    inline_strides: ArrayVec<[isize; INLINE_RANK]>,
    spilled: Option<Box<SpilledMeta>>,
}

impl Clone for ViewMeta {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inline_dims: self.inline_dims,
            inline_strides: self.inline_strides,
            spilled: self.spilled.clone(),
        }
    }
}

impl ViewMeta {
    #[inline(always)]
    fn from_slices(dims: &[usize], strides: &[isize]) -> Self {
        debug_assert_eq!(dims.len(), strides.len());
        if dims.len() <= INLINE_RANK {
            Self {
                inline_dims: arrayvec_from_slice(dims),
                inline_strides: arrayvec_from_slice(strides),
                spilled: None,
            }
        } else {
            Self {
                inline_dims: ArrayVec::new(),
                inline_strides: ArrayVec::new(),
                spilled: Some(Box::new(SpilledMeta {
                    dims: dims.into(),
                    strides: strides.into(),
                })),
            }
        }
    }

    #[inline]
    fn dims(&self) -> &[usize] {
        match &self.spilled {
            Some(spilled) => &spilled.dims,
            None => self.inline_dims.as_slice(),
        }
    }

    #[inline]
    fn strides(&self) -> &[isize] {
        match &self.spilled {
            Some(spilled) => &spilled.strides,
            None => self.inline_strides.as_slice(),
        }
    }
}

#[inline]
fn empty_layout(dims: &[usize]) -> bool {
    dims.contains(&0)
}

#[inline]
fn empty_const_ptr<T>() -> *const T {
    core::ptr::NonNull::<T>::dangling().as_ptr()
}

#[inline]
fn empty_mut_ptr<T>() -> *mut T {
    core::ptr::NonNull::<T>::dangling().as_ptr()
}

// ============================================================================
// Validation helpers
// ============================================================================

/// Validate that all accessed offsets stay within `[0, len)`.
#[inline]
pub(crate) fn validate_bounds(
    len: usize,
    dims: &[usize],
    strides: &[isize],
    offset: isize,
) -> Result<()> {
    if dims.len() != strides.len() {
        return Err(StridedError::StrideLengthMismatch);
    }
    // Empty array - no access needed
    if dims.contains(&0) {
        return Ok(());
    }
    // Compute min and max offsets
    let mut min_offset = offset;
    let mut max_offset = offset;
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        if dim > 1 {
            let last = dim
                .checked_sub(1)
                .and_then(|value| isize::try_from(value).ok())
                .ok_or(StridedError::OffsetOverflow)?;
            let end = stride
                .checked_mul(last)
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

/// Offset of `indices` inside `dims`/`strides`, relative to `base`.
///
/// # Panics
///
/// Panics when `indices` is not the full index for `dims`, an index is out of
/// bounds, or an indexed offset does not fit `isize`. Every panicking element
/// accessor below reports those conditions through this helper.
#[inline]
fn indexed_offset(dims: &[usize], strides: &[isize], indices: &[usize], base: isize) -> isize {
    assert_eq!(indices.len(), dims.len(), "wrong number of indices");
    let mut offset = base;
    for (i, &index) in indices.iter().enumerate() {
        assert!(
            index < dims[i],
            "index {} out of bounds for dim {}",
            index,
            dims[i]
        );
        offset = isize::try_from(index)
            .ok()
            .and_then(|index| index.checked_mul(strides[i]))
            .and_then(|step| offset.checked_add(step))
            .expect("index offset overflow");
    }
    offset
}

/// Number of elements of `dims`: the checked product of the extents.
///
/// A zero extent wins before the product, so a zero-sized shape is empty even
/// when its other extents would overflow. Returns
/// [`StridedError::OffsetOverflow`] when the product does not fit `usize`
/// (tensor4all/strided-rs#205).
#[inline]
fn checked_element_count(dims: &[usize]) -> Result<usize> {
    if dims.contains(&0) {
        return Ok(0);
    }
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or(StridedError::OffsetOverflow)
}

/// [`checked_element_count`] for the infallible `len` accessors.
///
/// # Panics
///
/// Panics when the element count does not fit `usize`, which only a
/// broadcast (stride-0) layout can reach.
#[inline]
fn element_count(dims: &[usize]) -> usize {
    checked_element_count(dims).expect("element count overflows usize")
}

/// Dense strides of `dims` with the axes visited fastest first in `order`.
///
/// Each stride is the product of the extents of the faster axes and must fit
/// `isize`; otherwise [`StridedError::OffsetOverflow`]. A zero-sized shape is
/// never dereferenced, so its strides saturate instead of failing: an empty
/// array with huge extents stays constructible.
fn checked_dense_strides(dims: &[usize], order: impl Iterator<Item = usize>) -> Result<StrideVec> {
    let empty = dims.contains(&0);
    let mut strides = StrideVec::new();
    strides.resize(dims.len(), 0isize);
    let mut step = 1isize;
    for axis in order {
        strides[axis] = step;
        let dim = isize::try_from(dims[axis]).ok();
        step = match dim.and_then(|dim| step.checked_mul(dim)) {
            Some(next) => next,
            None if empty => isize::MAX,
            // The last axis's step is never stored, but it is the element
            // count, which must also fit for the allocation and the offsets.
            None => return Err(StridedError::OffsetOverflow),
        };
    }
    Ok(strides)
}

fn checked_col_major_strides(dims: &[usize]) -> Result<StrideVec> {
    checked_dense_strides(dims, 0..dims.len())
}

fn checked_row_major_strides(dims: &[usize]) -> Result<StrideVec> {
    checked_dense_strides(dims, (0..dims.len()).rev())
}

/// Compute column-major strides (Julia default: first index varies fastest).
///
/// Returns an inline [`StrideVec`]: the result does not allocate at rank
/// `<= INLINE_RANK` and spills above it.
///
/// # Panics
///
/// Panics when a stride or the element count of a non-empty shape does not
/// fit `isize`. A shape with a zero extent never panics.
///
/// # Examples
///
/// ```
/// use strided_view::col_major_strides;
///
/// // Julia default: first index varies fastest.
/// assert_eq!(col_major_strides(&[2, 3]).as_slice(), &[1, 2]);
/// ```
#[inline]
pub fn col_major_strides(dims: &[usize]) -> StrideVec {
    checked_col_major_strides(dims).expect("column-major strides overflow isize")
}

/// Compute row-major strides (C default: last index varies fastest).
///
/// Returns an inline [`StrideVec`]: the result does not allocate at rank
/// `<= INLINE_RANK` and spills above it.
///
/// # Panics
///
/// Panics when a stride or the element count of a non-empty shape does not
/// fit `isize`. A shape with a zero extent never panics.
///
/// # Examples
///
/// ```
/// use strided_view::row_major_strides;
///
/// // C default: last index varies fastest.
/// assert_eq!(row_major_strides(&[2, 3]).as_slice(), &[3, 1]);
/// ```
#[inline]
pub fn row_major_strides(dims: &[usize]) -> StrideVec {
    checked_row_major_strides(dims).expect("row-major strides overflow isize")
}

// ============================================================================
// StridedView
// ============================================================================

/// Dynamic-rank immutable strided view with lazy element operations.
///
/// This is the Julia-equivalent `StridedView` type with:
/// - Dynamic rank, with metadata stored inline for up to [`INLINE_RANK`]
///   dimensions ([`StridedView::new`] copies the caller's layout, so the view
///   borrows only the element data)
/// - Lazy element operations via the `Op` type parameter
/// - Zero-copy transformations (permute, transpose, adjoint, conj)
///
/// # Type Parameters
/// - `'a`: Lifetime of the underlying data
/// - `T`: Element type
/// - `Op`: Element operation applied lazily on access (default: `Identity`)
///
/// # Size
///
/// The inline metadata makes the view 184 bytes on 64-bit targets for
/// `usize`/`isize` strides.
///
/// # Auto traits
///
/// The view borrows its elements, so it is `Send` only when `T: Sync` (a
/// shared slice is). A view of non-`Sync` data is not `Send`:
///
/// ```compile_fail
/// use std::cell::Cell;
/// use strided_view::StridedView;
///
/// fn assert_send<T: Send>() {}
/// assert_send::<StridedView<'static, Cell<i32>>>();
/// ```
pub struct StridedView<'a, T, Op = Identity> {
    ptr: *const T,
    data: &'a [T],
    meta: ViewMeta,
    offset: isize,
    _op: PhantomData<Op>,
}

// SAFETY: `StridedView` owns no `T`. It holds `&'a [T]` and a `*const T`
// derived from it, so moving the view moves a shared borrow, which is `Send`
// exactly when `T: Sync`. `Op` appears only in `PhantomData`, and `ViewMeta` is
// owned metadata (an inline `tinyvec::ArrayVec` plus an optional box).
unsafe impl<T: Sync, Op: Send> Send for StridedView<'_, T, Op> {}
// SAFETY: a shared `&StridedView` hands out only `&T` (`get`, `ptr`, the
// `ElementOp` read path) and never `&mut T`, so sharing is sound when `T: Sync`.
// `Op` is `Sync` and appears only in `PhantomData`.
unsafe impl<T: Sync, Op: Sync> Sync for StridedView<'_, T, Op> {}

impl<T, Op> Clone for StridedView<'_, T, Op> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            data: self.data,
            meta: self.meta.clone(),
            offset: self.offset,
            _op: PhantomData,
        }
    }
}

impl<T: std::fmt::Debug, Op> std::fmt::Debug for StridedView<'_, T, Op> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StridedView")
            .field("dims", &self.meta.dims())
            .field("strides", &self.meta.strides())
            .field("offset", &self.offset)
            .finish()
    }
}

/// The cached base pointer of a view over `data` at `offset`.
///
/// An empty layout gets the dangling pointer rather than `data + offset`: a
/// zero-sized extent may sit at an offset no element of `data` reaches.
#[inline]
fn base_ptr<T>(data: &[T], dims: &[usize], offset: isize) -> *const T {
    if empty_layout(dims) {
        empty_const_ptr()
    } else {
        // SAFETY: every caller reaches `base_ptr` from `from_meta`, whose
        // `dims`/`strides`/`offset` were either accepted by `validate_bounds`
        // (view.rs:187) — `StridedView::new`, `StridedViewMut::new`, the
        // re-validated diagonal path, or a metadata transform that only reorders
        // axes or zeroes strides and therefore cannot enlarge the reachable
        // offset set — or were supplied under the `new_unchecked` contract. The
        // empty-layout branch above excludes the case where `offset` may point
        // past `data`.
        unsafe { data.as_ptr().offset(offset) }
    }
}

/// A view from already-validated metadata, with the element op chosen here.
#[inline]
fn from_meta<'a, T, Op>(
    data: &'a [T],
    dims: &[usize],
    strides: &[isize],
    offset: isize,
) -> StridedView<'a, T, Op> {
    let meta = ViewMeta::from_slices(dims, strides);
    let ptr = base_ptr(data, meta.dims(), offset);
    StridedView {
        ptr,
        data,
        meta,
        offset,
        _op: PhantomData,
    }
}

impl<'a, T, Op> StridedView<'a, T, Op> {
    /// Create an immutable strided view, copying the caller's metadata inline.
    ///
    /// `dims`, `strides`, and `offset` are in units of `T` (elements, not
    /// bytes).
    ///
    /// The layout is copied into the view's inline storage, so it does not
    /// need to outlive the view and no heap allocation happens at rank
    /// `<= INLINE_RANK`. The view's lifetime is tied to `data` only.
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
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_view::StridedView;
    ///
    /// let data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let view = StridedView::<f64>::new(&data, &[2, 3], &[1, 2], 0).unwrap();
    /// assert_eq!(view.dims(), &[2, 3]);
    /// assert_eq!(view.get(&[1, 2]), 6.0);
    /// ```
    #[inline]
    pub fn new(data: &'a [T], dims: &[usize], strides: &[isize], offset: isize) -> Result<Self> {
        validate_bounds(data.len(), dims, strides, offset)?;
        Ok(from_meta(data, dims, strides, offset))
    }

    /// Create a view without bounds checking, copying the caller's metadata.
    ///
    /// `dims`, `strides`, and `offset` are in units of `T` (elements, not
    /// bytes).
    ///
    /// # Safety
    /// The caller must ensure all index combinations stay within bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_view::StridedView;
    ///
    /// let data = [1.0f64, 2.0, 3.0, 4.0];
    /// // SAFETY: the largest reachable offset, 1 * 1 + 1 * 2 = 3, is below
    /// // `data.len() == 4`.
    /// let view = unsafe { StridedView::<f64>::new_unchecked(&data, &[2, 2], &[1, 2], 0) };
    /// assert_eq!(view.ndim(), 2);
    /// ```
    #[inline]
    pub unsafe fn new_unchecked(
        data: &'a [T],
        dims: &[usize],
        strides: &[isize],
        offset: isize,
    ) -> Self {
        from_meta(data, dims, strides, offset)
    }

    /// Returns the shape (dimension sizes) of this view.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        self.meta.dims()
    }

    /// Returns the strides (in units of `T`) for each dimension.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.meta.strides()
    }

    /// Returns the offset into the backing data, in units of `T` (elements, not bytes).
    #[inline]
    pub fn offset(&self) -> isize {
        self.offset
    }

    /// Returns the number of dimensions (rank).
    #[inline]
    pub fn ndim(&self) -> usize {
        self.dims().len()
    }

    /// Returns the total number of elements.
    ///
    /// # Panics
    ///
    /// Panics when the element count does not fit `usize`, which only a
    /// broadcast (stride-0) layout can reach.
    #[inline]
    pub fn len(&self) -> usize {
        element_count(self.dims())
    }

    /// Returns `true` if any dimension is zero.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dims().contains(&0)
    }

    /// Returns a reference to the backing data slice.
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
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::RankMismatch`] when `perm` does not have one
    /// entry per axis, and [`StridedError::InvalidAxis`] when an entry
    /// names an axis outside the rank or repeats an axis.
    pub fn permute(&self, perm: &[usize]) -> Result<StridedView<'a, T, Op>> {
        let rank = self.dims().len();
        if perm.len() != rank {
            return Err(StridedError::RankMismatch(perm.len(), rank));
        }
        let mut seen: tinyvec::TinyVec<[bool; INLINE_RANK]> = tinyvec::TinyVec::new();
        seen.resize(rank, false);
        for &p in perm {
            if p >= rank {
                return Err(StridedError::InvalidAxis { axis: p, rank });
            }
            if seen[p] {
                return Err(StridedError::InvalidAxis { axis: p, rank });
            }
            seen[p] = true;
        }
        let new_dims: AxisVec = perm.iter().map(|&p| self.dims()[p]).collect();
        let new_strides: StrideVec = perm.iter().map(|&p| self.strides()[p]).collect();
        Ok(from_meta(self.data, &new_dims, &new_strides, self.offset))
    }

    /// Create a diagonal view by fusing repeated axis pairs via stride trick (zero-copy).
    ///
    /// Each pair `(a, b)` ties axes `a` and `b` to one index. Pairs that share
    /// an axis chain into one class (`[(0, 1), (1, 2)]` and `[(0, 1), (0, 2)]`
    /// both tie axes 0, 1 and 2), and each class becomes one axis:
    /// - New stride = the sum of the strides of the axes in the class
    /// - New dim = the minimum of their dims
    /// - The class sits at its lowest-numbered axis; the others are removed
    /// - Pairs use **original** axis numbering; their order does not matter,
    ///   and repeating a pair has no further effect
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::InvalidAxis`] when a pair names an axis outside
    /// the rank or names the same axis twice, and
    /// [`StridedError::OffsetOverflow`] when a fused stride does not fit
    /// `isize` or the fused layout reaches past the backing allocation.
    ///
    /// # Example
    /// `A[i,i,j]` shape=`[n,n,m]` strides=`[s0,s1,s2]` -> shape=`[n,m]` strides=`[s0+s1, s2]`
    pub fn diagonal_view(&self, axis_pairs: &[(usize, usize)]) -> Result<StridedView<'a, T, Op>> {
        let ndim = self.ndim();
        // Each pair ties two axes to one index, so the pairs partition the
        // axes into classes. Track each class's root as its lowest axis; the
        // result is then independent of pair order and of how pairs chain
        // (tensor4all/strided-rs#329).
        let mut root: tinyvec::TinyVec<[usize; INLINE_RANK]> = (0..ndim).collect();
        fn find(root: &mut [usize], mut ax: usize) -> usize {
            while root[ax] != ax {
                root[ax] = root[root[ax]];
                ax = root[ax];
            }
            ax
        }
        for &(a, b) in axis_pairs {
            let hi = a.max(b);
            if hi >= ndim {
                return Err(StridedError::InvalidAxis {
                    axis: hi,
                    rank: ndim,
                });
            }
            if a == b {
                return Err(StridedError::InvalidAxis {
                    axis: a,
                    rank: ndim,
                });
            }
            let (ra, rb) = (find(&mut root, a), find(&mut root, b));
            root[ra.max(rb)] = ra.min(rb);
        }

        // Every class collapses onto its root: the root's stride sums the
        // strides of all axes in the class and its dim is their minimum. A
        // root is lower than every other axis of its class, so it has already
        // been kept when a later member folds into it.
        let mut dims: AxisVec = AxisVec::new();
        let mut strides: StrideVec = StrideVec::new();
        let mut position: tinyvec::TinyVec<[usize; INLINE_RANK]> =
            tinyvec::TinyVec::with_capacity(ndim);
        for ax in 0..ndim {
            let r = find(&mut root, ax);
            if r == ax {
                position.push(dims.len());
                dims.push(self.dims()[ax]);
                strides.push(self.strides()[ax]);
            } else {
                position.push(usize::MAX);
                let p = position[r];
                strides[p] = strides[p]
                    .checked_add(self.strides()[ax])
                    .ok_or(StridedError::OffsetOverflow)?;
                dims[p] = dims[p].min(self.dims()[ax]);
            }
        }

        // A fused stride sums several input strides, so the result carries no
        // bound proof of its own and is validated here instead
        // (tensor4all/strided-rs#204).
        validate_bounds(self.data().len(), &dims, &strides, self.offset())?;

        Ok(from_meta(self.data(), &dims, &strides, self.offset()))
    }

    /// Broadcast this view to a target shape.
    ///
    /// Size-1 dimensions are expanded (stride set to 0) to match target.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::RankMismatch`] when `target_dims` does not
    /// have one entry per axis, and [`StridedError::ShapeMismatch`] when an
    /// axis is neither already the target extent nor of size one.
    pub fn broadcast(&self, target_dims: &[usize]) -> Result<StridedView<'a, T, Op>> {
        if self.dims().len() != target_dims.len() {
            return Err(StridedError::RankMismatch(
                self.dims().len(),
                target_dims.len(),
            ));
        }
        let mut new_strides = StrideVec::with_capacity(self.dims().len());
        for i in 0..self.dims().len() {
            if self.dims()[i] == target_dims[i] {
                new_strides.push(self.strides()[i]);
            } else if self.dims()[i] == 1 {
                new_strides.push(0);
            } else {
                return Err(StridedError::ShapeMismatch(
                    self.dims().to_vec(),
                    target_dims.to_vec(),
                ));
            }
        }
        Ok(from_meta(self.data, target_dims, &new_strides, self.offset))
    }
}

/// Composition methods: require `T: ElementOpApply` and `Op: ComposableElementOp<T>`.
impl<'a, T: Copy + ElementOpApply, Op: ComposableElementOp<T>> StridedView<'a, T, Op> {
    /// Transpose a 2D view: reverses dimensions and composes the Transpose element op.
    ///
    /// Julia equivalent: `Base.transpose(a::AbstractStridedView{<:Any, 2})`
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::RankMismatch`] when the view is not 2-D.
    pub fn transpose_2d(&self) -> Result<StridedView<'a, T, Op::ComposeTranspose>> {
        if self.dims().len() != 2 {
            return Err(StridedError::RankMismatch(self.dims().len(), 2));
        }
        Ok(from_meta(
            self.data,
            &[self.dims()[1], self.dims()[0]],
            &[self.strides()[1], self.strides()[0]],
            self.offset,
        ))
    }

    /// Adjoint (conjugate transpose) of a 2D view.
    ///
    /// Julia equivalent: `Base.adjoint(a::AbstractStridedView{<:Any, 2})`
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::RankMismatch`] when the view is not 2-D.
    pub fn adjoint_2d(&self) -> Result<StridedView<'a, T, Op::ComposeAdjoint>> {
        if self.dims().len() != 2 {
            return Err(StridedError::RankMismatch(self.dims().len(), 2));
        }
        Ok(from_meta(
            self.data,
            &[self.dims()[1], self.dims()[0]],
            &[self.strides()[1], self.strides()[0]],
            self.offset,
        ))
    }

    /// Complex conjugate (compose Conj without changing dims/strides).
    ///
    /// Julia equivalent: `Base.conj(a::AbstractStridedView)`
    #[inline]
    pub fn conj(&self) -> StridedView<'a, T, Op::ComposeConj> {
        StridedView {
            ptr: self.ptr,
            data: self.data,
            meta: self.meta.clone(),
            offset: self.offset,
            _op: PhantomData,
        }
    }
}

/// Element access: requires `Op: ElementOp<T>`.
impl<'a, T: Copy, Op: ElementOp<T>> StridedView<'a, T, Op> {
    /// Get an element with the element operation applied.
    ///
    /// # Panics
    ///
    /// Panics if `indices.len()` is not the view's rank, an index is out of
    /// bounds, or the indexed offset does not fit `isize`.
    pub fn get(&self, indices: &[usize]) -> T {
        let offset = indexed_offset(self.dims(), self.strides(), indices, 0);
        // SAFETY: `indexed_offset` (view.rs:240) asserts the index count and
        // `index < dims[i]`, so the offset it returns lies within the reachable
        // extent that `validate_bounds` (view.rs:187) proved stays inside
        // `data` for this view's construction (or that the `new_unchecked`
        // contract promised). A view from `StridedArray::view`
        // (view.rs:1461) carries the same proof; see the comment there.
        Op::apply(unsafe { *self.ptr.offset(offset) })
    }

    /// Get an element without bounds checking.
    ///
    /// # Safety
    ///
    /// `indices.len()` must equal `self.ndim()`, and `indices[i] < self.dims()[i]`
    /// for every axis `i`.
    ///
    /// A shorter `indices` does not mean "the missing axes are indexed with 0":
    /// the loop below skips the axes it was not given, so on a view with a
    /// zero-extent axis it dereferences the view's dangling base pointer.
    /// Debug builds reject a wrong `indices.len()` with `debug_assert_eq!`.
    #[inline]
    pub unsafe fn get_unchecked(&self, indices: &[usize]) -> T {
        debug_assert_eq!(indices.len(), self.ndim(), "wrong number of indices");
        let mut idx = 0isize;
        for (i, &index) in indices.iter().enumerate() {
            idx += index as isize * self.strides()[i];
        }
        // SAFETY: this function's `# Safety` contract (above) requires one
        // index per axis, `indices.len() == self.ndim()`, with `index < dims[i]`
        // for each. That makes `idx` an offset the view's construction proved
        // reachable: `validate_bounds` (view.rs:187), the dense constructors
        // (see `StridedArray::view`, view.rs:1461), or the `new_unchecked`
        // contract. `ptr` is the view's base, so `ptr + idx` stays inside
        // `data`. On a view with a zero-extent axis no index satisfies
        // `index < dims[i]`, so the contract, not the debug-only arity check,
        // is what keeps its dangling base unreached.
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
    /// Element base: `base` plus `offset`, or a dangling pointer for an empty
    /// layout. `get` and `set` go through it, so it stays cached.
    ///
    /// Nothing here reborrows the caller's `&mut [T]` any more: a stored slice
    /// field was retagged at construction and by `data_mut`, which invalidated
    /// this pointer and made every later `get`, `set`, and `as_view` access
    /// undefined behavior.
    ptr: *mut T,
    /// Base of the whole backing slice.
    ///
    /// `base` with `src_len` is the handle `data` and `data_mut` re-derive the
    /// slice from, so the backing extent stays independent of the shape (a
    /// `[0]` shape still has a backing slice).
    base: *mut T,
    /// Length of the whole backing slice.
    src_len: usize,
    meta: ViewMeta,
    offset: isize,
    /// The view borrows the caller's slice for `'a` without holding it.
    _marker: PhantomData<&'a mut [T]>,
}

// SAFETY: `StridedViewMut` uniquely owns its `&'a mut [T]` (held as
// `PhantomData<&'a mut [T]>` plus the two raw pointers `base` and `ptr` derived
// from it), and there is no `Sync` impl for it, so sending it moves that
// exclusive borrow; `T: Send` is the bound `&mut [T]: Send` requires.
unsafe impl<T: Send> Send for StridedViewMut<'_, T> {}

impl<T: std::fmt::Debug> std::fmt::Debug for StridedViewMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StridedViewMut")
            .field("dims", &self.meta.dims())
            .field("strides", &self.meta.strides())
            .field("offset", &self.offset)
            .finish()
    }
}

impl<'a, T> StridedViewMut<'a, T> {
    /// A view from already-validated metadata.
    #[inline]
    fn from_meta(data: &'a mut [T], dims: &[usize], strides: &[isize], offset: isize) -> Self {
        let meta = ViewMeta::from_slices(dims, strides);
        let src_len = data.len();
        let base = data.as_mut_ptr();
        let ptr = if empty_layout(meta.dims()) {
            empty_mut_ptr()
        } else {
            // SAFETY: as `base_ptr` above — `offset` was accepted by
            // `validate_bounds` (view.rs:187) in `StridedViewMut::new`, or by a
            // metadata transform that only reorders axes or zeroes strides, or
            // came from the `new_unchecked` contract. `base` is this view's own
            // live handle on `data` and the empty-layout branch above excludes
            // the case where `offset` may point past it.
            unsafe { base.offset(offset) }
        };
        Self {
            ptr,
            base,
            src_len,
            meta,
            offset,
            _marker: PhantomData,
        }
    }

    /// Create a mutable view, copying the caller's metadata inline.
    ///
    /// `dims`, `strides`, and `offset` are in units of `T` (elements, not
    /// bytes).
    ///
    /// The layout is copied, so it does not need to outlive the view and no
    /// heap allocation happens at rank `<= INLINE_RANK`.
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
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_view::StridedViewMut;
    ///
    /// let mut data = [0.0f64; 4];
    /// let mut view = StridedViewMut::<f64>::new(&mut data, &[2, 2], &[1, 2], 0).unwrap();
    /// view.set(&[1, 1], 5.0);
    /// assert_eq!(data[3], 5.0);
    /// ```
    #[inline]
    pub fn new(
        data: &'a mut [T],
        dims: &[usize],
        strides: &[isize],
        offset: isize,
    ) -> Result<Self> {
        validate_bounds(data.len(), dims, strides, offset)?;
        Ok(Self::from_meta(data, dims, strides, offset))
    }

    /// Create without bounds checking, copying the caller's metadata.
    ///
    /// `dims`, `strides`, and `offset` are in units of `T` (elements, not
    /// bytes).
    ///
    /// # Safety
    /// Caller must ensure all index combinations stay within bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_view::StridedViewMut;
    ///
    /// let mut data = [0.0f64; 4];
    /// // SAFETY: the largest reachable offset, 1 * 1 + 1 * 2 = 3, is below
    /// // `data.len() == 4`.
    /// let view = unsafe { StridedViewMut::<f64>::new_unchecked(&mut data, &[2, 2], &[1, 2], 0) };
    /// assert_eq!(view.ndim(), 2);
    /// ```
    #[inline]
    pub unsafe fn new_unchecked(
        data: &'a mut [T],
        dims: &[usize],
        strides: &[isize],
        offset: isize,
    ) -> Self {
        Self::from_meta(data, dims, strides, offset)
    }

    /// Returns the shape (dimension sizes) of this view.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        self.meta.dims()
    }

    /// Returns the strides (in units of `T`) for each dimension.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.meta.strides()
    }

    /// Returns the offset into the backing data, in units of `T` (elements, not bytes).
    #[inline]
    pub fn offset(&self) -> isize {
        self.offset
    }

    /// Returns the number of dimensions (rank).
    #[inline]
    pub fn ndim(&self) -> usize {
        self.dims().len()
    }

    /// Returns the total number of elements.
    ///
    /// # Panics
    ///
    /// Panics when the element count does not fit `usize`, which only a
    /// broadcast (stride-0) layout can reach.
    #[inline]
    pub fn len(&self) -> usize {
        element_count(self.dims())
    }

    /// Returns `true` if any dimension is zero.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dims().contains(&0)
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

    /// Returns a reference to the backing data slice.
    ///
    /// The length is the backing slice's, not the shape's: a view whose shape
    /// has a zero extent still reports the whole buffer.
    #[inline]
    pub fn data(&self) -> &[T] {
        // SAFETY: `base` is the caller's `data.as_mut_ptr()` and `src_len` is
        // that slice's length, both recorded at construction, so this rebuilds
        // exactly the borrowed slice. No length is derived from the shape, which
        // is why a zero-extent layout still reports the whole buffer. `T` is
        // initialized across it, and `&self` restricts this to a shared borrow.
        unsafe { std::slice::from_raw_parts(self.base, self.src_len) }
    }

    /// Returns a mutable reference to the backing data slice.
    ///
    /// The slice is re-derived from the view's storage handle, so `ptr`, `get`,
    /// and `set` stay valid after the returned borrow ends.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        // SAFETY: as `data` above — the same `base`/`src_len` pair, so this
        // rebuilds the caller's slice rather than a shape-sized one. `&mut self`
        // keeps the returned slice the only reference into the view while it
        // lives. In a non-empty layout `ptr` is `base + offset`, derived without
        // a retag, so re-deriving the slice here does not invalidate it — the
        // property `get`/`set` rely on; an empty layout caches a dangling `ptr`
        // that no element access dereferences.
        unsafe { std::slice::from_raw_parts_mut(self.base, self.src_len) }
    }

    /// Permute dimensions, consuming the mutable view.
    ///
    /// Returns a new mutable view with reordered dimensions and strides.
    /// Takes `self` by value to prevent aliasing of mutable views.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::RankMismatch`] when `perm` does not have one
    /// entry per axis, and [`StridedError::InvalidAxis`] when an entry
    /// names an axis outside the rank or repeats an axis.
    pub fn permute(self, perm: &[usize]) -> Result<StridedViewMut<'a, T>> {
        let rank = self.dims().len();
        if perm.len() != rank {
            return Err(StridedError::RankMismatch(perm.len(), rank));
        }
        let mut seen: tinyvec::TinyVec<[bool; INLINE_RANK]> = tinyvec::TinyVec::new();
        seen.resize(rank, false);
        for &p in perm {
            if p >= rank {
                return Err(StridedError::InvalidAxis { axis: p, rank });
            }
            if seen[p] {
                return Err(StridedError::InvalidAxis { axis: p, rank });
            }
            seen[p] = true;
        }
        let new_dims: AxisVec = perm.iter().map(|&p| self.dims()[p]).collect();
        let new_strides: StrideVec = perm.iter().map(|&p| self.strides()[p]).collect();
        Ok(Self {
            meta: ViewMeta::from_slices(&new_dims, &new_strides),
            ..self
        })
    }

    /// Reborrow as an immutable view.
    ///
    /// The reborrow copies this view's metadata into the immutable view's
    /// inline storage, which allocates nothing up to [`INLINE_RANK`] and
    /// spills above it (the mutable owner always keeps its own copy).
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_view::StridedViewMut;
    ///
    /// let mut data = [1.0f64, 2.0, 3.0];
    /// let view = StridedViewMut::<f64>::new(&mut data, &[3], &[1], 0).unwrap();
    /// let read = view.as_view();
    /// assert_eq!(read.get(&[2]), 3.0);
    /// ```
    #[inline]
    pub fn as_view(&self) -> StridedView<'_, T, Identity> {
        // SAFETY: `data()` re-derives the slice from this view's own handle
        // (`base` + `src_len`), and `ptr` is `base + offset` for the same
        // `offset`, so the immutable view ends up pointing at the same storage
        // the mutable view owns. A zero-extent layout yields a dangling pointer
        // on both sides (`empty_mut_ptr` here, the `empty_layout` branch of
        // `base_ptr` in the immutable view), so they agree there too. `&self`
        // keeps that storage exclusively borrowed for the returned view's
        // lifetime, so the shared view can only read it, and `T` is initialized
        // across the whole slice.
        unsafe { StridedView::new_unchecked(self.data(), self.dims(), self.strides(), self.offset) }
    }
}

impl<'a, T: Copy> StridedViewMut<'a, T> {
    /// Get an element.
    ///
    /// # Panics
    ///
    /// Panics if `indices.len()` is not the view's rank, an index is out of
    /// bounds, or the indexed offset does not fit `isize`.
    pub fn get(&self, indices: &[usize]) -> T {
        let offset = indexed_offset(self.dims(), self.strides(), indices, 0);
        // SAFETY: in a non-empty layout `ptr` is `base + offset`, derived from
        // the caller's `&'a mut [T]` through raw pointers only, so a reborrow of
        // the backing slice does not invalidate it: `data_mut` re-derives that
        // slice from `base` and never rewrites `ptr`, and offsetting a raw
        // pointer keeps `base`'s provenance. (An empty layout caches a dangling
        // `ptr`, but no index satisfies `index < dims[i]` there, so this access
        // is unreachable.) `indexed_offset` (view.rs:240) asserts the index
        // count and `index < dims[i]`, so the relative offset it returns keeps
        // `ptr + offset` inside the extent `validate_bounds` (view.rs:187)
        // proved reachable, i.e. inside `base..base + src_len`, where every
        // element is initialized because the view borrows an initialized
        // `&'a mut [T]` and no method shortens that borrow.
        unsafe { *self.ptr.offset(offset) }
    }

    /// Set an element.
    ///
    /// # Panics
    ///
    /// Panics if `indices.len()` is not the view's rank, an index is out of
    /// bounds, or the indexed offset does not fit `isize`.
    pub fn set(&mut self, indices: &[usize], value: T) {
        let offset = indexed_offset(self.dims(), self.strides(), indices, 0);
        // SAFETY: as `get` above — the same `ptr` survives an earlier `data_mut`
        // reborrow, and `indexed_offset` (view.rs:240) bounds-checks every index,
        // so the store lands inside `base..base + src_len`. `&mut self` makes
        // this the only access through the view for the call, and the view holds
        // the sole `&'a mut [T]` borrow of that storage, so nothing else can
        // alias the written place.
        unsafe {
            *self.ptr.offset(offset) = value;
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
    meta: ViewMeta,
    offset: isize,
}

impl<T: std::fmt::Debug> std::fmt::Debug for StridedArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StridedArray")
            .field("dims", &self.meta.dims())
            .field("strides", &self.meta.strides())
            .field("offset", &self.offset)
            .finish()
    }
}

impl<T: Clone> Clone for StridedArray<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            meta: self.meta.clone(),
            offset: self.offset,
        }
    }
}

impl<T: Clone + Default> StridedArray<T> {
    /// Create a column-major (Julia default) tensor filled with Default values.
    ///
    /// The shape and strides are stored inline, so a rank `<= INLINE_RANK`
    /// tensor allocates only its element buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_view::StridedArray;
    ///
    /// let a = StridedArray::<f64>::col_major(&[2, 3]);
    /// assert_eq!(a.dims(), &[2, 3]);
    /// assert_eq!(a.strides(), &[1, 2]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when the element count or a stride does not fit `isize`; use
    /// [`Self::try_col_major`] to get [`StridedError::OffsetOverflow`] instead.
    pub fn col_major(dims: &[usize]) -> Self {
        Self::try_col_major(dims).expect("col_major: shape overflows isize")
    }

    /// Fallible [`Self::col_major`].
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::OffsetOverflow`] when the element count or a
    /// stride does not fit `isize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_view::{StridedArray, StridedError};
    ///
    /// assert_eq!(StridedArray::<f64>::try_col_major(&[2, 3]).unwrap().strides(), &[1, 2]);
    /// assert!(matches!(
    ///     StridedArray::<f64>::try_col_major(&[1 << 62, 4]),
    ///     Err(StridedError::OffsetOverflow)
    /// ));
    /// ```
    pub fn try_col_major(dims: &[usize]) -> Result<Self> {
        let strides = checked_col_major_strides(dims)?;
        let total = checked_element_count(dims)?;
        Ok(Self {
            data: vec![T::default(); total],
            meta: ViewMeta::from_slices(dims, &strides),
            offset: 0,
        })
    }

    /// Create a row-major (C default) tensor filled with Default values.
    ///
    /// The shape and strides are stored inline, so a rank `<= INLINE_RANK`
    /// tensor allocates only its element buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use strided_view::StridedArray;
    ///
    /// let a = StridedArray::<f64>::row_major(&[2, 3]);
    /// assert_eq!(a.dims(), &[2, 3]);
    /// assert_eq!(a.strides(), &[3, 1]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when the element count or a stride does not fit `isize`; use
    /// [`Self::try_row_major`] to get [`StridedError::OffsetOverflow`] instead.
    pub fn row_major(dims: &[usize]) -> Self {
        Self::try_row_major(dims).expect("row_major: shape overflows isize")
    }

    /// Fallible [`Self::row_major`].
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::OffsetOverflow`] when the element count or a
    /// stride does not fit `isize`.
    pub fn try_row_major(dims: &[usize]) -> Result<Self> {
        let strides = checked_row_major_strides(dims)?;
        let total = checked_element_count(dims)?;
        Ok(Self {
            data: vec![T::default(); total],
            meta: ViewMeta::from_slices(dims, &strides),
            offset: 0,
        })
    }

    /// Create a column-major tensor with values produced by a function.
    ///
    /// The function is called with indices in column-major iteration order.
    ///
    /// # Panics
    ///
    /// Panics when the element count or a stride does not fit `isize`; use
    /// [`Self::try_from_fn_col_major`] to get [`StridedError::OffsetOverflow`] instead.
    pub fn from_fn_col_major(dims: &[usize], f: impl FnMut(&[usize]) -> T) -> Self {
        Self::try_from_fn_col_major(dims, f).expect("from_fn_col_major: shape overflows isize")
    }

    /// Fallible [`Self::from_fn_col_major`]; `f` is not called on error.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::OffsetOverflow`] when the element count or a
    /// stride does not fit `isize`.
    pub fn try_from_fn_col_major(dims: &[usize], mut f: impl FnMut(&[usize]) -> T) -> Result<Self> {
        let strides = checked_col_major_strides(dims)?;
        let total = checked_element_count(dims)?;
        let rank = dims.len();
        let mut data = Vec::with_capacity(total);
        let mut idx: AxisVec = AxisVec::new();
        idx.resize(rank, 0usize);
        // Borrow the cursor once; indexing the `TinyVec` per access would pay
        // its inline/heap match in the element-count loop.
        let idx: &mut [usize] = &mut idx[..];
        for _ in 0..total {
            data.push(f(idx));
            for d in 0..rank {
                idx[d] += 1;
                if idx[d] < dims[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok(Self {
            data,
            meta: ViewMeta::from_slices(dims, &strides),
            offset: 0,
        })
    }

    /// Create a row-major tensor with values produced by a function.
    ///
    /// The function is called with indices in row-major iteration order.
    ///
    /// # Panics
    ///
    /// Panics when the element count or a stride does not fit `isize`; use
    /// [`Self::try_from_fn_row_major`] to get [`StridedError::OffsetOverflow`] instead.
    pub fn from_fn_row_major(dims: &[usize], f: impl FnMut(&[usize]) -> T) -> Self {
        Self::try_from_fn_row_major(dims, f).expect("from_fn_row_major: shape overflows isize")
    }

    /// Fallible [`Self::from_fn_row_major`]; `f` is not called on error.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::OffsetOverflow`] when the element count or a
    /// stride does not fit `isize`.
    pub fn try_from_fn_row_major(dims: &[usize], mut f: impl FnMut(&[usize]) -> T) -> Result<Self> {
        let strides = checked_row_major_strides(dims)?;
        let total = checked_element_count(dims)?;
        let rank = dims.len();
        let mut data = Vec::with_capacity(total);
        let mut idx: AxisVec = AxisVec::new();
        idx.resize(rank, 0usize);
        // Borrow the cursor once; see `from_fn_col_major`.
        let idx: &mut [usize] = &mut idx[..];
        for _ in 0..total {
            data.push(f(idx));
            for d in (0..rank).rev() {
                idx[d] += 1;
                if idx[d] < dims[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok(Self {
            data,
            meta: ViewMeta::from_slices(dims, &strides),
            offset: 0,
        })
    }
}

impl<T> StridedArray<T> {
    /// Create from raw parts.
    ///
    /// `dims`, `strides`, and `offset` are in units of `T` (elements, not
    /// bytes).
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
    pub fn from_parts(
        data: Vec<T>,
        dims: &[usize],
        strides: &[isize],
        offset: isize,
    ) -> Result<Self> {
        validate_bounds(data.len(), dims, strides, offset)?;
        Ok(Self {
            data,
            meta: ViewMeta::from_slices(dims, strides),
            offset,
        })
    }

    /// Returns the shape (dimension sizes) of this array.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        self.meta.dims()
    }

    /// Returns the strides (in units of `T`) for each dimension.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.meta.strides()
    }

    /// Returns the number of dimensions (rank).
    #[inline]
    pub fn ndim(&self) -> usize {
        self.dims().len()
    }

    /// Returns the total number of elements.
    ///
    /// # Panics
    ///
    /// Panics when the element count does not fit `usize`, which only a
    /// broadcast (stride-0) layout can reach.
    #[inline]
    pub fn len(&self) -> usize {
        element_count(self.dims())
    }

    /// Returns `true` if any dimension is zero.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dims().contains(&0)
    }

    /// Returns a reference to the backing data slice.
    #[inline]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable reference to the backing data slice.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Create an immutable view over this tensor.
    #[inline]
    pub fn view(&self) -> StridedView<'_, T> {
        let ptr = if self.is_empty() {
            empty_const_ptr()
        } else {
            // SAFETY: `offset` is either 0 from a constructor or was accepted by
            // `validate_bounds` in `StridedArray::from_parts` (view.rs:1398), so
            // `data.as_ptr() + offset` stays inside `data`. The empty branch
            // above excludes a zero-sized layout, whose offset need not be
            // reachable. The view built below also needs the whole reachable
            // extent inside `data` (its `get` relies on it): `from_parts` proved
            // that with `validate_bounds`, and every dense constructor
            // (`try_col_major`, view.rs:1247, and its row-major, `from_fn`,
            // `from_buffer` and uninit siblings) stores exactly
            // `checked_element_count(dims)` (view.rs:266) elements under
            // `checked_dense_strides` (view.rs:292), whose highest offset is that
            // count minus one. `permuted` only reorders axes, and `data_mut`
            // hands out a slice, which cannot change the length.
            unsafe { self.data.as_ptr().offset(self.offset) }
        };
        StridedView {
            ptr,
            data: &self.data,
            meta: self.meta.clone(),
            offset: self.offset,
            _op: PhantomData,
        }
    }

    /// Create a mutable view over this tensor.
    #[inline]
    pub fn view_mut(&mut self) -> StridedViewMut<'_, T> {
        let src_len = self.data.len();
        let base = self.data.as_mut_ptr();
        let ptr = if self.is_empty() {
            empty_mut_ptr()
        } else {
            // SAFETY: as `view` above — `self.offset` is either 0 from a
            // constructor or was accepted by `validate_bounds` in
            // `StridedArray::from_parts` (view.rs:1398), and `&mut self` keeps
            // this reborrow exclusive. The empty branch above excludes a
            // zero-sized layout, whose offset need not be reachable. The whole
            // reachable extent stays inside `data` for the same reasons given in
            // `view`, which `StridedViewMut::get`/`set` rely on.
            unsafe { base.offset(self.offset) }
        };
        StridedViewMut {
            ptr,
            base,
            src_len,
            meta: self.meta.clone(),
            offset: self.offset,
            _marker: PhantomData,
        }
    }

    /// Permute dimensions (metadata-only reorder, no data copy).
    ///
    /// Returns a new array with reordered dims and strides.
    /// The underlying data buffer is not touched.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::RankMismatch`] when `perm` does not have one
    /// entry per axis, and [`StridedError::InvalidAxis`] when an entry
    /// names an axis outside the rank or repeats an axis.
    pub fn permuted(self, perm: &[usize]) -> Result<Self> {
        let rank = self.dims().len();
        if perm.len() != rank {
            return Err(StridedError::RankMismatch(perm.len(), rank));
        }
        let mut seen: tinyvec::TinyVec<[bool; INLINE_RANK]> = tinyvec::TinyVec::new();
        seen.resize(rank, false);
        for &p in perm {
            if p >= rank {
                return Err(StridedError::InvalidAxis { axis: p, rank });
            }
            if seen[p] {
                return Err(StridedError::InvalidAxis { axis: p, rank });
            }
            seen[p] = true;
        }
        let new_dims: AxisVec = perm.iter().map(|&p| self.dims()[p]).collect();
        let new_strides: StrideVec = perm.iter().map(|&p| self.strides()[p]).collect();
        Ok(Self {
            data: self.data,
            meta: ViewMeta::from_slices(&new_dims, &new_strides),
            offset: self.offset,
        })
    }

    /// Consume the array and return the backing `Vec<T>`.
    pub fn into_data(self) -> Vec<T> {
        self.data
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

impl<T: Default> StridedArray<T> {
    /// Create a column-major tensor reusing an existing buffer.
    ///
    /// If `buf` has at least `product(dims)` elements, it is truncated and
    /// zeroed.  Otherwise a fresh buffer is allocated.
    ///
    /// # Panics
    ///
    /// Panics when the element count or a stride does not fit `isize`; use
    /// [`Self::try_col_major_from_buffer`] to get
    /// [`StridedError::OffsetOverflow`] instead.
    pub fn col_major_from_buffer(buf: Vec<T>, dims: &[usize]) -> Self {
        Self::try_col_major_from_buffer(buf, dims)
            .expect("col_major_from_buffer: shape overflows isize")
    }

    /// Fallible [`Self::col_major_from_buffer`]; `buf` is dropped on error.
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::OffsetOverflow`] when the element count or a
    /// stride does not fit `isize`.
    pub fn try_col_major_from_buffer(mut buf: Vec<T>, dims: &[usize]) -> Result<Self> {
        let strides = checked_col_major_strides(dims)?;
        let total = checked_element_count(dims)?;
        if buf.len() >= total {
            buf.truncate(total);
        } else {
            buf.resize_with(total, T::default);
        }
        // Zero the reused region
        for v in buf.iter_mut() {
            *v = T::default();
        }
        Ok(Self {
            data: buf,
            meta: ViewMeta::from_slices(dims, &strides),
            offset: 0,
        })
    }
}

impl<T: Copy> StridedArray<T> {
    /// Create a column-major tensor with **uninitialized** data.
    ///
    /// The element type of the returned array is [`MaybeUninit<T>`], so no slot
    /// ever claims to hold a `T` before the caller wrote one — which is what
    /// `bool`, `char` and reference elements need (#413).
    ///
    /// # Safety
    ///
    /// This is the pre-#413 contract kept on the signature: a slot may be read
    /// through [`MaybeUninit::assume_init`] only after it was written. Allocating
    /// the slots carries no memory-safety precondition of its own; the per-slot
    /// rule is the one [`MaybeUninit`] already states.
    ///
    /// # Panics
    ///
    /// Panics when the element count or a stride does not fit `isize`; use
    /// [`Self::try_col_major_uninit`] to get [`StridedError::OffsetOverflow`]
    /// instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::mem::MaybeUninit;
    /// use strided_view::StridedArray;
    ///
    /// // SAFETY: each slot is written below before it is read.
    /// let mut a = unsafe { StridedArray::<bool>::col_major_uninit(&[2]) };
    /// a.set(&[0], MaybeUninit::new(true));
    /// a.set(&[1], MaybeUninit::new(false));
    /// // SAFETY: both slots were written above.
    /// assert!(unsafe { a.get(&[0]).assume_init() });
    /// ```
    pub unsafe fn col_major_uninit(dims: &[usize]) -> StridedArray<MaybeUninit<T>> {
        // SAFETY: `try_col_major_uninit` has the same contract as this function
        // (view.rs:1618), which our caller accepted: read a slot only after
        // writing it. This call reads nothing.
        Self::try_col_major_uninit(dims).expect("col_major_uninit: shape overflows isize")
    }

    /// Fallible [`Self::col_major_uninit`].
    ///
    /// # Safety
    ///
    /// As [`Self::col_major_uninit`].
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::OffsetOverflow`] when the element count or a
    /// stride does not fit `isize`.
    pub unsafe fn try_col_major_uninit(dims: &[usize]) -> Result<StridedArray<MaybeUninit<T>>> {
        let strides = checked_col_major_strides(dims)?;
        let total = checked_element_count(dims)?;
        Ok(StridedArray {
            data: vec![MaybeUninit::uninit(); total],
            meta: ViewMeta::from_slices(dims, &strides),
            offset: 0,
        })
    }

    /// Reuse an existing buffer as a column-major tensor **without zeroing**.
    ///
    /// The buffer is `Vec<MaybeUninit<T>>` for the same reason as
    /// [`Self::col_major_uninit`]: a slot the caller has not written yet is not a
    /// `T` (#413). Slots the buffer already holds are kept as they are.
    ///
    /// # Safety
    ///
    /// As [`Self::col_major_uninit`]: read a slot with
    /// [`MaybeUninit::assume_init`] only after it was written.
    ///
    /// # Panics
    ///
    /// Panics when the element count or a stride does not fit `isize`; use
    /// [`Self::try_col_major_from_buffer_uninit`] to get
    /// [`StridedError::OffsetOverflow`] instead.
    pub unsafe fn col_major_from_buffer_uninit(
        buf: Vec<MaybeUninit<T>>,
        dims: &[usize],
    ) -> StridedArray<MaybeUninit<T>> {
        // SAFETY: `try_col_major_from_buffer_uninit` has the same contract as
        // this function (view.rs:1618), which our caller accepted: read a slot
        // only after writing it. This call reads nothing.
        Self::try_col_major_from_buffer_uninit(buf, dims)
            .expect("col_major_from_buffer_uninit: shape overflows isize")
    }

    /// Fallible [`Self::col_major_from_buffer_uninit`]; `buf` is dropped on
    /// error.
    ///
    /// # Safety
    ///
    /// As [`Self::col_major_uninit`].
    ///
    /// # Errors
    ///
    /// Returns [`StridedError::OffsetOverflow`] when the element count or a
    /// stride does not fit `isize`.
    pub unsafe fn try_col_major_from_buffer_uninit(
        mut buf: Vec<MaybeUninit<T>>,
        dims: &[usize],
    ) -> Result<StridedArray<MaybeUninit<T>>> {
        let strides = checked_col_major_strides(dims)?;
        let total = checked_element_count(dims)?;
        // Truncates or grows to the shape; the retained prefix keeps whatever the
        // caller already put in it.
        buf.resize(total, MaybeUninit::uninit());
        Ok(StridedArray {
            data: buf,
            meta: ViewMeta::from_slices(dims, &strides),
            offset: 0,
        })
    }
}

impl<T: Copy> StridedArray<T> {
    /// Get an element by multi-dimensional index.
    ///
    /// # Panics
    ///
    /// Panics if `indices.len()` is not the array's rank, an index is out of
    /// bounds, or the indexed offset does not fit `isize`.
    pub fn get(&self, indices: &[usize]) -> T {
        self.view().get(indices)
    }

    /// Set an element by multi-dimensional index.
    ///
    /// # Panics
    ///
    /// Panics if `indices.len()` is not the array's rank, an index is out of
    /// bounds, or the indexed offset does not fit `isize`.
    pub fn set(&mut self, indices: &[usize], value: T) {
        let offset = indexed_offset(self.dims(), self.strides(), indices, self.offset);
        self.data[offset as usize] = value;
    }
}

impl<T: Copy> Index<&[usize]> for StridedArray<T> {
    type Output = T;

    fn index(&self, indices: &[usize]) -> &T {
        let offset = indexed_offset(self.dims(), self.strides(), indices, self.offset);
        &self.data[offset as usize]
    }
}

impl<T: Copy> IndexMut<&[usize]> for StridedArray<T> {
    fn index_mut(&mut self, indices: &[usize]) -> &mut T {
        let offset = indexed_offset(self.dims(), self.strides(), indices, self.offset);
        &mut self.data[offset as usize]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
