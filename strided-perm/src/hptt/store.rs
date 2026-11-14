//! The element store the engine writes through: plain copy or conjugation.
//!
//! HPTT carries conjugation as the `conjA` template parameter of its complex
//! entry points (`zTensorTranspose`, `cTensorTranspose`) and of its micro-kernel
//! (<https://github.com/springer13/hptt>). Here it is a monomorphized marker
//! type instead of a runtime flag, so one plan serves both stores and no store
//! loop pays a branch.
//!
//! Only the generic element store conjugates. The native `f64`/`f32`
//! micro-kernels copy bits and are therefore reachable only from a plain store
//! (`Store::PLAIN`).

use strided_view::ElementOpApply;

/// How the engine writes a destination element.
pub(crate) trait Store<T: Copy> {
    /// A plain bit copy: permits the `memcpy` fast path and the native float
    /// micro-kernels. A conjugating store must leave both alone.
    const PLAIN: bool;

    /// Write `*src` to `*dst` under this policy.
    ///
    /// # Safety
    ///
    /// `dst` must be writable for one `T`, `src` readable for one initialized
    /// `T`, and the two must not overlap.
    unsafe fn store(dst: *mut T, src: *const T);
}

/// `dest[i] = src[i]`.
pub(crate) struct PlainStore;

impl<T: Copy> Store<T> for PlainStore {
    const PLAIN: bool = true;

    #[inline(always)]
    unsafe fn store(dst: *mut T, src: *const T) {
        // SAFETY: the caller guarantees both accesses.
        unsafe { *dst = *src };
    }
}

/// `dest[i] = conj(src[i])`.
pub(crate) struct ConjStore;

impl<T: ElementOpApply> Store<T> for ConjStore {
    const PLAIN: bool = false;

    #[inline(always)]
    unsafe fn store(dst: *mut T, src: *const T) {
        // SAFETY: the caller guarantees both accesses.
        unsafe { *dst = (*src).conj() };
    }
}
