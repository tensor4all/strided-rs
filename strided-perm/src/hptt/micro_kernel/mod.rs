//! Architecture-specific micro-kernel trait and dispatch.
//!
//! The micro-kernel is the innermost building block: an N×N in-register
//! transpose where N = REGISTER_BITS / 8 / sizeof(T).

pub mod scalar;

/// Architecture-specific N×N transpose micro-kernel.
///
/// A micro-kernel transposes a MICRO × MICRO tile:
///   `dst[i + j*ldb] = src[i*lda + j]` for i,j in 0..MICRO
///
/// BLOCK = MICRO * 4 defines the macro-kernel tile size.
pub trait MicroKernel<T: Copy> {
    /// Micro-tile side length.
    /// e.g. 4 for f64 (scalar/AVX2), 8 for f32 (scalar/AVX2).
    const MICRO: usize;

    /// Macro-tile side length = MICRO * 4.
    const BLOCK: usize;

    /// Transpose a full MICRO × MICRO tile.
    ///
    /// # Safety
    /// - `src` must be readable for MICRO elements along stride-1 and MICRO rows of stride `lda`
    /// - `dst` must be writable for MICRO elements along stride-1 and MICRO rows of stride `ldb`
    unsafe fn transpose_micro(src: *const T, lda: isize, dst: *mut T, ldb: isize);
}

/// Marker type for scalar (non-SIMD) micro-kernels.
pub struct ScalarKernel;
