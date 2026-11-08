//! Generic scalar micro-kernel implementations.
//!
//! These use simple nested loops that LLVM auto-vectorizes effectively.
//! The 4×4 f64 loop compiles to 16 load-store pairs with known offsets,
//! matching HPTT C++'s scalar kernel performance.

use super::{MicroKernel, ScalarKernel};

impl MicroKernel<f64> for ScalarKernel {
    const MICRO: usize = 4;
    const BLOCK: usize = 16; // 4 * 4

    #[inline(always)]
    unsafe fn transpose_micro(src: *const f64, lda: isize, dst: *mut f64, ldb: isize) {
        for j in 0..4_isize {
            for i in 0..4_isize {
                *dst.offset(i + j * ldb) = *src.offset(i * lda + j);
            }
        }
    }
}

impl MicroKernel<f32> for ScalarKernel {
    const MICRO: usize = 8;
    const BLOCK: usize = 32; // 8 * 4

    #[inline(always)]
    unsafe fn transpose_micro(src: *const f32, lda: isize, dst: *mut f32, ldb: isize) {
        for j in 0..8_isize {
            for i in 0..8_isize {
                *dst.offset(i + j * ldb) = *src.offset(i * lda + j);
            }
        }
    }
}

#[cfg(test)]
mod tests;
