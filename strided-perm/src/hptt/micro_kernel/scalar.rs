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
mod tests {
    use super::*;

    #[test]
    fn test_scalar_f64_4x4() {
        // Source: 4×4 matrix in col-major (lda=4)
        // A = [[0,4,8,12],[1,5,9,13],[2,6,10,14],[3,7,11,15]]
        let src: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let mut dst = vec![0.0f64; 16];

        unsafe {
            ScalarKernel::transpose_micro(src.as_ptr(), 4, dst.as_mut_ptr(), 4);
        }

        // Expected: B[i + j*4] = A[i*4 + j]
        // B[0] = A[0] = 0, B[1] = A[4] = 4, B[2] = A[8] = 8, B[3] = A[12] = 12
        // B[4] = A[1] = 1, B[5] = A[5] = 5, ...
        for j in 0..4 {
            for i in 0..4 {
                assert_eq!(dst[i + j * 4], src[i * 4 + j], "mismatch at i={i}, j={j}");
            }
        }
    }

    #[test]
    fn test_scalar_f64_non_square_strides() {
        // src with lda=5 (5 elements per row), dst with ldb=6
        let mut src = vec![0.0f64; 20];
        for i in 0..4 {
            for j in 0..4 {
                src[i * 5 + j] = (i * 10 + j) as f64;
            }
        }
        let mut dst = vec![0.0f64; 24];

        unsafe {
            ScalarKernel::transpose_micro(src.as_ptr(), 5, dst.as_mut_ptr(), 6);
        }

        for j in 0..4 {
            for i in 0..4 {
                assert_eq!(dst[i + j * 6], src[i * 5 + j], "mismatch at i={i}, j={j}");
            }
        }
    }

    #[test]
    fn test_scalar_f32_8x8() {
        let src: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 64];

        unsafe {
            ScalarKernel::transpose_micro(src.as_ptr(), 8, dst.as_mut_ptr(), 8);
        }

        for j in 0..8 {
            for i in 0..8 {
                assert_eq!(dst[i + j * 8], src[i * 8 + j], "mismatch at i={i}, j={j}");
            }
        }
    }
}
