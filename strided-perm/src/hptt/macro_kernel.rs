//! Macro-kernel: processes a BLOCK × BLOCK tile using a grid of micro-kernels.
//!
//! Mirrors HPTT C++ `macro_kernel` (transpose.cpp lines 396-560).
//! Each macro-kernel call handles a tile of up to BLOCK × BLOCK elements,
//! invoking the micro-kernel for full MICRO × MICRO sub-tiles and scalar
//! loops for edge remainders.

use crate::hptt::micro_kernel::{MicroKernel, ScalarKernel};

/// Process a tile of `block_a × block_b` elements using f64 micro-kernels.
///
/// - `src` points to A[0,0] of the tile. A's stride-1 dimension is along dim_A.
/// - `lda` is A's stride along dim_B (the non-stride-1 dim in source).
/// - `dst` points to B[0,0] of the tile. B's stride-1 dimension is along dim_B.
/// - `ldb` is B's stride along dim_A (the non-stride-1 dim in dest).
///
/// The transpose operation: `B[j + i*ldb] = A[i + j*lda]`
/// where i iterates along dim_A (0..block_a) and j along dim_B (0..block_b).
///
/// # Safety
/// src/dst must be valid for the given block sizes and strides.
#[inline]
pub unsafe fn macro_kernel_f64(
    src: *const f64,
    lda: isize,
    block_a: usize,
    dst: *mut f64,
    ldb: isize,
    block_b: usize,
) {
    const MICRO: usize = <ScalarKernel as MicroKernel<f64>>::MICRO; // 4

    let full_a = block_a / MICRO;
    let rem_a = block_a % MICRO;
    let full_b = block_b / MICRO;
    let rem_b = block_b % MICRO;

    // Full MICRO × MICRO tiles
    for jb in 0..full_b {
        let j = (jb * MICRO) as isize;
        for ia in 0..full_a {
            let i = (ia * MICRO) as isize;
            ScalarKernel::transpose_micro(
                src.offset(i + j * lda),
                lda,
                dst.offset(j + i * ldb),
                ldb,
            );
        }
        // Remainder along dim_A (right edge)
        if rem_a > 0 {
            let i = (full_a * MICRO) as isize;
            for jj in 0..MICRO as isize {
                for ii in 0..rem_a as isize {
                    *dst.offset((j + jj) + (i + ii) * ldb) =
                        *src.offset((i + ii) + (j + jj) * lda);
                }
            }
        }
    }

    // Remainder along dim_B (bottom edge)
    if rem_b > 0 {
        let j = (full_b * MICRO) as isize;
        for ia in 0..full_a {
            let i = (ia * MICRO) as isize;
            for jj in 0..rem_b as isize {
                for ii in 0..MICRO as isize {
                    *dst.offset((j + jj) + (i + ii) * ldb) =
                        *src.offset((i + ii) + (j + jj) * lda);
                }
            }
        }
        // Corner remainder (both rem_a and rem_b)
        if rem_a > 0 {
            let i = (full_a * MICRO) as isize;
            for jj in 0..rem_b as isize {
                for ii in 0..rem_a as isize {
                    *dst.offset((j + jj) + (i + ii) * ldb) =
                        *src.offset((i + ii) + (j + jj) * lda);
                }
            }
        }
    }
}

/// Process a tile of `block_a × block_b` elements using f32 micro-kernels.
#[inline]
pub unsafe fn macro_kernel_f32(
    src: *const f32,
    lda: isize,
    block_a: usize,
    dst: *mut f32,
    ldb: isize,
    block_b: usize,
) {
    const MICRO: usize = <ScalarKernel as MicroKernel<f32>>::MICRO; // 8

    let full_a = block_a / MICRO;
    let rem_a = block_a % MICRO;
    let full_b = block_b / MICRO;
    let rem_b = block_b % MICRO;

    for jb in 0..full_b {
        let j = (jb * MICRO) as isize;
        for ia in 0..full_a {
            let i = (ia * MICRO) as isize;
            ScalarKernel::transpose_micro(
                src.offset(i + j * lda),
                lda,
                dst.offset(j + i * ldb),
                ldb,
            );
        }
        if rem_a > 0 {
            let i = (full_a * MICRO) as isize;
            for jj in 0..MICRO as isize {
                for ii in 0..rem_a as isize {
                    *dst.offset((j + jj) + (i + ii) * ldb) =
                        *src.offset((i + ii) + (j + jj) * lda);
                }
            }
        }
    }

    if rem_b > 0 {
        let j = (full_b * MICRO) as isize;
        for ia in 0..full_a {
            let i = (ia * MICRO) as isize;
            for jj in 0..rem_b as isize {
                for ii in 0..MICRO as isize {
                    *dst.offset((j + jj) + (i + ii) * ldb) =
                        *src.offset((i + ii) + (j + jj) * lda);
                }
            }
        }
        if rem_a > 0 {
            let i = (full_a * MICRO) as isize;
            for jj in 0..rem_b as isize {
                for ii in 0..rem_a as isize {
                    *dst.offset((j + jj) + (i + ii) * ldb) =
                        *src.offset((i + ii) + (j + jj) * lda);
                }
            }
        }
    }
}

/// ConstStride1 inner copy: simple memcpy or strided element-wise copy.
///
/// Used when dim_A == dim_B (both arrays have stride 1 along the same dim).
#[inline(always)]
pub unsafe fn const_stride1_copy<T: Copy>(
    src: *const T,
    dst: *mut T,
    count: usize,
    src_stride: isize,
    dst_stride: isize,
) {
    if src_stride == 1 && dst_stride == 1 {
        std::ptr::copy_nonoverlapping(src, dst, count);
    } else if dst_stride == 1 {
        let mut s = src;
        let mut d = dst;
        for _ in 0..count {
            *d = *s;
            s = s.offset(src_stride);
            d = d.add(1);
        }
    } else if src_stride == 1 {
        let mut s = src;
        let mut d = dst;
        for _ in 0..count {
            *d = *s;
            s = s.add(1);
            d = d.offset(dst_stride);
        }
    } else {
        let mut s = src;
        let mut d = dst;
        for _ in 0..count {
            *d = *s;
            s = s.offset(src_stride);
            d = d.offset(dst_stride);
        }
    }
}

/// Fallback element-by-element 2D copy for unsupported element sizes.
///
/// Performs the same transpose as macro_kernel but without micro-kernel
/// optimization. Used for types other than f64/f32.
#[inline]
pub unsafe fn macro_kernel_fallback<T: Copy>(
    src: *const T,
    lda: isize,
    block_a: usize,
    dst: *mut T,
    ldb: isize,
    block_b: usize,
) {
    for j in 0..block_b as isize {
        for i in 0..block_a as isize {
            *dst.offset(j + i * ldb) = *src.offset(i + j * lda);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro_kernel_f64_full_block() {
        // 16×16 tile, lda=16, ldb=16 (both square)
        let n = 16;
        let src: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
        let mut dst = vec![0.0f64; n * n];

        unsafe {
            macro_kernel_f64(src.as_ptr(), n as isize, n, dst.as_mut_ptr(), n as isize, n);
        }

        for j in 0..n {
            for i in 0..n {
                assert_eq!(
                    dst[j + i * n],
                    src[i + j * n],
                    "mismatch at i={i}, j={j}"
                );
            }
        }
    }

    #[test]
    fn test_macro_kernel_f64_with_remainder() {
        // 15×17 tile (both have remainders w.r.t. MICRO=4)
        let block_a = 15;
        let block_b = 17;
        let lda = 20isize; // src leading dim
        let ldb = 18isize; // dst leading dim

        let src: Vec<f64> = (0..(block_a as isize * 1 + (block_b - 1) as isize * lda + 1) as usize)
            .map(|i| i as f64)
            .collect();
        let mut dst =
            vec![0.0f64; ((block_b - 1) as isize * 1 + (block_a - 1) as isize * ldb + 1) as usize];

        unsafe {
            macro_kernel_f64(src.as_ptr(), lda, block_a, dst.as_mut_ptr(), ldb, block_b);
        }

        for j in 0..block_b {
            for i in 0..block_a {
                let s = src[(i as isize + j as isize * lda) as usize];
                let d = dst[(j as isize + i as isize * ldb) as usize];
                assert_eq!(d, s, "mismatch at i={i}, j={j}");
            }
        }
    }

    #[test]
    fn test_macro_kernel_f64_small() {
        // 3×5 tile (smaller than MICRO=4)
        let block_a = 3;
        let block_b = 5;
        let lda = 8isize;
        let ldb = 6isize;

        let src_len = ((block_a - 1) as isize + (block_b - 1) as isize * lda + 1) as usize;
        let dst_len = ((block_b - 1) as isize + (block_a - 1) as isize * ldb + 1) as usize;
        let src: Vec<f64> = (0..src_len).map(|i| i as f64).collect();
        let mut dst = vec![0.0f64; dst_len];

        unsafe {
            macro_kernel_f64(src.as_ptr(), lda, block_a, dst.as_mut_ptr(), ldb, block_b);
        }

        for j in 0..block_b {
            for i in 0..block_a {
                let s = src[(i as isize + j as isize * lda) as usize];
                let d = dst[(j as isize + i as isize * ldb) as usize];
                assert_eq!(d, s, "mismatch at i={i}, j={j}");
            }
        }
    }

    #[test]
    fn test_const_stride1_copy_contiguous() {
        let src = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mut dst = vec![0.0f64; 5];
        unsafe {
            const_stride1_copy(src.as_ptr(), dst.as_mut_ptr(), 5, 1, 1);
        }
        assert_eq!(dst, src);
    }

    #[test]
    fn test_const_stride1_copy_strided() {
        let src = vec![1.0f64, 0.0, 2.0, 0.0, 3.0];
        let mut dst = vec![0.0f64; 5];
        unsafe {
            const_stride1_copy(src.as_ptr(), dst.as_mut_ptr(), 3, 2, 1);
        }
        assert_eq!(dst[0], 1.0);
        assert_eq!(dst[1], 2.0);
        assert_eq!(dst[2], 3.0);
    }
}
