//! Macro-kernel: processes a BLOCK × BLOCK tile using a grid of micro-kernels.
//!
//! Mirrors HPTT C++ `macro_kernel` (transpose.cpp lines 396-560).
//! Each macro-kernel call handles a tile of up to BLOCK × BLOCK elements,
//! invoking the micro-kernel for full MICRO × MICRO sub-tiles and scalar
//! loops for edge remainders.

use crate::hptt::micro_kernel::{MicroKernel, ScalarKernel};
use crate::hptt::Store;

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
///
/// `src` must be valid for reads at `i + j * lda` and `dst` for writes at
/// `j + i * ldb` for every `i < block_a`, `j < block_b`, and the two extents
/// must not overlap.
#[inline]
pub unsafe fn macro_kernel_f64(
    src: *const f64,
    lda: isize,
    block_a: usize,
    dst: *mut f64,
    ldb: isize,
    block_b: usize,
) {
    // SAFETY: (covers the micro-kernel calls and the remainder loops) every
    // `offset` is `i + j * lda` on `src` or `j + i * ldb` on `dst` with
    // `i < block_a` and `j < block_b`: the full tiles start at multiples of
    // `MICRO` below `full_* * MICRO <= block_*`, and the remainder loops stop
    // at `rem_*`. So each access is a tile element under this function's
    // `# Safety` (macro_kernel.rs:22), and each micro-kernel call gets a full
    // `MICRO` x `MICRO` sub-tile (stride-1 along `i`, rows of `lda`/`ldb`), as
    // `transpose_micro` (micro_kernel/mod.rs:25) requires.
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
                    *dst.offset((j + jj) + (i + ii) * ldb) = *src.offset((i + ii) + (j + jj) * lda);
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
                    *dst.offset((j + jj) + (i + ii) * ldb) = *src.offset((i + ii) + (j + jj) * lda);
                }
            }
        }
        // Corner remainder (both rem_a and rem_b)
        if rem_a > 0 {
            let i = (full_a * MICRO) as isize;
            for jj in 0..rem_b as isize {
                for ii in 0..rem_a as isize {
                    *dst.offset((j + jj) + (i + ii) * ldb) = *src.offset((i + ii) + (j + jj) * lda);
                }
            }
        }
    }
}

/// Process a tile of `block_a × block_b` elements using f32 micro-kernels.
///
/// # Safety
///
/// As [`macro_kernel_f64`], with `f32` elements.
#[inline]
pub unsafe fn macro_kernel_f32(
    src: *const f32,
    lda: isize,
    block_a: usize,
    dst: *mut f32,
    ldb: isize,
    block_b: usize,
) {
    // SAFETY: (covers the micro-kernel calls and the remainder loops) every
    // `offset` is `i + j * lda` on `src` or `j + i * ldb` on `dst` with
    // `i < block_a` and `j < block_b`: the full tiles start at multiples of
    // `MICRO` below `full_* * MICRO <= block_*`, and the remainder loops stop
    // at `rem_*`. So each access is a tile element under this function's
    // `# Safety` (macro_kernel.rs:99), and each micro-kernel call gets a full
    // `MICRO` x `MICRO` sub-tile (stride-1 along `i`, rows of `lda`/`ldb`), as
    // `transpose_micro` (micro_kernel/mod.rs:25) requires.
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
                    *dst.offset((j + jj) + (i + ii) * ldb) = *src.offset((i + ii) + (j + jj) * lda);
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
                    *dst.offset((j + jj) + (i + ii) * ldb) = *src.offset((i + ii) + (j + jj) * lda);
                }
            }
        }
        if rem_a > 0 {
            let i = (full_a * MICRO) as isize;
            for jj in 0..rem_b as isize {
                for ii in 0..rem_a as isize {
                    *dst.offset((j + jj) + (i + ii) * ldb) = *src.offset((i + ii) + (j + jj) * lda);
                }
            }
        }
    }
}

/// ConstStride1 inner copy: memcpy for a plain unit-stride pair, otherwise an
/// element-wise copy through `S`. A conjugating store must never take the
/// `memcpy` branch.
///
/// Used when dim_A == dim_B (both arrays have stride 1 along the same dim).
///
/// # Safety
///
/// `src` must be valid for reads at `k * src_stride` and `dst` for writes at
/// `k * dst_stride` for every `k < count`, and the two extents must not
/// overlap. Both pointers must be non-null and aligned for `T` even when
/// `count == 0`, because the contiguous branch hands them to
/// `copy_nonoverlapping`.
#[inline(always)]
pub unsafe fn const_stride1_copy<T: Copy, S: Store<T>>(
    src: *const T,
    dst: *mut T,
    count: usize,
    src_stride: isize,
    dst_stride: isize,
) {
    // SAFETY: (covers all four branches) under this function's `# Safety`
    // (macro_kernel.rs:172) both runs of `count` elements are valid and disjoint,
    // and the pointers are non-null and aligned even for `count == 0`, which
    // is what `copy_nonoverlapping` needs when both strides are 1. The
    // loops dereference only the cursors for `k < count`; the advance after
    // the last copy may leave the allocation, so it uses `wrapping_offset`
    // and is never dereferenced.
    if S::PLAIN && src_stride == 1 && dst_stride == 1 {
        std::ptr::copy_nonoverlapping(src, dst, count);
    } else if dst_stride == 1 {
        let mut s = src;
        let mut d = dst;
        for _ in 0..count {
            // SAFETY: as above; the cursor advance after the last copy may
            // leave the allocation, so it uses `wrapping_offset`.
            unsafe { S::store(d, s) };
            s = s.wrapping_offset(src_stride);
            d = d.wrapping_offset(1);
        }
    } else if src_stride == 1 {
        let mut s = src;
        let mut d = dst;
        for _ in 0..count {
            // SAFETY: as above.
            unsafe { S::store(d, s) };
            s = s.wrapping_offset(1);
            d = d.wrapping_offset(dst_stride);
        }
    } else {
        let mut s = src;
        let mut d = dst;
        for _ in 0..count {
            // SAFETY: as above.
            unsafe { S::store(d, s) };
            s = s.wrapping_offset(src_stride);
            d = d.wrapping_offset(dst_stride);
        }
    }
}

/// Fallback element-by-element 2D copy for unsupported element sizes.
///
/// Performs the same transpose as macro_kernel but without micro-kernel
/// optimization. Used for types other than f64/f32, and for every element type
/// whose store conjugates.
///
/// # Safety
///
/// As [`macro_kernel_f64`], with `T` elements.
#[inline]
pub unsafe fn macro_kernel_fallback<T: Copy, S: Store<T>>(
    src: *const T,
    lda: isize,
    block_a: usize,
    dst: *mut T,
    ldb: isize,
    block_b: usize,
) {
    // SAFETY: `i < block_a` and `j < block_b`, so both accesses are tile
    // elements under this function's `# Safety` (macro_kernel.rs:231).
    for j in 0..block_b as isize {
        for i in 0..block_a as isize {
            unsafe { S::store(dst.offset(j + i * ldb), src.offset(i + j * lda)) };
        }
    }
}

#[cfg(test)]
mod tests;
