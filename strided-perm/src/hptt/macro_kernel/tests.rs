use super::*;

#[test]
fn test_macro_kernel_f64_full_block() {
    // 16×16 tile, lda=16, ldb=16 (both square)
    let n = 16;
    let src: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
    let mut dst = vec![0.0f64; n * n];

    // SAFETY: (macro_kernel.rs:22) separate buffers; the largest offset is 15 + 15 * 16 = 255 on
    // both sides, below both lengths (256).
    unsafe {
        macro_kernel_f64(src.as_ptr(), n as isize, n, dst.as_mut_ptr(), n as isize, n);
    }

    for j in 0..n {
        for i in 0..n {
            assert_eq!(dst[j + i * n], src[i + j * n], "mismatch at i={i}, j={j}");
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

    let src: Vec<f64> = (0..(block_a as isize + (block_b - 1) as isize * lda + 1) as usize)
        .map(|i| i as f64)
        .collect();
    let mut dst =
        vec![0.0f64; ((block_b - 1) as isize + (block_a - 1) as isize * ldb + 1) as usize];

    // SAFETY: (macro_kernel.rs:22) separate buffers; the largest offset is 14 + 16 * 20 = 334 <
    // 336 (source) and 16 + 14 * 18 = 268 < 269 (destination).
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

    // SAFETY: (macro_kernel.rs:22) separate buffers sized to the largest offset plus one: 2 + 4 *
    // 8 = 34 (source) and 4 + 2 * 6 = 16 (destination).
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
    // SAFETY: (macro_kernel.rs:172) separate five-element buffers; offsets 0..5 on both sides.
    unsafe {
        const_stride1_copy(src.as_ptr(), dst.as_mut_ptr(), 5, 1, 1);
    }
    assert_eq!(dst, src);
}

#[test]
fn test_const_stride1_copy_strided() {
    let src = [1.0f64, 0.0, 2.0, 0.0, 3.0];
    let mut dst = vec![0.0f64; 5];
    // SAFETY: (macro_kernel.rs:172) separate buffers; source offsets 0, 2, 4 < 5, destination
    // offsets 0, 1, 2 < 5.
    unsafe {
        const_stride1_copy(src.as_ptr(), dst.as_mut_ptr(), 3, 2, 1);
    }
    assert_eq!(dst[0], 1.0);
    assert_eq!(dst[1], 2.0);
    assert_eq!(dst[2], 3.0);
}

#[test]
fn const_stride1_leaf_advances_wrap_at_the_allocation_end() {
    // Two elements with source stride 3 live at offsets 0 and 3 in a
    // 4-element buffer; the copy's terminal advance would read offset 6.
    // The cursor is not dereferenced after the last copy, so the advance
    // must be a wrapping one; Miri flags the pre-fix `offset`.
    let src = [1.0f64, 2.0, 3.0, 4.0];
    let mut dst = vec![0.0f64; 2];
    // SAFETY: (macro_kernel.rs:172) separate buffers; source offsets 0, 3 < 4, destination
    // offsets 0, 1 < 2. The terminal advance wraps and is not read.
    unsafe {
        const_stride1_copy(src.as_ptr(), dst.as_mut_ptr(), 2, 3, 1);
    }
    assert_eq!(dst, vec![1.0, 4.0]);
}
