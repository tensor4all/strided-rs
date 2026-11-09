use super::*;

#[test]
fn test_scalar_f64_4x4() {
    // Source: 4×4 matrix in col-major (lda=4)
    // A = [[0,4,8,12],[1,5,9,13],[2,6,10,14],[3,7,11,15]]
    let src: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let mut dst = vec![0.0f64; 16];

    // SAFETY: (micro_kernel/mod.rs:25) separate 16-element buffers; the largest offset is 3 * 4 +
    // 3 = 15 on both sides.
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
    let mut src = [0.0f64; 20];
    for i in 0..4 {
        for j in 0..4 {
            src[i * 5 + j] = (i * 10 + j) as f64;
        }
    }
    let mut dst = vec![0.0f64; 24];

    // SAFETY: (micro_kernel/mod.rs:25) separate buffers; the largest offset is 3 * 5 + 3 = 18 <
    // 20 (source) and 3 + 3 * 6 = 21 < 24 (destination).
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

    // SAFETY: (micro_kernel/mod.rs:25) separate 64-element buffers; the largest offset is 7 * 8 +
    // 7 = 63 on both sides.
    unsafe {
        ScalarKernel::transpose_micro(src.as_ptr(), 8, dst.as_mut_ptr(), 8);
    }

    for j in 0..8 {
        for i in 0..8 {
            assert_eq!(dst[i + j * 8], src[i * 8 + j], "mismatch at i={i}, j={j}");
        }
    }
}
