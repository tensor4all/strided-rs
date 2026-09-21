#[cfg(feature = "simd")]
#[test]
fn sum_squares_simd_covers_unrolled_body_and_tail() {
    const LEN: usize = 285;
    let f32_values = vec![1.0_f32; LEN];
    let f64_values = vec![1.0_f64; LEN];

    assert_eq!(
        <f32 as super::MaybeSimdSumSquares>::try_simd_sum_squares(&f32_values),
        Some(LEN as f32)
    );
    assert_eq!(
        <f64 as super::MaybeSimdSumSquares>::try_simd_sum_squares(&f64_values),
        Some(LEN as f64)
    );
}

#[cfg(feature = "simd")]
#[test]
fn test_try_mul_contiguous_complex64() {
    let a = vec![
        num_complex::Complex64::new(1.0, 2.0),
        num_complex::Complex64::new(-3.0, 4.0),
        num_complex::Complex64::new(0.5, -0.25),
    ];
    let b = vec![
        num_complex::Complex64::new(5.0, -1.0),
        num_complex::Complex64::new(2.0, 0.25),
        num_complex::Complex64::new(-4.0, 3.0),
    ];
    let mut dst = vec![num_complex::Complex64::new(0.0, 0.0); a.len()];

    assert!(super::try_mul_contiguous(&mut dst, &a, &b));
    for i in 0..a.len() {
        assert_eq!(dst[i], a[i] * b[i]);
    }
}

#[cfg(feature = "simd")]
#[test]
fn test_try_mul_contiguous_complex32() {
    let a = vec![
        num_complex::Complex32::new(1.0, 2.0),
        num_complex::Complex32::new(-3.0, 4.0),
        num_complex::Complex32::new(0.5, -0.25),
    ];
    let b = vec![
        num_complex::Complex32::new(5.0, -1.0),
        num_complex::Complex32::new(2.0, 0.25),
        num_complex::Complex32::new(-4.0, 3.0),
    ];
    let mut dst = vec![num_complex::Complex32::new(0.0, 0.0); a.len()];

    assert!(super::try_mul_contiguous(&mut dst, &a, &b));
    for i in 0..a.len() {
        assert_eq!(dst[i], a[i] * b[i]);
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_transposed_scalar_rhs_2d_f64_source_contiguous() {
    let inner_len = 5usize;
    let row_len = 7usize;
    let src: Vec<f64> = (0..inner_len * row_len).map(|i| i as f64 + 0.25).collect();
    let scalar = 2.0f64;
    let mut dst = vec![0.0f64; inner_len * row_len];

    let used = unsafe {
        super::try_mul_transposed_scalar_rhs_2d::<f64, f64, f64>(
            dst.as_mut_ptr(),
            src.as_ptr(),
            &scalar,
            inner_len,
            row_len,
            row_len as isize,
            1,
        )
    };

    assert!(used);
    for row in 0..row_len {
        for inner in 0..inner_len {
            assert_eq!(
                dst[row * inner_len + inner],
                src[inner * row_len + row] * scalar
            );
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_transposed_scalar_lhs_2d_f32_source_contiguous() {
    let inner_len = 5usize;
    let row_len = 7usize;
    let src: Vec<f32> = (0..inner_len * row_len)
        .map(|i| i as f32 * 0.5 + 1.0)
        .collect();
    let scalar = 3.0f32;
    let mut dst = vec![0.0f32; inner_len * row_len];

    let used = unsafe {
        super::try_mul_transposed_scalar_lhs_2d::<f32, f32, f32>(
            dst.as_mut_ptr(),
            &scalar,
            src.as_ptr(),
            inner_len,
            row_len,
            row_len as isize,
            1,
        )
    };

    assert!(used);
    for row in 0..row_len {
        for inner in 0..inner_len {
            assert_eq!(
                dst[row * inner_len + inner],
                scalar * src[inner * row_len + row]
            );
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_transposed_scalar_rhs_2d_handles_short_rows() {
    let inner_len = 8usize;
    let row_len = 3usize;
    let src: Vec<f64> = (0..inner_len * row_len).map(|i| i as f64 + 1.0).collect();
    let scalar = 2.0f64;
    let mut dst = vec![0.0f64; inner_len * row_len];

    let used = unsafe {
        super::try_mul_transposed_scalar_rhs_2d::<f64, f64, f64>(
            dst.as_mut_ptr(),
            src.as_ptr(),
            &scalar,
            inner_len,
            row_len,
            row_len as isize,
            1,
        )
    };

    assert!(used);
    for row in 0..row_len {
        for inner in 0..inner_len {
            assert_eq!(
                dst[row * inner_len + inner],
                src[inner * row_len + row] * scalar
            );
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_transposed_scalar_2d_handles_small_square_tiles() {
    let inner_len = 4usize;
    let row_len = 4usize;
    let src: Vec<f64> = (0..inner_len * row_len).map(|i| i as f64 + 1.0).collect();
    let scalar = 2.0f64;
    let mut rhs_dst = vec![0.0f64; inner_len * row_len];
    let mut lhs_dst = vec![0.0f64; inner_len * row_len];

    let rhs_used = unsafe {
        super::try_mul_transposed_scalar_rhs_2d::<f64, f64, f64>(
            rhs_dst.as_mut_ptr(),
            src.as_ptr(),
            &scalar,
            inner_len,
            row_len,
            row_len as isize,
            1,
        )
    };
    let lhs_used = unsafe {
        super::try_mul_transposed_scalar_lhs_2d::<f64, f64, f64>(
            lhs_dst.as_mut_ptr(),
            &scalar,
            src.as_ptr(),
            inner_len,
            row_len,
            row_len as isize,
            1,
        )
    };

    assert!(rhs_used);
    assert!(lhs_used);
    for row in 0..row_len {
        for inner in 0..inner_len {
            assert_eq!(
                rhs_dst[row * inner_len + inner],
                src[inner * row_len + row] * scalar
            );
            assert_eq!(
                lhs_dst[row * inner_len + inner],
                scalar * src[inner * row_len + row]
            );
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_transposed_scalar_rhs_2d_complex64_source_contiguous() {
    let inner_len = 5usize;
    let row_len = 7usize;
    let src: Vec<num_complex::Complex64> = (0..inner_len * row_len)
        .map(|i| num_complex::Complex64::new(i as f64 + 0.25, i as f64 * -0.5))
        .collect();
    let scalar = num_complex::Complex64::new(2.0, -0.25);
    let mut dst = vec![num_complex::Complex64::new(0.0, 0.0); inner_len * row_len];

    let used = unsafe {
        super::try_mul_transposed_scalar_rhs_2d::<
            num_complex::Complex64,
            num_complex::Complex64,
            num_complex::Complex64,
        >(
            dst.as_mut_ptr(),
            src.as_ptr(),
            &scalar,
            inner_len,
            row_len,
            row_len as isize,
            1,
        )
    };

    assert!(used);
    for row in 0..row_len {
        for inner in 0..inner_len {
            assert_eq!(
                dst[row * inner_len + inner],
                src[inner * row_len + row] * scalar
            );
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_transposed_scalar_lhs_2d_complex32_source_contiguous() {
    let inner_len = 5usize;
    let row_len = 7usize;
    let src: Vec<num_complex::Complex32> = (0..inner_len * row_len)
        .map(|i| num_complex::Complex32::new(i as f32 * 0.5 + 1.0, i as f32 * 0.25))
        .collect();
    let scalar = num_complex::Complex32::new(3.0, -0.5);
    let mut dst = vec![num_complex::Complex32::new(0.0, 0.0); inner_len * row_len];

    let used = unsafe {
        super::try_mul_transposed_scalar_lhs_2d::<
            num_complex::Complex32,
            num_complex::Complex32,
            num_complex::Complex32,
        >(
            dst.as_mut_ptr(),
            &scalar,
            src.as_ptr(),
            inner_len,
            row_len,
            row_len as isize,
            1,
        )
    };

    assert!(used);
    for row in 0..row_len {
        for inner in 0..inner_len {
            assert_eq!(
                dst[row * inner_len + inner],
                scalar * src[inner * row_len + row]
            );
        }
    }
}
