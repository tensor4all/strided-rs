use super::*;

/// Highest and lowest representable `f64` powers used by the issue's cases.
fn c64(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

#[test]
fn normal_range_matches_the_textbook_formula() {
    let a = c64(3.0, 4.0);
    let b = c64(1.0, -2.0);
    let expected = a / b;
    let actual = robust_complex_divide_f64(a, b);
    assert!((actual.re - expected.re).abs() <= 1e-15 * expected.re.abs().max(1.0));
    assert!((actual.im - expected.im).abs() <= 1e-15 * expected.im.abs().max(1.0));
}

#[test]
fn huge_denominator_stays_representable() {
    // 1 / (2^600 + 2^600 i) == 2^-601 (1 - i)
    let scale = 2f64.powi(600);
    let b = c64(scale, scale);
    let actual = robust_complex_divide_f64(c64(1.0, 0.0), b);
    let expected = 2f64.powi(-601);
    assert!(
        (actual.re - expected).abs() <= expected * 1e-15,
        "{actual:?}"
    );
    assert!(
        (actual.im + expected).abs() <= expected * 1e-15,
        "{actual:?}"
    );
}

#[test]
fn tiny_denominator_stays_representable() {
    // 1 / (2^-600 + 2^-600 i) == 2^599 (1 - i)
    let scale = 2f64.powi(-600);
    let b = c64(scale, scale);
    let actual = robust_complex_divide_f64(c64(1.0, 0.0), b);
    let expected = 2f64.powi(599);
    assert!(
        (actual.re - expected).abs() <= expected * 1e-15,
        "{actual:?}"
    );
    assert!(
        (actual.im + expected).abs() <= expected * 1e-15,
        "{actual:?}"
    );
}

#[test]
fn infinite_components_produce_signed_zeros() {
    let actual = robust_complex_divide_f64(c64(1.0, 2.0), c64(f64::INFINITY, 1.0));
    assert_eq!(actual.re, 0.0);
    assert!(
        actual.im.is_sign_negative() && actual.im == 0.0,
        "{actual:?}"
    );
}

#[test]
fn f32_widens_so_the_squares_cannot_overflow() {
    let scale = 2f32.powi(60);
    let actual = robust_complex_divide_f32(Complex32::new(1.0, 0.0), Complex32::new(scale, scale));
    let expected = 2f32.powi(-61);
    assert!(
        (actual.re - expected).abs() <= expected * 1e-6,
        "{actual:?}"
    );
    assert!(
        (actual.im + expected).abs() <= expected * 1e-6,
        "{actual:?}"
    );
}
