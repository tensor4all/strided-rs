//! Scale-robust complex division.
//!
//! `num_complex`'s `Div` uses the textbook formula `(a·conj(b)) / (re(b)² +
//! im(b)²)`, whose denominator overflows to `inf` for `|b| ≳ 2^512` and
//! underflows to `0` for `|b| ≲ 2^-511`, even when the quotient is
//! representable. These helpers are a direct port of Julia Base
//! `base/complex.jl` (`/`, `cdiv`, `robust_cdiv1/2`, `scaling_cdiv`,
//! `scaleargs_cdiv`), which is Baudin–Smith with power-of-two scaling:
//!
//! * the normal-range case keeps the ratio form `(a + b·r)·t`;
//! * extreme numerator or denominator magnitudes are rescaled by powers of two
//!   and unscaled afterwards, so the result stays representable;
//! * `Complex<f32>` widens to `Complex<f64>` and narrows, as Julia does for
//!   `Complex{Float32}`.
//!
//! `inv(z) == 1/z` and `a/b == a·inv(b)`, so only the division is needed here.

use num_complex::{Complex32, Complex64};

/// Divide two `Complex<f64>` values without the `|z|²` overflow/underflow of
/// the textbook formula.
#[inline]
pub fn robust_complex_divide_f64(a: Complex64, b: Complex64) -> Complex64 {
    let (ar, ai) = (a.re, a.im);
    let (br, bi) = (b.re, b.im);
    if br.is_infinite() || bi.is_infinite() {
        if ar.is_finite() && ai.is_finite() {
            return Complex64::new(
                0.0 * ar.signum() * br.signum(),
                -0.0 * ai.signum() * bi.signum(),
            );
        }
        return Complex64::new(f64::NAN, f64::NAN);
    }
    // Julia deliberately uses a select instead of `max` here so that a NaN
    // component does not change the scaling branch (and stays branch-free).
    let abs_ar = ar.abs();
    let abs_ai = ai.abs();
    let ab = if abs_ar >= abs_ai { abs_ar } else { abs_ai };
    let abs_br = br.abs();
    let abs_bi = bi.abs();
    let cd = if abs_br >= abs_bi { abs_br } else { abs_bi };
    if ab >= 0.5 * f64::MAX
        || ab <= f64::MIN_POSITIVE * 2.0 / f64::EPSILON
        || cd >= 0.5 * f64::MAX
        || cd <= f64::MIN_POSITIVE * 2.0 / f64::EPSILON
    {
        scaling_cdiv_f64(ar, ai, br, bi, ab, cd)
    } else {
        cdiv_f64(ar, ai, br, bi)
    }
}

/// Divide two `Complex<f32>` values by widening to `Complex<f64>` (Julia's
/// `Complex{Float32}` route), so the `f32` squares cannot overflow.
#[inline]
pub fn robust_complex_divide_f32(a: Complex32, b: Complex32) -> Complex32 {
    let (ar, ai) = (f64::from(a.re), f64::from(a.im));
    let (br, bi) = (f64::from(b.re), f64::from(b.im));
    if br.is_infinite() || bi.is_infinite() {
        if ar.is_finite() && ai.is_finite() {
            return Complex32::new(
                (0.0 * ar.signum() * br.signum()) as f32,
                (-0.0 * ai.signum() * bi.signum()) as f32,
            );
        }
        return Complex32::new(f32::NAN, f32::NAN);
    }
    let mag = 1.0 / br.mul_add(br, bi * bi);
    let re = ar.mul_add(br, ai * bi);
    let im = ai.mul_add(br, -ar * bi);
    Complex32::new((re * mag) as f32, (im * mag) as f32)
}

#[inline]
fn cdiv_f64(a: f64, b: f64, c: f64, d: f64) -> Complex64 {
    if d.abs() <= c.abs() {
        robust_cdiv1(a, b, c, d)
    } else {
        let swapped = robust_cdiv1(b, a, d, c);
        Complex64::new(swapped.re, -swapped.im)
    }
}

#[inline]
fn scaling_cdiv_f64(a: f64, b: f64, c: f64, d: f64, ab: f64, cd: f64) -> Complex64 {
    let (a, b, c, d, s) = scale_cdiv_args(a, b, c, d, ab, cd);
    let quotient = cdiv_f64(a, b, c, d);
    Complex64::new(quotient.re * s, quotient.im * s)
}

fn scale_cdiv_args(a: f64, b: f64, c: f64, d: f64, ab: f64, cd: f64) -> (f64, f64, f64, f64, f64) {
    let half_ov = 0.5 * f64::MAX;
    let two_un_eps = f64::MIN_POSITIVE * 2.0 / f64::EPSILON;
    let big_scale = 2.0 / (f64::EPSILON * f64::EPSILON);
    let mut s = 1.0;
    let (mut a, mut b, mut c, mut d) = (a, b, c, d);
    if ab >= half_ov {
        a *= 0.5;
        b *= 0.5;
        s *= 2.0;
    } else if ab <= two_un_eps {
        a *= big_scale;
        b *= big_scale;
        s /= big_scale;
    }
    if cd >= half_ov {
        c *= 0.5;
        d *= 0.5;
        s *= 0.5;
    } else if cd <= two_un_eps {
        c *= big_scale;
        d *= big_scale;
        s *= big_scale;
    }
    (a, b, c, d, s)
}

#[inline]
fn robust_cdiv1(a: f64, b: f64, c: f64, d: f64) -> Complex64 {
    let r = d / c;
    let t = 1.0 / (c + d * r);
    Complex64::new(
        robust_cdiv2(a, b, c, d, r, t),
        robust_cdiv2(b, -a, c, d, r, t),
    )
}

#[inline]
fn robust_cdiv2(a: f64, b: f64, c: f64, d: f64, r: f64, t: f64) -> f64 {
    if r != 0.0 {
        let br = b * r;
        if br != 0.0 {
            (a + br) * t
        } else {
            a * t + (b * t) * r
        }
    } else {
        (a + d * (b / c)) * t
    }
}

#[cfg(test)]
mod tests {
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
        let actual =
            robust_complex_divide_f32(Complex32::new(1.0, 0.0), Complex32::new(scale, scale));
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
}
