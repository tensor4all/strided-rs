#![cfg(feature = "simd")]
use core::mem::MaybeUninit;
use num_complex::{Complex32, Complex64};
use strided_basic::{mul_into_uninit, StridedView, StridedViewMut};

macro_rules! check {
    ($name:ident, $ty:ty, $value:expr) => {
        #[test]
        fn $name() {
            let value: fn(usize) -> $ty = $value;
            for len in 0..130 {
                for stride in [1, 2] {
                    let a: Vec<_> = (0..2 * (len + 2)).map(value).collect();
                    let b: Vec<_> = (0..len + 2).map(|i| value(i + 2)).collect();
                    let sentinel = value(999);
                    let mut out = vec![MaybeUninit::uninit(); len + 2];
                    out[0].write(sentinel);
                    out[len + 1].write(sentinel);
                    let av = StridedView::<$ty>::new(&a, &[len], &[stride], 1).unwrap();
                    let bv = StridedView::<$ty>::new(&b, &[len], &[1], 1).unwrap();
                    let mut dst = StridedViewMut::new(&mut out, &[len], &[1], 1).unwrap();
                    mul_into_uninit(&mut dst, &av, &bv).unwrap();
                    // SAFETY: sentinels were written above; kernel success
                    // initializes only the requested interior view.
                    unsafe {
                        assert_eq!(out[0].assume_init(), sentinel);
                        assert_eq!(out[len + 1].assume_init(), sentinel);
                        for i in 0..len {
                            assert_eq!(
                                out[i + 1].assume_init(),
                                a[1 + i * stride as usize] * b[i + 1],
                                "length {len}, stride {stride}, element {i}"
                            );
                        }
                    }
                }
            }
        }
    };
}
check!(f32_unaligned_body_tail_and_fallback, f32, |i| i as f32
    / 8.0
    - 3.0);
check!(f64_unaligned_body_tail_and_fallback, f64, |i| i as f64
    / 8.0
    - 3.0);
check!(c32_unaligned_body_tail_and_fallback, Complex32, |i| {
    Complex32::new(i as f32 / 8.0 - 3.0, 0.5)
});
check!(c64_unaligned_body_tail_and_fallback, Complex64, |i| {
    Complex64::new(i as f64 / 8.0 - 3.0, 0.5)
});

#[test]
fn float_special_values() {
    let a = [
        f64::NAN,
        f64::INFINITY,
        -0.0,
        0.0,
        f64::MIN_POSITIVE,
        3.0,
        -0.0,
    ];
    let b = [1.0, -2.0, 2.0, -2.0, 0.5, f64::INFINITY, -2.0];
    let mut out = [MaybeUninit::uninit(); 7];
    mul_into_uninit(
        &mut StridedViewMut::new(&mut out, &[7], &[1], 0).unwrap(),
        &StridedView::<f64>::new(&a, &[7], &[1], 0).unwrap(),
        &StridedView::<f64>::new(&b, &[7], &[1], 0).unwrap(),
    )
    .unwrap();
    for i in 0..7 {
        // SAFETY: the kernel successfully initialized all seven values.
        let actual = unsafe { out[i].assume_init() };
        let expected = a[i] * b[i];
        assert!(if expected.is_nan() {
            actual.is_nan()
        } else {
            actual.to_bits() == expected.to_bits()
        });
    }
}
