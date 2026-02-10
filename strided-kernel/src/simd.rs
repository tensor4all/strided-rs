#[inline(always)]
pub(crate) fn dispatch<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(feature = "simd")]
    {
        pulp::Arch::new().dispatch(f)
    }
    #[cfg(not(feature = "simd"))]
    {
        f()
    }
}

#[inline(always)]
pub(crate) fn dispatch_if_large<R>(len: usize, f: impl FnOnce() -> R) -> R {
    // Avoid runtime-dispatch overhead for tiny loops (especially common for small-array cases).
    // This is a heuristic; correctness does not depend on it.
    if len >= 64 {
        dispatch(f)
    } else {
        f()
    }
}

/// Trait for types that may have SIMD-accelerated sum/dot operations.
///
/// Default implementations return `None` (no SIMD available).
/// f32/f64 override these with SIMD kernels when the `simd` feature is enabled.
pub trait MaybeSimdOps: Copy + Sized {
    fn try_simd_sum(_src: &[Self]) -> Option<Self> {
        None
    }
    fn try_simd_dot(_a: &[Self], _b: &[Self]) -> Option<Self> {
        None
    }
}

// Default (no-op) impls for integer types and Complex
macro_rules! impl_no_simd {
    ($($t:ty),*) => {
        $(impl MaybeSimdOps for $t {})*
    };
}

impl_no_simd!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

impl<T: num_traits::Num + Copy + Clone + std::ops::Neg<Output = T>> MaybeSimdOps
    for num_complex::Complex<T>
{
}

// f32/f64: SIMD-accelerated when feature enabled, no-op otherwise
#[cfg(not(feature = "simd"))]
impl MaybeSimdOps for f32 {}

#[cfg(not(feature = "simd"))]
impl MaybeSimdOps for f64 {}

#[cfg(feature = "simd")]
mod simd_impls {
    use super::MaybeSimdOps;
    use pulp::{Simd, WithSimd};

    impl MaybeSimdOps for f32 {
        fn try_simd_sum(src: &[f32]) -> Option<f32> {
            struct Sum<'a>(&'a [f32]);
            impl<'a> WithSimd for Sum<'a> {
                type Output = f32;

                #[inline(always)]
                fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                    let (head, tail) = S::as_simd_f32s(self.0);

                    let mut acc0 = simd.splat_f32s(0.0);
                    let mut acc1 = simd.splat_f32s(0.0);
                    let mut acc2 = simd.splat_f32s(0.0);
                    let mut acc3 = simd.splat_f32s(0.0);

                    let mut i = 0usize;
                    while i + 4 <= head.len() {
                        acc0 = simd.add_f32s(acc0, head[i]);
                        acc1 = simd.add_f32s(acc1, head[i + 1]);
                        acc2 = simd.add_f32s(acc2, head[i + 2]);
                        acc3 = simd.add_f32s(acc3, head[i + 3]);
                        i += 4;
                    }
                    for &v in &head[i..] {
                        acc0 = simd.add_f32s(acc0, v);
                    }

                    let acc = simd.add_f32s(simd.add_f32s(acc0, acc1), simd.add_f32s(acc2, acc3));
                    let mut sum = simd.reduce_sum_f32s(acc);
                    for &x in tail {
                        sum += x;
                    }
                    sum
                }
            }

            Some(pulp::Arch::new().dispatch(Sum(src)))
        }

        fn try_simd_dot(a: &[f32], b: &[f32]) -> Option<f32> {
            struct Dot<'a> {
                a: &'a [f32],
                b: &'a [f32],
            }
            impl<'a> WithSimd for Dot<'a> {
                type Output = f32;

                #[inline(always)]
                fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                    debug_assert_eq!(self.a.len(), self.b.len());
                    let (a_head, a_tail) = S::as_simd_f32s(self.a);
                    let (b_head, b_tail) = S::as_simd_f32s(self.b);
                    debug_assert_eq!(a_head.len(), b_head.len());
                    debug_assert_eq!(a_tail.len(), b_tail.len());

                    let mut acc0 = simd.splat_f32s(0.0);
                    let mut acc1 = simd.splat_f32s(0.0);
                    let mut acc2 = simd.splat_f32s(0.0);
                    let mut acc3 = simd.splat_f32s(0.0);

                    let mut i = 0usize;
                    while i + 4 <= a_head.len() {
                        acc0 = simd.mul_add_f32s(a_head[i], b_head[i], acc0);
                        acc1 = simd.mul_add_f32s(a_head[i + 1], b_head[i + 1], acc1);
                        acc2 = simd.mul_add_f32s(a_head[i + 2], b_head[i + 2], acc2);
                        acc3 = simd.mul_add_f32s(a_head[i + 3], b_head[i + 3], acc3);
                        i += 4;
                    }
                    for j in i..a_head.len() {
                        acc0 = simd.mul_add_f32s(a_head[j], b_head[j], acc0);
                    }

                    let acc = simd.add_f32s(simd.add_f32s(acc0, acc1), simd.add_f32s(acc2, acc3));
                    let mut sum = simd.reduce_sum_f32s(acc);
                    for (&x, &y) in a_tail.iter().zip(b_tail.iter()) {
                        sum += x * y;
                    }
                    sum
                }
            }

            Some(pulp::Arch::new().dispatch(Dot { a, b }))
        }
    }

    impl MaybeSimdOps for f64 {
        fn try_simd_sum(src: &[f64]) -> Option<f64> {
            struct Sum<'a>(&'a [f64]);
            impl<'a> WithSimd for Sum<'a> {
                type Output = f64;

                #[inline(always)]
                fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                    let (head, tail) = S::as_simd_f64s(self.0);

                    let mut acc0 = simd.splat_f64s(0.0);
                    let mut acc1 = simd.splat_f64s(0.0);
                    let mut acc2 = simd.splat_f64s(0.0);
                    let mut acc3 = simd.splat_f64s(0.0);

                    let mut i = 0usize;
                    while i + 4 <= head.len() {
                        acc0 = simd.add_f64s(acc0, head[i]);
                        acc1 = simd.add_f64s(acc1, head[i + 1]);
                        acc2 = simd.add_f64s(acc2, head[i + 2]);
                        acc3 = simd.add_f64s(acc3, head[i + 3]);
                        i += 4;
                    }
                    for &v in &head[i..] {
                        acc0 = simd.add_f64s(acc0, v);
                    }

                    let acc = simd.add_f64s(simd.add_f64s(acc0, acc1), simd.add_f64s(acc2, acc3));
                    let mut sum = simd.reduce_sum_f64s(acc);
                    for &x in tail {
                        sum += x;
                    }
                    sum
                }
            }

            Some(pulp::Arch::new().dispatch(Sum(src)))
        }

        fn try_simd_dot(a: &[f64], b: &[f64]) -> Option<f64> {
            struct Dot<'a> {
                a: &'a [f64],
                b: &'a [f64],
            }
            impl<'a> WithSimd for Dot<'a> {
                type Output = f64;

                #[inline(always)]
                fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                    debug_assert_eq!(self.a.len(), self.b.len());
                    let (a_head, a_tail) = S::as_simd_f64s(self.a);
                    let (b_head, b_tail) = S::as_simd_f64s(self.b);
                    debug_assert_eq!(a_head.len(), b_head.len());
                    debug_assert_eq!(a_tail.len(), b_tail.len());

                    let mut acc0 = simd.splat_f64s(0.0);
                    let mut acc1 = simd.splat_f64s(0.0);
                    let mut acc2 = simd.splat_f64s(0.0);
                    let mut acc3 = simd.splat_f64s(0.0);

                    let mut i = 0usize;
                    while i + 4 <= a_head.len() {
                        acc0 = simd.mul_add_f64s(a_head[i], b_head[i], acc0);
                        acc1 = simd.mul_add_f64s(a_head[i + 1], b_head[i + 1], acc1);
                        acc2 = simd.mul_add_f64s(a_head[i + 2], b_head[i + 2], acc2);
                        acc3 = simd.mul_add_f64s(a_head[i + 3], b_head[i + 3], acc3);
                        i += 4;
                    }
                    for j in i..a_head.len() {
                        acc0 = simd.mul_add_f64s(a_head[j], b_head[j], acc0);
                    }

                    let acc = simd.add_f64s(simd.add_f64s(acc0, acc1), simd.add_f64s(acc2, acc3));
                    let mut sum = simd.reduce_sum_f64s(acc);
                    for (&x, &y) in a_tail.iter().zip(b_tail.iter()) {
                        sum += x * y;
                    }
                    sum
                }
            }

            Some(pulp::Arch::new().dispatch(Dot { a, b }))
        }
    }
}
