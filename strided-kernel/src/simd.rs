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

#[cfg(feature = "simd")]
mod reduce {
    use pulp::{Simd, WithSimd};

    pub(crate) fn sum_f32(src: &[f32]) -> f32 {
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

        pulp::Arch::new().dispatch(Sum(src))
    }

    pub(crate) fn sum_f64(src: &[f64]) -> f64 {
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

        pulp::Arch::new().dispatch(Sum(src))
    }

    pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
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

        pulp::Arch::new().dispatch(Dot { a, b })
    }

    pub(crate) fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
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

        pulp::Arch::new().dispatch(Dot { a, b })
    }
}

#[cfg(feature = "simd")]
pub(crate) use reduce::{dot_f32, dot_f64, sum_f32, sum_f64};
