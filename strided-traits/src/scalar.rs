//! Scalar type bounds for strided operations and einsum.

/// Shared trait bounds for all element types usable with einsum, independent
/// of GEMM backend.
///
/// Unlike the previous design, `ScalarBase` does **not** require
/// `ElementOpApply`. This allows custom scalar types (e.g., tropical
/// semiring types) to satisfy einsum bounds without implementing
/// conj/transpose/adjoint.
pub trait ScalarBase:
    Copy
    + Send
    + Sync
    + std::ops::Mul<Output = Self>
    + std::ops::Add<Output = Self>
    + num_traits::Zero
    + num_traits::One
    + PartialEq
{
}

impl<T> ScalarBase for T where
    T: Copy
        + Send
        + Sync
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + num_traits::Zero
        + num_traits::One
        + PartialEq
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{One, Zero};

    fn assert_scalar_base<T: ScalarBase>() {}

    #[test]
    fn test_standard_types() {
        assert_scalar_base::<f32>();
        assert_scalar_base::<f64>();
        assert_scalar_base::<i32>();
        assert_scalar_base::<i64>();
        assert_scalar_base::<num_complex::Complex64>();
    }

    #[test]
    fn test_custom_type_without_element_op_apply() {
        // A custom type that implements the arithmetic traits
        // but NOT ElementOpApply — this should still satisfy ScalarBase
        #[derive(Debug, Clone, Copy, PartialEq)]
        struct TropicalLike(f64);

        impl std::ops::Add for TropicalLike {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                // tropical add = max
                TropicalLike(self.0.max(rhs.0))
            }
        }

        impl std::ops::Mul for TropicalLike {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self {
                // tropical mul = add
                TropicalLike(self.0 + rhs.0)
            }
        }

        impl num_traits::Zero for TropicalLike {
            fn zero() -> Self {
                TropicalLike(f64::NEG_INFINITY)
            }
            fn is_zero(&self) -> bool {
                self.0 == f64::NEG_INFINITY
            }
        }

        impl num_traits::One for TropicalLike {
            fn one() -> Self {
                TropicalLike(0.0)
            }
        }

        assert_scalar_base::<TropicalLike>();

        // Exercise the actual operations so coverage sees them
        let a = TropicalLike(3.0);
        let b = TropicalLike(5.0);
        assert_eq!((a + b).0, 5.0); // max(3, 5) = 5
        assert_eq!((a * b).0, 8.0); // 3 + 5 = 8
        assert_eq!(TropicalLike::zero().0, f64::NEG_INFINITY);
        assert!(TropicalLike::zero().is_zero());
        assert!(!a.is_zero());
        assert_eq!(TropicalLike::one().0, 0.0);
    }
}
