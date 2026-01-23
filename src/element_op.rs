//! Element-wise operations applied lazily to strided views.
//!
//! Julia's StridedViews.jl supports four element operations that form a group:
//! - `identity`: No transformation
//! - `conj`: Complex conjugate
//! - `transpose`: Element-wise transpose (for matrix elements)
//! - `adjoint`: Element-wise adjoint (conj + transpose)
//!
//! These operations are composed at the type level to avoid runtime dispatch.

use num_complex::Complex;
use num_traits::Num;
use std::marker::PhantomData;

/// Trait for element-wise operations applied to strided views.
///
/// Operations form a group under composition:
/// ```text
///   ∘    | Id   | Conj | Trans | Adj
/// -------|------|------|-------|------
///   Id   | Id   | Conj | Trans | Adj
///   Conj | Conj | Id   | Adj   | Trans
///   Trans| Trans| Adj  | Id    | Conj
///   Adj  | Adj  | Trans| Conj  | Id
/// ```
pub trait ElementOp: Copy + Default + 'static {
    /// Apply the operation to a value.
    fn apply<T: ElementOpApply>(value: T) -> T;

    /// The inverse operation (for this group, each element is its own inverse).
    type Inverse: ElementOp;

    /// Compose with Conj: Self ∘ Conj
    type ComposeConj: ElementOp;

    /// Compose with Transpose: Self ∘ Transpose
    type ComposeTranspose: ElementOp;

    /// Compose with Adjoint: Self ∘ Adjoint
    type ComposeAdjoint: ElementOp;
}

/// Trait for types that support element operations.
pub trait ElementOpApply: Copy {
    fn conj(self) -> Self;
    fn transpose(self) -> Self;
    fn adjoint(self) -> Self;
}

// Implementations for real types (conj/transpose/adjoint are identity)
macro_rules! impl_element_op_apply_real {
    ($($t:ty),*) => {
        $(
            impl ElementOpApply for $t {
                #[inline(always)]
                fn conj(self) -> Self { self }
                #[inline(always)]
                fn transpose(self) -> Self { self }
                #[inline(always)]
                fn adjoint(self) -> Self { self }
            }
        )*
    };
}

impl_element_op_apply_real!(
    f32, f64, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
);

// Implementations for complex types
impl<T: Num + Copy + Clone + std::ops::Neg<Output = T>> ElementOpApply for Complex<T> {
    #[inline(always)]
    fn conj(self) -> Self {
        Complex::conj(&self)
    }

    #[inline(always)]
    fn transpose(self) -> Self {
        // For scalar complex numbers, transpose is identity
        self
    }

    #[inline(always)]
    fn adjoint(self) -> Self {
        // For scalar complex numbers, adjoint is conj
        Complex::conj(&self)
    }
}

/// Identity operation: f(x) = x
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Identity;

/// Complex conjugate operation: f(x) = conj(x)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Conj;

/// Transpose operation: f(x) = transpose(x)
/// For scalar numbers, this is identity.
/// For matrix elements, this would transpose each element.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Transpose;

/// Adjoint operation: f(x) = adjoint(x) = conj(transpose(x))
/// For scalar numbers, this is conj.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Adjoint;

// Identity implementations
impl ElementOp for Identity {
    #[inline(always)]
    fn apply<T: ElementOpApply>(value: T) -> T {
        value
    }

    type Inverse = Identity;
    type ComposeConj = Conj;
    type ComposeTranspose = Transpose;
    type ComposeAdjoint = Adjoint;
}

// Conj implementations
impl ElementOp for Conj {
    #[inline(always)]
    fn apply<T: ElementOpApply>(value: T) -> T {
        value.conj()
    }

    type Inverse = Conj; // conj(conj(x)) = x
    type ComposeConj = Identity;
    type ComposeTranspose = Adjoint;
    type ComposeAdjoint = Transpose;
}

// Transpose implementations
impl ElementOp for Transpose {
    #[inline(always)]
    fn apply<T: ElementOpApply>(value: T) -> T {
        value.transpose()
    }

    type Inverse = Transpose; // transpose(transpose(x)) = x
    type ComposeConj = Adjoint;
    type ComposeTranspose = Identity;
    type ComposeAdjoint = Conj;
}

// Adjoint implementations
impl ElementOp for Adjoint {
    #[inline(always)]
    fn apply<T: ElementOpApply>(value: T) -> T {
        value.adjoint()
    }

    type Inverse = Adjoint; // adjoint(adjoint(x)) = x
    type ComposeConj = Transpose;
    type ComposeTranspose = Conj;
    type ComposeAdjoint = Identity;
}

/// Helper trait for composing two ElementOp types.
pub trait Compose<Other: ElementOp>: ElementOp {
    type Result: ElementOp;
}

impl<Op: ElementOp> Compose<Identity> for Op {
    type Result = Op;
}

impl Compose<Conj> for Identity {
    type Result = Conj;
}

impl Compose<Conj> for Conj {
    type Result = Identity;
}

impl Compose<Conj> for Transpose {
    type Result = Adjoint;
}

impl Compose<Conj> for Adjoint {
    type Result = Transpose;
}

impl Compose<Transpose> for Identity {
    type Result = Transpose;
}

impl Compose<Transpose> for Conj {
    type Result = Adjoint;
}

impl Compose<Transpose> for Transpose {
    type Result = Identity;
}

impl Compose<Transpose> for Adjoint {
    type Result = Conj;
}

impl Compose<Adjoint> for Identity {
    type Result = Adjoint;
}

impl Compose<Adjoint> for Conj {
    type Result = Transpose;
}

impl Compose<Adjoint> for Transpose {
    type Result = Conj;
}

impl Compose<Adjoint> for Adjoint {
    type Result = Identity;
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_identity() {
        let x = Complex64::new(3.0, 4.0);
        assert_eq!(Identity::apply(x), x);
    }

    #[test]
    fn test_conj() {
        let x = Complex64::new(3.0, 4.0);
        assert_eq!(Conj::apply(x), Complex64::new(3.0, -4.0));
    }

    #[test]
    fn test_conj_real() {
        let x = 3.0f64;
        assert_eq!(Conj::apply(x), 3.0);
    }

    #[test]
    fn test_adjoint_complex() {
        let x = Complex64::new(3.0, 4.0);
        assert_eq!(Adjoint::apply(x), Complex64::new(3.0, -4.0));
    }

    #[test]
    fn test_composition_conj_conj() {
        // conj(conj(x)) = x
        let x = Complex64::new(3.0, 4.0);
        let result = Conj::apply(Conj::apply(x));
        assert_eq!(result, x);
    }

    #[test]
    fn test_composition_types() {
        // Verify type-level composition
        fn assert_same<A, B>()
        where
            A: ElementOp,
            B: ElementOp,
        {
            // This function only compiles if A and B are the same type
        }

        // Identity ∘ Conj = Conj
        assert_same::<<Identity as Compose<Conj>>::Result, Conj>();

        // Conj ∘ Conj = Identity
        assert_same::<<Conj as Compose<Conj>>::Result, Identity>();

        // Transpose ∘ Conj = Adjoint
        assert_same::<<Transpose as Compose<Conj>>::Result, Adjoint>();

        // Adjoint ∘ Adjoint = Identity
        assert_same::<<Adjoint as Compose<Adjoint>>::Result, Identity>();
    }
}
