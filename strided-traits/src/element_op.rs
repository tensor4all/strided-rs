//! Element-wise operations applied lazily to strided views.
//!
//! Julia's StridedViews.jl supports four element operations that form a group:
//! - `identity`: No transformation
//! - `conj`: Complex conjugate
//! - `transpose`: Element-wise transpose (for matrix elements)
//! - `adjoint`: Element-wise adjoint (conj + transpose)
//!
//! These operations are composed at the type level to avoid runtime dispatch.
//!
//! # Key Design: `ElementOp<T>` is Generic Over T
//!
//! `Identity` implements `ElementOp<T>` for any `T: Copy`, requiring no
//! additional bounds. `Conj`, `Transpose`, and `Adjoint` require
//! `T: ElementOpApply`. This allows custom scalar types (e.g., tropical
//! semiring types) to use `Identity` views without implementing
//! `ElementOpApply`.
//!
//! Composition associated types (Inverse, ComposeConj, etc.) are separated
//! into `ComposableElementOp<T>`, only available when `T: ElementOpApply`.

use num_complex::Complex;
use num_traits::Num;

// ---------------------------------------------------------------------------
// ElementOpApply: trait for types that support conj/transpose/adjoint
// ---------------------------------------------------------------------------

/// Trait for types that support element operations (conj, transpose, adjoint).
///
/// Default implementations return `self` unchanged, so real-valued types
/// (and custom types that don't need complex operations) can simply write:
/// ```ignore
/// impl ElementOpApply for MyType {}
/// ```
pub trait ElementOpApply: Copy {
    #[inline(always)]
    fn conj(self) -> Self {
        self
    }
    #[inline(always)]
    fn transpose(self) -> Self {
        self
    }
    #[inline(always)]
    fn adjoint(self) -> Self {
        self
    }
}

// Real types: use default identity implementations
macro_rules! impl_element_op_apply_real {
    ($($t:ty),*) => {
        $(impl ElementOpApply for $t {})*
    };
}

impl_element_op_apply_real!(
    f32, f64, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
);

// Complex types: override with actual conjugation
impl<T: Num + Copy + Clone + std::ops::Neg<Output = T>> ElementOpApply for Complex<T> {
    #[inline(always)]
    fn conj(self) -> Self {
        Complex::conj(&self)
    }

    #[inline(always)]
    fn transpose(self) -> Self {
        self
    }

    #[inline(always)]
    fn adjoint(self) -> Self {
        Complex::conj(&self)
    }
}

// ---------------------------------------------------------------------------
// Marker types
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// ElementOp<T>: generic over element type
// ---------------------------------------------------------------------------

/// Trait for element-wise operations applied to strided views.
///
/// Generic over the element type `T`. `Identity` implements this for any
/// `T: Copy`, while `Conj`, `Transpose`, and `Adjoint` require
/// `T: ElementOpApply`.
///
/// Operations form a group under composition:
/// ```text
///   compose | Id   | Conj | Trans | Adj
/// ---------|------|------|-------|------
///   Id     | Id   | Conj | Trans | Adj
///   Conj   | Conj | Id   | Adj   | Trans
///   Trans  | Trans| Adj  | Id    | Conj
///   Adj    | Adj  | Trans| Conj  | Id
/// ```
pub trait ElementOp<T>: Copy + Default + 'static {
    /// Whether this operation is the identity (no-op).
    const IS_IDENTITY: bool = false;

    /// Apply the operation to a value.
    fn apply(value: T) -> T;
}

// Identity: works with ANY Copy type (no ElementOpApply needed)
impl<T: Copy> ElementOp<T> for Identity {
    const IS_IDENTITY: bool = true;

    #[inline(always)]
    fn apply(value: T) -> T {
        value
    }
}

// Conj, Transpose, Adjoint: only work with ElementOpApply types
impl<T: ElementOpApply> ElementOp<T> for Conj {
    #[inline(always)]
    fn apply(value: T) -> T {
        value.conj()
    }
}

impl<T: ElementOpApply> ElementOp<T> for Transpose {
    #[inline(always)]
    fn apply(value: T) -> T {
        value.transpose()
    }
}

impl<T: ElementOpApply> ElementOp<T> for Adjoint {
    #[inline(always)]
    fn apply(value: T) -> T {
        value.adjoint()
    }
}

// ---------------------------------------------------------------------------
// ComposableElementOp<T>: composition associated types
// ---------------------------------------------------------------------------

/// Trait for element operations that support type-level composition.
///
/// Only available when `T: ElementOpApply`, since composition with
/// `Conj`/`Transpose`/`Adjoint` requires the element type to support
/// those operations.
pub trait ComposableElementOp<T: ElementOpApply>: ElementOp<T> {
    /// The inverse operation (for this group, each element is its own inverse).
    type Inverse: ComposableElementOp<T>;

    /// Compose with Conj: Self then Conj
    type ComposeConj: ComposableElementOp<T>;

    /// Compose with Transpose: Self then Transpose
    type ComposeTranspose: ComposableElementOp<T>;

    /// Compose with Adjoint: Self then Adjoint
    type ComposeAdjoint: ComposableElementOp<T>;
}

impl<T: ElementOpApply> ComposableElementOp<T> for Identity {
    type Inverse = Identity;
    type ComposeConj = Conj;
    type ComposeTranspose = Transpose;
    type ComposeAdjoint = Adjoint;
}

impl<T: ElementOpApply> ComposableElementOp<T> for Conj {
    type Inverse = Conj;
    type ComposeConj = Identity;
    type ComposeTranspose = Adjoint;
    type ComposeAdjoint = Transpose;
}

impl<T: ElementOpApply> ComposableElementOp<T> for Transpose {
    type Inverse = Transpose;
    type ComposeConj = Adjoint;
    type ComposeTranspose = Identity;
    type ComposeAdjoint = Conj;
}

impl<T: ElementOpApply> ComposableElementOp<T> for Adjoint {
    type Inverse = Adjoint;
    type ComposeConj = Transpose;
    type ComposeTranspose = Conj;
    type ComposeAdjoint = Identity;
}

// ---------------------------------------------------------------------------
// Compose<Other>: helper trait for composing two ElementOp types
// ---------------------------------------------------------------------------

/// Helper trait for composing two ElementOp types.
///
/// Only available when `T: ElementOpApply`.
pub trait Compose<T: ElementOpApply, Other: ComposableElementOp<T>>:
    ComposableElementOp<T>
{
    type Result: ComposableElementOp<T>;
}

impl<T: ElementOpApply, Op: ComposableElementOp<T>> Compose<T, Identity> for Op {
    type Result = Op;
}

impl<T: ElementOpApply> Compose<T, Conj> for Identity {
    type Result = Conj;
}

impl<T: ElementOpApply> Compose<T, Conj> for Conj {
    type Result = Identity;
}

impl<T: ElementOpApply> Compose<T, Conj> for Transpose {
    type Result = Adjoint;
}

impl<T: ElementOpApply> Compose<T, Conj> for Adjoint {
    type Result = Transpose;
}

impl<T: ElementOpApply> Compose<T, Transpose> for Identity {
    type Result = Transpose;
}

impl<T: ElementOpApply> Compose<T, Transpose> for Conj {
    type Result = Adjoint;
}

impl<T: ElementOpApply> Compose<T, Transpose> for Transpose {
    type Result = Identity;
}

impl<T: ElementOpApply> Compose<T, Transpose> for Adjoint {
    type Result = Conj;
}

impl<T: ElementOpApply> Compose<T, Adjoint> for Identity {
    type Result = Adjoint;
}

impl<T: ElementOpApply> Compose<T, Adjoint> for Conj {
    type Result = Transpose;
}

impl<T: ElementOpApply> Compose<T, Adjoint> for Transpose {
    type Result = Conj;
}

impl<T: ElementOpApply> Compose<T, Adjoint> for Adjoint {
    type Result = Identity;
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_identity() {
        let x = Complex64::new(3.0, 4.0);
        assert_eq!(<Identity as ElementOp<Complex64>>::apply(x), x);
    }

    #[test]
    fn test_identity_custom_type() {
        // Custom type that is Copy but does NOT implement ElementOpApply
        #[derive(Debug, Clone, Copy, PartialEq)]
        struct MyCustom(f64);

        let x = MyCustom(42.0);
        assert_eq!(<Identity as ElementOp<MyCustom>>::apply(x), x);
    }

    #[test]
    fn test_conj() {
        let x = Complex64::new(3.0, 4.0);
        assert_eq!(
            <Conj as ElementOp<Complex64>>::apply(x),
            Complex64::new(3.0, -4.0)
        );
    }

    #[test]
    fn test_conj_real() {
        let x = 3.0f64;
        assert_eq!(<Conj as ElementOp<f64>>::apply(x), 3.0);
    }

    #[test]
    fn test_adjoint_complex() {
        let x = Complex64::new(3.0, 4.0);
        assert_eq!(
            <Adjoint as ElementOp<Complex64>>::apply(x),
            Complex64::new(3.0, -4.0)
        );
    }

    #[test]
    fn test_composition_conj_conj() {
        let x = Complex64::new(3.0, 4.0);
        let result =
            <Conj as ElementOp<Complex64>>::apply(<Conj as ElementOp<Complex64>>::apply(x));
        assert_eq!(result, x);
    }

    #[test]
    fn test_composable_types() {
        fn assert_same<A: 'static, B: 'static>() {
            assert_eq!(
                std::any::TypeId::of::<A>(),
                std::any::TypeId::of::<B>(),
                "types should be the same"
            );
        }

        // Identity composed with Conj = Conj
        assert_same::<<Identity as Compose<f64, Conj>>::Result, Conj>();

        // Conj composed with Conj = Identity
        assert_same::<<Conj as Compose<f64, Conj>>::Result, Identity>();

        // Transpose composed with Conj = Adjoint
        assert_same::<<Transpose as Compose<f64, Conj>>::Result, Adjoint>();

        // Adjoint composed with Adjoint = Identity
        assert_same::<<Adjoint as Compose<f64, Adjoint>>::Result, Identity>();
    }

    #[test]
    fn test_element_op_apply_defaults() {
        // A type using default identity implementations
        #[derive(Debug, Clone, Copy, PartialEq)]
        struct Real(f64);
        impl ElementOpApply for Real {}

        let x = Real(3.0);
        assert_eq!(x.conj(), x);
        assert_eq!(x.transpose(), x);
        assert_eq!(x.adjoint(), x);

        // Can use with all ops
        assert_eq!(<Conj as ElementOp<Real>>::apply(x), x);
        assert_eq!(<Transpose as ElementOp<Real>>::apply(x), x);
        assert_eq!(<Adjoint as ElementOp<Real>>::apply(x), x);
    }
}
