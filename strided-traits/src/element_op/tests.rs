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
    let result = <Conj as ElementOp<Complex64>>::apply(<Conj as ElementOp<Complex64>>::apply(x));
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
