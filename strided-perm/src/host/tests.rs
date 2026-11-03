use super::*;

#[test]
fn rows_per_class() {
    assert_eq!(copy_ns_per_element_of(HostClass::X86_64Avx512), 0.171);
    assert_eq!(copy_ns_per_element_of(HostClass::Aarch64Apple), 0.090);
    for c in [
        HostClass::X86_64Avx2,
        HostClass::X86_64Scalar,
        HostClass::Aarch64,
        HostClass::Generic,
    ] {
        assert_eq!(copy_ns_per_element_of(c), 0.090, "{}", c.name());
    }
    for (c, name) in [
        (HostClass::X86_64Avx512, "x86_64-avx512"),
        (HostClass::X86_64Avx2, "x86_64-avx2"),
        (HostClass::X86_64Scalar, "x86_64-scalar"),
        (HostClass::Aarch64Apple, "aarch64-apple"),
        (HostClass::Aarch64, "aarch64"),
        (HostClass::Generic, "generic"),
    ] {
        assert_eq!(c.name(), name);
    }
    assert_eq!(HostClass::detect(), HostClass::detect());
    assert_eq!(
        copy_ns_per_element(),
        copy_ns_per_element_of(HostClass::detect())
    );
}
