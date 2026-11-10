use super::*;

/// The beyond-L2 branch is a property of the class under test, not of the
/// machine that runs the test, so it is asserted with the class given
/// explicitly (strided-rs#364).
#[test]
fn a_footprint_past_l2_is_priced_above_the_row_of_its_class() {
    // Four MiB: the 512 x 512 f64 case of strided-rs#363, source plus
    // destination.
    assert_eq!(
        copy_ns_per_element_for_footprint(HostClass::X86_64Avx512, 4 << 20),
        3.0
    );
    // Up to and including the class's L2 size the row stands.
    assert_eq!(
        copy_ns_per_element_for_footprint(HostClass::X86_64Avx512, COPY_L2_BYTES),
        0.171
    );
    assert_eq!(
        copy_ns_per_element_for_footprint(HostClass::X86_64Avx512, 0),
        0.171
    );
    // One byte past it is priced at the beyond-L2 rate.
    assert_eq!(
        copy_ns_per_element_for_footprint(HostClass::X86_64Avx512, COPY_L2_BYTES + 1),
        3.0
    );
    // Every class without a beyond-L2 measurement keeps one rate at every
    // size.
    for class in [
        HostClass::Aarch64Apple,
        HostClass::X86_64Avx2,
        HostClass::X86_64Scalar,
        HostClass::Aarch64,
        HostClass::Generic,
    ] {
        assert_eq!(
            copy_ns_per_element_for_footprint(class, usize::MAX >> 8),
            copy_ns_per_element_of(class),
            "{} has no beyond-L2 measurement",
            class.name()
        );
    }
}
