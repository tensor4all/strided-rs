use super::*;
use core::mem::size_of;

/// A plan pays one cache line per run of its walk beyond the row, a hit line
/// while its working set fits the class's L2 and a DRAM line past it, so the
/// three in-L2 transposing rows of strided-rs#363's table and its two beyond-L2
/// rows are all reproduced within the tolerance that table was taken at
/// (strided-rs#364).
#[test]
fn a_plan_is_priced_by_its_runs_and_its_footprint() {
    let avx512 = HostClass::X86_64Avx512;
    let row = copy_ns_per_element_of(avx512);
    // A transposing n x n f64 copy walks one element per run and spans 2 * n^2
    // * 8 bytes. #363 measures, in ns per element: 64 -> 0.65-0.68, 128 -> 0.78,
    // 256 -> 0.90-0.94, 512 -> 3.09-3.24, 1024 -> 2.91-3.07.
    for (n, measured) in [
        (64usize, (0.65, 0.68)),
        (128, (0.78, 0.78)),
        (256, (0.90, 0.94)),
        (512, (3.09, 3.24)),
        (1024, (2.91, 3.07)),
    ] {
        let bytes = 2 * n * n * size_of::<f64>();
        let rate = copy_ns_per_element_for_plan(avx512, bytes, 1);
        assert!(
            (measured.0 / 1.25..=measured.1 * 1.25).contains(&rate),
            "{n} x {n}: the plan prices itself at {rate} ns per element, outside 1.25x \
             of the {measured:?} measured"
        );
    }
    // The table is monotone in n inside each regime, and the regime changes at
    // the class's L2, on the slow side: 256 x 256 f64 is 1 MiB of source and
    // destination together, so it sits exactly at the L2 and keeps the hit line,
    // and one byte past it is already priced at the DRAM line.
    assert_eq!(
        copy_ns_per_element_for_plan(avx512, COPY_L2_BYTES, 1),
        row + COPY_NS_PER_RUN_RESIDENT
    );
    assert_eq!(
        copy_ns_per_element_for_plan(avx512, COPY_L2_BYTES + 1, 1),
        COPY_BEYOND_L2_NS_PER_RUN
    );
    // A run of k elements pays the line once per run, not per element.
    assert_eq!(
        copy_ns_per_element_for_plan(avx512, 4 << 20, 16),
        row + (COPY_BEYOND_L2_NS_PER_RUN - row) / 16.0
    );
    // Long enough runs converge on the row from above and never below it.
    for run_len in [1usize, 2, 7, 64, 1024, usize::MAX] {
        assert!(copy_ns_per_element_for_plan(avx512, 4 << 20, run_len) >= row);
    }
    // Every class without a beyond-L2 measurement keeps one rate at every size
    // and every run length.
    for class in [
        HostClass::Aarch64Apple,
        HostClass::X86_64Avx2,
        HostClass::X86_64Scalar,
        HostClass::Aarch64,
        HostClass::Generic,
    ] {
        for (bytes, run_len) in [(0usize, 1usize), (4 << 20, 1), (usize::MAX >> 8, 1_000_000)] {
            assert_eq!(
                copy_ns_per_element_for_plan(class, bytes, run_len),
                copy_ns_per_element_of(class),
                "{} has no per-run term",
                class.name()
            );
        }
    }
}
