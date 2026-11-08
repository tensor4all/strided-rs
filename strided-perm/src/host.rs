//! Host class and the serial copy rate of each class.
//!
//! The rates of the strided stack are keyed by the kernel dispatch the machine
//! takes, not by the CPU model. [`HostClass`] is [`threadteam::HostClass`]
//! re-exported, so the class this crate reports is the one the placement token
//! uses; only the copy rate below is local (`strided-basic` reuses it for its own
//! table, which sits above this crate).

/// The class of the machine, keyed by the kernel dispatch it will take.
///
/// [`threadteam::HostClass`] re-exported, so the enum and its `detect` are the
/// ones the placement token uses and cannot drift from them. The upstream doc
/// below still names `strided-rs` among the crates that keep a hand-synced copy;
/// this crate no longer does.
///
/// # Compatibility
///
/// This is a **type-identity change**, not a rename: `strided_perm::HostClass`
/// is now the same type as `threadteam::HostClass` instead of a distinct copy.
/// A downstream crate that implemented its own trait for both types compiles
/// against the old pair and fails with `E0119` (conflicting implementations)
/// against this one. Code that names the type, compares it, or matches on its
/// variants is unaffected.
pub use threadteam::HostClass;

/// Serial throughput of a copy in nanoseconds per element, of a host class.
///
/// * `x86_64-avx512`: 0.171, the median of `seq` time per element of the strided
///   `copy` probe at 16384, 32768 and 65536 elements (L2 resident; 0.171, 0.131,
///   0.201) on an AMD Ryzen AI 9 HX 470 at 2.0 GHz (boost off), `faer-benchmark`
///   `data/probes/20261026-awake-team-linux/tables-confirm-exec3.md`.
/// * `aarch64-apple`: 0.090, Apple M5 Max (thresholds.md section 3.2).
/// * every other class: the `generic` row, the fastest known value (the minimum
///   over the measured classes), which keeps work serial: 0.090.
///
/// A row's rates and the dispatch costs of the same class in `threadteam` come
/// from the same machine in the same clock state (boost on or off moves both by
/// about 1.66x and keeps their ratio within a few percent).
pub const fn copy_ns_per_element_of(class: HostClass) -> f64 {
    match class {
        HostClass::X86_64Avx512 => 0.171,
        HostClass::Aarch64Apple => 0.090,
        HostClass::X86_64Avx2
        | HostClass::X86_64Scalar
        | HostClass::Aarch64
        | HostClass::Generic => 0.090,
    }
}

/// Serial throughput of a copy in nanoseconds per element on this host
/// ([`copy_ns_per_element_of`] of [`HostClass::detect`]).
///
/// The `Copy` class of the strided stack (`strided_basic::execution::KernelClass`
/// holds the class table; this crate sits below it and owns the one rate it
/// needs). `copy_into*` multiply it by the element count and ask the token
/// whether one section pays.
pub fn copy_ns_per_element() -> f64 {
    copy_ns_per_element_of(HostClass::detect())
}

#[cfg(test)]
mod tests;
