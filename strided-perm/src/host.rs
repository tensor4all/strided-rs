//! Host class and the serial copy rate of each class.
//!
//! The rates of the strided stack are keyed by the kernel dispatch the machine
//! takes, not by the CPU model. `HostClass` and `HostClass::detect` are the same
//! enum and the same detection as in `threadteam` (`threadteam::HostClass`), the
//! `tcontract` and the `faer` cost tables, and are kept in sync with them by
//! hand; this crate owns the copy of the strided workspace (`strided-basic`
//! reuses it for its own table, which sits above this crate).

use std::sync::OnceLock;

/// The class of the machine, keyed by the kernel dispatch it will take.
///
/// Kept in sync with `threadteam::HostClass`. Detected once per process by
/// [`HostClass::detect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HostClass {
    /// `x86_64` with AVX-512F at run time (`x86_64-avx512`).
    X86_64Avx512,
    /// `x86_64` with AVX2 and FMA, without AVX-512F (`x86_64-avx2`).
    X86_64Avx2,
    /// Any other `x86_64` (`x86_64-scalar`).
    X86_64Scalar,
    /// `aarch64` on macOS (`aarch64-apple`).
    Aarch64Apple,
    /// Any other `aarch64` (`aarch64`).
    Aarch64,
    /// Everything else (`generic`).
    Generic,
}

impl HostClass {
    /// The class of this host, detected at the first call and cached
    /// (`OnceLock`; `is_x86_feature_detected!` on x86_64).
    pub fn detect() -> Self {
        static CLASS: OnceLock<HostClass> = OnceLock::new();
        *CLASS.get_or_init(Self::detect_uncached)
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_uncached() -> Self {
        if std::arch::is_x86_feature_detected!("avx512f") {
            HostClass::X86_64Avx512
        } else if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            HostClass::X86_64Avx2
        } else {
            HostClass::X86_64Scalar
        }
    }

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    fn detect_uncached() -> Self {
        HostClass::Aarch64Apple
    }

    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    fn detect_uncached() -> Self {
        HostClass::Aarch64
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn detect_uncached() -> Self {
        HostClass::Generic
    }

    /// The class name used in the docs and the tables.
    pub const fn name(self) -> &'static str {
        match self {
            HostClass::X86_64Avx512 => "x86_64-avx512",
            HostClass::X86_64Avx2 => "x86_64-avx2",
            HostClass::X86_64Scalar => "x86_64-scalar",
            HostClass::Aarch64Apple => "aarch64-apple",
            HostClass::Aarch64 => "aarch64",
            HostClass::Generic => "generic",
        }
    }
}

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
mod tests {
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
}
