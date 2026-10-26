//! Serial cost classes and the parallel gate of the strided kernels.
//!
//! A kernel does not own a minimum length. It states how long its region takes
//! on one thread (elements times the serial throughput of its class) and asks
//! the placement token whether that pays: a region runs in parallel when the
//! serial time is at least [`threadteam::PAYS`] times the dispatch cost of every
//! section it opens (`Exec::pays`). The cost of a section belongs to the token
//! and the thread that calls, not to this crate; the design record is
//! `threadteam`'s `docs/design/thresholds.md`.
//!
//! # The class table
//!
//! This table is the only place the throughputs live (the `Copy` row is owned by
//! `strided-perm`, which sits below this crate). Every kernel of the workspace is
//! one of four classes, and the rate of a class depends on the *host class*
//! (`strided_perm::HostClass`, keyed by the kernel dispatch the machine takes,
//! detected once at first use; the enum and `detect()` are kept in sync with
//! `threadteam`'s). Serial, `f64`, in nanoseconds per element:
//!
//! | [`KernelClass`] | `x86_64-avx512` | `aarch64-apple` | `generic` (every other class) | kernels |
//! | --- | --- | --- | --- | --- |
//! | `Copy` | 0.171 | 0.090 | 0.090 | copies, fills, permutations, gathers, static indexing, transposes |
//! | `Map` | 0.173 | 0.100 | 0.100 | element-wise map, zip, add, mul, axpy, fma, in-place update, fused arithmetic chains |
//! | `Transcendental` | 5.60 | 1.38 | 1.38 | `exp`, `ln`, `tanh`, other libm-like functions (fused chains that contain one, softmax) |
//! | `Reduce` | 0.104 | 0.081 | 0.081 | sum, dot, reduce, zip-reduce, axis reductions, norms, scans and arg-reductions |
//!
//! * `aarch64-apple`: one Apple M5 Max core, memory resident (thresholds.md
//!   section 3.2).
//! * `x86_64-avx512`: an AMD Ryzen AI 9 HX 470 at 2.0 GHz (boost off), from the
//!   `seq` column of the strided probes (`copy`, `map`, `exp`, `sum`) in
//!   `faer-benchmark` `data/probes/20261026-awake-team-linux/tables-confirm-exec3.md`
//!   (width 8 table): the median over the L2-resident sizes 16384, 32768 and
//!   65536 elements of `seq us * 1000 / elements`. Copy 2.80, 4.29, 13.17 us give
//!   0.171, 0.131, 0.201; map 2.84, 5.21, 15.39 us give 0.173, 0.159, 0.235; exp
//!   91.77, 180, 419 us give 5.60, 5.49, 6.39; sum 1.71, 3.55, 6.65 us give 0.104,
//!   0.108, 0.102.
//! * `generic` (`x86_64-avx2`, `x86_64-scalar`, `aarch64`, `generic`: no
//!   measurement): the fastest known value of each row, the minimum over the
//!   measured classes. It is the conservative choice: a rate that is too small
//!   keeps work serial.
//!
//! The rates of a class and the dispatch costs of the same class in `threadteam`
//! come from the same machine in the same clock state: boost on or off moves both
//! by about 1.66x and leaves their ratio stable within a few percent
//! (`raw-exec3-boost`, `tables-*-exec3-boost.md`).
//!
//! A reduction counts the source elements it reads (outputs times reduced
//! elements). A kernel with a cost not in the table is classified by the nearest
//! row, never given a constant of its own.
//!
//! # Sections
//!
//! `sections` is the number of dispatches the entry performs, counted truthfully:
//!
//! * A *flat* entry (`Exec::for_ranges`: the copy and fill helpers, the
//!   permutation engine, the axis-chunked kernels) publishes the region once:
//!   1 section at the chosen width.
//! * A *tree* entry (the recursive halving of `parallel_map_reduce` and of the
//!   dimension splitter) forks twice per level and the children fork again on
//!   half of the budget, so the critical path holds `ceil(log2 width)` sections.
//!   The splitter gates every node on its own fork (1 section at width 2) because
//!   each child decides again for the work it has left.

use core::num::NonZeroUsize;
use core::time::Duration;
use strided_perm::{copy_ns_per_element_of, HostClass};
use threadteam::Exec;

/// Serial cost class of a kernel; see the module documentation of `strided-basic/src/cost.rs` for the
/// table.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::KernelClass;
///
/// assert!(KernelClass::Transcendental.ns_per_element() > KernelClass::Map.ns_per_element());
/// // 1e6 elements at the rate of this host's class.
/// let ns = KernelClass::Map.ns_per_element() * 1e6;
/// assert_eq!(KernelClass::Map.estimate(1_000_000).as_nanos() as f64, ns.round());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelClass {
    /// Copies, fills, permutations and index gathers.
    Copy,
    /// Element-wise arithmetic and fused arithmetic chains.
    Map,
    /// `exp`-like functions.
    Transcendental,
    /// Reductions and line kernels that read each source element once.
    Reduce,
}

impl KernelClass {
    /// Serial throughput of the class in nanoseconds per element on this host
    /// ([`ns_per_element_of`](Self::ns_per_element_of) of `HostClass::detect()`).
    pub fn ns_per_element(self) -> f64 {
        self.ns_per_element_of(HostClass::detect())
    }

    /// Serial throughput of the class in nanoseconds per element on a host class
    /// (the table in the module documentation).
    pub const fn ns_per_element_of(self, host: HostClass) -> f64 {
        match host {
            HostClass::X86_64Avx512 => match self {
                KernelClass::Copy => copy_ns_per_element_of(host),
                KernelClass::Map => 0.173,
                KernelClass::Transcendental => 5.60,
                KernelClass::Reduce => 0.104,
            },
            HostClass::Aarch64Apple => match self {
                KernelClass::Copy => copy_ns_per_element_of(host),
                KernelClass::Map => 0.100,
                KernelClass::Transcendental => 1.38,
                KernelClass::Reduce => 0.081,
            },
            HostClass::X86_64Avx2
            | HostClass::X86_64Scalar
            | HostClass::Aarch64
            | HostClass::Generic => match self {
                KernelClass::Copy => copy_ns_per_element_of(HostClass::Generic),
                KernelClass::Map => 0.100,
                KernelClass::Transcendental => 1.38,
                KernelClass::Reduce => 0.081,
            },
        }
    }

    /// Serial time of `elements` elements of this class.
    pub fn estimate(self, elements: usize) -> Duration {
        Duration::from_secs_f64(elements as f64 * self.ns_per_element() * 1e-9)
    }
}

const ONE: NonZeroUsize = NonZeroUsize::MIN;

/// Width of a flat entry (one section) that `exec` should use for `elements`
/// elements of `class`: the width in `1..=budget` that minimises the section
/// cost plus the share of the work, `1` when no width pays.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::{parallel_width, KernelClass};
/// use strided_basic::Exec;
///
/// assert_eq!(parallel_width(&Exec::serial(), KernelClass::Copy, usize::MAX >> 8).get(), 1);
/// ```
pub fn parallel_width(exec: &Exec<'_>, class: KernelClass, elements: usize) -> NonZeroUsize {
    if exec.is_serial() {
        return ONE;
    }
    exec.width_for(class.estimate(elements), 1)
}

/// Width of a tree entry (`parallel_map_reduce`): the width in `1..=budget`
/// that minimises `ceil(log2 k) * D(k) + estimate / k` among the widths at which
/// `ceil(log2 k)` sections pay, `1` when none does.
///
/// # Examples
///
/// ```
/// use strided_basic::execution::{tree_width, KernelClass};
/// use strided_basic::Exec;
///
/// assert_eq!(tree_width(&Exec::serial(), KernelClass::Reduce, usize::MAX >> 8).get(), 1);
/// ```
pub fn tree_width(exec: &Exec<'_>, class: KernelClass, elements: usize) -> NonZeroUsize {
    if exec.is_serial() {
        return ONE;
    }
    let estimate = class.estimate(elements);
    let mut best = (ONE, estimate);
    for k in 2..=exec.budget().get() {
        let width = NonZeroUsize::new(k).unwrap_or(ONE);
        let sections = k.next_power_of_two().trailing_zeros();
        if !exec.pays(estimate, sections, width) {
            continue;
        }
        let time = estimate / u32::try_from(k).unwrap_or(u32::MAX)
            + exec.dispatch_cost(width).saturating_mul(sections);
        if time < best.1 {
            best = (width, time);
        }
    }
    best.0
}

/// Narrows `exec` to the width a flat entry over `len` elements of `class` justifies.
pub(crate) fn for_len<'a>(exec: Exec<'a>, class: KernelClass, len: usize) -> Exec<'a> {
    let width = parallel_width(&exec, class, len);
    exec.limit(width)
}

/// Narrows `exec` to the width a tree entry over `len` elements of `class` justifies.
pub(crate) fn for_len_tree<'a>(exec: Exec<'a>, class: KernelClass, len: usize) -> Exec<'a> {
    let width = tree_width(&exec, class, len);
    exec.limit(width)
}

/// Whether one fork of `exec` over `len` elements of `class` pays: the gate of a
/// node of the dimension splitter and of the entries that call it.
pub(crate) fn is_parallel(class: KernelClass, len: usize, exec: &Exec<'_>) -> bool {
    if exec.is_serial() {
        return false;
    }
    let two = NonZeroUsize::new(2).unwrap_or(ONE);
    exec.pays(class.estimate(len), 1, two.min(exec.budget()))
}
