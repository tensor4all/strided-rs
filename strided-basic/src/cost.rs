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
//! This table is the only place the throughputs live. Every kernel of the
//! workspace is one of four classes (measured serially on one Apple M5 Max
//! core, `f64`, memory resident, see thresholds.md section 3.2):
//!
//! | [`KernelClass`] | ns per element | kernels |
//! | --- | --- | --- |
//! | `Copy` | 0.090 | copies, fills, permutations, gathers, static indexing, transposes |
//! | `Map` | 0.100 | element-wise map, zip, add, mul, axpy, fma, in-place update, fused arithmetic chains |
//! | `Transcendental` | 1.38 | `exp`, `ln`, `tanh`, other libm-like functions (fused chains that contain one, softmax) |
//! | `Reduce` | 0.081 | sum, dot, reduce, zip-reduce, axis reductions, norms, scans and arg-reductions |
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
/// assert_eq!(KernelClass::Map.estimate(1_000_000).as_micros(), 100);
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
    /// Serial throughput of the class in nanoseconds per element.
    pub const fn ns_per_element(self) -> f64 {
        match self {
            KernelClass::Copy => strided_perm::COPY_NS_PER_ELEMENT,
            KernelClass::Map => 0.100,
            KernelClass::Transcendental => 1.38,
            KernelClass::Reduce => 0.081,
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
