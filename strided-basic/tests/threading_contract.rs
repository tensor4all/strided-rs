//! The placement contract of [`docs/threading-contract.md`]: placement is opt-in
//! (issue #6, decision (a)), so a fan-out entry takes its `threadteam::Exec`
//! token as the last parameter and `Exec::serial()` never opens a second worker.
//! The one exception is the low-level helper `parallel_map_reduce`, which takes
//! it second; it is pinned that way too.
//!
//! Whether a *pooled* token stays within its budget is not checked here; that is
//! `execution_pool_*` and the `*_parallel` tests. These tests fail if a serial
//! token starts work off the calling thread, or if an entry listed in the
//! contract stops taking its token.
//!
//! [`docs/threading-contract.md`]: https://github.com/tensor4all/strided-rs/blob/main/docs/threading-contract.md

mod common;

use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::thread::ThreadId;
use std::time::Duration;

use strided_basic::execution::{parallel_map_reduce, parallel_width, tree_width, KernelClass};
use strided_basic::{
    add_acc, axpby_accum, axpy_acc, batched_outer_product_into, batched_outer_product_into_uninit,
    broadcast_mul_into, broadcast_mul_into_uninit, compare_into, compare_into_uninit,
    copy_conj_into, copy_into, copy_into_uninit, copy_scale_into, copy_transpose_scale_into,
    embed_diagonal_into_uninit, embed_diagonal_view_into_uninit, fma_acc, map_into,
    map_into_uninit_with_kernel, map_update_into, mul_acc, mul_into, mul_into_uninit, reduce,
    reduce_axis, reduce_axis_into, sum, symmetrize_conj_into, symmetrize_into,
    triangular_mask_into_uninit, triangular_mask_view_into_uninit, zip_map2_into, zip_map3_into,
    zip_map4_into, zip_reduce, zip_update2_into, zip_update3_into, CompareOp, CopyPlan, Exec,
    Identity, RawStridedMut, RawStridedRef, Result, StridedArray, StridedView, StridedViewMut,
    UpdatePlan,
};
use strided_perm::{
    copy_into as perm_copy_into, copy_into_col_major as perm_copy_into_col_major,
    copy_into_uninit as perm_copy_into_uninit,
};

const CLASSES: [KernelClass; 4] = [
    KernelClass::Copy,
    KernelClass::Map,
    KernelClass::Transcendental,
    KernelClass::Reduce,
];

/// The two width helpers answer `1` for a serial token whatever the size: the flat
/// entries ask `parallel_width`, the tree entries ask `tree_width`, and the gates
/// that skip them (`is_parallel`, the `exec.is_serial()` branches) check the token
/// themselves.
#[test]
fn a_serial_token_never_justifies_a_second_worker() {
    // Far above any real tensor, so the size cannot be what keeps it serial.
    let elements = usize::MAX >> 8;
    for class in CLASSES {
        assert_eq!(
            parallel_width(&Exec::serial(), class, elements).get(),
            1,
            "{class:?}: a serial token asked for a flat fan-out"
        );
        assert_eq!(
            tree_width(&Exec::serial(), class, elements).get(),
            1,
            "{class:?}: a serial token asked for a tree fan-out"
        );
    }
}

// The gate that decides a fork (`cost.rs`'s `parallel_width` and the
// `is_parallel` leaves) opens at `3 * dispatch_cost / ns_per_element`: 1732
// elements for `KernelClass::Map` against this pool's 100 ns section cost on an
// AVX-512 host (0.173 ns per element), 2995 on the generic and aarch64 rates
// (0.100). The estimate is rounded to whole nanoseconds before the comparison,
// so 2995 elements are 300 ns and clear the 300 ns that gate asks for, while
// 2994 (299 ns) does not. 512x512 is 87x that count, 151x the AVX-512 one, so
// the element count is not what decides this test; see [`Seen::record`] for the
// margin that does.
const ROWS: usize = 512;
const COLS: usize = 512;

/// How long a leaf waits for the second thread before giving up: a bound on how
/// long the test waits, not a claim about the host. A call that does fan out but
/// whose worker reaches the closure only after the bound also times out (a
/// failure injection that delays the second `record()` reproduces that), so the
/// failure message states what was observed — no second thread within the bound
/// — and not that the call stayed serial.
const SECOND_THREAD_LIMIT: Duration = Duration::from_secs(2);

/// The distinct threads the closures of one call ran on, gated on a second one
/// when `wait` is set.
struct Seen {
    threads: Mutex<Vec<ThreadId>>,
    opened: Condvar,
    open: AtomicBool,
    wait: bool,
}

impl Seen {
    fn new(wait: bool) -> Self {
        Self {
            threads: Mutex::new(Vec::new()),
            opened: Condvar::new(),
            open: AtomicBool::new(false),
            wait,
        }
    }

    /// Records the calling thread and, when gated, blocks until a second thread
    /// has recorded before letting it continue.
    ///
    /// Without the wait this test raced the pool instead of measuring
    /// placement. The shape that failed in CI (257x257, 66049 elements) runs
    /// 9.0 us on one thread at the 0.137 ns per element measured on an AVX-512
    /// host, so even a quarter of it is only ~2-3 us of work per leaf, while a
    /// worker needed 7-40 us to pick up the sibling (p50 8, p90 14, max 40 over 40 fresh pools
    /// on a loaded 32-core host): the calling thread could finish the whole
    /// region itself, and the same commit passed or failed with the host state.
    /// Waiting makes "a four-slot token opens a second worker" an assertion
    /// about the placement decision: it fails only when no second thread arrives
    /// within [`SECOND_THREAD_LIMIT`].
    fn record(&self) {
        if self.wait && self.open.load(Ordering::Relaxed) {
            return;
        }
        let id = std::thread::current().id();
        let mut threads = self.threads.lock().unwrap();
        if !threads.contains(&id) {
            threads.push(id);
            if threads.len() > 1 {
                self.open.store(true, Ordering::Relaxed);
            }
            self.opened.notify_all();
        }
        if !self.wait || threads.len() > 1 {
            return;
        }
        let (threads, timeout) = self
            .opened
            .wait_timeout_while(threads, SECOND_THREAD_LIMIT, |threads| threads.len() < 2)
            .unwrap();
        if timeout.timed_out() {
            // Let the remaining elements through, so one failure is reported
            // instead of every element waiting out the limit.
            self.open.store(true, Ordering::Relaxed);
            panic!(
                "no second thread within {SECOND_THREAD_LIMIT:?}: the closures stayed on {threads:?}"
            );
        }
    }

    fn threads(self) -> Vec<ThreadId> {
        self.threads.into_inner().unwrap()
    }
}

fn source() -> StridedArray<f64> {
    StridedArray::from_fn_col_major(&[ROWS, COLS], |index| (index[0] * COLS + index[1]) as f64)
}

fn run_map(exec: Exec<'_>, wait: bool) -> Vec<ThreadId> {
    let src = source();
    let mut dest = StridedArray::<f64>::col_major(&[ROWS, COLS]);
    let seen = Seen::new(wait);
    map_into(
        &mut dest.view_mut(),
        &src.view(),
        |x: f64| {
            seen.record();
            x * 2.0
        },
        exec,
    )
    .unwrap();
    seen.threads()
}

fn run_zip(exec: Exec<'_>, wait: bool) -> Vec<ThreadId> {
    let a = source();
    let b = source();
    let mut dest = StridedArray::<f64>::col_major(&[ROWS, COLS]);
    let seen = Seen::new(wait);
    zip_map2_into(
        &mut dest.view_mut(),
        &a.view(),
        &b.view(),
        |x: f64, y: f64| {
            seen.record();
            x + y
        },
        exec,
    )
    .unwrap();
    seen.threads()
}

#[test]
fn a_serial_token_runs_the_observed_entries_on_the_calling_thread() {
    let caller = std::thread::current().id();
    assert_eq!(
        run_map(Exec::serial(), false),
        [caller],
        "map_into left the calling thread under Exec::serial()"
    );
    assert_eq!(
        run_zip(Exec::serial(), false),
        [caller],
        "zip_map2_into left the calling thread under Exec::serial()"
    );
}

/// The same shapes do fan out when the caller hands in a pool, so the serial
/// assertion above is not vacuous.
///
/// The shapes are four times the work of the shape that failed under this
/// assertion in CI (#382) and 151x the element count at which the fork gate
/// opens; the run also waits for the second thread rather than racing it, which
/// is what makes the outcome independent of the host.
#[cfg(feature = "parallel")]
#[test]
fn a_pooled_token_opens_a_second_worker() {
    for threads in [
        run_map(Exec::new(common::shared_pool(), 4).unwrap(), true),
        run_zip(Exec::new(common::shared_pool(), 4).unwrap(), true),
    ] {
        assert!(
            threads.len() > 1,
            "a four-slot token over {ROWS}x{COLS} stayed on one thread: {threads:?}"
        );
    }
}

/// The typed entries of `strided-basic` and `strided-perm` that can fan out, each
/// pinned to take its token last — the same list `docs/threading-contract.md`
/// enumerates.
///
/// Every line fails to compile if the entry drops its trailing `Exec`, which is
/// the contract `docs/threading-contract.md` states. `impl Fn` arguments cannot be
/// named in a function-pointer type, so each entry is forwarded by a non-capturing
/// closure. The closures are never called: only their type is checked, so this
/// needs no tensor setup and no runtime.
///
/// The remaining plans and the erased entries of the other crates are not listed
/// here; each is replayed with a token by the tests that own it, so dropping the
/// parameter fails those instead.
#[test]
fn every_fan_out_entry_takes_its_exec_token_last() {
    type V<'a> = StridedView<'a, f64>;
    type M<'a> = StridedViewMut<'a, f64>;

    macro_rules! pins {
        ($($entry:expr => $ty:ty;)*) => {$(
            let _: $ty = $entry;
        )*};
    }

    pins! {
        |d, s, e| map_into(d, s, |x: f64| x, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, s, e| map_into_uninit_with_kernel(d, s, |_: &[f64], _: &mut [MaybeUninit<f64>]| false, |x: f64| x, e)
            => for<'a, 'b, 'e> fn(&'a mut StridedViewMut<'b, MaybeUninit<f64>>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, a, b, e| zip_map2_into(d, a, b, |x: f64, y: f64| x + y, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, a, b, c, e| zip_map3_into(d, a, b, c, |x: f64, y: f64, z: f64| x + y + z, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, &'a V<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, a, b, c, f, e| zip_map4_into(d, a, b, c, f, |x: f64, y: f64, z: f64, w: f64| x + y + z + w, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, &'a V<'b>, &'a V<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, a, b, e| mul_into(d, a, b, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, a, b, e| mul_into_uninit(d, a, b, e)
            => for<'a, 'b, 'e> fn(&'a mut StridedViewMut<'b, MaybeUninit<f64>>, &'a V<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, a, axes_a, b, axes_b, e| broadcast_mul_into(d, a, axes_a, b, axes_b, e)
            => for<'a, 'b, 'c, 'e> fn(&'a mut M<'b>, &'a V<'b>, &'a [usize], &'c V<'b>, &'c [usize], Exec<'e>) -> Result<()>;
        |d, a, axes_a, b, axes_b, e| broadcast_mul_into_uninit(d, a, axes_a, b, axes_b, e)
            => for<'a, 'b, 'c, 'e> fn(&'a mut StridedViewMut<'b, MaybeUninit<f64>>, &'a V<'b>, &'a [usize], &'c V<'b>, &'c [usize], Exec<'e>) -> Result<()>;
        |d, a, b, e| compare_into(d, a, b, CompareOp::Lt, e)
            => for<'a, 'b, 'e> fn(&'a mut StridedViewMut<'b, bool>, &'a V<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, a, b, e| compare_into_uninit(d, a, b, CompareOp::Lt, e)
            => for<'a, 'b, 'e> fn(&'a mut StridedViewMut<'b, MaybeUninit<bool>>, &'a V<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, a, b, lhs_free, rhs_free, e| batched_outer_product_into(d, a, b, lhs_free, rhs_free, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, &'a V<'b>, usize, usize, Exec<'e>) -> Result<()>;
        |d, a, b, lhs_free, rhs_free, e| batched_outer_product_into_uninit(d, a, b, lhs_free, rhs_free, e)
            => for<'a, 'b, 'e> fn(&'a mut StridedViewMut<'b, MaybeUninit<f64>>, &'a V<'b>, &'a V<'b>, usize, usize, Exec<'e>) -> Result<()>;
        |d, e| map_update_into::<f64, Identity>(d, |x: f64| x + 1.0, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, Exec<'e>) -> Result<()>;
        |d, a, e| zip_update2_into::<f64, f64, Identity, Identity>(d, a, |x: f64, y: f64| x + y, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, a, b, e| zip_update3_into::<f64, f64, f64, Identity, Identity, Identity>(d, a, b, |x: f64, y: f64, z: f64| x + y + z, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |s, e| reduce(s, |x: f64| x, |x: f64, y: f64| x + y, 0.0f64, e)
            => for<'a, 'b, 'e> fn(&'a V<'b>, Exec<'e>) -> Result<f64>;
        |a, b, e| zip_reduce(a, b, |x: f64, y: f64| x + y, |x: f64, y: f64| x + y, 0.0f64, e)
            => for<'a, 'b, 'e> fn(&'a V<'b>, &'a V<'b>, Exec<'e>) -> Result<f64>;
        |s, axis, e| reduce_axis(s, axis, |x: f64| x, |x: f64, y: f64| x + y, 0.0f64, e)
            => for<'a, 'b, 'e> fn(&'a V<'b>, usize, Exec<'e>) -> Result<StridedArray<f64>>;
        |d, s, axis, e| reduce_axis_into(d, s, axis, |x: f64| x, |x: f64, y: f64| x + y, 0.0f64, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, usize, Exec<'e>) -> Result<()>;
        |s, e| sum(s, e)
            => for<'a, 'b, 'e> fn(&'a V<'b>, Exec<'e>) -> Result<f64>;
        |d, s, e| copy_into(d, s, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, s, e| copy_into_uninit(d, s, e)
            => for<'a, 'b, 'e> fn(&'a mut StridedViewMut<'b, MaybeUninit<f64>>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, s, scale, e| copy_scale_into(d, s, scale, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, f64, Exec<'e>) -> Result<()>;
        |d, s, scale, e| copy_transpose_scale_into(d, s, scale, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, f64, Exec<'e>) -> Result<()>;
        |d, s, e| copy_conj_into(d, s, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, s, e| perm_copy_into(d, s, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, s, e| perm_copy_into_uninit(d, s, e)
            => for<'a, 'b, 'e> fn(&'a mut StridedViewMut<'b, MaybeUninit<f64>>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, s, e| perm_copy_into_col_major(d, s, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, s, e| add_acc(d, s, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, s, e| mul_acc(d, s, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, s, alpha, e| axpy_acc(d, s, alpha, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, f64, Exec<'e>) -> Result<()>;
        |d, a, b, e| fma_acc(d, a, b, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, s, e| symmetrize_into(d, s, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |d, s, e| symmetrize_conj_into(d, s, e)
            => for<'a, 'b, 'e> fn(&'a mut M<'b>, &'a V<'b>, Exec<'e>) -> Result<()>;
        |y, x, alpha, beta, e| axpby_accum(y, x, alpha, beta, e)
            => for<'a, 'b, 'e> fn(&'a mut [f64], &'b [f64], f64, f64, Exec<'e>) -> Result<()>;
        |o, i, shape, k, upper, fill, e| triangular_mask_into_uninit(o, i, shape, k, upper, fill, e)
            => for<'a, 'b, 'c, 'e> fn(&'a mut [MaybeUninit<f64>], &'b [f64], &'c [usize], i64, bool, f64, Exec<'e>) -> Result<()>;
        |o, s, k, upper, fill, e| triangular_mask_view_into_uninit(o, s, k, upper, fill, e)
            => for<'a, 'b, 'e> fn(&'a mut [MaybeUninit<f64>], &'b V<'b>, i64, bool, f64, Exec<'e>) -> Result<()>;
        |o, i, shape, axis, insert, zero, e| embed_diagonal_into_uninit(o, i, shape, axis, insert, zero, e)
            => for<'a, 'b, 'c, 'e> fn(&'a mut [MaybeUninit<f64>], &'b [f64], &'c [usize], usize, usize, f64, Exec<'e>) -> Result<()>;
        |o, s, axis, insert, zero, e| embed_diagonal_view_into_uninit(o, s, axis, insert, zero, e)
            => for<'a, 'b, 'e> fn(&'a mut [MaybeUninit<f64>], &'b V<'b>, usize, usize, f64, Exec<'e>) -> Result<()>;
        |p, d, s, e| p.execute(d, s, e)
            => for<'a, 'b, 'c, 'd, 'e> fn(&'a CopyPlan, &'b mut RawStridedMut<'c, f64>, &'d RawStridedRef<'c, f64>, Exec<'e>) -> Result<()>;
        |p, d, s, e| p.execute_uninit(d, s, e)
            => for<'a, 'b, 'c, 'd, 'e> fn(&'a CopyPlan, &'b mut RawStridedMut<'c, MaybeUninit<f64>>, &'d RawStridedRef<'c, f64>, Exec<'e>) -> Result<()>;
        |p, d, s, scale, e| p.execute_scale(d, s, scale, e)
            => for<'a, 'b, 'c, 'd, 'e> fn(&'a CopyPlan, &'b mut RawStridedMut<'c, f64>, &'d RawStridedRef<'c, f64>, f64, Exec<'e>) -> Result<()>;
        |p, d, s, e| p.execute_conj(d, s, e)
            => for<'a, 'b, 'c, 'd, 'e> fn(&'a CopyPlan, &'b mut RawStridedMut<'c, f64>, &'d RawStridedRef<'c, f64>, Exec<'e>) -> Result<()>;
        |p, d, s, alpha, beta, e| p.execute(d, s, alpha, beta, |x: f64| x, e)
            => for<'a, 'b, 'c, 'd, 'e> fn(&'a UpdatePlan, &'b mut RawStridedMut<'c, f64>, &'d RawStridedRef<'c, f64>, f64, f64, Exec<'e>) -> Result<()>;
        |p, d, s, alpha, e| p.execute_uninit(d, s, alpha, |x: f64| x, e)
            => for<'a, 'b, 'c, 'd, 'e> fn(&'a UpdatePlan, &'b mut RawStridedMut<'c, MaybeUninit<f64>>, &'d RawStridedRef<'c, f64>, f64, Exec<'e>) -> Result<()>;
        // The exception: the low-level map-reduce helper takes its token second.
        |range, e| parallel_map_reduce(range, e, &|r: std::ops::Range<usize>| r.len(), &|a: usize, b: usize| a + b)
            => for<'e> fn(std::ops::Range<usize>, Exec<'e>) -> usize;
    }
}
