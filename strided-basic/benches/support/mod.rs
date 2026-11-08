//! Shared plumbing for the `overhead_smoke` benches (tensor4all/strided-rs#306).
//!
//! Included from `strided-basic/benches/overhead_smoke.rs` and
//! `strided-fused/benches/overhead_smoke.rs` through `#[path]`, so the two
//! benches share one copy of the counting allocator, timing, result and table
//! code without adding another crate. The crate-specific case bodies stay in
//! each bench.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Process-wide allocation counter behind [`CountingAllocator`].
pub static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

/// A counting global allocator. Each bench registers it with
/// `#[global_allocator]`.
pub struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

/// Timed samples per path per attempt. The reported time is their median and the
/// tolerance comes from their spread, so this must be large enough for a
/// percentile to mean something: 9 keeps the p95 at the second-largest sample.
pub const REPS: usize = 9;
/// Attempts before a gate may fail; a case fails only when all fail.
pub const ATTEMPTS: usize = 3;
/// A parity gate is `1 + GATE_MULTIPLE * spread`, so a run whose spread is `s`
/// tolerates a ratio `2s` above 1 before it turns red. Two spreads is what the
/// paired samples of a quiet run justify: the median moves by about the
/// half-width of the distribution, and one attempt's median is not the next's.
pub const GATE_MULTIPLE: f64 = 2.0;
/// Floor for a parity gate, for a run whose measured spread is zero or near it.
/// Without it a perfectly repeatable run would demand an exact ratio of 1.0 and
/// fail on the rounding of a single sample.
pub const GATE_FLOOR: f64 = 0.05;
/// Calls per allocation-count window (the count is exact, so this is just
/// enough to average over).
pub const ALLOC_ITERS: usize = 64;

/// How a row's ratio is judged.
#[derive(Clone, Copy)]
pub enum Gate {
    /// A and B should cost the same. The tolerance is derived from this run's
    /// own spread ([`GATE_MULTIPLE`] x the wider of the two p95-over-median
    /// ratios, never below [`GATE_FLOOR`]), so a quiet machine gets a tight gate
    /// and a noisy one does not turn white noise into a red build.
    Parity,
    /// A deliberately loose first bound on a ratio that is *not* expected to be
    /// 1 (see #306: these start loose and tighten with history). Allocations are
    /// gated as well.
    Bound(f64),
    /// Report the ratio, gate allocations only.
    ReportGated,
    /// Report only.
    Report,
}

/// Nanoseconds per iteration of `body`.
pub fn time_once(iterations: usize, body: &mut impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        body();
    }
    start.elapsed().as_secs_f64() * 1e9 / iterations as f64
}

/// Allocations per call of `body`, from the process-wide counter.
pub fn allocations_per_op(body: &mut impl FnMut()) -> f64 {
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..ALLOC_ITERS {
        body();
    }
    let after = ALLOCATIONS.load(Ordering::Relaxed);
    (after - before) as f64 / ALLOC_ITERS as f64
}

/// Value at rank `q` (0..=1) of an already sorted slice, nearest rank.
fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let last = sorted.len() - 1;
    let rank = (q * last as f64).round() as usize;
    sorted[rank.min(last)]
}

/// One row of the overhead table.
pub struct CaseResult {
    id: &'static str,
    description: &'static str,
    a_med: f64,
    a_p95: f64,
    b_med: f64,
    b_p95: f64,
    ratio: f64,
    spread: f64,
    a_allocs: f64,
    b_allocs: f64,
    gate: Gate,
    /// The parity tolerance this run measured, if the row has one.
    tolerance: Option<f64>,
    attempts_exceeded: usize,
}

impl CaseResult {
    fn gate_value(&self) -> Option<f64> {
        match self.gate {
            Gate::Parity => self.tolerance,
            Gate::Bound(bound) => Some(bound),
            Gate::ReportGated | Gate::Report => None,
        }
    }

    fn allocations_gated(&self) -> bool {
        !matches!(self.gate, Gate::Report)
    }

    fn failed(&self) -> bool {
        if self.allocations_gated() && (self.a_allocs > 0.0 || self.b_allocs > 0.0) {
            return true;
        }
        match self.gate_value() {
            Some(_) => self.attempts_exceeded == ATTEMPTS,
            None => false,
        }
    }
}

/// Measure two paths on the same input and judge their ratio.
///
/// Both paths are warmed up first, then sampled [`REPS`] times per attempt in an
/// interleaved order, so a frequency or thermal drift cannot land entirely on
/// one side of the ratio. Per attempt this records the median and p95 of each
/// path and judges `median(A) / median(B)`; the spread that decides a
/// [`Gate::Parity`] tolerance is the wider of the two `p95 / median` ratios.
///
/// A gated row fails only when all [`ATTEMPTS`] attempts exceed its bound. One
/// attempt over the bound is a blip, not a verdict: attempt-to-attempt movement
/// of the same commit is larger than the gate on a shared runner (#395 measured
/// S1 at 1.504 and 0.998 for one commit on consecutive attempts).
pub fn measure_case(
    id: &'static str,
    description: &'static str,
    iterations: usize,
    gate: Gate,
    mut path_a: impl FnMut(),
    mut path_b: impl FnMut(),
) -> CaseResult {
    // Warm up page faults, lazy kernel state and thread-local scratch outside
    // every timed and counted window.
    path_a();
    path_b();

    let mut a_samples = Vec::with_capacity(REPS);
    let mut b_samples = Vec::with_capacity(REPS);
    let mut worst_ratio = -f64::INFINITY;
    let mut reported = None;
    let mut tolerance: Option<f64> = None;
    let mut max_a_allocs: f64 = 0.0;
    let mut max_b_allocs: f64 = 0.0;
    let mut attempts_exceeded = 0;

    for _ in 0..ATTEMPTS {
        a_samples.clear();
        b_samples.clear();
        for _ in 0..REPS {
            a_samples.push(time_once(iterations, &mut path_a));
            b_samples.push(time_once(iterations, &mut path_b));
        }
        a_samples.sort_by(f64::total_cmp);
        b_samples.sort_by(f64::total_cmp);

        let a_med = percentile_sorted(&a_samples, 0.5);
        let a_p95 = percentile_sorted(&a_samples, 0.95);
        let b_med = percentile_sorted(&b_samples, 0.5);
        let b_p95 = percentile_sorted(&b_samples, 0.95);
        let ratio = a_med / b_med;
        let spread = (a_p95 / a_med).max(b_p95 / b_med) - 1.0;
        let bound = match gate {
            Gate::Parity => Some(1.0 + (GATE_MULTIPLE * spread).max(GATE_FLOOR)),
            Gate::Bound(bound) => Some(bound),
            Gate::ReportGated | Gate::Report => None,
        };

        let a_allocs = allocations_per_op(&mut path_a);
        let b_allocs = allocations_per_op(&mut path_b);
        max_a_allocs = max_a_allocs.max(a_allocs);
        max_b_allocs = max_b_allocs.max(b_allocs);
        if let Some(bound) = bound {
            if ratio > bound {
                attempts_exceeded += 1;
            }
        }
        if let Some(bound) = bound {
            tolerance = Some(tolerance.map_or(bound, |t: f64| t.max(bound)));
        }
        // Report the worst attempt: a gate that only passes on its luckiest
        // samples is the failure mode #395 is about.
        if ratio > worst_ratio {
            worst_ratio = ratio;
            reported = Some((a_med, a_p95, b_med, b_p95, spread));
        }
    }

    let (a_med, a_p95, b_med, b_p95, spread) = reported.expect("ATTEMPTS > 0");
    CaseResult {
        id,
        description,
        a_med,
        a_p95,
        b_med,
        b_p95,
        ratio: worst_ratio,
        spread,
        a_allocs: max_a_allocs,
        b_allocs: max_b_allocs,
        gate,
        tolerance,
        attempts_exceeded,
    }
}

/// Print the markdown table and append it to the GitHub step summary, if any.
pub fn print_table(cases: &[CaseResult], title: &str) {
    let mut table = String::new();
    table.push_str(
        "| id | case | reps | A med ns | A p95 ns | B med ns | B p95 ns | ratio | spread | gate | \
         A allocs/op | B allocs/op |\n",
    );
    table.push_str(
        "|----|------|-----:|---------:|---------:|---------:|---------:|------:|-------:|------|\
         -----------:|-----------:|\n",
    );
    for case in cases {
        let bound = case.gate_value();
        let gate = match case.gate {
            Gate::Report => "report".to_string(),
            Gate::ReportGated if case.failed() => "FAIL (alloc)".to_string(),
            Gate::ReportGated => "report (alloc=0)".to_string(),
            Gate::Parity if case.failed() => format!("FAIL (>{:.3})", bound.unwrap_or(0.0)),
            Gate::Parity => format!("<= {:.3} (2x spread)", bound.unwrap_or(0.0)),
            Gate::Bound(_) if case.failed() => format!("FAIL (>{})", bound.unwrap_or(0.0)),
            Gate::Bound(bound) => format!("<= {bound}"),
        };
        table.push_str(&format!(
            "| {} | {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.3} | {:.3} | {} | {:.2} | {:.2} |\n",
            case.id,
            case.description,
            REPS,
            case.a_med,
            case.a_p95,
            case.b_med,
            case.b_p95,
            case.ratio,
            case.spread,
            gate,
            case.a_allocs,
            case.b_allocs,
        ));
    }
    print!("{table}");
    let _ = std::io::stdout().flush();
    if let Ok(path) = std::env::var("GITHUB_STEP_SUMMARY") {
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = file.write_all(format!("\n{title}\n\n").as_bytes());
            let _ = file.write_all(table.as_bytes());
        }
    }
}

/// Exit non-zero when `OVERHEAD_GATE=1` and any gated row failed.
pub fn exit_if_failed(cases: &[CaseResult]) {
    let failed = cases.iter().any(CaseResult::failed);
    if failed && std::env::var_os("OVERHEAD_GATE").is_some() {
        eprintln!("overhead smoke gate failed (see table above)");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(gate: Gate, ratio: f64, exceeded: usize) -> CaseResult {
        CaseResult {
            id: "T",
            description: "test",
            a_med: 2.0,
            a_p95: 2.0,
            b_med: 1.0,
            b_p95: 1.0,
            ratio,
            spread: 0.0,
            a_allocs: 0.0,
            b_allocs: 0.0,
            gate,
            tolerance: Some(1.1),
            attempts_exceeded: exceeded,
        }
    }

    #[test]
    fn a_parity_gate_needs_every_attempt() {
        assert!(case(Gate::Parity, 2.0, ATTEMPTS).failed());
        assert!(!case(Gate::Parity, 2.0, ATTEMPTS - 1).failed());
    }

    #[test]
    fn a_bound_gate_behaves_like_the_parity_one() {
        assert!(case(Gate::Bound(2.5), 2.6, ATTEMPTS).failed());
        assert!(!case(Gate::Bound(2.5), 2.4, ATTEMPTS).failed());
    }

    #[test]
    fn reporting_rows_only_gate_allocations_when_asked() {
        assert!(!case(Gate::Report, 9.0, ATTEMPTS).failed());
        let mut gated = case(Gate::ReportGated, 1.0, 0);
        assert!(!gated.failed());
        gated.a_allocs = 1.0 / ALLOC_ITERS as f64;
        assert!(gated.failed());
        gated.a_allocs = 0.0;
        gated.b_allocs = 1.0 / ALLOC_ITERS as f64;
        assert!(gated.failed());
    }

    #[test]
    fn percentiles_use_nearest_rank() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(percentile_sorted(&sorted, 0.5), 5.0);
        assert_eq!(percentile_sorted(&sorted, 0.95), 9.0);
        assert_eq!(percentile_sorted(&sorted, 0.0), 1.0);
        assert_eq!(percentile_sorted(&sorted, 1.0), 9.0);
    }
}
