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
/// tolerance comes from their spread. With 9 samples the nearest-rank p95 is the
/// largest sample and the median is the 5th, so the spread is computed from the
/// paired ratios rather than from p95.
pub const REPS: usize = 9;
/// Attempts before a gate may fail; a case fails only when all fail.
pub const ATTEMPTS: usize = 3;
/// `MAD` scaled to a standard deviation for a normal distribution
/// (`1 / Phi^-1(0.75)`). The ruling for #395 requires MAD rather than IQR: at
/// n ~ 5 an IQR is zero or jumps to a neighbour, while a scaled MAD stays
/// defined.
pub const MAD_TO_SIGMA: f64 = 1.4826;
/// A parity gate is `PARITY_ANCHOR + SIGMA_MULTIPLE * width`, computed from this
/// run's own paired samples: the width is derived from the run, not from a
/// constant baked in from another machine.
pub const SIGMA_MULTIPLE: f64 = 3.0;
/// The ratio a [`Gate::Parity`] row asserts: A and B cost the same.
pub const PARITY_ANCHOR: f64 = 1.0;
/// Floor on the width, as a fraction of the median. A perfectly repeatable run
/// would otherwise get a zero-width gate and fail on the rounding of one sample.
pub const GATE_FLOOR_FRACTION: f64 = 0.15;
/// Calls per allocation-count window (the count is exact, so this is just
/// enough to average over).
pub const ALLOC_ITERS: usize = 64;

/// How a row's ratio is judged.
///
/// The two benches share this file but gate different rows, so each compilation
/// uses a subset of these variants.
#[allow(dead_code)] // shared by two bench binaries; neither uses every variant
#[derive(Clone, Copy)]
pub enum Gate {
    /// A and B should cost the same. The bound is
    /// `1 + 3 * max(1.4826 * MAD, 0.15 * median)` over this attempt's
    /// paired ratios, so a quiet machine gets a tight gate and a run that is
    /// already swinging widens its own, instead of turning the runner's noise
    /// into a red build.
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

/// Width of a [`Gate::Parity`] tolerance: `max(1.4826 * MAD, 0.15 * median)`.
///
/// The floor is what keeps a run whose MAD measures exactly zero from getting a
/// zero-width gate: with `median > 0` the width is at least `0.15 * median`, so
/// the bound is at least `1 + 3 * 0.15 * median` and a repeatable row is still
/// allowed to move. Above parity that bound rises with the median, so a steady
/// ratio of `m` passes while `m <= 1 + 0.45 * m`, i.e. up to `m = 1.82`; see the
/// #395 worklog for why that band is the owner's decision. A `median` of zero means the timer measured nothing, which no row in
/// these benches can produce (`time_once` divides elapsed nanoseconds, not a
/// tick count).
fn parity_width(median: f64, mad: f64) -> f64 {
    (MAD_TO_SIGMA * mad).max(GATE_FLOOR_FRACTION * median)
}

/// Bound of a [`Gate::Parity`] row: `1 + 3 * width`.
///
/// The anchor is 1.0 — "the two paths cost the same" — not the measured median.
/// Anchoring at the median would let a *steady* regression admit itself: nine
/// identical ratios of 2.0 give `MAD = 0`, so the width is `0.15 * 2.0 = 0.3` and
/// a `median + 3 * width` bound of 2.9 would pass a row that is twice as slow as
/// it is allowed to be. With the 1.0 anchor the same row gets `1 + 0.9 = 1.9` and
/// fails, and a row sitting at parity gets `1 + 3 * max(1.4826 * MAD, 0.15)`,
/// i.e. ~1.45 for any median near 1.
///
/// MAD enters here and only here. [`Gate::Bound`] carries an absolute limit that
/// no dispersion measurement can move.
fn parity_bound(median: f64, mad: f64) -> f64 {
    PARITY_ANCHOR + SIGMA_MULTIPLE * parity_width(median, mad)
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
    spread: Option<f64>,
    mad: f64,
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
            // `ratio` is the worst attempt's median and `bound` is that same
            // attempt's bound, so this is "the worst attempt is over its limit".
            // One attempt over the limit is a blip; all of them is a verdict.
            Some(bound) => self.attempts_exceeded == ATTEMPTS && self.ratio > bound,
            None => false,
        }
    }
}

/// One attempt's verdict: the statistic, its width, and whether it was over.
struct Attempt {
    ratio: f64,
    mad: f64,
    width: Option<f64>,
    bound: Option<f64>,
    exceeded: bool,
}

/// Judge one attempt from its paired ratios.
///
/// This is the whole statistic-to-verdict path, split out so a test can drive it
/// with fixed samples instead of wall-clock readings: a timing-based test can
/// itself flake, because a genuinely 8x workload can measure as a legitimately
/// admitted ratio.
fn attempt_verdict(ratios: &[f64], gate: Gate) -> Attempt {
    let mut sorted = ratios.to_vec();
    sorted.sort_by(f64::total_cmp);
    let ratio = percentile_sorted(&sorted, 0.5);
    let mut deviations: Vec<f64> = sorted.iter().map(|r| (r - ratio).abs()).collect();
    deviations.sort_by(f64::total_cmp);
    let mad = percentile_sorted(&deviations, 0.5);
    let (width, bound) = match gate {
        Gate::Parity => {
            let width = parity_width(ratio, mad);
            (Some(width), Some(parity_bound(ratio, mad)))
        }
        Gate::Bound(bound) => (None, Some(bound)),
        Gate::ReportGated | Gate::Report => (None, None),
    };
    Attempt {
        ratio,
        mad,
        width,
        bound,
        exceeded: bound.is_some_and(|bound| ratio > bound),
    }
}

/// Measure two paths on the same input and judge their ratio.
///
/// Both paths are warmed up first, then sampled [`REPS`] times per attempt in an
/// interleaved order, so a frequency or thermal drift cannot land entirely on
/// one side of the ratio. The statistic is the median of the **paired** ratios
/// `A_i / B_i`, and a [`Gate::Parity`] width comes from the MAD of those same
/// ratios. The per-path medians and p95 values are recorded for diagnosis, not
/// for the verdict.
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
    let mut ratios = Vec::with_capacity(REPS);
    let mut worst_ratio = -f64::INFINITY;
    let mut reported = None;
    let mut tolerance: Option<f64> = None;
    let mut max_a_allocs: f64 = 0.0;
    let mut max_b_allocs: f64 = 0.0;
    let mut attempts_exceeded = 0;

    for _ in 0..ATTEMPTS {
        a_samples.clear();
        b_samples.clear();
        ratios.clear();
        for _ in 0..REPS {
            let a = time_once(iterations, &mut path_a);
            let b = time_once(iterations, &mut path_b);
            a_samples.push(a);
            b_samples.push(b);
            ratios.push(a / b);
        }
        a_samples.sort_by(f64::total_cmp);
        b_samples.sort_by(f64::total_cmp);

        let a_med = percentile_sorted(&a_samples, 0.5);
        let a_p95 = percentile_sorted(&a_samples, 0.95);
        let b_med = percentile_sorted(&b_samples, 0.5);
        let b_p95 = percentile_sorted(&b_samples, 0.95);
        let verdict = attempt_verdict(&ratios, gate);
        let (ratio, mad) = (verdict.ratio, verdict.mad);
        let (spread, bound) = (verdict.width, verdict.bound);
        if verdict.exceeded {
            attempts_exceeded += 1;
        }

        let a_allocs = allocations_per_op(&mut path_a);
        let b_allocs = allocations_per_op(&mut path_b);
        max_a_allocs = max_a_allocs.max(a_allocs);
        max_b_allocs = max_b_allocs.max(b_allocs);
        // Report the worst attempt, and the bound that attempt was judged
        // against: a gate that only passes on its luckiest samples is the
        // failure mode #395 is about.
        if ratio > worst_ratio {
            worst_ratio = ratio;
            tolerance = bound;
            reported = Some((a_med, a_p95, b_med, b_p95, spread, mad));
        }
    }

    let (a_med, a_p95, b_med, b_p95, spread, mad) = reported.expect("ATTEMPTS > 0");
    CaseResult {
        id,
        description,
        a_med,
        a_p95,
        b_med,
        b_p95,
        ratio: worst_ratio,
        spread,
        mad,
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
        "| id | case | reps | A med ns | A p95 ns | B med ns | B p95 ns | ratio | MAD | width | gate | \
         A allocs/op | B allocs/op |\n",
    );
    table.push_str(
        "|----|------|-----:|---------:|---------:|---------:|---------:|------:|------:|------:|------|\
         -----------:|-----------:|\n",
    );
    for case in cases {
        let bound = case.gate_value();
        let gate = match case.gate {
            Gate::Report => "report".to_string(),
            Gate::ReportGated if case.failed() => "FAIL (alloc)".to_string(),
            Gate::ReportGated => "report (alloc=0)".to_string(),
            // Never print a bare `<= bound`: a row passes while fewer than
            // ATTEMPTS attempts are over, so the reported ratio can sit above
            // the bound and still pass (#396 review). Say both numbers.
            Gate::Parity | Gate::Bound(_) if case.a_allocs > 0.0 || case.b_allocs > 0.0 => {
                "FAIL (alloc)".to_string()
            }
            Gate::Parity | Gate::Bound(_) if case.failed() => format!(
                "FAIL (>{:.3}, {}/{} over)",
                bound.unwrap_or(0.0),
                case.attempts_exceeded,
                ATTEMPTS
            ),
            Gate::Parity => format!(
                "bound {:.3} (parity+3w), {}/{} over",
                bound.unwrap_or(0.0),
                case.attempts_exceeded,
                ATTEMPTS
            ),
            Gate::Bound(bound) => {
                format!(
                    "bound {bound}, {}/{} over",
                    case.attempts_exceeded, ATTEMPTS
                )
            }
        };
        table.push_str(&format!(
            "| {} | {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.3} | {:.4} | {} | {} | {:.2} | {:.2} |\n",
            case.id,
            case.description,
            REPS,
            case.a_med,
            case.a_p95,
            case.b_med,
            case.b_p95,
            case.ratio,
            case.mad,
            case.spread.map_or_else(String::new, |w| format!("{w:.4}")),
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

    // `cargo bench` compiles this module without running the tests below.
    #[allow(dead_code)]
    fn case(gate: Gate, ratio: f64, exceeded: usize) -> CaseResult {
        CaseResult {
            id: "T",
            description: "test",
            a_med: 2.0,
            a_p95: 2.0,
            b_med: 1.0,
            b_p95: 1.0,
            ratio,
            spread: None,
            mad: 0.0,
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
    fn a_bound_gate_compares_the_ratio_with_the_bound() {
        assert!(case(Gate::Bound(2.5), 2.6, ATTEMPTS).failed());
        // Under the bound is a pass, however many attempts were counted.
        assert!(!case(Gate::Bound(2.5), 2.4, ATTEMPTS).failed());
        // Over the bound in one attempt only is a blip, not a verdict.
        assert!(!case(Gate::Bound(2.5), 2.6, ATTEMPTS - 1).failed());
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

    /// The floor is not a decoration: a zero MAD must still yield a usable gate.
    #[test]
    fn a_parity_width_is_never_zero() {
        for median in [0.5f64, 0.95, 1.0, 1.504, 2.0] {
            let width = parity_width(median, 0.0);
            assert!(width > 0.0, "median {median} gave a zero width");
            assert!((width - 0.15 * median).abs() < 1e-12, "floor not applied");
            assert!(
                parity_bound(median, 0.0) > PARITY_ANCHOR,
                "bound not above parity for median {median}"
            );
        }
        // MAD wins once it exceeds the floor.
        let wide = parity_width(1.0, 0.2);
        assert!((wide - 1.4826 * 0.2).abs() < 1e-12);
        assert!(wide > parity_width(1.0, 0.0));
    }

    /// A steady regression must not admit itself. Nine identical ratios give a
    /// zero MAD, so a bound anchored at the measured median would pass a row that
    /// is twice as slow as parity.
    #[test]
    fn a_steady_doubling_fails_the_parity_gate() {
        let bound = parity_bound(2.0, 0.0);
        assert!(bound < 2.0, "bound {bound} would pass a steady 2x slowdown");

        let mut steady = case(Gate::Parity, 2.0, ATTEMPTS);
        steady.tolerance = Some(bound);
        assert!(steady.failed());

        // Parity itself still passes with that same zero MAD.
        let mut same = case(Gate::Parity, 1.0, 0);
        same.tolerance = Some(parity_bound(1.0, 0.0));
        assert!(!same.failed());
    }

    /// `Gate::Bound` is an absolute limit: MAD cannot move it.
    #[test]
    fn a_bound_gate_ignores_dispersion() {
        let mut noisy = case(Gate::Bound(2.5), 2.4, ATTEMPTS);
        noisy.mad = 10.0;
        noisy.spread = Some(15.0);
        assert!(!noisy.failed());
        assert_eq!(noisy.gate_value(), Some(2.5));

        // The same dispersion on a parity row widens its bound instead.
        assert!(parity_bound(1.0, 10.0) > 2.5);
        let mut parity = case(Gate::Parity, 2.4, ATTEMPTS);
        parity.tolerance = Some(parity_bound(1.0, 10.0));
        assert!(!parity.failed());
    }

    /// The statistic-to-verdict path, driven with **fixed** paired samples.
    ///
    /// A wall-clock version of this test flaked: a genuinely 8x workload can
    /// measure as `[6,6,6,6,8,10,10,10,10]`, whose median 8 and MAD 2 give a
    /// bound of 9.896 and are therefore legitimately admitted (#396 review).
    /// Fixed samples exercise the same code with no host in the loop — and
    /// mutating the bound inside `attempt_verdict` to `f64::INFINITY` fails here,
    /// which the direct-`CaseResult` tests cannot catch.
    #[test]
    fn a_slowdown_is_judged_by_the_production_path() {
        let uniform = |value: f64| vec![value; REPS];

        let steady_two = attempt_verdict(&uniform(2.0), Gate::Parity);
        assert!(
            steady_two.exceeded,
            "steady 2x admitted: ratio {} against bound {:?}",
            steady_two.ratio, steady_two.bound
        );

        let parity = attempt_verdict(&uniform(1.0), Gate::Parity);
        assert!(!parity.exceeded);
        assert_eq!(parity.width, Some(parity_width(1.0, 0.0)));
        assert_eq!(parity.bound, Some(parity_bound(1.0, 0.0)));

        // A wide spread widens the bound, which is the point of using MAD.
        let noisy = attempt_verdict(&[0.5, 0.9, 1.0, 1.0, 1.1, 1.4, 1.6, 1.9, 2.6], Gate::Parity);
        assert!(noisy.width.unwrap_or(0.0) > parity.width.unwrap_or(0.0));

        // Bound rows ignore the dispersion and use the absolute limit.
        assert!(!attempt_verdict(&uniform(2.4), Gate::Bound(2.5)).exceeded);
        assert!(attempt_verdict(&uniform(2.6), Gate::Bound(2.5)).exceeded);
        // One high sample does not move a Bound row either: the statistic is the
        // median of the nine paired ratios, not the worst one.
        assert!(
            !attempt_verdict(
                &[0.5, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                Gate::Bound(2.5)
            )
            .exceeded
        );

        // Report rows have no bound at all.
        assert!(!attempt_verdict(&uniform(50.0), Gate::Report).exceeded);
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
