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
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

/// Timing rounds per attempt; the reported time is the minimum over these.
pub const ROUNDS: usize = 5;
/// Attempts before a ratio gate may fail; a case fails only when all fail.
pub const ATTEMPTS: usize = 3;
/// Calls per allocation-count window (the count is exact, so this is just
/// enough to average over).
pub const ALLOC_ITERS: usize = 64;

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

/// One row of the overhead table.
pub struct CaseResult {
    id: &'static str,
    description: &'static str,
    a_ns: f64,
    b_ns: f64,
    ratio: f64,
    a_allocs: f64,
    b_allocs: f64,
    threshold: Option<f64>,
    attempts_exceeded: usize,
}

impl CaseResult {
    fn failed(&self) -> bool {
        match self.threshold {
            Some(_) => {
                self.attempts_exceeded == ATTEMPTS || self.a_allocs > 0.0 || self.b_allocs > 0.0
            }
            None => false,
        }
    }
}

/// Measure two paths on the same input and gate their ratio.
///
/// Both paths are warmed up first, then measured round by round in an
/// interleaved order so a frequency or thermal drift cannot land entirely on
/// one side of the ratio. A gated row fails only when all [`ATTEMPTS`] attempts
/// exceed the threshold.
pub fn measure_case(
    id: &'static str,
    description: &'static str,
    iterations: usize,
    threshold: Option<f64>,
    mut path_a: impl FnMut(),
    mut path_b: impl FnMut(),
) -> CaseResult {
    // Warm up page faults, lazy kernel state and thread-local scratch outside
    // every timed and counted window.
    path_a();
    path_b();

    let mut best_ratio = f64::INFINITY;
    let mut reported_a = 0.0;
    let mut reported_b = 0.0;
    let mut max_a_allocs: f64 = 0.0;
    let mut max_b_allocs: f64 = 0.0;
    let mut attempts_exceeded = 0;
    for _ in 0..ATTEMPTS {
        let mut a_ns = f64::INFINITY;
        let mut b_ns = f64::INFINITY;
        for _ in 0..ROUNDS {
            a_ns = a_ns.min(time_once(iterations, &mut path_a));
            b_ns = b_ns.min(time_once(iterations, &mut path_b));
        }
        let ratio = a_ns / b_ns;
        let a_allocs = allocations_per_op(&mut path_a);
        let b_allocs = allocations_per_op(&mut path_b);
        max_a_allocs = max_a_allocs.max(a_allocs);
        max_b_allocs = max_b_allocs.max(b_allocs);
        if let Some(threshold) = threshold {
            if ratio > threshold {
                attempts_exceeded += 1;
            }
        }
        if ratio < best_ratio {
            best_ratio = ratio;
            reported_a = a_ns;
            reported_b = b_ns;
        }
    }

    CaseResult {
        id,
        description,
        a_ns: reported_a,
        b_ns: reported_b,
        ratio: best_ratio,
        a_allocs: max_a_allocs,
        b_allocs: max_b_allocs,
        threshold,
        attempts_exceeded,
    }
}

/// Print the markdown table and append it to the GitHub step summary, if any.
pub fn print_table(cases: &[CaseResult], title: &str) {
    let mut table = String::new();
    table
        .push_str("| id | case | A ns/op | B ns/op | ratio | A allocs/op | B allocs/op | gate |\n");
    table
        .push_str("|----|------|--------:|--------:|------:|------------:|------------:|------|\n");
    for case in cases {
        let gate = match case.threshold {
            Some(threshold) if case.failed() => format!("FAIL (>{threshold})"),
            Some(threshold) => format!("<= {threshold}"),
            None => "report".to_string(),
        };
        table.push_str(&format!(
            "| {} | {} | {:.1} | {:.1} | {:.3} | {:.2} | {:.2} | {} |\n",
            case.id,
            case.description,
            case.a_ns,
            case.b_ns,
            case.ratio,
            case.a_allocs,
            case.b_allocs,
            gate,
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
