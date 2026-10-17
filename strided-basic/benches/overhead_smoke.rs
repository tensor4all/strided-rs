//! Cheap per-call overhead smoke for the compiled-plan work in
//! tensor4all/strided-rs#306 (the "S3" and "S4" rows of the issue comment).
//!
//! `harness = false`: this is a few-second ratio guard, not a criterion suite.
//! Every case measures two paths that should cost about the same on the same
//! small input and gates on the ratio measured in this run, never on an
//! absolute time. A counting `#[global_allocator]` reports allocs/op beside
//! every ratio.
//!
//! Run with `OVERHEAD_GATE=1` to make a failed gate exit non-zero; without it
//! the table is printed and the process succeeds, so local runs and first CI
//! runs can collect numbers. The gate is deliberately forgiving: it fails only
//! when all [`ATTEMPTS`] attempts exceed the threshold, and the per-round
//! timing is the minimum of [`ROUNDS`] rounds.
//!
//! Rows (see `#306`):
//! - S3a/S3b: `map_into` replanning per call vs a compiled typed `UpdatePlan`
//!   (`alpha = 1`, `beta = 0`, identity), `n = 16` and `4x4x4` (pinned to
//!   sequential execution). These guard P4 per-call planning. The thresholds
//!   are deliberately loose for the first CI runs (measured at ~1.5 and ~2.2
//!   on the reference machine, so 2.0 and 3.0).
//! - S3c/S3d: the same `map_into` path vs the compiled `ErasedUpdatePlan` a
//!   downstream runtime replays; reported only, because the erased boundary
//!   adds its own overhead.
//! - S4a/S4b: permuted `copy_into` vs contiguous `copy_into`, `8x8x8` and
//!   `32x32x32` (reported only).
//!
//! ```text
//! OVERHEAD_GATE=1 cargo bench --profile release -p strided-basic --bench overhead_smoke
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use strided_basic::{
    copy_into, map_into, with_execution_policy, ErasedRawStridedMut, ErasedRawStridedRef,
    ErasedUpdatePlan, ExecContext, ExecutionPolicy, KernelDType, RawStridedMut, RawStridedRef,
    StridedArray, StridedView, StridedViewMut, UpdateOp, UpdatePlan,
};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

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

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Timing rounds per attempt; the reported time is the minimum over these.
const ROUNDS: usize = 5;
/// Attempts before a ratio gate may fail; a case fails only when all fail.
const ATTEMPTS: usize = 3;
/// Calls per allocation-count window (the count is exact, so this is just
/// enough to average over).
const ALLOC_ITERS: usize = 64;

fn time_once(iterations: usize, body: &mut impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        body();
    }
    start.elapsed().as_secs_f64() * 1e9 / iterations as f64
}

fn allocations_per_op(body: &mut impl FnMut()) -> f64 {
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..ALLOC_ITERS {
        body();
    }
    let after = ALLOCATIONS.load(Ordering::Relaxed);
    (after - before) as f64 / ALLOC_ITERS as f64
}

struct CaseResult {
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

fn measure_case(
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
        // Interleave the two paths round by round so a frequency or thermal
        // drift cannot land entirely on one side of the ratio.
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

fn col_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1isize;
    for &dim in dims {
        strides.push(stride);
        stride *= dim as isize;
    }
    strides
}

/// Which compiled plan the S3 row replays.
#[derive(Clone, Copy)]
enum S3Baseline {
    /// Typed `UpdatePlan`: the gated P4 comparison, same work as `map_into`
    /// with `alpha = 1`, `beta = 0` and an identity map.
    TypedUpdatePlan,
    /// Erased `ErasedUpdatePlan`: the plan a downstream runtime compiles per
    /// call. Reported only because the erased boundary adds its own overhead.
    ErasedUpdatePlan,
}

/// S3: `map_into` replans the traversal on every call; a compiled update plan
/// replays it. Views are built once, so the timed region holds only the
/// operation itself.
fn s3_case(
    dims: &[usize],
    label: &'static str,
    iterations: usize,
    baseline: S3Baseline,
    threshold: Option<f64>,
) -> CaseResult {
    let len: usize = dims.iter().product();
    let src_data: Vec<f64> = (0..len).map(|value| value as f64 + 1.0).collect();
    let strides = col_major_strides(dims);

    // Path A: `map_into`, which plans the traversal per call.
    let src = StridedArray::from_parts(src_data.clone(), dims, &strides, 0).unwrap();
    let mut dst = StridedArray::<f64>::col_major(dims);
    let src_view = src.view();
    let mut dst_view = dst.view_mut();

    match baseline {
        S3Baseline::TypedUpdatePlan => {
            let mut raw_dst = vec![0.0f64; len];
            let plan = UpdatePlan::compile(dims, &strides, &strides).unwrap();
            let src_ref = RawStridedRef::new(&src_data, dims, &strides, 0).unwrap();
            let mut dst_ref = RawStridedMut::new(&mut raw_dst, dims, &strides, 0).unwrap();
            measure_case(
                label,
                "map_into per call vs compiled typed UpdatePlan",
                iterations,
                threshold,
                || {
                    map_into(&mut dst_view, &src_view, |value| value).unwrap();
                    black_box(dst_view.as_mut_ptr());
                },
                || {
                    plan.execute(&mut dst_ref, &src_ref, 1.0, 0.0, |value| value)
                        .unwrap();
                    black_box(dst_ref.dims());
                },
            )
        }
        S3Baseline::ErasedUpdatePlan => {
            let mut raw_dst = vec![0.0f64; len];
            let plan =
                ErasedUpdatePlan::compile(KernelDType::F64, dims, &strides, &strides).unwrap();
            let src_ref = ErasedRawStridedRef::from_slice(&src_data, dims, &strides, 0).unwrap();
            let mut dst_ref =
                ErasedRawStridedMut::from_slice_mut(&mut raw_dst, dims, &strides, 0).unwrap();
            let ctx = ExecContext::serial();
            measure_case(
                label,
                "map_into per call vs compiled ErasedUpdatePlan",
                iterations,
                threshold,
                || {
                    map_into(&mut dst_view, &src_view, |value| value).unwrap();
                    black_box(dst_view.as_mut_ptr());
                },
                || {
                    plan.execute(&ctx, &mut dst_ref, &src_ref, 1.0, 0.0, UpdateOp::Identity)
                        .unwrap();
                    black_box(dst_ref.dims());
                },
            )
        }
    }
}

/// S4: a permuted source costs more than a contiguous one. Reported only.
fn s4_case(
    dims: &[usize],
    permuted_strides: &[isize],
    label: &'static str,
    iterations: usize,
) -> CaseResult {
    let len: usize = dims.iter().product();
    let src_data: Vec<f64> = (0..len).map(|value| value as f64 + 1.0).collect();
    let contiguous = col_major_strides(dims);
    let mut contiguous_dst = vec![0.0f64; len];
    let mut permuted_dst = vec![0.0f64; len];

    let permuted_src = StridedView::<f64>::new(&src_data, dims, permuted_strides, 0).unwrap();
    let contiguous_src = StridedView::<f64>::new(&src_data, dims, &contiguous, 0).unwrap();
    let mut permuted_out =
        StridedViewMut::<f64>::new(&mut permuted_dst, dims, &contiguous, 0).unwrap();
    let mut contiguous_out =
        StridedViewMut::<f64>::new(&mut contiguous_dst, dims, &contiguous, 0).unwrap();

    measure_case(
        label,
        "permuted copy_into vs contiguous copy_into",
        iterations,
        None,
        || {
            copy_into(&mut permuted_out, &permuted_src).unwrap();
            black_box(permuted_out.as_mut_ptr());
        },
        || {
            copy_into(&mut contiguous_out, &contiguous_src).unwrap();
            black_box(contiguous_out.as_mut_ptr());
        },
    )
}

fn print_table(cases: &[CaseResult]) {
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
            let _ = file.write_all(b"\n### strided-basic overhead smoke\n\n");
            let _ = file.write_all(table.as_bytes());
        }
    }
}

fn main() {
    // Pin the traversal to one thread so the ratios are comparable on any CI
    // runner; `with_execution_policy` is the crate's supported knob.
    let cases = with_execution_policy(ExecutionPolicy::Sequential, || {
        vec![
            // Gated against the typed compiled plan (P4 per-call planning).
            s3_case(&[16], "S3a", 20_000, S3Baseline::TypedUpdatePlan, Some(2.0)),
            s3_case(
                &[4, 4, 4],
                "S3b",
                10_000,
                S3Baseline::TypedUpdatePlan,
                Some(3.0),
            ),
            // The erased compiled plan downstream actually replays: report
            // only, because the erased boundary adds its own overhead.
            s3_case(&[16], "S3c", 20_000, S3Baseline::ErasedUpdatePlan, None),
            s3_case(
                &[4, 4, 4],
                "S3d",
                10_000,
                S3Baseline::ErasedUpdatePlan,
                None,
            ),
            s4_case(&[8, 8, 8], &[64, 8, 1], "S4a", 2_000),
            s4_case(&[32, 32, 32], &[1024, 32, 1], "S4b", 200),
        ]
    });

    print_table(&cases);

    let failed = cases.iter().any(CaseResult::failed);
    if failed && std::env::var_os("OVERHEAD_GATE").is_some() {
        eprintln!("overhead smoke gate failed (see table above)");
        std::process::exit(1);
    }
}
