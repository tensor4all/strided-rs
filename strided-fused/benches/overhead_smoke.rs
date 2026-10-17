//! Cheap per-call overhead smoke for the fused work in
//! tensor4all/strided-rs#306 (the "S1", "S2" and "S5" rows of the issue
//! comment; S3/S4 live in `strided-basic`).
//!
//! `harness = false`: this is a few-second ratio guard, not a criterion suite.
//! Every gated case measures two paths that should cost about the same on the
//! same small input and gates on the ratio measured in this run, never on an
//! absolute time. A counting `#[global_allocator]` reports allocs/op beside
//! every ratio.
//!
//! Run with `OVERHEAD_GATE=1` to make a failed gate exit non-zero; without it
//! the table is printed and the process succeeds. The gate is deliberately
//! forgiving: it fails only when all [`ATTEMPTS`] attempts exceed the
//! threshold, and the per-round timing is the minimum of [`ROUNDS`] rounds.
//!
//! Rows (see `#306`):
//! - S1: erased fused `execute` under `ExecContext::serial()` vs
//!   `ExecContext::max_threads(1)` for a two-op DAG, `8x8x8`; guards #305.
//! - S2: typed fused single `Add` vs `zip_map2_into` add, contiguous `8x8x8`;
//!   guards per-element dispatch in the single-op specialization.
//! - S5: unspecialized replay vs the specialized `add`-then-`multiply` DAG,
//!   `8x8x8` (reported only).
//!
//! ```text
//! OVERHEAD_GATE=1 cargo bench --profile release -p strided-fused --bench overhead_smoke
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use strided_basic::{
    zip_map2_into, ErasedRawStridedMut, ErasedRawStridedRef, ExecContext, KernelDType, StridedView,
    StridedViewMut,
};
use strided_fused::{fused_elementwise_into, ErasedFusedPlan, FusedInst, FusedOp, FusedPlan};

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

const ROUNDS: usize = 5;
const ATTEMPTS: usize = 3;
const ALLOC_ITERS: usize = 64;
const SIDE: usize = 8;
const LEN: usize = SIDE * SIDE * SIDE;

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

fn two_op_dag() -> FusedPlan {
    FusedPlan::new(
        2,
        vec![3],
        vec![
            FusedInst::new(FusedOp::Add, vec![0, 1]),
            FusedInst::new(FusedOp::Multiply, vec![2, 0]),
        ],
    )
}

/// `add` then `multiply`, with the multiply reading the second input. The
/// input order `[1, 0]` keeps `classify_static_specialization` from matching,
/// so this is the unspecialized counterpart of [`specialized_add_mul`].
fn unspecialized_add_mul() -> FusedPlan {
    FusedPlan::new(
        2,
        vec![3],
        vec![
            FusedInst::new(FusedOp::Add, vec![1, 0]),
            FusedInst::new(FusedOp::Multiply, vec![2, 0]),
        ],
    )
}

/// The exact shape `classify_static_specialization` recognizes as `AddMulLeft`.
fn specialized_add_mul() -> FusedPlan {
    FusedPlan::new(
        2,
        vec![3],
        vec![
            FusedInst::new(FusedOp::Add, vec![0, 1]),
            FusedInst::new(FusedOp::Multiply, vec![2, 0]),
        ],
    )
}

fn make_data(seed: f64) -> Vec<f64> {
    (0..LEN)
        .map(|index| seed + (index % 97) as f64 * 0.01)
        .collect()
}

/// S1: the same erased plan under a serial context vs a one-thread parallel
/// context.
fn s1_case() -> CaseResult {
    let dims = [SIDE, SIDE, SIDE];
    let strides = [1isize, SIDE as isize, (SIDE * SIDE) as isize];
    let lhs = make_data(1.0);
    let rhs = make_data(2.0);
    let lhs_ref = ErasedRawStridedRef::from_slice(&lhs, &dims, &strides, 0).unwrap();
    let rhs_ref = ErasedRawStridedRef::from_slice(&rhs, &dims, &strides, 0).unwrap();
    let inputs = [lhs_ref, rhs_ref];
    let plan = ErasedFusedPlan::compile(KernelDType::F64, two_op_dag()).unwrap();

    let mut out_serial = vec![0.0f64; LEN];
    let mut out_parallel = vec![0.0f64; LEN];
    let serial = ExecContext::serial();
    let parallel = ExecContext::max_threads(1).unwrap();

    measure_case(
        "S1",
        "fused serial ctx vs 1-thread parallel ctx",
        2_000,
        Some(1.5),
        || {
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut out_serial, &dims, &strides, 0).unwrap();
            plan.execute(&serial, &mut dest, &inputs).unwrap();
            black_box(dest.dims());
        },
        || {
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut out_parallel, &dims, &strides, 0).unwrap();
            plan.execute(&parallel, &mut dest, &inputs).unwrap();
            black_box(dest.dims());
        },
    )
}

/// S2: typed fused single `Add` vs `zip_map2_into`.
fn s2_case() -> CaseResult {
    let dims = [SIDE, SIDE, SIDE];
    let strides = [1isize, SIDE as isize, (SIDE * SIDE) as isize];
    let lhs = make_data(1.0);
    let rhs = make_data(2.0);
    let lhs_view = StridedView::<f64>::new(&lhs, &dims, &strides, 0).unwrap();
    let rhs_view = StridedView::<f64>::new(&rhs, &dims, &strides, 0).unwrap();
    let inputs = [lhs_view.clone(), rhs_view.clone()];
    let plan = FusedPlan::new(2, vec![2], vec![FusedInst::new(FusedOp::Add, vec![0, 1])]);

    let mut fused_out = vec![0.0f64; LEN];
    let mut map_out = vec![0.0f64; LEN];
    let mut fused_view = StridedViewMut::<f64>::new(&mut fused_out, &dims, &strides, 0).unwrap();
    let mut map_view = StridedViewMut::<f64>::new(&mut map_out, &dims, &strides, 0).unwrap();

    measure_case(
        "S2",
        "fused single add vs zip_map2_into add",
        5_000,
        Some(1.5),
        || {
            fused_elementwise_into(std::slice::from_mut(&mut fused_view), &inputs, &plan).unwrap();
            black_box(fused_view.as_mut_ptr());
        },
        || {
            zip_map2_into(&mut map_view, &lhs_view, &rhs_view, |x, y| x + y).unwrap();
            black_box(map_view.as_mut_ptr());
        },
    )
}

/// S5: unspecialized replay vs the recognized `add`-then-`multiply` shape.
/// Both compute `(a + b) * a`; only the classification differs.
fn s5_case() -> CaseResult {
    let dims = [SIDE, SIDE, SIDE];
    let strides = [1isize, SIDE as isize, (SIDE * SIDE) as isize];
    let lhs = make_data(1.0);
    let rhs = make_data(2.0);
    let lhs_view = StridedView::<f64>::new(&lhs, &dims, &strides, 0).unwrap();
    let rhs_view = StridedView::<f64>::new(&rhs, &dims, &strides, 0).unwrap();
    let inputs = [lhs_view.clone(), rhs_view.clone()];
    let unspecialized = unspecialized_add_mul();
    let specialized = specialized_add_mul();

    let mut unspecialized_out = vec![0.0f64; LEN];
    let mut specialized_out = vec![0.0f64; LEN];
    let mut unspecialized_view =
        StridedViewMut::<f64>::new(&mut unspecialized_out, &dims, &strides, 0).unwrap();
    let mut specialized_view =
        StridedViewMut::<f64>::new(&mut specialized_out, &dims, &strides, 0).unwrap();

    measure_case(
        "S5",
        "unspecialized replay vs specialized add*mul",
        5_000,
        None,
        || {
            fused_elementwise_into(
                std::slice::from_mut(&mut unspecialized_view),
                &inputs,
                &unspecialized,
            )
            .unwrap();
            black_box(unspecialized_view.as_mut_ptr());
        },
        || {
            fused_elementwise_into(
                std::slice::from_mut(&mut specialized_view),
                &inputs,
                &specialized,
            )
            .unwrap();
            black_box(specialized_view.as_mut_ptr());
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
            let _ = file.write_all(b"\n### strided-fused overhead smoke\n\n");
            let _ = file.write_all(table.as_bytes());
        }
    }
}

fn run_cases() -> Vec<CaseResult> {
    vec![s1_case(), s2_case(), s5_case()]
}

fn main() {
    // One thread for every row: the S1 comparison is serial vs a one-thread
    // parallel context, and the typed rows must not fan out either.
    #[cfg(feature = "parallel")]
    let cases = {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            assert_eq!(
                rayon::current_num_threads(),
                1,
                "overhead smoke must run on one thread"
            );
            run_cases()
        })
    };
    #[cfg(not(feature = "parallel"))]
    let cases = run_cases();

    print_table(&cases);

    let failed = cases.iter().any(CaseResult::failed);
    if failed && std::env::var_os("OVERHEAD_GATE").is_some() {
        eprintln!("overhead smoke gate failed (see table above)");
        std::process::exit(1);
    }
}
