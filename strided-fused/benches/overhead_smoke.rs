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
//! forgiving: it fails only when all [`support::ATTEMPTS`] attempts exceed the
//! threshold, and the per-round timing is the minimum of [`support::ROUNDS`]
//! rounds.
//!
//! Rows (see `#306`):
//! - S1: erased fused `execute` under `Exec::serial()` vs a
//!   budget-one `Exec::new(&pool, 1)` for a two-op DAG, `8x8x8`; guards #305.
//! - S2: typed fused single `Add` vs `zip_map2_into` add, contiguous `8x8x8`;
//!   guards per-element dispatch in the single-op specialization.
//! - S5: unspecialized replay vs the specialized `add`-then-`multiply` DAG,
//!   `8x8x8` (reported only).
//!
//! ```text
//! OVERHEAD_GATE=1 cargo bench --profile release -p strided-fused --bench overhead_smoke
//! ```

use std::hint::black_box;

use strided_basic::{
    zip_map2_into, ErasedRawStridedMut, ErasedRawStridedRef, Exec, KernelDType, Pool, StridedView,
    StridedViewMut,
};
use strided_fused::{fused_elementwise_into, ErasedFusedPlan, FusedInst, FusedOp, FusedPlan};

#[path = "../../strided-basic/benches/support/mod.rs"]
mod support;

#[global_allocator]
static GLOBAL: support::CountingAllocator = support::CountingAllocator;

use support::{measure_case, print_table, CaseResult};

const SIDE: usize = 8;
const LEN: usize = SIDE * SIDE * SIDE;

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
    let pool = Pool::builder().threads(1).build().unwrap();

    measure_case(
        "S1",
        "fused serial token vs budget-one pooled token",
        2_000,
        Some(1.5),
        || {
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut out_serial, &dims, &strides, 0).unwrap();
            plan.execute(&mut dest, &inputs, Exec::serial()).unwrap();
            black_box(dest.dims());
        },
        || {
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(&mut out_parallel, &dims, &strides, 0).unwrap();
            plan.execute(&mut dest, &inputs, Exec::new(&pool, 1).unwrap())
                .unwrap();
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
            fused_elementwise_into(
                std::slice::from_mut(&mut fused_view),
                &inputs,
                &plan,
                Exec::serial(),
            )
            .unwrap();
            black_box(fused_view.as_mut_ptr());
        },
        || {
            zip_map2_into(
                &mut map_view,
                &lhs_view,
                &rhs_view,
                |x, y| x + y,
                Exec::serial(),
            )
            .unwrap();
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
                Exec::serial(),
            )
            .unwrap();
            black_box(unspecialized_view.as_mut_ptr());
        },
        || {
            fused_elementwise_into(
                std::slice::from_mut(&mut specialized_view),
                &inputs,
                &specialized,
                Exec::serial(),
            )
            .unwrap();
            black_box(specialized_view.as_mut_ptr());
        },
    )
}

fn run_cases() -> Vec<CaseResult> {
    vec![s1_case(), s2_case(), s5_case()]
}

fn main() {
    // Every row carries its own token: S1 compares serial with a budget-one pooled
    // token, and the typed rows are serial, so nothing fans out.
    let cases = run_cases();

    print_table(&cases, "### strided-fused overhead smoke");
    support::exit_if_failed(&cases);
}
