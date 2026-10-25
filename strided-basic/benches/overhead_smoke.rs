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
//! when all [`support::ATTEMPTS`] attempts exceed the threshold, and the
//! per-round timing is the minimum of [`support::ROUNDS`] rounds.
//!
//! Rows (see `#306`):
//! - S3a/S3b: `map_into` replanning per call vs a compiled typed `UpdatePlan`
//!   (`alpha = 1`, `beta = 0`, identity), `n = 16` and `4x4x4` (pinned to
//!   sequential execution). These guard P4 per-call planning. The thresholds
//!   are deliberately loose for the first CI runs and set from the slowest
//!   machine seen: ~1.5 / ~2.2 on the reference EPYC host, and 1.87 / 3.15 on
//!   the GitHub-hosted runner after #317 made the beta == 0 compiled baseline
//!   write-only (faster), so 2.5 and 4.0. Tighten from CI history.
//! - S3c/S3d: the same `map_into` path vs the compiled `ErasedUpdatePlan` a
//!   downstream runtime replays; reported only, because the erased boundary
//!   adds its own overhead.
//! - S4a/S4b: permuted `copy_into` vs contiguous `copy_into`, `8x8x8` and
//!   `32x32x32` (reported only).
//!
//! ```text
//! OVERHEAD_GATE=1 cargo bench --profile release -p strided-basic --bench overhead_smoke
//! ```

use std::hint::black_box;

use strided_basic::Exec;
use strided_basic::{
    copy_into, map_into, ErasedRawStridedMut, ErasedRawStridedRef, ErasedUpdatePlan, KernelDType,
    RawStridedMut, RawStridedRef, StridedArray, StridedView, StridedViewMut, UpdateOp, UpdatePlan,
};

#[path = "support/mod.rs"]
mod support;

#[global_allocator]
static GLOBAL: support::CountingAllocator = support::CountingAllocator;

use support::{measure_case, print_table, CaseResult};

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
                    map_into(&mut dst_view, &src_view, |value| value, Exec::serial()).unwrap();
                    black_box(dst_view.as_mut_ptr());
                },
                || {
                    plan.execute(
                        &mut dst_ref,
                        &src_ref,
                        1.0,
                        0.0,
                        |value| value,
                        Exec::serial(),
                    )
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
            measure_case(
                label,
                "map_into per call vs compiled ErasedUpdatePlan",
                iterations,
                threshold,
                || {
                    map_into(&mut dst_view, &src_view, |value| value, Exec::serial()).unwrap();
                    black_box(dst_view.as_mut_ptr());
                },
                || {
                    plan.execute(
                        &mut dst_ref,
                        &src_ref,
                        1.0,
                        0.0,
                        UpdateOp::Identity,
                        Exec::serial(),
                    )
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
            copy_into(&mut permuted_out, &permuted_src, Exec::serial()).unwrap();
            black_box(permuted_out.as_mut_ptr());
        },
        || {
            copy_into(&mut contiguous_out, &contiguous_src, Exec::serial()).unwrap();
            black_box(contiguous_out.as_mut_ptr());
        },
    )
}

fn main() {
    // Pin the traversal to one thread so the ratios are comparable on any CI
    // runner; the serial token is the crate's supported knob.
    let cases = {
        vec![
            // Gated against the typed compiled plan (P4 per-call planning).
            s3_case(&[16], "S3a", 20_000, S3Baseline::TypedUpdatePlan, Some(2.5)),
            s3_case(
                &[4, 4, 4],
                "S3b",
                10_000,
                S3Baseline::TypedUpdatePlan,
                Some(4.0),
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
    };

    print_table(&cases, "### strided-basic overhead smoke");
    support::exit_if_failed(&cases);
}
