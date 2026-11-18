//! Fused executor vs the two-pass alternative (tensor4all/strided-rs#479 part 2).
//!
//! `harness = false`: one run prints a table, it gates nothing. Every row measures
//! a whole call on the same buffers, so two builds of this file can be compared
//! row by row.
//!
//! The alternative a fused executor has to beat is the two-pass one: materialise
//! every operand product (an `edges × segments × u` buffer — hundreds of MiB at
//! production sizes), then reduce it with the sorted-run plan. Both rows compute
//! the same numbers from the same inputs:
//!
//! - `fused/<case>`: `SegPolyPlan::execute_into`.
//! - `two-pass/<case>`: build the product with plain loops, then
//!   `RunSumPlan::execute_into`.
//!
//! View construction, the token and the scratch are outside the timed window in
//! both rows.
//!
//! ```text
//! cargo bench --profile release -p strided-multilinear --bench executor
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use strided_basic::{Exec, Pool, RawStridedMut, RawStridedRef, RunIndex, RunSumPlan};
use strided_multilinear::{IndexMode, Layout, Operation, Path, SegPolyPlan, Spec};

/// Wall time one sample aims for, so every row is measured over the same span.
const SAMPLE_TARGET: Duration = Duration::from_millis(20);
/// Samples per row; the median is reported.
const SAMPLES: usize = 15;
/// Worker slots of the pooled rows.
const POOL_SLOTS: usize = 4;

const CHAIN: Shape = Shape {
    name: "chain-8k",
    sources: 4096,
    batch: 8192,
    segments: 16,
    u: 32,
    runs: 2048,
};

const SMALL: Shape = Shape {
    name: "small-128",
    sources: 64,
    batch: 128,
    segments: 4,
    u: 8,
    runs: 32,
};

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    /// Distinct source rows, i.e. the gather extent.
    sources: usize,
    /// Batch points, i.e. edges.
    batch: usize,
    /// Segments per buffer.
    segments: usize,
    /// Lanes per segment.
    u: usize,
    /// Destination rows, i.e. runs.
    runs: usize,
}

fn payload(len: usize) -> Vec<f64> {
    (0..len)
        .map(|index| {
            let mantissa = ((index as u64).wrapping_mul(2_654_435_761) % 10_007) as f64 / 10_007.0;
            mantissa - 0.5
        })
        .collect()
}

/// Median nanoseconds per call.
fn measure(mut call: impl FnMut()) -> f64 {
    call(); // warm up: fault in the pages the samples reuse
    let mut times = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        let mut calls = 0u32;
        while start.elapsed() < SAMPLE_TARGET || calls == 0 {
            call();
            calls += 1;
        }
        times.push(start.elapsed().as_secs_f64() * 1e9 / f64::from(calls));
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn row(shape: Shape, pool: &Pool) {
    let segments = shape.segments;
    let u = shape.u;
    // Degree-1 message passing: out[run(e)] += c_s · x[src(e), s] · w[e, s].
    let source = payload(shape.sources * segments * u);
    let weight = payload(shape.batch * segments * u);
    let mut out = vec![0.0f64; shape.runs * segments * u];
    let gathers: Vec<i32> = (0..shape.batch)
        .map(|edge| (edge % shape.sources) as i32)
        .collect();
    let scatters: Vec<i32> = (0..shape.batch)
        .map(|edge| (edge % shape.runs) as i32)
        .collect();

    let source_dims = [shape.sources, segments, u];
    let batch_dims = [shape.batch, segments, u];
    let out_dims = [shape.runs, segments, u];
    let strides = [(segments * u) as isize, u as isize, 1];
    let runs = RunIndex::from_dest(&scatters, shape.runs).expect("in range");

    let path_segments: Vec<[u32; 3]> = (0..segments).map(|segment| [segment as u32; 3]).collect();
    let paths: Vec<Path<'_>> = path_segments
        .iter()
        .enumerate()
        .map(|(segment, ids)| Path {
            segments: ids,
            coefficient: 1.0 + segment as f64 * 0.125,
        })
        .collect();
    let operations = [Operation {
        buffers: &[0, 1, 2],
        path_start: 0,
        num_paths: segments as u32,
    }];
    let gather_mode = [IndexMode::Gather(&gathers)];
    let batched_mode = [IndexMode::Batched];
    let scatter_mode = [IndexMode::Scatter(&runs)];
    let indexes: [&[IndexMode<'_>]; 3] = [&gather_mode, &batched_mode, &scatter_mode];
    let spec = Spec {
        num_batch_axes: 1,
        operations: &operations,
        paths: &paths,
        indexes: &indexes,
    };
    let input_layouts = [
        Layout {
            dims: &source_dims,
            strides: &strides,
        },
        Layout {
            dims: &batch_dims,
            strides: &strides,
        },
    ];
    let out_layouts = [Layout {
        dims: &out_dims,
        strides: &strides,
    }];
    let plan = SegPolyPlan::<f64>::compile(&spec, &input_layouts, &out_layouts).expect("compiles");
    let mut scratch = vec![0.0f64; plan.scratch_len(POOL_SLOTS)];
    let macs = plan.macs() as f64;

    let source_ref = RawStridedRef::new(&source, &source_dims, &strides, 0).expect("view");
    let weight_ref = RawStridedRef::new(&weight, &batch_dims, &strides, 0).expect("view");

    macro_rules! fused {
        ($exec:expr) => {{
            let mut exec = $exec;
            measure(|| {
                // Both rows build their output view per call, so neither is
                // charged for it alone.
                let mut out_view =
                    RawStridedMut::new(&mut out, &out_dims, &strides, 0).expect("view");
                plan.execute_into(
                    std::slice::from_mut(&mut out_view),
                    &[source_ref, weight_ref],
                    &indexes,
                    &mut scratch,
                    black_box(exec.reborrow()),
                )
                .expect("executes")
            })
        }};
    }

    // The two-pass baseline: the same product, materialised, then reduced. It
    // applies the same per-segment coefficient as the fused row, or the two rows
    // would not compute the same numbers.
    let lanes = segments * u;
    let mut product = vec![0.0f64; shape.batch * lanes];
    let coefficients: Vec<f64> = (0..segments).map(|s| 1.0 + s as f64 * 0.125).collect();
    let prod_dims = [shape.batch, lanes];
    let prod_strides = [lanes as isize, 1];
    let dest_dims = [shape.runs, lanes];
    let run_plan = RunSumPlan::compile(&prod_dims, &prod_strides, &dest_dims, &prod_strides, &runs)
        .expect("compiles");
    /// The per-edge operand product, the buffer the fused row never allocates.
    fn build_product(
        product: &mut [f64],
        source: &[f64],
        weight: &[f64],
        gathers: &[i32],
        coefficients: &[f64],
        // (segments, u)
        shape: (usize, usize),
    ) {
        let (segments, u) = shape;
        let lanes = segments * u;
        for (edge, &from) in gathers.iter().enumerate() {
            let from = from as usize;
            for (segment, &coefficient) in coefficients.iter().enumerate().take(segments) {
                let offset = edge * lanes + segment * u;
                let source_row = from * lanes + segment * u;
                for lane in 0..u {
                    product[offset + lane] =
                        coefficient * source[source_row + lane] * weight[offset + lane];
                }
            }
        }
    }

    fn product_ref<'a>(
        product: &'a [f64],
        dims: &'a [usize; 2],
        strides: &'a [isize; 2],
    ) -> RawStridedRef<'a, f64> {
        RawStridedRef::new(product, dims, strides, 0).expect("view")
    }
    macro_rules! two_pass {
        ($exec:expr) => {{
            let mut exec = $exec;
            measure(|| {
                build_product(
                    &mut product,
                    &source,
                    &weight,
                    &gathers,
                    &coefficients,
                    (segments, u),
                );
                let mut dest =
                    RawStridedMut::new(&mut out, &dest_dims, &prod_strides, 0).expect("view");
                run_plan
                    .execute_into(
                        &mut dest,
                        &product_ref(&product, &prod_dims, &prod_strides),
                        &runs,
                        black_box(exec.reborrow()),
                    )
                    .expect("executes")
            })
        }};
    }

    // Both rows have to compute the same numbers before either is timed. They do
    // not agree bit for bit (the fused row combines fixed block partials), so the
    // check is a tolerance.
    {
        let mut out_view = RawStridedMut::new(&mut out, &out_dims, &strides, 0).expect("view");
        plan.execute_into(
            std::slice::from_mut(&mut out_view),
            &[source_ref, weight_ref],
            &indexes,
            &mut scratch,
            Exec::serial(),
        )
        .expect("executes");
        let fused_values = out.clone();
        build_product(
            &mut product,
            &source,
            &weight,
            &gathers,
            &coefficients,
            (segments, u),
        );
        let mut dest = RawStridedMut::new(&mut out, &dest_dims, &prod_strides, 0).expect("view");
        run_plan
            .execute_into(
                &mut dest,
                &product_ref(&product, &prod_dims, &prod_strides),
                &runs,
                Exec::serial(),
            )
            .expect("executes");
        let mut worst = 0.0f64;
        for (fused, pass) in fused_values.iter().zip(out.iter()) {
            worst = worst.max((fused - pass).abs() / fused.abs().max(pass.abs()).max(1.0));
        }
        assert!(
            worst < 1e-12,
            "{}: the two rows disagree by {worst:e}",
            shape.name
        );
    }

    let fused_serial = fused!(Exec::serial());
    let fused_pooled = fused!(Exec::new(pool, POOL_SLOTS).expect("token"));
    let pass_serial = two_pass!(Exec::serial());
    let pass_pooled = two_pass!(Exec::new(pool, POOL_SLOTS).expect("token"));
    let intermediate_mib =
        (shape.batch * lanes * core::mem::size_of::<f64>()) as f64 / f64::from(1 << 20);

    for (label, nanos, baseline) in [
        ("fused/serial", fused_serial, fused_serial),
        ("fused/pooled", fused_pooled, fused_serial),
        ("two-pass/serial", pass_serial, fused_serial),
        ("two-pass/pooled", pass_pooled, fused_serial),
    ] {
        println!(
            "{:<26} {:>10.0} {:>10.2} {:>11.1} {:>9.2}",
            format!("{}/{label}", shape.name),
            nanos,
            macs / nanos,
            intermediate_mib,
            nanos / baseline
        );
    }
    black_box(&out);
    black_box(&scratch);
}

fn main() {
    let pool = Pool::builder()
        .threads(POOL_SLOTS)
        .build()
        .expect("a bench pool");
    println!(
        "{:<26} {:>10} {:>10} {:>11} {:>9}",
        "row", "ns/call", "GFLOP/s", "MiB extra", "vs fused"
    );
    row(CHAIN, &pool);
    row(SMALL, &pool);
    println!(
        "\nGFLOP/s counts one multiply-add per path lane; it is a lower bound on the \
         arithmetic, not a hardware-counter measurement. `MiB extra` is the product \
         buffer the fused row never allocates."
    );
}
