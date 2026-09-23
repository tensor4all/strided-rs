//! Paired regression benchmark: dtype-erased uninit elementwise entries against
//! the typed uninit kernels they dispatch to, and the ternary entries against a
//! plain slice loop.
//!
//! The erased entries select the runtime operation once, outside the element
//! loop. This benchmark guards that the erased dispatch stays at typed-kernel
//! parity across sizes and layouts, including a transposed input that misses
//! the contiguous fast path. The ternary `select` and `clamp` rows guard the
//! per-call work around the loop too: a byte by byte `bool` validation once
//! cost more than the select itself, and a NaN test per `clamp` step doubled
//! its time, and neither showed up in any binary row.
//!
//! `RAYON_NUM_THREADS=1 cargo bench -p strided-kernel --bench erased_uninit_elementwise`
//!
//! Each `SUMMARY` line reports the typed and erased medians in milliseconds, the
//! paired geometric-mean ratio `erased / typed`, and its upper 95% bound.

use core::mem::MaybeUninit;
use std::hint::black_box;
use std::time::Instant;

use strided_kernel::{
    erased_clamp_into_uninit, erased_map_into_uninit, erased_select_into_uninit,
    erased_zip_into_uninit, map_into, zip_map2_into, ErasedMapOp, ErasedRawStridedPtr,
    ErasedRawStridedRef, ErasedRawStridedUninitMut, ErasedZipOp, ExecContext, KernelDType,
    StridedView, StridedViewMut,
};

const WARMUPS: usize = 6;
const SAMPLES: usize = 21;

fn values(len: usize, offset: f64) -> Vec<f64> {
    (0..len)
        .map(|index| (index % 251) as f64 * 0.00390625 - 0.5 + offset)
        .collect()
}

/// Repeat count that keeps one sample near a millisecond.
fn repeats(len: usize) -> usize {
    (1 << 20) / len.max(1) + 1
}

fn time_ms(repeats: usize, operation: &mut impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..repeats {
        operation();
    }
    start.elapsed().as_secs_f64() * 1e3 / repeats as f64
}

fn median(samples: &[f64]) -> f64 {
    let mut ordered = samples.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered[ordered.len() / 2]
}

fn measure_pair(case: &str, len: usize, mut typed: impl FnMut(), mut erased: impl FnMut()) {
    let repeats = repeats(len);
    for _ in 0..WARMUPS {
        typed();
        erased();
    }
    let mut typed_samples = Vec::with_capacity(SAMPLES);
    let mut erased_samples = Vec::with_capacity(SAMPLES);
    for sample in 0..SAMPLES {
        // Alternate the order so drift does not favor one side.
        if sample % 2 == 0 {
            typed_samples.push(time_ms(repeats, &mut typed));
            erased_samples.push(time_ms(repeats, &mut erased));
        } else {
            erased_samples.push(time_ms(repeats, &mut erased));
            typed_samples.push(time_ms(repeats, &mut typed));
        }
    }
    let logs: Vec<f64> = typed_samples
        .iter()
        .zip(&erased_samples)
        .map(|(&typed, &erased)| (erased / typed).ln())
        .collect();
    let mean = logs.iter().sum::<f64>() / logs.len() as f64;
    let variance =
        logs.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (logs.len() - 1) as f64;
    let upper95 = (mean + 1.96 * (variance / logs.len() as f64).sqrt()).exp();
    println!(
        "SUMMARY,{case},{:.6},{:.6},{:.3},{upper95:.3}",
        median(&typed_samples),
        median(&erased_samples),
        mean.exp()
    );
}

/// Input layout of one benchmark case.
#[derive(Clone, Copy)]
enum Layout {
    /// Rank-1 contiguous.
    Vector,
    /// Rank-2 column-major.
    Matrix,
    /// Rank-2 with the left input transposed (row-major), a fast-path miss.
    TransposedLhs,
}

impl Layout {
    fn label(self) -> &'static str {
        match self {
            Self::Vector => "vector",
            Self::Matrix => "matrix",
            Self::TransposedLhs => "transposed_lhs",
        }
    }

    /// Dims plus the (lhs, rhs, dest) strides for `len` elements.
    fn describe(self, len: usize) -> (Vec<usize>, Vec<isize>, Vec<isize>) {
        match self {
            Self::Vector => (vec![len], vec![1], vec![1]),
            Self::Matrix | Self::TransposedLhs => {
                let side = (len as f64).sqrt() as usize;
                let col_major = vec![1, side as isize];
                let lhs = match self {
                    Self::TransposedLhs => vec![side as isize, 1],
                    _ => col_major.clone(),
                };
                (vec![side, side], lhs, col_major)
            }
        }
    }
}

/// `typed_op` is a monomorphic closure, so the typed baseline inlines it.
fn bench_zip<F>(op: ErasedZipOp, label: &str, typed_op: F, layout: Layout, len: usize)
where
    F: Fn(f64, f64) -> f64 + Copy + Send + Sync,
{
    let (dims, lhs_strides, strides) = layout.describe(len);
    let len: usize = dims.iter().product();
    let lhs = values(len, 1.0);
    let rhs = values(len, 2.0);
    let mut typed_out = vec![MaybeUninit::<f64>::uninit(); len];
    let mut erased_out = vec![MaybeUninit::<f64>::uninit(); len];
    let lhs_view = StridedView::<f64>::new(&lhs, &dims, &lhs_strides, 0).unwrap();
    let rhs_view = StridedView::<f64>::new(&rhs, &dims, &strides, 0).unwrap();
    let lhs_ref = ErasedRawStridedRef::from_slice(&lhs, &dims, &lhs_strides, 0).unwrap();
    let rhs_ref = ErasedRawStridedRef::from_slice(&rhs, &dims, &strides, 0).unwrap();
    let ctx = ExecContext::serial();
    measure_pair(
        &format!("zip_{label},{},{len}", layout.label()),
        len,
        || {
            let mut dest = StridedViewMut::new(&mut typed_out, &dims, &strides, 0).unwrap();
            zip_map2_into(&mut dest, &lhs_view, &rhs_view, |a, b| {
                MaybeUninit::new(typed_op(a, b))
            })
            .unwrap();
            black_box(&typed_out);
        },
        || {
            let mut dest =
                ErasedRawStridedUninitMut::from_uninit_slice(&mut erased_out, &dims, &strides, 0)
                    .unwrap();
            erased_zip_into_uninit(
                KernelDType::F64,
                op,
                &ctx,
                &mut dest,
                &ErasedRawStridedPtr::from_ref(&lhs_ref),
                &ErasedRawStridedPtr::from_ref(&rhs_ref),
            )
            .unwrap();
            black_box(&erased_out);
        },
    );
}

/// `dest = if pred { on_true } else { on_false }` with an irregular predicate.
///
/// The ternary rows compare against a plain slice loop, not the typed
/// `zip_map3_into`, so a slow loop shared by the typed and erased paths also
/// shows in the ratio.
fn bench_select(len: usize) {
    let dims = [len];
    let strides = [1];
    let pred: Vec<bool> = (0..len).map(|index| (index * 7919) % 13 < 6).collect();
    let on_true = values(len, 1.0);
    let on_false = values(len, 2.0);
    let mut typed_out = vec![MaybeUninit::<f64>::uninit(); len];
    let mut erased_out = vec![MaybeUninit::<f64>::uninit(); len];
    let pred_ref = ErasedRawStridedRef::from_slice(&pred, &dims, &strides, 0).unwrap();
    let true_ref = ErasedRawStridedRef::from_slice(&on_true, &dims, &strides, 0).unwrap();
    let false_ref = ErasedRawStridedRef::from_slice(&on_false, &dims, &strides, 0).unwrap();
    let ctx = ExecContext::serial();
    measure_pair(
        &format!("select,vector,{len}"),
        len,
        || {
            for (((out, &p), &a), &b) in
                typed_out.iter_mut().zip(&pred).zip(&on_true).zip(&on_false)
            {
                *out = MaybeUninit::new(if p { a } else { b });
            }
            black_box(&typed_out);
        },
        || {
            let mut dest =
                ErasedRawStridedUninitMut::from_uninit_slice(&mut erased_out, &dims, &strides, 0)
                    .unwrap();
            erased_select_into_uninit(
                KernelDType::F64,
                &ctx,
                &mut dest,
                &ErasedRawStridedPtr::from_ref(&pred_ref),
                &ErasedRawStridedPtr::from_ref(&true_ref),
                &ErasedRawStridedPtr::from_ref(&false_ref),
            )
            .unwrap();
            black_box(&erased_out);
        },
    );
}

/// `clamp(x, lo, hi)` against a plain slice loop that tests NaN once per element.
fn bench_clamp(len: usize) {
    let dims = [len];
    let strides = [1];
    let x = values(len, 0.0);
    let lo = vec![-0.25; len];
    let hi = vec![0.25; len];
    let mut typed_out = vec![MaybeUninit::<f64>::uninit(); len];
    let mut erased_out = vec![MaybeUninit::<f64>::uninit(); len];
    let x_ref = ErasedRawStridedRef::from_slice(&x, &dims, &strides, 0).unwrap();
    let lo_ref = ErasedRawStridedRef::from_slice(&lo, &dims, &strides, 0).unwrap();
    let hi_ref = ErasedRawStridedRef::from_slice(&hi, &dims, &strides, 0).unwrap();
    let ctx = ExecContext::serial();
    measure_pair(
        &format!("clamp,vector,{len}"),
        len,
        || {
            for (((out, &x), &lo), &hi) in typed_out.iter_mut().zip(&x).zip(&lo).zip(&hi) {
                let raised = if lo >= x { lo } else { x };
                let lowered = if hi <= raised { hi } else { raised };
                *out = MaybeUninit::new(if x.is_nan() | lo.is_nan() | hi.is_nan() {
                    f64::NAN
                } else {
                    lowered
                });
            }
            black_box(&typed_out);
        },
        || {
            let mut dest =
                ErasedRawStridedUninitMut::from_uninit_slice(&mut erased_out, &dims, &strides, 0)
                    .unwrap();
            erased_clamp_into_uninit(
                KernelDType::F64,
                &ctx,
                &mut dest,
                &ErasedRawStridedPtr::from_ref(&x_ref),
                &ErasedRawStridedPtr::from_ref(&lo_ref),
                &ErasedRawStridedPtr::from_ref(&hi_ref),
            )
            .unwrap();
            black_box(&erased_out);
        },
    );
}

fn bench_map<F>(op: ErasedMapOp, label: &str, typed_op: F, len: usize)
where
    F: Fn(f64) -> f64 + Copy + Send + Sync,
{
    let dims = [len];
    let strides = [1];
    let input = values(len, 0.0);
    let mut typed_out = vec![MaybeUninit::<f64>::uninit(); len];
    let mut erased_out = vec![MaybeUninit::<f64>::uninit(); len];
    let input_view = StridedView::<f64>::new(&input, &dims, &strides, 0).unwrap();
    let input_ref = ErasedRawStridedRef::from_slice(&input, &dims, &strides, 0).unwrap();
    let ctx = ExecContext::serial();
    measure_pair(
        &format!("map_{label},vector,{len}"),
        len,
        || {
            let mut dest = StridedViewMut::new(&mut typed_out, &dims, &strides, 0).unwrap();
            map_into(&mut dest, &input_view, |a| MaybeUninit::new(typed_op(a))).unwrap();
            black_box(&typed_out);
        },
        || {
            let mut dest =
                ErasedRawStridedUninitMut::from_uninit_slice(&mut erased_out, &dims, &strides, 0)
                    .unwrap();
            erased_map_into_uninit(
                KernelDType::F64,
                op,
                &ctx,
                &mut dest,
                &ErasedRawStridedPtr::from_ref(&input_ref),
            )
            .unwrap();
            black_box(&erased_out);
        },
    );
}

fn main() {
    println!("CONFIG,warmups={WARMUPS},samples={SAMPLES},dtype=f64,context=serial");
    println!("HEADER,case,layout,len,typed_ms,erased_ms,ratio,upper95");
    let sizes = [1usize << 12, 1 << 16, 1 << 20];
    for &len in &sizes {
        for layout in [Layout::Vector, Layout::Matrix, Layout::TransposedLhs] {
            bench_zip(ErasedZipOp::Add, "add", |a, b| a + b, layout, len);
            bench_zip(ErasedZipOp::Multiply, "multiply", |a, b| a * b, layout, len);
        }
        bench_zip(
            ErasedZipOp::Maximum,
            "maximum",
            |a, b| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else if a >= b {
                    a
                } else {
                    b
                }
            },
            Layout::Vector,
            len,
        );
        bench_select(len);
        bench_clamp(len);
        bench_map(ErasedMapOp::Negate, "negate", |a| -a, len);
        bench_map(ErasedMapOp::Abs, "abs", |a: f64| a.abs(), len);
    }
    // A clamp that tested NaN at each step matched the loop at 1 << 20 but
    // took twice as long once its four streams left the cache.
    bench_select(1 << 23);
    bench_clamp(1 << 23);
}
