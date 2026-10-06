//! Regression benchmark for the single-axis plans: the fused norm against the
//! composed sequence of existing erased primitives, and the scan and
//! arg-reduction plans against raw pointer loops.
//!
//! `cargo bench -p strided-kernel --features parallel --bench axis_ops`
//!
//! Each case runs at one thread (`ExecContext::serial()`) and at four threads
//! (`ExecContext::max_threads(4)` inside a four-worker pool whose size is
//! asserted). Shapes cover the feature-first decode activations of
//! tenferro-rs#2006 (`(d, len, batch)` normalized over a unit-stride `d`), a
//! size past the last level cache, and a strided normalized axis, which runs
//! the block (panel) kernels instead of the contiguous line kernels.
//!
//! The composed norm reuses preallocated outputs, so it measures the extra
//! passes over memory but not the per-op allocation and dispatch a tensor
//! runtime adds on top. Each `SUMMARY` line reports medians in milliseconds.

use std::hint::black_box;
use std::time::Instant;

use strided_kernel::{
    erased_zip_into, map_into, ArgReduceOp, ErasedArgReducePlan, ErasedNormPlan,
    ErasedRawStridedMut, ErasedRawStridedPtr, ErasedRawStridedRef, ErasedReducePlan,
    ErasedScanPlan, ErasedZipOp, ExecContext, KernelDType, NormSpec, ReduceOp, ScanOp, ScanOptions,
    StridedView, StridedViewMut,
};

const WARMUPS: usize = 4;
const SAMPLES: usize = 15;

fn values(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 7919 + salt) % 1009) as f32 / 251.0 - 2.0)
        .collect()
}

fn repeats(elements: usize) -> usize {
    (1 << 22) / elements.max(1) + 1
}

fn median_ms(elements: usize, mut operation: impl FnMut()) -> f64 {
    let repeats = repeats(elements);
    for _ in 0..WARMUPS {
        operation();
    }
    let mut samples: Vec<f64> = (0..SAMPLES)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..repeats {
                operation();
            }
            start.elapsed().as_secs_f64() * 1e3 / repeats as f64
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn col_major(dims: &[usize]) -> Vec<isize> {
    let mut stride = 1isize;
    dims.iter()
        .map(|&d| {
            let s = stride;
            stride *= d as isize;
            s
        })
        .collect()
}

/// A normalization case: `dims` with the normalized `axis` and `strides`.
struct NormCase {
    label: &'static str,
    dims: Vec<usize>,
    strides: Vec<isize>,
    axis: usize,
}

fn norm_cases() -> Vec<NormCase> {
    let mut cases = Vec::new();
    for (label, d, len, batch) in [
        ("decode_d1024_len8_b1", 1024, 8, 1),
        ("decode_d1024_len64_b1", 1024, 64, 1),
        ("decode_d1024_len8_b8", 1024, 8, 8),
        ("decode_d1024_len64_b8", 1024, 64, 8),
        ("large_d1024_len16384", 1024, 16384, 1),
    ] {
        let dims = vec![d, len, batch];
        cases.push(NormCase {
            label,
            strides: col_major(&dims),
            dims,
            axis: 0,
        });
    }
    // Row-major (len, d): the normalized axis is strided, a fast-path miss
    // that runs the panel kernels.
    let dims = vec![512, 1024];
    cases.push(NormCase {
        label: "strided_axis_len512_d1024",
        strides: vec![1, 512],
        dims,
        axis: 1,
    });
    cases
}

/// Composed layer norm: sum, scale, subtract, sum of squares, rsqrt,
/// multiply, weight, bias, each a separate pass with a preallocated output.
struct Composed {
    sum: ErasedReducePlan,
    sumsq: ErasedReducePlan,
    stat_dims: Vec<usize>,
    stat_strides: Vec<isize>,
    /// The statistics broadcast back over the normalized axis.
    bcast_strides: Vec<isize>,
    /// The weight/bias vector broadcast over the other axes.
    param_strides: Vec<isize>,
}

impl Composed {
    fn new(case: &NormCase) -> Self {
        let mut stat_dims = case.dims.clone();
        stat_dims.remove(case.axis);
        let stat_strides = col_major(&stat_dims);
        let mut bcast_strides = stat_strides.clone();
        bcast_strides.insert(case.axis, 0);
        let mut param_strides = vec![0isize; case.dims.len()];
        param_strides[case.axis] = 1;
        let plan = |op| {
            ErasedReducePlan::compile_axes(
                KernelDType::F32,
                op,
                &case.dims,
                &case.strides,
                &stat_dims,
                &stat_strides,
                &[case.axis],
            )
            .unwrap()
        };
        Self {
            sum: plan(ReduceOp::Sum),
            sumsq: plan(ReduceOp::SumSquares),
            stat_dims,
            stat_strides,
            bcast_strides,
            param_strides,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        ctx: &ExecContext,
        case: &NormCase,
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        stat: &mut [f32],
        scaled: &mut [f32],
        centered: &mut [f32],
        out: &mut [f32],
    ) {
        let dims = &case.dims;
        let st = &case.strides;
        let n = dims[case.axis] as f32;
        let x_ref = ErasedRawStridedRef::from_slice(x, dims, st, 0).unwrap();
        // mean = sum(x) / n
        {
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(stat, &self.stat_dims, &self.stat_strides, 0)
                    .unwrap();
            self.sum.execute(ctx, &mut dest, &x_ref).unwrap();
        }
        map_into(
            &mut StridedViewMut::new(scaled, &[stat.len()], &[1], 0).unwrap(),
            &StridedView::<f32>::new(stat, &[stat.len()], &[1], 0).unwrap(),
            |s| s / n,
        )
        .unwrap();
        // centered = x - mean
        let zip = |op, out: &mut [f32], lhs: &[f32], ls: &[isize], rhs: &[f32], rs: &[isize]| {
            let lhs = ErasedRawStridedRef::from_slice(lhs, dims, ls, 0).unwrap();
            let rhs = ErasedRawStridedRef::from_slice(rhs, dims, rs, 0).unwrap();
            let mut dest = ErasedRawStridedMut::from_slice_mut(out, dims, st, 0).unwrap();
            erased_zip_into(
                KernelDType::F32,
                op,
                ctx,
                &mut dest,
                &ErasedRawStridedPtr::from_ref(&lhs),
                &ErasedRawStridedPtr::from_ref(&rhs),
            )
            .unwrap();
        };
        zip(
            ErasedZipOp::Subtract,
            centered,
            x,
            st,
            scaled,
            &self.bcast_strides,
        );
        // inv = rsqrt(sum(centered^2) / n + eps)
        {
            let c_ref = ErasedRawStridedRef::from_slice(centered, dims, st, 0).unwrap();
            let mut dest =
                ErasedRawStridedMut::from_slice_mut(stat, &self.stat_dims, &self.stat_strides, 0)
                    .unwrap();
            self.sumsq.execute(ctx, &mut dest, &c_ref).unwrap();
        }
        map_into(
            &mut StridedViewMut::new(scaled, &[stat.len()], &[1], 0).unwrap(),
            &StridedView::<f32>::new(stat, &[stat.len()], &[1], 0).unwrap(),
            |s| 1.0 / (s / n + 1e-5).sqrt(),
        )
        .unwrap();
        zip(
            ErasedZipOp::Multiply,
            out,
            centered,
            st,
            scaled,
            &self.bcast_strides,
        );
        centered.copy_from_slice(out);
        zip(
            ErasedZipOp::Multiply,
            out,
            centered,
            st,
            weight,
            &self.param_strides,
        );
        centered.copy_from_slice(out);
        zip(
            ErasedZipOp::Add,
            out,
            centered,
            st,
            bias,
            &self.param_strides,
        );
    }
}

/// Raw pointer two-pass layer norm over a unit-stride axis (column-major
/// lines), the credible naive baseline.
fn naive_layer_norm(x: &[f32], d: usize, weight: &[f32], bias: &[f32], out: &mut [f32]) {
    for (line, out) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mean = line.iter().sum::<f32>() / d as f32;
        let var = line.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for k in 0..d {
            out[k] = (line[k] - mean) * inv * weight[k] + bias[k];
        }
    }
}

fn bench_norm(pool4: &rayon::ThreadPool) {
    for case in norm_cases() {
        let len: usize = case.dims.iter().product();
        let d = case.dims[case.axis];
        let d_dims = [d];
        let x = values(len, 1);
        let weight = values(d, 2);
        let bias = values(d, 3);
        let mut out = vec![0.0f32; len];
        let mut centered = vec![0.0f32; len];
        let mut stat = vec![0.0f32; len / d];
        let mut scaled = vec![0.0f32; len / d];
        let spec = NormSpec::layer_norm(1e-5).with_weight(1).with_bias(1);
        let fused = ErasedNormPlan::compile(
            KernelDType::F32,
            spec,
            &case.dims,
            &case.strides,
            &case.strides,
            case.axis,
        )
        .unwrap();
        let composed = Composed::new(&case);

        for threads in [1usize, 4] {
            let ctx = if threads == 1 {
                ExecContext::serial()
            } else {
                ExecContext::max_threads(threads).unwrap()
            };
            let run = |f: &mut (dyn FnMut() + Send)| {
                if threads == 1 {
                    median_ms(len, f)
                } else {
                    pool4.install(|| {
                        assert_eq!(rayon::current_num_threads(), 4);
                        median_ms(len, f)
                    })
                }
            };
            let fused_ms = run(&mut || {
                let x = ErasedRawStridedRef::from_slice(&x, &case.dims, &case.strides, 0).unwrap();
                let w = ErasedRawStridedRef::from_slice(&weight, &d_dims, &[1], 0).unwrap();
                let b = ErasedRawStridedRef::from_slice(&bias, &d_dims, &[1], 0).unwrap();
                let mut dest =
                    ErasedRawStridedMut::from_slice_mut(&mut out, &case.dims, &case.strides, 0)
                        .unwrap();
                fused
                    .execute(&ctx, &mut dest, &x, Some(&w), Some(&b))
                    .unwrap();
                black_box(&mut dest);
            });
            let composed_ms = run(&mut || {
                composed.run(
                    &ctx,
                    &case,
                    &x,
                    &weight,
                    &bias,
                    &mut stat,
                    &mut scaled,
                    &mut centered,
                    &mut out,
                );
                black_box(&mut out);
            });
            let naive_ms = if case.axis == 0 && threads == 1 {
                format!(
                    "{:.6}",
                    median_ms(len, || {
                        naive_layer_norm(&x, d, &weight, &bias, &mut out);
                        black_box(&mut out);
                    })
                )
            } else {
                "-".to_string()
            };
            println!(
                "SUMMARY,layer_norm,{},{threads}T,fused={fused_ms:.6},composed={composed_ms:.6},naive={naive_ms},composed/fused={:.2}",
                case.label,
                composed_ms / fused_ms
            );
        }
    }
}

fn bench_scan_and_arg(pool4: &rayon::ThreadPool) {
    // (label, dims, axis): a unit-stride scanned axis, a strided one (panel
    // kernels) and a large unit-stride case.
    let cases: [(&str, [usize; 2], usize); 3] = [
        ("unit_axis_1024x64", [1024, 64], 0),
        ("strided_axis_1024x64", [1024, 64], 1),
        ("unit_axis_4096x4096", [4096, 4096], 0),
    ];
    for (label, dims, axis) in cases {
        let len = dims[0] * dims[1];
        let st = col_major(&dims);
        let x = values(len, 5);
        let mut out = vec![0.0f32; len];
        let out_dims = [dims[1 - axis]];
        let mut idx = vec![0i64; out_dims[0]];
        let scan = ErasedScanPlan::compile(
            KernelDType::F32,
            ScanOp::Sum,
            &dims,
            &st,
            &st,
            axis,
            ScanOptions::new(),
        )
        .unwrap();
        let arg = ErasedArgReducePlan::compile(
            KernelDType::F32,
            KernelDType::I64,
            ArgReduceOp::Max,
            &dims,
            &st,
            &out_dims,
            &[1],
            axis,
        )
        .unwrap();
        for threads in [1usize, 4] {
            let ctx = if threads == 1 {
                ExecContext::serial()
            } else {
                ExecContext::max_threads(threads).unwrap()
            };
            let run = |f: &mut (dyn FnMut() + Send)| {
                if threads == 1 {
                    median_ms(len, f)
                } else {
                    pool4.install(|| {
                        assert_eq!(rayon::current_num_threads(), 4);
                        median_ms(len, f)
                    })
                }
            };
            let scan_ms = run(&mut || {
                let src = ErasedRawStridedRef::from_slice(&x, &dims, &st, 0).unwrap();
                let mut dest =
                    ErasedRawStridedMut::from_slice_mut(&mut out, &dims, &st, 0).unwrap();
                scan.execute(&ctx, &mut dest, &src).unwrap();
                black_box(&mut dest);
            });
            let arg_ms = run(&mut || {
                let src = ErasedRawStridedRef::from_slice(&x, &dims, &st, 0).unwrap();
                let mut dest =
                    ErasedRawStridedMut::from_slice_mut(&mut idx, &out_dims, &[1], 0).unwrap();
                arg.execute(&ctx, &mut dest, &src).unwrap();
                black_box(&mut dest);
            });
            let (naive_scan, naive_arg) = if threads == 1 {
                let (n, lines, ls, ss) = if axis == 0 {
                    (dims[0], dims[1], dims[0], 1)
                } else {
                    (dims[1], dims[0], 1, dims[0])
                };
                let scan_ms = median_ms(len, || {
                    let src = x.as_ptr();
                    let dst = out.as_mut_ptr();
                    for line in 0..lines {
                        let mut acc = 0.0f32;
                        for k in 0..n {
                            let o = line * ls + k * ss;
                            // SAFETY: o < len for every line and k.
                            unsafe {
                                acc += *src.add(o);
                                *dst.add(o) = acc;
                            }
                        }
                    }
                    black_box(&mut out);
                });
                let arg_ms = median_ms(len, || {
                    let src = x.as_ptr();
                    for (line, idx) in idx.iter_mut().enumerate() {
                        let mut best = f32::NEG_INFINITY;
                        let mut best_k = 0;
                        for k in 0..n {
                            // SAFETY: the offset is < len.
                            let v = unsafe { *src.add(line * ls + k * ss) };
                            if v > best {
                                best = v;
                                best_k = k;
                            }
                        }
                        *idx = best_k as i64;
                    }
                    black_box(&mut idx);
                });
                (format!("{scan_ms:.6}"), format!("{arg_ms:.6}"))
            } else {
                ("-".to_string(), "-".to_string())
            };
            println!("SUMMARY,cumsum,{label},{threads}T,plan={scan_ms:.6},naive={naive_scan}");
            println!("SUMMARY,argmax,{label},{threads}T,plan={arg_ms:.6},naive={naive_arg}");
        }
    }
}

fn main() {
    if std::env::args().any(|arg| arg == "--list") {
        return;
    }
    let pool4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    bench_norm(&pool4);
    bench_scan_and_arg(&pool4);
}
