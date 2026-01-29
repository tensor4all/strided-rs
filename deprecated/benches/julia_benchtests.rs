#![cfg(feature = "mdarray")]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mdarray::Tensor;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::time::Duration;
use strided_rs::{
    copy_into, mapreducedim_capture_views_into, sum, Arg, CaptureArgs, Identity, StridedArrayView,
    StridedArrayViewMut,
};

fn julia_sizes() -> Vec<usize> {
    // Julia: sizes = ceil.(Int, 2 .^ (2:1.5:20))
    let mut out = Vec::new();
    let mut x = 2.0_f64;
    while x <= 20.0 + 1e-12 {
        out.push((2.0_f64.powf(x)).ceil() as usize);
        x += 1.5;
    }
    out
}

fn bench_sum_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("benchtests/sum_1d");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(10);

    let sizes = julia_sizes();
    // Keep memory bounded. Julia goes up to ~1e6 elements, which is fine for 1D.
    let max_len = 262_144usize; // Reduced from 1M to 256K

    for &s in sizes.iter().filter(|&&s| s <= max_len) {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE ^ (s as u64));
        let a = Tensor::<f64, _>::from_fn([s], |_| rng.sample(StandardNormal));

        group.throughput(Throughput::Elements(s as u64));

        group.bench_with_input(BenchmarkId::new("base", s), &s, |b, _| {
            b.iter(|| {
                let mut acc = 0.0f64;
                for i in 0..s {
                    acc += a[[i]];
                }
                black_box(acc)
            })
        });

        group.bench_with_input(BenchmarkId::new("strided", s), &s, |b, _| {
            b.iter(|| black_box(sum(a.as_ref()).unwrap()))
        });
    }

    group.finish();
}

fn permute_sizes_for_4d() -> Vec<usize> {
    // Julia uses the same `sizes`, but 4D (s,s,s,s) explodes quickly.
    // We pick the same generated list but stop once s^4 gets too large.
    let mut out = Vec::new();
    for s in julia_sizes() {
        let s4 = (s as u128)
            .saturating_mul(s as u128)
            .saturating_mul(s as u128)
            .saturating_mul(s as u128);
        // Keep <= ~4M elements (~32MB for f64) to finish quickly.
        if s4 > 4_000_000u128 {
            break;
        }
        out.push(s);
    }
    out
}

fn bench_permute_4d(c: &mut Criterion) {
    let mut group = c.benchmark_group("benchtests/permute_4d");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(10);

    // Julia permutations (1-based):
    // (4,3,2,1), (2,3,4,1), (3,4,1,2)
    // Rust 0-based:
    let perms: [([usize; 4], &str); 3] = [
        ([3, 2, 1, 0], "p4321"),
        ([1, 2, 3, 0], "p2341"),
        ([2, 3, 0, 1], "p3412"),
    ];

    for &s in &permute_sizes_for_4d() {
        let total = (s as u64) * (s as u64) * (s as u64) * (s as u64);
        group.throughput(Throughput::Elements(total));

        let mut rng = StdRng::seed_from_u64(0xBEEF ^ (s as u64));
        let a = Tensor::<f64, _>::from_fn([s, s, s, s], |_| rng.sample(StandardNormal));
        let a_view = a.as_ref();

        // Baseline copy!(B, A)
        group.bench_with_input(BenchmarkId::new("copy", s), &s, |b, _| {
            let mut b_out = Tensor::<f64, _>::zeros([s, s, s, s]);
            b.iter(|| {
                copy_into(&mut b_out, a_view).unwrap();
                black_box(&b_out);
            })
        });

        for (perm, perm_name) in perms {
            // Baseline permutedims!(B, A, p) equivalent: explicit index remap.
            group.bench_with_input(
                BenchmarkId::new(format!("permute_base/{perm_name}"), s),
                &s,
                |b, _| {
                    let mut b_out = Tensor::<f64, _>::zeros([s, s, s, s]);
                    b.iter(|| {
                        for o0 in 0..s {
                            for o1 in 0..s {
                                for o2 in 0..s {
                                    for o3 in 0..s {
                                        let out_idx = [o0, o1, o2, o3];
                                        let mut in_idx = [0usize; 4];
                                        in_idx[perm[0]] = out_idx[0];
                                        in_idx[perm[1]] = out_idx[1];
                                        in_idx[perm[2]] = out_idx[2];
                                        in_idx[perm[3]] = out_idx[3];
                                        b_out[[o0, o1, o2, o3]] =
                                            a_view[[in_idx[0], in_idx[1], in_idx[2], in_idx[3]]];
                                    }
                                }
                            }
                        }
                        black_box(&b_out);
                    })
                },
            );

            // Strided path: create a lazy permuted view, then copy into B.
            group.bench_with_input(
                BenchmarkId::new(format!("permute_strided/{perm_name}"), s),
                &s,
                |b, _| {
                    let a_perm = a_view.permute(perm);
                    let mut b_out = Tensor::<f64, _>::zeros([s, s, s, s]);
                    b.iter(|| {
                        copy_into(&mut b_out, &a_perm).unwrap();
                        black_box(&b_out);
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_mapreducedim_capture2(c: &mut Criterion) {
    let mut group = c.benchmark_group("benchtests/mapreducedim_capture2");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(10);

    // Keep runtime bounded; O(s^2) baseline.
    let max_s = 1024usize;

    for &s in julia_sizes().iter().filter(|&&s| s <= max_s) {
        let m = s;
        let n = s;
        group.throughput(Throughput::Elements((m as u64) * (n as u64)));

        let mut rng = StdRng::seed_from_u64(0xABCD ^ (s as u64));
        let mut a_data = Vec::with_capacity(m * n);
        for _ in 0..(m * n) {
            a_data.push(rng.sample(StandardNormal));
        }
        let mut b_data = Vec::with_capacity(n);
        for _ in 0..n {
            b_data.push(rng.sample(StandardNormal));
        }

        let capture = CaptureArgs::new(|x: f64, y: f64| x + y, (Arg, Arg));

        group.bench_with_input(BenchmarkId::new("base", s), &s, |bencher, _| {
            bencher.iter(|| {
                let mut out = vec![0.0f64; n];
                for j in 0..n {
                    let add = b_data[j];
                    let mut acc = 0.0f64;
                    for i in 0..m {
                        acc += a_data[i * n + j] + add;
                    }
                    out[j] = acc;
                }
                black_box(&out);
            })
        });

        group.bench_with_input(BenchmarkId::new("strided", s), &s, |bencher, _| {
            bencher.iter(|| {
                let a_view: StridedArrayView<'_, f64, 2, Identity> =
                    StridedArrayView::new(&a_data, [m, n], [n as isize, 1], 0).unwrap();
                let b_view: StridedArrayView<'_, f64, 2, Identity> =
                    StridedArrayView::new(&b_data, [1, n], [n as isize, 1], 0).unwrap();
                let mut out_data = vec![0.0f64; n];
                let mut dest: StridedArrayViewMut<'_, f64, 2, Identity> =
                    StridedArrayViewMut::new(&mut out_data, [1, n], [n as isize, 1], 0).unwrap();

                mapreducedim_capture_views_into(
                    &mut dest,
                    &[&a_view, &b_view],
                    &capture,
                    |a, b| a + b,
                    None,
                )
                .unwrap();
                black_box(dest.get([0, 0]));
            })
        });
    }

    group.finish();
}

criterion_group!(
    julia_benchtests,
    bench_sum_1d,
    bench_permute_4d,
    bench_mapreducedim_capture2
);
criterion_main!(julia_benchtests);
