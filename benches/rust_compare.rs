use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::hint::black_box;
use std::time::{Duration, Instant};
use strided_rs::{
    copy_into, copy_transpose_scale_into, map_into, zip_map2_into, zip_map4_into, StridedArray,
};

fn mean(durations: &[Duration]) -> Duration {
    let total_nanos: u128 = durations.iter().map(|d| d.as_nanos()).sum();
    Duration::from_nanos((total_nanos / durations.len() as u128) as u64)
}

fn bench_n(label: &str, warmup_iters: usize, iters: usize, mut f: impl FnMut()) -> Duration {
    for _ in 0..warmup_iters {
        f();
    }

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed());
    }

    let avg = mean(&samples);
    println!("{label}: {:.3} ms", avg.as_secs_f64() * 1e3);
    avg
}

fn make_random_2d(n: usize, seed: u64) -> StridedArray<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    StridedArray::<f64>::from_fn_row_major(&[n, n], |_| rng.sample(StandardNormal))
}

fn make_random_4d(n: usize, seed: u64) -> StridedArray<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    StridedArray::<f64>::from_fn_row_major(&[n, n, n, n], |_| rng.sample(StandardNormal))
}

fn main() {
    println!("Rust runner: benches/rust_compare.rs");
    println!("Note: single-threaded runner for parity with Julia.");
    println!();

    // 1) symmetrize_4000
    {
        println!("=== Benchmark 1: symmetrize_4000 ===");
        let n = 4000usize;
        let a = make_random_2d(n, 0);
        let a_view = a.view();
        let a_t = a_view.permute(&[1, 0]).unwrap();
        let mut b = StridedArray::<f64>::row_major(&[n, n]);

        bench_n("rust_naive", 1, 3, || {
            let av = a.view();
            for i in 0..n {
                for j in 0..n {
                    let x = av.get(&[i, j]);
                    let y = av.get(&[j, i]);
                    b.set(&[i, j], 0.5 * (x + y));
                }
            }
            black_box(&b);
        });

        bench_n("rust_strided", 1, 3, || {
            zip_map2_into(&mut b.view_mut(), &a_view, &a_t, |x, y| (x + y) * 0.5).unwrap();
            black_box(&b);
        });
        println!();
    }

    // 2) scale_transpose_1000
    {
        println!("=== Benchmark 2: scale_transpose_1000 ===");
        let n = 1000usize;
        let a = make_random_2d(n, 1);
        let a_view = a.view();
        let mut b = StridedArray::<f64>::row_major(&[n, n]);

        bench_n("rust_naive", 5, 10, || {
            let av = a.view();
            for i in 0..n {
                for j in 0..n {
                    b.set(&[i, j], 3.0 * av.get(&[j, i]));
                }
            }
            black_box(&b);
        });

        bench_n("rust_strided", 5, 10, || {
            copy_transpose_scale_into(&mut b.view_mut(), &a_view, 3.0).unwrap();
            black_box(&b);
        });
        println!();
    }

    // 2a) mwe_stridedview_scale_transpose_1000 (map_into)
    {
        println!("=== Benchmark 2a: mwe_stridedview_scale_transpose_1000 ===");
        let n = 1000usize;
        let mut rng = StdRng::seed_from_u64(11);
        let a = StridedArray::<f64>::from_fn_row_major(&[n, n], |_| rng.gen::<f64>());
        let a_view = a.view();
        let a_t = a_view.permute(&[1, 0]).unwrap();
        let mut b = StridedArray::<f64>::row_major(&[n, n]);

        bench_n("rust_naive", 5, 10, || {
            let av = a.view();
            for i in 0..n {
                for j in 0..n {
                    b.set(&[i, j], 3.0 * av.get(&[j, i]));
                }
            }
            black_box(&b);
        });

        bench_n("rust_strided_map", 5, 10, || {
            map_into(&mut b.view_mut(), &a_t, |x| 3.0 * x).unwrap();
            black_box(&b);
        });

        println!();
    }

    // 3) complex_elementwise_1000 (Float64)
    {
        println!("=== Benchmark 3: complex_elementwise_1000 (Float64) ===");
        let n = 1000usize;
        let a = make_random_2d(n, 2);
        let a_view = a.view();
        let mut b = StridedArray::<f64>::row_major(&[n, n]);

        bench_n("rust_naive", 3, 6, || {
            let av = a.view();
            for i in 0..n {
                for j in 0..n {
                    let x = av.get(&[i, j]);
                    b.set(&[i, j], x * (-2.0 * x).exp() + (x * x).sin());
                }
            }
            black_box(&b);
        });

        bench_n("rust_strided", 3, 6, || {
            map_into(&mut b.view_mut(), &a_view, |x| {
                x * (-2.0 * x).exp() + (x * x).sin()
            })
            .unwrap();
            black_box(&b);
        });
        println!();
    }

    // 4) permute_32_4d
    {
        println!("=== Benchmark 4: permute_32_4d ===");
        let n = 32usize;
        let a = make_random_4d(n, 3);
        let a_view = a.view();
        let a_perm = a_view.permute(&[3, 2, 1, 0]).unwrap();
        let mut b = StridedArray::<f64>::row_major(&[n, n, n, n]);

        bench_n("rust_naive", 20, 50, || {
            let ap = a.view().permute(&[3, 2, 1, 0]).unwrap();
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        for l in 0..n {
                            b.set(&[i, j, k, l], ap.get(&[i, j, k, l]));
                        }
                    }
                }
            }
            black_box(&b);
        });

        bench_n("rust_strided", 20, 50, || {
            copy_into(&mut b.view_mut(), &a_perm).unwrap();
            black_box(&b);
        });
        println!();
    }

    // 5) multiple_permute_sum_32_4d
    {
        println!("=== Benchmark 5: multiple_permute_sum_32_4d ===");
        let n = 32usize;
        let a = make_random_4d(n, 4);
        let a_view = a.view();

        let p1 = a_view.permute(&[0, 1, 2, 3]).unwrap();
        let p2 = a_view.permute(&[1, 2, 3, 0]).unwrap();
        let p3 = a_view.permute(&[2, 3, 0, 1]).unwrap();
        let p4 = a_view.permute(&[3, 0, 1, 2]).unwrap();
        let mut b = StridedArray::<f64>::row_major(&[n, n, n, n]);

        bench_n("rust_naive", 10, 30, || {
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        for l in 0..n {
                            let idx = &[i, j, k, l];
                            b.set(idx, p1.get(idx) + p2.get(idx) + p3.get(idx) + p4.get(idx));
                        }
                    }
                }
            }
            black_box(&b);
        });

        bench_n("rust_strided_fused", 10, 30, || {
            zip_map4_into(&mut b.view_mut(), &p1, &p2, &p3, &p4, |a, b, c, d| {
                a + b + c + d
            })
            .unwrap();
            black_box(&b);
        });
        println!();
    }
}
