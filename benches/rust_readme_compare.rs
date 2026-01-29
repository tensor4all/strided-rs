use mdarray::Tensor;
use strided_rs::{
    copy_into_pod, copy_transpose_scale_into_fast, map_into, symmetrize_into_f64, zip_map4_into,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn mean(durations: &[Duration]) -> Duration {
    let total_nanos: u128 = durations.iter().map(|d| d.as_nanos()).sum();
    Duration::from_nanos((total_nanos / durations.len() as u128) as u64)
}

fn bench_n(
    label: &str,
    warmup_iters: usize,
    iters: usize,
    mut f: impl FnMut(),
) -> Duration {
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

fn main() {
    println!("Rust runner: benches/rust_readme_compare.rs");
    println!("Note: set RAYON_NUM_THREADS=1 for parity (this runner is single-threaded).");
    println!();

    // 1) symmetrize_4000
    {
        println!("=== Benchmark 1: symmetrize_4000 ===");
        let n = 4000usize;
        let mut rng = StdRng::seed_from_u64(0);
        let a = Tensor::<f64, _>::from_fn([n, n], |_| rng.gen::<f64>());
        let a_view = a.as_ref();
        let mut b = Tensor::<f64, _>::zeros([n, n]);

        bench_n("rust_naive", 1, 3, || {
            for i in 0..n {
                for j in 0..n {
                    b[[i, j]] = 0.5 * (a_view[[i, j]] + a_view[[j, i]]);
                }
            }
            black_box(&b);
        });

        bench_n("rust_strided", 1, 3, || {
            symmetrize_into_f64(&mut b, a_view).unwrap();
            black_box(&b);
        });
        println!();
    }

    // 2) scale_transpose_1000
    {
        println!("=== Benchmark 2: scale_transpose_1000 ===");
        let n = 1000usize;
        let mut rng = StdRng::seed_from_u64(1);
        let a = Tensor::<f64, _>::from_fn([n, n], |_| rng.sample(StandardNormal));
        let a_view = a.as_ref();
        let mut b = Tensor::<f64, _>::zeros([n, n]);

        bench_n("rust_naive", 5, 10, || {
            for i in 0..n {
                for j in 0..n {
                    b[[i, j]] = 3.0 * a_view[[j, i]];
                }
            }
            black_box(&b);
        });

        bench_n("rust_strided", 5, 10, || {
            copy_transpose_scale_into_fast(&mut b, a_view, 3.0).unwrap();
            black_box(&b);
        });
        println!();
    }

    // 3) complex_elementwise_1000 (note: Julia script uses Float64, not Complex)
    {
        println!("=== Benchmark 3: complex_elementwise_1000 (Float64) ===");
        let n = 1000usize;
        let mut rng = StdRng::seed_from_u64(2);
        let a = Tensor::<f64, _>::from_fn([n, n], |_| rng.sample(StandardNormal));
        let a_view = a.as_ref();
        let mut b = Tensor::<f64, _>::zeros([n, n]);

        bench_n("rust_naive", 3, 6, || {
            for i in 0..n {
                for j in 0..n {
                    let x = a_view[[i, j]];
                    b[[i, j]] = x * (-2.0 * x).exp() + (x * x).sin();
                }
            }
            black_box(&b);
        });

        bench_n("rust_strided", 3, 6, || {
            map_into(&mut b, a_view, |x| x * (-2.0 * x).exp() + (x * x).sin()).unwrap();
            black_box(&b);
        });
        println!();
    }

    // 4) permute_32_4d
    {
        println!("=== Benchmark 4: permute_32_4d ===");
        let n = 32usize;
        let mut rng = StdRng::seed_from_u64(3);
        let a = Tensor::<f64, _>::from_fn([n, n, n, n], |_| rng.sample(StandardNormal));
        let mut b = Tensor::<f64, _>::zeros([n, n, n, n]);
        let a_perm = a.as_ref().permute([3, 2, 1, 0]);

        bench_n("mdarray_assign", 20, 50, || {
            b.assign(&a_perm);
            black_box(&b);
        });

        bench_n("rust_strided", 20, 50, || {
            copy_into_pod(&mut b, &a_perm).unwrap();
            black_box(&b);
        });
        println!();
    }

    // 5) multiple_permute_sum_32_4d
    {
        println!("=== Benchmark 5: multiple_permute_sum_32_4d ===");
        let n = 32usize;
        let mut rng = StdRng::seed_from_u64(4);
        let a = Tensor::<f64, _>::from_fn([n, n, n, n], |_| rng.sample(StandardNormal));
        let mut b = Tensor::<f64, _>::zeros([n, n, n, n]);
        let a_view = a.as_ref();

        let p1 = a_view.permute([0, 1, 2, 3]);
        let p2 = a_view.permute([1, 2, 3, 0]);
        let p3 = a_view.permute([2, 3, 0, 1]);
        let p4 = a_view.permute([3, 0, 1, 2]);

        // Julia "Base" equivalent: allocate temporaries for each permutedims() then sum.
        bench_n("mdarray_alloc4", 10, 30, || {
            let t1 = p1.to_tensor();
            let t2 = p2.to_tensor();
            let t3 = p3.to_tensor();
            let t4 = p4.to_tensor();
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        for l in 0..n {
                            b[[i, j, k, l]] = t1[[i, j, k, l]]
                                + t2[[i, j, k, l]]
                                + t3[[i, j, k, l]]
                                + t4[[i, j, k, l]];
                        }
                    }
                }
            }
            black_box(&b);
        });

        // Julia "@strided" equivalent: fused single-pass (no temporaries).
        bench_n("rust_strided_fused", 10, 30, || {
            zip_map4_into(&mut b, &p1, &p2, &p3, &p4, |a, b, c, d| a + b + c + d).unwrap();
            black_box(&b);
        });
        println!();
    }
}
