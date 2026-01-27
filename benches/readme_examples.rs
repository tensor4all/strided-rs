use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mdarray::Tensor;
use mdarray_strided::{copy_into, copy_transpose_scale_into_fast, zip_map2_into, zip_map4_into};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::time::Duration;

// Benchmark 1: B = (A + A') / 2 for 4000x4000 matrix
fn bench_symmetrize_4000(c: &mut Criterion) {
    let mut group = c.benchmark_group("readme/symmetrize_4000");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let mut rng = StdRng::seed_from_u64(42);
    let a = Tensor::<f64, _>::from_fn([4000, 4000], |_| rng.sample(StandardNormal));
    let mut b = Tensor::<f64, _>::zeros([4000, 4000]);

    group.bench_function("base", |bencher| {
        bencher.iter(|| {
            for i in 0..4000 {
                for j in 0..4000 {
                    b[[i, j]] = (a[[i, j]] + a[[j, i]]) / 2.0;
                }
            }
            black_box(&b);
        })
    });

    group.bench_function("strided", |bencher| {
        bencher.iter(|| {
            // Match Julia's algorithm: B .= (A .+ A') ./ 2
            // This processes all N×N elements, not just upper triangle
            let a_view = a.as_ref();
            let a_t = a_view.permute([1, 0]);
            zip_map2_into(&mut b, a_view, &a_t, |&x, &y| (x + y) / 2.0).unwrap();
            black_box(&b);
        })
    });

    group.finish();
}

// Benchmark 2: B = 3 * A' for 1000x1000 matrix
fn bench_scale_transpose_1000(c: &mut Criterion) {
    let mut group = c.benchmark_group("readme/scale_transpose_1000");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let mut rng = StdRng::seed_from_u64(43);
    let a = Tensor::<f64, _>::from_fn([1000, 1000], |_| rng.sample(StandardNormal));
    let mut b = Tensor::<f64, _>::zeros([1000, 1000]);

    group.bench_function("base", |bencher| {
        bencher.iter(|| {
            for i in 0..1000 {
                for j in 0..1000 {
                    b[[i, j]] = 3.0 * a[[j, i]];
                }
            }
            black_box(&b);
        })
    });

    group.bench_function("strided", |bencher| {
        bencher.iter(|| {
            copy_transpose_scale_into_fast(&mut b, a.as_ref(), 3.0).unwrap();
            black_box(&b);
        })
    });

    group.finish();
}

// Benchmark 3: B = A * exp(-2*A) + sin(A*A) for 1000x1000 matrix
fn bench_complex_elementwise_1000(c: &mut Criterion) {
    let mut group = c.benchmark_group("readme/complex_elementwise_1000");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let mut rng = StdRng::seed_from_u64(44);
    let a = Tensor::<f64, _>::from_fn([1000, 1000], |_| rng.sample(StandardNormal));
    let mut b = Tensor::<f64, _>::zeros([1000, 1000]);

    group.bench_function("base", |bencher| {
        bencher.iter(|| {
            for i in 0..1000 {
                for j in 0..1000 {
                    let val = a[[i, j]];
                    b[[i, j]] = val * (-2.0 * val).exp() + (val * val).sin();
                }
            }
            black_box(&b);
        })
    });

    group.bench_function("strided", |bencher| {
        bencher.iter(|| {
            mdarray_strided::map_into(&mut b, a.as_ref(), |&x| {
                x * (-2.0 * x).exp() + (x * x).sin()
            })
            .unwrap();
            black_box(&b);
        })
    });

    group.finish();
}

// Benchmark 4: permutedims!(B, A, (4,3,2,1)) for 32x32x32x32 array
fn bench_permute_32_4d(c: &mut Criterion) {
    let mut group = c.benchmark_group("readme/permute_32_4d");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let mut rng = StdRng::seed_from_u64(45);
    let a = Tensor::<f64, _>::from_fn([32, 32, 32, 32], |_| rng.sample(StandardNormal));
    let mut b = Tensor::<f64, _>::zeros([32, 32, 32, 32]);

    // Julia permutation (4,3,2,1) is 1-based, so in Rust (0-based) it's (3,2,1,0)
    let perm = [3, 2, 1, 0];

    group.bench_function("base", |bencher| {
        bencher.iter(|| {
            for o0 in 0..32 {
                for o1 in 0..32 {
                    for o2 in 0..32 {
                        for o3 in 0..32 {
                            let out_idx = [o0, o1, o2, o3];
                            let mut in_idx = [0usize; 4];
                            in_idx[perm[0]] = out_idx[0];
                            in_idx[perm[1]] = out_idx[1];
                            in_idx[perm[2]] = out_idx[2];
                            in_idx[perm[3]] = out_idx[3];
                            b[[o0, o1, o2, o3]] =
                                a[[in_idx[0], in_idx[1], in_idx[2], in_idx[3]]];
                        }
                    }
                }
            }
            black_box(&b);
        })
    });

    group.bench_function("strided", |bencher| {
        bencher.iter(|| {
            let a_view = a.as_ref();
            let a_perm = a_view.permute(perm);
            copy_into(&mut b, &a_perm).unwrap();
            black_box(&b);
        })
    });

    group.finish();
}

// Benchmark 5: B = permutedims(A,(1,2,3,4)) + permutedims(A,(2,3,4,1)) +
//                  permutedims(A,(3,4,1,2)) + permutedims(A,(4,1,2,3))
fn bench_multiple_permute_sum_32_4d(c: &mut Criterion) {
    let mut group = c.benchmark_group("readme/multiple_permute_sum_32_4d");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let mut rng = StdRng::seed_from_u64(46);
    let a = Tensor::<f64, _>::from_fn([32, 32, 32, 32], |_| rng.sample(StandardNormal));
    let mut b = Tensor::<f64, _>::zeros([32, 32, 32, 32]);

    // Julia permutations (1-based): (1,2,3,4), (2,3,4,1), (3,4,1,2), (4,1,2,3)
    // Rust permutations (0-based): (0,1,2,3), (1,2,3,0), (2,3,0,1), (3,0,1,2)
    let perms: [[usize; 4]; 4] = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]];

    group.bench_function("base", |bencher| {
        bencher.iter(|| {
            // First zero out B
            for i in 0..32 {
                for j in 0..32 {
                    for k in 0..32 {
                        for l in 0..32 {
                            b[[i, j, k, l]] = 0.0;
                        }
                    }
                }
            }

            // Add each permutation
            for perm in &perms {
                for o0 in 0..32 {
                    for o1 in 0..32 {
                        for o2 in 0..32 {
                            for o3 in 0..32 {
                                let out_idx = [o0, o1, o2, o3];
                                let mut in_idx = [0usize; 4];
                                in_idx[perm[0]] = out_idx[0];
                                in_idx[perm[1]] = out_idx[1];
                                in_idx[perm[2]] = out_idx[2];
                                in_idx[perm[3]] = out_idx[3];
                                b[[o0, o1, o2, o3]] +=
                                    a[[in_idx[0], in_idx[1], in_idx[2], in_idx[3]]];
                            }
                        }
                    }
                }
            }
            black_box(&b);
        })
    });

    group.bench_function("strided", |bencher| {
        bencher.iter(|| {
            let a_view = a.as_ref();

            // Compute all 4 permutations at once using zip_map4_into
            let a_perm0 = a_view.permute(perms[0]);
            let a_perm1 = a_view.permute(perms[1]);
            let a_perm2 = a_view.permute(perms[2]);
            let a_perm3 = a_view.permute(perms[3]);
            zip_map4_into(&mut b, &a_perm0, &a_perm1, &a_perm2, &a_perm3, |&p0, &p1, &p2, &p3| p0 + p1 + p2 + p3).unwrap();
            
            black_box(&b);
        })
    });

    group.finish();
}

criterion_group!(
    readme_examples,
    bench_symmetrize_4000,
    bench_scale_transpose_1000,
    bench_complex_elementwise_1000,
    bench_permute_32_4d,
    bench_multiple_permute_sum_32_4d,
);
criterion_main!(readme_examples);
