#![cfg(feature = "mdarray")]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mdarray::Tensor;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::time::Duration;
use strided::{
    copy_into, copy_transpose_scale_into_fast, map_into, sum, symmetrize_into, zip_map2_into,
    zip_map4_into,
};

// Julia equivalent for `bench_copy_permuted`:
// ```julia
// using BenchmarkTools
// using Strided
//let
//    A = rand(1000, 1000)
//    B = similar(A)
//    permAlazy = PermutedDimsArray(A, (2, 1))  # lazy permuted view (no copy)
//    @benchmark $B .= $permAlazy
//    @benchmark @strided $B .= $permAlazy
//end
// ```
fn bench_copy_permuted(c: &mut Criterion) {
    let mut group = c.benchmark_group("copy_permuted");
    for size in [1000usize] {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));

        let a = Tensor::<f64, _>::from_fn([size, size], |idx| (idx[0] * size + idx[1]) as f64);
        let a_t = a.as_ref().permute([1, 0]);

        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b, _| {
            b.iter(|| a_t.to_tensor());
        });

        group.bench_with_input(BenchmarkId::new("strided", size), &size, |b, _| {
            b.iter(|| {
                let mut out = Tensor::zeros([size, size]);
                if let Err(err) = copy_into(&mut out, &a_t) {
                    panic!("copy_into failed: {err}");
                }
                out
            })
        });
    }
    group.finish();
}

// Julia equivalent for `bench_zip_map_mixed_strides`:
// ```julia
// using BenchmarkTools
// using Strided
//let
//    A = rand(1000, 1000)
//    B = similar(A)
//    permAlazy = PermutedDimsArray(A, (2, 1))
//    out = similar(A)
//    @benchmark $out .= $permAlazy .+ $B
//    @benchmark @strided $out .= $permAlazy .+ $B
//end
// ```
fn bench_zip_map_mixed_strides(c: &mut Criterion) {
    let mut group = c.benchmark_group("zip_map_mixed");
    for size in [1000usize] {
        let a = Tensor::<f64, _>::from_fn([size, size], |idx| (idx[0] * size + idx[1]) as f64);
        let b = Tensor::<f64, _>::from_fn([size, size], |idx| (idx[0] + idx[1]) as f64);
        let a_t = a.as_ref().permute([1, 0]);

        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b_iter, _| {
            b_iter.iter(|| {
                let mut out = Tensor::zeros([size, size]);
                for i in 0..size {
                    for j in 0..size {
                        out[[i, j]] = a_t[[i, j]] + b[[i, j]];
                    }
                }
                out
            })
        });

        group.bench_with_input(BenchmarkId::new("strided", size), &size, |b_iter, _| {
            b_iter.iter(|| {
                let mut out = Tensor::zeros([size, size]);
                if let Err(err) = zip_map2_into(&mut out, &a_t, &b.as_ref(), |x, y| x + y) {
                    panic!("zip_map2_into failed: {err}");
                }
                out
            })
        });
    }
    group.finish();
}

// Julia equivalent for `bench_reduce_transposed`:
// ```julia
// using BenchmarkTools
// using Strided
// let
//    A = rand(1000, 1000)
//    @benchmark sum($A')
//    @benchmark @strided sum($A')
// end
// ```
fn bench_reduce_transposed(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce_transposed");
    for size in [1000usize] {
        let a = Tensor::<f64, _>::from_fn([size, size], |idx| (idx[0] * size + idx[1]) as f64);
        let a_t = a.as_ref().permute([1, 0]);

        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b, _| {
            b.iter(|| {
                let mut sum_val = 0.0;
                for i in 0..size {
                    for j in 0..size {
                        sum_val += a_t[[i, j]];
                    }
                }
                sum_val
            })
        });

        group.bench_with_input(BenchmarkId::new("strided", size), &size, |b, _| {
            b.iter(|| match sum(&a_t) {
                Ok(v) => v,
                Err(err) => panic!("sum failed: {err}"),
            })
        });
    }
    group.finish();
}

// Julia equivalent for `bench_symmetrize_aat`:
// ```julia
// using BenchmarkTools
// let
//    A = rand(4000, 4000)
//    B = similar(A)
//    @benchmark $B .= ($A .+ $A') ./ 2;
//    @benchmark @strided $B .= ($A .+ $A') ./ 2;
// end
// ```
fn bench_symmetrize_aat(c: &mut Criterion) {
    let mut group = c.benchmark_group("symmetrize_aat");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    let size = 4000usize;
    let elements = size * size;
    group.throughput(Throughput::Elements(elements as u64));

    let mut rng = StdRng::seed_from_u64(0);
    let a = Tensor::<f64, _>::from_fn([size, size], |_| rng.gen::<f64>());
    let a_view = a.as_ref();
    let a_t = a_view.permute([1, 0]);

    group.bench_function("naive", |b| {
        b.iter(|| {
            let mut out = Tensor::zeros([size, size]);
            for i in 0..size {
                for j in 0..size {
                    out[[i, j]] = 0.5 * (a_view[[i, j]] + a_t[[i, j]]);
                }
            }
            out
        })
    });

    group.bench_function("strided", |b| {
        b.iter(|| {
            let mut out = Tensor::zeros([size, size]);
            if let Err(err) = zip_map2_into(&mut out, a_view, &a_t, |&x, &y| (x + y) * 0.5) {
                panic!("zip_map2_into (symmetrize_aat) failed: {err}");
            }
            out
        })
    });

    group.finish();
}

// Julia equivalent for `bench_scale_transpose`:
// ```julia
// let
//    A = rand(1000, 1000)
//    B = similar(A)
//    @benchmark $B .= 3 .* $A';
//    @benchmark @strided $B .= 3 .* $A';
// end
//```
fn bench_scale_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale_transpose");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    let size = 1000usize;
    let elements = size * size;
    group.throughput(Throughput::Elements(elements as u64));

    let mut rng = StdRng::seed_from_u64(1);
    let a = Tensor::<f64, _>::from_fn([size, size], |_| rng.sample(StandardNormal));
    let a_t = a.as_ref().permute([1, 0]);

    group.bench_function("naive", |b| {
        b.iter(|| {
            let mut out = Tensor::zeros([size, size]);
            for i in 0..size {
                for j in 0..size {
                    out[[i, j]] = 3.0 * a_t[[i, j]];
                }
            }
            out
        })
    });

    group.bench_function("strided", |b| {
        b.iter(|| {
            let mut out = Tensor::zeros([size, size]);
            if let Err(err) = copy_transpose_scale_into_fast(&mut out, a.as_ref(), 3.0) {
                panic!("copy_transpose_scale_into_fast failed: {err}");
            }
            out
        })
    });

    group.finish();
}

// Julia equivalent for `bench_nonlinear_map`:
// ```julia
// let
//    A = rand(1000, 1000)
//    B = similar(A)
//    @benchmark $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);
//    @benchmark @strided $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);
// end
//```
fn bench_nonlinear_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("nonlinear_map");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    let size = 1000usize;
    let elements = size * size;
    group.throughput(Throughput::Elements(elements as u64));

    let mut rng = StdRng::seed_from_u64(2);
    let a = Tensor::<f64, _>::from_fn([size, size], |_| rng.sample(StandardNormal));
    let a_view = a.as_ref();

    group.bench_function("naive", |b| {
        b.iter(|| {
            let mut out = Tensor::zeros([size, size]);
            for i in 0..size {
                for j in 0..size {
                    let x = a_view[[i, j]];
                    out[[i, j]] = x * (-2.0 * x).exp() + (x * x).sin();
                }
            }
            out
        })
    });

    group.bench_function("strided", |b| {
        b.iter(|| {
            let mut out = Tensor::zeros([size, size]);
            if let Err(err) = map_into(&mut out, a_view, |x| x * (-2.0 * x).exp() + (x * x).sin()) {
                panic!("map_into failed: {err}");
            }
            out
        })
    });

    group.finish();
}

// Julia equivalent for `bench_permutedims_4d`:
// ```julia
// using BenchmarkTools
//
// function bench_permutedims_4d_julia()
//     A = randn(32, 32, 32, 32)
//     B = similar(A)
//
//     # Base permutedims!
//     @btime permutedims!($B, $A, (4, 3, 2, 1))
// end
// ```
fn bench_permutedims_4d(c: &mut Criterion) {
    let mut group = c.benchmark_group("permutedims_4d");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    let size = 32usize;
    let elements = size * size * size * size;
    group.throughput(Throughput::Elements(elements as u64));

    let mut rng = StdRng::seed_from_u64(3);
    let a = Tensor::<f64, _>::from_fn([size, size, size, size], |_| rng.sample(StandardNormal));
    let a_view = a.as_ref();
    // Julia (4,3,2,1) -> Rust [3,2,1,0] (0-indexed)
    let a_perm = a_view.permute([3, 2, 1, 0]);

    group.bench_function("naive", |b| {
        b.iter(|| {
            let mut out = Tensor::zeros([size, size, size, size]);
            for i in 0..size {
                for j in 0..size {
                    for k in 0..size {
                        for l in 0..size {
                            out[[i, j, k, l]] = a_perm[[i, j, k, l]];
                        }
                    }
                }
            }
            out
        })
    });

    group.bench_function("strided", |b| {
        b.iter(|| {
            let mut out = Tensor::zeros([size, size, size, size]);
            if let Err(err) = copy_into(&mut out, &a_perm) {
                panic!("copy_into failed: {err}");
            }
            out
        })
    });

    group.finish();
}

// Julia equivalent for `bench_multi_permute_sum`:
// ```julia
// using BenchmarkTools
// using Strided
// function bench_multi_permute_sum_julia()
//     A = randn(32, 32, 32, 32)
//     B = similar(A)
//
//     # naive: allocate temporaries for each permutation
//     @btime $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+
//                  permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3))
//     @btime @strided $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+
//                  permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3))
// end
// ```
fn bench_multi_permute_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_permute_sum");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    let size = 32usize;
    let elements = size * size * size * size;
    group.throughput(Throughput::Elements(elements as u64));

    let mut rng = StdRng::seed_from_u64(4);
    let a = Tensor::<f64, _>::from_fn([size, size, size, size], |_| rng.sample(StandardNormal));
    let a_view = a.as_ref();

    // Julia (1,2,3,4) -> Rust [0,1,2,3] (identity)
    // Julia (2,3,4,1) -> Rust [1,2,3,0]
    // Julia (3,4,1,2) -> Rust [2,3,0,1]
    // Julia (4,1,2,3) -> Rust [3,0,1,2]
    let p1 = a_view.permute([0, 1, 2, 3]); // identity
    let p2 = a_view.permute([1, 2, 3, 0]);
    let p3 = a_view.permute([2, 3, 0, 1]);
    let p4 = a_view.permute([3, 0, 1, 2]);

    group.bench_function("naive", |b| {
        b.iter(|| {
            let mut out = Tensor::zeros([size, size, size, size]);
            for i in 0..size {
                for j in 0..size {
                    for k in 0..size {
                        for l in 0..size {
                            out[[i, j, k, l]] = p1[[i, j, k, l]]
                                + p2[[i, j, k, l]]
                                + p3[[i, j, k, l]]
                                + p4[[i, j, k, l]];
                        }
                    }
                }
            }
            out
        })
    });

    // Using zip_map4_into for fused single-pass operations (no temporaries)
    group.bench_function("strided_fused", |b| {
        b.iter(|| {
            let mut out = Tensor::zeros([size, size, size, size]);
            if let Err(err) =
                zip_map4_into(&mut out, &p1, &p2, &p3, &p4, |a, b, c, d| a + b + c + d)
            {
                panic!("zip_map4_into failed: {err}");
            }
            out
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_copy_permuted,
    bench_zip_map_mixed_strides,
    bench_reduce_transposed,
    bench_symmetrize_aat,
    bench_scale_transpose,
    bench_nonlinear_map,
    bench_permutedims_4d,
    bench_multi_permute_sum
);
criterion_main!(benches);
