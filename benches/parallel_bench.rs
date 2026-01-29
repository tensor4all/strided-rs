//! Parallel benchmarks for strided-rs
//!
//! This benchmark compares sequential vs parallel implementations
//! to measure the benefit of the `parallel` feature.
//!
//! Run with: cargo bench --features parallel --bench parallel_bench

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mdarray::Tensor;
use strided_rs::{par_zip_map2_into, zip_map2_into};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::time::Duration;

/// Compare sequential vs parallel zip_map2_into for mixed strides
///
/// Julia equivalent:
/// ```julia
/// using BenchmarkTools
/// using Strided
/// let
///     A = rand(N, N)
///     B = similar(A)
///     out = similar(A)
///     # Sequential
///     @benchmark $out .= $A' .+ $B
///     # Parallel (with @strided macro and multiple threads)
///     @benchmark @strided $out .= $A' .+ $B
/// end
/// ```
fn bench_par_zip_map2_mixed_strides(c: &mut Criterion) {
    let mut group = c.benchmark_group("par_zip_map2_mixed");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for size in [500, 1000, 2000, 4000] {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));

        let mut rng = StdRng::seed_from_u64(42);
        let a = Tensor::<f64, _>::from_fn([size, size], |_| rng.sample(StandardNormal));
        let b = Tensor::<f64, _>::from_fn([size, size], |_| rng.sample(StandardNormal));
        let a_t = a.as_ref().permute([1, 0]); // Transposed view

        // Sequential version
        group.bench_with_input(BenchmarkId::new("sequential", size), &size, |bench, _| {
            bench.iter(|| {
                let mut out = Tensor::zeros([size, size]);
                zip_map2_into(&mut out, &a_t, &b.as_ref(), |x, y| x + y).unwrap();
                out
            })
        });

        // Parallel version
        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |bench, _| {
            bench.iter(|| {
                let mut out = Tensor::zeros([size, size]);
                par_zip_map2_into(&mut out, &a_t, &b.as_ref(), |x, y| x + y).unwrap();
                out
            })
        });
    }
    group.finish();
}

/// Compare sequential vs parallel for contiguous arrays
///
/// This tests the overhead of parallel dispatch for contiguous data
/// where memory access is already optimal.
fn bench_par_zip_map2_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("par_zip_map2_contiguous");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for size in [500, 1000, 2000, 4000] {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));

        let mut rng = StdRng::seed_from_u64(42);
        let a = Tensor::<f64, _>::from_fn([size, size], |_| rng.sample(StandardNormal));
        let b = Tensor::<f64, _>::from_fn([size, size], |_| rng.sample(StandardNormal));

        // Sequential version
        group.bench_with_input(BenchmarkId::new("sequential", size), &size, |bench, _| {
            bench.iter(|| {
                let mut out = Tensor::zeros([size, size]);
                zip_map2_into(&mut out, &a.as_ref(), &b.as_ref(), |x, y| x + y).unwrap();
                out
            })
        });

        // Parallel version
        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |bench, _| {
            bench.iter(|| {
                let mut out = Tensor::zeros([size, size]);
                par_zip_map2_into(&mut out, &a.as_ref(), &b.as_ref(), |x, y| x + y).unwrap();
                out
            })
        });
    }
    group.finish();
}

/// Benchmark parallel operation with compute-intensive function
///
/// This tests whether parallel dispatch helps when the per-element
/// computation is more expensive.
fn bench_par_zip_map2_compute_heavy(c: &mut Criterion) {
    let mut group = c.benchmark_group("par_zip_map2_compute_heavy");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for size in [500, 1000, 2000] {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));

        let mut rng = StdRng::seed_from_u64(42);
        let a = Tensor::<f64, _>::from_fn([size, size], |_| rng.sample(StandardNormal));
        let b = Tensor::<f64, _>::from_fn([size, size], |_| rng.sample(StandardNormal));

        // Compute-heavy function: exp, sin, cos
        let heavy_fn = |x: &f64, y: &f64| (x * y).exp() + (x + y).sin() * (x - y).cos();

        // Sequential version
        group.bench_with_input(BenchmarkId::new("sequential", size), &size, |bench, _| {
            bench.iter(|| {
                let mut out = Tensor::zeros([size, size]);
                zip_map2_into(&mut out, &a.as_ref(), &b.as_ref(), heavy_fn).unwrap();
                out
            })
        });

        // Parallel version
        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |bench, _| {
            bench.iter(|| {
                let mut out = Tensor::zeros([size, size]);
                par_zip_map2_into(&mut out, &a.as_ref(), &b.as_ref(), heavy_fn).unwrap();
                out
            })
        });
    }
    group.finish();
}

/// Benchmark parallel with both operands transposed (worst case memory access)
fn bench_par_zip_map2_both_transposed(c: &mut Criterion) {
    let mut group = c.benchmark_group("par_zip_map2_both_transposed");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for size in [500, 1000, 2000, 4000] {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));

        let mut rng = StdRng::seed_from_u64(42);
        let a = Tensor::<f64, _>::from_fn([size, size], |_| rng.sample(StandardNormal));
        let b = Tensor::<f64, _>::from_fn([size, size], |_| rng.sample(StandardNormal));
        let a_t = a.as_ref().permute([1, 0]);
        let b_t = b.as_ref().permute([1, 0]);

        // Sequential version
        group.bench_with_input(BenchmarkId::new("sequential", size), &size, |bench, _| {
            bench.iter(|| {
                let mut out = Tensor::zeros([size, size]);
                zip_map2_into(&mut out, &a_t, &b_t, |x, y| x + y).unwrap();
                out
            })
        });

        // Parallel version
        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |bench, _| {
            bench.iter(|| {
                let mut out = Tensor::zeros([size, size]);
                par_zip_map2_into(&mut out, &a_t, &b_t, |x, y| x + y).unwrap();
                out
            })
        });
    }
    group.finish();
}

/// Benchmark parallel 4D operations
fn bench_par_zip_map2_4d(c: &mut Criterion) {
    let mut group = c.benchmark_group("par_zip_map2_4d");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for size in [20, 32, 48] {
        let elements = size * size * size * size;
        group.throughput(Throughput::Elements(elements as u64));

        let mut rng = StdRng::seed_from_u64(42);
        let a =
            Tensor::<f64, _>::from_fn([size, size, size, size], |_| rng.sample(StandardNormal));
        let b =
            Tensor::<f64, _>::from_fn([size, size, size, size], |_| rng.sample(StandardNormal));

        // Permuted views
        let a_perm = a.as_ref().permute([3, 2, 1, 0]);
        let b_perm = b.as_ref().permute([1, 0, 3, 2]);

        // Sequential version
        group.bench_with_input(BenchmarkId::new("sequential", size), &size, |bench, _| {
            bench.iter(|| {
                let mut out = Tensor::zeros([size, size, size, size]);
                zip_map2_into(&mut out, &a_perm, &b_perm, |x, y| x + y).unwrap();
                out
            })
        });

        // Parallel version
        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |bench, _| {
            bench.iter(|| {
                let mut out = Tensor::zeros([size, size, size, size]);
                par_zip_map2_into(&mut out, &a_perm, &b_perm, |x, y| x + y).unwrap();
                out
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_par_zip_map2_mixed_strides,
    bench_par_zip_map2_contiguous,
    bench_par_zip_map2_compute_heavy,
    bench_par_zip_map2_both_transposed,
    bench_par_zip_map2_4d,
);
criterion_main!(benches);
