//! BLAS benchmarks for mdarray-strided
//!
//! This benchmark compares generic implementations vs BLAS-backed implementations
//! to measure the benefit of the `blas` feature.
//!
//! Run with: cargo bench --features blas --bench blas_bench

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mdarray_strided::{
    blas_axpy, blas_dot, blas_gemm, generic_axpy, generic_dot, generic_gemm, Identity,
    StridedArrayView, StridedArrayViewMut,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::time::Duration;

/// Compare generic vs BLAS dot product
fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for size in [1000, 10000, 100000, 1000000] {
        group.throughput(Throughput::Elements(size as u64));

        let mut rng = StdRng::seed_from_u64(42);
        let x_data: Vec<f64> = (0..size).map(|_| rng.sample(StandardNormal)).collect();
        let y_data: Vec<f64> = (0..size).map(|_| rng.sample(StandardNormal)).collect();

        let x: StridedArrayView<'_, f64, 1, Identity> =
            StridedArrayView::new(&x_data, [size], [1], 0).unwrap();
        let y: StridedArrayView<'_, f64, 1, Identity> =
            StridedArrayView::new(&y_data, [size], [1], 0).unwrap();

        // Generic version
        group.bench_with_input(BenchmarkId::new("generic", size), &size, |bench, _| {
            bench.iter(|| generic_dot(&x, &y).unwrap())
        });

        // BLAS version
        group.bench_with_input(BenchmarkId::new("blas", size), &size, |bench, _| {
            bench.iter(|| blas_dot(&x, &y).unwrap())
        });
    }
    group.finish();
}

/// Compare generic vs BLAS axpy (y = alpha * x + y)
fn bench_axpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("axpy");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for size in [1000, 10000, 100000, 1000000] {
        group.throughput(Throughput::Elements(size as u64));

        let mut rng = StdRng::seed_from_u64(42);
        let x_data: Vec<f64> = (0..size).map(|_| rng.sample(StandardNormal)).collect();
        let y_template: Vec<f64> = (0..size).map(|_| rng.sample(StandardNormal)).collect();

        let x: StridedArrayView<'_, f64, 1, Identity> =
            StridedArrayView::new(&x_data, [size], [1], 0).unwrap();

        // Generic version
        group.bench_with_input(BenchmarkId::new("generic", size), &size, |bench, _| {
            bench.iter(|| {
                let mut y_data = y_template.clone();
                let mut y: StridedArrayViewMut<'_, f64, 1, Identity> =
                    StridedArrayViewMut::new(&mut y_data, [size], [1], 0).unwrap();
                generic_axpy(2.5, &x, &mut y).unwrap();
                y_data
            })
        });

        // BLAS version
        group.bench_with_input(BenchmarkId::new("blas", size), &size, |bench, _| {
            bench.iter(|| {
                let mut y_data = y_template.clone();
                let mut y: StridedArrayViewMut<'_, f64, 1, Identity> =
                    StridedArrayViewMut::new(&mut y_data, [size], [1], 0).unwrap();
                blas_axpy(2.5, &x, &mut y).unwrap();
                y_data
            })
        });
    }
    group.finish();
}

/// Compare generic vs BLAS gemm (matrix multiplication)
fn bench_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for size in [64, 128, 256, 512, 1024] {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));

        let mut rng = StdRng::seed_from_u64(42);
        let a_data: Vec<f64> = (0..elements).map(|_| rng.sample(StandardNormal)).collect();
        let b_data: Vec<f64> = (0..elements).map(|_| rng.sample(StandardNormal)).collect();

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [size, size], [size as isize, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [size, size], [size as isize, 1], 0).unwrap();

        // Generic version (skip very large sizes as it's O(n³))
        if size <= 256 {
            group.bench_with_input(BenchmarkId::new("generic", size), &size, |bench, _| {
                bench.iter(|| {
                    let mut c_data = vec![0.0f64; elements];
                    let mut c: StridedArrayViewMut<'_, f64, 2, Identity> =
                        StridedArrayViewMut::new(&mut c_data, [size, size], [size as isize, 1], 0)
                            .unwrap();
                    generic_gemm(1.0, &a, &b, 0.0, &mut c).unwrap();
                    c_data
                })
            });
        }

        // BLAS version
        group.bench_with_input(BenchmarkId::new("blas", size), &size, |bench, _| {
            bench.iter(|| {
                let mut c_data = vec![0.0f64; elements];
                let mut c: StridedArrayViewMut<'_, f64, 2, Identity> =
                    StridedArrayViewMut::new(&mut c_data, [size, size], [size as isize, 1], 0)
                        .unwrap();
                blas_gemm(1.0, &a, &b, 0.0, &mut c).unwrap();
                c_data
            })
        });
    }
    group.finish();
}

/// Compare gemm with alpha/beta scaling
fn bench_gemm_scaled(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_scaled");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for size in [64, 128, 256, 512] {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));

        let mut rng = StdRng::seed_from_u64(42);
        let a_data: Vec<f64> = (0..elements).map(|_| rng.sample(StandardNormal)).collect();
        let b_data: Vec<f64> = (0..elements).map(|_| rng.sample(StandardNormal)).collect();
        let c_template: Vec<f64> = (0..elements).map(|_| rng.sample(StandardNormal)).collect();

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [size, size], [size as isize, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [size, size], [size as isize, 1], 0).unwrap();

        // Generic version (skip very large sizes)
        if size <= 128 {
            group.bench_with_input(BenchmarkId::new("generic", size), &size, |bench, _| {
                bench.iter(|| {
                    let mut c_data = c_template.clone();
                    let mut c: StridedArrayViewMut<'_, f64, 2, Identity> =
                        StridedArrayViewMut::new(&mut c_data, [size, size], [size as isize, 1], 0)
                            .unwrap();
                    generic_gemm(2.0, &a, &b, 0.5, &mut c).unwrap();
                    c_data
                })
            });
        }

        // BLAS version
        group.bench_with_input(BenchmarkId::new("blas", size), &size, |bench, _| {
            bench.iter(|| {
                let mut c_data = c_template.clone();
                let mut c: StridedArrayViewMut<'_, f64, 2, Identity> =
                    StridedArrayViewMut::new(&mut c_data, [size, size], [size as isize, 1], 0)
                        .unwrap();
                blas_gemm(2.0, &a, &b, 0.5, &mut c).unwrap();
                c_data
            })
        });
    }
    group.finish();
}

/// Benchmark gemm with column-major input (transposed)
fn bench_gemm_column_major(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_column_major");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for size in [64, 128, 256, 512] {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));

        let mut rng = StdRng::seed_from_u64(42);
        let a_data: Vec<f64> = (0..elements).map(|_| rng.sample(StandardNormal)).collect();
        let b_data: Vec<f64> = (0..elements).map(|_| rng.sample(StandardNormal)).collect();

        // Column-major: stride = [1, size]
        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [size, size], [1, size as isize], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [size, size], [1, size as isize], 0).unwrap();

        // Generic version
        if size <= 128 {
            group.bench_with_input(BenchmarkId::new("generic", size), &size, |bench, _| {
                bench.iter(|| {
                    let mut c_data = vec![0.0f64; elements];
                    let mut c: StridedArrayViewMut<'_, f64, 2, Identity> =
                        StridedArrayViewMut::new(&mut c_data, [size, size], [size as isize, 1], 0)
                            .unwrap();
                    generic_gemm(1.0, &a, &b, 0.0, &mut c).unwrap();
                    c_data
                })
            });
        }

        // BLAS version (handles column-major via transpose)
        group.bench_with_input(BenchmarkId::new("blas", size), &size, |bench, _| {
            bench.iter(|| {
                let mut c_data = vec![0.0f64; elements];
                let mut c: StridedArrayViewMut<'_, f64, 2, Identity> =
                    StridedArrayViewMut::new(&mut c_data, [size, size], [size as isize, 1], 0)
                        .unwrap();
                blas_gemm(1.0, &a, &b, 0.0, &mut c).unwrap();
                c_data
            })
        });
    }
    group.finish();
}

/// Benchmark f32 vs f64 performance
fn bench_gemm_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_f32");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for size in [256, 512, 1024] {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));

        let mut rng = StdRng::seed_from_u64(42);
        let a_data: Vec<f32> = (0..elements).map(|_| rng.sample(StandardNormal)).collect();
        let b_data: Vec<f32> = (0..elements).map(|_| rng.sample(StandardNormal)).collect();

        let a: StridedArrayView<'_, f32, 2, Identity> =
            StridedArrayView::new(&a_data, [size, size], [size as isize, 1], 0).unwrap();
        let b: StridedArrayView<'_, f32, 2, Identity> =
            StridedArrayView::new(&b_data, [size, size], [size as isize, 1], 0).unwrap();

        // BLAS version (f32)
        group.bench_with_input(BenchmarkId::new("blas_f32", size), &size, |bench, _| {
            bench.iter(|| {
                let mut c_data = vec![0.0f32; elements];
                let mut c: StridedArrayViewMut<'_, f32, 2, Identity> =
                    StridedArrayViewMut::new(&mut c_data, [size, size], [size as isize, 1], 0)
                        .unwrap();
                blas_gemm(1.0f32, &a, &b, 0.0f32, &mut c).unwrap();
                c_data
            })
        });
    }

    // Also benchmark f64 for comparison
    for size in [256, 512, 1024] {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));

        let mut rng = StdRng::seed_from_u64(42);
        let a_data: Vec<f64> = (0..elements).map(|_| rng.sample(StandardNormal)).collect();
        let b_data: Vec<f64> = (0..elements).map(|_| rng.sample(StandardNormal)).collect();

        let a: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&a_data, [size, size], [size as isize, 1], 0).unwrap();
        let b: StridedArrayView<'_, f64, 2, Identity> =
            StridedArrayView::new(&b_data, [size, size], [size as isize, 1], 0).unwrap();

        // BLAS version (f64)
        group.bench_with_input(BenchmarkId::new("blas_f64", size), &size, |bench, _| {
            bench.iter(|| {
                let mut c_data = vec![0.0f64; elements];
                let mut c: StridedArrayViewMut<'_, f64, 2, Identity> =
                    StridedArrayViewMut::new(&mut c_data, [size, size], [size as isize, 1], 0)
                        .unwrap();
                blas_gemm(1.0, &a, &b, 0.0, &mut c).unwrap();
                c_data
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dot,
    bench_axpy,
    bench_gemm,
    bench_gemm_scaled,
    bench_gemm_column_major,
    bench_gemm_f32,
);
criterion_main!(benches);
