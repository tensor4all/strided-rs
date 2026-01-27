use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mdarray::Tensor;
use mdarray_strided::copy_into;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::time::Duration;

// Compare mdarray's assign (iterator-based) with mdarray-strided's copy_into (blocked)
// for a 4D permutation: (32, 32, 32, 32) -> [3, 2, 1, 0]
fn bench_permute_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare/permute_32_4d");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let mut rng = StdRng::seed_from_u64(42);
    let a = Tensor::<f64, _>::from_fn([32, 32, 32, 32], |_| rng.sample(StandardNormal));
    let mut b = Tensor::<f64, _>::zeros([32, 32, 32, 32]);
    let perm = [3, 2, 1, 0];

    // 1. mdarray's built-in assign (iterator-based)
    group.bench_function("mdarray_assign", |bencher| {
        bencher.iter(|| {
            let a_perm = a.permute(perm);
            b.assign(&a_perm);
            black_box(&b);
        })
    });

    // 2. mdarray-strided's copy_into (blocked/optimized)
    group.bench_function("strided_copy_into", |bencher| {
        bencher.iter(|| {
            let a_perm = a.permute(perm);
            copy_into(&mut b, &a_perm).unwrap();
            black_box(&b);
        })
    });

    group.finish();
}

// Compare 2000x2000 transpose
fn bench_transpose_compare_2000(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare/transpose_2000");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let mut rng = StdRng::seed_from_u64(43);
    let a = Tensor::<f64, _>::from_fn([2000, 2000], |_| rng.sample(StandardNormal));
    let mut b = Tensor::<f64, _>::zeros([2000, 2000]);

    // 1. mdarray assign
    group.bench_function("mdarray_assign", |bencher| {
        bencher.iter(|| {
            let a_t = a.transpose();
            b.assign(&a_t);
            black_box(&b);
        })
    });

    // 2. mdarray-strided copy_into
    group.bench_function("strided_copy_into", |bencher| {
        bencher.iter(|| {
            let a_t = a.transpose();
            copy_into(&mut b, &a_t).unwrap();
            black_box(&b);
        })
    });

    group.finish();
}

// Compare 4000x4000 transpose
fn bench_transpose_compare_4000(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare/transpose_4000");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let mut rng = StdRng::seed_from_u64(43);
    let a = Tensor::<f64, _>::from_fn([4000, 4000], |_| rng.sample(StandardNormal));
    let mut b = Tensor::<f64, _>::zeros([4000, 4000]);

    // 1. mdarray assign
    group.bench_function("mdarray_assign", |bencher| {
        bencher.iter(|| {
            let a_t = a.transpose();
            b.assign(&a_t);
            black_box(&b);
        })
    });

    // 2. mdarray-strided copy_into
    group.bench_function("strided_copy_into", |bencher| {
        bencher.iter(|| {
            let a_t = a.transpose();
            copy_into(&mut b, &a_t).unwrap();
            black_box(&b);
        })
    });

    group.finish();
}

criterion_group!(
    mdarray_compare,
    bench_permute_compare,
    bench_transpose_compare_2000,
    bench_transpose_compare_4000,
);
criterion_main!(mdarray_compare);
