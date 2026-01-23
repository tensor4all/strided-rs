use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mdarray::Tensor;
use mdarray_strided::{
    copy_into, copy_into_uninit, copy_transpose_scale_into_fast,
    copy_transpose_scale_into_tiled, map_into, sum, symmetrize_into, zip_map2_into, zip_map4_into,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::mem::MaybeUninit;
use std::time::Duration;

// Julia equivalent for `bench_copy_permuted`:
// ```julia
// using BenchmarkTools
//
// function bench_copy_permuted_julia()
//     for size in (100, 500, 1000)
//         a  = reshape(collect(0.0:(size*size-1)), size, size)
//         at = PermutedDimsArray(a, (2, 1))  # lazy permuted view (no copy)
//
//         # "naive": allocate by materializing the permuted view
//         @btime copy($at)
//
//         # "strided"-like: copy into a preallocated output
//         out = similar(a)
//         @btime copyto!($out, $at)
//     end
// end
// ```
fn bench_copy_permuted(c: &mut Criterion) {
    let mut group = c.benchmark_group("copy_permuted");
    for size in [100usize, 500, 1000] {
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
//
// function bench_zip_map_mixed_strides_julia()
//     for size in (100, 500, 1000)
//         a  = reshape(collect(0.0:(size*size-1)), size, size)
//         b  = reshape([i + j for i in 1:size, j in 1:size], size, size)
//         at = PermutedDimsArray(a, (2, 1)) # mixed strides vs `b`
//
//         # naive: nested loops + allocation
//         @btime begin
//             out = zeros(size, size)
//             @inbounds for i in 1:size, j in 1:size
//                 out[i, j] = at[i, j] + b[i, j]
//             end
//             out
//         end
//
//         # strided-like: fused broadcast into preallocated output
//         out = zeros(size, size)
//         @btime @. $out = $at + $b
//     end
// end
// ```
fn bench_zip_map_mixed_strides(c: &mut Criterion) {
    let mut group = c.benchmark_group("zip_map_mixed");
    for size in [100usize, 500, 1000] {
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
//
// function bench_reduce_transposed_julia()
//     for size in (100, 500, 1000)
//         a  = reshape(collect(0.0:(size*size-1)), size, size)
//         at = PermutedDimsArray(a, (2, 1))
//
//         # naive
//         @btime begin
//             s = 0.0
//             @inbounds for i in 1:size, j in 1:size
//                 s += at[i, j]
//             end
//             s
//         end
//
//         # strided: reduction over the permuted view
//         @btime sum($at)
//     end
// end
// ```
fn bench_reduce_transposed(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce_transposed");
    for size in [100usize, 500, 1000] {
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
//
// function bench_symmetrize_aat_julia()
//     size = 4000
//     a = rand(size, size)
//
//     # naive: allocate + nested loops
//     @btime begin
//         out = zeros(size, size)
//         at = PermutedDimsArray(a, (2, 1))
//         @inbounds for i in 1:size, j in 1:size
//             out[i, j] = 0.5 * (a[i, j] + at[i, j])
//         end
//         out
//     end
//
//     # strided-like: fused broadcast into a preallocated output
//     out = similar(a)
//     at = PermutedDimsArray(a, (2, 1))
//     @btime @. $out = 0.5 * ($a + $at)
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
            if let Err(err) = symmetrize_into(&mut out, a_view) {
                panic!("symmetrize_into failed: {err}");
            }
            out
        })
    });

    group.finish();
}

// Julia equivalent for `bench_scale_transpose`:
// ```julia
// using BenchmarkTools
//
// function bench_scale_transpose_julia()
//     size = 1000
//     a  = randn(size, size)
//     at = PermutedDimsArray(a, (2, 1))
//
//     # naive: nested loops
//     @btime begin
//         out = zeros(size, size)
//         @inbounds for i in 1:size, j in 1:size
//             out[i, j] = 3.0 * at[i, j]
//         end
//         out
//     end
//
//     # strided-like: broadcast into preallocated output
//     out = similar(a)
//     @btime @. $out = 3.0 * $at
// end
// ```
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

    for tile in [16usize, 24, 32] {
        group.bench_with_input(BenchmarkId::new("strided_tile", tile), &tile, |b, &tile| {
            b.iter(|| {
                let mut out = Tensor::zeros([size, size]);
                if let Err(err) = copy_transpose_scale_into_tiled(&mut out, a.as_ref(), 3.0, tile) {
                    panic!("copy_transpose_scale_into_tiled failed: {err}");
                }
                out
            })
        });
    }

    group.finish();
}

// Julia equivalent for `bench_nonlinear_map`:
// ```julia
// using BenchmarkTools
//
// function bench_nonlinear_map_julia()
//     size = 1000
//     a = randn(size, size)
//
//     f(x) = x * exp(-2x) + sin(x^2)
//
//     # naive: nested loops
//     @btime begin
//         out = zeros(size, size)
//         @inbounds for i in 1:size, j in 1:size
//             out[i, j] = f(a[i, j])
//         end
//         out
//     end
//
//     # strided-like: map into a preallocated output
//     out = similar(a)
//     @btime map!($f, $out, $a)
// end
// ```
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

// Julia equivalent for `bench_copy_contiguous`:
// ```julia
// using BenchmarkTools
//
// function bench_copy_contiguous_julia()
//     for size in (100, 500, 1000)
//         a = reshape(collect(0.0:(size*size-1)), size, size)
//
//         # naive: allocate copy
//         @btime copy($a)
//
//         # strided-like: copy into preallocated output
//         out = similar(a)
//         @btime copyto!($out, $a)
//     end
// end
// ```
fn bench_copy_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("copy_contiguous");
    for size in [100usize, 500, 1000] {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));

        let a = Tensor::<f64, _>::from_fn([size, size], |idx| (idx[0] * size + idx[1]) as f64);
        let a_view = a.as_ref();

        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b, _| {
            b.iter(|| a_view.to_tensor());
        });

        group.bench_with_input(BenchmarkId::new("strided", size), &size, |b, _| {
            b.iter(|| {
                let mut out = Tensor::zeros([size, size]);
                if let Err(err) = copy_into(&mut out, a_view) {
                    panic!("copy_into failed: {err}");
                }
                out
            })
        });

        group.bench_with_input(BenchmarkId::new("uninit", size), &size, |b, &size| {
            let mut buf = vec![MaybeUninit::<f64>::uninit(); size * size];
            let dims = [size, size];
            let strides = [size as isize, 1isize];
            b.iter(|| unsafe {
                if let Err(err) = copy_into_uninit(buf.as_mut_ptr(), &dims, &strides, a_view) {
                    panic!("copy_into_uninit failed: {err}");
                }
            })
        });
    }
    group.finish();
}

// Julia equivalent for `bench_zip_map_contiguous`:
// ```julia
// using BenchmarkTools
//
// function bench_zip_map_contiguous_julia()
//     for size in (100, 500, 1000)
//         a = reshape(collect(0.0:(size*size-1)), size, size)
//         b = reshape([i + j for i in 1:size, j in 1:size], size, size)
//
//         # naive: nested loops + allocation
//         @btime begin
//             out = zeros(size, size)
//             @inbounds for i in 1:size, j in 1:size
//                 out[i, j] = a[i, j] + b[i, j]
//             end
//             out
//         end
//
//         # strided-like: fused broadcast into preallocated output
//         out = zeros(size, size)
//         @btime @. $out = $a + $b
//     end
// end
// ```
fn bench_zip_map_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("zip_map_contiguous");
    for size in [100usize, 500, 1000] {
        let a = Tensor::<f64, _>::from_fn([size, size], |idx| (idx[0] * size + idx[1]) as f64);
        let b = Tensor::<f64, _>::from_fn([size, size], |idx| (idx[0] + idx[1]) as f64);
        let a_view = a.as_ref();
        let b_view = b.as_ref();

        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b_iter, _| {
            b_iter.iter(|| {
                let mut out = Tensor::zeros([size, size]);
                for i in 0..size {
                    for j in 0..size {
                        out[[i, j]] = a_view[[i, j]] + b_view[[i, j]];
                    }
                }
                out
            })
        });

        group.bench_with_input(BenchmarkId::new("strided", size), &size, |b_iter, _| {
            b_iter.iter(|| {
                let mut out = Tensor::zeros([size, size]);
                if let Err(err) = zip_map2_into(&mut out, a_view, b_view, |x, y| x + y) {
                    panic!("zip_map2_into failed: {err}");
                }
                out
            })
        });
    }
    group.finish();
}

// Julia equivalent for `bench_reduce_contiguous`:
// ```julia
// using BenchmarkTools
//
// function bench_reduce_contiguous_julia()
//     for size in (100, 500, 1000)
//         a = reshape(collect(0.0:(size*size-1)), size, size)
//
//         # naive
//         @btime begin
//             s = 0.0
//             @inbounds for i in 1:size, j in 1:size
//                 s += a[i, j]
//             end
//             s
//         end
//
//         # strided: reduction on contiguous array
//         @btime sum($a)
//     end
// end
// ```
fn bench_reduce_contiguous(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce_contiguous");
    for size in [100usize, 500, 1000] {
        let a = Tensor::<f64, _>::from_fn([size, size], |idx| (idx[0] * size + idx[1]) as f64);
        let a_view = a.as_ref();

        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b, _| {
            b.iter(|| {
                let mut sum_val = 0.0;
                for i in 0..size {
                    for j in 0..size {
                        sum_val += a_view[[i, j]];
                    }
                }
                sum_val
            })
        });

        group.bench_with_input(BenchmarkId::new("strided", size), &size, |b, _| {
            b.iter(|| match sum(a_view) {
                Ok(v) => v,
                Err(err) => panic!("sum failed: {err}"),
            })
        });
    }
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
//
// function bench_multi_permute_sum_julia()
//     A = randn(32, 32, 32, 32)
//     B = similar(A)
//
//     # naive: allocate temporaries for each permutation
//     @btime $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+
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
            if let Err(err) = zip_map4_into(&mut out, &p1, &p2, &p3, &p4, |a, b, c, d| a + b + c + d)
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
    bench_copy_contiguous,
    bench_zip_map_contiguous,
    bench_reduce_contiguous,
    bench_symmetrize_aat,
    bench_scale_transpose,
    bench_nonlinear_map,
    bench_permutedims_4d,
    bench_multi_permute_sum
);
criterion_main!(benches);
