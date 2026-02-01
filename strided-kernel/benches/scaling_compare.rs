//! Scaling benchmarks matching Strided.jl's benchtests.jl.
//!
//! Measures performance across exponentially-scaled array sizes to show
//! the crossover point where strided's ordering/blocking overhead pays off.
//!
//! - benchmark_sum: 1D sum, sizes 2^2 .. 2^20
//! - benchmark_permute: 4D permutedims, sizes 4..64 (s^4 elements)

use rand::{rngs::StdRng, SeedableRng};
use rand_distr::StandardNormal;
use std::hint::black_box;
use std::time::{Duration, Instant};
use strided_kernel::{copy_into, sum, StridedArray};

fn median(durations: &mut [Duration]) -> Duration {
    durations.sort();
    durations[durations.len() / 2]
}

/// Adaptive bench: run enough iterations to get stable timing.
fn bench_adaptive(mut f: impl FnMut()) -> Duration {
    // Warmup
    for _ in 0..3 {
        f();
    }

    // Calibrate: find how many iters fit in ~100ms
    let t0 = Instant::now();
    f();
    let single = t0.elapsed();
    let iters = if single.as_nanos() == 0 {
        10000
    } else {
        ((100_000_000u128 / single.as_nanos()) as usize).clamp(3, 10000)
    };

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed());
    }
    median(&mut samples)
}

/// Julia's sizes: ceil.(Int, 2 .^ (2:1.5:20))
fn julia_sizes() -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut exp = 2.0f64;
    while exp <= 20.0 {
        sizes.push(2.0f64.powf(exp).ceil() as usize);
        exp += 1.5;
    }
    sizes
}

fn make_random_1d(n: usize, seed: u64) -> StridedArray<f64> {
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(seed);
    StridedArray::<f64>::from_fn_col_major(&[n], |_| rng.sample(StandardNormal))
}

fn make_random_4d(s: usize, seed: u64) -> StridedArray<f64> {
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(seed);
    StridedArray::<f64>::from_fn_col_major(&[s, s, s, s], |_| rng.sample(StandardNormal))
}

fn benchmark_sum() {
    println!("=== benchmark_sum (1D) ===");
    println!(
        "{:>10} {:>12} {:>12} {:>8}",
        "size", "naive (us)", "strided (us)", "ratio"
    );

    let sizes = julia_sizes();
    for (i, &s) in sizes.iter().enumerate() {
        let a = make_random_1d(s, i as u64);
        let a_ptr = a.data().as_ptr();

        // Naive: raw pointer sum
        let t_naive = bench_adaptive(|| {
            let mut acc = 0.0f64;
            for k in 0..s {
                acc += unsafe { *a_ptr.add(k) };
            }
            black_box(acc);
        });

        // Strided sum
        let a_view = a.view();
        let t_strided = bench_adaptive(|| {
            let r = sum(&a_view).unwrap();
            black_box(r);
        });

        let ratio = t_strided.as_nanos() as f64 / t_naive.as_nanos().max(1) as f64;
        println!(
            "{:>10} {:>12.3} {:>12.3} {:>8.2}x",
            s,
            t_naive.as_nanos() as f64 / 1e3,
            t_strided.as_nanos() as f64 / 1e3,
            ratio
        );
    }
    println!();
}

fn benchmark_permute(perm: &[usize], label: &str) {
    println!("=== benchmark_permute {} ===", label);
    println!(
        "{:>6} {:>10} {:>12} {:>12} {:>12} {:>8}",
        "s", "s^4", "copy (us)", "naive (us)", "strided (us)", "ratio"
    );

    // Practical sizes: s^4 * 8 bytes per array, 2 arrays
    // s=4: 4KB, s=12: 324KB, s=32: 16MB, s=64: 128MB
    let sizes: Vec<usize> = vec![4, 8, 12, 16, 24, 32, 48, 64];

    for (i, &s) in sizes.iter().enumerate() {
        let total = s * s * s * s;
        let a = make_random_4d(s, 100 + i as u64);
        let mut b = StridedArray::<f64>::col_major(&[s, s, s, s]);
        let a_ptr = a.data().as_ptr();
        let b_ptr = b.data_mut().as_mut_ptr();

        // Contiguous copy baseline
        let t_copy = bench_adaptive(|| {
            unsafe {
                std::ptr::copy_nonoverlapping(a_ptr, b_ptr, total);
            }
            black_box(b_ptr);
        });

        // Naive permute with precomputed strides
        let s2 = s * s;
        let s3 = s2 * s;
        let t_naive = bench_adaptive(|| {
            // permutedims!(B, A, perm) where perm is 0-indexed
            // B[i0,i1,i2,i3] = A[i_{perm[0]}, i_{perm[1]}, i_{perm[2]}, i_{perm[3]}]
            // Col-major: index = i0 + i1*s + i2*s^2 + i3*s^3
            // A source index uses perm to remap
            let src_strides = [1usize, s, s2, s3];
            let a_strides: [usize; 4] = [
                src_strides[perm[0]],
                src_strides[perm[1]],
                src_strides[perm[2]],
                src_strides[perm[3]],
            ];
            for i3 in 0..s {
                for i2 in 0..s {
                    for i1 in 0..s {
                        let b_base = i1 * s + i2 * s2 + i3 * s3;
                        let a_base = i1 * a_strides[1] + i2 * a_strides[2] + i3 * a_strides[3];
                        for i0 in 0..s {
                            unsafe {
                                *b_ptr.add(b_base + i0) = *a_ptr.add(a_base + i0 * a_strides[0]);
                            }
                        }
                    }
                }
            }
            black_box(b_ptr);
        });

        // Strided permutedims
        let a_view = a.view();
        let a_perm = a_view.permute(perm).unwrap();
        let t_strided = bench_adaptive(|| {
            copy_into(&mut b.view_mut(), &a_perm).unwrap();
            black_box(b_ptr);
        });

        let ratio = t_strided.as_nanos() as f64 / t_naive.as_nanos().max(1) as f64;
        println!(
            "{:>6} {:>10} {:>12.3} {:>12.3} {:>12.3} {:>8.2}x",
            s,
            total,
            t_copy.as_nanos() as f64 / 1e3,
            t_naive.as_nanos() as f64 / 1e3,
            t_strided.as_nanos() as f64 / 1e3,
            ratio
        );
    }
    println!();
}

fn main() {
    println!("Scaling benchmarks (cf. Strided.jl benchtests.jl)");
    println!("Column-major layout. Median timing.");
    println!();

    benchmark_sum();
    benchmark_permute(&[3, 2, 1, 0], "(4,3,2,1)");
    benchmark_permute(&[1, 2, 3, 0], "(2,3,4,1)");
    benchmark_permute(&[2, 3, 0, 1], "(3,4,1,2)");
}
