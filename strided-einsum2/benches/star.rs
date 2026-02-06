//! Star contraction benchmark (ein"ij,ik,il->jkl": D[j,k,l] = sum_i A[i,j]*B[i,k]*C[i,l]).
//! Matches Julia OMEinsum suite "star", large size only (50,50).
//! 3-ary contraction: implemented as explicit loop (no einsum3).

use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::hint::black_box;
use std::time::{Duration, Instant};
use strided_view::StridedArray;

fn mean(durations: &[Duration]) -> Duration {
    let total_nanos: u128 = durations.iter().map(|d| d.as_nanos()).sum();
    Duration::from_nanos((total_nanos / durations.len() as u128) as u64)
}

fn bench_n(label: &str, warmup: usize, iters: usize, mut f: impl FnMut()) -> Duration {
    for _ in 0..warmup {
        f();
    }
    let samples: Vec<Duration> = (0..iters)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed()
        })
        .collect();
    let avg = mean(&samples);
    println!("{label}: {:.3} ms", avg.as_secs_f64() * 1e3);
    avg
}

#[inline(never)]
fn run_star_f64(
    d: &mut StridedArray<f64>,
    a: &StridedArray<f64>,
    b: &StridedArray<f64>,
    c: &StridedArray<f64>,
    n: usize,
) {
    for j in 0..n {
        for k in 0..n {
            for l in 0..n {
                let mut s = 0.0f64;
                for i in 0..n {
                    s += a.get(&[i, j]) * b.get(&[i, k]) * c.get(&[i, l]);
                }
                d.set(&[j, k, l], s);
            }
        }
    }
}

#[inline(never)]
fn run_star_complex64(
    d: &mut StridedArray<Complex64>,
    a: &StridedArray<Complex64>,
    b: &StridedArray<Complex64>,
    c: &StridedArray<Complex64>,
    n: usize,
) {
    for j in 0..n {
        for k in 0..n {
            for l in 0..n {
                let mut s = Complex64::new(0.0, 0.0);
                for i in 0..n {
                    s += a.get(&[i, j]) * b.get(&[i, k]) * c.get(&[i, l]);
                }
                d.set(&[j, k, l], s);
            }
        }
    }
}

fn main() {
    let n = 50usize;
    let shape_2 = [n, n];
    let shape_3 = [n, n, n];
    println!(
        "strided-einsum2 bench: star (ein \"ij,ik,il->jkl\"), large ({}x{})",
        n, n
    );

    let mut rng = StdRng::seed_from_u64(0);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_2, |_| rng.gen::<f64>());
    let b = StridedArray::<f64>::from_fn_col_major(&shape_2, |_| rng.gen::<f64>());
    let c = StridedArray::<f64>::from_fn_col_major(&shape_2, |_| rng.gen::<f64>());
    let mut d = StridedArray::<f64>::col_major(&shape_3);
    println!("star (Float64):");
    bench_n("star_f64_large", 1, 3, || {
        run_star_f64(&mut d, &a, &b, &c, n);
        black_box(d.data().as_ptr());
    });

    let mut rng_c = StdRng::seed_from_u64(1);
    let ac = StridedArray::<Complex64>::from_fn_col_major(&shape_2, |_| {
        Complex64::new(rng_c.gen(), rng_c.gen())
    });
    let bc = StridedArray::<Complex64>::from_fn_col_major(&shape_2, |_| {
        Complex64::new(rng_c.gen(), rng_c.gen())
    });
    let cc = StridedArray::<Complex64>::from_fn_col_major(&shape_2, |_| {
        Complex64::new(rng_c.gen(), rng_c.gen())
    });
    let mut dc = StridedArray::<Complex64>::col_major(&shape_3);
    println!("star (ComplexF64):");
    bench_n("star_Complex64_large", 1, 3, || {
        run_star_complex64(&mut dc, &ac, &bc, &cc, n);
        black_box(dc.data().as_ptr());
    });
}
