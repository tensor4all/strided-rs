//! Index-sum benchmark (ein"ijk->ik": out[i,k] = sum_j A[i,j,k]).
//! Matches Julia OMEinsum suite "indexsum", large size only (100,100,100).
//! Unary reduction: implemented as explicit loop.

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
    for _ in 0..warmup { f(); }
    let samples: Vec<Duration> = (0..iters).map(|_| { let t = Instant::now(); f(); t.elapsed() }).collect();
    let avg = mean(&samples);
    println!("{label}: {:.3} ms", avg.as_secs_f64() * 1e3);
    avg
}

#[inline(never)]
fn run_indexsum_f64(out: &mut StridedArray<f64>, a: &StridedArray<f64>, n: usize) {
    for i in 0..n {
        for k in 0..n {
            let s: f64 = (0..n).map(|j| a.get(&[i, j, k])).sum();
            out.set(&[i, k], s);
        }
    }
}

#[inline(never)]
fn run_indexsum_complex64(out: &mut StridedArray<Complex64>, a: &StridedArray<Complex64>, n: usize) {
    for i in 0..n {
        for k in 0..n {
            let s: Complex64 = (0..n).map(|j| a.get(&[i, j, k])).sum();
            out.set(&[i, k], s);
        }
    }
}

fn main() {
    let n = 100usize;
    let shape_3 = [n, n, n];
    let shape_2 = [n, n];
    println!("strided-einsum2 bench: indexsum (ein \"ijk->ik\"), large ({}^3)", n);

    let mut rng = StdRng::seed_from_u64(0);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_3, |_| rng.gen::<f64>());
    let mut out = StridedArray::<f64>::col_major(&shape_2);
    println!("indexsum (Float64):");
    bench_n("indexsum_f64_large", 1, 3, || { run_indexsum_f64(&mut out, &a, n); black_box(out.data().as_ptr()); });

    let mut rng_c = StdRng::seed_from_u64(1);
    let ac = StridedArray::<Complex64>::from_fn_col_major(&shape_3, |_| Complex64::new(rng_c.gen(), rng_c.gen()));
    let mut outc = StridedArray::<Complex64>::col_major(&shape_2);
    println!("indexsum (ComplexF64):");
    bench_n("indexsum_Complex64_large", 1, 3, || { run_indexsum_complex64(&mut outc, &ac, n); black_box(outc.data().as_ptr()); });
}
