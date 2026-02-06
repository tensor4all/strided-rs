//! Star-and-contract benchmark (ein"ij,ik,ik->j": out[j] = sum_ik A[i,j]*B[i,k]*C[i,k]).
//! Matches Julia OMEinsum suite "starandcontract", large size only (100,100).
//! 3-ary: implemented as explicit loop.

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
fn run_starandcontract_f64(
    out: &mut StridedArray<f64>,
    a: &StridedArray<f64>,
    b: &StridedArray<f64>,
    c: &StridedArray<f64>,
    n: usize,
) {
    for j in 0..n {
        let mut s = 0.0f64;
        for i in 0..n {
            for k in 0..n {
                s += a.get(&[i, j]) * b.get(&[i, k]) * c.get(&[i, k]);
            }
        }
        out.set(&[j], s);
    }
}

#[inline(never)]
fn run_starandcontract_complex64(
    out: &mut StridedArray<Complex64>,
    a: &StridedArray<Complex64>,
    b: &StridedArray<Complex64>,
    c: &StridedArray<Complex64>,
    n: usize,
) {
    for j in 0..n {
        let mut s = Complex64::new(0.0, 0.0);
        for i in 0..n {
            for k in 0..n {
                s += a.get(&[i, j]) * b.get(&[i, k]) * c.get(&[i, k]);
            }
        }
        out.set(&[j], s);
    }
}

fn main() {
    let n = 100usize;
    let shape_2 = [n, n];
    let shape_1 = [n];
    println!(
        "strided-einsum2 bench: starandcontract (ein \"ij,ik,ik->j\"), large ({}x{})",
        n, n
    );

    let mut rng = StdRng::seed_from_u64(0);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_2, |_| rng.gen::<f64>());
    let b = StridedArray::<f64>::from_fn_col_major(&shape_2, |_| rng.gen::<f64>());
    let c = StridedArray::<f64>::from_fn_col_major(&shape_2, |_| rng.gen::<f64>());
    let mut out = StridedArray::<f64>::col_major(&shape_1);
    println!("starandcontract (Float64):");
    bench_n("starandcontract_f64_large", 1, 3, || {
        run_starandcontract_f64(&mut out, &a, &b, &c, n);
        black_box(out.data().as_ptr());
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
    let mut outc = StridedArray::<Complex64>::col_major(&shape_1);
    println!("starandcontract (ComplexF64):");
    bench_n("starandcontract_Complex64_large", 1, 3, || {
        run_starandcontract_complex64(&mut outc, &ac, &bc, &cc, n);
        black_box(outc.data().as_ptr());
    });
}
