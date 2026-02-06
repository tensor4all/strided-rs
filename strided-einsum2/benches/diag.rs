//! Diagonal extraction benchmark (ein"ijj->ij": C[i,j] = A[i,j,j]).
//!
//! Matches Julia OMEinsum ein"ijj->ij" with shape (100, 100, 100).
//! Unary in Julia; we copy the diagonal slice (no contraction).
//! Benchmarks both f64 and Complex64. Column-major for Julia parity.

use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::hint::black_box;
use std::time::{Duration, Instant};
use strided_view::StridedArray;

fn mean(durations: &[Duration]) -> Duration {
    let total_nanos: u128 = durations.iter().map(|d| d.as_nanos()).sum();
    Duration::from_nanos((total_nanos / durations.len() as u128) as u64)
}

fn bench_n(label: &str, warmup_iters: usize, iters: usize, mut f: impl FnMut()) -> Duration {
    for _ in 0..warmup_iters {
        f();
    }

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed());
    }

    let avg = mean(&samples);
    println!("{label}: {:.3} ms", avg.as_secs_f64() * 1e3);
    avg
}

/// C[i,j] = A[i,j,j]; #[inline(never)] prevents hoisting out of the loop.
#[inline(never)]
fn run_diag_f64(c: &mut StridedArray<f64>, a: &StridedArray<f64>, n: usize) {
    for i in 0..n {
        for j in 0..n {
            c.set(&[i, j], a.get(&[i, j, j]));
        }
    }
}

#[inline(never)]
fn run_diag_complex64(c: &mut StridedArray<Complex64>, a: &StridedArray<Complex64>, n: usize) {
    for i in 0..n {
        for j in 0..n {
            c.set(&[i, j], a.get(&[i, j, j]));
        }
    }
}

fn main() {
    println!("strided-einsum2 bench: diag (ein\"ijj->ij\")");
    println!("Shape (100, 100, 100) -> (100, 100). Column-major for Julia parity.");
    println!();

    let n = 100usize;
    let shape_3 = [n, n, n];
    let shape_2 = [n, n];

    // f64
    let mut rng = StdRng::seed_from_u64(0);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_3, |_| rng.gen::<f64>());
    let mut c = StridedArray::<f64>::col_major(&shape_2);

    println!("diag: Float64");
    bench_n("diag_f64", 2, 5, || {
        run_diag_f64(&mut c, &a, n);
        black_box(c.data().as_ptr());
    });
    println!();

    // Complex64
    let mut rng_c = StdRng::seed_from_u64(1);
    let a_c = StridedArray::<Complex64>::from_fn_col_major(&shape_3, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });
    let mut c_c = StridedArray::<Complex64>::col_major(&shape_2);

    println!("diag: ComplexF64");
    bench_n("diag_Complex64", 2, 5, || {
        run_diag_complex64(&mut c_c, &a_c, n);
        black_box(c_c.data().as_ptr());
    });
}
