//! Partial trace benchmark (ein"iij->j": C[j] = sum_i A[i,i,j]).
//!
//! Matches Julia OMEinsum ein"iij->j" with shape (100, 100, 100).
//! Trace is unary in Julia; we implement it as a diagonal sum over the first
//! two indices (same workload). Benchmarks both f64 and Complex64.
//! Column-major for Julia parity.

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

/// C[j] = sum_i A[i,i,j]; #[inline(never)] prevents hoisting out of the loop.
#[inline(never)]
fn run_ptrace_f64(c: &mut StridedArray<f64>, a: &StridedArray<f64>, n: usize) {
    for j in 0..n {
        let s: f64 = (0..n).map(|i| a.get(&[i, i, j])).sum();
        c.set(&[j], s);
    }
}

#[inline(never)]
fn run_ptrace_complex64(c: &mut StridedArray<Complex64>, a: &StridedArray<Complex64>, n: usize) {
    for j in 0..n {
        let s: Complex64 = (0..n).map(|i| a.get(&[i, i, j])).sum();
        c.set(&[j], s);
    }
}

fn main() {
    println!("strided-einsum2 bench: ptrace (ein\"iij->j\")");
    println!("Shape (100, 100, 100) -> (100). Column-major for Julia parity.");
    println!();

    let n = 100usize;
    let shape_3 = [n, n, n];
    let shape_1 = [n];

    // f64
    let mut rng = StdRng::seed_from_u64(0);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_3, |_| rng.gen::<f64>());
    let mut c = StridedArray::<f64>::col_major(&shape_1);

    println!("ptrace: Float64");
    bench_n("ptrace_f64", 2, 5, || {
        run_ptrace_f64(&mut c, &a, n);
        black_box(c.data().as_ptr());
    });
    println!();

    // Complex64
    let mut rng_c = StdRng::seed_from_u64(1);
    let a_c = StridedArray::<Complex64>::from_fn_col_major(&shape_3, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });
    let mut c_c = StridedArray::<Complex64>::col_major(&shape_1);

    println!("ptrace: ComplexF64");
    bench_n("ptrace_Complex64", 2, 5, || {
        run_ptrace_complex64(&mut c_c, &a_c, n);
        black_box(c_c.data().as_ptr());
    });
}
