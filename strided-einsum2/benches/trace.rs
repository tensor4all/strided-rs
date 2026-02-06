//! Trace benchmark (sum of diagonal: ein"ii->").
//!
//! Matches Julia OMEinsum ein"ii->" with 1000×1000 matrix.
//! Trace is unary in Julia; we implement it as a diagonal sum (same workload).
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

/// Sum diagonal elements; #[inline(never)] prevents hoisting out of the loop.
#[inline(never)]
fn run_trace_f64(a: &StridedArray<f64>, n: usize) -> f64 {
    (0..n).map(|i| a.get(&[i, i])).sum()
}

#[inline(never)]
fn run_trace_complex64(a: &StridedArray<Complex64>, n: usize) -> Complex64 {
    (0..n).map(|i| a.get(&[i, i])).sum()
}

fn main() {
    println!("strided-einsum2 bench: trace (ein\"ii->\")");
    println!("Shape (1000, 1000), scalar output. Column-major for Julia parity.");
    println!();

    let n = 1000usize;
    let shape_2 = [n, n];

    // f64
    let mut rng = StdRng::seed_from_u64(0);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_2, |_| rng.gen::<f64>());

    println!("trace: Float64");
    bench_n("trace_f64", 2, 5, || {
        let s = run_trace_f64(&a, n);
        black_box(s);
    });
    println!();

    // Complex64
    let mut rng_c = StdRng::seed_from_u64(1);
    let a_c = StridedArray::<Complex64>::from_fn_col_major(&shape_2, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });

    println!("trace: ComplexF64");
    bench_n("trace_Complex64", 2, 5, || {
        let s = run_trace_complex64(&a_c, n);
        black_box(s);
    });
}
