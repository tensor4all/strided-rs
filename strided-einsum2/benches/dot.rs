//! Dot product benchmark (einsum2 scalar C = sum_ijk A_ijk * B_ijk).
//!
//! Matches Julia OMEinsum ein"ijk,ijk->" with shape (100, 100, 100).
//! Benchmarks both f64 and Complex64. Column-major for Julia parity.
//!
//! Uses #[inline(never)] on the run_dot_* helpers so the compiler cannot
//! hoist the computation out of the timed loop. If Rust times are much
//! lower than Julia (e.g. 0.2ms vs 4ms), that is likely due to different
//! backends (faer vs OMEinsum), not benchmark artifact.

use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::hint::black_box;
use std::time::{Duration, Instant};
use strided_einsum2::einsum2_into;
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

/// Run dot once; #[inline(never)] prevents the compiler from hoisting
/// the computation out of the benchmark loop (same inputs → same output).
#[inline(never)]
fn run_dot_f64(
    c: &mut StridedArray<f64>,
    a: &StridedArray<f64>,
    b: &StridedArray<f64>,
) -> Result<(), strided_einsum2::EinsumError> {
    let ic_empty: &[char] = &[];
    einsum2_into(
        c.view_mut(),
        &a.view(),
        &b.view(),
        ic_empty,
        &['i', 'j', 'k'],
        &['i', 'j', 'k'],
        1.0,
        0.0,
    )
}

#[inline(never)]
fn run_dot_complex64(
    c: &mut StridedArray<Complex64>,
    a: &StridedArray<Complex64>,
    b: &StridedArray<Complex64>,
    one: Complex64,
    zero: Complex64,
) -> Result<(), strided_einsum2::EinsumError> {
    let ic_empty: &[char] = &[];
    einsum2_into(
        c.view_mut(),
        &a.view(),
        &b.view(),
        ic_empty,
        &['i', 'j', 'k'],
        &['i', 'j', 'k'],
        one,
        zero,
    )
}

fn main() {
    println!("strided-einsum2 bench: dot (einsum2 ijk,ijk->)");
    println!("Shape (100, 100, 100), scalar output. Column-major for Julia parity.");
    println!();

    let shape_3 = [100, 100, 100];

    // f64
    let mut rng = StdRng::seed_from_u64(0);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_3, |_| rng.gen::<f64>());
    let b = StridedArray::<f64>::from_fn_col_major(&shape_3, |_| rng.gen::<f64>());
    let mut c = StridedArray::<f64>::col_major(&[]);

    println!("dot (einsum2_into): Float64");
    bench_n("dot_f64", 2, 5, || {
        run_dot_f64(&mut c, &a, &b).unwrap();
        black_box(c.get(&[]));
    });
    println!();

    // Complex64
    let mut rng_c = StdRng::seed_from_u64(1);
    let a_c = StridedArray::<Complex64>::from_fn_col_major(&shape_3, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });
    let b_c = StridedArray::<Complex64>::from_fn_col_major(&shape_3, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });
    let mut c_c = StridedArray::<Complex64>::col_major(&[]);

    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);

    println!("dot (einsum2_into): ComplexF64");
    bench_n("dot_Complex64", 2, 5, || {
        run_dot_complex64(&mut c_c, &a_c, &b_c, one, zero).unwrap();
        black_box(c_c.get(&[]));
    });
}
