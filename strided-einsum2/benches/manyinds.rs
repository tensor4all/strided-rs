//! Many-index einsum benchmark (OMEinsum-style).
//!
//! Contraction: "abcdefghijklmnop,flnqrcipstujvgamdwxyz->bcdeghkmnopqrstuvwxyz"
//! All dimensions size 2. Single-threaded for parity with Julia @benchmark.
//! Benchmarks both f64 and ComplexF64 (Complex64).

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

/// Axis labels for the many-index contraction (same as Julia OMEinsum example).
/// Left:  abcdefghijklmnop (16 dims), Right: flnqrcipstujvgamdwxyz (21 dims),
/// Output: bcdeghkmnopqrstuvwxyz (21 dims). All dimensions size 2.
const IA: [char; 16] = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
];
const IB: [char; 21] = [
    'f', 'l', 'n', 'q', 'r', 'c', 'i', 'p', 's', 't', 'u', 'j', 'v', 'g', 'a', 'm', 'd', 'w', 'x',
    'y', 'z',
];
const IC: [char; 21] = [
    'b', 'c', 'd', 'e', 'g', 'h', 'k', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z',
];

fn main() {
    println!("strided-einsum2 bench: manyinds (many-index binary einsum)");
    println!("Contract: abcdefghijklmnop, flnqrcipstujvgamdwxyz -> bcdeghkmnopqrstuvwxyz");
    println!("All dimensions size 2. Column-major for Julia parity.");
    println!();

    let dim = 2usize;
    let shape_a: Vec<usize> = (0..16).map(|_| dim).collect();
    let shape_b: Vec<usize> = (0..21).map(|_| dim).collect();
    let shape_c: Vec<usize> = (0..21).map(|_| dim).collect();

    let mut rng = StdRng::seed_from_u64(0);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_a, |_| rng.gen::<f64>());
    let b = StridedArray::<f64>::from_fn_col_major(&shape_b, |_| rng.gen::<f64>());
    let mut c = StridedArray::<f64>::col_major(&shape_c);

    println!("=== manyinds f64 (einsum2_into) ===");
    bench_n("einsum2_into_f64", 1, 5, || {
        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &IC[..],
            &IA[..],
            &IB[..],
            1.0,
            0.0,
        )
        .unwrap();
        black_box(c.data().as_ptr());
    });
    println!();

    // ComplexF64 (Complex64) version — same contraction, complex arrays
    let mut rng_c = StdRng::seed_from_u64(1);
    let a_c = StridedArray::<Complex64>::from_fn_col_major(&shape_a, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });
    let b_c = StridedArray::<Complex64>::from_fn_col_major(&shape_b, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });
    let mut c_c = StridedArray::<Complex64>::col_major(&shape_c);

    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);

    println!("=== manyinds Complex64 (einsum2_into) ===");
    bench_n("einsum2_into_Complex64", 1, 5, || {
        einsum2_into(
            c_c.view_mut(),
            &a_c.view(),
            &b_c.view(),
            &IC[..],
            &IA[..],
            &IB[..],
            one,
            zero,
        )
        .unwrap();
        black_box(c_c.data().as_ptr());
    });
}

