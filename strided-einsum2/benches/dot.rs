//! Dot product (einsum2 scalar C = sum_ijk A_ijk * B_ijk).
//!
//! Shape (n1, n2, n3). Three cases:
//! - (1) square: n1 = n2 = n3
//! - (2) n1 = n3 >> n2 (tall/skinny in middle dim)
//! - (3) n1 = n3 << n2 (short/wide in middle dim)
//! Benchmarks both f64 and Complex64. Column-major for Julia parity.

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
    println!("  {label}: {:.3} ms", avg.as_secs_f64() * 1e3);
    avg
}

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

fn run_dot_case(case_name: &str, n1: usize, n2: usize, n3: usize, seed_f64: u64, seed_c: u64) {
    let shape_3 = [n1, n2, n3];

    let mut rng = StdRng::seed_from_u64(seed_f64);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_3, |_| rng.gen::<f64>());
    let b = StridedArray::<f64>::from_fn_col_major(&shape_3, |_| rng.gen::<f64>());
    let mut c = StridedArray::<f64>::col_major(&[]);

    println!("  Float64:");
    bench_n(&format!("{case_name}_f64_{n1}x{n2}x{n3}"), 2, 5, || {
        run_dot_f64(&mut c, &a, &b).unwrap();
        black_box(c.get(&[]));
    });

    let mut rng_c = StdRng::seed_from_u64(seed_c);
    let a_c = StridedArray::<Complex64>::from_fn_col_major(&shape_3, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });
    let b_c = StridedArray::<Complex64>::from_fn_col_major(&shape_3, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });
    let mut c_c = StridedArray::<Complex64>::col_major(&[]);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);

    println!("  ComplexF64:");
    bench_n(
        &format!("{case_name}_Complex64_{n1}x{n2}x{n3}"),
        2,
        5,
        || {
            run_dot_complex64(&mut c_c, &a_c, &b_c, one, zero).unwrap();
            black_box(c_c.get(&[]));
        },
    );
}

fn main() {
    println!("strided-einsum2 bench: dot (einsum2 ijk,ijk->)");
    println!("Shape (n1, n2, n3), scalar. Column-major.");
    println!();

    println!("(1) square: n1 = n2 = n3 = 100");
    run_dot_case("square", 100, 100, 100, 0, 1);
    println!();

    println!("(2) n1 = n3 >> n2: (2000, 50, 2000)");
    run_dot_case("tall_skinny", 2000, 50, 2000, 2, 3);
    println!();

    println!("(3) n1 = n3 << n2: (50, 2000, 50)");
    run_dot_case("short_wide", 50, 2000, 50, 4, 5);
}
