//! Batched matrix multiplication (einsum2 C_ilk = A_ijk * B_jlk).
//!
//! Per batch: (n1, n2) * (n2, n3) = (n1, n3). Three cases:
//! - (1) square: n1 = n2 = n3
//! - (2) n1 = n3 >> n2 (tall/skinny)
//! - (3) n1 = n3 << n2 (short/wide)
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

fn run_batchmul_case(
    case_name: &str,
    batch: usize,
    n1: usize,
    n2: usize,
    n3: usize,
    seed_f64: u64,
    seed_c: u64,
) {
    let shape_a = [n1, n2, batch];
    let shape_b = [n2, n3, batch];
    let shape_c = [n1, n3, batch];

    let mut rng = StdRng::seed_from_u64(seed_f64);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_a, |_| rng.gen::<f64>());
    let b = StridedArray::<f64>::from_fn_col_major(&shape_b, |_| rng.gen::<f64>());
    let mut c = StridedArray::<f64>::col_major(&shape_c);

    println!("  Float64:");
    bench_n(
        &format!("{case_name}_f64_b{batch}_{n1}x{n2}x{n3}"),
        2,
        5,
        || {
            einsum2_into(
                c.view_mut(),
                &a.view(),
                &b.view(),
                &['i', 'l', 'k'],
                &['i', 'j', 'k'],
                &['j', 'l', 'k'],
                1.0,
                0.0,
            )
            .unwrap();
            black_box(c.data().as_ptr());
        },
    );

    let mut rng_c = StdRng::seed_from_u64(seed_c);
    let a_c = StridedArray::<Complex64>::from_fn_col_major(&shape_a, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });
    let b_c = StridedArray::<Complex64>::from_fn_col_major(&shape_b, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });
    let mut c_c = StridedArray::<Complex64>::col_major(&shape_c);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);

    println!("  ComplexF64:");
    bench_n(
        &format!("{case_name}_Complex64_b{batch}_{n1}x{n2}x{n3}"),
        2,
        5,
        || {
            einsum2_into(
                c_c.view_mut(),
                &a_c.view(),
                &b_c.view(),
                &['i', 'l', 'k'],
                &['i', 'j', 'k'],
                &['j', 'l', 'k'],
                one,
                zero,
            )
            .unwrap();
            black_box(c_c.data().as_ptr());
        },
    );
}

fn main() {
    let batch = 3usize;
    println!("strided-einsum2 bench: batchmul (einsum2 ijk,jlk->ilk)");
    println!("Per batch (n1,n2)*(n2,n3)=(n1,n3). Column-major.");
    println!();

    println!("(1) square: n1 = n2 = n3 = 1000, batch = {}", batch);
    run_batchmul_case("square", batch, 1000, 1000, 1000, 0, 1);
    println!();

    println!(
        "(2) n1 = n3 >> n2: (2000, 50) * (50, 2000), batch = {}",
        batch
    );
    run_batchmul_case("tall_skinny", batch, 2000, 50, 2000, 2, 3);
    println!();

    println!(
        "(3) n1 = n3 << n2: (50, 2000) * (2000, 50), batch = {}",
        batch
    );
    run_batchmul_case("short_wide", batch, 50, 2000, 50, 4, 5);
}
