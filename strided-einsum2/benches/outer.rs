//! Outer product (ein"ij,kl->ijkl": D[i,j,k,l] = A[i,j]*B[k,l]).
//!
//! A(n1,n2), B(n3,n4) -> C(n1,n2,n3,n4). Sizes kept moderate to limit memory
//! (output has n1*n2*n3*n4 elements). Three cases:
//! - (1) square: n1=n2=n3=n4
//! - (2) n1=n2 >> n3=n4 (tall A, short B)
//! - (3) n1=n2 << n3=n4 (short A, tall B)
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
    println!("  {label}: {:.3} ms", avg.as_secs_f64() * 1e3);
    avg
}

fn run_outer_case(
    case_name: &str,
    n1: usize,
    n2: usize,
    n3: usize,
    n4: usize,
    seed_f64: u64,
    seed_c: u64,
) {
    let shape_a = [n1, n2];
    let shape_b = [n3, n4];
    let shape_c = [n1, n2, n3, n4];

    let mut rng = StdRng::seed_from_u64(seed_f64);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_a, |_| rng.gen::<f64>());
    let b = StridedArray::<f64>::from_fn_col_major(&shape_b, |_| rng.gen::<f64>());
    let mut d = StridedArray::<f64>::col_major(&shape_c);

    println!("  Float64:");
    bench_n(
        &format!("{case_name}_f64_{n1}x{n2}_x_{n3}x{n4}"),
        1,
        3,
        || {
            einsum2_into(
                d.view_mut(),
                &a.view(),
                &b.view(),
                &['i', 'j', 'k', 'l'],
                &['i', 'j'],
                &['k', 'l'],
                1.0,
                0.0,
            )
            .unwrap();
            black_box(d.data().as_ptr());
        },
    );

    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let mut rng_c = StdRng::seed_from_u64(seed_c);
    let ac = StridedArray::<Complex64>::from_fn_col_major(&shape_a, |_| {
        Complex64::new(rng_c.gen(), rng_c.gen())
    });
    let bc = StridedArray::<Complex64>::from_fn_col_major(&shape_b, |_| {
        Complex64::new(rng_c.gen(), rng_c.gen())
    });
    let mut dc = StridedArray::<Complex64>::col_major(&shape_c);
    println!("  ComplexF64:");
    bench_n(
        &format!("{case_name}_Complex64_{n1}x{n2}_x_{n3}x{n4}"),
        1,
        3,
        || {
            einsum2_into(
                dc.view_mut(),
                &ac.view(),
                &bc.view(),
                &['i', 'j', 'k', 'l'],
                &['i', 'j'],
                &['k', 'l'],
                one,
                zero,
            )
            .unwrap();
            black_box(dc.data().as_ptr());
        },
    );
}

fn main() {
    println!("strided-einsum2 bench: outer (ein \"ij,kl->ijkl\")");
    println!("A(n1,n2), B(n3,n4) -> C(n1,n2,n3,n4). Column-major.");
    println!();

    println!("(1) square: n1 = n2 = n3 = n4 = 40 (output 40^4)");
    run_outer_case("square", 40, 40, 40, 40, 0, 1);
    println!();

    println!("(2) n1=n2 >> n3=n4: (80,20) x (20,80)");
    run_outer_case("tall_skinny", 80, 20, 20, 80, 2, 3);
    println!();

    println!("(3) n1=n2 << n3=n4: (20,80) x (80,20)");
    run_outer_case("short_wide", 20, 80, 80, 20, 4, 5);
}
