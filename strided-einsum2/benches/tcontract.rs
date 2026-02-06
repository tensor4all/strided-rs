//! Tensor contraction (einsum2 C_il = sum_jk A_ijk * B_jlk).
//!
//! A(n_i, n_j, n_k), B(n_j, n_l, n_k), C(n_i, n_l). Three cases:
//! - (1) square: n_i = n_j = n_k = n_l
//! - (2) n_i = n_l >> n_j (n_j = n_k small)
//! - (3) n_i = n_l << n_j (n_j = n_k large)
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

fn run_tcontract_case(
    case_name: &str,
    ni: usize,
    nj: usize,
    nk: usize,
    nl: usize,
    seed_f64: u64,
    seed_c: u64,
) {
    let shape_a = [ni, nj, nk];
    let shape_b = [nj, nl, nk];
    let shape_c = [ni, nl];

    let mut rng = StdRng::seed_from_u64(seed_f64);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_a, |_| rng.gen::<f64>());
    let b = StridedArray::<f64>::from_fn_col_major(&shape_b, |_| rng.gen::<f64>());
    let mut c = StridedArray::<f64>::col_major(&shape_c);

    println!("  Float64:");
    bench_n(
        &format!("{case_name}_f64_{ni}x{nj}x{nk}_x{nl}"),
        2,
        5,
        || {
            einsum2_into(
                c.view_mut(),
                &a.view(),
                &b.view(),
                &['i', 'l'],
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
        &format!("{case_name}_Complex64_{ni}x{nj}x{nk}_x{nl}"),
        2,
        5,
        || {
            einsum2_into(
                c_c.view_mut(),
                &a_c.view(),
                &b_c.view(),
                &['i', 'l'],
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
    println!("strided-einsum2 bench: tcontract (einsum2 ijk,jlk->il)");
    println!("A(n_i,n_j,n_k), B(n_j,n_l,n_k) -> C(n_i,n_l). Column-major.");
    println!();

    println!("(1) square: n_i = n_j = n_k = n_l = 30");
    run_tcontract_case("square", 30, 30, 30, 30, 0, 1);
    println!();

    println!("(2) n_i = n_l >> n_j: (2000,50,50) * (50,2000,50) -> (2000,2000)");
    run_tcontract_case("tall_skinny", 2000, 50, 50, 2000, 2, 3);
    println!();

    println!("(3) n_i = n_l << n_j: (50,2000,2000) * (2000,50,2000) -> (50,50)");
    run_tcontract_case("short_wide", 50, 2000, 2000, 50, 4, 5);
}
