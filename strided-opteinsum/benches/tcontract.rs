//! Tensor contraction benchmark via opteinsum (ijk,jlk->il).

use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::hint::black_box;
use std::time::{Duration, Instant};
use strided_opteinsum::{parse_einsum, EinsumOperand};
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

    let code = parse_einsum("ijk,jlk->il").unwrap();

    let mut rng = StdRng::seed_from_u64(seed_f64);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_a, |_| rng.gen::<f64>());
    let b = StridedArray::<f64>::from_fn_col_major(&shape_b, |_| rng.gen::<f64>());
    let a_view = a.view();
    let b_view = b.view();

    println!("  Float64:");
    bench_n(
        &format!("{case_name}_f64_{ni}x{nj}x{nk}_x{nl}"),
        2,
        5,
        || {
            let result = code
                .evaluate(vec![
                    EinsumOperand::from_view_f64(&a_view),
                    EinsumOperand::from_view_f64(&b_view),
                ])
                .unwrap();
            match result {
                EinsumOperand::F64(data) => black_box(data.as_array().data().as_ptr()),
                _ => unreachable!("expected f64 output"),
            };
        },
    );

    let mut rng_c = StdRng::seed_from_u64(seed_c);
    let a_c = StridedArray::<Complex64>::from_fn_col_major(&shape_a, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });
    let b_c = StridedArray::<Complex64>::from_fn_col_major(&shape_b, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });
    let a_c_view = a_c.view();
    let b_c_view = b_c.view();

    println!("  ComplexF64:");
    bench_n(
        &format!("{case_name}_Complex64_{ni}x{nj}x{nk}_x{nl}"),
        2,
        5,
        || {
            let result = code
                .evaluate(vec![
                    EinsumOperand::from_view_c64(&a_c_view),
                    EinsumOperand::from_view_c64(&b_c_view),
                ])
                .unwrap();
            match result {
                EinsumOperand::C64(data) => black_box(data.as_array().data().as_ptr()),
                _ => unreachable!("expected complex output"),
            };
        },
    );
}

fn main() {
    println!("strided-opteinsum bench: tcontract (ijk,jlk->il)");
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
