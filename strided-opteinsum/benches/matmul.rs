//! Matrix multiplication benchmark (opteinsum C_ik = A_ij * B_jk).

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

fn run_matmul_case(case_name: &str, n1: usize, n2: usize, n3: usize, seed_f64: u64, seed_c: u64) {
    let shape_a = [n1, n2];
    let shape_b = [n2, n3];

    let code = parse_einsum("ij,jk->ik").unwrap();

    let mut rng = StdRng::seed_from_u64(seed_f64);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_a, |_| rng.gen::<f64>());
    let b = StridedArray::<f64>::from_fn_col_major(&shape_b, |_| rng.gen::<f64>());
    let a_view = a.view();
    let b_view = b.view();

    println!("  Float64:");
    bench_n(&format!("{case_name}_f64_{n1}x{n2}x{n3}"), 2, 5, || {
        let result = code
            .evaluate(
                vec![
                    EinsumOperand::from_view(&a_view),
                    EinsumOperand::from_view(&b_view),
                ],
                None,
            )
            .unwrap();
        match result {
            EinsumOperand::F64(data) => black_box(data.as_array().data().as_ptr()),
            _ => unreachable!("expected f64 output"),
        };
    });

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
        &format!("{case_name}_Complex64_{n1}x{n2}x{n3}"),
        2,
        5,
        || {
            let result = code
                .evaluate(
                    vec![
                        EinsumOperand::from_view(&a_c_view),
                        EinsumOperand::from_view(&b_c_view),
                    ],
                    None,
                )
                .unwrap();
            match result {
                EinsumOperand::C64(data) => black_box(data.as_array().data().as_ptr()),
                _ => unreachable!("expected complex output"),
            };
        },
    );
}

fn main() {
    println!("strided-opteinsum bench: matmul (ij,jk->ik)");
    println!("(n1,n2) * (n2,n3) = (n1,n3). Column-major.");
    println!();

    println!("(1) square: n1 = n2 = n3 = 1000");
    run_matmul_case("square", 1000, 1000, 1000, 0, 1);
    println!();

    println!("(2) n1 = n3 >> n2: (2000, 50) * (50, 2000)");
    run_matmul_case("tall_skinny", 2000, 50, 2000, 2, 3);
    println!();

    println!("(3) n1 = n3 << n2: (50, 2000) * (2000, 50)");
    run_matmul_case("short_wide", 50, 2000, 50, 4, 5);
}
