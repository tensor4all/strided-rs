//! Partial trace benchmark via opteinsum (ein"iij->j").

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
    println!("{label}: {:.3} ms", avg.as_secs_f64() * 1e3);
    avg
}

fn main() {
    println!("strided-opteinsum bench: ptrace (iij->j)");
    println!("Shape (100, 100, 100) -> (100). Column-major.");
    println!();

    let n = 100usize;
    let shape_3 = [n, n, n];
    let code = parse_einsum("iij->j").unwrap();

    let mut rng = StdRng::seed_from_u64(0);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_3, |_| rng.gen::<f64>());
    let a_view = a.view();

    println!("ptrace: Float64");
    bench_n("ptrace_f64", 2, 5, || {
        let result = code
            .evaluate(vec![EinsumOperand::from_view_f64(&a_view)])
            .unwrap();
        match result {
            EinsumOperand::F64(data) => black_box(data.as_array().data().as_ptr()),
            _ => unreachable!("expected f64 output"),
        };
    });
    println!();

    let mut rng_c = StdRng::seed_from_u64(1);
    let a_c = StridedArray::<Complex64>::from_fn_col_major(&shape_3, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });
    let a_c_view = a_c.view();

    println!("ptrace: ComplexF64");
    bench_n("ptrace_Complex64", 2, 5, || {
        let result = code
            .evaluate(vec![EinsumOperand::from_view_c64(&a_c_view)])
            .unwrap();
        match result {
            EinsumOperand::C64(data) => black_box(data.as_array().data().as_ptr()),
            _ => unreachable!("expected complex output"),
        };
    });
}
