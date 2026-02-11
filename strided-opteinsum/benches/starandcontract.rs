//! Star-and-contract benchmark via opteinsum (ein"ij,ik,ik->j").

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
    println!("{label}: {:.3} ms", avg.as_secs_f64() * 1e3);
    avg
}

fn main() {
    let n = 50usize;
    let shape_2 = [n, n];
    let code = parse_einsum("ij,ik,ik->j").unwrap();
    println!(
        "strided-opteinsum bench: starandcontract (ein \"ij,ik,ik->j\"), large ({}x{})",
        n, n
    );

    let mut rng = StdRng::seed_from_u64(0);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_2, |_| rng.gen::<f64>());
    let b = StridedArray::<f64>::from_fn_col_major(&shape_2, |_| rng.gen::<f64>());
    let c = StridedArray::<f64>::from_fn_col_major(&shape_2, |_| rng.gen::<f64>());
    let a_view = a.view();
    let b_view = b.view();
    let c_view = c.view();
    println!("starandcontract (Float64):");
    bench_n("starandcontract_f64_large", 1, 3, || {
        let result = code
            .evaluate(vec![
                EinsumOperand::from_view(&a_view),
                EinsumOperand::from_view(&b_view),
                EinsumOperand::from_view(&c_view),
            ])
            .unwrap();
        match result {
            EinsumOperand::F64(data) => black_box(data.as_array().data().as_ptr()),
            _ => unreachable!("expected f64 output"),
        };
    });

    let mut rng_c = StdRng::seed_from_u64(1);
    let ac = StridedArray::<Complex64>::from_fn_col_major(&shape_2, |_| {
        Complex64::new(rng_c.gen(), rng_c.gen())
    });
    let bc = StridedArray::<Complex64>::from_fn_col_major(&shape_2, |_| {
        Complex64::new(rng_c.gen(), rng_c.gen())
    });
    let cc = StridedArray::<Complex64>::from_fn_col_major(&shape_2, |_| {
        Complex64::new(rng_c.gen(), rng_c.gen())
    });
    let ac_view = ac.view();
    let bc_view = bc.view();
    let cc_view = cc.view();
    println!("starandcontract (ComplexF64):");
    bench_n("starandcontract_Complex64_large", 1, 3, || {
        let result = code
            .evaluate(vec![
                EinsumOperand::from_view(&ac_view),
                EinsumOperand::from_view(&bc_view),
                EinsumOperand::from_view(&cc_view),
            ])
            .unwrap();
        match result {
            EinsumOperand::C64(data) => black_box(data.as_array().data().as_ptr()),
            _ => unreachable!("expected complex output"),
        };
    });
}
