//! Many-index einsum benchmark via opteinsum (OMEinsum-style).

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
    println!("strided-opteinsum bench: manyinds (many-index einsum)");
    println!("Contract: abcdefghijkl, flnqrcipstuj -> abdeghkqrpstu");
    println!("All dimensions size 2. Column-major.");
    println!();

    let notation = "abcdefghijkl,flnqrcipstuj->abdeghkqrpstu";
    let code = parse_einsum(notation).unwrap();

    let dim = 2usize;
    let shape_a: Vec<usize> = (0..12).map(|_| dim).collect();
    let shape_b: Vec<usize> = (0..12).map(|_| dim).collect();

    let mut rng = StdRng::seed_from_u64(0);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_a, |_| rng.gen::<f64>());
    let b = StridedArray::<f64>::from_fn_col_major(&shape_b, |_| rng.gen::<f64>());
    let a_view = a.view();
    let b_view = b.view();

    println!("=== manyinds f64 (opteinsum) ===");
    bench_n("opteinsum_f64", 1, 5, || {
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
    });
    println!();

    let mut rng_c = StdRng::seed_from_u64(1);
    let a_c = StridedArray::<Complex64>::from_fn_col_major(&shape_a, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });
    let b_c = StridedArray::<Complex64>::from_fn_col_major(&shape_b, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });
    let a_c_view = a_c.view();
    let b_c_view = b_c.view();

    println!("=== manyinds Complex64 (opteinsum) ===");
    bench_n("opteinsum_Complex64", 1, 5, || {
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
    });
}
