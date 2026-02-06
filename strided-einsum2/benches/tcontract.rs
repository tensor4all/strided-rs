//! Tensor contraction benchmark (einsum2 C_il = sum_jk A_ijk * B_jlk).
//!
//! Matches Julia OMEinsum ein"ijk,jlk->il" with shape (30, 30, 30).
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
    println!("{label}: {:.3} ms", avg.as_secs_f64() * 1e3);
    avg
}

fn main() {
    println!("strided-einsum2 bench: tcontract (einsum2 ijk,jlk->il)");
    println!("Shape (30, 30, 30) -> (30, 30). Column-major for Julia parity.");
    println!();

    let n = 30usize;
    let shape_3 = [n, n, n];
    let shape_2 = [n, n];

    // f64
    let mut rng = StdRng::seed_from_u64(0);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_3, |_| rng.gen::<f64>());
    let b = StridedArray::<f64>::from_fn_col_major(&shape_3, |_| rng.gen::<f64>());
    let mut c = StridedArray::<f64>::col_major(&shape_2);

    println!("tcontract (einsum2_into): Float64");
    bench_n("tcontract_f64", 2, 5, || {
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
    });
    println!();

    // Complex64
    let mut rng_c = StdRng::seed_from_u64(1);
    let a_c = StridedArray::<Complex64>::from_fn_col_major(&shape_3, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });
    let b_c = StridedArray::<Complex64>::from_fn_col_major(&shape_3, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });
    let mut c_c = StridedArray::<Complex64>::col_major(&shape_2);

    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);

    println!("tcontract (einsum2_into): ComplexF64");
    bench_n("tcontract_Complex64", 2, 5, || {
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
    });
}
