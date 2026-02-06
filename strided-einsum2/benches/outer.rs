//! Outer product benchmark (ein"ij,kl->ijkl": D[i,j,k,l] = A[i,j]*B[k,l]).
//! Matches Julia OMEinsum suite "outer", large size only (100,100).
//! Uses einsum2_into.

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
    println!("{label}: {:.3} ms", avg.as_secs_f64() * 1e3);
    avg
}

fn main() {
    let n = 100usize;
    let shape_2 = [n, n];
    let shape_4 = [n, n, n, n];
    println!(
        "strided-einsum2 bench: outer (ein \"ij,kl->ijkl\"), large ({}x{})",
        n, n
    );

    let mut rng = StdRng::seed_from_u64(0);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_2, |_| rng.gen::<f64>());
    let b = StridedArray::<f64>::from_fn_col_major(&shape_2, |_| rng.gen::<f64>());
    let mut d = StridedArray::<f64>::col_major(&shape_4);
    println!("outer (Float64):");
    bench_n("outer_f64_large", 1, 3, || {
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
    });

    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let mut rng_c = StdRng::seed_from_u64(1);
    let ac = StridedArray::<Complex64>::from_fn_col_major(&shape_2, |_| {
        Complex64::new(rng_c.gen(), rng_c.gen())
    });
    let bc = StridedArray::<Complex64>::from_fn_col_major(&shape_2, |_| {
        Complex64::new(rng_c.gen(), rng_c.gen())
    });
    let mut dc = StridedArray::<Complex64>::col_major(&shape_4);
    println!("outer (ComplexF64):");
    bench_n("outer_Complex64_large", 1, 3, || {
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
    });
}
