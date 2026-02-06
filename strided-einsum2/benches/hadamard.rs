//! Hadamard (element-wise) benchmark (ein"ijk,ijk->ijk": C = A .* B).
//! Matches Julia OMEinsum suite "hadamard", large size only (100,100,100).
//! Uses strided_kernel::zip_map2_into.

use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::hint::black_box;
use std::time::{Duration, Instant};
use strided_kernel::zip_map2_into;
use strided_view::StridedArray;

fn mean(durations: &[Duration]) -> Duration {
    let total_nanos: u128 = durations.iter().map(|d| d.as_nanos()).sum();
    Duration::from_nanos((total_nanos / durations.len() as u128) as u64)
}

fn bench_n(label: &str, warmup: usize, iters: usize, mut f: impl FnMut()) -> Duration {
    for _ in 0..warmup { f(); }
    let samples: Vec<Duration> = (0..iters).map(|_| { let t = Instant::now(); f(); t.elapsed() }).collect();
    let avg = mean(&samples);
    println!("{label}: {:.3} ms", avg.as_secs_f64() * 1e3);
    avg
}

fn main() {
    let n = 100usize;
    let shape_3 = [n, n, n];
    println!("strided-einsum2 bench: hadamard (ein \"ijk,ijk->ijk\"), large ({}^3)", n);

    let mut rng = StdRng::seed_from_u64(0);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_3, |_| rng.gen::<f64>());
    let b = StridedArray::<f64>::from_fn_col_major(&shape_3, |_| rng.gen::<f64>());
    let mut c = StridedArray::<f64>::col_major(&shape_3);
    println!("hadamard (Float64):");
    bench_n("hadamard_f64_large", 1, 3, || {
        zip_map2_into(&mut c.view_mut(), &a.view(), &b.view(), |x, y| x * y).unwrap();
        black_box(c.data().as_ptr());
    });

    let mut rng_c = StdRng::seed_from_u64(1);
    let ac = StridedArray::<Complex64>::from_fn_col_major(&shape_3, |_| Complex64::new(rng_c.gen(), rng_c.gen()));
    let bc = StridedArray::<Complex64>::from_fn_col_major(&shape_3, |_| Complex64::new(rng_c.gen(), rng_c.gen()));
    let mut cc = StridedArray::<Complex64>::col_major(&shape_3);
    println!("hadamard (ComplexF64):");
    bench_n("hadamard_Complex64_large", 1, 3, || {
        zip_map2_into(&mut cc.view_mut(), &ac.view(), &bc.view(), |x, y| x * y).unwrap();
        black_box(cc.data().as_ptr());
    });
}
