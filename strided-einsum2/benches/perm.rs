//! Permutation benchmark (ein"ijkl->ljki": reorder dimensions).
//!
//! Matches Julia OMEinsum ein"ijkl->ljki" with shape (30, 30, 30, 30).
//! Unary in Julia; we permute then copy via strided_kernel::copy_into.
//! Benchmarks both f64 and Complex64. Column-major for Julia parity.

use num_complex::Complex64;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::hint::black_box;
use std::time::{Duration, Instant};
use strided_kernel::copy_into;
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

/// Permute ijkl -> ljki (perm [3,1,2,0]) then copy; #[inline(never)] prevents hoisting.
#[inline(never)]
fn run_perm_f64(c: &mut StridedArray<f64>, a: &StridedArray<f64>) -> Result<(), strided_view::StridedError> {
    let permuted = a.view().permute(&[3, 1, 2, 0])?;
    copy_into(&mut c.view_mut(), &permuted)
}

#[inline(never)]
fn run_perm_complex64(
    c: &mut StridedArray<Complex64>,
    a: &StridedArray<Complex64>,
) -> Result<(), strided_view::StridedError> {
    let permuted = a.view().permute(&[3, 1, 2, 0])?;
    copy_into(&mut c.view_mut(), &permuted)
}

fn main() {
    println!("strided-einsum2 bench: perm (ein\"ijkl->ljki\")");
    println!("Shape (30, 30, 30, 30). Column-major for Julia parity.");
    println!();

    let n = 30usize;
    let shape_4 = [n, n, n, n];

    // f64
    let mut rng = StdRng::seed_from_u64(0);
    let a = StridedArray::<f64>::from_fn_col_major(&shape_4, |_| rng.gen::<f64>());
    let mut c = StridedArray::<f64>::col_major(&shape_4);

    println!("perm: Float64");
    bench_n("perm_f64", 2, 5, || {
        run_perm_f64(&mut c, &a).unwrap();
        black_box(c.data().as_ptr());
    });
    println!();

    // Complex64
    let mut rng_c = StdRng::seed_from_u64(1);
    let a_c = StridedArray::<Complex64>::from_fn_col_major(&shape_4, |_| {
        Complex64::new(rng_c.gen::<f64>(), rng_c.gen::<f64>())
    });
    let mut c_c = StridedArray::<Complex64>::col_major(&shape_4);

    println!("perm: ComplexF64");
    bench_n("perm_Complex64", 2, 5, || {
        run_perm_complex64(&mut c_c, &a_c).unwrap();
        black_box(c_c.data().as_ptr());
    });
}
