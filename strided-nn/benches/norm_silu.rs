//! A3 bench: fused LayerNorm -> SiLU versus the two-pass A1 + A2 composition.
//!
//! Plain `std::time` harness (no new dependency). Serial `Exec`; pin the process to one core
//! (`taskset -c N`). Prints median / min nanoseconds over `SAMPLES` samples per cell.
//!
//! `cargo bench -p strided-nn --bench norm_silu`  (env `SAMPLES`, default 31)

use std::hint::black_box;
use std::time::Instant;

use strided_basic::Exec;
use strided_nn::activation as act;
use strided_nn::norm as nm;
use strided_nn::norm_silu as fu;

fn samples() -> usize {
    std::env::var("SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(31)
}

/// Median and min ns of one call, with the inner repeat count chosen to run ~2 ms per sample.
fn time(mut f: impl FnMut()) -> (f64, f64) {
    f();
    let t = Instant::now();
    f();
    let one = t.elapsed().as_nanos().max(1) as f64;
    let reps = ((2.0e6 / one).ceil() as usize).clamp(1, 100_000);
    let mut v: Vec<f64> = (0..samples())
        .map(|_| {
            let t = Instant::now();
            for _ in 0..reps {
                f();
            }
            t.elapsed().as_nanos() as f64 / reps as f64
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[0])
}

macro_rules! grid {
    ($t:ty, $name:literal, $ln:ident, $ln_in:ident, $silu:ident, $silu_vjp:ident,
     $fln:ident, $fln_in:ident) => {{
        for &width in &[512usize, 1024, 2048, 4096] {
            for &rows in &[1usize, 16, 256, 4096] {
                let n = rows * width;
                let x: Vec<$t> = (0..n).map(|i| ((i * 2654435761) % 1000) as $t / 250.0 - 2.0).collect();
                let w: Vec<$t> = (0..width).map(|i| 0.5 + (i % 7) as $t * 0.1).collect();
                let b: Vec<$t> = (0..width).map(|i| (i % 5) as $t * 0.05).collect();
                let dy = x.clone();
                let (mut z, mut y, mut d, mut dz, mut dx) =
                    (vec![0.0 as $t; n], vec![0.0 as $t; n], vec![0.0 as $t; n], vec![0.0 as $t; n], vec![0.0 as $t; n]);
                let (mut m, mut s) = (vec![0.0 as $t; rows], vec![0.0 as $t; rows]);
                let mut asc = vec![0.0 as $t; act::scratch_len(n, 1)];
                let mut fsc = vec![0.0 as $t; fu::norm_silu_scratch_len(rows, width, 1).unwrap()];
                let mut e = Exec::serial();

                let two_fwd = time(|| {
                    nm::$ln(width, 1e-5, black_box(&x), Some(&w), Some(&b), &mut z, Some((&mut m, &mut s)), e.reborrow()).unwrap();
                    act::$silu(&z, &mut y, Some(&mut d), &mut asc, e.reborrow()).unwrap();
                    black_box(&y);
                });
                let a1 = time(|| {
                    nm::$ln(width, 1e-5, black_box(&x), Some(&w), Some(&b), &mut z, Some((&mut m, &mut s)), e.reborrow()).unwrap();
                    black_box(&z);
                });
                let a2 = time(|| {
                    act::$silu(black_box(&z), &mut y, Some(&mut d), &mut asc, e.reborrow()).unwrap();
                    black_box(&y);
                });
                let fused_fwd = time(|| {
                    fu::$fln(width, 1e-5, black_box(&x), Some(&w), Some(&b), &mut y, Some((&mut m, &mut s)), Some(&mut d), &mut fsc, e.reborrow()).unwrap();
                    black_box(&y);
                });
                let two_vjp = time(|| {
                    act::$silu_vjp(black_box(&dy), &d, &mut dz, e.reborrow()).unwrap();
                    nm::$ln_in(width, &x, &m, &s, Some(&w), &dz, &mut dx, e.reborrow()).unwrap();
                    black_box(&dx);
                });
                let fused_vjp = time(|| {
                    fu::$fln_in(width, &x, &m, &s, Some(&w), &d, black_box(&dy), &mut dx, e.reborrow()).unwrap();
                    black_box(&dx);
                });
                println!(
                    "{:<4} fwd w={:<5} rows={:<5} two-pass {:>12.0} ns  fused {:>12.0} ns  ratio {:.3}   (min {:.0} / {:.0})  [A1 alone {:.0}, A2 alone {:.0}]",
                    $name, width, rows, two_fwd.0, fused_fwd.0, two_fwd.0 / fused_fwd.0, two_fwd.1, fused_fwd.1, a1.0, a2.0
                );
                println!(
                    "{:<4} vjp w={:<5} rows={:<5} two-pass {:>12.0} ns  fused {:>12.0} ns  ratio {:.3}   (min {:.0} / {:.0})",
                    $name, width, rows, two_vjp.0, fused_vjp.0, two_vjp.0 / fused_vjp.0, two_vjp.1, fused_vjp.1
                );
            }
        }
    }};
}

fn main() {
    println!("samples per cell: {}", samples());
    grid!(
        f32,
        "f32",
        layer_norm_forward_f32,
        layer_norm_vjp_input_f32,
        silu_forward_f32,
        silu_vjp_f32,
        layer_norm_silu_forward_f32,
        layer_norm_silu_vjp_input_f32
    );
    grid!(
        f64,
        "f64",
        layer_norm_forward_f64,
        layer_norm_vjp_input_f64,
        silu_forward_f64,
        silu_vjp_f64,
        layer_norm_silu_forward_f64,
        layer_norm_silu_vjp_input_f64
    );
}
