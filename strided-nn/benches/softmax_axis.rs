//! A4 bench: softmax over a contiguous axis (`inner == 1`) and a strided axis, against
//! `softmax_axis0`, forward and VJP.
//!
//! Plain `std::time` harness (no new dependency). Serial `Exec`; pin the process to one core
//! (`taskset -c N`). Prints median / min nanoseconds over `SAMPLES` samples per cell.
//!
//! `cargo bench -p strided-nn --bench softmax_axis`  (env `SAMPLES`, default 31)
//!
//! Shapes are `[width, rows]` column-major. "contig" is the softmax over `width` with stride 1
//! (`inner = 1, axis_len = width, outer = rows`); "strided" is the softmax over the axis of
//! length `width` at stride `rows` (`inner = rows, axis_len = width, outer = 1`) -- the same
//! element count and the same reduction length. "gather" is the materialized-copy
//! alternative for the strided case: transpose into contiguous columns, `softmax_axis0`,
//! transpose back.

use core::mem::MaybeUninit;
use std::hint::black_box;
use std::time::Instant;

use strided_basic::Exec;
use strided_nn::{
    scratch_len, softmax_axis0_f32, softmax_axis0_f64, softmax_axis_f32, softmax_axis_f64,
    softmax_axis_scratch_len, softmax_vjp_axis_f32, softmax_vjp_axis_f64,
    softmax_vjp_axis_scratch_len, AxisShape,
};

fn samples() -> usize {
    std::env::var("SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(31)
}

/// Median and min ns of one call, inner repeat count chosen to run ~2 ms per sample.
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
    ($t:ty, $name:literal, $axis0:ident, $fwd:ident, $vjp:ident) => {{
        for &width in &[512usize, 1024, 2048, 4096] {
            for &rows in &[16usize, 256, 4096] {
                let n = width * rows;
                let x: Vec<$t> = (0..n)
                    .map(|i| ((i * 2654435761) % 1000) as $t / 40.0 - 12.0)
                    .collect();
                let dy = x.clone();
                let mut out = vec![MaybeUninit::<$t>::uninit(); n];
                let mut y = vec![0.0 as $t; n];
                let mut dx = vec![0.0 as $t; n];
                let mut e = Exec::serial();

                // Contiguous axis: the existing kernel and the generalised entry.
                let mut sc0 = vec![0.0 as $t; scratch_len::<$t>(width, n, 1)];
                let old = time(|| {
                    $axis0(width, black_box(&x), &mut out, &mut sc0, e.reborrow()).unwrap();
                });
                let c = AxisShape { inner: 1, axis_len: width, outer: rows };
                let mut scc = vec![0.0 as $t; softmax_axis_scratch_len::<$t>(c, 1)];
                let new = time(|| {
                    $fwd(c, black_box(&x), &mut out, &mut scc, e.reborrow()).unwrap();
                });
                // Forward output for the VJP.
                $fwd(c, &x, &mut out, &mut scc, e.reborrow()).unwrap();
                for (d, s) in y.iter_mut().zip(&out) {
                    // SAFETY: the call above wrote every slot.
                    *d = unsafe { s.assume_init() };
                }
                let vjp_c = time(|| {
                    $vjp(c, black_box(&y), &dy, &mut dx, &mut [], e.reborrow()).unwrap();
                });
                println!(
                    "{} w={:<5} rows={:<5} contig  fwd axis0 {:>10.0} ns  axis {:>10.0} ns  ratio {:.3}   vjp {:>10.0} ns",
                    $name, width, rows, old.0, new.0, new.0 / old.0, vjp_c.0
                );

                // Strided axis: stride `rows`, native walk versus gather-copy.
                let s = AxisShape { inner: rows, axis_len: width, outer: 1 };
                let mut scs = vec![0.0 as $t; softmax_axis_scratch_len::<$t>(s, 1)];
                let walk = time(|| {
                    $fwd(s, black_box(&x), &mut out, &mut scs, e.reborrow()).unwrap();
                });
                let mut tin = vec![0.0 as $t; n];
                let mut tout = vec![MaybeUninit::<$t>::uninit(); n];
                let mut sct = vec![0.0 as $t; scratch_len::<$t>(width, n, 1)];
                let gather = time(|| {
                    // [rows, width] column-major -> [width, rows] columns.
                    let src = black_box(&x);
                    for j in 0..width {
                        for i in 0..rows {
                            tin[i * width + j] = src[j * rows + i];
                        }
                    }
                    $axis0(width, &tin, &mut tout, &mut sct, e.reborrow()).unwrap();
                    for j in 0..width {
                        for i in 0..rows {
                            // SAFETY: `axis0` wrote every slot of `tout`.
                            let v = unsafe { tout[i * width + j].assume_init() };
                            out[j * rows + i] = MaybeUninit::new(v);
                        }
                    }
                });
                $fwd(s, &x, &mut out, &mut scs, e.reborrow()).unwrap();
                for (d, o) in y.iter_mut().zip(&out) {
                    // SAFETY: written by the call above.
                    *d = unsafe { o.assume_init() };
                }
                let mut vs = vec![0.0 as $t; softmax_vjp_axis_scratch_len::<$t>(s, 1)];
                let vjp_s = time(|| {
                    $vjp(s, black_box(&y), &dy, &mut dx, &mut vs, e.reborrow()).unwrap();
                });
                println!(
                    "{} w={:<5} rows={:<5} strided fwd walk  {:>10.0} ns  gather {:>10.0} ns  walk/axis0-contig {:.3}  gather/walk {:.3}   vjp {:>10.0} ns",
                    $name, width, rows, walk.0, gather.0, walk.0 / old.0, gather.0 / walk.0, vjp_s.0
                );
            }
        }
    }};
}

fn main() {
    println!("samples per cell: {}", samples());
    grid!(f32, "f32", softmax_axis0_f32, softmax_axis_f32, softmax_vjp_axis_f32);
    grid!(f64, "f64", softmax_axis0_f64, softmax_axis_f64, softmax_vjp_axis_f64);
}
