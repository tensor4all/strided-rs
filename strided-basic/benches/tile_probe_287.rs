//! Diagnostic for tensor4all/strided-rs#287 (not a gate): `mul_into` with
//! D, A column-major and B row-major through the production walk at forced
//! tiles, beside a hand scalar loop that walks the same tiles in the same
//! order (axis-1 tiles outer, axis-0 tiles inner, axis 0 innermost).
//!
//! ```text
//! taskset -c <core> cargo bench -p strided-basic --features bench-hooks,parallel \
//!     --bench tile_probe_287 -- <m> <n> f64|f32 <b0>x<b1> ...
//! ```

use std::hint::black_box;
use std::time::Instant;

use strided_basic::bench_hooks::FORCED_TILES;
use strided_basic::{mul_into, Exec, StridedView, StridedViewMut};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// `d[i + j m] = a[i + j m] * b[i n + j]`, tile by tile.
fn hand<T: Copy + std::ops::Mul<Output = T>>(
    d: &mut [T],
    a: &[T],
    b: &[T],
    m: usize,
    n: usize,
    b0: usize,
    b1: usize,
) {
    let (dp, ap, bp) = (d.as_mut_ptr(), a.as_ptr(), b.as_ptr());
    for j0 in (0..n).step_by(b1) {
        for i0 in (0..m).step_by(b0) {
            for j in j0..(j0 + b1).min(n) {
                for i in i0..(i0 + b0).min(m) {
                    // SAFETY: `i < m`, `j < n`, and every buffer holds `m * n`.
                    unsafe { *dp.add(i + j * m) = *ap.add(i + j * m) * *bp.add(i * n + j) };
                }
            }
        }
    }
}

fn probe<T>(m: usize, n: usize, tiles: &[(usize, usize)], from: fn(usize) -> T, name: &str)
where
    T: Copy + std::ops::Mul<Output = T> + Send + Sync + 'static,
{
    let a: Vec<T> = (0..m * n).map(from).collect();
    let b: Vec<T> = (0..m * n).map(|k| from(k * 7 + 3)).collect();
    let mut d: Vec<T> = vec![from(0); m * n];
    let (ds, bs) = ([1isize, m as isize], [n as isize, 1]);
    for round in 0..5 {
        for &(b0, b1) in tiles {
            let mut walk = || {
                *FORCED_TILES.lock().unwrap() = vec![b0, b1];
                let mut dv = StridedViewMut::<T>::new(&mut d, &[m, n], &ds, 0).unwrap();
                let av = StridedView::<T>::new(&a, &[m, n], &ds, 0).unwrap();
                let bv = StridedView::<T>::new(&b, &[m, n], &bs, 0).unwrap();
                mul_into(&mut dv, &av, &bv, Exec::serial()).unwrap();
                FORCED_TILES.lock().unwrap().clear();
            };
            let time = |f: &mut dyn FnMut()| {
                f();
                median(
                    (0..11)
                        .map(|_| {
                            let t = Instant::now();
                            f();
                            t.elapsed().as_secs_f64() * 1e3
                        })
                        .collect(),
                )
            };
            let w = time(&mut walk);
            let h = time(&mut || {
                hand(&mut d, &a, &b, m, n, b0, b1);
                black_box(d.as_mut_ptr());
            });
            println!("{round}\t{m}x{n}\t{name}\t[{b0},{b1}]\twalk_ms {w:.3}\thand_ms {h:.3}");
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| a != "--bench")
        .collect();
    let (m, n): (usize, usize) = (args[0].parse().unwrap(), args[1].parse().unwrap());
    let tiles: Vec<(usize, usize)> = args[3..]
        .iter()
        .map(|t| {
            let (x, y) = t.split_once('x').unwrap();
            (x.parse().unwrap(), y.parse().unwrap())
        })
        .collect();
    match args[2].as_str() {
        "f64" => probe::<f64>(m, n, &tiles, |k| (k % 1013) as f64 * 0.37, "f64"),
        _ => probe::<f32>(m, n, &tiles, |k| (k % 1013) as f32 * 0.37, "f32"),
    }
}
