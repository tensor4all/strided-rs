//! Small unfusable tiles under a long axis (tensor4all/strided-rs#526), with
//! the regression rows the change has to keep.
//!
//! `harness = false`: a table, not a criterion suite. Every row is timed with
//! `Exec::serial()`; the number is the median over [`REPS`] samples of the
//! nanoseconds per element (per `n x n` block for the tile rows as well).
//!
//! ```text
//! taskset -c 40 cargo bench -p strided-basic --bench small_tile_526
//! ```
//!
//! Rows:
//!
//! - `[n, n, B]`, n = 2, 3, 4 (and 8, which is not a tile), B = 4096 and
//!   16384, f64 and `Complex<f64>`, column-major with `b` permuted
//!   `[1, 0, 2]` (strides `[n, 1, n²]`): `zip_map2_into` a − bᵀ, `copy_into`
//!   bᵀ, `copy_into_uninit` bᵀ, the same on contiguous operands, and a naive
//!   pointer loop over the batch (block unrolled inside), which is the
//!   baseline the issue measures against.
//! - Regression: the `[512, 512]` transpose (`copy_into`, `copy_into_uninit`,
//!   `zip_map2_into`), the strided-rs#466 power-of-two permutations (3D
//!   `[d, d, d]` with perm `[1, 2, 0]` and `[2, 1, 0]`, d = 31, 32, 64; 2D
//!   `[32, 1024]` and `[64, 2048]` transposed), and a contiguous `[1 << 20]`
//!   zip and copy.
//!
//! The source uses only the public API, so the same file measures any
//! revision that has it.

use std::hint::black_box;
use std::mem::MaybeUninit;
use std::ops::Sub;
use std::time::Instant;

use num_complex::Complex64;
use strided_basic::{
    copy_into, copy_into_uninit, zip_map2_into, Exec, StridedView, StridedViewMut,
};

/// Timed samples per row.
const REPS: usize = 11;
/// Elements each sample covers at least; small rows repeat the call.
const SAMPLE_ELEMENTS: usize = 1 << 21;

trait Elem: Copy + Send + Sync + Default + PartialEq + Sub<Output = Self> + 'static {
    const NAME: &'static str;
    fn make(i: usize) -> Self;
}

impl Elem for f64 {
    const NAME: &'static str = "f64";
    fn make(i: usize) -> Self {
        ((i * 13) % 31) as f64 - 15.0
    }
}

impl Elem for Complex64 {
    const NAME: &'static str = "c64";
    fn make(i: usize) -> Self {
        Complex64::new(((i * 13) % 31) as f64 - 15.0, ((i * 5) % 17) as f64)
    }
}

fn col_major(dims: &[usize]) -> Vec<isize> {
    let mut strides = vec![1isize; dims.len()];
    for i in 1..dims.len() {
        strides[i] = strides[i - 1] * dims[i - 1] as isize;
    }
    strides
}

/// Median nanoseconds per element of `body`, which touches `elements`.
fn median_ns(elements: usize, mut body: impl FnMut()) -> f64 {
    let calls = SAMPLE_ELEMENTS.div_ceil(elements).max(1);
    body();
    body();
    let mut samples: Vec<f64> = (0..REPS)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..calls {
                body();
            }
            start.elapsed().as_secs_f64() * 1e9 / (calls * elements) as f64
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[REPS / 2]
}

fn print_row(label: &str, per_block: usize, ns: &[(&str, f64)]) {
    let cells: Vec<String> = ns
        .iter()
        .map(|(name, v)| format!("{name} {:8.3} ({:7.2})", v, v * per_block as f64))
        .collect();
    println!("{label:24} | {}", cells.join(" | "));
}

fn tile_row<T: Elem>(n: usize, batch: usize) {
    let dims = [n, n, batch];
    let strides = col_major(&dims);
    let len = n * n * batch;
    let a: Vec<T> = (0..len).map(T::make).collect();
    let b: Vec<T> = (0..len).map(|i| T::make(i * 7 + 3)).collect();
    let mut out = vec![T::default(); len];
    let mut scratch = vec![MaybeUninit::<T>::uninit(); len];
    let av = StridedView::<T>::new(&a, &dims, &strides, 0).unwrap();
    let bv = StridedView::<T>::new(&b, &dims, &strides, 0).unwrap();
    let bt = bv.permute(&[1, 0, 2]).unwrap();

    let zip_t = median_ns(len, || {
        let mut dv = StridedViewMut::new(&mut out, &dims, &strides, 0).unwrap();
        zip_map2_into(&mut dv, &av, black_box(&bt), |x, y| x - y, Exec::serial()).unwrap();
    });
    let zip_c = median_ns(len, || {
        let mut dv = StridedViewMut::new(&mut out, &dims, &strides, 0).unwrap();
        zip_map2_into(&mut dv, &av, black_box(&bv), |x, y| x - y, Exec::serial()).unwrap();
    });
    let copy_t = median_ns(len, || {
        let mut dv = StridedViewMut::new(&mut out, &dims, &strides, 0).unwrap();
        copy_into(&mut dv, black_box(&bt), Exec::serial()).unwrap();
    });
    let copy_c = median_ns(len, || {
        let mut dv = StridedViewMut::new(&mut out, &dims, &strides, 0).unwrap();
        copy_into(&mut dv, black_box(&bv), Exec::serial()).unwrap();
    });
    let uninit_t = median_ns(len, || {
        let mut dv = StridedViewMut::new(&mut scratch, &dims, &strides, 0).unwrap();
        copy_into_uninit(&mut dv, black_box(&bt), Exec::serial()).unwrap();
    });
    let naive = median_ns(len, || {
        let (dp, ap, bp) = (out.as_mut_ptr(), a.as_ptr(), black_box(b.as_ptr()));
        let nn = n * n;
        for k in 0..batch {
            for j in 0..n {
                for i in 0..n {
                    // SAFETY: every index is below `len` = n * n * batch.
                    unsafe {
                        *dp.add(k * nn + i + n * j) =
                            *ap.add(k * nn + i + n * j) - *bp.add(k * nn + j + n * i);
                    }
                }
            }
        }
    });
    print_row(
        &format!("{} [{n},{n},{batch}]", T::NAME),
        n * n,
        &[
            ("zipT", zip_t),
            ("zip", zip_c),
            ("naiveT", naive),
            ("copyT", copy_t),
            ("copy", copy_c),
            ("uninitT", uninit_t),
        ],
    );
}

/// `copy_into` and `copy_into_uninit` of a source with column-major strides of
/// `src_dims` permuted by `perm` into a column-major destination, and
/// `zip_map2_into` of the destination layout with that source.
fn perm_row<T: Elem>(src_dims: &[usize], perm: &[usize]) {
    let len: usize = src_dims.iter().product();
    let src: Vec<T> = (0..len).map(T::make).collect();
    let other: Vec<T> = (0..len).map(|i| T::make(i + 5)).collect();
    let src_strides = col_major(src_dims);
    let sv = StridedView::<T>::new(&src, src_dims, &src_strides, 0)
        .unwrap()
        .permute(perm)
        .unwrap();
    let dims: Vec<usize> = sv.dims().to_vec();
    let strides = col_major(&dims);
    let ov = StridedView::<T>::new(&other, &dims, &strides, 0).unwrap();
    let mut out = vec![T::default(); len];
    let mut scratch = vec![MaybeUninit::<T>::uninit(); len];
    let copy = median_ns(len, || {
        let mut dv = StridedViewMut::new(&mut out, &dims, &strides, 0).unwrap();
        copy_into(&mut dv, black_box(&sv), Exec::serial()).unwrap();
    });
    let uninit = median_ns(len, || {
        let mut dv = StridedViewMut::new(&mut scratch, &dims, &strides, 0).unwrap();
        copy_into_uninit(&mut dv, black_box(&sv), Exec::serial()).unwrap();
    });
    let zip = median_ns(len, || {
        let mut dv = StridedViewMut::new(&mut out, &dims, &strides, 0).unwrap();
        zip_map2_into(&mut dv, &ov, black_box(&sv), |x, y| x - y, Exec::serial()).unwrap();
    });
    print_row(
        &format!("{} {src_dims:?} {perm:?}", T::NAME),
        1,
        &[("copy", copy), ("uninit", uninit), ("zip", zip)],
    );
}

fn contiguous_row<T: Elem>(len: usize) {
    let dims = [len];
    let strides = [1isize];
    let a: Vec<T> = (0..len).map(T::make).collect();
    let b: Vec<T> = (0..len).map(|i| T::make(i + 1)).collect();
    let mut out = vec![T::default(); len];
    let av = StridedView::<T>::new(&a, &dims, &strides, 0).unwrap();
    let bv = StridedView::<T>::new(&b, &dims, &strides, 0).unwrap();
    let zip = median_ns(len, || {
        let mut dv = StridedViewMut::new(&mut out, &dims, &strides, 0).unwrap();
        zip_map2_into(&mut dv, &av, black_box(&bv), |x, y| x - y, Exec::serial()).unwrap();
    });
    let copy = median_ns(len, || {
        let mut dv = StridedViewMut::new(&mut out, &dims, &strides, 0).unwrap();
        copy_into(&mut dv, black_box(&bv), Exec::serial()).unwrap();
    });
    print_row(
        &format!("{} [{len}] contiguous", T::NAME),
        1,
        &[("zip", zip), ("copy", copy)],
    );
}

fn main() {
    println!("ns/element (ns/block in parentheses); median of {REPS}, Exec::serial()");
    println!("-- small tiles under a long axis (#526)");
    for (n, batch) in [
        (2usize, 4096usize),
        (2, 16384),
        (3, 4096),
        (3, 16384),
        (4, 4096),
        (4, 16384),
        (8, 4096),
    ] {
        tile_row::<f64>(n, batch);
        tile_row::<Complex64>(n, batch);
    }
    println!("-- [512, 512] transpose");
    perm_row::<f64>(&[512, 512], &[1, 0]);
    perm_row::<Complex64>(&[512, 512], &[1, 0]);
    println!("-- strided-rs#466 power-of-two permutations");
    for d in [31usize, 32, 64] {
        for perm in [[1usize, 2, 0], [2, 1, 0]] {
            perm_row::<f64>(&[d, d, d], &perm);
            perm_row::<Complex64>(&[d, d, d], &perm);
        }
    }
    for dims in [[32usize, 1024], [64, 2048]] {
        perm_row::<f64>(&dims, &[1, 0]);
        perm_row::<Complex64>(&dims, &[1, 0]);
    }
    println!("-- contiguous");
    contiguous_row::<f64>(1 << 20);
    contiguous_row::<Complex64>(1 << 20);
}
