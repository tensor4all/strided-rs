//! Paired `2^d` against `2^d ± 1` permutation rows for tensor4all/strided-rs#466.
//!
//! `harness = false`: a few-second table, not a criterion suite. Each row is
//! [`copy_into`] on a column-major source permuted onto a contiguous
//! column-major destination, in nanoseconds per element.
//!
//! The shapes come in a `2^d - 1`, `2^d`, `2^d + 1` triple (and around 96, the
//! issue's non-power-of-two case). #466 is the middle cell: the destination
//! axis that the source streams along is then a multiple of the 4 KiB L1 way
//! away from its neighbour, so every row of a macro tile lands in one L1 set.
//! The neighbours are the miss case — the same work over a similar cache
//! footprint without the aliasing — so `2^d/nb` is the aliasing penalty itself
//! rather than a comparison across unrelated shapes.
//!
//! Both element types and both thread counts are in the table because they take
//! different kernels: `Complex<f64>` the scalar fallback, `f64` the native
//! micro-kernel. A fix has to repair the first without costing the second.
//!
//! The ns/e columns are the **median over [`REPS`] samples**; `min`, `max` and
//! the half-spread `(max - min) / median` sit beside it. `2^d/nb` is the
//! **median of the per-sample ratios**: each sample times the triple back to
//! back in a rotating order, so background load lands on all three in the same
//! sample and cancels in that sample's ratio instead of on one independently
//! selected minimum.
//!
//! `digest` fingerprints the destination after the copy (element count, the sum
//! of the elements, and a position-weighted sum, read as integers). Two
//! revisions that print the same digests wrote the same elements; a row whose
//! digest moves is a correctness change, not a performance one.
//!
//! ```text
//! cargo bench -j 4 --features parallel -p strided-perm --bench l1_sets_466
//! L1_SETS_466_ONLY=c128 cargo bench -j 4 --features parallel -p strided-perm --bench l1_sets_466
//! ```
//!
//! `L1_SETS_466_ONLY` keeps only the rows whose `threads element` label
//! contains the value (`1T`, `4T`, `f64`, `c128`), so a revision can be re-timed
//! in pieces. The thread count the table reports is read back from the token
//! and asserted; the pool is built with exactly [`THREADS`] threads.

use std::hint::black_box;
use std::time::Instant;

use num_complex::Complex64;
use strided_perm::copy_into;
use strided_view::{StridedArray, StridedView, StridedViewMut};
use threadteam::{Exec, Pool};

/// Timed samples per cell.
const REPS: usize = 9;
/// Target number of elements one sample times: small shapes are copied several
/// times per sample so a sample is long against the clock, large ones once.
const SAMPLE_ELEMENTS: usize = 8_000_000;
/// Untimed samples before the first measured one.
const WARMUP: usize = 1;
/// Threads in the pooled token.
const THREADS: usize = 4;

/// One element type the table covers.
trait Elem: Copy + Default + Send + Sync + 'static {
    fn make(i: usize) -> Self;
    /// The element's bits as one wide integer, so a digest is exact.
    fn bits(self) -> u128;
    fn name() -> &'static str;
}

impl Elem for f64 {
    fn make(i: usize) -> Self {
        i as f64
    }
    fn bits(self) -> u128 {
        self.to_bits() as u128
    }
    fn name() -> &'static str {
        "f64"
    }
}

impl Elem for Complex64 {
    fn make(i: usize) -> Self {
        Complex64::new(i as f64, 1.0)
    }
    fn bits(self) -> u128 {
        self.re.to_bits() as u128 | ((self.im.to_bits() as u128) << 64)
    }
    fn name() -> &'static str {
        "c128"
    }
}

fn permuted_dims(dims: &[usize], perm: &[usize]) -> Vec<usize> {
    perm.iter().map(|&p| dims[p]).collect()
}

/// Where a cell's copies run: the calling thread, or a section of the pool.
#[derive(Clone, Copy)]
struct Placement<'a> {
    pool: Option<&'a Pool>,
}

impl Placement<'_> {
    /// A token for one call. Minted before the clock starts, as the copy itself
    /// does not own the token's set-up.
    fn token(&self) -> Exec<'_> {
        match self.pool {
            Some(pool) => Exec::new(pool, THREADS).expect("the pool holds THREADS threads"),
            None => Exec::serial(),
        }
    }

    fn label(&self) -> &'static str {
        if self.pool.is_some() {
            "4T"
        } else {
            "1T"
        }
    }
}

/// Nanoseconds per element of one sample: `reps` copies, each timed on its own
/// so the token's set-up stays outside the clock.
fn timed<T: Elem>(src: &StridedView<T>, dst: &mut StridedViewMut<T>, at: Placement<'_>) -> f64 {
    let n: usize = src.dims().iter().product();
    let reps = (SAMPLE_ELEMENTS / n.max(1)).clamp(1, 32);
    let mut elapsed = 0.0;
    for _ in 0..reps {
        let exec = at.token();
        let start = Instant::now();
        copy_into(black_box(dst), black_box(src), exec).unwrap();
        elapsed += start.elapsed().as_secs_f64() * 1e9;
    }
    elapsed / (reps * n) as f64
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// The destination fingerprint, read once outside the timed region: element
/// count, sum, and a position-weighted sum.
fn digest<T: Elem>(dst: &StridedView<T>) -> (usize, u128, u128) {
    let data = dst.data();
    let sum = data
        .iter()
        .fold(0u128, |acc, &v| acc.wrapping_add(v.bits()));
    let weighted = data.iter().enumerate().fold(0u128, |acc, (i, &v)| {
        acc.wrapping_add(v.bits().wrapping_mul(i as u128 + 1))
    });
    (data.len(), sum, weighted)
}

/// Times a `2^d - 1`, `2^d`, `2^d + 1` triple of one element type, permuted by
/// `perm`, and prints its three rows.
fn triple<T: Elem>(
    shape: impl Fn(usize) -> Vec<usize>,
    sides: [usize; 3],
    perm: &[usize],
    at: Placement<'_>,
    only: Option<&str>,
) {
    if let Some(only) = only {
        if !format!("{} {}", at.label(), T::name()).contains(only) {
            return;
        }
    }
    // Build every arm of the triple up front so the timed pass can rotate. The
    // source buffers come first: each permuted view borrows one of them.
    let dims_list: Vec<Vec<usize>> = sides.iter().map(|&side| shape(side)).collect();
    let src_bufs: Vec<Vec<T>> = dims_list
        .iter()
        .map(|dims| {
            let n: usize = dims.iter().product();
            (0..n).map(T::make).collect()
        })
        .collect();
    let mut dst_arrays: Vec<StridedArray<T>> = dims_list
        .iter()
        .map(|dims| StridedArray::<T>::col_major(&permuted_dims(dims, perm)))
        .collect();
    let srcs: Vec<StridedView<T>> = src_bufs
        .iter()
        .zip(&dims_list)
        .map(|(buf, dims)| {
            let source_strides = strided_view::col_major_strides(dims);
            let strides: Vec<isize> = perm.iter().map(|&p| source_strides[p]).collect();
            StridedView::new(buf, &permuted_dims(dims, perm), &strides, 0).unwrap()
        })
        .collect();
    let mut arms: Vec<(usize, &StridedView<T>, &mut StridedArray<T>)> = sides
        .iter()
        .zip(&srcs)
        .zip(dst_arrays.iter_mut())
        .map(|((&side, src), dst)| (side, src, dst))
        .collect();
    for (_, src, dst) in arms.iter_mut() {
        for _ in 0..WARMUP {
            timed(src, &mut dst.view_mut(), at);
        }
    }
    let mut samples: Vec<[f64; 3]> = Vec::with_capacity(REPS);
    for rep in 0..REPS {
        let mut row = [0.0f64; 3];
        for slot in 0..3 {
            let i = (rep + slot) % 3;
            let (_, src, dst) = &mut arms[i];
            row[i] = timed(src, &mut dst.view_mut(), at);
        }
        samples.push(row);
    }
    let ratio = {
        let mut per_sample: Vec<f64> = samples
            .iter()
            .map(|row| row[1] / ((row[0] + row[2]) / 2.0))
            .collect();
        median(&mut per_sample)
    };
    for (i, (side, src, dst)) in arms.iter_mut().enumerate() {
        let fingerprint = digest(&dst.view());
        let mut values: Vec<f64> = samples.iter().map(|row| row[i]).collect();
        let med = median(&mut values);
        let (min, max) = (values[0], values[values.len() - 1]);
        println!(
            "{:<2} {:<4} {:<14} {:<11} {:>4} {:>4} {:>8.3} {:>8.3} {:>8.3} {:>+6.1}% {:>7} {:>9} {:>10} {:>20}",
            at.label(),
            T::name(),
            format!("{:?}", src.dims()),
            format!("{perm:?}"),
            side,
            if i == 1 { "2^d" } else { "" },
            med,
            min,
            max,
            (max - min) / med * 100.0,
            if i == 1 {
                format!("{ratio:.2}x")
            } else {
                "-".to_string()
            },
            fingerprint.0,
            format!("{:x}", fingerprint.1),
            format!("{:x}", fingerprint.2),
        );
    }
}

fn main() {
    let only = std::env::var("L1_SETS_466_ONLY").ok();
    let only = only.as_deref();
    let pool = Pool::builder().threads(THREADS).build().unwrap();

    println!(
        "{:<2} {:<4} {:<14} {:<11} {:>4} {:>4} {:>8} {:>8} {:>8} {:>7} {:>7} {:>9} {:>10} {:>20}",
        "thr",
        "elem",
        "dims",
        "perm",
        "side",
        "",
        "median",
        "min",
        "max",
        "spread",
        "2^d/nb",
        "elements",
        "digest",
        "digest-weighted"
    );
    for at in [Placement { pool: None }, Placement { pool: Some(&pool) }] {
        // The floor of the reported thread count: `Exec::budget`, asserted here
        // so a clamped or ignored request cannot be reported as a pinned one.
        assert_eq!(
            at.token().budget().get(),
            at.pool.map_or(1, |_| THREADS),
            "the token must use the thread count this table reports"
        );
        // 2D `[32, n]` and `[64, n]` transposed. The middle cell is the
        // aliasing one: the destination's strided axis is `n` elements wide.
        for d in [10u32, 11] {
            let n = 1usize << d;
            for m in [32usize, 64] {
                triple::<f64>(|n| vec![m, n], [n - 1, n, n + 1], &[1, 0], at, only);
                triple::<Complex64>(|n| vec![m, n], [n - 1, n, n + 1], &[1, 0], at, only);
            }
        }
        // Square `[s, s]` transposed: the issue's 2D table. Here both operands
        // have a strided axis, so the source aliases as well as the destination.
        for sides in [[255usize, 256, 257], [1023, 1024, 1025]] {
            let shape = |s: usize| vec![s, s];
            triple::<f64>(shape, sides, &[1, 0], at, only);
            triple::<Complex64>(shape, sides, &[1, 0], at, only);
        }
        // 3D `[s, s, s]` permuted by `[1, 2, 0]`, the issue's table: the
        // destination axis is `s * s` elements wide. 96 is the non-power-of-two
        // case, 96 * 96 * 8 bytes being a multiple of 16 KiB.
        for sides in [
            [31usize, 32, 33],
            [63, 64, 65],
            [95, 96, 97],
            [127, 128, 129],
        ] {
            let shape = |s: usize| vec![s, s, s];
            triple::<f64>(shape, sides, &[1, 2, 0], at, only);
            triple::<Complex64>(shape, sides, &[1, 2, 0], at, only);
        }
        // The miss case at a fixed footprint: `[64, 64, 64]` permuted by
        // `[2, 1, 0]` and `[2, 0, 1]` — the same extents, other destinations.
        for perm in [[2usize, 1, 0], [2, 0, 1]] {
            let shape = |s: usize| vec![s, s, s];
            triple::<f64>(shape, [63, 64, 65], &perm, at, only);
            triple::<Complex64>(shape, [63, 64, 65], &perm, at, only);
        }
    }
}
