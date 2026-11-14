//! Paired plain/conjugating permutation rows for tensor4all/strided-rs#467.
//!
//! `harness = false`: a few-second ratio table, not a criterion suite. Each row
//! runs [`copy_into`] and [`copy_conj_into`] on the same `Complex<f64>` layout
//! and reports ns per element plus two ratios. The 2D extents come in `2^d` and
//! `2^d ± 1` pairs, so the power-of-two shapes and their remainder-carrying
//! neighbours are both visible.
//!
//! The ns/e columns are the minimum over [`REPS`] samples. The ratio columns are
//! the **median of the per-sample ratios**: each sample times the stores back to
//! back in a rotating order, so background load lands on all of them in the same
//! sample and cancels in that sample's ratio, instead of landing on one
//! independently selected minimum.
//!
//! ```text
//! cargo bench -j 4 --features parallel -p strided-perm --bench conj_copy
//! CONJ_COPY_GATE=1 cargo bench -j 4 --features parallel -p strided-perm --bench conj_copy
//! ```
//!
//! Without `CONJ_COPY_GATE` the table is printed and the process succeeds; with
//! it, a row whose median conj/plain ratio exceeds [`GATE_RATIO`] exits non-zero.
//! The threaded columns are reported, not gated: a pool's dispatch cost on a busy
//! host is not what this table is for.

use std::hint::black_box;
use std::time::Instant;

use num_complex::Complex64;
use strided_perm::{copy_conj_into, copy_into};
use strided_view::{StridedArray, StridedView, StridedViewMut};
use threadteam::{Exec, Pool};

/// Conj/plain ratio a gated run rejects. Deliberately loose: the measured spread
/// on this harness is 0.9-1.7 (the conjugation costs one floating-point sign
/// flip per stored element, which shows up where the plain store is closest to
/// raw memory bandwidth), so the gate catches a store that falls off the plain
/// path's kernel entirely rather than a few percent.
const GATE_RATIO: f64 = 2.0;
/// Timed samples per row.
const REPS: usize = 21;
/// Untimed samples before the first measured one.
const WARMUP: usize = 2;
/// Threads in the pooled token.
const THREADS: usize = 4;

/// One measured layout: source dims and the permutation to apply.
struct Case {
    label: &'static str,
    src_dims: Vec<usize>,
    perm: Vec<usize>,
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for d in [8u32, 10, 11, 12] {
        let n = 1usize << d;
        for side in [n.saturating_sub(1), n, n + 1] {
            cases.push(Case {
                label: Box::leak(format!("2d {side}x{side} [1,0]").into_boxed_str()),
                src_dims: vec![side, side],
                perm: vec![1, 0],
            });
        }
        // Same layout on both sides: the unit-stride store path, no 2D tile loop.
        cases.push(Case {
            label: Box::leak(format!("contig {n}x{n} [0,1]").into_boxed_str()),
            src_dims: vec![n, n],
            perm: vec![0, 1],
        });
    }
    for (dims, perm) in [
        (vec![64usize, 64, 64], vec![2usize, 0, 1]),
        (vec![128, 128, 128], vec![2, 0, 1]),
        (vec![32, 32, 32, 32], vec![3, 1, 0, 2]),
        (vec![100, 2, 100], vec![2, 0, 1]),
    ] {
        cases.push(Case {
            label: Box::leak(format!("{dims:?} perm {perm:?}").into_boxed_str()),
            src_dims: dims,
            perm,
        });
    }
    cases
}

fn permuted_dims(dims: &[usize], perm: &[usize]) -> Vec<usize> {
    perm.iter().map(|&p| dims[p]).collect()
}

/// The store a sample times.
#[derive(Clone, Copy)]
enum Store {
    /// [`copy_into`] on the calling thread.
    Plain,
    /// [`copy_conj_into`] on the calling thread.
    Conj,
    /// [`copy_conj_into`] on `THREADS`.
    ConjThreaded,
}

/// One timed run of one store, in nanoseconds per element.
fn run(
    src: &StridedView<Complex64>,
    dst: &mut StridedViewMut<Complex64>,
    store: Store,
    pool: &Pool,
) -> f64 {
    let elements: usize = src.dims().iter().product();
    // Minted before the clock: a token's own set-up is not part of the copy.
    let threaded = Exec::new(pool, THREADS).unwrap();
    let start = Instant::now();
    match store {
        Store::Plain => copy_into(black_box(dst), black_box(src), Exec::serial()).unwrap(),
        Store::Conj => copy_conj_into(black_box(dst), black_box(src), Exec::serial()).unwrap(),
        Store::ConjThreaded => copy_conj_into(black_box(dst), black_box(src), threaded).unwrap(),
    }
    start.elapsed().as_secs_f64() * 1e9 / elements as f64
}

fn min(samples: &[f64]) -> f64 {
    samples.iter().fold(f64::INFINITY, |best, &x| best.min(x))
}

/// Median of a sample set; the caller passes a non-empty slice.
fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

/// Per-row measurements: min ns/e of each store and the two median ratios.
struct Row {
    plain: f64,
    conj: f64,
    conj_threaded: f64,
    conj_over_plain: f64,
    threaded_over_serial: f64,
}

fn measure(src: &StridedView<Complex64>, dst: &mut StridedViewMut<Complex64>, pool: &Pool) -> Row {
    const STORES: [Store; 3] = [Store::Plain, Store::Conj, Store::ConjThreaded];
    for _ in 0..WARMUP {
        for store in STORES {
            run(src, dst, store, pool);
        }
    }
    let mut ns = [[0.0f64; 3]; REPS];
    for (rep, row) in ns.iter_mut().enumerate() {
        // Rotate the order: no store always sees the same neighbour in a sample.
        for slot in 0..STORES.len() {
            let i = (rep + slot) % STORES.len();
            row[i] = run(src, dst, STORES[i], pool);
        }
    }
    let column = |i: usize| ns.iter().map(|row| row[i]).collect::<Vec<_>>();
    let (plain, conj, conj_threaded) = (column(0), column(1), column(2));
    let per_sample = |num: &[f64], den: &[f64]| -> f64 {
        median(
            &num.iter()
                .zip(den)
                .map(|(&n, &d)| n / d)
                .collect::<Vec<_>>(),
        )
    };
    Row {
        plain: min(&plain),
        conj: min(&conj),
        conj_threaded: min(&conj_threaded),
        conj_over_plain: per_sample(&conj, &plain),
        threaded_over_serial: per_sample(&conj_threaded, &conj),
    }
}

fn main() {
    let gated = std::env::var("CONJ_COPY_GATE").is_ok_and(|v| v != "0");
    let mut failures = Vec::new();
    let pool = Pool::builder().threads(THREADS).build().unwrap();

    println!(
        "{:<28} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "case", "plain", "conj", "conj4T", "conj/p", "4T/1T", "elements"
    );
    for case in cases() {
        let src_data = StridedArray::<Complex64>::from_fn_col_major(&case.src_dims, |idx| {
            Complex64::new(idx[0] as f64, idx[1] as f64)
        });
        let src = src_data.view().permute(&case.perm).unwrap();
        let mut dst =
            StridedArray::<Complex64>::col_major(&permuted_dims(&case.src_dims, &case.perm));

        let row = measure(&src, &mut dst.view_mut(), &pool);
        println!(
            "{:<28} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9}",
            case.label,
            row.plain,
            row.conj,
            row.conj_threaded,
            row.conj_over_plain,
            row.threaded_over_serial,
            src.dims().iter().product::<usize>()
        );
        if row.conj_over_plain > GATE_RATIO {
            failures.push((case.label, row.conj_over_plain));
        }
    }

    if !failures.is_empty() {
        for (label, ratio) in &failures {
            println!("over {GATE_RATIO:.2}x: {label} at {ratio:.3}");
        }
        if gated {
            std::process::exit(1);
        }
        println!("(not gated; set CONJ_COPY_GATE=1 to fail on this)");
    }
}
