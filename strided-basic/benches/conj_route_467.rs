//! Route evidence for tensor4all/strided-rs#467: the element map or the
//! permutation engine's conjugating store for `strided_basic::copy_conj_into`.
//!
//! `harness = false`: a few-second ratio table, not a criterion suite. Every row
//! runs three stores on the same `Complex<f64>` layout, in a rotating order
//! inside each sample:
//!
//! - `map` is `copy_into(dst, &src.conj(), ..)`, the route `copy_conj_into` took
//!   before the gate;
//! - `engine` is `strided_perm::copy_conj_into`, the permutation engine's
//!   conjugating store (HPTT's `conjA`, landed in #478);
//! - `prod` is `strided_basic::copy_conj_into`, which the gate decides.
//!
//! ns/e columns are minima over [`REPS`] samples; the ratio columns are the
//! **median of the per-sample ratios**, so background load lands on all three
//! stores in the same sample and cancels.
//!
//! ```text
//! taskset -c 0-7,16-23 cargo bench -j 8 -p strided-basic --bench conj_route_467
//! CONJ_ROUTE_GATE=1 taskset -c 0-7,16-23 cargo bench -j 8 -p strided-basic --bench conj_route_467
//! ```
//!
//! Without `CONJ_ROUTE_GATE` the table is printed and the process succeeds; with
//! it, a row whose two routes differ by more than the noise band of their own
//! paired samples must have `prod` inside the noise band of the faster one, or
//! the process exits non-zero. The band is the formula the rest of this crate's
//! benches use (`benches/support/mod.rs`): `3 * max(1.4826 * MAD, 0.15 * median)`
//! over the per-sample ratios, so a quiet run gets a tighter bound and a noisy
//! run widens its own instead of turning the host's load into a red gate. The
//! map column varies by 10-25 % between runs on this host, which is why a fixed
//! few-percent bar would not be a routing statement.
//!
//! The 2D crosses, the 3D size scan and the contiguous rows are all in the set:
//! the first locates the power-of-two/aliasing class (#466), the second the size
//! crossing, the third the clause that keeps contiguous copies on the map.

use std::hint::black_box;
use std::time::Instant;

use num_complex::Complex64;
use strided_basic::{copy_conj_into, copy_into, Exec, StridedView, StridedViewMut};
use strided_view::StridedArray;

/// `MAD` scaled to a standard deviation for a normal distribution.
const MAD_TO_SIGMA: f64 = 1.4826;
/// Floor on the band, as a fraction of the median (`benches/support/mod.rs`).
const GATE_FLOOR_FRACTION: f64 = 0.15;
/// Widths of the band, in scaled MADs.
const SIGMA_MULTIPLE: f64 = 3.0;
/// A row where one route is ahead by this factor must take it within
/// [`STRONG_SLACK`], whatever the noise band says: the check that the gate is
/// neither inverted nor dropped.
const STRONG_ROUTE: f64 = 1.5;
/// How far `prod` may sit from the faster route on a [`STRONG_ROUTE`] row.
const STRONG_SLACK: f64 = 1.10;
/// Timed samples per row.
const REPS: usize = 21;
/// Untimed samples before the first measured one.
const WARMUP: usize = 2;

/// One measured layout: source dims and the permutation to apply.
struct Case {
    label: &'static str,
    src_dims: Vec<usize>,
    perm: Vec<usize>,
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let mut push = |label: String, src_dims: Vec<usize>, perm: Vec<usize>| {
        cases.push(Case {
            label: Box::leak(label.into_boxed_str()),
            src_dims,
            perm,
        })
    };
    // The issue's 2D rows and their 2^d neighbours, kept around the 4 KiB
    // destination-row boundary (254^2 is 4064 B, 256^2 is 4096 B) that a removed
    // routing clause used to decide, so a regression on either side is visible.
    for side in [128usize, 254, 255, 256, 257, 512, 1023, 1024, 1025, 2048] {
        push(
            format!("2d {side}x{side} [1,0]"),
            vec![side, side],
            vec![1, 0],
        );
    }
    // The rest of the issue's rows.
    for (dims, perm) in [
        (vec![64usize, 64, 64], vec![2usize, 0, 1]),
        (vec![65, 65, 65], vec![2, 0, 1]),
        (vec![128, 128, 128], vec![2, 0, 1]),
        (vec![32, 32, 32, 32], vec![3, 1, 0, 2]),
        (vec![100, 2, 100], vec![2, 0, 1]),
    ] {
        push(format!("{dims:?} perm {perm:?}"), dims, perm);
    }
    // The size crossing between 64^3 (engine) and 128^3 (map), probed at
    // 2^21 elements (128^3) from both sides.
    for d in [32usize, 48, 64, 80, 96, 100, 108, 112, 116, 120, 124] {
        push(
            format!("{}^3 perm [2,0,1]", d),
            vec![d, d, d],
            vec![2, 0, 1],
        );
    }
    // Contiguous copies, which the gate must leave on the map.
    for side in [64usize, 1024] {
        push(
            format!("contig {side}x{side} [0,1]"),
            vec![side, side],
            vec![0, 1],
        );
    }
    cases
}

fn permuted_dims(dims: &[usize], perm: &[usize]) -> Vec<usize> {
    perm.iter().map(|&p| dims[p]).collect()
}

/// The store a sample times.
#[derive(Clone, Copy)]
enum Store {
    /// The element map through `copy_into` on a conjugating view.
    Map,
    /// `strided_perm::copy_conj_into` directly.
    Engine,
    /// `strided_basic::copy_conj_into`, i.e. the gate's own decision.
    Prod,
}

/// One timed run of one store, in nanoseconds per element.
fn run(src: &StridedView<Complex64>, dst: &mut StridedViewMut<Complex64>, store: Store) -> f64 {
    let elements: usize = src.dims().iter().product();
    let start = Instant::now();
    match store {
        Store::Map => {
            let src_conj = src.conj();
            copy_into(black_box(dst), &src_conj, Exec::serial()).unwrap()
        }
        Store::Engine => {
            strided_perm::copy_conj_into(black_box(dst), black_box(src), Exec::serial()).unwrap()
        }
        Store::Prod => copy_conj_into(black_box(dst), black_box(src), Exec::serial()).unwrap(),
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

/// Median absolute deviation, `MAD` scaled to a standard deviation.
fn scaled_mad(samples: &[f64], center: f64) -> f64 {
    let deviations: Vec<f64> = samples.iter().map(|x| (x - center).abs()).collect();
    MAD_TO_SIGMA * median(&deviations)
}

/// Half-width of a ratio's noise band: the bound `benches/support/mod.rs` uses
/// for a parity row, so this table widens its own gate on a busy host.
fn band(samples: &[f64]) -> f64 {
    let center = median(samples);
    SIGMA_MULTIPLE * scaled_mad(samples, center).max(GATE_FLOOR_FRACTION * center.abs())
}

/// Per-row measurements: min ns/e of each store and the two median ratios.
struct Row {
    map: f64,
    engine: f64,
    prod: f64,
    map_over_engine: f64,
    map_over_engine_band: f64,
    prod_over_best: f64,
    prod_over_best_band: f64,
}

fn measure(src: &StridedView<Complex64>, dst: &mut StridedViewMut<Complex64>) -> Row {
    const STORES: [Store; 3] = [Store::Map, Store::Engine, Store::Prod];
    for _ in 0..WARMUP {
        for store in STORES {
            run(src, dst, store);
        }
    }
    let mut ns = [[0.0f64; 3]; REPS];
    for (rep, row) in ns.iter_mut().enumerate() {
        // Rotate the order: no store always sees the same neighbour in a sample.
        for slot in 0..STORES.len() {
            let i = (rep + slot) % STORES.len();
            row[i] = run(src, dst, STORES[i]);
        }
    }
    let column = |i: usize| ns.iter().map(|row| row[i]).collect::<Vec<_>>();
    let (map, engine, prod) = (column(0), column(1), column(2));
    let per_sample = |num: &[f64], den: &[f64]| -> Vec<f64> {
        num.iter().zip(den).map(|(&n, &d)| n / d).collect()
    };
    let best: Vec<f64> = map.iter().zip(&engine).map(|(&m, &e)| m.min(e)).collect();
    let map_over_engine = per_sample(&map, &engine);
    let prod_over_best = per_sample(&prod, &best);
    Row {
        map: min(&map),
        engine: min(&engine),
        prod: min(&prod),
        map_over_engine: median(&map_over_engine),
        map_over_engine_band: band(&map_over_engine),
        prod_over_best: median(&prod_over_best),
        prod_over_best_band: band(&prod_over_best),
    }
}

fn main() {
    let gated = std::env::var("CONJ_ROUTE_GATE").is_ok_and(|v| v != "0");
    let mut failures = Vec::new();

    println!(
        "{:<26} {:>9} {:>9} {:>9} {:>9} {:>8} {:>9} {:>8} {:>10}",
        "case", "map", "engine", "prod", "map/eng", "+-", "prod/best", "+-", "elements"
    );
    for case in cases() {
        let src_data = StridedArray::<Complex64>::from_fn_col_major(&case.src_dims, |idx| {
            Complex64::new(idx[0] as f64, idx[1] as f64)
        });
        let src = src_data.view().permute(&case.perm).unwrap();
        let mut dst =
            StridedArray::<Complex64>::col_major(&permuted_dims(&case.src_dims, &case.perm));

        let row = measure(&src, &mut dst.view_mut());
        println!(
            "{:<26} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>8.3} {:>9.3} {:>8.3} {:>10}",
            case.label,
            row.map,
            row.engine,
            row.prod,
            row.map_over_engine,
            row.map_over_engine_band,
            row.prod_over_best,
            row.prod_over_best_band,
            src.dims().iter().product::<usize>()
        );
        // A row is decided when the two routes differ by more than their own
        // paired-sample noise, or by [`STRONG_ROUTE`] either way. The factor is
        // symmetric: 0.59 and 1.70 are the same 1.70x.
        let factor = row.map_over_engine.max(1.0 / row.map_over_engine);
        let strong = factor >= STRONG_ROUTE;
        let spread = factor - 1.0;
        let decided = strong || spread > row.map_over_engine_band;
        let slack = if strong {
            STRONG_SLACK
        } else {
            1.0 + row.prod_over_best_band
        };
        if decided && row.prod_over_best > slack {
            failures.push((
                case.label,
                row.map_over_engine,
                row.prod_over_best,
                row.prod_over_best_band,
            ));
        }
    }

    if !failures.is_empty() {
        for (label, map_over_engine, prod_over_best, band) in &failures {
            println!(
                "prod is not on the faster route: {label} at map/eng {map_over_engine:.3}, \
                 prod/best {prod_over_best:.3} (band {band:.3})"
            );
        }
        if gated {
            std::process::exit(1);
        }
        println!("(not gated; set CONJ_ROUTE_GATE=1 to fail on this)");
    }
}
