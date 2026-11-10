//! Evidence for tensor4all/strided-rs#287: the planner's tiles fitted to the
//! L1 sets (`block::fit_l1_sets`) against Strided.jl's model tiles, on a
//! declared set of transposed, permuted and broadcast layouts at one and four
//! threads, `f64` and `f32`.
//!
//! One process times both answers on the same buffers by storing
//! `bench_hooks::L1_SET_BUDGET` between samples: `0` is the model's tiles
//! (current), 6 (`L1_WAYS - L1_WAYS / 4`) the fitted ones (candidate). Each cell runs five
//! rounds; a round times current, candidate and current again (an A/A arm
//! through the same toggle, for the noise floor), in an order that alternates
//! between rounds. Only ratios inside one round are compared; the spread of
//! the absolute medians across rounds is reported beside them.
//!
//! Every arm's output is compared bitwise with the first current arm's before
//! any timing: the tiles must not change a result.
//!
//! ```text
//! taskset -c <cores> cargo bench -p strided-basic \
//!     --features bench-hooks,parallel --bench block_l1_sets_287
//! ```
//!
//! `BENCH_287_THREADS` (default `1,4`), `BENCH_287_FILTER` (a substring of
//! `op layout dims`, for diagnostics) and `BENCH_287_CANDIDATE` (default 6,
//! the per-set budget of the candidate arm) select what runs. Rows go to
//! stdout as tab-separated `CELL` and `ROUND` lines. Nothing here is a gate by
//! itself; the decision rule lives with the experiment's protocol.

use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

use strided_basic::bench_hooks::{pair_blocks, pointwise_blocks, L1_SET_BUDGET};
use strided_basic::{
    copy_into, mul_into, replay_copy_raw, Exec, Pool, RawStridedMut, RawStridedRef, StridedView,
    StridedViewMut,
};

const ROUNDS: usize = 5;
const SAMPLES: usize = 9;
/// The production budget, `L1_WAYS - L1_WAYS / 4`.
const LIVE_LINES_PER_SET: usize = 6;

/// Column-major strides of `dims`.
fn col_major(dims: &[usize]) -> Vec<isize> {
    stored(dims, &(0..dims.len()).collect::<Vec<_>>())
}

/// Strides of `dims` stored with axis `storage[0]` fastest, then `storage[1]`, ...
fn stored(dims: &[usize], storage: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; dims.len()];
    let mut acc = 1isize;
    for &axis in storage {
        strides[axis] = acc;
        acc *= dims[axis] as isize;
    }
    strides
}

/// Elements a view of `dims` and `strides` reaches (`max offset + 1`).
fn span(dims: &[usize], strides: &[isize]) -> usize {
    dims.iter()
        .zip(strides)
        .map(|(&d, &s)| (d - 1) * s.unsigned_abs())
        .sum::<usize>()
        + 1
}

#[derive(Clone)]
struct Cell {
    op: &'static str,
    layout: &'static str,
    dims: Vec<usize>,
    /// Destination first, then the sources.
    strides: Vec<Vec<isize>>,
}

/// The declared layout set.
fn cells() -> Vec<Cell> {
    let mut cells = Vec::new();
    // Power-of-two shapes (the #287 family) and non-power-of-two controls.
    for (m, n) in [
        (2048, 2048),
        (1024, 4096),
        (4096, 1024),
        (2047, 2047),
        (1000, 4100),
    ] {
        let dims = vec![m, n];
        let d = col_major(&dims);
        let t = stored(&dims, &[1, 0]);
        let mul = |layout, a: &Vec<isize>, b: &Vec<isize>| Cell {
            op: "mul_into",
            layout,
            dims: dims.clone(),
            strides: vec![d.clone(), a.clone(), b.clone()],
        };
        cells.push(mul("B transposed", &d, &t));
        cells.push(mul("A,B transposed", &t, &t));
        cells.push(mul("A row-bcast, B transposed", &vec![0, 1], &t));
        cells.push(mul("B row-bcast", &d, &vec![0, 1]));
        cells.push(mul("B col-bcast, A transposed", &t, &vec![1, 0]));
        cells.push(Cell {
            op: "copy_into",
            layout: "transposed",
            dims: dims.clone(),
            strides: vec![d.clone(), t.clone()],
        });
        cells.push(Cell {
            op: "replay_copy_raw",
            layout: "transposed",
            dims: dims.clone(),
            strides: vec![d.clone(), t.clone()],
        });
    }
    for dims in [vec![128, 128, 256], vec![127, 129, 255]] {
        let d = col_major(&dims);
        for (layout, storage) in [
            ("B perm (2,0,1)", [2, 0, 1]),
            ("B perm (1,2,0)", [1, 2, 0]),
            ("B perm (2,1,0)", [2, 1, 0]),
        ] {
            cells.push(Cell {
                op: "mul_into",
                layout,
                dims: dims.clone(),
                strides: vec![d.clone(), d.clone(), stored(&dims, &storage)],
            });
        }
        cells.push(Cell {
            op: "copy_into",
            layout: "perm (2,1,0)",
            dims: dims.clone(),
            strides: vec![d.clone(), stored(&dims, &[2, 1, 0])],
        });
        cells.push(Cell {
            op: "replay_copy_raw",
            layout: "perm (2,1,0)",
            dims: dims.clone(),
            strides: vec![d.clone(), stored(&dims, &[2, 1, 0])],
        });
    }
    cells
}

trait Elem: Copy + PartialEq + Send + Sync + std::ops::Mul<Output = Self> + 'static {
    const NAME: &'static str;
    fn from_index(i: usize) -> Self;
    fn bits(self) -> u64;
}

impl Elem for f64 {
    const NAME: &'static str = "f64";
    fn from_index(i: usize) -> Self {
        (i % 1013) as f64 * 0.37 - 11.0
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

impl Elem for f32 {
    const NAME: &'static str = "f32";
    fn from_index(i: usize) -> Self {
        (i % 1013) as f32 * 0.37 - 11.0
    }
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn run_cell<T: Elem>(cell: &Cell, pool: &Pool, threads: usize, candidate: usize) {
    let dims = &cell.dims;
    let sources: Vec<Vec<T>> = cell.strides[1..]
        .iter()
        .map(|s| (0..span(dims, s)).map(T::from_index).collect())
        .collect();
    let mut out: Vec<T> = vec![T::from_index(0); span(dims, &cell.strides[0])];
    let elem = std::mem::size_of::<T>();

    let refs: Vec<&[isize]> = cell.strides.iter().map(|s| s.as_slice()).collect();
    let sizes = vec![elem; refs.len()];
    let mut blocks = Vec::new();
    for budget in [0, candidate] {
        L1_SET_BUDGET.store(budget, Ordering::Relaxed);
        blocks.push(if cell.op == "replay_copy_raw" {
            let tiles = pair_blocks::<T>(dims, &cell.strides[0], &cell.strides[1]);
            (dims.clone(), tiles.unwrap_or_else(|| dims.clone()))
        } else {
            pointwise_blocks(dims, &refs, &sizes, 0)
        });
    }

    let call = |budget: usize, out: &mut Vec<T>| {
        L1_SET_BUDGET.store(budget, Ordering::Relaxed);
        let exec = Exec::new(pool, threads).unwrap();
        assert_eq!(exec.budget().get(), threads, "thread budget clamped");
        if cell.op == "replay_copy_raw" {
            let mut dst = RawStridedMut::new(&mut out[..], dims, &cell.strides[0], 0).unwrap();
            let src = RawStridedRef::new(&sources[0], dims, &cell.strides[1], 0).unwrap();
            replay_copy_raw(&mut dst, &src, exec).unwrap();
            black_box(out.as_mut_ptr());
            return;
        }
        let mut dst = StridedViewMut::<T>::new(&mut out[..], dims, &cell.strides[0], 0).unwrap();
        let a = StridedView::<T>::new(&sources[0], dims, &cell.strides[1], 0).unwrap();
        if cell.op == "copy_into" {
            copy_into(&mut dst, &a, exec).unwrap();
        } else {
            let b = StridedView::<T>::new(&sources[1], dims, &cell.strides[2], 0).unwrap();
            mul_into(&mut dst, &a, &b, exec).unwrap();
        }
        black_box(out.as_mut_ptr());
    };

    // Checked before timed: both arms write the same bits.
    call(0, &mut out);
    let reference: Vec<u64> = out.iter().map(|x| x.bits()).collect();
    call(candidate, &mut out);
    let identical = out.iter().map(|x| x.bits()).eq(reference.iter().copied());
    assert!(
        identical,
        "{} {}: candidate changed the output",
        cell.op, cell.layout
    );

    let time = |budget: usize, out: &mut Vec<T>| {
        let mut samples: Vec<f64> = (0..SAMPLES)
            .map(|_| {
                let start = Instant::now();
                call(budget, out);
                start.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        median(&mut samples)
    };

    let id = format!(
        "{}\t{}\t{:?}\t{}\t{}T",
        cell.op,
        cell.layout,
        dims,
        T::NAME,
        threads
    );
    // Prime: time-based and the same for every cell.
    let prime = Instant::now();
    while prime.elapsed().as_millis() < 300 {
        call(0, &mut out);
        call(candidate, &mut out);
    }
    let mut ratios = Vec::new();
    let mut aa = Vec::new();
    let mut current_ms = Vec::new();
    for round in 0..ROUNDS {
        // Alternate the order between rounds: C X C', then X C C'.
        let (cur, cand, cur2) = if round % 2 == 0 {
            let c = time(0, &mut out);
            let x = time(candidate, &mut out);
            (c, x, time(0, &mut out))
        } else {
            let x = time(candidate, &mut out);
            let c = time(0, &mut out);
            (c, x, time(0, &mut out))
        };
        // > 1: candidate faster than current in this round.
        ratios.push(cur / cand);
        aa.push(cur / cur2);
        current_ms.push(cur);
        println!("ROUND\t{id}\t{round}\t{cur:.4}\t{cand:.4}\t{cur2:.4}");
    }
    let spread = {
        let mut v = current_ms.clone();
        v.sort_by(f64::total_cmp);
        v[ROUNDS - 1] / v[0] - 1.0
    };
    let mut r = ratios.clone();
    r.sort_by(f64::total_cmp);
    let mut a = aa.clone();
    a.sort_by(f64::total_cmp);
    println!(
        "CELL\t{id}\t{:?}\t{:?}\t{:?}\t{:.4}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}",
        blocks[0].0,
        blocks[0].1,
        blocks[1].1,
        median(&mut current_ms),
        r[ROUNDS / 2],
        r[0],
        r[ROUNDS - 1],
        a[ROUNDS / 2],
        a.iter().map(|x| (x - 1.0).abs()).fold(0.0, f64::max),
        spread
    );
    L1_SET_BUDGET.store(LIVE_LINES_PER_SET, Ordering::Relaxed);
}

fn main() {
    let threads: Vec<usize> = std::env::var("BENCH_287_THREADS")
        .unwrap_or_else(|_| "1,4".into())
        .split(',')
        .map(|t| t.trim().parse().unwrap())
        .collect();
    let candidate: usize = std::env::var("BENCH_287_CANDIDATE")
        .map(|c| c.parse().unwrap())
        .unwrap_or(LIVE_LINES_PER_SET);
    println!(
        "# columns: CELL op layout dims dtype threads fused_dims model_tiles candidate_tiles \
         current_ms ratio_med ratio_min ratio_max aa_med aa_maxdev current_spread"
    );
    println!("# ratio = current / candidate within a round (> 1: candidate faster)");
    for &t in &threads {
        let pool = Pool::builder().threads(t).build().unwrap();
        assert_eq!(pool.width().get(), t, "pool width");
        let filter = std::env::var("BENCH_287_FILTER").unwrap_or_default();
        for cell in cells()
            .into_iter()
            .filter(|c| format!("{} {} {:?}", c.op, c.layout, c.dims).contains(&filter))
        {
            run_cell::<f64>(&cell, &pool, t, candidate);
            run_cell::<f32>(&cell, &pool, t, candidate);
        }
    }
}
