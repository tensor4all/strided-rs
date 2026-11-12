//! Per-row timings of the plan-free `replay_*_raw` entries (tensor4all/strided-rs#469).
//!
//! `harness = false`. One run prints, per row, the median of
//! [`support::REPS`] timed samples and the allocations per call. The rows are
//! meant for a paired comparison of two builds of this same file: an external
//! driver runs the two binaries in ABBA order and forms the per-pair ratios, so
//! this binary gates nothing itself.
//!
//! Rows (f64, release profile, default L1 budget):
//! - `R1`: small replays at ranks 1-8, totals 64 and 4096, with a contiguous
//!   source, a reversed (permuted) source, and a source whose inner axis runs
//!   backwards from a nonzero base (the `-1` source-stride run). Serial. The
//!   routes are mixed: every row is untiled except the reversed-source rows of
//!   total 4096 at ranks 3-8, which the cache-block model tiles on both builds
//!   (checked through `bench_hooks::pair_blocks`); their ids say `perm-tiled`.
//! - `R2`: inline tiled replays, a `1024x1024` transpose and the rank-3
//!   `[4096, 17, 16]` layout, serial and on a pooled four-slot token.
//! - `R3`: a rank-3 contiguous replay of 2^22 elements on the pooled token.
//! - `R4`: wide replays, serial: `[2; 9]` reversed into column-major, `[2; 12]`
//!   reversed, rank 200 with the nine moving axes of `[2; 9]`, and `4^9`
//!   reversed (tiled).
//!
//! ```text
//! cargo bench --profile release -p strided-basic --features parallel --bench replay_raw
//! ```

use std::hint::black_box;

use strided_basic::execution::{parallel_width, KernelClass};
use strided_basic::{replay_copy_raw, Exec, Pool, RawStridedMut, RawStridedRef};

#[allow(dead_code)] // the shared support module also serves the overhead benches
#[path = "support/mod.rs"]
mod support;

#[global_allocator]
static GLOBAL: support::CountingAllocator = support::CountingAllocator;

/// Pooled rows run on this many slots.
const POOL_THREADS: usize = 4;
/// Elements touched per timed sample, so that every sample lasts about the
/// same time whatever the row's size.
const ELEMENTS_PER_SAMPLE: usize = 1 << 21;

fn col_major(dims: &[usize]) -> Vec<isize> {
    let mut stride = 1isize;
    dims.iter()
        .map(|&dim| {
            let s = stride;
            stride *= dim as isize;
            s
        })
        .collect()
}

/// Strides of a column-major buffer of the reversed dims, read in the given
/// axis order: the source of a full axis reversal.
fn reversed(dims: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; dims.len()];
    let mut stride = 1isize;
    for axis in (0..dims.len()).rev() {
        strides[axis] = stride;
        stride *= dims[axis] as isize;
    }
    strides
}

/// Elements of a buffer that holds every offset reachable from `offset`.
fn span(dims: &[usize], strides: &[isize], offset: isize) -> usize {
    let (mut lo, mut hi) = (offset, offset);
    for (&dim, &stride) in dims.iter().zip(strides) {
        let reach = (dim as isize - 1) * stride;
        if reach < 0 {
            lo += reach;
        } else {
            hi += reach;
        }
    }
    assert!(lo >= 0, "base offset below the lowest reachable slot");
    (hi + 1) as usize
}

struct Row {
    id: String,
    dims: Vec<usize>,
    dst: Vec<isize>,
    src: Vec<isize>,
    src_offset: isize,
    pooled: bool,
}

impl Row {
    fn new(id: impl Into<String>, dims: &[usize], dst: Vec<isize>, src: Vec<isize>) -> Self {
        Self {
            id: id.into(),
            dims: dims.to_vec(),
            dst,
            src,
            src_offset: 0,
            pooled: false,
        }
    }

    fn pooled(mut self) -> Self {
        self.pooled = true;
        self
    }
}

/// `total` split over `rank` axes as evenly as powers of two allow; ranks above
/// `log2(total)` get extent-1 axes.
fn shape(rank: usize, total: usize) -> Vec<usize> {
    let bits = total.trailing_zeros() as usize;
    (0..rank)
        .map(|axis| 1usize << (bits / rank + usize::from(axis < bits % rank)))
        .collect()
}

fn rows() -> Vec<Row> {
    let mut rows = Vec::new();
    for total in [64usize, 4096] {
        for rank in 1..=8usize {
            let dims = shape(rank, total);
            let dst = col_major(&dims);
            rows.push(Row::new(
                format!("R1 contig r{rank} n{total}"),
                &dims,
                dst.clone(),
                dst.clone(),
            ));
            rows.push(Row::new(
                if total == 4096 && rank >= 3 {
                    format!("R1 perm-tiled r{rank} n{total}")
                } else {
                    format!("R1 perm r{rank} n{total}")
                },
                &dims,
                dst.clone(),
                reversed(&dims),
            ));
            // The inner axis runs backwards from its last slot.
            let mut src = dst.clone();
            src[0] = -1;
            let mut row = Row::new(format!("R1 neg r{rank} n{total}"), &dims, dst, src);
            row.src_offset = dims[0] as isize - 1;
            rows.push(row);
        }
    }
    let square = [1024usize, 1024];
    rows.push(Row::new(
        "R2 transpose 1024^2 serial",
        &square,
        vec![1, 1024],
        vec![1024, 1],
    ));
    rows.push(
        Row::new(
            "R2 transpose 1024^2 pooled",
            &square,
            vec![1, 1024],
            vec![1024, 1],
        )
        .pooled(),
    );
    let deep = [4096usize, 17, 16];
    rows.push(Row::new(
        "R2 [4096,17,16] serial",
        &deep,
        vec![1, 65_536, 4096],
        vec![1, 4096, 4096],
    ));
    rows.push(
        Row::new(
            "R2 [4096,17,16] pooled",
            &deep,
            vec![1, 65_536, 4096],
            vec![1, 4096, 4096],
        )
        .pooled(),
    );
    let big = [256usize, 128, 128];
    rows.push(
        Row::new(
            "R3 contig 2^22 pooled",
            &big,
            col_major(&big),
            col_major(&big),
        )
        .pooled(),
    );
    for (label, dims) in [
        ("R4 [2;9] reversed", vec![2usize; 9]),
        ("R4 [2;12] reversed", vec![2; 12]),
    ] {
        rows.push(Row::new(label, &dims, col_major(&dims), reversed(&dims)));
    }
    // Rank 200: the nine moving axes of `[2; 9]` among 191 extent-1 axes.
    let nine = [2usize; 9];
    let (nine_dst, nine_src) = (col_major(&nine), reversed(&nine));
    let mut padded = (Vec::new(), Vec::new(), Vec::new());
    for axis in 0..200usize {
        if axis % 22 == 0 && axis / 22 < 9 {
            padded.0.push(2usize);
            padded.1.push(nine_dst[axis / 22]);
            padded.2.push(nine_src[axis / 22]);
        } else {
            padded.0.push(1);
            padded.1.push(axis as isize);
            padded.2.push(-(axis as isize));
        }
    }
    rows.push(Row::new(
        "R4 rank200 9 moving",
        &padded.0,
        padded.1,
        padded.2,
    ));
    let wide = [4usize; 9];
    rows.push(Row::new(
        "R4 4^9 reversed tiled",
        &wide,
        col_major(&wide),
        reversed(&wide),
    ));
    rows
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() {
    let pool = Pool::builder().threads(POOL_THREADS).build().unwrap();
    println!("row\tns_per_call\tallocs_per_call");
    for row in rows() {
        let total: usize = row.dims.iter().product();
        let src_len = span(&row.dims, &row.src, row.src_offset);
        let dst_len = span(&row.dims, &row.dst, 0);
        let src: Vec<f64> = (0..src_len).map(|v| v as f64).collect();
        let mut out = vec![0.0f64; dst_len];
        let source = RawStridedRef::new(&src, &row.dims, &row.src, row.src_offset).unwrap();
        let mut dest = RawStridedMut::new(&mut out, &row.dims, &row.dst, 0).unwrap();
        let exec = || {
            if row.pooled {
                Exec::new(&pool, POOL_THREADS).unwrap()
            } else {
                Exec::serial()
            }
        };
        if row.pooled {
            // Setup, outside the timed region: the token must fan out.
            let width = parallel_width(&exec(), KernelClass::Copy, total).get();
            assert!(width >= 2, "{}: pooled row runs at width {width}", row.id);
        }
        let iterations = (ELEMENTS_PER_SAMPLE / total).max(1);
        let mut call = || {
            replay_copy_raw(black_box(&mut dest), black_box(&source), exec()).unwrap();
        };
        for _ in 0..iterations.min(64) {
            call();
        }
        let samples = (0..support::REPS)
            .map(|_| support::time_once(iterations, &mut call))
            .collect();
        let allocations = if row.pooled {
            f64::NAN
        } else {
            support::allocations_per_op(&mut call)
        };
        println!("{}\t{:.2}\t{}", row.id, median(samples), allocations);
    }
}
