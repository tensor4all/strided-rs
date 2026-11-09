//! Whole-plan parallel `ConcatenatePlan` against the per-segment `CopyPlan`
//! loop it replaces, for strided-rs#427.
//!
//! `harness = false`: a report, not a gate. Every case concatenates f64
//! `[rows, cols]` segments along axis 1 into a column-major destination. A
//! segment is either transposed (source strides `[cols, 1]`, the layout the
//! cache tiles exist for) or contiguous (`[1, rows]`). Per repetition the
//! three paths run back to back on the same buffers:
//!
//! - `whole-plan`: `ConcatenatePlan::execute` on a token of `THREADS` slots;
//! - `per-segment`: `CopyPlan::execute` of each segment in turn on that token,
//!   i.e. the serial per-segment route of `ConcatenatePlan::execute`, where
//!   each segment may still split across the token itself;
//! - `serial`: `ConcatenatePlan::execute` on `Exec::serial()`.
//!
//! The three run in one loop and their order rotates every repetition
//! (whole/per-segment/serial, then per-segment/serial/whole, ...), so a
//! drifting machine and the state a predecessor leaves in the caches and the
//! pool are spread over all three paths instead of always falling on the same
//! one. Plans, inputs and the destination are built once, and every path
//! builds its token and destination views before its timer starts (the
//! per-segment path one view per disjoint segment chunk), so the timed
//! region holds only the replay. Each row prints the median of `REPS` samples
//! (after `WARMUP` unrecorded ones), the bandwidth counting one read and one
//! write per element, the element rate and the 10th to 90th percentile, then
//! `whole/per-segment` of the medians.
//!
//! ```text
//! cargo bench -p strided-basic --features parallel --bench concat_parallel
//! CONCAT_BENCH_THREADS=8 CONCAT_BENCH_REPS=101 cargo bench -p strided-basic --features parallel --bench concat_parallel
//! ```

use std::hint::black_box;
use std::time::Instant;

use strided_basic::{ConcatenatePlan, CopyPlan, Exec, Pool, RawStridedMut, RawStridedRef};

const WARMUP: usize = 3;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// The 10th percentile, median and 90th percentile of `samples` (nearest rank).
fn quantiles(mut samples: Vec<f64>) -> [f64; 3] {
    samples.sort_by(f64::total_cmp);
    let at = |q: f64| samples[((samples.len() - 1) as f64 * q).round() as usize];
    [at(0.1), at(0.5), at(0.9)]
}

/// One concatenation of `transposed.len()` segments of `[rows, cols]`.
fn case(
    name: &str,
    rows: usize,
    cols: usize,
    transposed: &[bool],
    pool: &Pool,
    threads: usize,
    reps: usize,
) {
    let count = transposed.len();
    let dims = [rows, cols];
    let strides: Vec<[isize; 2]> = transposed
        .iter()
        .map(|&t| {
            if t {
                [cols as isize, 1]
            } else {
                [1, rows as isize]
            }
        })
        .collect();
    let data: Vec<Vec<f64>> = (0..count)
        .map(|k| {
            (0..rows * cols)
                .map(|i| (k * rows * cols + i) as f64)
                .collect()
        })
        .collect();
    let dest_dims = [rows, count * cols];
    let dest_strides = [1isize, rows as isize];
    let dims_refs: Vec<&[usize]> = (0..count).map(|_| &dims[..]).collect();
    let strides_refs: Vec<&[isize]> = strides.iter().map(|s| &s[..]).collect();
    let plan = ConcatenatePlan::compile(&dims_refs, &strides_refs, &dest_dims, &dest_strides, 1)
        .expect("a valid concatenation");
    let copies: Vec<CopyPlan> = strides
        .iter()
        .map(|s| CopyPlan::compile(&dims, &dest_strides, s).expect("a valid segment copy"))
        .collect();
    let inputs: Vec<RawStridedRef<'_, f64>> = data
        .iter()
        .zip(&strides)
        .map(|(d, s)| RawStridedRef::new(d, &dims, s, 0).expect("an in-bounds input"))
        .collect();
    let mut dest = vec![0.0f64; rows * count * cols];

    // Time one path (0 whole-plan, 1 per-segment, 2 serial) once.
    let mut time_path = |path: usize| -> f64 {
        let elapsed = match path {
            0 => {
                let exec = Exec::new(pool, threads).expect("an idle pool");
                let mut view =
                    RawStridedMut::new(&mut dest, &dest_dims, &dest_strides, 0).expect("dest");
                let start = Instant::now();
                plan.execute(&mut view, &inputs, exec).expect("whole-plan");
                start.elapsed()
            }
            1 => {
                let mut exec = Exec::new(pool, threads).expect("an idle pool");
                // Segment `k` occupies the destination elements
                // `k * rows * cols..(k + 1) * rows * cols` (column-major,
                // axis 1), so its view is built on that disjoint chunk at
                // offset 0, before the timer, as the other paths build theirs.
                let mut views: Vec<RawStridedMut<'_, f64>> = dest
                    .chunks_mut(rows * cols)
                    .map(|chunk| {
                        RawStridedMut::new(chunk, &dims, &dest_strides, 0).expect("segment")
                    })
                    .collect();
                let start = Instant::now();
                for ((copy, input), view) in copies.iter().zip(&inputs).zip(&mut views) {
                    copy.execute(view, input, exec.reborrow())
                        .expect("per-segment");
                }
                let elapsed = start.elapsed();
                drop(views);
                elapsed
            }
            _ => {
                let mut view =
                    RawStridedMut::new(&mut dest, &dest_dims, &dest_strides, 0).expect("dest");
                let start = Instant::now();
                plan.execute(&mut view, &inputs, Exec::serial())
                    .expect("serial");
                start.elapsed()
            }
        };
        black_box(&dest);
        elapsed.as_secs_f64()
    };

    let mut samples: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(reps));
    for rep in 0..WARMUP + reps {
        // Rotate the order every repetition, so that each path runs first,
        // second and third equally often (to within one repetition) and none
        // always inherits the cache and pool state of the same predecessor.
        for slot in 0..3 {
            let path = (rep + slot) % 3;
            let time = time_path(path);
            if rep >= WARMUP {
                samples[path].push(time);
            }
        }
    }
    let [whole, per_segment, serial] = samples;
    let elems = (rows * cols * count) as f64;
    let bytes = 2.0 * 8.0 * elems;
    let (whole, per_segment, serial) =
        (quantiles(whole), quantiles(per_segment), quantiles(serial));
    for (label, [p10, time, p90]) in [
        (format!("whole-plan {threads}T"), whole),
        (format!("per-segment {threads}T"), per_segment),
        ("serial 1T".to_string(), serial),
    ] {
        println!(
            "{name:<24} {label:<16} {:>8.3} ms {:>7.1} GB/s {:>7.3} Gelem/s  p10..p90 {:.3}..{:.3} ms",
            time * 1e3,
            bytes / time / 1e9,
            elems / time / 1e9,
            p10 * 1e3,
            p90 * 1e3
        );
    }
    let (whole, per_segment) = (whole[1], per_segment[1]);
    println!("{name:<24} whole/per-segment = {:.3}", whole / per_segment);
}

fn main() {
    let threads = env_usize("CONCAT_BENCH_THREADS", 4);
    let reps = env_usize("CONCAT_BENCH_REPS", 41);
    assert!(threads > 0, "CONCAT_BENCH_THREADS must be at least 1");
    assert!(
        reps > 0,
        "CONCAT_BENCH_REPS must be at least 1: zero reps leaves every sample vector empty, so the median would index an empty slice"
    );
    let pool = Pool::builder().threads(threads).build().expect("a pool");
    println!("threads {threads}, median of {reps} reps after {WARMUP} warm-up");
    case(
        "4x[1024x1024] all T",
        1024,
        1024,
        &[true; 4],
        &pool,
        threads,
        reps,
    );
    case(
        "4x[1024x1024] T,C,T,C",
        1024,
        1024,
        &[true, false, true, false],
        &pool,
        threads,
        reps,
    );
    case(
        "4x[256x256] all T",
        256,
        256,
        &[true; 4],
        &pool,
        threads,
        reps,
    );
    case(
        "8x[512x512] all T",
        512,
        512,
        &[true; 8],
        &pool,
        threads,
        reps,
    );
    case(
        "4x[1024x1024] all C",
        1024,
        1024,
        &[false; 4],
        &pool,
        threads,
        reps,
    );
}
