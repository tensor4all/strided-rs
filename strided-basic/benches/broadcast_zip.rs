//! Regression benchmark for the threaded split of a broadcast `zip_map2_into`.
//!
//! A stride-0 broadcast operand on the blocked axis used to leave the whole
//! tensor in one cache block, so `ThreadedMapReduce::run` declined to split and
//! the four-thread row ran in a single partition. The shared block
//! normalization in `threading` opens the split; this bench reports the one and
//! four worker rows so a regression is visible.
//!
//! Every row runs in an explicitly bounded Rayon pool whose width is asserted,
//! so the number on a `SUMMARY` line always belongs to the configuration named
//! there.
//!
//! ```text
//! cargo bench -p strided-basic --features parallel --bench broadcast_zip
//! ```

use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::Instant;

use strided_basic::{
    with_execution_policy, zip_map2_into, ExecutionPolicy, StridedView, StridedViewMut,
};

const WARMUPS: usize = 4;
const SAMPLES: usize = 15;
const THREAD_COUNTS: [usize; 2] = [1, 4];
/// Target work per sample, in elements; `repeats` is derived from it.
const SAMPLE_ELEMENTS: usize = 1 << 20;

/// Repeat count that keeps one sample near a millisecond at this host's
/// sub-nanosecond per-element cost.
fn repeats(elements: usize) -> usize {
    (SAMPLE_ELEMENTS / elements.max(1)).max(1)
}

fn time_ms(repeats: usize, operation: &mut impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..repeats {
        operation();
    }
    start.elapsed().as_secs_f64() * 1e3 / repeats as f64
}

fn median(samples: &[f64]) -> f64 {
    let mut ordered = samples.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered[ordered.len() / 2]
}

fn measure(label: &str, rows: usize, cols: usize, threads: usize) {
    let dims = [rows, cols];
    let out_strides = [1isize, rows as isize];
    // `a` is constant along the blocked axis 0 and contiguous along axis 1.
    let broadcast_strides = [0isize, 1isize];
    let peer_strides = [1isize, rows as isize];
    let len = rows * cols;

    let a: Vec<f64> = (0..cols).map(|c| c as f64 * 0.25 - 1.0).collect();
    let b: Vec<f64> = (0..len).map(|index| (index % 17) as f64 - 8.0).collect();
    let mut out = vec![0.0f64; len];

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        assert_eq!(
            rayon::current_num_threads(),
            threads,
            "the pool must actually hold the thread count the line reports"
        );
        with_execution_policy(
            ExecutionPolicy::Rayon {
                max_threads: NonZeroUsize::new(threads).unwrap(),
            },
            || {
                let repeats = repeats(len);
                let mut run = || {
                    let a_view = StridedView::<f64>::new(&a, &dims, &broadcast_strides, 0).unwrap();
                    let b_view = StridedView::<f64>::new(&b, &dims, &peer_strides, 0).unwrap();
                    let mut dest =
                        StridedViewMut::<f64>::new(&mut out, &dims, &out_strides, 0).unwrap();
                    zip_map2_into(&mut dest, &a_view, &b_view, |x, y| x + y).unwrap();
                };
                for _ in 0..WARMUPS {
                    run();
                }
                let mut samples = Vec::with_capacity(SAMPLES);
                for _ in 0..SAMPLES {
                    samples.push(time_ms(repeats, &mut run));
                }
                let ms = median(&samples);
                println!(
                    "SUMMARY f64 {threads}T n={len:<9} {label:<18} ms={ms:>9.4} ns/el={:>7.3}",
                    ms * 1e6 / len as f64,
                );
            },
        );
    });
}

fn main() {
    println!("zip_map2_into with a stride-0 inner broadcast operand");
    for threads in THREAD_COUNTS {
        for (rows, cols) in [(1024usize, 64usize), (4096, 256), (1023, 67)] {
            measure("broadcast_add", rows, cols, threads);
        }
    }
    black_box(());
}
