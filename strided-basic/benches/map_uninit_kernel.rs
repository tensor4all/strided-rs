//! Paired regression benchmark for `map_into_uninit_with_kernel`.
//!
//! The entry point exists so that a caller's vectorized kernel can reach the runs this crate
//! already knows are flat — the whole buffer on the sequential contiguous path and one block at a
//! time on the parallel path. It must therefore cost nothing measurable over the typed entry on
//! the layouts where a flat run exists (the fast-path hit), and must not add overhead to the ones
//! where none does (the miss), because on a miss it falls through to exactly that entry.
//!
//! Every case runs at one and four workers in an explicitly bounded Rayon pool whose thread count
//! is asserted, so the number on a `SUMMARY` line always belongs to the configuration named there.
//! The kernel writes element by element on purpose: this measures the plumbing, not a SIMD math
//! library, and the pair is reported as `kernel / typed`, so a value near 1.0 is the expected
//! result rather than a speedup.
//!
//! ```text
//! cargo bench -p strided-basic --features parallel --bench map_uninit_kernel
//! ```

use core::mem::MaybeUninit;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::Instant;

use strided_basic::{
    map_into, map_into_uninit_with_kernel, with_execution_policy, ExecutionPolicy, Identity,
    StridedView, StridedViewMut,
};

const WARMUPS: usize = 4;
const SAMPLES: usize = 15;
const THREAD_COUNTS: [usize; 2] = [1, 4];
/// Target work per sample, in elements; `repeats` is derived from it.
const SAMPLE_ELEMENTS: usize = 1 << 20;

/// Repeat count that keeps one sample near a millisecond at this host's ~0.3 ns/element.
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

/// How many elements a buffer has to hold for `dims`/`strides`, which is more than their product
/// for a gapped layout.
fn span(dims: &[usize], strides: &[isize]) -> usize {
    dims.iter()
        .zip(strides)
        .filter(|(&dim, _)| dim > 1)
        .map(|(&dim, &stride)| (dim - 1) * stride as usize)
        .sum::<usize>()
        + 1
}

/// Measures `typed` and `kernel` alternately under `threads` workers and prints the pair.
fn measure_pair(
    label: &str,
    dtype: &str,
    threads: usize,
    elements: usize,
    mut typed: impl FnMut() + Send,
    mut kernel: impl FnMut() + Send,
) {
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
                let repeats = repeats(elements);
                for _ in 0..WARMUPS {
                    typed();
                    kernel();
                }
                let mut typed_samples = Vec::with_capacity(SAMPLES);
                let mut kernel_samples = Vec::with_capacity(SAMPLES);
                for sample in 0..SAMPLES {
                    // Alternate the order so drift does not favour one side.
                    if sample % 2 == 0 {
                        typed_samples.push(time_ms(repeats, &mut typed));
                        kernel_samples.push(time_ms(repeats, &mut kernel));
                    } else {
                        kernel_samples.push(time_ms(repeats, &mut kernel));
                        typed_samples.push(time_ms(repeats, &mut typed));
                    }
                }
                let (typed_ms, kernel_ms) = (median(&typed_samples), median(&kernel_samples));
                println!(
                    "SUMMARY {dtype:<3} {threads}T n={elements:<9} {label:<15} \
                     typed={typed_ms:>9.4}ms kernel={kernel_ms:>9.4}ms ratio={:>6.3} \
                     ns/el={:>7.3}",
                    kernel_ms / typed_ms,
                    kernel_ms * 1e6 / elements as f64,
                );
            },
        );
    });
}

/// The cases: `(label, dims, strides)`. A pair whose strides are contiguous in the same order is
/// a hit; a gapped layout is not contiguous in either order and is the fast-path miss.
fn cases(len: usize) -> Vec<(String, Vec<usize>, Vec<isize>)> {
    let side = (len as f64).sqrt().round().max(1.0) as usize;
    let rows = (len / side).max(1);
    let cube = (len as f64).cbrt().round().max(1.0) as usize;
    let depth = (len / (cube * cube)).max(1);
    let gap_rows = (len / (2 * side)).max(1);
    vec![
        ("1d contiguous".into(), vec![len], vec![1]),
        (
            "2d contiguous".into(),
            vec![rows, side],
            vec![1, rows as isize],
        ),
        (
            "3d contiguous".into(),
            vec![depth, cube, cube],
            vec![1, depth as isize, (depth * cube) as isize],
        ),
        ("1d gapped".into(), vec![len / 2], vec![2]),
        (
            "2d gapped".into(),
            vec![gap_rows, side],
            vec![1, (2 * gap_rows) as isize],
        ),
    ]
}

macro_rules! run_dtype {
    ($dtype:ty, $name:literal, $len:expr, $threads:expr) => {
        for (label, dims, strides) in cases($len) {
            let elements: usize = dims.iter().product();
            let allocated = span(&dims, &strides);
            let source: Vec<$dtype> = (0..allocated).map(|i| (i % 251) as $dtype).collect();
            let mut typed_out = vec![0 as $dtype; allocated];
            let mut kernel_out = vec![MaybeUninit::<$dtype>::uninit(); allocated];

            measure_pair(
                &label,
                $name,
                $threads,
                elements,
                || {
                    let mut dest = StridedViewMut::new(&mut typed_out, &dims, &strides, 0).unwrap();
                    let view =
                        StridedView::<$dtype, Identity>::new(&source, &dims, &strides, 0).unwrap();
                    map_into(&mut dest, &view, |x| x + x).unwrap();
                },
                || {
                    let mut dest =
                        StridedViewMut::new(&mut kernel_out, &dims, &strides, 0).unwrap();
                    let view =
                        StridedView::<$dtype, Identity>::new(&source, &dims, &strides, 0).unwrap();
                    map_into_uninit_with_kernel(
                        &mut dest,
                        &view,
                        |src: &[$dtype], dst: &mut [MaybeUninit<$dtype>]| {
                            for (slot, value) in dst.iter_mut().zip(src) {
                                slot.write(*value + *value);
                            }
                            true
                        },
                        |x| x + x,
                    )
                    .unwrap();
                },
            );
        }
    };
}

fn main() {
    println!("map_into_uninit_with_kernel, paired against map_into: the ratio is kernel / typed");
    for threads in THREAD_COUNTS {
        for len in [1usize << 16, 1 << 20, 1 << 22] {
            run_dtype!(f64, "f64", len, threads);
            if len <= 1 << 20 {
                run_dtype!(f32, "f32", len, threads);
            }
        }
    }
    black_box(());
}
