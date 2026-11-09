//! Evidence for tensor4all/strided-rs#428: the view path (`copy_into`,
//! `map_into`) now takes the `slabs_share_no_line` decline that plan and
//! replay took in [#422]. One process times both answers on the same layout by
//! clearing `bench_hooks::DECLINE_TILES` between samples: A is the decline
//! (the flat walk), B is the model's tiles.
//!
//! [#422]: https://github.com/tensor4all/strided-rs/pull/422
//!
//! `harness = false`, like the other benches here: rows are reported, and the
//! `Gate::Parity` rows additionally assert that the views whose tiling does not
//! change cost the same either way. Nothing here is a product gate.
//!
//! ```text
//! cargo bench -p strided-basic --features bench-hooks --bench view_decline_428
//! ```

use std::hint::black_box;
use std::sync::atomic::Ordering;

use strided_basic::bench_hooks::{planned_blocks, DECLINE_TILES};
use strided_basic::{copy_into, map_into, Exec, StridedView, StridedViewMut};

#[path = "support/mod.rs"]
mod support;

#[global_allocator]
static GLOBAL: support::CountingAllocator = support::CountingAllocator;

use support::{measure_case, print_table, CaseResult, Gate};

/// F3 (`strided-basic/src/raw_ops/tests/tiled_tests.rs`): both operands stream
/// whole 32 KiB runs along axis 0 and the tail axes only permute those runs.
/// The model offers `[4096, 16, 16]`, and no visit order reuses a line.
const F3: (&[usize], &[isize], &[isize]) =
    (&[4096, 17, 33], &[1, 4096, 69_632], &[1, 135_168, 4096]);

/// The views below are built from the layout's own reach, so a sparse tail
/// needs no exact-size buffer: `max offset + 1` elements.
fn span(dims: &[usize], strides: &[isize]) -> usize {
    let reach: usize = strides
        .iter()
        .zip(dims)
        .map(|(&stride, &dim)| (dim - 1) * stride.unsigned_abs())
        .sum();
    reach + 1
}

fn run(dst: &mut StridedViewMut<'_, f64>, src: &StridedView<'_, f64>, map: bool) {
    if map {
        map_into(dst, src, |value| value, Exec::serial()).unwrap();
    } else {
        copy_into(dst, src, Exec::serial()).unwrap();
    }
}

/// One layout, both routing answers: A keeps the flat walk, B tiles.
///
/// Both arms write the same buffer, so the ratio cannot be a page-placement
/// difference between two allocations; the per-call view rebuild costs the
/// same on each side.
fn pair(
    id: &'static str,
    description: &'static str,
    (dims, dst_strides, src_strides): (&[usize], &[isize], &[isize]),
    iterations: usize,
    map: bool,
    gate: Gate,
) -> CaseResult {
    let src_data: Vec<f64> = (0..span(dims, src_strides)).map(|k| k as f64).collect();
    let src = StridedView::<f64>::new(&src_data, dims, src_strides, 0).unwrap();
    let dst_data = std::cell::RefCell::new(vec![0.0f64; span(dims, dst_strides)]);
    let write = |flat: bool| {
        DECLINE_TILES.store(flat, Ordering::Relaxed);
        let mut data = dst_data.borrow_mut();
        let mut dst = StridedViewMut::<f64>::new(&mut data[..], dims, dst_strides, 0).unwrap();
        run(&mut dst, &src, map);
        black_box(dst.as_mut_ptr());
    };

    measure_case(
        id,
        description,
        iterations,
        gate,
        || write(true),
        || write(false),
    )
}

fn main() {
    // Every row is single-threaded, so the ratio is the routing decision and
    // not the thread team's chunking.

    // F3's axes are already in iteration order (the planner's order is the
    // identity and nothing fuses), so this is the block vector `copy_into`
    // walks in each arm of F3c/F3m below.
    let (dims, dst, src) = F3;
    let strides: [&[isize]; 2] = [dst, src];
    let elem_sizes = [std::mem::size_of::<f64>(); 2];
    for flat in [true, false] {
        DECLINE_TILES.store(flat, Ordering::Relaxed);
        println!(
            "F3 blocks, decline {}: {:?}",
            if flat { "on " } else { "off" },
            planned_blocks(dims, &strides, &elem_sizes)
        );
    }
    DECLINE_TILES.store(true, Ordering::Relaxed);

    let cases = vec![
        // The measured case from #422, both view entries.
        pair(
            "F3c",
            "F3 [4096,17,33] copy_into: flat walk vs the model's tiles",
            F3,
            2,
            false,
            Gate::Report,
        ),
        pair(
            "F3m",
            "F3 [4096,17,33] map_into: flat walk vs the model's tiles",
            F3,
            2,
            true,
            Gate::Report,
        ),
        // Strided.jl-parity rows: their tiling must not change, so A and B run
        // the same blocks and the row must sit at parity.
        pair(
            "Pc",
            "contiguous [2048,512] copy_into (no tiles either way)",
            (&[2048, 512], &[1, 2048], &[1, 2048]),
            5,
            false,
            Gate::Parity,
        ),
        pair(
            "Pt",
            "transposed [1024,1024] copy_into (tiles, not declined)",
            (&[1024, 1024], &[1024, 1], &[1, 1024]),
            3,
            false,
            Gate::Parity,
        ),
        pair(
            "Pb",
            "broadcast [512,512] copy_into (tiles, not declined)",
            (&[512, 512], &[1, 512], &[0, 1]),
            5,
            false,
            Gate::Parity,
        ),
    ];

    print_table(&cases, "### #428 view-path decline vs the model's tiles");
    support::exit_if_failed(&cases);
}
