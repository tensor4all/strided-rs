//! Zero-allocation windows for the executor's hot path (tensor4all/strided-rs#479).
//!
//! `SegPolyPlan::compile` allocates at plan time, and the harness builds every
//! index, view and `RunIndex` outside the counted windows. `execute_into` and
//! `execute_acc` must not allocate: the per-call state is a stack `Call`, the
//! scratch is the caller's, and the block combine works on slices of it.
//!
//! The windows count the calling thread only, and they are serial: in a debug
//! build threadteam's token instrumentation allocates on the dispatching side of
//! every `fork`, which is not a per-call allocation of this kernel. The fanned-out
//! path's own allocations belong to `threadteam` and are out of scope here, as in
//! `strided-basic`'s sorted-run counter.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use strided_basic::{Exec, RawStridedMut, RawStridedRef};
use strided_segpoly::{IndexMode, SegPolyPlan};

#[allow(dead_code)] // each test binary uses a different part of the harness
#[path = "support/mod.rs"]
mod support;

use support::{Buffer, Case, Mode, Op, PathSpec};

thread_local! {
    static THREAD_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn record_allocation() {
    let _ = THREAD_ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
}

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn count_thread_allocations(run: impl FnOnce()) -> usize {
    let before = THREAD_ALLOCATIONS.with(Cell::get);
    run();
    THREAD_ALLOCATIONS.with(Cell::get) - before
}

fn payload(len: usize) -> Vec<f64> {
    (0..len)
        .map(|index| (index % 17) as f64 * 0.125 - 1.0)
        .collect()
}

/// A message-passing case: gathered rows, a batched per-edge weight, a scattered
/// output and a shared output, so the window walks every mode at once.
fn message_passing(batch: usize, segments: usize, u: usize) -> Case {
    let paths: Vec<PathSpec> = (0..segments)
        .map(|segment| PathSpec {
            segments: vec![segment as u32; 4],
            coefficient: 1.0 / (1.0 + segment as f64),
        })
        .collect();
    let source_rows = batch / 2;
    let runs = batch / 4;
    Case {
        batch,
        inputs: vec![
            Buffer::dense(
                [source_rows, segments, u],
                Mode::Gather((0..batch).map(|row| (row % source_rows) as i32).collect()),
                payload(source_rows * segments * u),
            ),
            Buffer::dense(
                [batch, segments, u],
                Mode::Batched,
                payload(batch * segments * u),
            ),
            Buffer::dense([1, segments, u], Mode::Shared, payload(segments * u)),
        ],
        outputs: vec![
            Buffer::dense(
                [runs, segments, u],
                Mode::Scatter((0..batch).map(|row| (row % runs) as i32).collect()),
                vec![0.0; runs * segments * u],
            ),
            Buffer::dense([1, segments, u], Mode::Shared, vec![0.0; segments * u]),
        ],
        ops: vec![
            // The scattered output: one operation over (x, w, x, out).
            Op {
                buffers: vec![0, 1, 0, 3],
                paths: paths.clone(),
            },
            // The shared output: the same product summed over the batch.
            Op {
                buffers: vec![0, 2, 4],
                paths: paths
                    .iter()
                    .map(|path| PathSpec {
                        segments: vec![path.segments[0]; 3],
                        coefficient: path.coefficient,
                    })
                    .collect(),
            },
        ],
        expected: vec![],
    }
}

fn exercise(name: &str, case: &Case, plan: &SegPolyPlan<f64>) {
    let num_inputs = case.inputs.len();
    let mut data = case.data();
    let mut scratch = vec![0.0f64; plan.scratch_len(1)];
    let (input_data, output_data) = data.split_at_mut(num_inputs);
    let inputs: Vec<RawStridedRef<'_, f64>> = input_data
        .iter()
        .zip(case.inputs.iter())
        .map(|(storage, buffer)| {
            RawStridedRef::new(
                storage,
                buffer.dims.as_slice(),
                buffer.strides.as_slice(),
                buffer.offset,
            )
            .unwrap()
        })
        .collect();
    let modes: Vec<Vec<IndexMode<'_>>> = case
        .inputs
        .iter()
        .chain(case.outputs.iter())
        .map(|buffer| vec![buffer.mode.of(buffer.runs.as_ref())])
        .collect();
    let indexes: Vec<&[IndexMode<'_>]> = modes.iter().map(|mode| mode.as_slice()).collect();

    // The views, the token and the scratch are the caller's work; only the plan
    // call itself is counted.
    let mut outputs: Vec<RawStridedMut<'_, f64>> = output_data
        .iter_mut()
        .zip(case.outputs.iter())
        .map(|(storage, buffer)| {
            RawStridedMut::new(
                storage,
                buffer.dims.as_slice(),
                buffer.strides.as_slice(),
                buffer.offset,
            )
            .unwrap()
        })
        .collect();
    let mut exec = Exec::serial();
    for (label, acc) in [("into", false), ("acc", true)] {
        let counted = count_thread_allocations(|| {
            let result = if acc {
                plan.execute_acc(
                    &mut outputs,
                    &inputs,
                    &indexes,
                    &mut scratch,
                    exec.reborrow(),
                )
            } else {
                plan.execute_into(
                    &mut outputs,
                    &inputs,
                    &indexes,
                    &mut scratch,
                    exec.reborrow(),
                )
            };
            result.expect("the case executes");
        });
        assert_eq!(
            counted, 0,
            "{name} execute_{label} allocated {counted} times"
        );
    }
}

#[test]
fn execute_allocates_nothing() {
    for (batch, segments, u) in [
        (16usize, 1usize, 1usize),
        (32, 4, 8),
        (64, 3, 1),
        // Past the stack lane tile, so the untitled lane loop runs.
        (8, 2, 128),
        // More rows than one block: the combine path with block partials.
        (600, 2, 4),
    ] {
        let case = message_passing(batch, segments, u);
        let plan = case.compile().expect("the case compiles");
        exercise(
            &format!("batch {batch} segments {segments} u {u}"),
            &case,
            &plan,
        );
    }
}
