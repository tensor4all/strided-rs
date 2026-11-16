//! Unit tests for compilation and the block order.
//!
//! The rejection table is the D1/D3/D5/D6 contract: every rule a caller can
//! break is a typed error, and every rejection names itself. The order tests pin
//! the summation association the crate documents, which the integration tests
//! then check against the reference implementation.

use strided_basic::{Exec, RawStridedMut, RawStridedRef, RunIndex};

use super::*;
use crate::spec::{IndexMode, Layout, Operation, Path, Spec};

const BATCHED: &[IndexMode] = &[IndexMode::Batched];
const SHARED: &[IndexMode] = &[IndexMode::Shared];

fn layout<'a>(dims: &'a [usize], strides: &'a [isize]) -> Layout<'a> {
    Layout { dims, strides }
}

fn spec<'a>(
    operations: &'a [Operation<'a>],
    paths: &'a [Path<'a>],
    indexes: &'a [&'a [IndexMode<'a>]],
) -> Spec<'a> {
    Spec {
        num_batch_axes: 1,
        operations,
        paths,
        indexes,
    }
}

/// The error `compile` returns for this descriptor and these layouts.
fn compile_error(spec: &Spec<'_>, inputs: &[Layout<'_>], outputs: &[Layout<'_>]) -> SegPolyError {
    SegPolyPlan::<f64>::compile(spec, inputs, outputs).expect_err("compile rejects this descriptor")
}

#[test]
fn rejects_more_than_one_batch_axis() {
    let axes = 2;
    let modes: &[IndexMode] = &[IndexMode::Batched, IndexMode::Batched];
    let indexes = vec![modes];
    let mut spec = spec(&[], &[], &indexes);
    spec.num_batch_axes = axes;
    let error = compile_error(&spec, &[layout(&[1, 1, 1], &[1, 1, 1])], &[]);
    assert!(matches!(error, SegPolyError::MultiBatchAxis { axes: 2 }));
}

#[test]
fn rejects_disagreeing_lane_extents() {
    let x = [2, 2, 3];
    let w = [2, 2, 5];
    let strides = [6, 3, 1];
    let error = compile_error(
        &spec(&[], &[], &[BATCHED, BATCHED]),
        &[layout(&x, &strides), layout(&w, &strides)],
        &[],
    );
    assert!(matches!(
        error,
        SegPolyError::LaneExtent { lane: 3, u: 5, .. }
    ));
}

#[test]
fn rejects_a_lane_broadcast_output() {
    let x = [2, 2, 3];
    let strides = [6, 3, 1];
    let y = [2, 2, 1];
    let error = compile_error(
        &spec(&[], &[], &[BATCHED, BATCHED]),
        &[layout(&x, &strides)],
        &[layout(&y, &strides)],
    );
    assert!(matches!(
        error,
        SegPolyError::OutputLaneBroadcast { lane: 1, u: 3, .. }
    ));
}

#[test]
fn rejects_a_mode_count_mismatch() {
    let index = [0i32];
    let error = compile_error(
        &spec(&[], &[], &[BATCHED, BATCHED]),
        &[layout(&[1, 1, 1], &[1, 1, 1])],
        &[],
    );
    assert!(matches!(error, SegPolyError::BufferModeCount { .. }));

    // One mode list per buffer, but two axes' worth of modes for that buffer.
    let two: &[IndexMode] = &[IndexMode::Batched, IndexMode::Batched];
    let indexes = vec![two];
    let mut spec = spec(&[], &[], &indexes);
    spec.num_batch_axes = 2;
    assert!(matches!(
        compile_error(&spec, &[], &[]),
        SegPolyError::MultiBatchAxis { .. }
    ));
    let _ = index;
}

#[test]
fn rejects_a_mode_a_buffer_may_not_use() {
    // An input cannot be scattered into, and an output cannot be gathered from.
    let runs = RunIndex::from_ptr(vec![0, 0]).unwrap();
    let scatter = &[IndexMode::Scatter(&runs)];
    let error = compile_error(
        &spec(&[], &[], &[scatter]),
        &[layout(&[1, 1, 1], &[1, 1, 1])],
        &[],
    );
    assert!(matches!(
        error,
        SegPolyError::IndexModeNotAllowed { buffer: 0 }
    ));

    let index = [0i32];
    let gather = &[IndexMode::Gather(&index)];
    let error = compile_error(
        &spec(&[], &[], &[gather]),
        &[],
        &[layout(&[1, 1, 1], &[1, 1, 1])],
    );
    assert!(matches!(
        error,
        SegPolyError::IndexModeNotAllowed { buffer: 0 }
    ));
}

#[test]
fn rejects_an_unused_buffer() {
    let buffers = [0u32, 2];
    let segments = [0u32, 0];
    let operations = [Operation {
        buffers: &buffers,
        path_start: 0,
        num_paths: 1,
    }];
    let paths = [Path {
        segments: &segments,
        coefficient: 1.0,
    }];
    let error = compile_error(
        &spec(&operations, &paths, &[BATCHED, BATCHED, BATCHED]),
        &[layout(&[1, 1, 1], &[1, 1, 1]); 2],
        &[layout(&[1, 1, 1], &[1, 1, 1])],
    );
    assert!(matches!(error, SegPolyError::UnusedBuffer { buffer: 1 }));
}

#[test]
fn rejects_a_buffer_in_the_wrong_role() {
    // The output is not the last operand.
    let buffers = [2u32, 0];
    let segments = [0u32, 0];
    let operations = [Operation {
        buffers: &buffers,
        path_start: 0,
        num_paths: 1,
    }];
    let paths = [Path {
        segments: &segments,
        coefficient: 1.0,
    }];
    let error = compile_error(
        &spec(&operations, &paths, &[BATCHED, BATCHED, BATCHED]),
        &[layout(&[1, 1, 1], &[1, 1, 1]); 2],
        &[layout(&[1, 1, 1], &[1, 1, 1])],
    );
    assert!(
        matches!(error, SegPolyError::BufferRole { buffer: 2, .. }),
        "{error:?}"
    );
}

#[test]
fn rejects_an_arity_mismatch() {
    let buffers = [0u32, 1];
    let segments = [0u32, 0, 0];
    let operations = [Operation {
        buffers: &buffers,
        path_start: 0,
        num_paths: 1,
    }];
    let paths = [Path {
        segments: &segments,
        coefficient: 1.0,
    }];
    let error = compile_error(
        &spec(&operations, &paths, &[BATCHED, BATCHED]),
        &[layout(&[1, 1, 1], &[1, 1, 1])],
        &[layout(&[1, 1, 1], &[1, 1, 1])],
    );
    assert!(matches!(
        error,
        SegPolyError::ArityMismatch {
            operands: 2,
            segments: 3,
            ..
        }
    ));
}

#[test]
fn rejects_a_path_range_past_the_descriptor() {
    let buffers = [0u32, 1];
    let segments = [0u32, 0];
    let operations = [Operation {
        buffers: &buffers,
        path_start: 1,
        num_paths: 1,
    }];
    let paths = [Path {
        segments: &segments,
        coefficient: 1.0,
    }];
    let error = compile_error(
        &spec(&operations, &paths, &[BATCHED, BATCHED]),
        &[layout(&[1, 1, 1], &[1, 1, 1])],
        &[layout(&[1, 1, 1], &[1, 1, 1])],
    );
    assert!(matches!(
        error,
        SegPolyError::PathRange {
            path_start: 1,
            num_paths: 1,
            paths: 1,
            ..
        }
    ));
}

#[test]
fn rejects_a_segment_past_its_buffer() {
    let buffers = [0u32, 1];
    let segments = [0u32, 4];
    let operations = [Operation {
        buffers: &buffers,
        path_start: 0,
        num_paths: 1,
    }];
    let paths = [Path {
        segments: &segments,
        coefficient: 1.0,
    }];
    let error = compile_error(
        &spec(&operations, &paths, &[BATCHED, BATCHED]),
        &[layout(&[1, 2, 1], &[2, 1, 1])],
        &[layout(&[1, 2, 1], &[2, 1, 1])],
    );
    assert!(matches!(
        error,
        SegPolyError::SegmentOutOfRange {
            segment: 4,
            segments: 2,
            ..
        }
    ));
}

#[test]
fn rejects_disagreeing_batch_extents() {
    let buffers = [0u32, 1];
    let segments = [0u32, 0];
    let operations = [Operation {
        buffers: &buffers,
        path_start: 0,
        num_paths: 1,
    }];
    let paths = [Path {
        segments: &segments,
        coefficient: 1.0,
    }];
    let error = compile_error(
        &spec(&operations, &paths, &[BATCHED, BATCHED]),
        &[layout(&[3, 1, 1], &[1, 1, 1])],
        &[layout(&[2, 1, 1], &[1, 1, 1])],
    );
    assert!(matches!(
        error,
        SegPolyError::BatchExtentMismatch {
            extent: 2,
            batch: 3,
            ..
        }
    ));
}

#[test]
fn rejects_a_scatter_output_whose_rows_are_not_its_runs() {
    let dest = [0i32, 0];
    let runs = RunIndex::from_dest(&dest, 2).unwrap();
    let mode: &[IndexMode] = &[IndexMode::Scatter(&runs)];
    let buffers = [0u32, 1];
    let segments = [0u32, 0];
    let operations = [Operation {
        buffers: &buffers,
        path_start: 0,
        num_paths: 1,
    }];
    let paths = [Path {
        segments: &segments,
        coefficient: 1.0,
    }];
    // Three rows in the view, two runs in the topology: the executor would write
    // row 1 of a two-row topology past the extent it was given.
    let error = compile_error(
        &spec(&operations, &paths, &[BATCHED, mode]),
        &[layout(&[2, 1, 1], &[1, 1, 1])],
        &[layout(&[3, 1, 1], &[1, 1, 1])],
    );
    assert!(matches!(
        error,
        SegPolyError::BatchExtentMismatch {
            extent: 3,
            batch: 2,
            ..
        }
    ));
}

#[test]
fn rejects_an_unrepresentable_buffer_count() {
    let modes: Vec<&[IndexMode]> = (0..=MAX_OUTPUTS).map(|_| BATCHED).collect();
    let error = compile_error(
        &spec(&[], &[], &modes),
        &[layout(&[1, 1, 1], &[1, 1, 1])],
        &vec![layout(&[1, 1, 1], &[1, 1, 1]); MAX_OUTPUTS + 1],
    );
    assert!(matches!(error, SegPolyError::TooManyBuffers { .. }));
}

/// One input row per batch point, one shared output: `Σ_n x[n]`.
fn shared_sum_case(rows: usize, lanes: usize) -> (Vec<usize>, [isize; 3]) {
    (vec![rows, 1, lanes], [lanes as isize, lanes as isize, 1])
}

/// The documented order, in scalar code: fixed blocks, combined left to right.
fn shared_sum_reference(values: &[f64], lanes: usize, block_rows: usize) -> Vec<f64> {
    let rows = values.len() / lanes;
    let mut total = vec![0.0; lanes];
    let mut start = 0;
    while start < rows {
        let end = (start + block_rows).min(rows);
        let mut partial = vec![0.0; lanes];
        for row in start..end {
            for lane in 0..lanes {
                partial[lane] += values[row * lanes + lane];
            }
        }
        for lane in 0..lanes {
            total[lane] += partial[lane];
        }
        start = end;
    }
    total
}

fn run_shared_sum(values: &[f64], lanes: usize, block_rows: usize) -> Vec<f64> {
    let rows = values.len() / lanes;
    let (input_dims, input_strides) = shared_sum_case(rows, lanes);
    let out_dims = [1usize, 1, lanes];
    let out_strides = [lanes as isize, lanes as isize, 1];
    let buffers = [0u32, 1];
    let segments = [0u32, 0];
    let operations = [Operation {
        buffers: &buffers,
        path_start: 0,
        num_paths: 1,
    }];
    let paths = [Path {
        segments: &segments,
        coefficient: 1.0,
    }];
    let descriptor = spec(&operations, &paths, &[BATCHED, SHARED]);
    let plan = SegPolyPlan::<f64>::compile_with_block_rows(
        &descriptor,
        &[layout(&input_dims, &input_strides)],
        &[layout(&out_dims, &out_strides)],
        block_rows,
    )
    .expect("compiles");
    let mut out = vec![0.0; lanes];
    let mut scratch = vec![0.0; plan.scratch_len(1)];
    let input = RawStridedRef::new(values, &input_dims, &input_strides, 0).unwrap();
    {
        let mut outputs = [RawStridedMut::new(&mut out, &out_dims, &out_strides, 0).unwrap()];
        plan.execute_into(
            &mut outputs,
            &[input],
            descriptor.indexes,
            &mut scratch,
            Exec::serial(),
        )
        .expect("executes");
    }
    out
}

#[test]
fn block_partials_combine_in_the_documented_order() {
    // Values whose sum depends on the association: 1 + 1e16 - 1e16 is 1 in one
    // order and 0 in another.
    let lanes = 1;
    let values = [1.0, 1e16, -1e16, 1.0, -1e16, 2.0, 1e16, 3.0];
    for block_rows in [1, 2, 3, 5, BLOCK_ROWS] {
        let actual = run_shared_sum(&values, lanes, block_rows);
        let expected = shared_sum_reference(&values, lanes, block_rows);
        assert_eq!(
            actual, expected,
            "block_rows {block_rows}: bits must follow the documented order"
        );
    }
    // Different blockings do move the bits, which is why the block size is a
    // documented constant and not a tuning knob.
    assert_ne!(
        run_shared_sum(&values, lanes, 2),
        run_shared_sum(&values, lanes, BLOCK_ROWS)
    );
}

#[test]
fn rows_below_one_block_are_block_invariant() {
    // A unit that fits one block must not change when the block size shrinks to
    // it, so a plan compiled with the default and a plan with a smaller block
    // size agree bit for bit on short units.
    let values: Vec<f64> = (0..8).map(|i| (i as f64) * 0.5 - 0.25).collect();
    let reference = run_shared_sum(&values, 3, BLOCK_ROWS);
    for block_rows in [8, 16, BLOCK_ROWS] {
        assert_eq!(run_shared_sum(&values, 3, block_rows), reference);
    }
}

#[test]
fn empty_runs_write_zero_and_accumulate_nothing() {
    let dest = [0i32, 0];
    let runs = RunIndex::from_dest(&dest, 2).unwrap();
    let out_mode: &[IndexMode] = &[IndexMode::Scatter(&runs)];
    let buffers = [0u32, 1];
    let segments = [0u32, 0];
    let operations = [Operation {
        buffers: &buffers,
        path_start: 0,
        num_paths: 1,
    }];
    let paths = [Path {
        segments: &segments,
        coefficient: 3.0,
    }];
    let modes: [&[IndexMode]; 2] = [BATCHED, out_mode];
    let descriptor = spec(&operations, &paths, &modes);
    let input_dims = [2usize, 1, 1];
    let out_dims = [2usize, 1, 1];
    let strides = [1isize, 1, 1];
    let plan = SegPolyPlan::<f64>::compile(
        &descriptor,
        &[layout(&input_dims, &strides)],
        &[layout(&out_dims, &strides)],
    )
    .expect("compiles");
    let x = [2.0f64, 5.0];
    let mut scratch = vec![0.0; plan.scratch_len(1)];
    let mut y = [7.0f64, 11.0];
    let input = RawStridedRef::new(&x, &input_dims, &strides, 0).unwrap();
    plan.execute_into(
        &mut [RawStridedMut::new(&mut y, &out_dims, &strides, 0).unwrap()],
        &[input],
        descriptor.indexes,
        &mut scratch,
        Exec::serial(),
    )
    .unwrap();
    // Run 0 holds both batch points (3 * (2 + 5)); run 1 is empty and is zeroed.
    assert_eq!(y, [21.0, 0.0]);

    plan.execute_acc(
        &mut [RawStridedMut::new(&mut y, &out_dims, &strides, 0).unwrap()],
        &[input],
        descriptor.indexes,
        &mut scratch,
        Exec::serial(),
    )
    .unwrap();
    assert_eq!(y, [42.0, 0.0]);
}

#[test]
fn zero_extent_calls_are_accepted() {
    let buffers = [0u32, 1];
    let segments = [0u32, 0];
    let operations = [Operation {
        buffers: &buffers,
        path_start: 0,
        num_paths: 1,
    }];
    let paths = [Path {
        segments: &segments,
        coefficient: 1.0,
    }];
    let descriptor = spec(&operations, &paths, &[BATCHED, BATCHED]);
    let dims = [0usize, 1, 1];
    let strides = [1isize, 1, 1];
    let plan = SegPolyPlan::<f64>::compile(
        &descriptor,
        &[layout(&dims, &strides)],
        &[layout(&dims, &strides)],
    )
    .expect("a zero batch compiles");
    let mut scratch = vec![0.0; plan.scratch_len(1)];
    plan.execute_into(
        &mut [RawStridedMut::new(&mut [], &dims, &strides, 0).unwrap()],
        &[RawStridedRef::new(&[], &dims, &strides, 0).unwrap()],
        descriptor.indexes,
        &mut scratch,
        Exec::serial(),
    )
    .expect("a zero batch executes");
}

#[test]
fn rejects_a_view_that_is_not_the_compiled_layout() {
    let buffers = [0u32, 1];
    let segments = [0u32, 0];
    let operations = [Operation {
        buffers: &buffers,
        path_start: 0,
        num_paths: 1,
    }];
    let paths = [Path {
        segments: &segments,
        coefficient: 1.0,
    }];
    let descriptor = spec(&operations, &paths, &[BATCHED, BATCHED]);
    let dims = [2usize, 1, 1];
    let strides = [1isize, 1, 1];
    let plan = SegPolyPlan::<f64>::compile(
        &descriptor,
        &[layout(&dims, &strides)],
        &[layout(&dims, &strides)],
    )
    .expect("compiles");
    let x = [1.0f64, 2.0, 3.0, 4.0];
    let mut y = [0.0f64; 4];
    let mut scratch = vec![0.0; plan.scratch_len(1)];
    // Same extents, a different stride: in bounds, but not the compiled layout.
    let other_dims = [2usize, 1, 1];
    let other_strides = [2isize, 2, 1];
    let view = RawStridedRef::new(&x, &other_dims, &other_strides, 0);
    let output = RawStridedMut::new(&mut y, &other_dims, &other_strides, 0);
    let error = plan
        .execute_into(
            &mut [output.expect("in bounds")],
            &[view.expect("in bounds")],
            descriptor.indexes,
            &mut scratch,
            Exec::serial(),
        )
        .expect_err("a different layout is a mismatch");
    assert!(matches!(
        error,
        SegPolyError::Strided(StridedError::PlanLayoutMismatch)
    ));
}

#[test]
fn rejects_gather_indices_out_of_range() {
    let index = [0i32, -1];
    let gather: &[IndexMode] = &[IndexMode::Gather(&index)];
    let buffers = [0u32, 1];
    let segments = [0u32, 0];
    let operations = [Operation {
        buffers: &buffers,
        path_start: 0,
        num_paths: 1,
    }];
    let paths = [Path {
        segments: &segments,
        coefficient: 1.0,
    }];
    let modes: [&[IndexMode]; 2] = [gather, BATCHED];
    let descriptor = spec(&operations, &paths, &modes);
    let source_dims = [4usize, 1, 1];
    let out_dims = [2usize, 1, 1];
    let strides = [1isize, 1, 1];
    let plan = SegPolyPlan::<f64>::compile(
        &descriptor,
        &[layout(&source_dims, &strides)],
        &[layout(&out_dims, &strides)],
    )
    .expect("compiles");
    let x = [0.0f64; 4];
    let mut y = [0.0f64; 2];
    let mut scratch = vec![0.0; plan.scratch_len(1)];
    let error = plan
        .execute_into(
            &mut [RawStridedMut::new(&mut y, &out_dims, &strides, 0).unwrap()],
            &[RawStridedRef::new(&x, &source_dims, &strides, 0).unwrap()],
            descriptor.indexes,
            &mut scratch,
            Exec::serial(),
        )
        .expect_err("a negative index is rejected");
    assert!(matches!(
        error,
        SegPolyError::IndexOutOfRange {
            buffer: 0,
            index: -1,
            extent: 4
        }
    ));
}

#[test]
fn rejects_scratch_below_the_block_floor() {
    // Two runs of three rows each, a 1-element block: every block partial must
    // be able to exist at once, so the scratch floor is the unit's block count
    // plus one running tile.
    let dest = [0i32; 8];
    let runs = RunIndex::from_dest(&dest, 2).unwrap();
    let mode: &[IndexMode] = &[IndexMode::Scatter(&runs)];
    let buffers = [0u32, 1];
    let segments = [0u32, 0];
    let operations = [Operation {
        buffers: &buffers,
        path_start: 0,
        num_paths: 1,
    }];
    let paths = [Path {
        segments: &segments,
        coefficient: 1.0,
    }];
    let modes: [&[IndexMode]; 2] = [BATCHED, mode];
    let descriptor = spec(&operations, &paths, &modes);
    let input_dims = [8usize, 1, 1];
    let out_dims = [2usize, 1, 1];
    let strides = [1isize, 1, 1];
    let plan = SegPolyPlan::<f64>::compile_with_block_rows(
        &descriptor,
        &[layout(&input_dims, &strides)],
        &[layout(&out_dims, &strides)],
        1,
    )
    .expect("compiles");
    assert_eq!(plan.tile_elems(), 1);
    let need = plan.scratch_len(1);
    assert!(
        need >= 5,
        "two units of four blocks need their partials: {need}"
    );
    let x = [1.0f64; 8];
    let mut y = [0.0f64; 2];
    let mut scratch = vec![0.0; need - 1];
    let error = plan
        .execute_into(
            &mut [RawStridedMut::new(&mut y, &out_dims, &strides, 0).unwrap()],
            &[RawStridedRef::new(&x, &input_dims, &strides, 0).unwrap()],
            descriptor.indexes,
            &mut scratch,
            Exec::serial(),
        )
        .expect_err("one element short is rejected");
    assert!(matches!(error, SegPolyError::ScratchTooSmall { .. }));
}
