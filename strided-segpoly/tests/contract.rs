//! Contract checks the fast path leans on: the descriptor rules a plan rejects, and
//! the layout and mode properties a plan must still execute correctly.
//!
//! Every test here pins something the post-implementation review found missing: a
//! gather shorter than the batch, a lane axis that is not unit-stride, a
//! destination that must be added once and not once per block, an empty output
//! beside a full one, an output written by two operations out of descriptor order,
//! an operation with more operands than the replay tables hold, a reversed batch
//! stride, a segment axis whose table cannot be allocated, and a unit-lane input
//! whose (arbitrary) lane stride must not be stepped.

#[allow(dead_code)] // each test binary uses a different part of the harness
#[path = "support/mod.rs"]
mod support;

use strided_basic::{Exec, RawStridedMut, RawStridedRef};
use strided_segpoly::{IndexMode, Layout, Operation, Path, SegPolyError, SegPolyPlan, Spec};
use support::{assert_close, Buffer, Case, Mode, Op, PathSpec};

/// Deterministic payload with mixed magnitudes, so regrouping a sum moves the
/// bits.
fn payload(len: usize) -> Vec<f64> {
    (0..len)
        .map(|index| {
            let mantissa = ((index as u64).wrapping_mul(2_654_435_761) % 10_007) as f64 / 10_007.0;
            let scale = 2f64.powi((index % 30) as i32 - 15);
            (mantissa - 0.5) * 2.0 * scale
        })
        .collect()
}

/// `y[b, s, l] = c_s · w[0, s, l] · x[b, s, l]` over one batched input, whose mode
/// the caller supplies (so a gather can be too short).
fn weight_times_input(batch: usize, segments: usize, u: usize, mode: Mode, rows: usize) -> Case {
    let paths: Vec<PathSpec> = (0..segments)
        .map(|segment| PathSpec {
            segments: vec![segment as u32; 3],
            coefficient: 0.5 + segment as f64,
        })
        .collect();
    Case {
        batch,
        inputs: vec![
            Buffer::dense([1, segments, u], Mode::Shared, payload(segments * u)),
            Buffer::dense([rows, segments, u], mode, payload(rows * segments * u)),
        ],
        outputs: vec![Buffer::dense(
            [batch, segments, u],
            Mode::Batched,
            vec![0.0; batch * segments * u],
        )],
        ops: vec![Op {
            buffers: vec![0, 1, 2],
            paths,
        }],
        expected: vec![],
    }
}

#[test]
fn a_gather_must_cover_the_batch() {
    // A gather is read once per batch point without a bound, so a shorter list
    // would be read past its end.
    let short = weight_times_input(4, 2, 2, Mode::Gather(vec![0, 1]), 2);
    let error = short.compile().expect_err("a short gather is rejected");
    assert!(
        matches!(error, SegPolyError::BatchExtentMismatch { .. }),
        "{error:?}"
    );

    let long = weight_times_input(4, 2, 2, Mode::Gather64(vec![0, 1, 0, 1, 0]), 2);
    let error = long.compile().expect_err("a long gather is rejected");
    assert!(
        matches!(error, SegPolyError::BatchExtentMismatch { .. }),
        "{error:?}"
    );

    // The same length rule at call time: the plan was compiled for four points
    // and the call hands it two.
    let full = weight_times_input(4, 2, 2, Mode::Gather64(vec![0, 0, 1, 1]), 2);
    let plan = full.compile().expect("a full gather compiles");
    let index = [0i64, 1];
    let gather = [IndexMode::Gather64(&index)];
    let shared = [IndexMode::Shared];
    let batched = [IndexMode::Batched];
    let indexes: [&[IndexMode<'_>]; 3] = [&shared, &gather, &batched];
    let mut data = full.data();
    let mut scratch = vec![0.0f64; plan.scratch_len(1)];
    let error = full
        .execute_with_modes(
            &plan,
            &mut data,
            &mut scratch,
            Exec::serial(),
            true,
            &indexes,
        )
        .expect_err("a short gather is rejected at call time too");
    assert!(
        matches!(error, SegPolyError::BatchExtentMismatch { .. }),
        "{error:?}"
    );
}

#[test]
fn an_input_reads_its_lanes_at_the_compiled_stride() {
    // The same logical payload twice: once lane-unit-stride, once with a gap
    // (lane stride 2) and a batch stride that skips the padding. Both plans must
    // produce the same output, at two budgets.
    let (batch, segments, u) = (5usize, 3usize, 4usize);
    let logical = payload(batch * segments * u);
    let dense = weight_times_input(batch, segments, u, Mode::Batched, batch);
    let mut padded = dense.clone();
    // Lane stride 2, segment stride 2u + 2, batch stride segments * (2u + 2) + 2:
    // every row is followed by one sentinel element per (segment, lane) group.
    let segment_stride = 2 * u + 2;
    let batch_stride = segments * segment_stride + 2;
    let mut storage = vec![-1.0f64; batch * batch_stride];
    for row in 0..batch {
        for segment in 0..segments {
            for lane in 0..u {
                let logical_index = (row * segments + segment) * u + lane;
                storage[row * batch_stride + segment * segment_stride + lane * 2] =
                    logical[logical_index];
            }
        }
    }
    padded.inputs[1] = Buffer::strided(
        [batch, segments, u],
        [batch_stride as isize, segment_stride as isize, 2],
        0,
        Mode::Batched,
        storage,
    );

    let dense_plan = dense.compile().expect("the dense case compiles");
    let padded_plan = padded.compile().expect("a lane-strided input compiles");
    let mut reference: Option<Vec<f64>> = None;
    for budget in [1usize, 2] {
        let mut dense_data = dense.data();
        let mut scratch = vec![0.0f64; dense_plan.scratch_len(budget)];
        dense
            .execute(
                &dense_plan,
                &mut dense_data,
                &mut scratch,
                Exec::serial(),
                true,
            )
            .expect("the dense case executes");
        let mut padded_data = padded.data();
        let mut scratch = vec![0.0f64; padded_plan.scratch_len(budget)];
        padded
            .execute(
                &padded_plan,
                &mut padded_data,
                &mut scratch,
                Exec::serial(),
                true,
            )
            .expect("the padded case executes");
        assert_eq!(
            dense_data[dense.inputs.len()],
            padded_data[padded.inputs.len()],
            "the lane stride must not change the result"
        );
        match &reference {
            None => reference = Some(padded_data[padded.inputs.len()].clone()),
            Some(reference) => assert_eq!(
                *reference,
                padded_data[padded.inputs.len()],
                "budget {budget} changed the bits"
            ),
        }
    }
}

#[test]
fn execute_acc_adds_the_destination_once_per_unit() {
    // A shared output sums 600 rows, which is more than one fixed block, so its
    // partials are combined. The destination value is the first term of the sum,
    // not one term per block.
    let (batch, segments, u) = (600usize, 2usize, 2usize);
    let mut case = weight_times_input(batch, segments, u, Mode::Batched, batch);
    case.outputs[0].mode = Mode::Shared;
    case.outputs[0].dims = [1, segments, u];
    case.outputs[0].strides = [(segments * u) as isize, u as isize, 1];
    case.outputs[0].data = vec![0.0; segments * u];

    let plan = case.compile().expect("the case compiles");
    assert!(
        batch > 2 * strided_segpoly::BLOCK_ROWS,
        "more than two blocks"
    );

    let mut zeroed = case.data();
    let mut scratch = vec![0.0f64; plan.scratch_len(1)];
    case.execute(&plan, &mut zeroed, &mut scratch, Exec::serial(), true)
        .expect("into executes");
    let sum = &zeroed[case.inputs.len()];

    let seed: Vec<f64> = sum.iter().map(|value| value * 0.5 + 5.0).collect();
    let mut seeded = case.data();
    seeded[case.inputs.len()] = seed.clone();
    let mut scratch = vec![0.0f64; plan.scratch_len(1)];
    case.execute(&plan, &mut seeded, &mut scratch, Exec::serial(), false)
        .expect("acc executes");
    let expected: Vec<f64> = seed.iter().zip(sum.iter()).map(|(s, v)| s + v).collect();
    assert_close(
        "accumulated destination",
        &seeded[case.inputs.len()],
        &expected,
        1e-12,
    );
    // The review's failure mode added the seed once per block: 2·seed + Σ would
    // be far off, so name it rather than letting the tolerance hide it.
    let doubled: Vec<f64> = seed
        .iter()
        .zip(sum.iter())
        .map(|(s, v)| 2.0 * s + v)
        .collect();
    assert!(
        seeded[case.inputs.len()]
            .iter()
            .zip(doubled.iter())
            .any(|(actual, wrong)| (actual - wrong).abs() > 1e-9 * wrong.abs().max(1.0)),
        "the destination was added once per block"
    );
}

#[test]
fn an_empty_output_beside_a_full_one_executes() {
    // A zero-segment output has no elements to write, so its empty view's pointer
    // must not be offset; the other output still has to be written.
    let (batch, segments, u) = (3usize, 2usize, 2usize);
    let mut case = weight_times_input(batch, segments, u, Mode::Batched, batch);
    let mut empty = Buffer::dense([1, 0, u], Mode::Shared, vec![]);
    empty.strides = [u as isize, u as isize, 1];
    case.outputs.push(empty);
    // The empty output is written by its own (path-less) operation, so it is not
    // an unused buffer.
    case.ops.push(Op {
        buffers: vec![0, 1, 3],
        paths: vec![],
    });
    let plan = case.compile().expect("an empty output compiles");
    let mut data = case.data();
    let mut scratch = vec![0.0f64; plan.scratch_len(1)];
    case.execute(&plan, &mut data, &mut scratch, Exec::serial(), true)
        .expect("an empty output executes");
    assert!(data[case.inputs.len()].iter().all(|value| *value != 0.0));
}

#[test]
fn an_output_written_by_two_operations_runs_both_in_descriptor_order() {
    // Operations 0 and 2 write output 0, operation 1 writes output 1. A per-output
    // range over the operation list would run operation 1 for output 0 (and could
    // run operation 0 twice).
    let (batch, segments, u) = (2usize, 1usize, 1usize);
    let paths = |coefficient: f64| {
        vec![PathSpec {
            segments: vec![0; 3],
            coefficient,
        }]
    };
    let case = Case {
        batch,
        inputs: vec![
            Buffer::dense([1, segments, u], Mode::Shared, vec![2.0]),
            Buffer::dense([batch, segments, u], Mode::Batched, vec![3.0, 5.0]),
            Buffer::dense([1, segments, u], Mode::Shared, vec![7.0]),
        ],
        outputs: vec![
            Buffer::dense([batch, segments, u], Mode::Batched, vec![0.0; batch]),
            Buffer::dense([batch, segments, u], Mode::Batched, vec![0.0; batch]),
        ],
        ops: vec![
            // out0 += 1 · w · x
            Op {
                buffers: vec![0, 1, 3],
                paths: paths(1.0),
            },
            // out1 += 10 · u · x
            Op {
                buffers: vec![2, 1, 4],
                paths: paths(10.0),
            },
            // out0 += 100 · x, a binary operation, so its paths name two buffers
            Op {
                buffers: vec![1, 3],
                paths: vec![PathSpec {
                    segments: vec![0, 0],
                    coefficient: 100.0,
                }],
            },
        ],
        expected: vec![],
    };
    let plan = case.compile().expect("the case compiles");
    let mut data = case.data();
    let mut scratch = vec![0.0f64; plan.scratch_len(1)];
    case.execute(&plan, &mut data, &mut scratch, Exec::serial(), true)
        .expect("the case executes");
    assert_eq!(
        data[case.inputs.len()],
        vec![1.0 * 2.0 * 3.0 + 100.0 * 3.0, 1.0 * 2.0 * 5.0 + 100.0 * 5.0]
    );
    assert_eq!(
        data[case.inputs.len() + 1],
        vec![10.0 * 7.0 * 3.0, 10.0 * 7.0 * 5.0]
    );
}

#[test]
fn an_operation_may_not_hold_more_operands_than_the_replay_tables() {
    // `MAX_INPUTS` bounds the replay's stack arrays per *operand*, not per distinct
    // buffer, so one input repeated past the capacity has to be rejected.
    let mut case = weight_times_input(2, 1, 1, Mode::Batched, 2);
    let mut operands = vec![1u32; strided_segpoly::MAX_INPUTS + 1];
    operands.push(2);
    case.ops = vec![Op {
        buffers: operands,
        paths: vec![PathSpec {
            segments: vec![0; strided_segpoly::MAX_INPUTS + 2],
            coefficient: 1.0,
        }],
    }];
    let error = case.compile().expect_err("oversized arity is rejected");
    assert!(
        matches!(error, SegPolyError::TooManyOperands { .. }),
        "{error:?}"
    );
}

#[test]
fn a_reversed_batch_stride_is_a_representable_layout() {
    // A negative batch stride reaches as far as a positive one; a view with the
    // matching offset is valid, so compile must not call it an overflow.
    let mut case = weight_times_input(2, 1, 1, Mode::Batched, 2);
    case.inputs[0].data = vec![2.0];
    case.inputs[1] = Buffer::strided([2, 1, 1], [-1, 2, 1], 1, Mode::Batched, vec![4.0, 9.0]);
    let plan = case.compile().expect("a reversed batch stride compiles");
    let mut data = case.data();
    let mut scratch = vec![0.0f64; plan.scratch_len(1)];
    case.execute(&plan, &mut data, &mut scratch, Exec::serial(), true)
        .expect("the case executes");
    // Row 0 is storage[1] and row 1 is storage[0]; the path's coefficient is 0.5.
    assert_eq!(
        data[case.inputs.len()],
        vec![0.5 * 2.0 * 9.0, 0.5 * 2.0 * 4.0]
    );
}

#[test]
fn a_segment_axis_whose_table_cannot_be_allocated_is_a_typed_error() {
    // A zero-length lane axis makes the element count zero while the segment axis
    // is unbounded, so the per-segment table's allocation is the only thing that
    // notices. It must be an error, not a capacity-overflow abort.
    let case = Case {
        batch: 1,
        inputs: vec![Buffer::strided(
            [1, 1, 1],
            [0, 0, 0],
            0,
            Mode::Shared,
            vec![1.0],
        )],
        outputs: vec![Buffer::strided(
            [1, usize::MAX, 0],
            [0, 0, 0],
            0,
            Mode::Shared,
            vec![],
        )],
        ops: vec![Op {
            buffers: vec![0, 1],
            paths: vec![],
        }],
        expected: vec![],
    };
    let error = case
        .compile()
        .expect_err("an unallocatable table is rejected");
    assert!(
        matches!(
            error,
            SegPolyError::Strided(strided_basic::StridedError::OffsetOverflow)
                | SegPolyError::LaneExtent { .. }
                | SegPolyError::OutputLaneBroadcast { .. }
        ),
        "{error:?}"
    );
}

#[test]
fn an_f32_plan_rounds_the_coefficient_at_compile_time() {
    // 2^24 + 1 is not representable in f32: rounding it once at compile time is
    // visible, and it is what keeps the replay a single multiply-add.
    let coefficient = 16_777_217.0f64;
    let dims = [1usize, 1, 1];
    let strides = [1isize, 1, 1];
    let input = [2.0f32];
    let mut output = [0.0f32];
    let operations = [strided_segpoly::Operation {
        buffers: &[0, 1],
        path_start: 0,
        num_paths: 1,
    }];
    let segments = [0u32; 2];
    let paths = [strided_segpoly::Path {
        segments: &segments,
        coefficient,
    }];
    let batched = [IndexMode::Batched];
    let indexes: [&[IndexMode<'_>]; 2] = [&batched, &batched];
    let spec = strided_segpoly::Spec {
        num_batch_axes: 1,
        operations: &operations,
        paths: &paths,
        indexes: &indexes,
    };
    let layouts = [strided_segpoly::Layout {
        dims: &dims,
        strides: &strides,
    }; 2];
    let (inputs, outputs) = layouts.split_at(1);
    let plan = SegPolyPlan::<f32>::compile(&spec, inputs, outputs).expect("an f32 plan compiles");
    let mut scratch = vec![0.0f32; plan.scratch_len(1)];
    let mut out_view =
        RawStridedMut::new(&mut output, &dims, &strides, 0).expect("the output view");
    plan.execute_into(
        std::slice::from_mut(&mut out_view),
        &[RawStridedRef::new(&input, &dims, &strides, 0).expect("the input view")],
        &indexes,
        &mut scratch,
        Exec::serial(),
    )
    .expect("the f32 plan executes");
    assert_eq!(output, [16_777_216.0f32 * 2.0]);
}

#[test]
fn a_unit_lane_input_broadcasts_whatever_its_lane_stride() {
    // A lane extent of one broadcasts over the plan's `u` lanes. The stride of a
    // length-1 axis is arbitrary in a valid view, so the replay must not step by
    // it: stepping by 8 here would read elements 8, 16 and 24 of the backing
    // slice, outside the `[1, 1, 1]` view. The backing slice is long enough that
    // those reads stay inside the allocation and return the sentinel instead of
    // the broadcast value, so the old replay fails deterministically.
    let sentinel = 1.0e3f64;
    let mut input = vec![sentinel; 25];
    input[0] = 8.0;
    let input_dims = [1usize, 1, 1];
    let input_strides = [1isize, 1, 8];
    let out_dims = [1usize, 1, 4];
    let out_strides = [4isize, 4, 1];

    let ids = [0u32, 0];
    let paths = [Path {
        segments: &ids,
        coefficient: 1.0,
    }];
    let operations = [Operation {
        buffers: &[0, 1],
        path_start: 0,
        num_paths: 1,
    }];
    let batched = [IndexMode::Batched];
    let indexes: [&[IndexMode<'_>]; 2] = [&batched, &batched];
    let spec = Spec {
        num_batch_axes: 1,
        operations: &operations,
        paths: &paths,
        indexes: &indexes,
    };
    let plan = SegPolyPlan::<f64>::compile(
        &spec,
        &[Layout {
            dims: &input_dims,
            strides: &input_strides,
        }],
        &[Layout {
            dims: &out_dims,
            strides: &out_strides,
        }],
    )
    .expect("a unit-lane input with a nonzero lane stride is a valid layout");
    assert_eq!(plan.u(), 4);

    let input_view =
        RawStridedRef::new(&input, &input_dims, &input_strides, 0).expect("input view");
    let mut out = vec![-1.0f64; 4];
    let mut scratch = vec![0.0f64; plan.scratch_len(1)];
    {
        let mut out_view =
            RawStridedMut::new(&mut out, &out_dims, &out_strides, 0).expect("output view");
        plan.execute_into(
            std::slice::from_mut(&mut out_view),
            &[input_view],
            &indexes,
            &mut scratch,
            Exec::serial(),
        )
        .expect("executes");
    }
    assert_eq!(out, vec![8.0; 4]);
}
