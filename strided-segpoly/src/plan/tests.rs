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
fn rejects_an_output_lane_that_is_neither_one_nor_u() {
    // A lane extent of 1 on an output is a lane reduction (see the test below); a
    // lane extent between 1 and `u` is neither that nor the uniform extent.
    let x = [2, 2, 3];
    let strides = [6, 3, 1];
    let y = [2, 2, 2];
    let error = compile_error(
        &spec(&[], &[], &[BATCHED, BATCHED]),
        &[layout(&x, &strides)],
        &[layout(&y, &strides)],
    );
    assert!(matches!(
        error,
        SegPolyError::OutputLaneBroadcast { lane: 2, u: 3, .. }
    ));
}

/// Lanes in the lane-reduction cases: more than one vector on every host here, so
/// the vector loop *and* its scalar remainder run.
const LANE_U: usize = 19;
/// Three `Batched` buffers, the mode lists for the single-path cases below.
const INDEXES_3: [&[IndexMode<'_>]; 3] = [BATCHED, BATCHED, BATCHED];
/// Batch rows in the lane-reduction cases. Their destinations are `Batched`, so a
/// unit is one row and no block partials arise; this only has to exceed one unit
/// and one block so the row stride and the scheduler's row split are exercised.
/// `a_shared_lane_reduction_combines_block_partials_first` owns the block case.
const LANE_ROWS: usize = 2 * BLOCK_ROWS + 5;

/// `x[b, s, l]` = `c · x · w` summed over the plans paths, then over the lanes:
/// the documented order, in scalar code.
///
/// Lane `l` accumulates its own paths in plan order and only then are the lanes
/// added in increasing order, with the seed as lane 0's initial value. (A unit
/// here is one row, so there is no block combination to model; that order is the
/// subject of the shared case below.) A reference that sums the lanes *inside* a path
/// and then adds path totals is a different association: `[1e16, 1]` and
/// `[-1e16, 1]` give 2 in this order and 0 in that one.
// A reference is a transcription of the order; a struct would only rename the
// parameters.
#[allow(clippy::too_many_arguments)]
fn lane_reference(
    rows: usize,
    segments: usize,
    u: usize,
    span: usize,
    lane_stride: isize,
    x: &[f64],
    x_offset: isize,
    w: &[f64],
    paths: &[[u32; 3]],
    coeffs: &[f64],
    seed: f64,
) -> Vec<f64> {
    let mut out = vec![0.0f64; rows * segments];
    for row in 0..rows {
        let mut lanes = vec![vec![0.0f64; u]; segments];
        for lane in lanes.iter_mut() {
            lane[0] = seed;
        }
        for (path, &coefficient) in paths.iter().zip(coeffs.iter()) {
            let (sx, sw, sy) = (path[0] as usize, path[1] as usize, path[2] as usize);
            for (lane, slot) in lanes[sy].iter_mut().enumerate() {
                let index = row * segments * span
                    + sx * span
                    + (x_offset + lane as isize * lane_stride).unsigned_abs();
                let term = coefficient * x[index] * w[sw * u + lane];
                *slot += term;
            }
        }
        for (segment, lane_sums) in lanes.iter().enumerate() {
            let mut total = lane_sums[0];
            for &value in &lane_sums[1..] {
                total += value;
            }
            out[row * segments + segment] = total;
        }
    }
    out
}

/// A lane-reduced plan and the same sum in the documented order.
///
/// `lane_stride` is the x view's lane stride: 1 reads it in place, anything else
/// makes the replay pack it (0 is a splat, covered separately).
fn lane_reduction_case(lane_stride: isize, budget: usize, seed: f64) -> (Vec<f64>, Vec<f64>) {
    let (rows, segments, u) = (LANE_ROWS, 2usize, LANE_U);
    let span = (u - 1) * lane_stride.unsigned_abs() + 1;
    // Cancellation-sensitive payload: the association is visible in the bits.
    let x: Vec<f64> = (0..rows * segments * span)
        .map(|index| match index % 4 {
            0 => 1e16,
            1 => -1e16,
            2 => 1.0,
            _ => (index % 7) as f64 * 0.5 - 1.0,
        })
        .collect();
    let w: Vec<f64> = (0..segments * u)
        .map(|index| 1.0 + (index % 5) as f64 * 0.25)
        .collect();
    let mut out = vec![seed; rows * segments];
    let x_dims = [rows, segments, u];
    let x_strides = [(segments * span) as isize, span as isize, lane_stride];
    // A negative lane stride walks backwards, so the view starts at the last lane.
    let x_offset = if lane_stride < 0 {
        (u as isize - 1) * -lane_stride
    } else {
        0
    };
    let w_dims = [1usize, segments, u];
    let w_strides = [u as isize, u as isize, 1];
    let out_dims = [rows, segments, 1];
    let out_strides = [segments as isize, 1, 1];
    // (x segment, w segment, out segment); the out segment ids interleave so the
    // compile-time sort has to reorder the accumulation.
    let paths: [[u32; 3]; 4] = [[0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1]];
    let coeffs = [1.0f64, -0.5, 0.25, 2.0];
    let operations = [Operation {
        buffers: &[0, 1, 2],
        path_start: 0,
        num_paths: paths.len() as u32,
    }];
    let path_list: Vec<Path<'_>> = paths
        .iter()
        .zip(coeffs.iter())
        .map(|(segments, &coefficient)| Path {
            segments,
            coefficient,
        })
        .collect();
    let descriptor = spec(&operations, &path_list, &[BATCHED, SHARED, BATCHED]);
    let plan = SegPolyPlan::<f64>::compile(
        &descriptor,
        &[
            Layout {
                dims: &x_dims,
                strides: &x_strides,
            },
            layout(&w_dims, &w_strides),
        ],
        &[layout(&out_dims, &out_strides)],
    )
    .expect("a lane-unit output is a lane reduction");
    let mut scratch = vec![0.0f64; plan.scratch_len(budget)];
    let inputs = [
        RawStridedRef::new(&x, &x_dims, &x_strides, x_offset).expect("x view"),
        RawStridedRef::new(&w, &w_dims, &w_strides, 0).expect("w view"),
    ];
    {
        let mut outputs = [RawStridedMut::new(&mut out, &out_dims, &out_strides, 0).expect("out")];
        // A seed means `execute_acc`; otherwise the destination starts at zero.
        if seed == 0.0 {
            plan.execute_into(
                &mut outputs,
                &inputs,
                descriptor.indexes,
                &mut scratch,
                Exec::serial(),
            )
            .expect("executes");
        } else {
            plan.execute_acc(
                &mut outputs,
                &inputs,
                descriptor.indexes,
                &mut scratch,
                Exec::serial(),
            )
            .expect("executes");
        }
    }
    let expected = lane_reference(
        rows,
        segments,
        u,
        span,
        lane_stride,
        &x,
        x_offset,
        &w,
        &paths,
        &coeffs,
        seed,
    );
    (out, expected)
}

#[test]
fn a_lane_unit_output_sums_its_lanes_in_order() {
    // The vector loop and its remainder both run: `LANE_U` is not a multiple of
    // any supported width.
    let (out, expected) = lane_reduction_case(1, 1, 0.0);
    assert_eq!(
        out, expected,
        "lane reduction must follow the documented order"
    );
}

#[test]
fn a_packed_lane_matches_the_scalar_reference_bit_for_bit() {
    // A lane stride that is neither 0 nor 1 is packed into a tile first. Packing is
    // a copy, so the result must be the same bits the scalar order gives.
    for stride in [1, 2, 3, 5, -1, -2] {
        let (out, expected) = lane_reduction_case(stride, 1, 0.0);
        assert_eq!(
            out, expected,
            "lane stride {stride} did not match the reference"
        );
    }
}

#[test]
fn a_seeded_lane_reduction_adds_the_seed_once() {
    // `execute_acc` keeps the destination and adds this call's sum to it; a lane
    // reduction reads the destination as lane 0's first term, so the seed enters
    // the sum exactly once.
    let seed = 0.75f64;
    for stride in [1, 3] {
        let (out, expected) = lane_reduction_case(stride, 1, seed);
        assert_eq!(
            out, expected,
            "lane stride {stride} seeded reduction differs"
        );
    }
}

/// `y[0, s, l] = Σ_paths c · x[0, s, l] · w[0, s, l]` with x's lane stride given:
/// the three lane classifications (slice, splat, pack) as the plan sees them.
///
/// `x_operands = 4` builds `x · x · x · x · w` instead -- arity six with the
/// output, past the lane kernel's three, so the scalar fallback takes it.
fn single_path_plan(
    u: usize,
    x_lane_stride: isize,
    x_lane_extent: usize,
    x_operands: usize,
) -> (SegPolyPlan<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let segments = 2usize;
    let span = (u - 1) * x_lane_stride.unsigned_abs() + 1;
    let x_len = segments * span;
    let x: Vec<f64> = (0..x_len).map(|index| 1.0 + index as f64).collect();
    let w: Vec<f64> = (0..segments * u)
        .map(|index| 1.0 + index as f64 * 0.5)
        .collect();
    let out = vec![0.0f64; segments * u];
    let x_dims = [1usize, segments, x_lane_extent];
    let x_strides = [
        segments as isize * span as isize,
        span as isize,
        x_lane_stride,
    ];
    let w_dims = [1usize, segments, u];
    let w_strides = [u as isize, u as isize, 1];
    let out_dims = [1usize, segments, u];
    let out_strides = [u as isize, u as isize, 1];
    let buffers: Vec<u32> = if x_operands == 4 {
        vec![0, 0, 0, 0, 1, 2]
    } else {
        vec![0, 1, 2]
    };
    // One segment id per operand, all the same: `Σ c · x · ... · w`.
    let seg_lists: Vec<Vec<u32>> = (0..segments)
        .map(|segment| vec![segment as u32; buffers.len()])
        .collect();
    let paths: Vec<Path<'_>> = seg_lists
        .iter()
        .map(|segments| Path {
            segments,
            coefficient: 1.0,
        })
        .collect();
    let operations = [Operation {
        buffers: &buffers,
        path_start: 0,
        num_paths: paths.len() as u32,
    }];
    let indexes: Vec<&[IndexMode<'_>]> = (0..3).map(|_| BATCHED).collect();
    let plan = SegPolyPlan::<f64>::compile(
        &spec(&operations, &paths, &indexes),
        &[
            Layout {
                dims: &x_dims,
                strides: &x_strides,
            },
            layout(&w_dims, &w_strides),
        ],
        &[layout(&out_dims, &out_strides)],
    )
    .expect("compiles");
    (plan, x, w, out)
}

#[test]
fn a_stride_zero_lane_is_a_broadcast_not_a_pack() {
    // Every lane reads the same element, which is a splat: the replay must read one
    // value, not `u` contiguous lanes (or a packed tile at buffer-space offsets).
    let (u, segments) = (4usize, 2usize);
    let (plan, x, w, mut out) = single_path_plan(u, 0, u, 1);
    let x_dims = [1usize, segments, u];
    let x_strides = [segments as isize, 1, 0];
    let w_dims = [1usize, segments, u];
    let w_strides = [u as isize, u as isize, 1];
    let out_dims = [1usize, segments, u];
    let out_strides = [u as isize, u as isize, 1];
    let x_span = segments;
    let x: Vec<f64> = x[..x_span].to_vec();
    let mut scratch = vec![0.0f64; plan.scratch_len(1)];
    let inputs = [
        RawStridedRef::new(&x, &x_dims, &x_strides, 0).expect("x view"),
        RawStridedRef::new(&w, &w_dims, &w_strides, 0).expect("w view"),
    ];
    {
        let mut outputs = [RawStridedMut::new(&mut out, &out_dims, &out_strides, 0).expect("out")];
        plan.execute_into(
            &mut outputs,
            &inputs,
            &INDEXES_3,
            &mut scratch,
            Exec::serial(),
        )
        .expect("executes");
    }
    let mut expected = Vec::new();
    for segment in 0..segments {
        for lane in 0..u {
            expected.push(x[segment] * w[segment * u + lane]);
        }
    }
    assert_eq!(out, expected, "a stride-zero lane must read one element");
}

#[test]
fn a_unit_lane_plan_keeps_the_buffer_strides() {
    // `u = 1`: no lane reduction, no packing, and the offsets stay the buffer's own
    // segment stride even though the lane stride is not 1.
    let segments = 3usize;
    let x_dims = [1usize, segments, 1];
    let x_strides = [segments as isize * 4, 4, 2];
    let x: Vec<f64> = (0..(segments - 1) * 4 + 1)
        .map(|index| index as f64)
        .collect();
    let w_dims = [1usize, segments, 1];
    let w_strides = [segments as isize, 1, 1];
    let w: Vec<f64> = vec![2.0, 3.0, 5.0];
    let out_dims = [1usize, segments, 1];
    let out_strides = [segments as isize, 1, 1];
    let mut out = vec![0.0f64; segments];
    let seg_lists: Vec<[u32; 3]> = (0..segments).map(|segment| [segment as u32; 3]).collect();
    let paths: Vec<Path<'_>> = seg_lists
        .iter()
        .map(|segments| Path {
            segments,
            coefficient: 1.0,
        })
        .collect();
    let operations = [Operation {
        buffers: &[0, 1, 2],
        path_start: 0,
        num_paths: segments as u32,
    }];
    let plan = SegPolyPlan::<f64>::compile(
        &spec(&operations, &paths, &[BATCHED, BATCHED, BATCHED]),
        &[
            Layout {
                dims: &x_dims,
                strides: &x_strides,
            },
            layout(&w_dims, &w_strides),
        ],
        &[layout(&out_dims, &out_strides)],
    )
    .expect("compiles");
    let mut scratch = vec![0.0f64; plan.scratch_len(1)];
    let inputs = [
        RawStridedRef::new(&x, &x_dims, &x_strides, 0).expect("x view"),
        RawStridedRef::new(&w, &w_dims, &w_strides, 0).expect("w view"),
    ];
    {
        let mut outputs = [RawStridedMut::new(&mut out, &out_dims, &out_strides, 0).expect("out")];
        plan.execute_into(
            &mut outputs,
            &inputs,
            &INDEXES_3,
            &mut scratch,
            Exec::serial(),
        )
        .expect("executes");
    }
    let expected: Vec<f64> = (0..segments).map(|s| x[s * 4] * w[s]).collect();
    assert_eq!(
        out, expected,
        "a unit lane plan must read its buffer strides"
    );
}

#[test]
fn a_deep_operation_reads_the_buffer_offsets() {
    // Five operands, so the scalar fallback -- which walks the buffer it names --
    // takes this path: its offsets must stay in buffer space even though the lane
    // stride is 2.
    let (u, segments) = (3usize, 2usize);
    let (plan, x, _, mut out) = single_path_plan(u, 2, u, 4);
    let span = (u - 1) * 2 + 1;
    let x_dims = [1usize, segments, u];
    let x_strides = [segments as isize * span as isize, span as isize, 2];
    let w_dims = [1usize, segments, u];
    let w_strides = [u as isize, u as isize, 1];
    let out_dims = [1usize, segments, u];
    let out_strides = [u as isize, u as isize, 1];
    let x = &x[..segments * span];
    let w: Vec<f64> = (0..segments * u).map(|index| 1.0 + index as f64).collect();
    let mut scratch = vec![0.0f64; plan.scratch_len(1)];
    let inputs = [
        RawStridedRef::new(x, &x_dims, &x_strides, 0).expect("x view"),
        RawStridedRef::new(&w, &w_dims, &w_strides, 0).expect("w view"),
    ];
    {
        let mut outputs = [RawStridedMut::new(&mut out, &out_dims, &out_strides, 0).expect("out")];
        plan.execute_into(
            &mut outputs,
            &inputs,
            &INDEXES_3,
            &mut scratch,
            Exec::serial(),
        )
        .expect("executes");
    }
    // `x * x * x * x` along the lane stride, then `* w`.
    let mut expected = Vec::new();
    for segment in 0..segments {
        for lane in 0..u {
            let value = x[segment * span + lane * 2];
            expected.push(value * value * value * value * w[segment * u + lane]);
        }
    }
    assert_eq!(out, expected, "the fallback must use buffer-space offsets");
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

/// The same sum with a *lane-reduced* shared output:
/// `y[0, 0, 0] = Σ_lanes Σ_rows x[row, 0, lane]`, block partials combined
/// lane-wise before the lanes are summed.
fn run_shared_lane_reduction(values: &[f64], lanes: usize, block_rows: usize) -> Vec<f64> {
    let rows = values.len() / lanes;
    let input_dims = [rows, 1, lanes];
    let input_strides = [lanes as isize, lanes as isize, 1];
    let out_dims = [1usize, 1, 1];
    let out_strides = [1isize, 1, 1];
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
    let mut out = vec![0.0];
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
fn a_shared_lane_reduction_combines_block_partials_first() {
    // Two blocks whose lane partials cancel: block 1 sums to `[1e16, 1]` per lane
    // and block 2 to `[-1e16, 1]`. Combined lane-wise first, the lanes are
    // `[0, 2]` and the reduction is 2; reducing each block's lanes first gives
    // `1e16 + 1 -> 1e16` and `-1e16 + 1 -> -1e16`, i.e. 0.
    let values = [1e16, 1.0, -1e16, 1.0];
    for block_rows in [1, 2, BLOCK_ROWS] {
        assert_eq!(
            run_shared_lane_reduction(&values, 2, block_rows),
            vec![2.0],
            "block_rows {block_rows}: the lane sum must follow the block partials"
        );
    }
    // The same value has to come out of one block, where no combination happens:
    // the test above is about the order, not about the block count.
    assert_eq!(
        run_shared_lane_reduction(&[1e16, 1.0], 2, BLOCK_ROWS),
        vec![1e16]
    );
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

/// A minimal lane-reducing plan whose x operand has the given lane stride.
fn lane_reduction_plan(lane_stride: isize) -> SegPolyPlan<f64> {
    let u = 5;
    let span = (u as isize - 1) * lane_stride.unsigned_abs() as isize + 1;
    let x_dims = [1usize, 2, u];
    let x_strides = [2 * span, span, lane_stride];
    let w_dims = [1usize, 2, u];
    let w_strides = [u as isize, u as isize, 1];
    let out_dims = [1usize, 2, 1];
    let out_strides = [2isize, 1, 1];
    let first = [0u32; 3];
    let second = [0u32, 1, 1];
    let operations = [Operation {
        buffers: &[0, 1, 2],
        path_start: 0,
        num_paths: 2,
    }];
    let paths = [
        Path {
            segments: &first,
            coefficient: 1.0,
        },
        Path {
            segments: &second,
            coefficient: -2.0,
        },
    ];
    SegPolyPlan::<f64>::compile(
        &spec(&operations, &paths, &[BATCHED, SHARED, BATCHED]),
        &[
            Layout {
                dims: &x_dims,
                strides: &x_strides,
            },
            layout(&w_dims, &w_strides),
        ],
        &[layout(&out_dims, &out_strides)],
    )
    .expect("compiles")
}

#[test]
fn scratch_reserves_a_pack_area_per_worker() {
    // A lane-strided operand is packed into a tile before the vector loop, so the
    // scratch grows with the budget by one tile per packed operand on top of the
    // block partials, and the block partials are a compile-time floor.
    let packed = lane_reduction_plan(3);
    assert!(packed.pack_elems > 0, "a lane stride of 3 has to be packed");
    assert_eq!(
        packed.acc_stride, 0,
        "a batched-only plan has no block accumulator"
    );
    assert_eq!(packed.row_elems, packed.tile_elems);
    for budget in 1..=4 {
        let expected = packed.block_slots * packed.acc_stride
            + budget * (packed.acc_stride + packed.row_elems + packed.pack_elems);
        assert_eq!(
            packed.scratch_len(budget),
            expected,
            "budget {budget}: scratch must cover the blocks and one pack area per worker"
        );
    }
    // An unpacked plan pays nothing for a pack area, and a zero budget is the one
    // worker floor rather than no workers.
    let contiguous = lane_reduction_plan(1);
    assert_eq!(contiguous.pack_elems, 0);
    assert_eq!(contiguous.scratch_len(0), contiguous.scratch_len(1));
}

#[test]
fn one_group_accumulator_covers_every_output_of_one_traversal() {
    // Two scattered outputs with the same topology share one traversal and one
    // accumulator region per block slot, so the block partials are accounted once
    // per group, not once per output.
    let segments = 2usize;
    let u = 3usize;
    let dest = [0i32; 3];
    let runs = RunIndex::from_dest(&dest, 1).unwrap();
    let scatter: &[IndexMode] = &[IndexMode::Scatter(&runs)];
    let strides = [segments as isize * u as isize, u as isize, 1];
    let outputs = [1usize, segments, u];
    let out_strides = [segments as isize * u as isize, u as isize, 1];
    let x_dims = [3usize, segments, u];
    let first = [0u32; 2];
    let operations = [
        Operation {
            buffers: &[0, 1],
            path_start: 0,
            num_paths: 1,
        },
        Operation {
            buffers: &[0, 2],
            path_start: 0,
            num_paths: 1,
        },
    ];
    let paths = [Path {
        segments: &first,
        coefficient: 1.0,
    }];
    let indexes: [&[IndexMode]; 3] = [BATCHED, scatter, scatter];
    let plan = SegPolyPlan::<f64>::compile_with_block_rows(
        &spec(&operations, &paths, &indexes),
        &[layout(&x_dims, &strides)],
        &[
            layout(&outputs, &out_strides),
            layout(&outputs, &out_strides),
        ],
        1,
    )
    .expect("compiles");
    // One run of three rows, one row per block: three block slots, each holding
    // both outputs' `segments × u` accumulators.
    assert_eq!(plan.block_slots, 3);
    assert_eq!(plan.acc_stride, 2 * segments * u);
    assert_eq!(plan.row_elems, 0, "no batched output streams a row");
    assert_eq!(plan.tile_elems, segments * u);
    for budget in 1..=4 {
        assert_eq!(
            plan.scratch_len(budget),
            3 * 2 * segments * u + budget * (2 * segments * u),
            "budget {budget}: the two outputs share the block slots and the child region"
        );
    }
    assert_eq!(plan.groups.len(), 1);
    assert_eq!(plan.groups[0].acc.len(), 2);
    assert!(plan.units.iter().all(|unit| unit.group == 0));
}

#[test]
fn scatter_outputs_with_different_run_orders_get_their_own_group() {
    // The same windows in a different order sum in a different order, so two
    // scattered outputs are only interchangeable when their permutations match
    // too. `[0,1,0,1]` and `[1,0,1,0]` share `ptr` and differ in `perm`.
    let segments = 2usize;
    let u = 2usize;
    let runs_a = RunIndex::from_dest(&[0i32, 1, 0, 1], 2).expect("a run index");
    let runs_b = RunIndex::from_dest(&[1i32, 0, 1, 0], 2).expect("a run index");
    assert_eq!(runs_a.ptr(), runs_b.ptr(), "the windows have to agree");
    assert_ne!(
        runs_a.perm(),
        runs_b.perm(),
        "and the orders have to differ"
    );
    let scatter_a: &[IndexMode] = &[IndexMode::Scatter(&runs_a)];
    let scatter_b: &[IndexMode] = &[IndexMode::Scatter(&runs_b)];
    let strides = [segments as isize * u as isize, u as isize, 1];
    let x_dims = [4usize, segments, u];
    let outputs = [2usize, segments, u];
    let first = [0u32; 2];
    let operations = [
        Operation {
            buffers: &[0, 1],
            path_start: 0,
            num_paths: 1,
        },
        Operation {
            buffers: &[0, 2],
            path_start: 0,
            num_paths: 1,
        },
    ];
    let paths = [Path {
        segments: &first,
        coefficient: 1.0,
    }];
    let indexes: [&[IndexMode]; 3] = [BATCHED, scatter_a, scatter_b];
    let plan = SegPolyPlan::<f64>::compile_with_block_rows(
        &spec(&operations, &paths, &indexes),
        &[layout(&x_dims, &strides)],
        &[layout(&outputs, &strides), layout(&outputs, &strides)],
        4,
    )
    .expect("compiles");
    assert_eq!(plan.groups.len(), 2, "one traversal each, not one shared");
    assert_eq!(plan.acc_stride, segments * u, "one output per group");
    assert_eq!(plan.block_slots, 0, "two rows per run fit one block");
    assert_eq!(plan.units.len(), 4, "two runs per group");
    assert_ne!(plan.out_group[0], plan.out_group[1]);
    assert_eq!(plan.tasks.len(), 4, "one task per unit");
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
