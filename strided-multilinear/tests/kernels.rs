//! Arithmetic checks that do not need generated fixtures: budget bitwise
//! equality at real fan-out, a finite-difference derivative, and the discrete
//! adjoint identity (which is also how a shared output's batch reduction — the
//! gradient of a shared weight — is checked against `cuEquivariance`, whose naive
//! method clamps instead of summing there).

#[allow(dead_code)] // each test binary uses a different part of the harness
#[path = "support/mod.rs"]
mod support;

use strided_basic::execution::{tree_width, KernelClass};
use strided_basic::{Exec, RawStridedMut, RawStridedRef};
use support::{assert_close, Buffer, Case, Mode, Op, PathSpec};

/// Deterministic payload with mixed magnitudes, so regrouping a sum moves the
/// bits: a plain smooth field would agree bit for bit for the wrong reason.
fn payload(len: usize) -> Vec<f64> {
    (0..len)
        .map(|index| {
            let mantissa = ((index as u64).wrapping_mul(2_654_435_761) % 10_007) as f64 / 10_007.0;
            let scale = 2f64.powi((index % 30) as i32 - 15);
            (mantissa - 0.5) * 2.0 * scale
        })
        .collect()
}

fn pool(threads: usize) -> strided_basic::Pool {
    use threadteam::{DispatchCosts, DispatchKind, HostClass, WidthCosts};
    let cheap = WidthCosts::flat(core::time::Duration::from_nanos(100));
    strided_basic::Pool::builder()
        .threads(threads)
        .dispatch_costs(
            DispatchCosts::for_class(HostClass::Generic)
                .with(DispatchKind::Team, cheap)
                .with(DispatchKind::InPool, cheap)
                .with(DispatchKind::Outside, cheap),
        )
        .build()
        .expect("a test pool")
}

#[test]
fn budget_split_is_bitwise_for_a_batched_and_a_shared_output() {
    // A `Batched` output splits by row; a `Shared` output is one unit of 4096
    // rows, so it splits across fixed block partials instead. Both must agree bit
    // for bit with the serial order, at a size where the gate really opens.
    let (batch, segments, u) = (4096usize, 8usize, 8usize);
    let paths: Vec<PathSpec> = (0..segments)
        .map(|segment| PathSpec {
            segments: vec![segment as u32; 4],
            coefficient: 1.0 + segment as f64,
        })
        .collect();
    for (name, mode, rows) in [
        ("batched", Mode::Batched, batch),
        ("shared", Mode::Shared, 1),
    ] {
        let weights = Buffer::dense([1, segments, u], Mode::Shared, payload(segments * u));
        let input = Buffer::dense(
            [batch, segments, u],
            Mode::Batched,
            payload(batch * segments * u),
        );
        let other = Buffer::dense(
            [batch, segments, u],
            Mode::Batched,
            payload(batch * segments * u),
        );
        let output = Buffer::dense([rows, segments, u], mode, vec![0.0; rows * segments * u]);
        let case = Case {
            batch,
            inputs: vec![weights, input, other],
            outputs: vec![output],
            ops: vec![Op {
                // A degree-2 operation: `w[0] * x[s] * other[s]`, so both inner
                // operands are batched and the weight is shared.
                buffers: vec![0, 1, 2, 3],
                paths: paths.clone(),
            }],
            expected: vec![],
        };
        let plan = case.compile().expect("the case compiles");
        assert!(
            plan.macs() > 1 << 16,
            "{name}: the case has to be worth splitting"
        );

        // Two pool sizes, so the claim covers more than one worker count.
        let mut reference: Option<Vec<Vec<f64>>> = None;
        for threads in [2usize, 4] {
            let pool = pool(threads);
            for budget in [1usize, 2, 4] {
                if budget > threads {
                    break;
                }
                let exec = if budget == 1 {
                    Exec::serial()
                } else {
                    Exec::new(&pool, budget).expect("a token")
                };
                if budget > 1 {
                    assert_eq!(
                        tree_width(&exec, KernelClass::Map, plan.macs()).get(),
                        budget,
                        "{name}: the width gate has to open for the split to exist"
                    );
                }
                let mut data = case.data();
                let mut scratch = vec![0.0f64; plan.scratch_len(budget)];
                case.execute(&plan, &mut data, &mut scratch, exec, true)
                    .expect("the case executes");
                match reference {
                    None => reference = Some(data),
                    Some(ref reference) => assert_eq!(
                        reference, &data,
                        "{name}: {threads} threads at budget {budget} changed the bits"
                    ),
                }
            }
        }
    }
}

#[test]
fn a_lane_reduction_is_bitwise_across_budgets_and_block_partials() {
    // The output has lane extent 1 (a lane reduction) and mode `Shared` (a batch
    // reduction), so one unit is every row: `3 * BLOCK_ROWS + 7` rows, i.e. four
    // block partials that must combine lane-wise before the lanes are summed. The
    // payload stays flag-shaped so the sum is cancellation-sensitive: a different
    // association would move the bits.
    let (batch, segments, u) = (3 * strided_multilinear::BLOCK_ROWS + 7, 4usize, 19usize);
    let paths: Vec<PathSpec> = (0..segments)
        .map(|segment| PathSpec {
            segments: vec![segment as u32, segment as u32, segment as u32],
            coefficient: (1.0 + 0.25 * segment as f64) * if segment % 2 == 0 { 1.0 } else { -1.0 },
        })
        .collect();
    let weights = Buffer::dense([1, segments, u], Mode::Shared, payload(segments * u));
    let input = Buffer::dense(
        [batch, segments, u],
        Mode::Batched,
        payload(batch * segments * u),
    );
    let output = Buffer::dense([1, segments, 1], Mode::Shared, vec![0.0; segments]);
    let case = Case {
        batch,
        inputs: vec![weights, input],
        outputs: vec![output],
        ops: vec![Op {
            buffers: vec![0, 1, 2],
            paths,
        }],
        expected: vec![],
    };
    let plan = case.compile().expect("the case compiles");
    let mut reference: Option<Vec<Vec<f64>>> = None;
    for threads in [2usize, 4] {
        let pool = pool(threads);
        for budget in [1usize, 2, 4] {
            if budget > threads {
                break;
            }
            let exec = if budget == 1 {
                Exec::serial()
            } else {
                Exec::new(&pool, budget).expect("a token")
            };
            let mut data = case.data();
            let mut scratch = vec![0.0f64; plan.scratch_len(budget)];
            case.execute(&plan, &mut data, &mut scratch, exec, true)
                .expect("the case executes");
            match reference {
                None => reference = Some(data),
                Some(ref reference) => assert_eq!(
                    reference, &data,
                    "{threads} threads at budget {budget} changed the bits"
                ),
            }
        }
    }
    // The reduction really did sum something: every destination segment is nonzero
    // and no lane was left unsummed.
    let result = reference.expect("at least one execution");
    let out = &result[case.inputs.len()];
    assert_eq!(out.len(), segments);
    for (segment, value) in out.iter().enumerate() {
        assert!(
            value.is_finite() && *value != 0.0,
            "segment {segment} did not accumulate: {value}"
        );
    }
}

/// `y[b, s, l] = c_s · w[0, s, l] · x[b, s, l] · x[b, s, l]`, whose derivative in
/// `x[b, s, l]` is `2 · c_s · w[0, s, l] · x[b, s, l]`.
fn squared_case(batch: usize, segments: usize, u: usize) -> Case {
    let paths: Vec<PathSpec> = (0..segments)
        .map(|segment| PathSpec {
            segments: vec![
                segment as u32,
                segment as u32,
                segment as u32,
                segment as u32,
            ],
            coefficient: 1.0 + 0.25 * segment as f64,
        })
        .collect();
    let weights = Buffer::dense([1, segments, u], Mode::Shared, payload(segments * u));
    let input = Buffer::dense(
        [batch, segments, u],
        Mode::Batched,
        payload(batch * segments * u),
    );
    let output = Buffer::dense(
        [batch, segments, u],
        Mode::Batched,
        vec![0.0; batch * segments * u],
    );
    Case {
        batch,
        inputs: vec![weights, input],
        outputs: vec![output],
        ops: vec![Op {
            buffers: vec![0, 1, 1, 2],
            paths,
        }],
        expected: vec![],
    }
}

#[test]
fn finite_differences_match_the_analytic_derivative() {
    let (batch, segments, u) = (3usize, 4usize, 2usize);
    let case = squared_case(batch, segments, u);
    let plan = case.compile().expect("the case compiles");

    let evaluate = |data: &mut [Vec<f64>]| -> Vec<f64> {
        let mut scratch = vec![0.0f64; plan.scratch_len(1)];
        case.execute(&plan, data, &mut scratch, Exec::serial(), true)
            .expect("the case executes");
        data[case.inputs.len()].clone()
    };

    let step = 1e-6;
    let base = case.data();
    for element in 0..batch * segments * u {
        let mut plus = base.clone();
        let mut minus = base.clone();
        plus[1][element] += step;
        minus[1][element] -= step;
        let derivative =
            (evaluate(&mut plus)[element] - evaluate(&mut minus)[element]) / (2.0 * step);
        // Invert the payload to place the element on the `[b, s, l]` grid: the
        // buffers are lane-unit-stride, so the flat order is already that grid.
        let segment = (element / u) % segments;
        let analytic = 2.0
            * (1.0 + 0.25 * segment as f64)
            * case.inputs[0].data[segment * u + element % u]
            * case.inputs[1].data[element];
        assert_close(
            &format!("d/dx[{element}]"),
            &[derivative],
            &[analytic],
            1e-6,
        );
    }
}

/// The forward polynomial of the adjoint test: `y[b, s, l] = c_s · w[0, s, l] ·
/// x[b, s, l]`, with `w` shared across the batch.
fn adjoint_cases(batch: usize, segments: usize, u: usize) -> (Case, Case) {
    let paths: Vec<PathSpec> = (0..segments)
        .map(|segment| PathSpec {
            segments: vec![segment as u32, segment as u32, segment as u32],
            coefficient: 0.5 + segment as f64,
        })
        .collect();
    let weights = Buffer::dense([1, segments, u], Mode::Shared, payload(segments * u));
    let input = Buffer::dense(
        [batch, segments, u],
        Mode::Batched,
        payload(batch * segments * u),
    );
    let output = Buffer::dense(
        [batch, segments, u],
        Mode::Batched,
        vec![0.0; batch * segments * u],
    );
    let forward = Case {
        batch,
        inputs: vec![weights.clone(), input.clone()],
        outputs: vec![output.clone()],
        ops: vec![Op {
            buffers: vec![0, 1, 2],
            paths: paths.clone(),
        }],
        expected: vec![],
    };
    // Reverse mode: `x̄[b, s] += c_s · ȳ[b, s] · w[0, s]` is batched, and
    // `w̄[0, s] += Σ_b c_s · ȳ[b, s] · x[b, s]` is the batch reduction a shared
    // output performs.
    let cotangent = Buffer::dense(
        [batch, segments, u],
        Mode::Batched,
        payload(batch * segments * u),
    );
    let x_grad = Buffer::dense(
        [batch, segments, u],
        Mode::Batched,
        vec![0.0; batch * segments * u],
    );
    let w_grad = Buffer::dense([1, segments, u], Mode::Shared, vec![0.0; segments * u]);
    let reverse = Case {
        batch,
        // (ȳ, w, x)
        inputs: vec![cotangent, weights, input],
        outputs: vec![x_grad, w_grad],
        ops: vec![
            // x̄ += c · ȳ · w
            Op {
                buffers: vec![0, 1, 3],
                paths: paths.clone(),
            },
            // w̄ += c · ȳ · x, summed over the batch
            Op {
                buffers: vec![0, 2, 4],
                paths: paths
                    .iter()
                    .map(|path| PathSpec {
                        // (ȳ, x, w̄) all sit on the same segment.
                        segments: vec![path.segments[1]; 3],
                        coefficient: path.coefficient,
                    })
                    .collect(),
            },
        ],
        expected: vec![],
    };
    (forward, reverse)
}

#[test]
fn the_transpose_is_the_discrete_adjoint() {
    // ⟨ȳ, A(x, w)⟩ must equal ⟨x̄, x⟩ + ⟨w̄, w⟩, which pins the shared output's
    // batch reduction: a transpose that dropped or reordered it would not match.
    let (batch, segments, u) = (7usize, 3usize, 5usize);
    let (forward, reverse) = adjoint_cases(batch, segments, u);

    let mut forward_data = forward.data();
    let forward_plan = forward.compile().expect("the case compiles");
    let mut scratch = vec![0.0f64; forward_plan.scratch_len(1)];
    forward
        .execute(
            &forward_plan,
            &mut forward_data,
            &mut scratch,
            Exec::serial(),
            true,
        )
        .expect("the case executes");
    let y = &forward_data[2];

    let mut reverse_data = reverse.data();
    let reverse_plan = reverse.compile().expect("the case compiles");
    let mut scratch = vec![0.0f64; reverse_plan.scratch_len(1)];
    reverse
        .execute(
            &reverse_plan,
            &mut reverse_data,
            &mut scratch,
            Exec::serial(),
            true,
        )
        .expect("the case executes");
    let (x_grad, w_grad) = (&reverse_data[3], &reverse_data[4]);

    let cotangent = &reverse.inputs[0].data;
    let dot = |left: &[f64], right: &[f64]| -> f64 {
        left.iter().zip(right.iter()).map(|(l, r)| l * r).sum()
    };
    let value = dot(cotangent, y);
    // The forward map is bilinear, so each input's adjoint identity holds on its
    // own: `⟨ȳ, A(x, w)⟩` must equal `⟨x̄, x⟩` and, separately, `⟨w̄, w⟩`.
    assert_close(
        "x adjoint",
        &[value],
        &[dot(x_grad, &forward.inputs[1].data)],
        1e-10,
    );
    assert_close(
        "w adjoint (shared output)",
        &[value],
        &[dot(w_grad, &forward.inputs[0].data)],
        1e-10,
    );
}

#[test]
fn a_plan_rejects_an_inconsistent_call_without_panicking() {
    let case = squared_case(2, 2, 2);
    let plan = case.compile().expect("the case compiles");
    let mut data = case.data();
    // A gather index on an input the plan compiled as batched.
    let index = [0i32, 1];
    let gather = [strided_multilinear::IndexMode::Gather(&index)];
    let shared = [strided_multilinear::IndexMode::Shared];
    let batched = [strided_multilinear::IndexMode::Batched];
    let indexes: [&[strided_multilinear::IndexMode<'_>]; 3] = [&shared, &gather, &batched];
    let (input_data, output_data) = data.split_at_mut(2);
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
    let mut outputs = [RawStridedMut::new(
        &mut output_data[0],
        case.outputs[0].dims.as_slice(),
        case.outputs[0].strides.as_slice(),
        case.outputs[0].offset,
    )
    .unwrap()];
    let mut scratch = vec![0.0f64; plan.scratch_len(1)];
    let error = plan
        .execute_into(
            &mut outputs,
            &inputs,
            &indexes,
            &mut scratch,
            Exec::serial(),
        )
        .expect_err("an input mode that disagrees with the compile is rejected");
    assert!(
        matches!(
            error,
            strided_multilinear::SegPolyError::Strided(
                strided_basic::StridedError::PlanLayoutMismatch
            ) | strided_multilinear::SegPolyError::IndexModeNotAllowed { .. }
        ),
        "{error:?}"
    );
}

#[test]
fn a_scatter_call_must_carry_the_compiled_run_order() {
    // `[0,1,0,1]` and `[1,0,1,0]` give the same windows -- two runs of two rows --
    // and a different row order, which is a different sum. The plan traverses the
    // order it compiled, so the second call is rejected before it writes anything.
    let dest_a = [0i32, 1, 0, 1];
    let dest_b = [1i32, 0, 1, 0];
    let runs_a = strided_basic::RunIndex::from_dest(&dest_a, 2).expect("a run index");
    let runs_b = strided_basic::RunIndex::from_dest(&dest_b, 2).expect("a run index");
    assert_eq!(runs_a.ptr(), runs_b.ptr(), "the two windows have to agree");
    assert_ne!(
        runs_a.perm(),
        runs_b.perm(),
        "and the orders have to differ"
    );

    let (batch, segments, u) = (4usize, 2usize, 2usize);
    let x = Buffer::dense(
        [batch, segments, u],
        Mode::Batched,
        payload(batch * segments * u),
    );
    let m = Buffer::dense(
        [2, segments, u],
        Mode::Gather(dest_a.to_vec()),
        payload(2 * segments * u),
    );
    let gx = Buffer::dense(
        [2, segments, u],
        Mode::Scatter(dest_a.to_vec()),
        vec![0.0; 2 * segments * u],
    );
    let paths: Vec<PathSpec> = (0..segments)
        .map(|segment| PathSpec {
            segments: vec![segment as u32; 3],
            coefficient: 0.25 + segment as f64,
        })
        .collect();
    let case = Case {
        batch,
        inputs: vec![x, m],
        outputs: vec![gx],
        ops: vec![Op {
            buffers: vec![0, 1, 2],
            paths,
        }],
        expected: vec![],
    };
    let plan = case.compile().expect("the case compiles");
    let mut data = case.data();
    let sentinel = data[case.inputs.len()..].to_vec();
    let mut scratch = vec![0.0f64; plan.scratch_len(1)];
    let modes: Vec<Vec<strided_multilinear::IndexMode<'_>>> = vec![
        vec![strided_multilinear::IndexMode::Batched],
        vec![strided_multilinear::IndexMode::Gather(&dest_a)],
        vec![strided_multilinear::IndexMode::Scatter(&runs_b)],
    ];
    let indexes: Vec<&[strided_multilinear::IndexMode<'_>]> =
        modes.iter().map(|mode| mode.as_slice()).collect();
    for into in [true, false] {
        let error = case
            .execute_with_modes(
                &plan,
                &mut data,
                &mut scratch,
                Exec::serial(),
                into,
                &indexes,
            )
            .expect_err("a different run order is refused");
        assert!(
            matches!(
                error,
                strided_multilinear::SegPolyError::Strided(
                    strided_basic::StridedError::PlanLayoutMismatch
                )
            ),
            "into={into}: {error:?}"
        );
    }
    assert_eq!(
        &data[case.inputs.len()..],
        sentinel.as_slice(),
        "the refused calls must not have written"
    );
    // The same call with the compiled order still runs, so the guard is about the
    // order and not about rejecting the topology.
    case.execute(&plan, &mut data, &mut scratch, Exec::serial(), true)
        .expect("the compiled order is accepted");
    assert_ne!(
        &data[case.inputs.len()..],
        sentinel.as_slice(),
        "and it writes"
    );
}

#[test]
fn a_zero_batch_call_is_a_no_op() {
    // A zero batch must return `Ok` and leave the destination alone rather than
    // slice on a zero extent or write a spurious zero.
    let case = squared_case(0, 2, 3);
    let plan = case.compile().expect("a zero batch compiles");
    assert_eq!(plan.batch(), 0);
    let mut data = case.data();
    let sentinel: Vec<f64> = data[case.inputs.len()].clone();
    let mut scratch = vec![0.0f64; plan.scratch_len(1)];
    case.execute(&plan, &mut data, &mut scratch, Exec::serial(), true)
        .expect("a zero batch executes");
    assert_eq!(data[case.inputs.len()], sentinel, "nothing to write");
}

/// A VJP-shaped multi-output case: two scattered outputs sharing one run
/// topology and one batched output, all computed from the same inputs. The
/// split cases form the same three outputs one plan at a time, so the fused
/// plan can be compared with them bit for bit.
///
/// `strided` puts every lane at stride 2, so the inputs are packed into the
/// child's tile and the two accumulating outputs are not `Direct`: one child then
/// holds accumulators, a streamed row and a pack tile at once, and a long run adds
/// the block partials on top of that (#498's accumulator layout).
fn fused_case(
    batch: usize,
    sources: usize,
    segments: usize,
    u: usize,
    strided: bool,
) -> (Case, Vec<Case>) {
    let dest: Vec<i32> = (0..batch).map(|row| ((row * 7) % sources) as i32).collect();
    // Lane stride 2 with every other element unused, or a dense buffer.
    let lane_buffer = |dims: [usize; 3], mode: Mode| {
        let count = dims[0] * dims[1] * dims[2];
        if strided {
            Buffer::strided(
                dims,
                [(dims[1] * dims[2] * 2) as isize, (dims[2] * 2) as isize, 2],
                0,
                mode,
                payload(count * 2),
            )
        } else {
            Buffer::dense(dims, mode, payload(count))
        }
    };
    let x = lane_buffer([batch, segments, u], Mode::Batched);
    let w = Buffer::dense(
        [batch, segments, u],
        Mode::Batched,
        payload(batch * segments * u),
    );
    let m = Buffer::dense(
        [sources, segments, u],
        Mode::Gather(dest.clone()),
        payload(sources * segments * u),
    );
    let scattered = |seed: f64| {
        if strided {
            Buffer::strided(
                [sources, segments, u],
                [(segments * u * 2) as isize, (u * 2) as isize, 2],
                0,
                Mode::Scatter(dest.clone()),
                vec![seed; sources * segments * u * 2],
            )
        } else {
            Buffer::dense(
                [sources, segments, u],
                Mode::Scatter(dest.clone()),
                vec![seed; sources * segments * u],
            )
        }
    };
    let gx1 = scattered(0.0);
    let gx2 = scattered(0.0);
    let gw = Buffer::dense(
        [batch, segments, u],
        Mode::Batched,
        vec![0.0; batch * segments * u],
    );
    // The lane-reducing output (`[batch, segments, 1]`): its block is summed over
    // the lanes on the way out, so it exercises the streamed reduce-lane path.
    let gy = Buffer::dense(
        [batch, segments, 1],
        Mode::Batched,
        vec![0.0; batch * segments],
    );
    let paths = |arity: usize, coefficient: f64| -> Vec<PathSpec> {
        (0..segments)
            .map(|segment| PathSpec {
                segments: vec![segment as u32; arity],
                coefficient: coefficient + segment as f64 * 0.25,
            })
            .collect()
    };
    let x_paths = paths(4, 0.75);
    let w_paths = paths(3, -0.5);
    let fused = Case {
        batch,
        inputs: vec![x.clone(), w.clone(), m.clone()],
        outputs: vec![gx1.clone(), gx2.clone(), gw.clone(), gy.clone()],
        ops: vec![
            // gx1 += c · x · w · m, and the same into gx2 from the same inputs.
            Op {
                buffers: vec![0, 1, 2, 3],
                paths: x_paths.clone(),
            },
            Op {
                buffers: vec![0, 1, 2, 4],
                paths: x_paths,
            },
            // gw += c · x · m
            Op {
                buffers: vec![0, 2, 5],
                paths: w_paths.clone(),
            },
            // gy += c · x · m, summed over the lanes (`[batch, segments, 1]`).
            Op {
                buffers: vec![0, 2, 6],
                paths: w_paths.clone(),
            },
        ],
        expected: vec![],
    };
    let split = [
        Case {
            batch,
            inputs: vec![x.clone(), w.clone(), m.clone()],
            outputs: vec![gx1],
            ops: vec![Op {
                buffers: vec![0, 1, 2, 3],
                paths: paths(4, 0.75),
            }],
            expected: vec![],
        },
        Case {
            batch,
            inputs: vec![x.clone(), w.clone(), m.clone()],
            outputs: vec![gx2],
            ops: vec![Op {
                buffers: vec![0, 1, 2, 3],
                paths: paths(4, 0.75),
            }],
            expected: vec![],
        },
        Case {
            batch,
            inputs: vec![x.clone(), m.clone()],
            outputs: vec![gw],
            ops: vec![Op {
                buffers: vec![0, 1, 2],
                paths: w_paths.clone(),
            }],
            expected: vec![],
        },
        Case {
            batch,
            inputs: vec![x, m],
            outputs: vec![gy],
            ops: vec![Op {
                buffers: vec![0, 1, 2],
                paths: w_paths,
            }],
            expected: vec![],
        },
    ];
    (fused, split.to_vec())
}

/// `assert_eq!` on floats is numerical equality (`-0.0 == 0.0`, `NaN != NaN`); the
/// claim in these tests is bit identity, so compare the bit patterns.
fn assert_same_bits(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: lengths differ");
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{what}: element {index} is {actual:?}, not {expected:?}"
        );
    }
}

/// Run one case with a fresh copy of its data, returning the output vectors.
fn run_outputs(
    case: &Case,
    plan: &strided_multilinear::SegPolyPlan<f64>,
    exec: Exec<'_>,
    into: bool,
) -> Vec<Vec<f64>> {
    let mut data = case.data();
    let mut scratch = vec![0.0f64; plan.scratch_len(exec.budget().get())];
    case.execute(plan, &mut data, &mut scratch, exec, into)
        .expect("the case executes");
    data.split_off(case.inputs.len())
}

fn assert_fused_matches_split(
    batch: usize,
    sources: usize,
    segments: usize,
    u: usize,
    strided: bool,
) {
    let (fused, split) = fused_case(batch, sources, segments, u, strided);
    let fused_plan = fused.compile().expect("the fused case compiles");
    let split_plans: Vec<_> = split
        .iter()
        .map(|case| case.compile().expect("a split case compiles"))
        .collect();
    for into in [true, false] {
        let mut fused_case = fused.clone();
        let mut split_cases = split.clone();
        if !into {
            // A non-zero seed makes `execute_acc` read every destination, which is
            // where the Direct/Child/Block-0/Block>0 seeding rules differ.
            for (dense, output) in fused_case.outputs.iter_mut().enumerate() {
                output.data = payload(output.data.len());
                split_cases[dense].outputs[0].data = output.data.clone();
            }
        }
        let expected = run_outputs(&fused_case, &fused_plan, Exec::serial(), into);
        let actual: Vec<Vec<f64>> = split_cases
            .iter()
            .zip(split_plans.iter())
            .map(|(case, plan)| run_outputs(case, plan, Exec::serial(), into)[0].clone())
            .collect();
        assert_eq!(actual.len(), expected.len());
        for (dense, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_same_bits(
                actual,
                expected,
                &format!(
                    "{batch}x{sources} seg={segments} u={u} output {dense} into={into}: \
                     the fused plan must keep every output's bits"
                ),
            );
        }
    }
}

#[test]
fn a_fused_plan_matches_the_split_plans_bit_for_bit() {
    assert_fused_matches_split(5, 3, 2, 3, false);
    // One run longer than one block, so the scattered accumulators split into
    // block partials and the combine runs for both of them.
    assert_fused_matches_split(300, 1, 2, 2, false);
    // The same, with packed lane-strided inputs and non-contiguous accumulators;
    // 300 rows put those in a block slot, 5 rows in a `Child` slot.
    assert_fused_matches_split(300, 1, 2, 2, true);
    assert_fused_matches_split(5, 3, 2, 3, true);
}

#[test]
fn a_fused_plan_is_budget_and_thread_invariant() {
    // Two shapes: 300 rows in one run splits into two block partials (two tasks),
    // and 600 rows over two sources makes two 300-row runs, i.e. four block tasks.
    // How many of those the executor actually forks is the host's cost model and
    // the pool it was handed, not the plan's (a baseline aarch64 host narrows this
    // plan's 9 600 macs to two children), so the test does not assert a width: the
    // plan's own task count is pinned in `plan::tests`, and the bits below must not
    // depend on the width either way.
    for (batch, sources) in [(300usize, 1usize), (600, 2)] {
        let (fused, _) = fused_case(batch, sources, 2, 2, false);
        let plan = fused.compile().expect("the fused case compiles");
        let threads = 4usize;
        let pool = pool(threads);
        for into in [true, false] {
            let mut case = fused.clone();
            if !into {
                // A non-zero seed makes `execute_acc` read every destination, so a
                // wrong seed for a later block or a child region cannot hide.
                for output in case.outputs.iter_mut() {
                    output.data = payload(output.data.len());
                }
            }
            let reference = run_outputs(&case, &plan, Exec::serial(), into);
            for budget in [1usize, 2, 4] {
                let exec = if budget == 1 {
                    Exec::serial()
                } else {
                    Exec::new(&pool, budget).expect("a token")
                };
                let actual = run_outputs(&case, &plan, exec, into);
                assert_same_bits(
                    &actual.concat(),
                    &reference.concat(),
                    &format!(
                        "{batch}x{sources} {threads} threads at budget {budget} into={into}: \
                         the fused bits moved"
                    ),
                );
            }
        }
    }
}

#[test]
fn scatter_outputs_with_different_topologies_keep_their_own_order() {
    // The two scattered outputs have different run indices, so they cannot share
    // one traversal; each keeps its own run order and the batched output is
    // written exactly once (by the first group).
    let (batch, sources, segments, u) = (7usize, 3usize, 2usize, 2usize);
    let dest_a: Vec<i32> = (0..batch).map(|row| (row % sources) as i32).collect();
    let dest_b: Vec<i32> = (0..batch)
        .map(|row| ((row * 2 + 1) % sources) as i32)
        .collect();
    let x = Buffer::dense(
        [batch, segments, u],
        Mode::Batched,
        payload(batch * segments * u),
    );
    let m = Buffer::dense(
        [sources, segments, u],
        Mode::Gather(dest_a.clone()),
        payload(sources * segments * u),
    );
    let paths = || -> Vec<PathSpec> {
        (0..segments)
            .map(|segment| PathSpec {
                segments: vec![segment as u32; 3],
                coefficient: 0.5 + segment as f64,
            })
            .collect()
    };
    let outputs = || {
        [
            Buffer::dense(
                [sources, segments, u],
                Mode::Scatter(dest_a.clone()),
                vec![0.0; sources * segments * u],
            ),
            Buffer::dense(
                [sources, segments, u],
                Mode::Scatter(dest_b.clone()),
                vec![0.0; sources * segments * u],
            ),
            Buffer::dense(
                [batch, segments, u],
                Mode::Batched,
                vec![0.0; batch * segments * u],
            ),
        ]
    };
    let fused = Case {
        batch,
        inputs: vec![x.clone(), m.clone()],
        outputs: outputs().to_vec(),
        ops: vec![
            Op {
                buffers: vec![0, 1, 2],
                paths: paths(),
            },
            Op {
                buffers: vec![0, 1, 3],
                paths: paths(),
            },
            // The batched output reads the same two inputs.
            Op {
                buffers: vec![0, 1, 4],
                paths: paths(),
            },
        ],
        expected: vec![],
    };
    let split = [
        Case {
            batch,
            inputs: vec![x.clone(), m.clone()],
            outputs: vec![outputs()[0].clone()],
            ops: vec![Op {
                buffers: vec![0, 1, 2],
                paths: paths(),
            }],
            expected: vec![],
        },
        Case {
            batch,
            inputs: vec![x.clone(), m.clone()],
            outputs: vec![outputs()[1].clone()],
            ops: vec![Op {
                buffers: vec![0, 1, 2],
                paths: paths(),
            }],
            expected: vec![],
        },
        Case {
            batch,
            inputs: vec![x, m],
            outputs: vec![outputs()[2].clone()],
            ops: vec![Op {
                buffers: vec![0, 1, 2],
                paths: paths(),
            }],
            expected: vec![],
        },
    ];
    let fused_plan = fused.compile().expect("the fused case compiles");
    let expected = run_outputs(&fused, &fused_plan, Exec::serial(), true);
    for (dense, case) in split.iter().enumerate() {
        let plan = case.compile().expect("a split case compiles");
        let actual = run_outputs(case, &plan, Exec::serial(), true);
        assert_eq!(
            actual[0], expected[dense],
            "output {dense} must keep its own topology's order"
        );
    }
}
