//! The generated fixture suite (decision D9).
//!
//! [`cases`] is emitted by `tests/fixtures/generate.py` from cuEquivariance at a
//! pinned commit: the descriptor is that library's `uniform_1d` lowering and the
//! expectation is its `naive` method on the same buffers. Nothing here is
//! hand-written, so a passing run is an outside check of both the contract and the
//! arithmetic.
//!
//! The suite covers a lane axis of 4 (`u = 4`, symmetric contractions) and of 1
//! (spherical harmonics), an operation per degree, shared weight inputs, a
//! gather/scatter message-passing case, and the linearised (`jvp`) and transposed
//! (`backward`) descriptors.

#[allow(dead_code)] // each test binary uses a different part of the harness
#[path = "support/mod.rs"]
mod support;

// `cases.rs` imports the harness types it builds, `assert_close` included.

use strided_basic::{Exec, RawStridedMut, RawStridedRef};
use strided_segpoly::{IndexMode, SegPolyError};

include!("fixtures/cases.rs");

/// Relative tolerance against cuEquivariance's `naive` method, whose summation
/// order differs from the plan's fixed one.
const REFERENCE_TOLERANCE: f64 = 1e-12;

/// A pool whose dispatch is cheap enough that the width gate opens on fixtures
/// this small, so "budget invariant" is tested against real fan-out.
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

fn describe(case: &Case) -> String {
    let shapes: Vec<String> = case
        .inputs
        .iter()
        .chain(case.outputs.iter())
        .map(|buffer| format!("{}x{}x{}", buffer.dims[0], buffer.dims[1], buffer.dims[2]))
        .collect();
    shapes.join(" ")
}

#[test]
fn fixtures_match_the_reference() {
    for case in cases() {
        case.check(Exec::serial(), REFERENCE_TOLERANCE);
    }
}

#[test]
fn fixtures_are_budget_invariant() {
    // D8: the bits must not depend on how the work is split.
    let pool = pool(4);
    for case in cases() {
        let plan = case.compile().expect("the case compiles");
        let mut reference: Option<Vec<Vec<f64>>> = None;
        for budget in [1usize, 2, 4] {
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
                    *reference,
                    data,
                    "{}: budget {budget} changed the bits",
                    describe(&case)
                ),
            }
        }
    }
}

#[test]
fn fixtures_accumulate_on_top_of_their_seed() {
    // `execute_acc` keeps the destination as the first term of the sum.
    for case in cases() {
        let plan = case.compile().expect("the case compiles");
        let seeds = case.data();
        let mut data = seeds.clone();
        let mut scratch = vec![0.0f64; plan.scratch_len(1)];
        case.execute(&plan, &mut data, &mut scratch, Exec::serial(), false)
            .expect("the case executes");
        let num_inputs = case.inputs.len();
        for (dense, expected) in case.expected.iter().enumerate() {
            let reference: Vec<f64> = seeds[num_inputs + dense]
                .iter()
                .zip(expected.iter())
                .map(|(seed, value)| seed + value)
                .collect();
            assert_close(
                &format!("{} output {dense} (acc)", describe(&case)),
                &data[num_inputs + dense],
                &reference,
                REFERENCE_TOLERANCE,
            );
        }
    }
}

#[test]
fn fixtures_reject_a_foreign_scatter_topology() {
    // A `Scatter` output must not accept a run index with different row windows:
    // the plan owns the windows, so a mismatch is a typed error rather than a
    // silently different sum.
    for case in cases() {
        let Some(dense) = case
            .outputs
            .iter()
            .position(|output| matches!(output.mode, Mode::Scatter(_)))
        else {
            continue;
        };
        let plan = case.compile().expect("the case compiles");
        // One run holding every row, where the compiled topology has several.
        let other = Buffer::strided(
            case.outputs[dense].dims,
            case.outputs[dense].strides,
            case.outputs[dense].offset,
            Mode::Scatter(vec![0; case.batch]),
            case.outputs[dense].data.clone(),
        );
        let mut buffers: Vec<&Buffer> = case.inputs.iter().collect();
        buffers.extend(case.outputs.iter());
        buffers[case.inputs.len() + dense] = &other;
        let modes: Vec<Vec<IndexMode<'_>>> = buffers
            .iter()
            .map(|buffer| vec![buffer.mode.of(buffer.runs.as_ref())])
            .collect();
        let indexes: Vec<&[IndexMode<'_>]> = modes.iter().map(|mode| mode.as_slice()).collect();

        let mut data = case.data();
        let num_inputs = case.inputs.len();
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
                .expect("in bounds")
            })
            .collect();
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
                .expect("in bounds")
            })
            .collect();
        let mut scratch = vec![0.0f64; plan.scratch_len(1)];
        let error = plan
            .execute_into(
                &mut outputs,
                &inputs,
                &indexes,
                &mut scratch,
                Exec::serial(),
            )
            .expect_err("a different run topology is rejected");
        assert!(
            matches!(
                error,
                SegPolyError::Strided(strided_basic::StridedError::ShapeMismatch(..))
            ),
            "{}: {error:?}",
            describe(&case)
        );
    }
}
