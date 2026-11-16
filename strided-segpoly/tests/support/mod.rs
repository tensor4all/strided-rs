//! Case builder, the naive reference and view plumbing shared by the
//! `strided-segpoly` integration tests.
//!
//! A [`Case`] owns everything a call needs — buffer shapes, data, index modes and
//! the descriptor — so a test can build the `Spec` ([`Case::with_spec`]) and
//! replay it ([`Case::execute`]) without fighting borrows. The expected values are
//! not produced here: they come from the generated `.npy` fixtures.

pub mod npy;

use strided_basic::{Exec, RawStridedMut, RawStridedRef, RunIndex};
use strided_segpoly::{IndexMode, Layout, Operation, Path, SegPolyError, SegPolyPlan, Spec};

/// A buffer's batch-axis mode, carrying the index data it needs.
#[derive(Clone, Debug)]
pub enum Mode {
    Batched,
    Shared,
    Gather(Vec<i32>),
    Gather64(Vec<i64>),
    Scatter(Vec<i32>),
}

impl Mode {
    /// The executor's index mode for this buffer.
    pub fn of<'a>(&'a self, runs: Option<&'a RunIndex>) -> IndexMode<'a> {
        match self {
            Self::Batched => IndexMode::Batched,
            Self::Shared => IndexMode::Shared,
            Self::Gather(index) => IndexMode::Gather(index),
            Self::Gather64(index) => IndexMode::Gather64(index),
            Self::Scatter(_) => IndexMode::Scatter(runs.expect("scatter mode builds a run index")),
        }
    }
}

/// One `[batch, num_segments, u]` buffer of a case.
#[derive(Clone, Debug)]
pub struct Buffer {
    pub dims: [usize; 3],
    pub strides: [isize; 3],
    pub offset: isize,
    pub mode: Mode,
    /// The scatter topology, for [`Mode::Scatter`].
    pub runs: Option<RunIndex>,
    /// Row-major payload of the whole buffer.
    pub data: Vec<f64>,
}

impl Buffer {
    /// A lane-unit-stride buffer: strides `[segments * u, u, 1]`.
    pub fn dense(dims: [usize; 3], mode: Mode, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), dims[0] * dims[1] * dims[2], "data for {dims:?}");
        let strides = [(dims[1] * dims[2]) as isize, dims[2] as isize, 1];
        Self::strided(dims, strides, 0, mode, data)
    }

    /// A buffer with explicit strides and offset; `data` covers the whole storage.
    pub fn strided(
        dims: [usize; 3],
        strides: [isize; 3],
        offset: isize,
        mode: Mode,
        data: Vec<f64>,
    ) -> Self {
        let runs = match &mode {
            Mode::Scatter(dest) => {
                Some(RunIndex::from_dest(dest, dims[0]).expect("scatter destinations are in range"))
            }
            _ => None,
        };
        Self {
            dims,
            strides,
            offset,
            mode,
            runs,
            data,
        }
    }
}

/// One path of an operation.
#[derive(Clone, Debug)]
pub struct PathSpec {
    /// Segment id per operand, in buffer order.
    pub segments: Vec<u32>,
    pub coefficient: f64,
}

/// One operation: operand buffer ids (last is the output) and its paths.
#[derive(Clone, Debug)]
pub struct Op {
    pub buffers: Vec<u32>,
    pub paths: Vec<PathSpec>,
}

/// A whole call: buffers (inputs first, outputs last), operations and the batch.
#[derive(Clone, Debug)]
pub struct Case {
    pub batch: usize,
    pub inputs: Vec<Buffer>,
    pub outputs: Vec<Buffer>,
    pub ops: Vec<Op>,
    /// The reference result per output buffer, when the case carries one.
    pub expected: Vec<Vec<f64>>,
}

/// A fixture payload: `tests/fixtures/<relative>`, shape-checked.
pub fn npy(relative: &str, shape: &[usize]) -> Vec<f64> {
    npy::read_f64(fixture(relative), shape)
}

/// An index payload: `tests/fixtures/<relative>`, shape-checked.
pub fn npy_i32(relative: &str, shape: &[usize]) -> Vec<i32> {
    npy::read_i32(fixture(relative), shape)
}

fn fixture(relative: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(relative)
}

impl Case {
    /// Inputs' data followed by the outputs' data: the initial values of a call.
    pub fn data(&self) -> Vec<Vec<f64>> {
        self.inputs
            .iter()
            .chain(self.outputs.iter())
            .map(|buffer| buffer.data.clone())
            .collect()
    }

    fn buffers(&self) -> impl Iterator<Item = &Buffer> {
        self.inputs.iter().chain(self.outputs.iter())
    }

    /// Build the descriptor over this case's metadata and hand it to `f`.
    pub fn with_spec<R>(&self, f: impl FnOnce(&Spec<'_>, &[Layout<'_>], &[Layout<'_>]) -> R) -> R {
        let mut operations: Vec<Operation<'_>> = Vec::with_capacity(self.ops.len());
        let mut segments: Vec<Vec<u32>> = Vec::new();
        let mut coefficients: Vec<f64> = Vec::new();
        let mut path_start = 0u32;
        for op in &self.ops {
            for path in &op.paths {
                segments.push(path.segments.clone());
                coefficients.push(path.coefficient);
            }
            operations.push(Operation {
                buffers: &op.buffers,
                path_start,
                num_paths: u32::try_from(op.paths.len()).expect("path count fits u32"),
            });
            path_start += op.paths.len() as u32;
        }
        let paths: Vec<Path<'_>> = segments
            .iter()
            .zip(coefficients.iter())
            .map(|(segments, &coefficient)| Path {
                segments,
                coefficient,
            })
            .collect();
        let modes: Vec<Vec<IndexMode<'_>>> = self
            .buffers()
            .map(|buffer| vec![buffer.mode.of(buffer.runs.as_ref())])
            .collect();
        let indexes: Vec<&[IndexMode<'_>]> = modes.iter().map(|mode| mode.as_slice()).collect();
        let layouts: Vec<Layout<'_>> = self
            .buffers()
            .map(|buffer| Layout {
                dims: &buffer.dims,
                strides: &buffer.strides,
            })
            .collect();
        let spec = Spec {
            num_batch_axes: 1,
            operations: &operations,
            paths: &paths,
            indexes: &indexes,
        };
        f(
            &spec,
            &layouts[..self.inputs.len()],
            &layouts[self.inputs.len()..],
        )
    }

    /// Compile this case, with the default block size.
    pub fn compile(&self) -> Result<SegPolyPlan<f64>, SegPolyError> {
        self.with_spec(SegPolyPlan::compile)
    }

    /// Compile, replay into a fresh copy of [`data`](Self::data) and compare
    /// against the reference with a relative tolerance.
    pub fn check(&self, exec: Exec<'_>, tolerance: f64) {
        let plan = self.compile().expect("the case compiles");
        let mut data = self.data();
        let mut scratch = vec![0.0f64; plan.scratch_len(exec.budget().get())];
        self.execute(&plan, &mut data, &mut scratch, exec, true)
            .expect("the case executes");
        let num_inputs = self.inputs.len();
        for (dense, expected) in self.expected.iter().enumerate() {
            assert_close(
                &format!("output {dense}"),
                &data[num_inputs + dense],
                expected,
                tolerance,
            );
        }
    }

    /// Replay a compiled plan on `data` (inputs then outputs, mutated in place).
    pub fn execute(
        &self,
        plan: &SegPolyPlan<f64>,
        data: &mut [Vec<f64>],
        scratch: &mut [f64],
        exec: Exec<'_>,
        into: bool,
    ) -> Result<(), SegPolyError> {
        let modes: Vec<Vec<IndexMode<'_>>> = self
            .buffers()
            .map(|buffer| vec![buffer.mode.of(buffer.runs.as_ref())])
            .collect();
        let indexes: Vec<&[IndexMode<'_>]> = modes.iter().map(|mode| mode.as_slice()).collect();
        self.execute_with_modes(plan, data, scratch, exec, into, &indexes)
    }

    /// Replay with an explicit mode list, so a test can hand a compiled plan a
    /// call it should reject.
    pub fn execute_with_modes(
        &self,
        plan: &SegPolyPlan<f64>,
        data: &mut [Vec<f64>],
        scratch: &mut [f64],
        exec: Exec<'_>,
        into: bool,
        indexes: &[&[IndexMode<'_>]],
    ) -> Result<(), SegPolyError> {
        let num_inputs = self.inputs.len();
        // Inputs and outputs are disjoint ranges of `data`, so one split serves
        // both view kinds without aliasing them.
        let (input_data, output_data) = data.split_at_mut(num_inputs);
        let inputs: Vec<RawStridedRef<'_, f64>> = input_data
            .iter()
            .zip(self.inputs.iter())
            .map(|(storage, buffer)| {
                RawStridedRef::new(
                    storage,
                    buffer.dims.as_slice(),
                    buffer.strides.as_slice(),
                    buffer.offset,
                )
                .expect("input view in bounds")
            })
            .collect();
        let mut outputs: Vec<RawStridedMut<'_, f64>> = output_data
            .iter_mut()
            .zip(self.outputs.iter())
            .map(|(storage, buffer)| {
                RawStridedMut::new(
                    storage,
                    buffer.dims.as_slice(),
                    buffer.strides.as_slice(),
                    buffer.offset,
                )
                .expect("output view in bounds")
            })
            .collect();
        if into {
            plan.execute_into(&mut outputs, &inputs, indexes, scratch, exec)
        } else {
            plan.execute_acc(&mut outputs, &inputs, indexes, scratch, exec)
        }
    }
}

/// Assert two float vectors agree within a relative tolerance.
pub fn assert_close(what: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let scale = actual.abs().max(expected.abs()).max(1.0);
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance * scale,
            "{what}[{index}]: {actual:.17e} vs {expected:.17e} (rel {:.3e})",
            error / scale
        );
    }
}
