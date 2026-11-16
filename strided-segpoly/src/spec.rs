//! The lowered segmented-polynomial descriptor and the buffer index modes.
//!
//! [`Spec`] is the flat form `cuEquivariance` lowers a segmented polynomial to
//! before it calls its `uniform_1d` kernel (see
//! [tensor4all/strided-rs#479](https://github.com/tensor4all/strided-rs/issues/479)):
//! every buffer is `[batch, num_segments, u]`, every operation is
//! `(buffer ids, path range)`, and every path is `(segment ids, scalar
//! coefficient)`. Building this form from a polynomial is the descriptor crate's
//! job; this crate only executes it.

use strided_basic::RunIndex;

/// A lowered segmented polynomial.
///
/// Buffers are numbered in one list: `indexes.len()` entries, the inputs first
/// and the outputs last, so `num_inputs = indexes.len() - outputs.len()`.
#[derive(Clone, Debug)]
pub struct Spec<'a> {
    /// Batch axes the index modes describe. Exactly one is accepted.
    pub num_batch_axes: usize,
    /// The operations, in polynomial order.
    pub operations: &'a [Operation<'a>],
    /// Every operation's paths, concatenated in operation order.
    pub paths: &'a [Path<'a>],
    /// Per buffer, one index mode per batch axis. Inputs first, outputs last.
    pub indexes: &'a [&'a [IndexMode<'a>]],
}

/// One operation: the buffers it reads and the output it accumulates into.
///
/// `buffers[..len - 1]` are input buffer ids and `buffers[len - 1]` is the
/// output buffer id. `paths[path_start .. path_start + num_paths]` of the
/// owning [`Spec`] belong to this operation.
#[derive(Clone, Copy, Debug)]
pub struct Operation<'a> {
    /// Operand buffer ids; the last is the output.
    pub buffers: &'a [u32],
    /// First path of [`Spec::paths`] this operation owns.
    pub path_start: u32,
    /// Number of paths this operation owns.
    pub num_paths: u32,
}

/// One path: a segment id per operand and a scalar coefficient.
#[derive(Clone, Copy, Debug)]
pub struct Path<'a> {
    /// Segment id per operand, in [`Operation::buffers`] order, so
    /// `segments.len() == buffers.len()` and `segments.last()` selects the
    /// output segment.
    pub segments: &'a [u32],
    /// The path's coefficient, rounded to the plan's element type at compile.
    pub coefficient: f64,
}

/// How one buffer's batch axis is indexed.
///
/// The variants are the lowerings of `cuEquivariance`'s index modes, plus the
/// scattered output's normalised sorted-run form.
#[derive(Clone, Copy, Debug)]
pub enum IndexMode<'a> {
    /// Buffer row `n` is batch point `n`; the buffer's batch extent is the
    /// call's batch.
    Batched,
    /// The buffer has batch extent one. As an input it is read at row 0 for
    /// every batch point; as an output it receives the **sum over the batch**
    /// (the gradient of a shared weight, decision D4).
    Shared,
    /// Input row `idx[n]` is batch point `n`.
    Gather(&'a [i32]),
    /// [`IndexMode::Gather`] over 64-bit indices.
    Gather64(&'a [i64]),
    /// Output row `run(n)` is the sum of every batch point in its run: the
    /// normalised form of a scatter-add output. Build the `RunIndex` once per
    /// topology with [`RunIndex::from_dest`] (`strided-basic`).
    Scatter(&'a RunIndex),
}

/// How a buffer is indexed, without the index data itself.
///
/// A plan compiles against one of these per buffer and a call must supply the
/// same one: the row mapping decides the result, so a call that silently changed
/// it would change the bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModeKind {
    Batched,
    Shared,
    Gather,
    Scatter,
}

impl IndexMode<'_> {
    /// This mode's [`ModeKind`], for the call-versus-compile check.
    pub(crate) fn kind(&self) -> ModeKind {
        match self {
            Self::Batched => ModeKind::Batched,
            Self::Shared => ModeKind::Shared,
            Self::Gather(_) | Self::Gather64(_) => ModeKind::Gather,
            Self::Scatter(_) => ModeKind::Scatter,
        }
    }

    /// Whether this mode indexes an output (a scatter) rather than an input.
    pub(crate) fn is_output_mode(&self) -> bool {
        matches!(self, Self::Scatter(_))
    }

    /// Whether this mode reads an index array.
    pub(crate) fn is_gather(&self) -> bool {
        matches!(self, Self::Gather(_) | Self::Gather64(_))
    }
}

/// A buffer's compiled layout: the extents and strides of its
/// `[batch, num_segments, u]` view.
///
/// Borrowed so a plan compiled from it owns nothing of the caller's metadata.
#[derive(Clone, Copy, Debug)]
pub struct Layout<'a> {
    /// Extents: batch, segments, lanes.
    pub dims: &'a [usize],
    /// Strides: batch, segments, lanes.
    pub strides: &'a [isize],
}
