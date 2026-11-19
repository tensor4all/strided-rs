//! Errors a segmented-polynomial plan returns.
//!
//! [`SegPolyError::Strided`] carries the layout, collation and overflow errors
//! the plan reuses from `strided-basic` (`StridedError`); every other variant is
//! a rule of the segmented-polynomial contract itself.

use strided_basic::StridedError;

/// A `strided-multilinear` call whose descriptor, layouts or scratch violate the
/// plan's contract.
#[derive(Debug, thiserror::Error)]
pub enum SegPolyError {
    /// The descriptor declares more than one batch axis, which the first
    /// increment rejects (decision D1 of
    /// [tensor4all/strided-rs#479](https://github.com/tensor4all/strided-rs/issues/479)).
    #[error("{} batch axes declared; the executor accepts exactly one", .axes)]
    MultiBatchAxis { axes: usize },

    /// A buffer's index mode list does not have one entry per batch axis.
    #[error("buffer {buffer} declares {modes} index modes for {axes} batch axes")]
    BufferModeCount {
        buffer: usize,
        modes: usize,
        axes: usize,
    },

    /// A gather index mode is attached to an output buffer, or a scatter mode to
    /// an input buffer (decision D6: inputs are gathered and outputs are
    /// scattered, never the other way around).
    #[error("buffer {buffer}: gather applies to inputs and scatter to outputs")]
    IndexModeNotAllowed { buffer: usize },

    /// A buffer's last axis is neither unit-length nor the plan's uniform lane
    /// extent.
    #[error(
        "buffer {buffer} has lane extent {lane}; the plan's uniform extent is {u} (1 or u accepted)"
    )]
    LaneExtent {
        buffer: usize,
        lane: usize,
        u: usize,
    },

    /// An output buffer's lane extent is neither the plan's uniform extent nor
    /// one. An extent of one is accepted -- it is a *lane reduction*, the row's `u`
    /// lanes summed in increasing order -- but nothing between 1 and `u` is a
    /// destination shape this executor has (decision D5).
    #[error(
        "output buffer {buffer} has lane extent {lane}; the plan's uniform extent is {u} \
         (1 or {u} accepted, 1 being a lane reduction)"
    )]
    OutputLaneBroadcast {
        buffer: usize,
        lane: usize,
        u: usize,
    },

    /// A buffer is never read and never written, which the lowered contract
    /// ("all buffers must be used") does not allow.
    #[error("buffer {buffer} is not used by any operation")]
    UnusedBuffer { buffer: usize },

    /// An operation names a buffer id that does not exist.
    #[error("buffer id {buffer} is out of range")]
    UnknownBuffer { buffer: u32 },

    /// An operation's last operand is an input buffer, or one of its other
    /// operands is an output buffer. The last operand is the operation's output.
    #[error("operation {operation}: buffer {buffer} is not of the expected role")]
    BufferRole { operation: usize, buffer: u32 },

    /// An operation's operand count disagrees with a path's segment-id count.
    #[error(
        "operation {operation} has {operands} operands but its path declares {segments} segments"
    )]
    ArityMismatch {
        operation: usize,
        operands: usize,
        segments: usize,
    },

    /// An operation's path range leaves the descriptor's path list.
    #[error("operation {operation}: {num_paths} paths from {path_start} leave the {paths}-path descriptor")]
    PathRange {
        operation: usize,
        path_start: u32,
        num_paths: u32,
        paths: usize,
    },

    /// A path names a segment id at or past the buffer's segment count.
    #[error("buffer {buffer}: segment {segment} is not below {segments}")]
    SegmentOutOfRange {
        buffer: u32,
        segment: u32,
        segments: usize,
    },

    /// Two non-broadcast batch extents disagree (decision D7).
    #[error("buffer {buffer} has batch extent {extent} but the call's batch is {batch}")]
    BatchExtentMismatch {
        buffer: usize,
        extent: usize,
        batch: usize,
    },

    /// A gather index is negative or at past the gathered buffer's batch extent
    /// (decision D3).
    #[error("buffer {buffer}: index {index} is outside 0..{extent}")]
    IndexOutOfRange {
        buffer: usize,
        index: i64,
        extent: usize,
    },

    /// The plan has more buffers than the replay's stack tables hold
    /// ([`MAX_INPUTS`](crate::MAX_INPUTS) / [`MAX_OUTPUTS`](crate::MAX_OUTPUTS)).
    #[error("{inputs} inputs and {outputs} outputs exceed the executor's stack tables")]
    TooManyBuffers { inputs: usize, outputs: usize },

    /// An operation has more operands than the replay's stack tables hold. The
    /// operand count, not the distinct buffer count, is what bounds them: one
    /// input may appear as many operands.
    #[error("operation {operation}: {operands} operands exceed the executor's {max}-slot tables")]
    TooManyOperands {
        operation: usize,
        operands: usize,
        max: usize,
    },

    /// The caller's scratch cannot hold the plan's block partials.
    #[error("scratch holds {got} elements; {need} needed")]
    ScratchTooSmall { need: usize, got: usize },

    /// A layout, collation or offset rule of `strided-basic` was violated.
    #[error(transparent)]
    Strided(#[from] StridedError),
}
