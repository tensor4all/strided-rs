//! Inline (heap-free up to a bound) metadata collections shared by the
//! strided-basic planner and plan types.
//!
//! [`strided_view::INLINE_RANK`] is the workspace capacity constant ([#306]
//! P2). [`InlineVec`] is the rank-bounded metadata vector, and [`OperandVec`]
//! holds the destination plus up to four inputs (five slots, matching
//! `zip_map4_into` and erased fusion). Both are `tinyvec` [`TinyVec`]s: they
//! store up to their bound inline and spill to the heap above it, never
//! erroring ([#306] P1, P4; crate policy in [#312]).
//!
//! [#306]: https://github.com/tensor4all/strided-rs/issues/306
//! [#312]: https://github.com/tensor4all/strided-rs/issues/312

/// Inline vector of at most [`strided_view::INLINE_RANK`] elements.
pub type InlineVec<T> = tinyvec::TinyVec<[T; strided_view::INLINE_RANK]>;

/// Number of operand slots: destination plus up to four inputs.
pub const OPERAND_CAP: usize = 5;

/// Inline vector of at most [`OPERAND_CAP`] operands (strides, offsets, ...).
pub type OperandVec<T> = tinyvec::TinyVec<[T; OPERAND_CAP]>;
