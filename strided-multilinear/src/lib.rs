//! `strided-multilinear` — an executor for cuEquivariance-style segmented
//! polynomials, i.e. sparse multilinear forms.
//!
//! A segmented polynomial is `Σ_paths c · Π_operands x[segment]`: a sparse
//! multilinear form over segments. `cuEquivariance` lowers one to a flat
//! *uniform-1d* contract before it calls a kernel, and this crate executes that
//! contract on the CPU:
//!
//! ```text
//! buffers      [batch, num_segments, u]      (u may be 1: a scalar per segment)
//! operations   (buffer ids, path_start, num_paths)
//! paths        (one segment id per operand, scalar coefficient)
//! for each operation:
//!   for each batch point b:
//!     for each path (s_0..s_n, c):
//!       out[idx_out(b)][s_n][0..u] += c · Π_t in_t[idx_t(b)][s_t][0..u]
//! ```
//!
//! This is the message-passing inner loop of an equivariant GNN
//! (`h[r] += Σ_{e: recv(e)=r} Σ_paths c · x[src(e)] · Y[e] · W[e]`) evaluated
//! without materialising the per-edge products.
//!
//! # What the plan guarantees
//!
//! - **Deterministic.** One output element is a sum whose association is fixed
//!   by the descriptor, the inputs and the topology: rows in run order,
//!   operations in descriptor order, paths in `sort_paths(-1)` order, and a unit
//!   with more rows than [`BLOCK_ROWS`] as fixed blocks combined left to right.
//!   Every [`Exec`](strided_basic::Exec) budget, worker count and destination
//!   layout therefore produces identical bits (up to the sign of zero, which a
//!   zero-seeded block partial can normalise).
//! - **No atomics, no locks, no scratch allocation.** Output rows are owned by one
//!   task each, and the caller supplies the scratch.
//! - **One pass per plan.** A plan with several outputs evaluates the compatible
//!   ones in one traversal of the batch points (scattered outputs with the same
//!   compiled run topology share it, batched outputs ride along, a shared output
//!   keeps its own batch reduction). Each output keeps the summation order below;
//!   no `edges × segments × u` intermediate is materialised.
//! - **Typed errors, no panics** on any caller-reachable path: indices, extents,
//!   segment ids, path ranges, layouts and scratch are all checked.
//! - **Aliasing.** Inputs may alias one another. An output may not alias any other
//!   buffer: `compile` rejects an output whose own elements repeat, and two buffers
//!   over one allocation stay the caller's obligation, as for every plan taking
//!   [`RawStridedRef`](strided_basic::RawStridedRef) /
//!   [`RawStridedMut`](strided_basic::RawStridedMut).
//! - **`f32` and `f64` only**, one element type per plan, coefficients rounded to
//!   it once at compile time (decision D5 of
//!   [tensor4all/strided-rs#479](https://github.com/tensor4all/strided-rs/issues/479)).
//!
//! # Deliberately out of scope
//!
//! Building the descriptor from a polynomial (that is the separate descriptor-lowering
//! crate), element-indexed inputs (`REPEATED` / `indexed_linear`, decision D6),
//! half precision, CUDA, and shared partial products for degree 2-3 operands.
//!
//! # Divergences from the reference implementation
//!
//! `cuEquivariance`'s `naive` method follows JAX index semantics: a negative
//! gather index wraps, an out-of-range gather clamps and an out-of-range scatter
//! drops the update. `strided-basic`'s `ScatterPlan` clamps too. This crate
//! rejects both cases with [`SegPolyError::IndexOutOfRange`] (decision D3), so a
//! descriptor or index array that would silently produce a different value there
//! is an error here.

#![doc = include_str!("../README.md")]

mod error;
mod plan;
mod simd;
mod spec;

pub use error::SegPolyError;
pub use plan::{lane_width, SegPolyElem, SegPolyPlan, BLOCK_ROWS, MAX_INPUTS, MAX_OUTPUTS};
pub use spec::{IndexMode, Layout, Operation, Path, Spec};
