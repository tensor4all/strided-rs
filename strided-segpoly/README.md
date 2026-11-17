# strided-segpoly

Executor for **lowered segmented polynomials**: fused sparse gather, segment
product and accumulation over strided CPU buffers, with no per-edge intermediate.

A segmented polynomial is `Σ_paths c · Π_operands x[segment]` — a sparse
multilinear form whose sparsity is a list of *paths*. `cuEquivariance` lowers one
to a flat contract before it calls a kernel, and this crate executes that
contract on the CPU:

```text
buffers      [batch, num_segments, u]      (u may be 1: one scalar per segment)
operations   (buffer ids, path_start, num_paths)
paths        (one segment id per operand, scalar coefficient)

for each operation (bufs, paths):
  for each batch point b:
    for each path (s_0..s_n, c):
      out[idx_out(b)][s_n][0..u] += c · Π_t in_t[idx_t(b)][s_t][0..u]
```

That is the message-passing inner loop of an equivariant GNN,
`h[r] += Σ_{e: recv(e)=r} Σ_paths c · x[src(e)] · Y[e] · W[e]`, evaluated without
materialising the per-edge products (an edge × segments × u buffer would be
hundreds of MiB at production sizes).

## What the plan guarantees

- **Deterministic.** An output element is a sum whose association is fixed by the
  descriptor, the inputs and the topology: rows in run order, operations in
  descriptor order, paths in `sort_paths(-1)` order, and a unit with more rows
  than `BLOCK_ROWS` as fixed blocks combined left to right. A destination whose
  lane extent is 1 while the inputs' lane extent is `u` is a lane *reduction*: it
  sums each row's `u` lanes in increasing lane order, after the block partials are
  combined. Every `Exec` budget, worker count and destination layout produces
  identical bits (up to the sign of zero, which a zero-seeded block partial can
  normalise).
- **No atomics, no locks, no allocation.** One task owns each destination row
  family, and the caller supplies the scratch.
- **One pass per plan.** A plan with several outputs evaluates the compatible ones
  in one traversal of the batch points: scattered outputs with the same compiled
  run topology share it, batched outputs ride along (their row is one batch
  point), and a shared output keeps its own batch reduction. Each output keeps
  its documented order, and no edge × segments × u intermediate is materialised.
- **Typed errors, no panics** on caller-reachable paths: indices, extents, segment
  ids, path ranges, layouts and scratch are all checked.
- **Aliasing.** Inputs may alias one another. An output may not alias any other
  buffer: `compile` rejects an output whose own elements repeat, and two buffers
  over one allocation stay the caller's obligation, as for every plan taking
  `RawStridedRef` / `RawStridedMut`.
- **`f32` and `f64` only**, one element type per plan, with path coefficients
  rounded to it once at compile time.

## Example

A shared weight applied per segment (`y[b, s, :] += c_s · x[b, s, :] · w[0, s, :]`):

```rust
use strided_basic::{Exec, RawStridedMut, RawStridedRef};
use strided_segpoly::{IndexMode, Layout, Operation, Path, SegPolyPlan, Spec};

let (batch, segments, u) = (2usize, 2usize, 3usize);
let x: Vec<f64> = (0..batch * segments * u).map(|i| i as f64).collect();
let w: Vec<f64> = (0..segments * u).map(|i| 1.0 + i as f64).collect();
let mut y = vec![0.0f64; batch * segments * u];

// `[batch, num_segments, u]`, lane axis unit-stride.
let x_dims = [batch, segments, u];
let x_strides = [(segments * u) as isize, u as isize, 1];
let w_dims = [1, segments, u];
let w_strides = x_strides;
let y_dims = [batch, segments, u];
let y_strides = x_strides;

let spec = Spec {
    num_batch_axes: 1,
    operations: &[Operation {
        buffers: &[0, 1, 2],
        path_start: 0,
        num_paths: 2,
    }],
    paths: &[
        Path { segments: &[0, 0, 0], coefficient: 1.0 },
        Path { segments: &[1, 1, 1], coefficient: -0.5 },
    ],
    // Inputs first, the output last: batched, shared weight, batched output.
    indexes: &[
        &[IndexMode::Batched],
        &[IndexMode::Shared],
        &[IndexMode::Batched],
    ],
};

let plan = SegPolyPlan::<f64>::compile(
    &spec,
    &[
        Layout { dims: &x_dims, strides: &x_strides },
        Layout { dims: &w_dims, strides: &w_strides },
    ],
    &[Layout { dims: &y_dims, strides: &y_strides }],
)?;

let inputs = [
    RawStridedRef::new(&x, &x_dims, &x_strides, 0)?,
    RawStridedRef::new(&w, &w_dims, &w_strides, 0)?,
];
let mut outputs = [RawStridedMut::new(&mut y, &y_dims, &y_strides, 0)?];
let mut scratch = vec![0.0f64; plan.scratch_len(1)];
plan.execute_into(&mut outputs, &inputs, spec.indexes, &mut scratch, Exec::serial())?;

// Index [1, 0, 2] of a [2, 2, 3] lane-unit-stride buffer: 1.0 * x * w = 8 * 3.
assert_eq!(outputs[0].data()[1 * segments * u + 2], 24.0);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`execute_into` zeroes each destination block and accumulates ([`D2`]); every
output can also be seeded by the caller and updated with `execute_acc`. A plan
whose scattered outputs were compiled from the same run topology evaluates them
in one pass; the call must supply each output's `RunIndex` with that compiled
row order, and a different row order is a typed error rather than a silently
different sum.

## Indexing

Per buffer, per batch axis (one axis in this increment):

| Mode | Input | Output |
| --- | --- | --- |
| `Batched` | row `b` | row `b` |
| `Shared` | row 0, broadcast | row 0, **summed over the batch** |
| `Gather(&[i32])` / `Gather64` | row `idx[b]` | — |
| `Scatter(&RunIndex)` | — | sorted-run accumulation into row `run(b)` |

Lane extents are uniform across a plan's buffers, except that an output may have a
lane extent of 1, which sums the row's lanes (a lane reduction). An input with lane
extent 1 is broadcast along the lane axis; an input whose lane stride is neither 0
nor 1 is copied into a contiguous tile before the kernel reads it, so the
arithmetic stays per-lane and the copy cannot move a bit.

A `Shared` output is the transpose of a shared input, i.e. the gradient of a
shared weight; it is a batch reduction and follows the same fixed order.

## Deliberately out of scope

Building the descriptor from a polynomial (a separate `segpoly-desc` crate),
element-indexed inputs (`REPEATED` / `indexed_linear`), half precision, CUDA, and
shared partial products for degree 2-3 operands. Operations with four or more
operands take the scalar per-lane loop (`shortcut:` in `plan/exec.rs`); one to
three operands run a runtime-dispatched SIMD body.

## Divergences from the reference

`cuEquivariance`'s `naive` method follows JAX index semantics: a negative gather
index wraps, an out-of-range gather clamps and an out-of-range scatter drops the
update. `strided-basic`'s `ScatterPlan` clamps too. This crate rejects both with
a typed error, so a descriptor or index array that would silently produce a
different value there fails here instead.

## Provenance

The execution contract is modelled on `cuEquivariance`'s `uniform_1d` lowering
(NVIDIA, Apache-2.0) and is validated against its `naive` method at a pinned
commit; no upstream code is ported. The sorted-run topology is
`strided-basic`'s `RunIndex` (`tensor4all/strided-rs#479`, part 1).

[`D2`]: https://github.com/tensor4all/strided-rs/issues/479
