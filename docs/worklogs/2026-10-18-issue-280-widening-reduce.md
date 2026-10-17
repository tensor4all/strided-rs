# 2026-10-18: strided-rs #280 residual — widening two-input reduction

Release notes for the additive follow-up to 0.6.0 that closes the remaining
substance of [tensor4all/strided-rs#280](https://github.com/tensor4all/strided-rs/issues/280).
Only `strided-basic` changes; the release (0.6.1, `strided-basic` only) is a
separate PR. No version is bumped here.

## What #280 asked for and what is still needed

#280 was filed against 0.4.4, where `StridedView::new` copied `dims`/`strides`
into `Arc`s and the non-contiguous path built a `Vec` plan on every call. 0.6.0
(#306) made view metadata and kernel planning inline up to
`INLINE_RANK = 8`, so `dot`/`sum`/`reduce`/`reduce_axis` over borrowed views are
allocation-free at rank ≤ 8 without a second raw entry point. The raw variants
in #280 are therefore superseded (`R-core-01`: no duplicate raw path), except
for the one capability that is not expressible today:

- **widening two-operand reduction.** `dot` requires `A: Mul<B, Output = R>`, so
  `f32` operands cannot be accumulated in `f64`.

## Added

- `strided_basic::zip_reduce(a, b, map, combine, init)` — the two-operand
  sibling of `reduce`. It takes the accumulator type from the map closure
  (`Fn(A, B) -> U`), honours each operand's `ElementOp` without materializing a
  conjugated copy, supports permuted, gapped, negative and stride-0 operands,
  and reuses the existing fused planner and threading tree (no second traversal
  engine). Allocation-free at rank ≤ 8, including in the parallel branch.
  Exported unchanged through `strided-kernel` (`pub use strided_basic::*`).

## Documented

- `reduce`, `zip_reduce`, `sum` and `dot` now state the evaluation order
  precisely: a result that is independent of the ambient pool width requires the
  sequential policy (`ExecContext::serial()` /
  `with_execution_policy(ExecutionPolicy::Sequential, ..)`); under a parallel
  policy the result is repeatable for a fixed thread count only; no entry is a
  strict left-to-right fold, and none guarantees cross-machine bitwise equality.
  Each entry also states the rank ≤ `INLINE_RANK` allocation bound.
- `docs/design/erased-execution-policy.md` gains a "Reduction evaluation order
  and the deterministic route" section, plus `zip_reduce` in the covered-family
  list.

## Tests

- `strided-basic/tests/alloc_matrix.rs`: rows for `sum`, `reduce` and
  `zip_reduce` across rank 1..=8, both sides of `SMALL_TENSOR_THRESHOLD`,
  contiguous / permuted / stride-0 operands, serial and two-worker windows, and
  the tensor-sized shapes above `MINTHREADLENGTH`.
- `strided-basic/tests/zip_reduce.rs`: f32→f64 widening case (`dot` loses unit
  terms, `zip_reduce` does not; contiguous, permuted and gapped layouts); naive
  reference equality for six layouts with exact integer data; `ElementOp`
  honouring (`Conj`); empty operand; shape/rank rejection; the parallel branch
  above the threading threshold; `ExecContext::serial()` bitwise identity for
  `zip_reduce`, `reduce`, `sum` and `dot` inside 1-, 2- and 4-worker pools;
  parallel-policy repeatability for a fixed thread count; and a positive
  control (`fold_order_is_observable`) proving the determinism data can see a
  reordered fold, so the pool-width test is not vacuous.

## Performance

No new fast path and no performance claim. `zip_reduce` reuses the existing
fused planner, the `FOLD_LANES` lane fold and the `mapreduce_threaded_fold`
tree, so its cost is the `reduce` cost plus one operand stream. The repository
gate run for this change is the per-call overhead smoke
(`OVERHEAD_GATE=1 cargo bench --profile release -p strided-basic --bench
overhead_smoke`, pass). Scaling rows for a typed two-input reduction belong to
[`strided-rs-benchmark-suite`](https://github.com/tensor4all/strided-rs-benchmark-suite)
and are a follow-up; nothing in this change makes a speed claim that would
require them here.

## Not changed

- No version bump, no publish. `Cargo.toml` files are untouched.
- `dot` keeps its semantics and its serial-only traversal (its docs now say so).
