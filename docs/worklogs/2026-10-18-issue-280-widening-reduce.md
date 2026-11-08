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

## Fixed (post-review)

- **Empty views no longer reach the block planner.** A zero-extent dimension
  makes any stride representable, so a valid empty view such as
  `dims [0, 2]`, `strides [0, isize::MAX]` reached `block.rs` and overflowed
  `isize::MAX * 8` in `zip_reduce`, `reduce` and `sum`. All four reduction
  entries (`reduce`, `zip_reduce`, `sum`, `dot`) now return their identity for
  zero elements right after shape validation and checked element counting, with
  a test per entry. `reduce_axis`/`reduce_axis_into` already handled the empty
  reduced axis and were unaffected.
- **Parallel contiguous leaf dispatch.** The serial contiguous path wraps
  `fold_contiguous2` in `simd::dispatch_if_large`; the parallel leaf now does the
  same, matching `reduce`, so both routes take the same runtime target-feature
  dispatch.
- The forwarding-only `zip_reduce_impl` wrapper is gone; the body lives in
  `zip_reduce`.

## Documented

- `reduce`, `zip_reduce`, `sum` and `dot` now state the evaluation order
  precisely: for the entries that can fan out, a result that is independent of
  the ambient pool width requires the sequential policy
  (`ExecContext::serial()` /
  `with_execution_policy(ExecutionPolicy::Sequential, ..)`); under a parallel
  policy the result is repeatable for a fixed thread count only; `dot` never
  enters a parallel branch, so its result does not depend on the pool under any
  policy. No entry is a strict left-to-right fold, and none guarantees
  cross-machine bitwise equality. Each entry also states the rank ≤
  `INLINE_RANK` allocation bound.
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
  honouring (`Conj`); empty operand and the non-contiguous empty layout
  (`dims [0, 2]`, `strides [0, isize::MAX]`) for all four entries; shape/rank
  rejection; both above-threshold layouts (contiguous and gapped) checked
  against the naive reference under an explicit multi-worker policy, so the
  contiguous parallel leaf is result-checked and not only allocation-checked;
  `ExecContext::serial()` bitwise identity for `zip_reduce`, `reduce`, `sum` and
  `dot` inside 1-, 2- and 4-worker pools on both layouts; parallel-policy
  repeatability for a fixed thread count; and a positive control
  (`fold_order_is_observable`) proving the determinism data can see a reordered
  fold, so the pool-width test is not vacuous.

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

One measured negative result, recorded so nobody "cleans it up" later: routing
`fold_contiguous` through a generic "map index to value" lane helper (the way to
share its scheduling with `fold_contiguous2`) cost 0.34 ns/element against
0.11 ns/element on a contiguous `f64` `reduce` of 1 MiB (`cargo test --release
-j 4`, min of 7): the `chunks_exact` + `zip` form is what keeps the 16 lanes
bound-check free and vectorized. `fold_contiguous2` therefore mirrors the
scheduling by hand and its docs say so.

## Not changed

- No version bump, no publish. `Cargo.toml` files are untouched.
- `dot` keeps its semantics and its serial-only traversal (its docs now say so).
