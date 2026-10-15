# tenferro-rs#2150 — reduce a product in source order, not across lanes

## Scope

`strided-basic`'s erased product reduction (`ErasedReducePlan` with
`ReduceOp::Product`) folded each strided run into lane vectors and multiplied the
lane partials at the end, so the reduction was reassociated away from source
order.

A long axis whose true product is exactly representable then returned `NaN`. For
the alternating `2.0, 0.5` axis of tenferro-rs#2150, one lane saw only `2.0` and
overflowed to `inf` while another saw only `0.5` and underflowed to `0`, and
`inf * 0 = NaN` — while the sequential product is exactly `1.0`. `f64` at 1e6
elements failed the same way, so this is not an f32 precision threshold.

Reproduced from the tenferro-rs side:
`crates/tenferro-cpu/src/tests/cpu_tests/basic_ops.rs`,
`test_reduce_prod_survives_lane_reassociation` (currently `#[ignore]`d with a
pointer at this change).

## Design

- `ReduceKernel` gains `reassociable()`. The product kernel is the one that is
  **not** reassociable, so the planner keeps its runs in source order and carries
  **one** accumulator through every run instead of folding each run
  independently and combining the partials. The ordered, unsplit exception is
  documented in the `ErasedReducePlan` evaluation-order docs for the full,
  parallel and axis-reduction paths.
- Everything else keeps its existing association: sum/min/max are associative
  and unchanged.

## Verification

- `strided-basic/tests/erased_reduce_plan.rs`: the representable product through
  the unit-stride and a strided two-run layout; a genuine overflow; a full
  reduction whose traversal splits one output into `[2^-600, 1]` and
  `[2^600, 2^600]` (a `NaN` at the never-visited offset makes an out-of-run read
  fail loudly); an axis `ContiguousRuns` reduction under the serial and a
  two-worker context (65,536 elements, above `MINTHREADLENGTH`, explicit
  two-thread pool with `current_num_threads() == 2` asserted).
- Both new regressions fail on pristine `v0.5.0` (`inf` instead of `2^600`), and
  they fail when `reassociable()`'s guard is replaced by the unconditional split.
- `cargo test -j 16 --workspace --no-fail-fast` and
  `cargo test -j 16 -p strided-basic --features parallel`: green.
  `cargo fmt --all --check`: clean.
- Merged as PR #302 (merge commit `31205d9`).

## Release

Single-crate patch release: `strided-basic` `0.5.1` -> `0.5.2`. The core crates
(`strided-traits`, `strided-view`, `strided-perm`, `strided-fused`,
`strided-kernel`) stay at `0.5.0`, and the workspace dependency entry moves to
`0.5.2` as a caret range so dependents' next published manifest admits it.

Publish order for the touched crate: `strided-basic` only, after
`strided-traits` and `strided-view` (already on the registry at `0.5.0`).

Downstream: tenferro-rs bumps its `strided-basic` pin, drops the `#[ignore]` on
the regression test, and covers the selected contract across
contiguous/strided, multi-axis, batch/parallel and f32/f64 with zero/sign and
genuine-overflow cases.
