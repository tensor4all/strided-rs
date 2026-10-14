# Issue #300 — remove the raw copy/scale/axpy rank-8 cliff

## Scope

`strided-basic`'s borrowed-metadata raw family (`copy_scale_raw`,
`copy_scale_conj_raw`, `axpy_raw`, `axpy_conj_raw`, `CopyPlan`) fused the
destination/source stride pair only when the *original* rank was at most
`RAW_FUSED_RANK_LIMIT`. Above that it built a view plan on every call
(`compute_order` + `compute_block_sizes`, 46-112 allocations per call measured
for ranks 9-16). The guard ran before size-1 axes were dropped and contiguous
axes merged, so a high-rank layout that fuses to a few axes took the planning
path as well. `CopyPlan` stored the same optional fused layout and fell back per
execution.

Baseline: `7bb8022` (origin/main). Candidate: this branch.

## Design

Fusion now happens before the representation is chosen, and the guard is on the
*fused* rank:

- `fuse_axes` is the single sort/drop/merge core (unchanged semantics, now
  shared). `fuse_pair_traversal` returns `PairTraversal::{Inline, Spilled}`:
  inline storage for fused ranks at most `RAW_FUSED_RANK_LIMIT`, otherwise one
  `Vec` bounded by the fused rank. Any original rank is accepted.
- `apply_spilled_range` is the carry loop for spilled axes (one rank-bounded
  cursor `Vec`, incremental offsets, no per-element coordinate decode).
- The raw entry points and `CopyPlan` dispatch once between the two
  representations. `FusedPairLayout`, `apply_fused_pair`, `apply_fused_range`,
  `replay_fused_raw` and the window plans' `fuse_pair_layout` keep their previous
  shapes, so the common path (original rank and fused rank at most 8) compiles to
  the loop it had before. Two deliberate details protect that:
  `apply_spilled_range` is `#[inline(never)]`, and `apply_spilled_pair` is a
  separate function because sharing one entry with the inline replay was
  measured to slow the inline path by 10-47% depending on the instantiation.
- `CopyPlan::fused` becomes a `PairTraversal`; `fused_layout()` (the parallel
  concatenate fast path in `static_indexing_plan`) keeps its "inline only"
  meaning. `gather_plan` and `static_indexing_plan` are unchanged.
- `CopyPlan`'s four `None =>` view-kernel arms are gone; the plan always has a
  traversal. `map_raw_into` became test-only and is gated with `cfg(test)`.
- The view side gets one fix of the same kind: `simd::dispatch_if_large` in
  `map_parts_into_validated` captured the operation closure by reference, so LLVM
  never vectorized that leaf (0.73 ns/element instead of 0.19 on a contiguous
  `f64` pair). `map_contiguous`, an out-of-line helper that takes the closure by
  value, fixes it for `map_into` and therefore `copy_scale`/`copy_conj`.

No public API was added. `RAW_FUSED_RANK_LIMIT` is unchanged as a value; its
documentation now describes the inline-storage limit rather than a fallback
boundary.

## Verification

- `cargo test -j 16 --workspace --all-features --no-fail-fast`: 52 targets pass;
  `strided-fused --test execution_policy` fails in the same two tests on the
  unmodified baseline (source-contract/thread-count assertions), so it is
  pre-existing. `cargo fmt --all --check` passes. `cargo clippy` reports no
  finding in the touched files; the `strided-view` `manual_is_multiple_of`
  findings are pre-existing.
- New tests: rank 8/9/12/16 correctness against a dense reference for
  copy-scale and both accumulating variants, padded/permuted/reversed/broadcast/
  zero-extent layouts, fusion-before-storage assertions, and a rank-9 window case
  that keeps the window plans' non-fused replay covered.
- `strided-kernel/tests/copy_plan_alloc.rs` extends its counting-allocator check:
  a rank-9 raw call allocates exactly one spilled layout plus one cursor, a
  rank-9 `CopyPlan::execute` exactly one cursor, and everything at fused rank at
  most 8 stays at zero allocations.

## Measurements

Protocol, harness and full tables:
`strided-rs-benchmark-suite/benchmarks/strided_benchmarks/raw_rank_boundary/`
(AMD EPYC 7713P, one worker, 15 interleaved repetitions, per-case median; a null
A/A control puts protocol resolution at about +-1%).

Two predeclared gates did **not** pass, and are recorded as such:

- *Primary*: every cliff case (original rank > 8 and fused rank > 8) improves by
  at least 1.3x. Run 1: 34 of 36 do (1.04x-3.13x); the two exceptions are
  `axpy_raw` on padded layouts with a long inner axis at 1M elements (ranks 9,
  12), which are 3.4-3.9% slower because the baseline was already at the
  memory-bandwidth floor (0.307-0.321 ns/element) there, so the gate demanded a
  speedup where there was nothing left to win. Run 2 (same candidate, tests and
  comments only added in between): worst ratio 1.007, 0.996 for those two cases.
- *Non-regression*: no case with original rank <= 8 loses more than 5% in median
  time. Run 1: of the 14 cases whose baseline spread is under 5%, the worst is
  +5.5% (`axpy_raw` rank 6 `padded-short-inner`); the next are +2.3% and +1.8%.
  Run 2: worst +1.0% over 21 clean cases. The remaining cases have baseline
  spreads above 5% and cannot resolve a 5% difference. The 5% target is
  therefore met only within this host's run-to-run uncertainty.

Two complete runs of the same candidate binary differ by up to about 5% per-case
median on this shared host, which is the same order as the one common-path
exception above. Both runs are published side by side rather than reporting only
the better one.

The change is recommended on the merits of the cliff numbers below, not on those
two gates.

- Cliff cases (original rank > 8, fused rank > 8, 36 cases): allocations per call
  46-112 -> 1-2; 34 of 36 cases 1.04x-3.13x faster. The two exceptions are
  `axpy_raw` on padded long-inner layouts at 1M elements (ranks 9, 12), where the
  baseline was already at the memory-bandwidth floor (0.307-0.321 ns/element) and
  the fused replay is 3.4-3.9% slower.
- Common path (original rank <= 8): the touched code does not run differently for
  these; of the 14 cases whose baseline spread is under 5%, the worst is +5.5%
  (`axpy_raw` rank 6 `padded-short-inner`).
- High-rank layouts that fuse to at most 8 axes: mostly much faster (contiguous
  `CopyPlan::execute` 2.93 -> 0.185 ns/element, 15x). The exception is
  `axpy_raw` on fully contiguous 4096-element layouts, 0.21 -> 0.27-0.30
  ns/element (1.28-1.44x), because the baseline's fallback reached the view
  kernels' runtime-dispatched contiguous axpy; at 1M elements those cases are at
  parity. A `pulp::Arch::dispatch` inside the replay was measured at 0.70
  ns/element and rejected, so no SIMD leaf was added to the raw path.
- Four threads, 1M elements: the cliff improves as at one thread; high-rank
  contiguous `axpy_raw` is 1.07-1.30x slower than the baseline's parallel view
  leaf (the raw path is serial by contract at every rank).

## Follow-up measurements (not in this change)

These were measured on the same host with the same harnesses and are recorded in
the benchmark page; they are deliberately not part of this change.

- Runtime-SIMD dispatch for the *raw* contiguous case (0.26 -> 0.19 ns/element,
  i.e. -25%, matching a `-C target-cpu=native` build) works only when the loop
  sits inside the dispatched closure with the operations moved in *by value*:
  captured through `&Apply`/`&Op`, the same loop stays scalar with a reload per
  element and is 2.7x slower. Instruction-set detection is not the cost:
  `pulp::Arch::new()` is 0.55 ns/call and `fearless_simd::Level::new()` 0.71
  ns/call (both cached), about 0.1% of a 4096-element call.
- That raw dispatch was nevertheless dropped here. In every placement tried
  (entry point, `apply_fused_pair`, with and without `#[inline(never)]` on the
  replay functions) it cost the *short-run* layouts 10-24% in exchange for the
  25-30% contiguous win, which is the wrong trade in absolute time: +0.31
  ns/element on a rank-2 padded 4096-element call against -0.06 ns/element on a
  contiguous one.
- Whole-crate codegen sensitivity: a control that contains only the raw change
  and leaves `ops_view`/`map_view` untouched still moves the *view* path's
  padded layouts by +8-13%. The short-run carry loops in this crate are that
  sensitive to any change in the functions around them, which is also why the
  raw/view dispatch work above needs a structural fix (an op-typed kernel whose
  SIMD leaf is monomorphized on its own) rather than another branch placement.
- The view path still has the planner cliff for its own callers: `axpy`/`copy_scale`
  on a rank-9 padded 4096-element layout cost 7.0-7.4 ns/element and 64
  allocations per call, where the raw replay now costs 2.5-2.6 and 1-2. Routing
  the two-operand view ops through the fused traversal is the fix, but it changes
  the view API's blocking/parallel strategy and needs its own protocol.

## Release

Only `strided-basic` changed, so only it moves: `0.5.0 -> 0.5.1` (patch; no public
API change). It now carries an explicit `[package] version` instead of
`version.workspace = true`, and `[workspace.dependencies] strided-basic` is
`0.5.1`. `strided-traits`, `strided-view`, `strided-perm`, `strided-fused` and
`strided-kernel` stay at `0.5.0`.

The packaged manifest confirms the split: `strided-basic 0.5.1` requires
`strided-view`/`strided-perm` at `0.5.0` (a caret range), so releasing it does not
force new releases of the core crates. `strided-kernel`'s test-only change does
not warrant a bump. Publication order stays
`strided-traits` -> `strided-view` -> `strided-perm`/`strided-basic` ->
`strided-fused`/`strided-kernel`, and is a human step; the version-bump recipe is
in the `version-bump` skill.

## Remaining

- The single-run contiguous `axpy_raw` case above is a real, narrow regression on
  L1-resident fully contiguous high-rank layouts. Matching it would need a
  runtime-dispatched SIMD leaf for the raw replay, which is out of scope here.
- Two cliff cases were already at the bandwidth floor in the baseline and stay
  there, so they show no speedup (no regression beyond 3.9%).
- The window plans (`DynamicSlicePlan`, `DynamicUpdateSlicePlan`) still keep
  their original-rank boundary and `WindowReplay` fallback; only fusion before
  that boundary changed for them.
