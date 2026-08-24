# Issue 238 dynamic indexed replay

## Task and contract

Issue #238 is the second P1 child of #213. `DynamicSlicePlan` and
`DynamicUpdateSlicePlan` still reconstruct full coordinates and scan source and
destination strides per logical element outside their compact rank-one fast
paths. Dynamic slice must return the clamped fixed window. Dynamic update must
copy the complete operand first and then overwrite the clamped update window;
initialized and uninitialized destinations share that semantic contract.

Baseline source: `acdeea3f620b9a515c6915a44730022d19c4e71a`, the
current integration head after #242. Selected reviewer: read-only
`reviewer-flash`, high thinking, for the design and exact final diff. No
benchmark or production implementation starts before a Correct-to-merge design
verdict.

## Inputs reviewed

- issues #213/#238 and PRs #236/#241/#242
- repository and shared performance/layout/threading rules
- `docs/design/erased-execution-policy.md`
- `DynamicSlicePlan` and `DynamicUpdateSlicePlan` compile, fast, generic,
  serial/parallel, initialized/uninitialized, and copy-before-update paths in
  `strided-kernel/src/gather_plan.rs`
- erased wrappers and tests in `erased_indexed_write_plan.rs`,
  `issue_187_uninit_indexed.rs`, and `erased_policy_parallel.rs`
- current threshold benchmark groups for dynamic slice/update

## Benchmark-first experiment

A benchmark-only commit adds `erased_dynamic_slice_generic_rank_layout` and
`erased_dynamic_update_generic_rank_layout`. Each group uses the unchanged
threshold profile (`2^12`, `2^15`, `2^18`, `2^20`) and serial plus
`max_threads(4)` contexts. Five variants produce/update the same logical N
values:

- compact rank 2, 4, and 8;
- rank-2 non-unit source layout;
- rank-2 negative-stride source layout.

For compact rank `r`, the first `r - 1` axes have extent 2 and the final axis
has extent `N / 2^(r-1)`. Dynamic-slice operand dims add 128 only to the final
axis; starts are zero except 64 on that axis, so the requested window has N
values and one compact column-major destination stream at every rank.
Dynamic-update operand/destination dims use the padded shape, update dims use
the N-value window shape, and the same start vector is used. The update group
times the required full operand copy plus update replay, matching the public
operation.

Non-unit dynamic slice varies only operand strides while retaining the compact
destination. Negative dynamic slice uses a negative final operand stride and a
validated base offset. Non-unit/negative dynamic update varies only update
strides; operand/destination stay compact so initial-copy cost is comparable.
Physical buffers are sized from validated reachable spans. Start vectors,
allocation, plan compilation, and raw descriptor construction remain outside
timing. Existing rank-one groups remain fast-path controls.

Criterion uses 300 ms warmup, 10 samples, one-second measurement, release mode,
`RAYON_NUM_THREADS=4`, and benchmark thread override 4. Runs are sequential on
AMD EPYC 7713P. Before each complete baseline/candidate suite, select four cores
in one L3/CCD, require every selected core below 2% busy over four seconds and
every sibling below 20%, and pin the process with `taskset`; otherwise classify
the complete run INCONCLUSIVE.

Need-before-implementation gate at `medium_n262144`:

- generic rank 4 or rank 8 serial must exceed the matching rank-one serial
  control by at least 2x or exceed 1.0 ms absolute in each operation family;
- and rank 8 must cost at least 25% more per logical value than rank 2, or a
  non-unit/negative case must cost at least 25% more than compact rank 2.

If either operation family fails both signals, retain its benchmark evidence
but do not optimize that family.

Predeclared candidate gates, Criterion point estimates:

- dynamic slice medium serial rank 4/rank 8: at least 3x faster; four-thread:
  at least 2x faster; non-unit/negative: at least 2x faster;
- dynamic update medium serial rank 4/rank 8: at least 2x faster; four-thread:
  at least 1.5x faster; non-unit/negative: at least 1.5x faster;
- candidate medium rank-8/rank-2 cost ratio no more than 1.5 for each family.

Validity/non-regression gates:

- no selected generic or existing rank-one control regresses by more than 10%;
- every primary improvement has `p < 0.05`, and all declared cases complete;
- exact values match for rank 2/4/8, lower/upper clamping, nonzero raw offsets,
  non-unit/negative source and destination layouts, zero/empty outputs,
  initialized/uninitialized destinations, and serial/parallel below/exact/above
  threshold;
- update tests prove full operand copy, window overwrite, untouched regions,
  and no uninitialized read;
- formatting, default/parallel feature tests, workspace tests, modified-file
  coverage, docs, repository-rules review, exact-diff review, and hosted CI pass.

Cases and gates are frozen before baseline. A failed host gate invalidates the
complete run; no selective reruns or post-hoc exclusions are allowed.

## Implementation design

Keep rank-one contiguous fast paths and all public APIs unchanged. Add one
private, same-shaped window replay helper shared only by dynamic slice and
dynamic update. It stores, per logical window axis:

- source and destination step;
- checked source and destination reset/carry delta.

Compile validates source/destination signed layout spans and every delta,
including negative strides. Execution reads/clamps starts once and computes the
checked logical window base once. Serial replay decodes state once; parallel
replay decodes once per worker range. It then advances source/destination
offsets incrementally in column-major order. `RawStridedRef`/writer validation
plus exact `check_call` layout matching proves allocation reachability; nearby
`// INVARIANT:`/`// SAFETY:` comments must identify that proof. No per-element
checked-offset helper or full-coordinate rebuild remains.

Dynamic update retains its existing initial `CopyPlan` and begins incremental
update replay only after that copy. The uninitialized path still obtains the
initialized writer from `execute_uninit_then` before overwrite replay. Do not
refactor the already-merged GatherReplay or extract a cross-family cursor; this
helper is justified only because slice and update have the same two-offset
window traversal in this task.

## Correctness and review

Add focused ground-truth and differential tests for rank 2/4/8, compact and
both unusual layouts, lower/upper starts, offsets, explicit i32/i64 starts,
initialized/uninitialized output, zero dimensions, and threshold-boundary
serial/parallel equality. Preserve broad existing dtype coverage rather than
duplicating it.

The implementer runs focused default/parallel tests and checks. The parent
reviews all unsafe/invariant changes, runs paired candidate timing and complete
repository gates, records every estimate/CI and disposition here, then requests
an exact-final-diff `reviewer-flash` verdict before PR creation.

## Gate status

Design review, benchmark implementation, baseline, candidate, and verification
are pending.
