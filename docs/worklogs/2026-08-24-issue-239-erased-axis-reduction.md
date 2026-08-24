# Issue 239 erased axis-reduction replay

## Task and dependency check

Optimize `ErasedReducePlan` axis replay without changing reduction order,
identity semantics, dtype behavior, public API, or threading policy. Baseline
source is integration commit `75fb0f70f6138bab3a5bc033d59bfc64bfd033f3`.
Selected reviewer: read-only `reviewer-flash`, high thinking, for the design and
exact final diff.

Issues #207 and #231 were checked before design. They own the separate typed
`reduce_view.rs::reduce_axis` implementation: #207 concerns empty-axis pointer
formation, and #231 concerns an arbitrary caller seed dropped by its contiguous
fast path. #239 changes only fixed-identity `ErasedReducePlan` replay in
`erased.rs`; it never calls typed `reduce_axis`. #231 explicitly records
`ErasedReducePlan` as unaffected. The independence evidence is recorded in
#239, so both typed issues remain open and untouched while this task proceeds.

## Evidence and contracts reviewed

- #213/#239 and #207/#231
- repository/shared performance, numerical, unsafe, threading, and benchmark
  rules
- `ErasedReducePlan::compile_axes`, `ReduceLayout::Axes`, serial/parallel
  execution, initialized/uninitialized writers, and reduction scalar semantics
- erased reduction tests, source-contract tests, policy tests, allocation tests,
  and `erased_policy_thresholds.rs`

Current serial and parallel loops rebuild a full source coordinate and call
rank-scanning checked-offset helpers inside each of
`dest_total * reduce_total` visits. Parallelism partitions only destination
outputs, leaving the same inner cost.

## Benchmark-first experiment

Add `erased_axis_reduce_generic_rank_layout` while retaining the existing
rank-2/two-row axis-reduce group as a control. All cases use F64 sum, compact
column-major destination, and exactly `N` source values:

- one reduced axis of extent 2 at compact rank 2, 4, and 8; output count `N/2`;
- multiple reduced axes at rank 4 (three extent-2 axes) and rank 8 (seven
  extent-2 axes), keeping only the final axis; total reduction visits remain N;
- rank-4 non-unit source layout;
- rank-4 negative-stride source layout with a validated base offset.

The compact rank cases use first `rank-1` extents of 2 and final extent
`N / 2^(rank-1)`. Non-unit and negative variants retain the rank-4 logical
shape and reduction axes while varying only source strides/storage. Allocation,
plan compile, raw descriptor construction, and output setup remain outside
timing; timed regions execute and black-box the output.

Use the unchanged threshold profile (`2^12`, `2^15`, `2^18`, `2^20`), serial
and `max_threads(4)`, release mode, 300 ms warmup, 10 samples, one-second
measurement, `RAYON_NUM_THREADS=4`, and benchmark thread override 4. Runs are
sequential on AMD EPYC 7713P. Before each complete baseline/candidate run,
select four cores in one L3/CCD, require selected cores below 2% busy for four
seconds and every sibling below 20%, and pin with `taskset`; otherwise classify
the complete run INCONCLUSIVE.

Need-before-implementation gate at medium N:

- compact single-axis rank 4 or rank 8 serial must be at least 2x slower than
  the existing rank-2 control or exceed 1.0 ms absolute; and
- rank-8/rank-2 single-axis per-source cost must increase by at least 25%, or
  one multi-axis/non-unit/negative case must exceed compact rank 2 by 25%.

If the gate fails, retain evidence and do not change production.

Predeclared candidate gates, Criterion point estimates:

- compact single-axis rank 4/rank 8 serial: at least 3x faster; four-thread:
  at least 2x faster;
- multi-axis rank 4/rank 8 serial: at least 3x faster; four-thread at least 2x;
- non-unit/negative rank 4 serial: at least 2x faster;
- candidate medium single-axis rank-8/rank-2 cost ratio no more than 1.5.

Validity/non-regression gates:

- no selected generic or existing rank-2 control point estimate regresses more
  than 10%; every primary improvement has `p < 0.05`;
- bitwise-equal results for serial/parallel and initialized/uninitialized paths;
- preserve reduction association/order, fixed identities, wrapping integer
  arithmetic, sum-squares rounding, nonfinite behavior, and output order;
- cover rank0, zero retained/reduced extents, all-axes reduction, multiple and
  reordered axes, transposed/non-unit/negative strides, nonzero offsets, and
  below/exact/above threshold contexts;
- formatting, focused/default/parallel/workspace tests, allocation contract,
  modified-file coverage, docs, rules review, exact-diff review, and hosted CI
  pass.

Cases/gates freeze before baseline. No selective reruns or post-hoc exclusions.

## Implementation design

Extend only private `ReduceLayout::Axes` metadata with two prepared cursors:

1. an outer paired cursor over kept/output axes, carrying source-base and
   destination offsets;
2. an inner source cursor over reduced axes, restarted from the current outer
   source base for each output.

Compile maps kept axes in source order and reduced axes in the caller-supplied
order, validates signed source/destination spans, and precomputes checked
step/reset deltas. The initial implementation will not compress adjacent axes;
that optional metadata optimization is deferred unless a frozen candidate gate
fails and a separately reviewed design delta proves identical visit order.

Serial execution decodes the outer cursor once. Parallel execution decodes once
per destination worker range. For each output, reduction starts at the current
outer source base, traverses exactly `reduce_total` values in the original axis
order with incremental source offsets, writes the accumulator to the current
destination offset, then advances the outer cursor. Empty reduced domains write
the fixed operation identity without forming a source pointer. Rank-0 and
all-axes reductions preserve existing behavior.

Unchecked hot-loop arithmetic is permitted only with concrete nearby
`// INVARIANT:` and `// SAFETY:` proofs naming the complete three-link chain:
(1) compile-time checked source/destination spans and checked step/reset deltas,
including `-(extent-1)*stride`; (2) execute-time raw descriptor/pointer bounds
validation; and (3) exact plan-layout equality before dispatch. Keep full reductions,
typed `reduce_view`, public enums/APIs, accumulation lanes/order, and threading
threshold unchanged. Use existing rank-bounded scratch to preserve
allocation-free execution through rank 8; do not introduce a generic cross-plan
cursor abstraction in this task.

## Review and verification

Benchmark implementation starts only after a Correct-to-merge design verdict.
After a valid need gate, implementation gets focused ground-truth and
source-contract tests, paired candidate timing, complete local gates, and an
exact-final-diff `reviewer-flash` verdict before PR creation.

## Gate status

`reviewer-flash` reviewed exact design commit `a7880dd` with high thinking and
a read-only boundary. Verdict: **Correct-to-merge**; benchmark implementation
may proceed. The safety-proof chain and no-compression initial sequencing are
now explicit above. The existing rank-2 control uses the same generic axes
replay and may improve; it remains a valid non-regression control because its
point estimate may improve but must not regress by more than 10%.

Benchmark implementation, baseline, candidate, and final verification are
pending.
