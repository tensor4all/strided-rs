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

## Baseline evidence

Benchmark-only commit `a00e057` ran the complete baseline sequentially on CPUs 17-20 in L3 domain 16-23 after a valid four-second gate (selected 0.0-0.7%, sibling maximum 5.0%).

| family | variant/context | size | estimate `[low, high]` |
|---|---|---|---:|
| generic | compact_single_rank2_serial | small_n4096 | 30.774 µs `[29.826 µs, 31.382 µs]` |
| generic | compact_single_rank4_serial | small_n4096 | 40.426 µs `[39.671 µs, 41.528 µs]` |
| generic | compact_single_rank8_serial | small_n4096 | 65.522 µs `[62.019 µs, 68.765 µs]` |
| generic | compact_multi_rank4_serial | small_n4096 | 31.572 µs `[31.014 µs, 32.320 µs]` |
| generic | compact_multi_rank8_serial | small_n4096 | 61.227 µs `[59.898 µs, 63.677 µs]` |
| generic | rank4_nonunit_source_serial | small_n4096 | 40.378 µs `[39.691 µs, 41.099 µs]` |
| generic | rank4_negative_source_serial | small_n4096 | 39.785 µs `[39.595 µs, 40.223 µs]` |
| generic | compact_single_rank2_max_threads_4 | small_n4096 | 30.382 µs `[29.786 µs, 31.042 µs]` |
| generic | compact_single_rank4_max_threads_4 | small_n4096 | 41.381 µs `[39.723 µs, 44.011 µs]` |
| generic | compact_single_rank8_max_threads_4 | small_n4096 | 65.194 µs `[63.465 µs, 67.235 µs]` |
| generic | compact_multi_rank4_max_threads_4 | small_n4096 | 31.930 µs `[30.909 µs, 32.932 µs]` |
| generic | compact_multi_rank8_max_threads_4 | small_n4096 | 61.058 µs `[59.354 µs, 62.750 µs]` |
| generic | rank4_nonunit_source_max_threads_4 | small_n4096 | 40.199 µs `[39.551 µs, 40.961 µs]` |
| generic | rank4_negative_source_max_threads_4 | small_n4096 | 41.312 µs `[39.889 µs, 42.828 µs]` |
| generic | compact_single_rank2_serial | near_threshold_n32768 | 247.68 µs `[240.37 µs, 256.87 µs]` |
| generic | compact_single_rank4_serial | near_threshold_n32768 | 336.67 µs `[323.53 µs, 352.23 µs]` |
| generic | compact_single_rank8_serial | near_threshold_n32768 | 515.28 µs `[506.97 µs, 534.07 µs]` |
| generic | compact_multi_rank4_serial | near_threshold_n32768 | 257.79 µs `[251.32 µs, 264.26 µs]` |
| generic | compact_multi_rank8_serial | near_threshold_n32768 | 407.49 µs `[394.08 µs, 419.12 µs]` |
| generic | rank4_nonunit_source_serial | near_threshold_n32768 | 339.31 µs `[330.24 µs, 355.25 µs]` |
| generic | rank4_negative_source_serial | near_threshold_n32768 | 357.55 µs `[344.95 µs, 369.17 µs]` |
| generic | compact_single_rank2_max_threads_4 | near_threshold_n32768 | 252.23 µs `[243.66 µs, 263.38 µs]` |
| generic | compact_single_rank4_max_threads_4 | near_threshold_n32768 | 330.94 µs `[323.58 µs, 341.84 µs]` |
| generic | compact_single_rank8_max_threads_4 | near_threshold_n32768 | 502.62 µs `[493.78 µs, 514.06 µs]` |
| generic | compact_multi_rank4_max_threads_4 | near_threshold_n32768 | 261.86 µs `[253.81 µs, 270.18 µs]` |
| generic | compact_multi_rank8_max_threads_4 | near_threshold_n32768 | 400.12 µs `[393.91 µs, 407.71 µs]` |
| generic | rank4_nonunit_source_max_threads_4 | near_threshold_n32768 | 337.08 µs `[325.37 µs, 349.72 µs]` |
| generic | rank4_negative_source_max_threads_4 | near_threshold_n32768 | 340.41 µs `[324.99 µs, 356.37 µs]` |
| generic | compact_single_rank2_serial | medium_n262144 | 2.0329 ms `[1.8847 ms, 2.1357 ms]` |
| generic | compact_single_rank4_serial | medium_n262144 | 2.5877 ms `[2.5574 ms, 2.6500 ms]` |
| generic | compact_single_rank8_serial | medium_n262144 | 4.2516 ms `[4.1455 ms, 4.3802 ms]` |
| generic | compact_multi_rank4_serial | medium_n262144 | 2.0340 ms `[1.9871 ms, 2.0723 ms]` |
| generic | compact_multi_rank8_serial | medium_n262144 | 3.9605 ms `[3.6421 ms, 4.1128 ms]` |
| generic | rank4_nonunit_source_serial | medium_n262144 | 2.6691 ms `[2.6242 ms, 2.7426 ms]` |
| generic | rank4_negative_source_serial | medium_n262144 | 2.6393 ms `[2.6063 ms, 2.6854 ms]` |
| generic | compact_single_rank2_max_threads_4 | medium_n262144 | 524.61 µs `[524.34 µs, 524.84 µs]` |
| generic | compact_single_rank4_max_threads_4 | medium_n262144 | 694.55 µs `[693.53 µs, 695.34 µs]` |
| generic | compact_single_rank8_max_threads_4 | medium_n262144 | 1.2369 ms `[1.2340 ms, 1.2392 ms]` |
| generic | compact_multi_rank4_max_threads_4 | medium_n262144 | 2.1008 ms `[2.0336 ms, 2.1832 ms]` |
| generic | compact_multi_rank8_max_threads_4 | medium_n262144 | 4.0601 ms `[3.8806 ms, 4.2018 ms]` |
| generic | rank4_nonunit_source_max_threads_4 | medium_n262144 | 675.59 µs `[671.90 µs, 678.06 µs]` |
| generic | rank4_negative_source_max_threads_4 | medium_n262144 | 689.87 µs `[687.98 µs, 693.22 µs]` |
| generic | compact_single_rank2_serial | large_n1048576 | 8.0595 ms `[7.8222 ms, 8.2804 ms]` |
| generic | compact_single_rank4_serial | large_n1048576 | 11.179 ms `[10.914 ms, 11.538 ms]` |
| generic | compact_single_rank8_serial | large_n1048576 | 19.445 ms `[19.056 ms, 19.809 ms]` |
| generic | compact_multi_rank4_serial | large_n1048576 | 8.2678 ms `[8.0969 ms, 8.4493 ms]` |
| generic | compact_multi_rank8_serial | large_n1048576 | 15.279 ms `[14.547 ms, 15.617 ms]` |
| generic | rank4_nonunit_source_serial | large_n1048576 | 11.126 ms `[10.716 ms, 11.476 ms]` |
| generic | rank4_negative_source_serial | large_n1048576 | 10.939 ms `[10.466 ms, 11.482 ms]` |
| generic | compact_single_rank2_max_threads_4 | large_n1048576 | 2.0891 ms `[2.0696 ms, 2.0982 ms]` |
| generic | compact_single_rank4_max_threads_4 | large_n1048576 | 2.6962 ms `[2.6798 ms, 2.7052 ms]` |
| generic | compact_single_rank8_max_threads_4 | large_n1048576 | 4.7120 ms `[4.5243 ms, 4.8291 ms]` |
| generic | compact_multi_rank4_max_threads_4 | large_n1048576 | 5.7072 ms `[5.6620 ms, 5.7490 ms]` |
| generic | compact_multi_rank8_max_threads_4 | large_n1048576 | 16.251 ms `[15.650 ms, 17.014 ms]` |
| generic | rank4_nonunit_source_max_threads_4 | large_n1048576 | 2.7147 ms `[2.7082 ms, 2.7280 ms]` |
| generic | rank4_negative_source_max_threads_4 | large_n1048576 | 2.6976 ms `[2.6898 ms, 2.7083 ms]` |
| rank2 control | serial | small_n4096 | 30.705 µs `[30.121 µs, 31.785 µs]` |
| rank2 control | max_threads_4 | small_n4096 | 31.051 µs `[30.150 µs, 31.945 µs]` |
| rank2 control | serial | near_threshold_n32768 | 248.91 µs `[246.02 µs, 252.98 µs]` |
| rank2 control | max_threads_4 | near_threshold_n32768 | 255.02 µs `[243.08 µs, 270.70 µs]` |
| rank2 control | serial | medium_n262144 | 2.0449 ms `[1.9803 ms, 2.0996 ms]` |
| rank2 control | max_threads_4 | medium_n262144 | 533.09 µs `[525.59 µs, 543.23 µs]` |
| rank2 control | serial | large_n1048576 | 8.3363 ms `[8.0298 ms, 8.6054 ms]` |
| rank2 control | max_threads_4 | large_n1048576 | 2.0558 ms `[2.0434 ms, 2.0649 ms]` |

The need-before-implementation gate is **PASS**. At medium size, compact single-axis rank 4/8 serial measured 2.5877/4.2516 ms versus the rank-2 control 2.0449 ms; both exceed 1.0 ms and rank 8 is 2.08x the control. The rank-8/rank-2 per-source ratio is 2.09, above the 25% signal. Cases and gates were frozen before production implementation.

## Candidate attempt 1 and design delta

Incremental-cursor candidate `1b4265b` completed the full suite after a valid
load gate. Every case improved with `p < 0.05`, and all gates passed except one:
medium compact single-axis rank 4 serial improved from 2.5877 to 0.88426 ms,
only 2.93x versus the predeclared 3x gate. Attempt 1 is therefore **FAIL** and
cannot be promoted. Rank 8 serial improved 4.41x; all four-thread, multi-axis,
non-unit, negative, rank-ratio, and control gates passed. No gate or case is
changed.

The failure activates the design's previously deferred metadata compression.
Compile will scan each cursor's axes left-to-right in the exact decode order and
fuse adjacent axes only when the next stride equals the first step times the
checked accumulated extent. For the outer cursor, this relation must hold for
both source and destination strides; for the inner cursor it must hold for the
source stride. Combined extents and recomputed resets remain checked. Negative
or noncontiguous boundaries stay separate. The generic execution loop,
reduction order, identities, accumulation functions, partitioning, and public
API are unchanged; this is private metadata compression, not a new execution
branch.

`reviewer-flash` reviewed exact design-delta commit `c1eea8d` and returned
**Correct-to-merge**. Implementation must use checked multiplication with the
accumulated fused extent and recompute resets through `checked_reduce_reset`;
wrapping or immediate-predecessor-only checks are forbidden. The complete
baseline/candidate suite will be rerun under unchanged protocol; failure of the
3x gate again blocks promotion.
