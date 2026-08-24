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

The need-before-implementation gate is **PASS**. At medium size, compact single-axis rank 4/8 serial measured 2.5877/4.2516 ms versus the rank-2 control 2.0449 ms; both exceed 1.0 ms and rank 8 is 2.08x the control. The rank-8/rank-2 per-source ratio is 2.08, above the 25% signal. Cases and gates were frozen before production implementation.

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

## Final paired experiment

Named baseline `issue239-final-base` used exact benchmark-only commit `a00e057`; final candidate `247b88a` adds reviewed cursor-axis metadata compression. Baseline ran on CPUs 8-11 after a valid gate (selected 0.0-0.2%, sibling maximum 1.7%); candidate ran on CPUs 8,9,10,12 after a valid gate (selected 0.5-1.3%, sibling maximum 2.8%).

| phase | family | variant/context | size | estimate `[low, high]` |
|---|---|---|---|---:|
| final baseline | rank2 control | serial | small_n4096 | 30.596 µs `[29.613 µs, 32.055 µs]` |
| final baseline | rank2 control | max_threads_4 | small_n4096 | 30.280 µs `[29.634 µs, 30.735 µs]` |
| final baseline | rank2 control | serial | near_threshold_n32768 | 250.14 µs `[243.88 µs, 258.16 µs]` |
| final baseline | rank2 control | max_threads_4 | near_threshold_n32768 | 252.17 µs `[241.38 µs, 260.60 µs]` |
| final baseline | rank2 control | serial | medium_n262144 | 2.0155 ms `[1.9731 ms, 2.0888 ms]` |
| final baseline | rank2 control | max_threads_4 | medium_n262144 | 524.92 µs `[524.71 µs, 525.17 µs]` |
| final baseline | rank2 control | serial | large_n1048576 | 8.0622 ms `[7.8717 ms, 8.3312 ms]` |
| final baseline | rank2 control | max_threads_4 | large_n1048576 | 2.0828 ms `[2.0664 ms, 2.0944 ms]` |
| final baseline | generic | compact_single_rank2_serial | small_n4096 | 30.088 µs `[29.605 µs, 30.535 µs]` |
| final baseline | generic | compact_single_rank4_serial | small_n4096 | 40.802 µs `[40.033 µs, 41.783 µs]` |
| final baseline | generic | compact_single_rank8_serial | small_n4096 | 66.087 µs `[63.375 µs, 68.879 µs]` |
| final baseline | generic | compact_multi_rank4_serial | small_n4096 | 29.996 µs `[29.292 µs, 31.075 µs]` |
| final baseline | generic | compact_multi_rank8_serial | small_n4096 | 53.881 µs `[51.556 µs, 56.119 µs]` |
| final baseline | generic | rank4_nonunit_source_serial | small_n4096 | 42.105 µs `[41.262 µs, 43.226 µs]` |
| final baseline | generic | rank4_negative_source_serial | small_n4096 | 41.771 µs `[41.220 µs, 42.354 µs]` |
| final baseline | generic | compact_single_rank2_max_threads_4 | small_n4096 | 30.310 µs `[29.746 µs, 30.877 µs]` |
| final baseline | generic | compact_single_rank4_max_threads_4 | small_n4096 | 43.695 µs `[41.838 µs, 45.547 µs]` |
| final baseline | generic | compact_single_rank8_max_threads_4 | small_n4096 | 64.531 µs `[62.701 µs, 67.127 µs]` |
| final baseline | generic | compact_multi_rank4_max_threads_4 | small_n4096 | 31.596 µs `[30.766 µs, 32.499 µs]` |
| final baseline | generic | compact_multi_rank8_max_threads_4 | small_n4096 | 52.223 µs `[50.377 µs, 54.007 µs]` |
| final baseline | generic | rank4_nonunit_source_max_threads_4 | small_n4096 | 42.413 µs `[40.478 µs, 44.397 µs]` |
| final baseline | generic | rank4_negative_source_max_threads_4 | small_n4096 | 41.161 µs `[40.414 µs, 41.905 µs]` |
| final baseline | generic | compact_single_rank2_serial | near_threshold_n32768 | 244.27 µs `[240.51 µs, 249.45 µs]` |
| final baseline | generic | compact_single_rank4_serial | near_threshold_n32768 | 332.55 µs `[322.69 µs, 342.68 µs]` |
| final baseline | generic | compact_single_rank8_serial | near_threshold_n32768 | 589.05 µs `[580.98 µs, 598.06 µs]` |
| final baseline | generic | compact_multi_rank4_serial | near_threshold_n32768 | 232.54 µs `[228.82 µs, 236.70 µs]` |
| final baseline | generic | compact_multi_rank8_serial | near_threshold_n32768 | 474.64 µs `[469.94 µs, 481.18 µs]` |
| final baseline | generic | rank4_nonunit_source_serial | near_threshold_n32768 | 324.51 µs `[320.33 µs, 329.69 µs]` |
| final baseline | generic | rank4_negative_source_serial | near_threshold_n32768 | 332.34 µs `[326.29 µs, 339.09 µs]` |
| final baseline | generic | compact_single_rank2_max_threads_4 | near_threshold_n32768 | 247.64 µs `[239.85 µs, 255.84 µs]` |
| final baseline | generic | compact_single_rank4_max_threads_4 | near_threshold_n32768 | 321.48 µs `[318.02 µs, 327.71 µs]` |
| final baseline | generic | compact_single_rank8_max_threads_4 | near_threshold_n32768 | 611.84 µs `[603.15 µs, 632.39 µs]` |
| final baseline | generic | compact_multi_rank4_max_threads_4 | near_threshold_n32768 | 234.84 µs `[229.76 µs, 240.43 µs]` |
| final baseline | generic | compact_multi_rank8_max_threads_4 | near_threshold_n32768 | 489.86 µs `[470.47 µs, 515.24 µs]` |
| final baseline | generic | rank4_nonunit_source_max_threads_4 | near_threshold_n32768 | 322.43 µs `[319.66 µs, 327.72 µs]` |
| final baseline | generic | rank4_negative_source_max_threads_4 | near_threshold_n32768 | 341.34 µs `[329.02 µs, 351.27 µs]` |
| final baseline | generic | compact_single_rank2_serial | medium_n262144 | 1.9327 ms `[1.9108 ms, 1.9576 ms]` |
| final baseline | generic | compact_single_rank4_serial | medium_n262144 | 2.5954 ms `[2.5414 ms, 2.6328 ms]` |
| final baseline | generic | compact_single_rank8_serial | medium_n262144 | 4.8418 ms `[4.6967 ms, 5.0655 ms]` |
| final baseline | generic | compact_multi_rank4_serial | medium_n262144 | 2.1708 ms `[2.0959 ms, 2.2612 ms]` |
| final baseline | generic | compact_multi_rank8_serial | medium_n262144 | 3.9640 ms `[3.8096 ms, 4.0996 ms]` |
| final baseline | generic | rank4_nonunit_source_serial | medium_n262144 | 2.6445 ms `[2.6026 ms, 2.7130 ms]` |
| final baseline | generic | rank4_negative_source_serial | medium_n262144 | 2.6326 ms `[2.5846 ms, 2.6778 ms]` |
| final baseline | generic | compact_single_rank2_max_threads_4 | medium_n262144 | 586.53 µs `[581.84 µs, 588.93 µs]` |
| final baseline | generic | compact_single_rank4_max_threads_4 | medium_n262144 | 695.56 µs `[691.86 µs, 698.58 µs]` |
| final baseline | generic | compact_single_rank8_max_threads_4 | medium_n262144 | 1.2478 ms `[1.2418 ms, 1.2522 ms]` |
| final baseline | generic | compact_multi_rank4_max_threads_4 | medium_n262144 | 2.2159 ms `[2.1426 ms, 2.2750 ms]` |
| final baseline | generic | compact_multi_rank8_max_threads_4 | medium_n262144 | 3.9500 ms `[3.8515 ms, 4.0800 ms]` |
| final baseline | generic | rank4_nonunit_source_max_threads_4 | medium_n262144 | 696.65 µs `[691.59 µs, 700.79 µs]` |
| final baseline | generic | rank4_negative_source_max_threads_4 | medium_n262144 | 700.12 µs `[699.42 µs, 700.78 µs]` |
| final baseline | generic | compact_single_rank2_serial | large_n1048576 | 8.3616 ms `[8.0401 ms, 8.6125 ms]` |
| final baseline | generic | compact_single_rank4_serial | large_n1048576 | 10.666 ms `[10.311 ms, 11.138 ms]` |
| final baseline | generic | compact_single_rank8_serial | large_n1048576 | 19.807 ms `[19.465 ms, 20.467 ms]` |
| final baseline | generic | compact_multi_rank4_serial | large_n1048576 | 8.3350 ms `[8.1642 ms, 8.4609 ms]` |
| final baseline | generic | compact_multi_rank8_serial | large_n1048576 | 15.669 ms `[15.511 ms, 15.949 ms]` |
| final baseline | generic | rank4_nonunit_source_serial | large_n1048576 | 10.720 ms `[10.443 ms, 11.069 ms]` |
| final baseline | generic | rank4_negative_source_serial | large_n1048576 | 10.798 ms `[10.657 ms, 10.932 ms]` |
| final baseline | generic | compact_single_rank2_max_threads_4 | large_n1048576 | 2.0889 ms `[2.0714 ms, 2.0993 ms]` |
| final baseline | generic | compact_single_rank4_max_threads_4 | large_n1048576 | 2.7612 ms `[2.7534 ms, 2.7682 ms]` |
| final baseline | generic | compact_single_rank8_max_threads_4 | large_n1048576 | 4.8399 ms `[4.8235 ms, 4.8520 ms]` |
| final baseline | generic | compact_multi_rank4_max_threads_4 | large_n1048576 | 5.4449 ms `[5.4015 ms, 5.4901 ms]` |
| final baseline | generic | compact_multi_rank8_max_threads_4 | large_n1048576 | 15.688 ms `[15.422 ms, 15.967 ms]` |
| final baseline | generic | rank4_nonunit_source_max_threads_4 | large_n1048576 | 2.7544 ms `[2.7457 ms, 2.7635 ms]` |
| final baseline | generic | rank4_negative_source_max_threads_4 | large_n1048576 | 2.7668 ms `[2.7492 ms, 2.8107 ms]` |
| final candidate | rank2 control | serial | small_n4096 | 11.462 µs `[11.359 µs, 11.688 µs]` |
| final candidate | rank2 control | max_threads_4 | small_n4096 | 11.417 µs `[11.329 µs, 11.616 µs]` |
| final candidate | rank2 control | serial | near_threshold_n32768 | 96.550 µs `[92.864 µs, 100.30 µs]` |
| final candidate | rank2 control | max_threads_4 | near_threshold_n32768 | 97.648 µs `[95.790 µs, 100.82 µs]` |
| final candidate | rank2 control | serial | medium_n262144 | 767.93 µs `[747.78 µs, 793.03 µs]` |
| final candidate | rank2 control | max_threads_4 | medium_n262144 | 236.38 µs `[236.25 µs, 236.62 µs]` |
| final candidate | rank2 control | serial | large_n1048576 | 3.1794 ms `[3.0754 ms, 3.2547 ms]` |
| final candidate | rank2 control | max_threads_4 | large_n1048576 | 976.42 µs `[971.96 µs, 981.06 µs]` |
| final candidate | generic | compact_single_rank2_serial | small_n4096 | 11.487 µs `[11.383 µs, 11.755 µs]` |
| final candidate | generic | compact_single_rank4_serial | small_n4096 | 12.218 µs `[11.649 µs, 12.866 µs]` |
| final candidate | generic | compact_single_rank8_serial | small_n4096 | 11.421 µs `[11.307 µs, 11.556 µs]` |
| final candidate | generic | compact_multi_rank4_serial | small_n4096 | 7.6241 µs `[7.2063 µs, 7.9209 µs]` |
| final candidate | generic | compact_multi_rank8_serial | small_n4096 | 6.1143 µs `[5.9724 µs, 6.1798 µs]` |
| final candidate | generic | rank4_nonunit_source_serial | small_n4096 | 11.974 µs `[11.445 µs, 12.616 µs]` |
| final candidate | generic | rank4_negative_source_serial | small_n4096 | 11.901 µs `[11.745 µs, 12.152 µs]` |
| final candidate | generic | compact_single_rank2_max_threads_4 | small_n4096 | 11.425 µs `[11.309 µs, 11.630 µs]` |
| final candidate | generic | compact_single_rank4_max_threads_4 | small_n4096 | 11.933 µs `[11.714 µs, 12.234 µs]` |
| final candidate | generic | compact_single_rank8_max_threads_4 | small_n4096 | 11.513 µs `[11.400 µs, 11.711 µs]` |
| final candidate | generic | compact_multi_rank4_max_threads_4 | small_n4096 | 7.4975 µs `[7.3776 µs, 7.6915 µs]` |
| final candidate | generic | compact_multi_rank8_max_threads_4 | small_n4096 | 6.0744 µs `[5.9863 µs, 6.2006 µs]` |
| final candidate | generic | rank4_nonunit_source_max_threads_4 | small_n4096 | 11.576 µs `[11.300 µs, 11.981 µs]` |
| final candidate | generic | rank4_negative_source_max_threads_4 | small_n4096 | 12.327 µs `[11.806 µs, 12.662 µs]` |
| final candidate | generic | compact_single_rank2_serial | near_threshold_n32768 | 93.931 µs `[92.550 µs, 95.138 µs]` |
| final candidate | generic | compact_single_rank4_serial | near_threshold_n32768 | 94.120 µs `[91.410 µs, 96.701 µs]` |
| final candidate | generic | compact_single_rank8_serial | near_threshold_n32768 | 92.647 µs `[90.763 µs, 94.733 µs]` |
| final candidate | generic | compact_multi_rank4_serial | near_threshold_n32768 | 57.834 µs `[56.978 µs, 58.933 µs]` |
| final candidate | generic | compact_multi_rank8_serial | near_threshold_n32768 | 48.239 µs `[47.436 µs, 50.007 µs]` |
| final candidate | generic | rank4_nonunit_source_serial | near_threshold_n32768 | 92.346 µs `[91.406 µs, 94.061 µs]` |
| final candidate | generic | rank4_negative_source_serial | near_threshold_n32768 | 95.015 µs `[94.243 µs, 95.945 µs]` |
| final candidate | generic | compact_single_rank2_max_threads_4 | near_threshold_n32768 | 91.934 µs `[90.907 µs, 92.915 µs]` |
| final candidate | generic | compact_single_rank4_max_threads_4 | near_threshold_n32768 | 92.384 µs `[90.980 µs, 94.287 µs]` |
| final candidate | generic | compact_single_rank8_max_threads_4 | near_threshold_n32768 | 92.657 µs `[90.755 µs, 94.905 µs]` |
| final candidate | generic | compact_multi_rank4_max_threads_4 | near_threshold_n32768 | 61.044 µs `[57.705 µs, 64.507 µs]` |
| final candidate | generic | compact_multi_rank8_max_threads_4 | near_threshold_n32768 | 49.218 µs `[47.737 µs, 50.738 µs]` |
| final candidate | generic | rank4_nonunit_source_max_threads_4 | near_threshold_n32768 | 91.470 µs `[90.636 µs, 92.743 µs]` |
| final candidate | generic | rank4_negative_source_max_threads_4 | near_threshold_n32768 | 95.141 µs `[94.186 µs, 97.123 µs]` |
| final candidate | generic | compact_single_rank2_serial | medium_n262144 | 743.27 µs `[731.00 µs, 762.12 µs]` |
| final candidate | generic | compact_single_rank4_serial | medium_n262144 | 750.00 µs `[728.36 µs, 788.16 µs]` |
| final candidate | generic | compact_single_rank8_serial | medium_n262144 | 783.54 µs `[745.49 µs, 813.01 µs]` |
| final candidate | generic | compact_multi_rank4_serial | medium_n262144 | 466.96 µs `[458.84 µs, 477.24 µs]` |
| final candidate | generic | compact_multi_rank8_serial | medium_n262144 | 385.31 µs `[378.15 µs, 394.67 µs]` |
| final candidate | generic | rank4_nonunit_source_serial | medium_n262144 | 753.07 µs `[736.02 µs, 776.41 µs]` |
| final candidate | generic | rank4_negative_source_serial | medium_n262144 | 789.32 µs `[764.78 µs, 824.46 µs]` |
| final candidate | generic | compact_single_rank2_max_threads_4 | medium_n262144 | 235.81 µs `[235.11 µs, 236.36 µs]` |
| final candidate | generic | compact_single_rank4_max_threads_4 | medium_n262144 | 230.48 µs `[228.55 µs, 233.07 µs]` |
| final candidate | generic | compact_single_rank8_max_threads_4 | medium_n262144 | 234.97 µs `[232.54 µs, 236.85 µs]` |
| final candidate | generic | compact_multi_rank4_max_threads_4 | medium_n262144 | 464.66 µs `[459.41 µs, 476.60 µs]` |
| final candidate | generic | compact_multi_rank8_max_threads_4 | medium_n262144 | 383.00 µs `[379.60 µs, 385.82 µs]` |
| final candidate | generic | rank4_nonunit_source_max_threads_4 | medium_n262144 | 235.21 µs `[233.93 µs, 236.51 µs]` |
| final candidate | generic | rank4_negative_source_max_threads_4 | medium_n262144 | 244.50 µs `[244.42 µs, 244.60 µs]` |
| final candidate | generic | compact_single_rank2_serial | large_n1048576 | 2.9467 ms `[2.9086 ms, 3.0191 ms]` |
| final candidate | generic | compact_single_rank4_serial | large_n1048576 | 3.0503 ms `[2.9534 ms, 3.1388 ms]` |
| final candidate | generic | compact_single_rank8_serial | large_n1048576 | 3.0965 ms `[2.9919 ms, 3.1881 ms]` |
| final candidate | generic | compact_multi_rank4_serial | large_n1048576 | 1.8572 ms `[1.8363 ms, 1.8908 ms]` |
| final candidate | generic | compact_multi_rank8_serial | large_n1048576 | 1.5522 ms `[1.5143 ms, 1.5880 ms]` |
| final candidate | generic | rank4_nonunit_source_serial | large_n1048576 | 2.9389 ms `[2.9020 ms, 3.0145 ms]` |
| final candidate | generic | rank4_negative_source_serial | large_n1048576 | 3.0940 ms `[3.0344 ms, 3.2246 ms]` |
| final candidate | generic | compact_single_rank2_max_threads_4 | large_n1048576 | 913.91 µs `[913.55 µs, 914.34 µs]` |
| final candidate | generic | compact_single_rank4_max_threads_4 | large_n1048576 | 901.35 µs `[892.05 µs, 907.53 µs]` |
| final candidate | generic | compact_single_rank8_max_threads_4 | large_n1048576 | 910.59 µs `[903.35 µs, 913.36 µs]` |
| final candidate | generic | compact_multi_rank4_max_threads_4 | large_n1048576 | 622.20 µs `[621.37 µs, 623.06 µs]` |
| final candidate | generic | compact_multi_rank8_max_threads_4 | large_n1048576 | 1.5465 ms `[1.5079 ms, 1.6180 ms]` |
| final candidate | generic | rank4_nonunit_source_max_threads_4 | large_n1048576 | 915.52 µs `[914.31 µs, 916.28 µs]` |
| final candidate | generic | rank4_negative_source_max_threads_4 | large_n1048576 | 944.66 µs `[943.98 µs, 945.21 µs]` |

All final gates are **PASS**. At medium size, compact single-axis rank 4/8 serial speedups are 3.46x/6.18x and four-thread speedups are 3.02x/5.31x. Multi-axis rank 4/8 serial speedups are 4.65x/10.29x and four-thread speedups are 4.77x/10.31x. Non-unit/negative serial speedups are 3.51x/3.34x. Candidate single-axis rank-8/rank-2 ratio is 1.054, below 1.5. Every selected generic and rank-2 control case improved with `p < 0.05`; no control regressed.

Attempt 1 remains recorded as a failed 2.93x result; no gate or case changed. Correctness, repository verification, and exact-final review remain pending.

## Final promotion pair

After preflight findings, candidate attempts that reused inner scratch through rank 8 were retained as non-promotable because they missed the frozen 3x rank-4 serial gate. Final candidate `d9c1446` keeps inline construction through rank 8 and reuses heap scratch only above the inline limit. A fresh complete pair used named baseline `issue239-final2-base`: baseline and candidate both passed their load gates; all cases remained frozen.

| phase | family | variant/context | size | estimate `[low, high]` |
|---|---|---|---|---:|
| promotion baseline | rank2 control | serial | small_n4096 | 31.446 µs `[30.299 µs, 32.685 µs]` |
| promotion baseline | rank2 control | max_threads_4 | small_n4096 | 31.252 µs `[30.745 µs, 32.249 µs]` |
| promotion baseline | rank2 control | serial | near_threshold_n32768 | 249.53 µs `[244.60 µs, 253.66 µs]` |
| promotion baseline | rank2 control | max_threads_4 | near_threshold_n32768 | 245.67 µs `[243.09 µs, 248.16 µs]` |
| promotion baseline | rank2 control | serial | medium_n262144 | 2.0358 ms `[2.0052 ms, 2.0707 ms]` |
| promotion baseline | rank2 control | max_threads_4 | medium_n262144 | 532.58 µs `[532.43 µs, 532.85 µs]` |
| promotion baseline | rank2 control | serial | large_n1048576 | 7.9776 ms `[7.8159 ms, 8.2314 ms]` |
| promotion baseline | rank2 control | max_threads_4 | large_n1048576 | 2.0285 ms `[2.0034 ms, 2.0608 ms]` |
| promotion baseline | generic | compact_single_rank2_serial | small_n4096 | 31.309 µs `[30.509 µs, 31.808 µs]` |
| promotion baseline | generic | compact_single_rank4_serial | small_n4096 | 43.163 µs `[41.967 µs, 45.448 µs]` |
| promotion baseline | generic | compact_single_rank8_serial | small_n4096 | 81.555 µs `[77.981 µs, 83.117 µs]` |
| promotion baseline | generic | compact_multi_rank4_serial | small_n4096 | 31.955 µs `[30.473 µs, 33.312 µs]` |
| promotion baseline | generic | compact_multi_rank8_serial | small_n4096 | 64.529 µs `[63.350 µs, 65.628 µs]` |
| promotion baseline | generic | rank4_nonunit_source_serial | small_n4096 | 42.915 µs `[41.121 µs, 44.471 µs]` |
| promotion baseline | generic | rank4_negative_source_serial | small_n4096 | 42.988 µs `[41.343 µs, 44.836 µs]` |
| promotion baseline | generic | compact_single_rank2_max_threads_4 | small_n4096 | 32.162 µs `[31.045 µs, 33.358 µs]` |
| promotion baseline | generic | compact_single_rank4_max_threads_4 | small_n4096 | 42.558 µs `[40.857 µs, 44.543 µs]` |
| promotion baseline | generic | compact_single_rank8_max_threads_4 | small_n4096 | 66.778 µs `[64.943 µs, 67.982 µs]` |
| promotion baseline | generic | compact_multi_rank4_max_threads_4 | small_n4096 | 34.398 µs `[33.720 µs, 35.297 µs]` |
| promotion baseline | generic | compact_multi_rank8_max_threads_4 | small_n4096 | 55.411 µs `[53.966 µs, 56.853 µs]` |
| promotion baseline | generic | rank4_nonunit_source_max_threads_4 | small_n4096 | 41.481 µs `[40.899 µs, 42.176 µs]` |
| promotion baseline | generic | rank4_negative_source_max_threads_4 | small_n4096 | 41.655 µs `[40.886 µs, 42.369 µs]` |
| promotion baseline | generic | compact_single_rank2_serial | near_threshold_n32768 | 252.59 µs `[247.07 µs, 261.74 µs]` |
| promotion baseline | generic | compact_single_rank4_serial | near_threshold_n32768 | 335.42 µs `[324.18 µs, 352.35 µs]` |
| promotion baseline | generic | compact_single_rank8_serial | near_threshold_n32768 | 640.75 µs `[622.33 µs, 661.81 µs]` |
| promotion baseline | generic | compact_multi_rank4_serial | near_threshold_n32768 | 290.91 µs `[276.50 µs, 300.75 µs]` |
| promotion baseline | generic | compact_multi_rank8_serial | near_threshold_n32768 | 511.69 µs `[499.95 µs, 525.51 µs]` |
| promotion baseline | generic | rank4_nonunit_source_serial | near_threshold_n32768 | 353.80 µs `[342.65 µs, 367.12 µs]` |
| promotion baseline | generic | rank4_negative_source_serial | near_threshold_n32768 | 352.82 µs `[348.87 µs, 362.21 µs]` |
| promotion baseline | generic | compact_single_rank2_max_threads_4 | near_threshold_n32768 | 265.84 µs `[259.23 µs, 274.88 µs]` |
| promotion baseline | generic | compact_single_rank4_max_threads_4 | near_threshold_n32768 | 361.16 µs `[348.55 µs, 378.36 µs]` |
| promotion baseline | generic | compact_single_rank8_max_threads_4 | near_threshold_n32768 | 576.82 µs `[557.44 µs, 597.13 µs]` |
| promotion baseline | generic | compact_multi_rank4_max_threads_4 | near_threshold_n32768 | 281.39 µs `[273.02 µs, 289.23 µs]` |
| promotion baseline | generic | compact_multi_rank8_max_threads_4 | near_threshold_n32768 | 503.59 µs `[493.58 µs, 516.39 µs]` |
| promotion baseline | generic | rank4_nonunit_source_max_threads_4 | near_threshold_n32768 | 361.45 µs `[348.22 µs, 371.40 µs]` |
| promotion baseline | generic | rank4_negative_source_max_threads_4 | near_threshold_n32768 | 357.49 µs `[342.46 µs, 368.50 µs]` |
| promotion baseline | generic | compact_single_rank2_serial | medium_n262144 | 2.0998 ms `[2.0482 ms, 2.1455 ms]` |
| promotion baseline | generic | compact_single_rank4_serial | medium_n262144 | 2.7166 ms `[2.6761 ms, 2.7823 ms]` |
| promotion baseline | generic | compact_single_rank8_serial | medium_n262144 | 4.9495 ms `[4.8841 ms, 5.0404 ms]` |
| promotion baseline | generic | compact_multi_rank4_serial | medium_n262144 | 2.2165 ms `[2.1644 ms, 2.2646 ms]` |
| promotion baseline | generic | compact_multi_rank8_serial | medium_n262144 | 3.9443 ms `[3.8957 ms, 3.9835 ms]` |
| promotion baseline | generic | rank4_nonunit_source_serial | medium_n262144 | 2.7872 ms `[2.6840 ms, 2.9085 ms]` |
| promotion baseline | generic | rank4_negative_source_serial | medium_n262144 | 2.7500 ms `[2.6622 ms, 2.8355 ms]` |
| promotion baseline | generic | compact_single_rank2_max_threads_4 | medium_n262144 | 532.60 µs `[532.45 µs, 532.87 µs]` |
| promotion baseline | generic | compact_single_rank4_max_threads_4 | medium_n262144 | 697.71 µs `[694.48 µs, 703.47 µs]` |
| promotion baseline | generic | compact_single_rank8_max_threads_4 | medium_n262144 | 1.2313 ms `[1.2286 ms, 1.2362 ms]` |
| promotion baseline | generic | compact_multi_rank4_max_threads_4 | medium_n262144 | 2.1548 ms `[2.1049 ms, 2.2217 ms]` |
| promotion baseline | generic | compact_multi_rank8_max_threads_4 | medium_n262144 | 4.1707 ms `[4.0635 ms, 4.2558 ms]` |
| promotion baseline | generic | rank4_nonunit_source_max_threads_4 | medium_n262144 | 701.47 µs `[699.90 µs, 702.96 µs]` |
| promotion baseline | generic | rank4_negative_source_max_threads_4 | medium_n262144 | 698.43 µs `[696.39 µs, 699.71 µs]` |
| promotion baseline | generic | compact_single_rank2_serial | large_n1048576 | 8.3124 ms `[8.1030 ms, 8.4941 ms]` |
| promotion baseline | generic | compact_single_rank4_serial | large_n1048576 | 10.830 ms `[10.593 ms, 11.143 ms]` |
| promotion baseline | generic | compact_single_rank8_serial | large_n1048576 | 20.533 ms `[20.022 ms, 21.105 ms]` |
| promotion baseline | generic | compact_multi_rank4_serial | large_n1048576 | 8.6614 ms `[8.4424 ms, 8.9650 ms]` |
| promotion baseline | generic | compact_multi_rank8_serial | large_n1048576 | 15.850 ms `[15.437 ms, 16.487 ms]` |
| promotion baseline | generic | rank4_nonunit_source_serial | large_n1048576 | 10.635 ms `[10.509 ms, 10.768 ms]` |
| promotion baseline | generic | rank4_negative_source_serial | large_n1048576 | 10.884 ms `[10.565 ms, 11.276 ms]` |
| promotion baseline | generic | compact_single_rank2_max_threads_4 | large_n1048576 | 2.0706 ms `[2.0657 ms, 2.0763 ms]` |
| promotion baseline | generic | compact_single_rank4_max_threads_4 | large_n1048576 | 2.7541 ms `[2.7436 ms, 2.7636 ms]` |
| promotion baseline | generic | compact_single_rank8_max_threads_4 | large_n1048576 | 4.7716 ms `[4.6874 ms, 4.8180 ms]` |
| promotion baseline | generic | compact_multi_rank4_max_threads_4 | large_n1048576 | 5.4478 ms `[5.4158 ms, 5.4878 ms]` |
| promotion baseline | generic | compact_multi_rank8_max_threads_4 | large_n1048576 | 15.837 ms `[15.618 ms, 16.188 ms]` |
| promotion baseline | generic | rank4_nonunit_source_max_threads_4 | large_n1048576 | 2.7558 ms `[2.7458 ms, 2.7650 ms]` |
| promotion baseline | generic | rank4_negative_source_max_threads_4 | large_n1048576 | 2.7437 ms `[2.7408 ms, 2.7481 ms]` |
| promotion candidate | rank2 control | serial | small_n4096 | 12.740 µs `[12.399 µs, 13.148 µs]` |
| promotion candidate | rank2 control | max_threads_4 | small_n4096 | 12.482 µs `[12.370 µs, 12.684 µs]` |
| promotion candidate | rank2 control | serial | near_threshold_n32768 | 105.29 µs `[103.68 µs, 107.12 µs]` |
| promotion candidate | rank2 control | max_threads_4 | near_threshold_n32768 | 106.06 µs `[103.49 µs, 108.54 µs]` |
| promotion candidate | rank2 control | serial | medium_n262144 | 829.55 µs `[808.81 µs, 856.31 µs]` |
| promotion candidate | rank2 control | max_threads_4 | medium_n262144 | 228.15 µs `[227.92 µs, 228.47 µs]` |
| promotion candidate | rank2 control | serial | large_n1048576 | 3.4791 ms `[3.4771 ms, 3.4820 ms]` |
| promotion candidate | rank2 control | max_threads_4 | large_n1048576 | 894.47 µs `[891.79 µs, 896.49 µs]` |
| promotion candidate | generic | compact_single_rank2_serial | small_n4096 | 12.603 µs `[12.426 µs, 12.973 µs]` |
| promotion candidate | generic | compact_single_rank4_serial | small_n4096 | 12.768 µs `[12.448 µs, 13.097 µs]` |
| promotion candidate | generic | compact_single_rank8_serial | small_n4096 | 12.510 µs `[12.253 µs, 12.846 µs]` |
| promotion candidate | generic | compact_multi_rank4_serial | small_n4096 | 9.9890 µs `[9.6818 µs, 10.254 µs]` |
| promotion candidate | generic | compact_multi_rank8_serial | small_n4096 | 9.2588 µs `[9.0255 µs, 9.4984 µs]` |
| promotion candidate | generic | rank4_nonunit_source_serial | small_n4096 | 12.640 µs `[11.932 µs, 13.282 µs]` |
| promotion candidate | generic | rank4_negative_source_serial | small_n4096 | 12.976 µs `[12.725 µs, 13.392 µs]` |
| promotion candidate | generic | compact_single_rank2_max_threads_4 | small_n4096 | 12.949 µs `[12.666 µs, 13.110 µs]` |
| promotion candidate | generic | compact_single_rank4_max_threads_4 | small_n4096 | 12.988 µs `[12.805 µs, 13.353 µs]` |
| promotion candidate | generic | compact_single_rank8_max_threads_4 | small_n4096 | 13.009 µs `[12.763 µs, 13.333 µs]` |
| promotion candidate | generic | compact_multi_rank4_max_threads_4 | small_n4096 | 10.140 µs `[9.9293 µs, 10.298 µs]` |
| promotion candidate | generic | compact_multi_rank8_max_threads_4 | small_n4096 | 9.2991 µs `[9.1036 µs, 9.5170 µs]` |
| promotion candidate | generic | rank4_nonunit_source_max_threads_4 | small_n4096 | 13.040 µs `[12.899 µs, 13.213 µs]` |
| promotion candidate | generic | rank4_negative_source_max_threads_4 | small_n4096 | 13.858 µs `[13.855 µs, 13.864 µs]` |
| promotion candidate | generic | compact_single_rank2_serial | near_threshold_n32768 | 103.86 µs `[101.81 µs, 106.39 µs]` |
| promotion candidate | generic | compact_single_rank4_serial | near_threshold_n32768 | 101.56 µs `[100.25 µs, 102.59 µs]` |
| promotion candidate | generic | compact_single_rank8_serial | near_threshold_n32768 | 103.84 µs `[101.91 µs, 105.31 µs]` |
| promotion candidate | generic | compact_multi_rank4_serial | near_threshold_n32768 | 82.084 µs `[80.710 µs, 83.086 µs]` |
| promotion candidate | generic | compact_multi_rank8_serial | near_threshold_n32768 | 74.197 µs `[73.020 µs, 75.134 µs]` |
| promotion candidate | generic | rank4_nonunit_source_serial | near_threshold_n32768 | 104.52 µs `[103.06 µs, 106.03 µs]` |
| promotion candidate | generic | rank4_negative_source_serial | near_threshold_n32768 | 103.64 µs `[100.85 µs, 106.23 µs]` |
| promotion candidate | generic | compact_single_rank2_max_threads_4 | near_threshold_n32768 | 102.68 µs `[100.70 µs, 105.37 µs]` |
| promotion candidate | generic | compact_single_rank4_max_threads_4 | near_threshold_n32768 | 99.426 µs `[98.670 µs, 100.82 µs]` |
| promotion candidate | generic | compact_single_rank8_max_threads_4 | near_threshold_n32768 | 100.81 µs `[99.553 µs, 103.58 µs]` |
| promotion candidate | generic | compact_multi_rank4_max_threads_4 | near_threshold_n32768 | 79.325 µs `[76.437 µs, 82.190 µs]` |
| promotion candidate | generic | compact_multi_rank8_max_threads_4 | near_threshold_n32768 | 72.343 µs `[70.882 µs, 74.231 µs]` |
| promotion candidate | generic | rank4_nonunit_source_max_threads_4 | near_threshold_n32768 | 103.23 µs `[101.51 µs, 105.01 µs]` |
| promotion candidate | generic | rank4_negative_source_max_threads_4 | near_threshold_n32768 | 107.49 µs `[105.58 µs, 109.28 µs]` |
| promotion candidate | generic | compact_single_rank2_serial | medium_n262144 | 816.37 µs `[800.66 µs, 845.08 µs]` |
| promotion candidate | generic | compact_single_rank4_serial | medium_n262144 | 814.69 µs `[797.04 µs, 839.51 µs]` |
| promotion candidate | generic | compact_single_rank8_serial | medium_n262144 | 846.08 µs `[822.24 µs, 863.88 µs]` |
| promotion candidate | generic | compact_multi_rank4_serial | medium_n262144 | 639.79 µs `[626.34 µs, 650.90 µs]` |
| promotion candidate | generic | compact_multi_rank8_serial | medium_n262144 | 578.13 µs `[564.32 µs, 589.55 µs]` |
| promotion candidate | generic | rank4_nonunit_source_serial | medium_n262144 | 827.94 µs `[814.79 µs, 839.76 µs]` |
| promotion candidate | generic | rank4_negative_source_serial | medium_n262144 | 822.11 µs `[788.66 µs, 860.42 µs]` |
| promotion candidate | generic | compact_single_rank2_max_threads_4 | medium_n262144 | 229.27 µs `[228.85 µs, 229.76 µs]` |
| promotion candidate | generic | compact_single_rank4_max_threads_4 | medium_n262144 | 228.87 µs `[228.76 µs, 229.03 µs]` |
| promotion candidate | generic | compact_single_rank8_max_threads_4 | medium_n262144 | 228.63 µs `[228.54 µs, 228.76 µs]` |
| promotion candidate | generic | compact_multi_rank4_max_threads_4 | medium_n262144 | 635.30 µs `[624.55 µs, 645.36 µs]` |
| promotion candidate | generic | compact_multi_rank8_max_threads_4 | medium_n262144 | 612.58 µs `[612.39 µs, 612.90 µs]` |
| promotion candidate | generic | rank4_nonunit_source_max_threads_4 | medium_n262144 | 232.65 µs `[232.17 µs, 233.21 µs]` |
| promotion candidate | generic | rank4_negative_source_max_threads_4 | medium_n262144 | 244.72 µs `[243.77 µs, 246.39 µs]` |
| promotion candidate | generic | compact_single_rank2_serial | large_n1048576 | 3.3830 ms `[3.3370 ms, 3.4345 ms]` |
| promotion candidate | generic | compact_single_rank4_serial | large_n1048576 | 3.2730 ms `[3.1933 ms, 3.4061 ms]` |
| promotion candidate | generic | compact_single_rank8_serial | large_n1048576 | 3.3315 ms `[3.2913 ms, 3.3916 ms]` |
| promotion candidate | generic | compact_multi_rank4_serial | large_n1048576 | 2.6086 ms `[2.5754 ms, 2.6642 ms]` |
| promotion candidate | generic | compact_multi_rank8_serial | large_n1048576 | 2.3723 ms `[2.3538 ms, 2.4107 ms]` |
| promotion candidate | generic | rank4_nonunit_source_serial | large_n1048576 | 3.4269 ms `[3.3760 ms, 3.4848 ms]` |
| promotion candidate | generic | rank4_negative_source_serial | large_n1048576 | 3.5786 ms `[3.5389 ms, 3.6002 ms]` |
| promotion candidate | generic | compact_single_rank2_max_threads_4 | large_n1048576 | 894.10 µs `[893.02 µs, 894.80 µs]` |
| promotion candidate | generic | compact_single_rank4_max_threads_4 | large_n1048576 | 894.89 µs `[892.94 µs, 896.06 µs]` |
| promotion candidate | generic | compact_single_rank8_max_threads_4 | large_n1048576 | 892.75 µs `[891.36 µs, 893.95 µs]` |
| promotion candidate | generic | compact_multi_rank4_max_threads_4 | large_n1048576 | 685.09 µs `[684.40 µs, 686.26 µs]` |
| promotion candidate | generic | compact_multi_rank8_max_threads_4 | large_n1048576 | 2.4542 ms `[2.4523 ms, 2.4552 ms]` |
| promotion candidate | generic | rank4_nonunit_source_max_threads_4 | large_n1048576 | 910.03 µs `[907.21 µs, 911.70 µs]` |
| promotion candidate | generic | rank4_negative_source_max_threads_4 | large_n1048576 | 969.98 µs `[967.89 µs, 972.04 µs]` |

All promotion gates are **PASS**. At medium size, compact single-axis rank 4/8 serial speedups are 3.33x/5.85x and four-thread speedups are 3.05x/5.39x. Multi-axis rank 4/8 serial speedups are 3.46x/6.82x and four-thread speedups are 3.39x/6.81x. Non-unit/negative serial speedups are 3.37x/3.35x. Candidate rank-8/rank-2 ratio is 1.036. Every generic and rank-2 control case improved with `p < 0.05`; no control regressed.

All failed attempts remain recorded above; no gate/case/exclusion changed.

## Verification and review

Verification on promoted runtime candidate `d9c1446` plus the source-only helper
rename at `5a098c3`:

- focused default erased reduction/uninitialized tests: 42 passed
- focused parallel erased reduction/uninitialized/policy tests: 54 passed
- allocation contract: pass
- source-contract tests: 6 passed
- `cargo fmt --all -- --check`: pass
- `cargo test --workspace`: 912 passed, 9 ignored
- `cargo doc --workspace --no-deps`: pass
- deterministic repository-rules review: pass, no findings
- repository-rules review script: 83 passed

Local `cargo llvm-cov --workspace --features parallel` completed. Modified
`erased.rs` reached 88.17% line coverage, above the repository's 80% threshold.
The global checker still reports the same three unmodified files below their
configured thresholds (`reduce_view.rs`, `static_indexing_plan.rs`, and
`strided-perm/src/hptt/execute.rs`); hosted CI is authoritative for the PR gate.

`reviewer-flash` preflight of candidate `247b88a` found no Critical or Important
issues. Minor dispositions:

- initialized raw descriptors rely on Rust's non-aliasing borrow contract and
  the module-level descriptor non-overlap contract; the uninitialized pointer
  entry point keeps its explicit runtime overlap check;
- `MaxThreads(1)` intentionally selects the serial full-reduction path to avoid
  entering Rayon, so no behavior change was made;
- rank>8 inner scratch is now reused once per execution/worker while rank<=8
  keeps the faster inline construction selected by the frozen performance gate;
- the overflow-only span helper was renamed
  `check_reduce_layout_offset_arithmetic` to avoid overstating bounds ownership;
- a partial inner-axis fusion ground-truth test was added;
- the final promotion pair used the same L3 domain for baseline/candidate and
  all margins remain above their gates;
- rank-2 control improvement was predeclared and retained in the complete table;
- the large multi-axis rank-8 four-thread candidate (2.4542 ms) is 3.4% slower
  than its serial candidate (2.3723 ms), but still improves 6.45x over the
  matching four-thread baseline and does not affect any gate. It is retained
  rather than selectively excluded.

The helper rename does not alter timed execution, so the promotion benchmark
carries forward. The exact-final safety and evidence reviews both returned
**Correct-to-merge**, with no Critical or Important findings; the two evidence
Minors are corrected/disclosed above. Hosted CI remains the final merge gate.
