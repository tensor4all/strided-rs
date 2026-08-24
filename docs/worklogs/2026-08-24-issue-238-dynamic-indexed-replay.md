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

Dynamic-slice cases use F64 values and I64 starts, matching their rank-one
control. Dynamic-update cases use I32 values and I32 starts, likewise matching
that control. Non-unit dynamic slice varies only operand strides while
retaining the compact destination. Negative dynamic slice uses a negative final
operand stride and a validated base offset. Non-unit/negative dynamic update
varies only update strides; operand/destination stay compact so initial-copy
cost is comparable.
Physical buffers are sized from validated reachable spans. Start vectors,
allocation, plan compilation, and raw descriptor construction remain outside
timing. Existing rank-one groups remain fast-path controls.

Criterion uses 300 ms warmup, 10 samples, one-second measurement, release mode,
`RAYON_NUM_THREADS=4`, and benchmark thread override 4. Exactly `2^15` remains
serial even under the four-thread context because the policy parallelizes only
lengths greater than `MINTHREADLENGTH`; tests, not this timing row, cover
below/exact/above-threshold execution equivalence. Runs are sequential on
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
but do not optimize that family. This is an accepted possible outcome for
update: its mandatory full operand copy is included in both generic and
rank-one controls and may dominate the 1.0 ms fallback signal.

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

Compile reuses the existing private `validate_layout_span` and
`checked_replay_reset` helpers to validate source/destination signed spans and
every delta, including negative strides; it does not re-derive that arithmetic.
Execution reads/clamps starts once and computes the
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

`reviewer-flash` reviewed exact design commit `9fa1480` with high thinking and
a read-only boundary. Verdict: **Correct-to-merge**; benchmark implementation
may proceed. Its four Minor amendments (matching dtypes, accepted update-family
need-gate failure, helper reuse, and exact-threshold serial behavior) are folded
into the text above.

## Baseline evidence

Benchmark-only commit: `de42f23`. The complete baseline ran sequentially on CPUs 49-52 in L3 domain 48-55. The accepted four-second gate measured selected cores at 0.0-0.5% busy and every sibling at at most 0.5%, so the run is valid.

| family | variant/context | size | estimate `[low, high]` |
|---|---|---|---:|
| slice | compact_rank2_serial | small_n4096 | 75.179 µs `[74.173 µs, 76.644 µs]` |
| slice | compact_rank4_serial | small_n4096 | 141.99 µs `[140.61 µs, 145.47 µs]` |
| slice | compact_rank8_serial | small_n4096 | 279.44 µs `[277.57 µs, 283.65 µs]` |
| slice | rank2_nonunit_source_serial | small_n4096 | 73.961 µs `[72.642 µs, 75.780 µs]` |
| slice | rank2_negative_source_serial | small_n4096 | 72.460 µs `[71.525 µs, 74.305 µs]` |
| slice | compact_rank2_max_threads_4 | small_n4096 | 72.992 µs `[72.177 µs, 74.567 µs]` |
| slice | compact_rank4_max_threads_4 | small_n4096 | 142.49 µs `[136.92 µs, 147.81 µs]` |
| slice | compact_rank8_max_threads_4 | small_n4096 | 285.25 µs `[280.68 µs, 289.57 µs]` |
| slice | rank2_nonunit_source_max_threads_4 | small_n4096 | 71.932 µs `[71.185 µs, 73.023 µs]` |
| slice | rank2_negative_source_max_threads_4 | small_n4096 | 72.097 µs `[69.877 µs, 73.979 µs]` |
| slice | compact_rank2_serial | near_threshold_n32768 | 630.98 µs `[605.47 µs, 647.20 µs]` |
| slice | compact_rank4_serial | near_threshold_n32768 | 1.1794 ms `[1.1456 ms, 1.2150 ms]` |
| slice | compact_rank8_serial | near_threshold_n32768 | 2.3017 ms `[2.2396 ms, 2.3744 ms]` |
| slice | rank2_nonunit_source_serial | near_threshold_n32768 | 580.19 µs `[569.91 µs, 597.97 µs]` |
| slice | rank2_negative_source_serial | near_threshold_n32768 | 579.88 µs `[563.39 µs, 601.65 µs]` |
| slice | compact_rank2_max_threads_4 | near_threshold_n32768 | 578.17 µs `[567.30 µs, 596.12 µs]` |
| slice | compact_rank4_max_threads_4 | near_threshold_n32768 | 1.1506 ms `[1.1247 ms, 1.1741 ms]` |
| slice | compact_rank8_max_threads_4 | near_threshold_n32768 | 2.2253 ms `[2.1709 ms, 2.3333 ms]` |
| slice | rank2_nonunit_source_max_threads_4 | near_threshold_n32768 | 578.10 µs `[566.14 µs, 591.39 µs]` |
| slice | rank2_negative_source_max_threads_4 | near_threshold_n32768 | 582.94 µs `[571.79 µs, 599.08 µs]` |
| slice | compact_rank2_serial | medium_n262144 | 4.6385 ms `[4.5120 ms, 4.7837 ms]` |
| slice | compact_rank4_serial | medium_n262144 | 9.2671 ms `[8.9052 ms, 9.6984 ms]` |
| slice | compact_rank8_serial | medium_n262144 | 18.602 ms `[17.794 ms, 19.179 ms]` |
| slice | rank2_nonunit_source_serial | medium_n262144 | 4.7057 ms `[4.6052 ms, 4.8197 ms]` |
| slice | rank2_negative_source_serial | medium_n262144 | 4.6549 ms `[4.5562 ms, 4.7511 ms]` |
| slice | compact_rank2_max_threads_4 | medium_n262144 | 514.60 µs `[514.48 µs, 514.72 µs]` |
| slice | compact_rank4_max_threads_4 | medium_n262144 | 844.76 µs `[844.35 µs, 845.29 µs]` |
| slice | compact_rank8_max_threads_4 | medium_n262144 | 1.8646 ms `[1.8069 ms, 1.8881 ms]` |
| slice | rank2_nonunit_source_max_threads_4 | medium_n262144 | 514.57 µs `[514.27 µs, 515.21 µs]` |
| slice | rank2_negative_source_max_threads_4 | medium_n262144 | 514.20 µs `[514.02 µs, 514.46 µs]` |
| slice | compact_rank2_serial | large_n1048576 | 18.989 ms `[18.696 ms, 19.391 ms]` |
| slice | compact_rank4_serial | large_n1048576 | 37.419 ms `[36.610 ms, 38.306 ms]` |
| slice | compact_rank8_serial | large_n1048576 | 73.989 ms `[71.920 ms, 76.288 ms]` |
| slice | rank2_nonunit_source_serial | large_n1048576 | 18.864 ms `[18.484 ms, 19.687 ms]` |
| slice | rank2_negative_source_serial | large_n1048576 | 18.942 ms `[18.541 ms, 19.570 ms]` |
| slice | compact_rank2_max_threads_4 | large_n1048576 | 2.0241 ms `[2.0234 ms, 2.0248 ms]` |
| slice | compact_rank4_max_threads_4 | large_n1048576 | 3.3389 ms `[3.3374 ms, 3.3402 ms]` |
| slice | compact_rank8_max_threads_4 | large_n1048576 | 6.6720 ms `[6.6701 ms, 6.6734 ms]` |
| slice | rank2_nonunit_source_max_threads_4 | large_n1048576 | 2.0286 ms `[2.0279 ms, 2.0293 ms]` |
| slice | rank2_negative_source_max_threads_4 | large_n1048576 | 2.0248 ms `[2.0242 ms, 2.0253 ms]` |
| update | compact_rank2_serial | small_n4096 | 73.436 µs `[72.043 µs, 75.826 µs]` |
| update | compact_rank4_serial | small_n4096 | 160.12 µs `[157.04 µs, 163.83 µs]` |
| update | compact_rank8_serial | small_n4096 | 305.89 µs `[296.74 µs, 315.70 µs]` |
| update | rank2_nonunit_source_serial | small_n4096 | 74.939 µs `[73.547 µs, 76.378 µs]` |
| update | rank2_negative_source_serial | small_n4096 | 75.676 µs `[73.827 µs, 77.908 µs]` |
| update | compact_rank2_max_threads_4 | small_n4096 | 78.180 µs `[75.756 µs, 81.311 µs]` |
| update | compact_rank4_max_threads_4 | small_n4096 | 151.97 µs `[148.04 µs, 158.64 µs]` |
| update | compact_rank8_max_threads_4 | small_n4096 | 306.37 µs `[298.22 µs, 317.50 µs]` |
| update | rank2_nonunit_source_max_threads_4 | small_n4096 | 76.282 µs `[73.645 µs, 79.371 µs]` |
| update | rank2_negative_source_max_threads_4 | small_n4096 | 77.924 µs `[75.774 µs, 79.765 µs]` |
| update | compact_rank2_serial | near_threshold_n32768 | 598.46 µs `[580.13 µs, 621.48 µs]` |
| update | compact_rank4_serial | near_threshold_n32768 | 1.2740 ms `[1.2437 ms, 1.2999 ms]` |
| update | compact_rank8_serial | near_threshold_n32768 | 2.4287 ms `[2.3740 ms, 2.5020 ms]` |
| update | rank2_nonunit_source_serial | near_threshold_n32768 | 623.25 µs `[594.72 µs, 648.89 µs]` |
| update | rank2_negative_source_serial | near_threshold_n32768 | 594.43 µs `[587.43 µs, 609.76 µs]` |
| update | compact_rank2_max_threads_4 | near_threshold_n32768 | 620.78 µs `[601.97 µs, 645.01 µs]` |
| update | compact_rank4_max_threads_4 | near_threshold_n32768 | 1.1895 ms `[1.1740 ms, 1.2154 ms]` |
| update | compact_rank8_max_threads_4 | near_threshold_n32768 | 2.4440 ms `[2.3664 ms, 2.5364 ms]` |
| update | rank2_nonunit_source_max_threads_4 | near_threshold_n32768 | 617.04 µs `[600.39 µs, 647.13 µs]` |
| update | rank2_negative_source_max_threads_4 | near_threshold_n32768 | 632.06 µs `[617.55 µs, 646.96 µs]` |
| update | compact_rank2_serial | medium_n262144 | 5.2119 ms `[5.0754 ms, 5.2986 ms]` |
| update | compact_rank4_serial | medium_n262144 | 9.4503 ms `[9.1611 ms, 9.7755 ms]` |
| update | compact_rank8_serial | medium_n262144 | 19.210 ms `[18.573 ms, 19.871 ms]` |
| update | rank2_nonunit_source_serial | medium_n262144 | 4.9769 ms `[4.8593 ms, 5.1007 ms]` |
| update | rank2_negative_source_serial | medium_n262144 | 4.9850 ms `[4.7566 ms, 5.2830 ms]` |
| update | compact_rank2_max_threads_4 | medium_n262144 | 583.37 µs `[582.97 µs, 584.12 µs]` |
| update | compact_rank4_max_threads_4 | medium_n262144 | 1.0080 ms `[1.0052 ms, 1.0100 ms]` |
| update | compact_rank8_max_threads_4 | medium_n262144 | 1.6914 ms `[1.6905 ms, 1.6925 ms]` |
| update | rank2_nonunit_source_max_threads_4 | medium_n262144 | 585.76 µs `[585.32 µs, 586.60 µs]` |
| update | rank2_negative_source_max_threads_4 | medium_n262144 | 584.27 µs `[584.05 µs, 584.83 µs]` |
| update | compact_rank2_serial | large_n1048576 | 19.261 ms `[18.841 ms, 19.828 ms]` |
| update | compact_rank4_serial | large_n1048576 | 39.249 ms `[38.216 ms, 40.383 ms]` |
| update | compact_rank8_serial | large_n1048576 | 76.441 ms `[73.653 ms, 79.374 ms]` |
| update | rank2_nonunit_source_serial | large_n1048576 | 19.931 ms `[18.959 ms, 20.692 ms]` |
| update | rank2_negative_source_serial | large_n1048576 | 19.964 ms `[19.562 ms, 20.685 ms]` |
| update | compact_rank2_max_threads_4 | large_n1048576 | 2.2373 ms `[2.2346 ms, 2.2407 ms]` |
| update | compact_rank4_max_threads_4 | large_n1048576 | 3.8501 ms `[3.6248 ms, 3.9586 ms]` |
| update | compact_rank8_max_threads_4 | large_n1048576 | 6.8116 ms `[6.7697 ms, 6.8740 ms]` |
| update | rank2_nonunit_source_max_threads_4 | large_n1048576 | 2.2455 ms `[2.2432 ms, 2.2482 ms]` |
| update | rank2_negative_source_max_threads_4 | large_n1048576 | 2.2338 ms `[2.2289 ms, 2.2376 ms]` |
| slice control | serial | small_n4096 | 665.97 ns `[646.40 ns, 686.91 ns]` |
| slice control | max_threads_4 | small_n4096 | 668.10 ns `[650.39 ns, 689.76 ns]` |
| slice control | serial | near_threshold_n32768 | 6.0472 µs `[5.8797 µs, 6.2108 µs]` |
| slice control | max_threads_4 | near_threshold_n32768 | 5.9493 µs `[5.8067 µs, 6.0958 µs]` |
| slice control | serial | medium_n262144 | 52.501 µs `[51.846 µs, 53.200 µs]` |
| slice control | max_threads_4 | medium_n262144 | 54.128 µs `[52.799 µs, 56.486 µs]` |
| slice control | serial | large_n1048576 | 222.25 µs `[213.54 µs, 231.98 µs]` |
| slice control | max_threads_4 | large_n1048576 | 212.66 µs `[208.14 µs, 217.28 µs]` |
| update control | serial | small_n4096 | 664.02 ns `[628.45 ns, 688.89 ns]` |
| update control | max_threads_4 | small_n4096 | 670.82 ns `[650.20 ns, 684.88 ns]` |
| update control | serial | near_threshold_n32768 | 5.7779 µs `[5.5974 µs, 5.9982 µs]` |
| update control | max_threads_4 | near_threshold_n32768 | 5.7419 µs `[5.6081 µs, 5.8647 µs]` |
| update control | serial | medium_n262144 | 51.635 µs `[50.750 µs, 52.815 µs]` |
| update control | max_threads_4 | medium_n262144 | 53.939 µs `[52.566 µs, 55.170 µs]` |
| update control | serial | large_n1048576 | 264.35 µs `[255.33 µs, 274.31 µs]` |
| update control | max_threads_4 | large_n1048576 | 268.19 µs `[265.20 µs, 273.61 µs]` |

The need-before-implementation gate is **PASS for both families**. At medium size, serial slice rank 4/8 measured 9.2671/18.602 ms versus the rank-one 0.052501 ms control, and serial update rank 4/8 measured 9.4503/19.210 ms versus the rank-one 0.051635 ms control. Rank-8/rank-2 cost ratios were 4.01 (slice) and 3.69 (update), both above the 25% scaling signal. Production implementation proceeded for both families without changing any case or gate.

## Candidate attempt 1 and design delta

The first production implementation used incremental `WindowReplay` state. A
complete named-baseline paired rerun was performed because an exploratory
tuning run had advanced Criterion's default history. Exact baseline commit
`de42f23` was saved as `issue238-final-base`; candidate commit `56b2ad0` ran all
96 declared cases against that baseline. Both four-second load gates passed.

All primary speedup gates passed except one hard gate. At medium size:

- slice compact rank 2/4/8 serial: 4.6250/9.1742/18.650 ms baseline and
  0.58643/0.68698/0.89878 ms candidate;
- update compact rank 2/4/8 serial: 4.8585/9.8359/19.677 ms baseline and
  0.63894/0.72891/0.88775 ms candidate.

The candidate slice rank-8/rank-2 ratio was 1.5327, above the predeclared 1.5
limit. Candidate attempt 1 is therefore **FAIL** and cannot be promoted, despite
all other generic cases improving significantly. The update ratio was 1.389.
No threshold, case, or exclusion is changed.

The failure isolates amortized carry propagation across the eight logical axes.
A private `WindowReplay::compile` refinement will compress adjacent logical
axes only when both source and destination layouts prove the same contiguous
fusion boundary:

```text
next_source_stride == current_source_step * fused_extent
next_dest_stride   == current_dest_step   * fused_extent
```

All products are checked. Fusion scans axes left-to-right in the same
fastest-axis-first order used by `decode`, accumulating the checked fused extent
for each subsequent boundary test. A fused axis retains the first
source/destination step, checked combined extent, and recomputed checked resets.
Negative or
otherwise noncontiguous boundaries remain separate (the negative rank-2 case
therefore exercises the unfused path). This changes only private replay
metadata/state dimensionality; starts, clamping, logical order, offsets,
copy-before-update, uninitialized lifecycle, and public APIs are unchanged. It
is not a new fast path: every generic case still executes the same replay loop,
with compile-time-equivalent adjacent axes represented once.

`reviewer-flash` reviewed exact design-delta commit `a85deb3` and returned
**Correct-to-merge** before implementation. The carry-propagation explanation
remains a hypothesis until the unchanged paired rerun drives the ratio to 1.5
or below; if it does not, the candidate fails again and no fallback gate is
substituted. After implementation, the entire named-baseline baseline/candidate
suite will be rerun under the unchanged host protocol; no selective candidate
row will be promoted. Full attempt-1 and final paired tables will be retained
in this worklog before PR creation.

## Candidate attempt 2 and design delta

After the approved fusion, candidate commit `c6a8256` completed another full
96-case named-baseline pair. All generic gates passed: at medium size compact
slice rank 2/4/8 was 0.47291/0.45819/0.53338 ms and update was
0.51859/0.50821/0.55401 ms, giving rank ratios 1.128 and 1.068. Every generic
case improved with `p < 0.05`.

The attempt is nevertheless **FAIL** because unchanged rank-one controls crossed
the 10% non-regression gate: update serial 4,096 rose from 0.64003 to 0.73246 us
(+14.4%), and update `max_threads(4)` at 32,768 rose from 5.6335 to 6.3354 us
(+12.5%). These fast paths branch before generic replay, so the likely source is
the 256-byte inline `WindowReplay` metadata added to every plan, including
rank-one plans, which changes plan size/cache behavior. The gate is not relaxed
and the rows are not excluded.

A second private design delta removes stored replay metadata from
`DynamicSlicePlan` and `DynamicUpdateSlicePlan`. Compile will invoke
`WindowReplay::prepare` once and discard the result to validate all spans,
fusion products, steps, and resets. Generic execution, only after the unchanged
rank-one fast-path branch, will prepare the same rank-bounded replay state once
per operation; `AxisVec` remains inline through rank 8, preserving the documented
no-allocation contract. Parallel execution prepares once before partitioning and
shares immutable metadata; each worker still performs only one checked range
start decode. Exact `check_call` layout matching guarantees execution prepares
the same validated strides and shapes. The per-element loop remains unchanged.

This restores rank-one plan layout and avoids any replay preparation on rank-one
execution while retaining generic axis fusion. Under the benchmarked `parallel`
feature, temporary rank-bounded metadata uses the existing inline `SmallVec`;
under the default non-parallel feature, the plan retains compile-time `Vec`
metadata so execution preserves its existing no-allocation-through-rank-8
contract. It changes O(rank) operation setup only, not the tensor-sized loop,
and introduces no public API. Implementation is blocked until `reviewer-flash` approves this
design delta. A fresh complete named-baseline pair under the unchanged protocol
will determine the final result; failure of any control or generic gate blocks
promotion again.

`reviewer-flash` reviewed exact design-delta commit `490a7a3` and returned
**Correct-to-merge**. It also confirmed that replay metadata preparation is
compile-guaranteed for exact checked layouts. The pre-existing dynamic-update
ordering can still copy the operand before a pathological runtime-base offset
error; this task neither introduces nor changes that contract; it requires a
pathological isize-magnitude stride/base combination and is not part of the
replay-performance change. The implementation follows the feature-aware
storage disposition above and keeps all rank-one execution branches free of
replay preparation.
