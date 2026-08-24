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

## Final paired experiment

Final named baseline `issue238-final2-base` used exact benchmark-only commit `de42f23`; candidate attempt 3 commit `1607a92` retained fused generic replay while restoring rank-one plan layout. The baseline ran on CPUs 26-29 after a valid gate (selected 0.0-0.5%, L3 sibling maximum 6.0%). The candidate ran on CPUs 0,2,3,4 after a valid gate (selected 0.0-0.7%, sibling maximum 4.5%). Both complete 96-case suites ran sequentially with the frozen protocol.

| phase | family | variant/context | size | estimate `[low, high]` |
|---|---|---|---|---:|
| final baseline | slice control | serial | small_n4096 | 733.59 ns `[712.67 ns, 748.66 ns]` |
| final baseline | slice control | max_threads_4 | small_n4096 | 711.19 ns `[696.96 ns, 729.79 ns]` |
| final baseline | slice control | serial | near_threshold_n32768 | 5.9528 µs `[5.8362 µs, 6.0975 µs]` |
| final baseline | slice control | max_threads_4 | near_threshold_n32768 | 6.2653 µs `[6.0710 µs, 6.4605 µs]` |
| final baseline | slice control | serial | medium_n262144 | 67.774 µs `[66.210 µs, 70.566 µs]` |
| final baseline | slice control | max_threads_4 | medium_n262144 | 52.642 µs `[50.419 µs, 54.364 µs]` |
| final baseline | slice control | serial | large_n1048576 | 223.63 µs `[217.12 µs, 230.41 µs]` |
| final baseline | slice control | max_threads_4 | large_n1048576 | 288.03 µs `[281.69 µs, 295.36 µs]` |
| final baseline | slice | compact_rank2_serial | small_n4096 | 76.902 µs `[72.880 µs, 79.031 µs]` |
| final baseline | slice | compact_rank4_serial | small_n4096 | 141.27 µs `[139.44 µs, 144.73 µs]` |
| final baseline | slice | compact_rank8_serial | small_n4096 | 285.10 µs `[277.74 µs, 292.47 µs]` |
| final baseline | slice | rank2_nonunit_source_serial | small_n4096 | 72.224 µs `[71.364 µs, 73.982 µs]` |
| final baseline | slice | rank2_negative_source_serial | small_n4096 | 73.659 µs `[72.280 µs, 74.887 µs]` |
| final baseline | slice | compact_rank2_max_threads_4 | small_n4096 | 74.021 µs `[71.148 µs, 79.173 µs]` |
| final baseline | slice | compact_rank4_max_threads_4 | small_n4096 | 148.77 µs `[141.33 µs, 154.02 µs]` |
| final baseline | slice | compact_rank8_max_threads_4 | small_n4096 | 281.01 µs `[278.89 µs, 285.28 µs]` |
| final baseline | slice | rank2_nonunit_source_max_threads_4 | small_n4096 | 75.415 µs `[72.396 µs, 78.292 µs]` |
| final baseline | slice | rank2_negative_source_max_threads_4 | small_n4096 | 71.241 µs `[70.136 µs, 73.398 µs]` |
| final baseline | slice | compact_rank2_serial | near_threshold_n32768 | 585.29 µs `[572.42 µs, 593.71 µs]` |
| final baseline | slice | compact_rank4_serial | near_threshold_n32768 | 1.1331 ms `[1.1195 ms, 1.1619 ms]` |
| final baseline | slice | compact_rank8_serial | near_threshold_n32768 | 2.3954 ms `[2.3048 ms, 2.4638 ms]` |
| final baseline | slice | rank2_nonunit_source_serial | near_threshold_n32768 | 586.00 µs `[570.50 µs, 612.43 µs]` |
| final baseline | slice | rank2_negative_source_serial | near_threshold_n32768 | 592.84 µs `[570.26 µs, 618.96 µs]` |
| final baseline | slice | compact_rank2_max_threads_4 | near_threshold_n32768 | 613.98 µs `[581.33 µs, 635.98 µs]` |
| final baseline | slice | compact_rank4_max_threads_4 | near_threshold_n32768 | 1.1478 ms `[1.1207 ms, 1.1797 ms]` |
| final baseline | slice | compact_rank8_max_threads_4 | near_threshold_n32768 | 2.3150 ms `[2.2274 ms, 2.4212 ms]` |
| final baseline | slice | rank2_nonunit_source_max_threads_4 | near_threshold_n32768 | 598.84 µs `[571.03 µs, 619.92 µs]` |
| final baseline | slice | rank2_negative_source_max_threads_4 | near_threshold_n32768 | 606.22 µs `[581.63 µs, 628.41 µs]` |
| final baseline | slice | compact_rank2_serial | medium_n262144 | 4.7049 ms `[4.6080 ms, 4.8367 ms]` |
| final baseline | slice | compact_rank4_serial | medium_n262144 | 9.2190 ms `[9.0355 ms, 9.4167 ms]` |
| final baseline | slice | compact_rank8_serial | medium_n262144 | 17.956 ms `[17.666 ms, 18.427 ms]` |
| final baseline | slice | rank2_nonunit_source_serial | medium_n262144 | 4.8242 ms `[4.6196 ms, 4.9663 ms]` |
| final baseline | slice | rank2_negative_source_serial | medium_n262144 | 4.6537 ms `[4.5617 ms, 4.8349 ms]` |
| final baseline | slice | compact_rank2_max_threads_4 | medium_n262144 | 514.69 µs `[514.53 µs, 514.79 µs]` |
| final baseline | slice | compact_rank4_max_threads_4 | medium_n262144 | 839.75 µs `[839.31 µs, 840.20 µs]` |
| final baseline | slice | compact_rank8_max_threads_4 | medium_n262144 | 1.6692 ms `[1.6686 ms, 1.6702 ms]` |
| final baseline | slice | rank2_nonunit_source_max_threads_4 | medium_n262144 | 515.38 µs `[515.19 µs, 515.58 µs]` |
| final baseline | slice | rank2_negative_source_max_threads_4 | medium_n262144 | 510.60 µs `[506.93 µs, 515.02 µs]` |
| final baseline | slice | compact_rank2_serial | large_n1048576 | 18.733 ms `[18.374 ms, 19.145 ms]` |
| final baseline | slice | compact_rank4_serial | large_n1048576 | 36.684 ms `[36.015 ms, 37.380 ms]` |
| final baseline | slice | compact_rank8_serial | large_n1048576 | 72.847 ms `[71.279 ms, 74.436 ms]` |
| final baseline | slice | rank2_nonunit_source_serial | large_n1048576 | 18.907 ms `[18.470 ms, 19.245 ms]` |
| final baseline | slice | rank2_negative_source_serial | large_n1048576 | 19.156 ms `[18.769 ms, 19.694 ms]` |
| final baseline | slice | compact_rank2_max_threads_4 | large_n1048576 | 2.0257 ms `[2.0250 ms, 2.0266 ms]` |
| final baseline | slice | compact_rank4_max_threads_4 | large_n1048576 | 3.3243 ms `[3.3224 ms, 3.3260 ms]` |
| final baseline | slice | compact_rank8_max_threads_4 | large_n1048576 | 6.6797 ms `[6.6764 ms, 6.6829 ms]` |
| final baseline | slice | rank2_nonunit_source_max_threads_4 | large_n1048576 | 2.0289 ms `[2.0274 ms, 2.0305 ms]` |
| final baseline | slice | rank2_negative_source_max_threads_4 | large_n1048576 | 2.0267 ms `[2.0256 ms, 2.0276 ms]` |
| final baseline | update | compact_rank2_serial | small_n4096 | 75.628 µs `[73.078 µs, 78.665 µs]` |
| final baseline | update | compact_rank4_serial | small_n4096 | 155.43 µs `[153.29 µs, 157.84 µs]` |
| final baseline | update | compact_rank8_serial | small_n4096 | 297.41 µs `[290.28 µs, 307.43 µs]` |
| final baseline | update | rank2_nonunit_source_serial | small_n4096 | 76.079 µs `[74.046 µs, 77.371 µs]` |
| final baseline | update | rank2_negative_source_serial | small_n4096 | 77.680 µs `[75.261 µs, 81.295 µs]` |
| final baseline | update | compact_rank2_max_threads_4 | small_n4096 | 75.669 µs `[73.802 µs, 77.342 µs]` |
| final baseline | update | compact_rank4_max_threads_4 | small_n4096 | 153.82 µs `[148.15 µs, 158.10 µs]` |
| final baseline | update | compact_rank8_max_threads_4 | small_n4096 | 309.46 µs `[297.41 µs, 320.50 µs]` |
| final baseline | update | rank2_nonunit_source_max_threads_4 | small_n4096 | 75.668 µs `[74.035 µs, 78.188 µs]` |
| final baseline | update | rank2_negative_source_max_threads_4 | small_n4096 | 76.428 µs `[75.166 µs, 77.657 µs]` |
| final baseline | update | compact_rank2_serial | near_threshold_n32768 | 619.06 µs `[605.71 µs, 639.56 µs]` |
| final baseline | update | compact_rank4_serial | near_threshold_n32768 | 1.2327 ms `[1.1742 ms, 1.2965 ms]` |
| final baseline | update | compact_rank8_serial | near_threshold_n32768 | 2.3359 ms `[2.2752 ms, 2.4101 ms]` |
| final baseline | update | rank2_nonunit_source_serial | near_threshold_n32768 | 616.24 µs `[601.94 µs, 631.09 µs]` |
| final baseline | update | rank2_negative_source_serial | near_threshold_n32768 | 623.06 µs `[596.98 µs, 654.25 µs]` |
| final baseline | update | compact_rank2_max_threads_4 | near_threshold_n32768 | 617.89 µs `[593.71 µs, 632.37 µs]` |
| final baseline | update | compact_rank4_max_threads_4 | near_threshold_n32768 | 1.2137 ms `[1.1713 ms, 1.2635 ms]` |
| final baseline | update | compact_rank8_max_threads_4 | near_threshold_n32768 | 2.4082 ms `[2.3592 ms, 2.4894 ms]` |
| final baseline | update | rank2_nonunit_source_max_threads_4 | near_threshold_n32768 | 628.79 µs `[614.11 µs, 645.57 µs]` |
| final baseline | update | rank2_negative_source_max_threads_4 | near_threshold_n32768 | 612.54 µs `[593.38 µs, 641.81 µs]` |
| final baseline | update | compact_rank2_serial | medium_n262144 | 4.8560 ms `[4.7583 ms, 4.9727 ms]` |
| final baseline | update | compact_rank4_serial | medium_n262144 | 9.6588 ms `[9.3648 ms, 10.065 ms]` |
| final baseline | update | compact_rank8_serial | medium_n262144 | 19.572 ms `[18.543 ms, 20.255 ms]` |
| final baseline | update | rank2_nonunit_source_serial | medium_n262144 | 4.7965 ms `[4.6748 ms, 4.8979 ms]` |
| final baseline | update | rank2_negative_source_serial | medium_n262144 | 5.0220 ms `[4.8022 ms, 5.2230 ms]` |
| final baseline | update | compact_rank2_max_threads_4 | medium_n262144 | 583.64 µs `[583.05 µs, 584.51 µs]` |
| final baseline | update | compact_rank4_max_threads_4 | medium_n262144 | 899.41 µs `[899.05 µs, 899.72 µs]` |
| final baseline | update | compact_rank8_max_threads_4 | medium_n262144 | 1.6999 ms `[1.6992 ms, 1.7006 ms]` |
| final baseline | update | rank2_nonunit_source_max_threads_4 | medium_n262144 | 586.06 µs `[585.13 µs, 587.19 µs]` |
| final baseline | update | rank2_negative_source_max_threads_4 | medium_n262144 | 587.50 µs `[585.17 µs, 591.14 µs]` |
| final baseline | update | compact_rank2_serial | large_n1048576 | 19.845 ms `[19.316 ms, 20.590 ms]` |
| final baseline | update | compact_rank4_serial | large_n1048576 | 39.245 ms `[38.076 ms, 40.590 ms]` |
| final baseline | update | compact_rank8_serial | large_n1048576 | 75.234 ms `[74.455 ms, 76.036 ms]` |
| final baseline | update | rank2_nonunit_source_serial | large_n1048576 | 19.300 ms `[18.873 ms, 19.950 ms]` |
| final baseline | update | rank2_negative_source_serial | large_n1048576 | 19.184 ms `[18.708 ms, 19.569 ms]` |
| final baseline | update | compact_rank2_max_threads_4 | large_n1048576 | 2.2394 ms `[2.2332 ms, 2.2443 ms]` |
| final baseline | update | compact_rank4_max_threads_4 | large_n1048576 | 3.9130 ms `[3.9081 ms, 3.9185 ms]` |
| final baseline | update | compact_rank8_max_threads_4 | large_n1048576 | 6.7839 ms `[6.7787 ms, 6.7894 ms]` |
| final baseline | update | rank2_nonunit_source_max_threads_4 | large_n1048576 | 2.2462 ms `[2.2414 ms, 2.2492 ms]` |
| final baseline | update | rank2_negative_source_max_threads_4 | large_n1048576 | 2.2377 ms `[2.2352 ms, 2.2407 ms]` |
| final baseline | update control | serial | small_n4096 | 640.03 ns `[613.00 ns, 673.99 ns]` |
| final baseline | update control | max_threads_4 | small_n4096 | 677.79 ns `[659.77 ns, 698.17 ns]` |
| final baseline | update control | serial | near_threshold_n32768 | 5.5456 µs `[5.4565 µs, 5.6180 µs]` |
| final baseline | update control | max_threads_4 | near_threshold_n32768 | 5.6335 µs `[5.5536 µs, 5.7509 µs]` |
| final baseline | update control | serial | medium_n262144 | 69.796 µs `[68.662 µs, 71.859 µs]` |
| final baseline | update control | max_threads_4 | medium_n262144 | 50.147 µs `[48.902 µs, 52.346 µs]` |
| final baseline | update control | serial | large_n1048576 | 265.09 µs `[260.59 µs, 269.30 µs]` |
| final baseline | update control | max_threads_4 | large_n1048576 | 210.62 µs `[202.94 µs, 215.37 µs]` |
| final candidate | slice control | serial | small_n4096 | 732.49 ns `[714.29 ns, 760.36 ns]` |
| final candidate | slice control | max_threads_4 | small_n4096 | 725.20 ns `[695.74 ns, 765.84 ns]` |
| final candidate | slice control | serial | near_threshold_n32768 | 5.8248 µs `[5.6375 µs, 6.0936 µs]` |
| final candidate | slice control | max_threads_4 | near_threshold_n32768 | 5.9549 µs `[5.7708 µs, 6.1787 µs]` |
| final candidate | slice control | serial | medium_n262144 | 51.456 µs `[49.808 µs, 52.892 µs]` |
| final candidate | slice control | max_threads_4 | medium_n262144 | 52.720 µs `[51.185 µs, 54.506 µs]` |
| final candidate | slice control | serial | large_n1048576 | 208.32 µs `[206.31 µs, 211.45 µs]` |
| final candidate | slice control | max_threads_4 | large_n1048576 | 273.32 µs `[264.84 µs, 282.29 µs]` |
| final candidate | slice | compact_rank2_serial | small_n4096 | 9.3692 µs `[9.2519 µs, 9.5878 µs]` |
| final candidate | slice | compact_rank4_serial | small_n4096 | 9.4155 µs `[9.3120 µs, 9.6669 µs]` |
| final candidate | slice | compact_rank8_serial | small_n4096 | 9.4789 µs `[9.3008 µs, 9.8264 µs]` |
| final candidate | slice | rank2_nonunit_source_serial | small_n4096 | 9.3853 µs `[9.2832 µs, 9.4881 µs]` |
| final candidate | slice | rank2_negative_source_serial | small_n4096 | 11.752 µs `[11.502 µs, 11.942 µs]` |
| final candidate | slice | compact_rank2_max_threads_4 | small_n4096 | 10.095 µs `[9.9306 µs, 10.329 µs]` |
| final candidate | slice | compact_rank4_max_threads_4 | small_n4096 | 10.578 µs `[10.319 µs, 10.809 µs]` |
| final candidate | slice | compact_rank8_max_threads_4 | small_n4096 | 9.5056 µs `[9.4039 µs, 9.7252 µs]` |
| final candidate | slice | rank2_nonunit_source_max_threads_4 | small_n4096 | 9.4559 µs `[9.3050 µs, 9.7265 µs]` |
| final candidate | slice | rank2_negative_source_max_threads_4 | small_n4096 | 10.903 µs `[10.525 µs, 11.399 µs]` |
| final candidate | slice | compact_rank2_serial | near_threshold_n32768 | 73.006 µs `[72.576 µs, 73.445 µs]` |
| final candidate | slice | compact_rank4_serial | near_threshold_n32768 | 73.672 µs `[73.273 µs, 74.458 µs]` |
| final candidate | slice | compact_rank8_serial | near_threshold_n32768 | 75.934 µs `[73.929 µs, 79.390 µs]` |
| final candidate | slice | rank2_nonunit_source_serial | near_threshold_n32768 | 76.049 µs `[75.106 µs, 77.752 µs]` |
| final candidate | slice | rank2_negative_source_serial | near_threshold_n32768 | 83.619 µs `[83.161 µs, 84.480 µs]` |
| final candidate | slice | compact_rank2_max_threads_4 | near_threshold_n32768 | 74.712 µs `[74.109 µs, 75.985 µs]` |
| final candidate | slice | compact_rank4_max_threads_4 | near_threshold_n32768 | 74.017 µs `[73.382 µs, 75.155 µs]` |
| final candidate | slice | compact_rank8_max_threads_4 | near_threshold_n32768 | 73.914 µs `[73.130 µs, 75.097 µs]` |
| final candidate | slice | rank2_nonunit_source_max_threads_4 | near_threshold_n32768 | 74.938 µs `[73.910 µs, 76.766 µs]` |
| final candidate | slice | rank2_negative_source_max_threads_4 | near_threshold_n32768 | 84.643 µs `[83.569 µs, 85.956 µs]` |
| final candidate | slice | compact_rank2_serial | medium_n262144 | 589.46 µs `[585.33 µs, 596.91 µs]` |
| final candidate | slice | compact_rank4_serial | medium_n262144 | 638.84 µs `[622.37 µs, 674.13 µs]` |
| final candidate | slice | compact_rank8_serial | medium_n262144 | 637.14 µs `[616.32 µs, 660.29 µs]` |
| final candidate | slice | rank2_nonunit_source_serial | medium_n262144 | 596.98 µs `[585.37 µs, 611.11 µs]` |
| final candidate | slice | rank2_negative_source_serial | medium_n262144 | 681.72 µs `[672.33 µs, 692.77 µs]` |
| final candidate | slice | compact_rank2_max_threads_4 | medium_n262144 | 162.66 µs `[162.44 µs, 162.96 µs]` |
| final candidate | slice | compact_rank4_max_threads_4 | medium_n262144 | 162.67 µs `[162.60 µs, 162.76 µs]` |
| final candidate | slice | compact_rank8_max_threads_4 | medium_n262144 | 187.62 µs `[187.37 µs, 188.02 µs]` |
| final candidate | slice | rank2_nonunit_source_max_threads_4 | medium_n262144 | 162.59 µs `[162.46 µs, 162.81 µs]` |
| final candidate | slice | rank2_negative_source_max_threads_4 | medium_n262144 | 192.06 µs `[191.99 µs, 192.17 µs]` |
| final candidate | slice | compact_rank2_serial | large_n1048576 | 2.5703 ms `[2.4295 ms, 2.6654 ms]` |
| final candidate | slice | compact_rank4_serial | large_n1048576 | 2.4326 ms `[2.3359 ms, 2.5415 ms]` |
| final candidate | slice | compact_rank8_serial | large_n1048576 | 2.4080 ms `[2.3234 ms, 2.4997 ms]` |
| final candidate | slice | rank2_nonunit_source_serial | large_n1048576 | 2.4051 ms `[2.3740 ms, 2.4579 ms]` |
| final candidate | slice | rank2_negative_source_serial | large_n1048576 | 2.9001 ms `[2.7506 ms, 3.0631 ms]` |
| final candidate | slice | compact_rank2_max_threads_4 | large_n1048576 | 607.64 µs `[599.64 µs, 614.23 µs]` |
| final candidate | slice | compact_rank4_max_threads_4 | large_n1048576 | 615.40 µs `[614.79 µs, 616.13 µs]` |
| final candidate | slice | compact_rank8_max_threads_4 | large_n1048576 | 616.56 µs `[615.56 µs, 617.47 µs]` |
| final candidate | slice | rank2_nonunit_source_max_threads_4 | large_n1048576 | 618.65 µs `[617.87 µs, 619.84 µs]` |
| final candidate | slice | rank2_negative_source_max_threads_4 | large_n1048576 | 736.23 µs `[735.25 µs, 737.44 µs]` |
| final candidate | update | compact_rank2_serial | small_n4096 | 8.4521 µs `[8.3334 µs, 8.6156 µs]` |
| final candidate | update | compact_rank4_serial | small_n4096 | 8.8817 µs `[8.6106 µs, 9.1506 µs]` |
| final candidate | update | compact_rank8_serial | small_n4096 | 10.072 µs `[9.9027 µs, 10.432 µs]` |
| final candidate | update | rank2_nonunit_source_serial | small_n4096 | 8.4992 µs `[8.3677 µs, 8.7487 µs]` |
| final candidate | update | rank2_negative_source_serial | small_n4096 | 10.599 µs `[10.300 µs, 11.157 µs]` |
| final candidate | update | compact_rank2_max_threads_4 | small_n4096 | 8.4859 µs `[8.3643 µs, 8.7123 µs]` |
| final candidate | update | compact_rank4_max_threads_4 | small_n4096 | 8.7077 µs `[8.6072 µs, 8.8654 µs]` |
| final candidate | update | compact_rank8_max_threads_4 | small_n4096 | 10.226 µs `[9.7698 µs, 10.875 µs]` |
| final candidate | update | rank2_nonunit_source_max_threads_4 | small_n4096 | 8.4634 µs `[8.3396 µs, 8.6862 µs]` |
| final candidate | update | rank2_negative_source_max_threads_4 | small_n4096 | 10.330 µs `[10.168 µs, 10.556 µs]` |
| final candidate | update | compact_rank2_serial | near_threshold_n32768 | 67.150 µs `[66.337 µs, 68.539 µs]` |
| final candidate | update | compact_rank4_serial | near_threshold_n32768 | 69.902 µs `[68.497 µs, 71.681 µs]` |
| final candidate | update | compact_rank8_serial | near_threshold_n32768 | 71.454 µs `[69.853 µs, 73.833 µs]` |
| final candidate | update | rank2_nonunit_source_serial | near_threshold_n32768 | 69.109 µs `[67.417 µs, 71.800 µs]` |
| final candidate | update | rank2_negative_source_serial | near_threshold_n32768 | 81.730 µs `[80.907 µs, 83.068 µs]` |
| final candidate | update | compact_rank2_max_threads_4 | near_threshold_n32768 | 67.289 µs `[66.570 µs, 68.563 µs]` |
| final candidate | update | compact_rank4_max_threads_4 | near_threshold_n32768 | 68.873 µs `[67.023 µs, 71.485 µs]` |
| final candidate | update | compact_rank8_max_threads_4 | near_threshold_n32768 | 70.045 µs `[68.804 µs, 71.746 µs]` |
| final candidate | update | rank2_nonunit_source_max_threads_4 | near_threshold_n32768 | 68.475 µs `[67.042 µs, 69.801 µs]` |
| final candidate | update | rank2_negative_source_max_threads_4 | near_threshold_n32768 | 83.824 µs `[80.187 µs, 87.919 µs]` |
| final candidate | update | compact_rank2_serial | medium_n262144 | 560.81 µs `[549.30 µs, 586.10 µs]` |
| final candidate | update | compact_rank4_serial | medium_n262144 | 561.75 µs `[547.03 µs, 577.28 µs]` |
| final candidate | update | compact_rank8_serial | medium_n262144 | 615.70 µs `[604.84 µs, 628.03 µs]` |
| final candidate | update | rank2_nonunit_source_serial | medium_n262144 | 555.08 µs `[545.76 µs, 569.98 µs]` |
| final candidate | update | rank2_negative_source_serial | medium_n262144 | 669.64 µs `[662.24 µs, 683.88 µs]` |
| final candidate | update | compact_rank2_max_threads_4 | medium_n262144 | 215.78 µs `[212.28 µs, 217.73 µs]` |
| final candidate | update | compact_rank4_max_threads_4 | medium_n262144 | 214.59 µs `[214.18 µs, 214.85 µs]` |
| final candidate | update | compact_rank8_max_threads_4 | medium_n262144 | 245.87 µs `[244.08 µs, 247.37 µs]` |
| final candidate | update | rank2_nonunit_source_max_threads_4 | medium_n262144 | 213.72 µs `[212.50 µs, 216.31 µs]` |
| final candidate | update | rank2_negative_source_max_threads_4 | medium_n262144 | 346.18 µs `[342.91 µs, 351.20 µs]` |
| final candidate | update | compact_rank2_serial | large_n1048576 | 2.3187 ms `[2.2174 ms, 2.4521 ms]` |
| final candidate | update | compact_rank4_serial | large_n1048576 | 2.3198 ms `[2.2198 ms, 2.4500 ms]` |
| final candidate | update | compact_rank8_serial | large_n1048576 | 2.2141 ms `[2.1804 ms, 2.2750 ms]` |
| final candidate | update | rank2_nonunit_source_serial | large_n1048576 | 2.3962 ms `[2.3178 ms, 2.4828 ms]` |
| final candidate | update | rank2_negative_source_serial | large_n1048576 | 2.8480 ms `[2.7534 ms, 2.9444 ms]` |
| final candidate | update | compact_rank2_max_threads_4 | large_n1048576 | 757.17 µs `[749.06 µs, 768.61 µs]` |
| final candidate | update | compact_rank4_max_threads_4 | large_n1048576 | 758.44 µs `[756.73 µs, 759.92 µs]` |
| final candidate | update | compact_rank8_max_threads_4 | large_n1048576 | 781.32 µs `[767.27 µs, 787.90 µs]` |
| final candidate | update | rank2_nonunit_source_max_threads_4 | large_n1048576 | 752.13 µs `[749.82 µs, 756.31 µs]` |
| final candidate | update | rank2_negative_source_max_threads_4 | large_n1048576 | 1.2713 ms `[1.2545 ms, 1.2857 ms]` |
| final candidate | update control | serial | small_n4096 | 603.61 ns `[597.40 ns, 615.22 ns]` |
| final candidate | update control | max_threads_4 | small_n4096 | 615.63 ns `[605.40 ns, 630.65 ns]` |
| final candidate | update control | serial | near_threshold_n32768 | 5.2912 µs `[5.2435 µs, 5.3933 µs]` |
| final candidate | update control | max_threads_4 | near_threshold_n32768 | 5.4793 µs `[5.3086 µs, 5.6328 µs]` |
| final candidate | update control | serial | medium_n262144 | 49.352 µs `[48.704 µs, 50.716 µs]` |
| final candidate | update control | max_threads_4 | medium_n262144 | 50.467 µs `[48.895 µs, 52.056 µs]` |
| final candidate | update control | serial | large_n1048576 | 255.30 µs `[251.35 µs, 260.95 µs]` |
| final candidate | update control | max_threads_4 | large_n1048576 | 197.77 µs `[193.89 µs, 206.32 µs]` |

Candidate attempt 3 passed every performance gate, but the subsequent full workspace run found one allocation-contract failure: default-feature dynamic-slice execution allocated once per call because ephemeral replay coordinates used `Vec`. Attempt 3 is therefore **FAIL** and not promotable. The implementation replaced those coordinates with the existing rank-bounded `CoordScratch`; benchmarked parallel-feature loop semantics stayed unchanged, but a complete candidate rerun was still required for exact-revision evidence.

### Final candidate after allocation-contract fix

Exact final performance candidate `75d4921` ran the complete 96-case suite on CPUs 33-36 after a valid four-second gate (selected 0.2-1.0%, sibling maximum 2.0%) against unchanged named baseline `issue238-final2-base`.

| family | variant/context | size | estimate `[low, high]` |
|---|---|---|---:|
| slice control | serial | small_n4096 | 713.86 ns `[691.41 ns, 745.58 ns]` |
| slice control | max_threads_4 | small_n4096 | 716.07 ns `[698.17 ns, 739.36 ns]` |
| slice control | serial | near_threshold_n32768 | 5.9851 µs `[5.8638 µs, 6.1142 µs]` |
| slice control | max_threads_4 | near_threshold_n32768 | 5.9831 µs `[5.8604 µs, 6.1431 µs]` |
| slice control | serial | medium_n262144 | 52.074 µs `[50.629 µs, 54.212 µs]` |
| slice control | max_threads_4 | medium_n262144 | 52.914 µs `[51.079 µs, 55.310 µs]` |
| slice control | serial | large_n1048576 | 227.25 µs `[215.87 µs, 234.10 µs]` |
| slice control | max_threads_4 | large_n1048576 | 229.83 µs `[217.82 µs, 238.05 µs]` |
| slice | compact_rank2_serial | small_n4096 | 9.4831 µs `[9.4808 µs, 9.4853 µs]` |
| slice | compact_rank4_serial | small_n4096 | 9.0269 µs `[8.8167 µs, 9.2245 µs]` |
| slice | compact_rank8_serial | small_n4096 | 9.5889 µs `[9.5120 µs, 9.6609 µs]` |
| slice | rank2_nonunit_source_serial | small_n4096 | 9.4867 µs `[9.3808 µs, 9.6035 µs]` |
| slice | rank2_negative_source_serial | small_n4096 | 10.004 µs `[9.7261 µs, 10.327 µs]` |
| slice | compact_rank2_max_threads_4 | small_n4096 | 8.8363 µs `[8.3677 µs, 9.1291 µs]` |
| slice | compact_rank4_max_threads_4 | small_n4096 | 8.6870 µs `[8.5396 µs, 8.8905 µs]` |
| slice | compact_rank8_max_threads_4 | small_n4096 | 8.6902 µs `[8.4217 µs, 8.9959 µs]` |
| slice | rank2_nonunit_source_max_threads_4 | small_n4096 | 8.4417 µs `[8.3177 µs, 8.5962 µs]` |
| slice | rank2_negative_source_max_threads_4 | small_n4096 | 10.360 µs `[10.113 µs, 10.625 µs]` |
| slice | compact_rank2_serial | near_threshold_n32768 | 66.454 µs `[65.266 µs, 67.850 µs]` |
| slice | compact_rank4_serial | near_threshold_n32768 | 66.476 µs `[65.530 µs, 67.507 µs]` |
| slice | compact_rank8_serial | near_threshold_n32768 | 75.201 µs `[74.292 µs, 75.823 µs]` |
| slice | rank2_nonunit_source_serial | near_threshold_n32768 | 67.324 µs `[65.940 µs, 68.635 µs]` |
| slice | rank2_negative_source_serial | near_threshold_n32768 | 86.095 µs `[80.025 µs, 89.651 µs]` |
| slice | compact_rank2_max_threads_4 | near_threshold_n32768 | 68.877 µs `[67.495 µs, 70.835 µs]` |
| slice | compact_rank4_max_threads_4 | near_threshold_n32768 | 67.689 µs `[66.921 µs, 68.251 µs]` |
| slice | compact_rank8_max_threads_4 | near_threshold_n32768 | 75.869 µs `[75.864 µs, 75.877 µs]` |
| slice | rank2_nonunit_source_max_threads_4 | near_threshold_n32768 | 75.824 µs `[75.723 µs, 75.919 µs]` |
| slice | rank2_negative_source_max_threads_4 | near_threshold_n32768 | 84.877 µs `[83.298 µs, 86.931 µs]` |
| slice | compact_rank2_serial | medium_n262144 | 538.59 µs `[524.50 µs, 564.17 µs]` |
| slice | compact_rank4_serial | medium_n262144 | 534.19 µs `[524.51 µs, 554.09 µs]` |
| slice | compact_rank8_serial | medium_n262144 | 617.72 µs `[595.40 µs, 644.65 µs]` |
| slice | rank2_nonunit_source_serial | medium_n262144 | 537.66 µs `[524.95 µs, 563.16 µs]` |
| slice | rank2_negative_source_serial | medium_n262144 | 648.11 µs `[638.37 µs, 668.77 µs]` |
| slice | compact_rank2_max_threads_4 | medium_n262144 | 184.31 µs `[184.18 µs, 184.42 µs]` |
| slice | compact_rank4_max_threads_4 | medium_n262144 | 184.22 µs `[184.14 µs, 184.32 µs]` |
| slice | compact_rank8_max_threads_4 | medium_n262144 | 218.70 µs `[218.58 µs, 218.83 µs]` |
| slice | rank2_nonunit_source_max_threads_4 | medium_n262144 | 184.42 µs `[184.33 µs, 184.49 µs]` |
| slice | rank2_negative_source_max_threads_4 | medium_n262144 | 217.80 µs `[217.70 µs, 217.91 µs]` |
| slice | compact_rank2_serial | large_n1048576 | 2.2237 ms `[2.1693 ms, 2.2706 ms]` |
| slice | compact_rank4_serial | large_n1048576 | 2.1609 ms `[2.1104 ms, 2.2264 ms]` |
| slice | compact_rank8_serial | large_n1048576 | 2.1871 ms `[2.1138 ms, 2.2482 ms]` |
| slice | rank2_nonunit_source_serial | large_n1048576 | 2.2043 ms `[2.1585 ms, 2.2950 ms]` |
| slice | rank2_negative_source_serial | large_n1048576 | 2.6053 ms `[2.5457 ms, 2.6564 ms]` |
| slice | compact_rank2_max_threads_4 | large_n1048576 | 702.93 µs `[702.42 µs, 703.33 µs]` |
| slice | compact_rank4_max_threads_4 | large_n1048576 | 702.69 µs `[702.50 µs, 702.88 µs]` |
| slice | compact_rank8_max_threads_4 | large_n1048576 | 702.42 µs `[701.95 µs, 702.95 µs]` |
| slice | rank2_nonunit_source_max_threads_4 | large_n1048576 | 704.01 µs `[703.63 µs, 704.52 µs]` |
| slice | rank2_negative_source_max_threads_4 | large_n1048576 | 837.14 µs `[836.89 µs, 837.59 µs]` |
| update | compact_rank2_serial | small_n4096 | 9.1878 µs `[8.9730 µs, 9.4095 µs]` |
| update | compact_rank4_serial | small_n4096 | 9.3940 µs `[9.0173 µs, 9.9770 µs]` |
| update | compact_rank8_serial | small_n4096 | 10.360 µs `[10.160 µs, 10.506 µs]` |
| update | rank2_nonunit_source_serial | small_n4096 | 9.6236 µs `[8.9343 µs, 10.054 µs]` |
| update | rank2_negative_source_serial | small_n4096 | 12.070 µs `[12.069 µs, 12.070 µs]` |
| update | compact_rank2_max_threads_4 | small_n4096 | 8.9834 µs `[8.8616 µs, 9.1914 µs]` |
| update | compact_rank4_max_threads_4 | small_n4096 | 9.6563 µs `[9.2494 µs, 10.083 µs]` |
| update | compact_rank8_max_threads_4 | small_n4096 | 10.339 µs `[10.130 µs, 10.530 µs]` |
| update | rank2_nonunit_source_max_threads_4 | small_n4096 | 9.2268 µs `[9.0110 µs, 9.7024 µs]` |
| update | rank2_negative_source_max_threads_4 | small_n4096 | 11.195 µs `[10.768 µs, 11.661 µs]` |
| update | compact_rank2_serial | near_threshold_n32768 | 74.991 µs `[72.779 µs, 78.532 µs]` |
| update | compact_rank4_serial | near_threshold_n32768 | 72.558 µs `[69.972 µs, 75.659 µs]` |
| update | compact_rank8_serial | near_threshold_n32768 | 73.659 µs `[72.374 µs, 75.100 µs]` |
| update | rank2_nonunit_source_serial | near_threshold_n32768 | 71.490 µs `[70.764 µs, 72.723 µs]` |
| update | rank2_negative_source_serial | near_threshold_n32768 | 84.467 µs `[82.278 µs, 88.471 µs]` |
| update | compact_rank2_max_threads_4 | near_threshold_n32768 | 77.386 µs `[73.746 µs, 82.452 µs]` |
| update | compact_rank4_max_threads_4 | near_threshold_n32768 | 72.708 µs `[70.630 µs, 76.249 µs]` |
| update | compact_rank8_max_threads_4 | near_threshold_n32768 | 79.739 µs `[78.128 µs, 82.225 µs]` |
| update | rank2_nonunit_source_max_threads_4 | near_threshold_n32768 | 74.520 µs `[71.825 µs, 77.825 µs]` |
| update | rank2_negative_source_max_threads_4 | near_threshold_n32768 | 85.638 µs `[84.419 µs, 86.995 µs]` |
| update | compact_rank2_serial | medium_n262144 | 602.92 µs `[595.04 µs, 607.91 µs]` |
| update | compact_rank4_serial | medium_n262144 | 594.99 µs `[578.92 µs, 621.05 µs]` |
| update | compact_rank8_serial | medium_n262144 | 632.04 µs `[625.87 µs, 640.12 µs]` |
| update | rank2_nonunit_source_serial | medium_n262144 | 598.86 µs `[587.51 µs, 623.18 µs]` |
| update | rank2_negative_source_serial | medium_n262144 | 680.35 µs `[675.41 µs, 690.02 µs]` |
| update | compact_rank2_max_threads_4 | medium_n262144 | 233.57 µs `[233.09 µs, 234.05 µs]` |
| update | compact_rank4_max_threads_4 | medium_n262144 | 235.91 µs `[235.55 µs, 236.29 µs]` |
| update | compact_rank8_max_threads_4 | medium_n262144 | 253.93 µs `[253.30 µs, 254.58 µs]` |
| update | rank2_nonunit_source_max_threads_4 | medium_n262144 | 238.19 µs `[235.65 µs, 240.73 µs]` |
| update | rank2_negative_source_max_threads_4 | medium_n262144 | 264.54 µs `[263.18 µs, 266.90 µs]` |
| update | compact_rank2_serial | large_n1048576 | 2.4827 ms `[2.3825 ms, 2.6220 ms]` |
| update | compact_rank4_serial | large_n1048576 | 2.4185 ms `[2.3558 ms, 2.4846 ms]` |
| update | compact_rank8_serial | large_n1048576 | 2.3481 ms `[2.3306 ms, 2.3833 ms]` |
| update | rank2_nonunit_source_serial | large_n1048576 | 2.3920 ms `[2.3252 ms, 2.4945 ms]` |
| update | rank2_negative_source_serial | large_n1048576 | 3.0498 ms `[2.9066 ms, 3.1502 ms]` |
| update | compact_rank2_max_threads_4 | large_n1048576 | 851.71 µs `[836.41 µs, 860.77 µs]` |
| update | compact_rank4_max_threads_4 | large_n1048576 | 844.35 µs `[841.47 µs, 847.14 µs]` |
| update | compact_rank8_max_threads_4 | large_n1048576 | 858.07 µs `[843.70 µs, 865.24 µs]` |
| update | rank2_nonunit_source_max_threads_4 | large_n1048576 | 862.07 µs `[854.85 µs, 867.12 µs]` |
| update | rank2_negative_source_max_threads_4 | large_n1048576 | 976.59 µs `[970.13 µs, 984.91 µs]` |
| update control | serial | small_n4096 | 617.82 ns `[601.73 ns, 645.59 ns]` |
| update control | max_threads_4 | small_n4096 | 675.11 ns `[650.76 ns, 695.79 ns]` |
| update control | serial | near_threshold_n32768 | 5.2698 µs `[5.1444 µs, 5.4714 µs]` |
| update control | max_threads_4 | near_threshold_n32768 | 5.0990 µs `[5.0520 µs, 5.1696 µs]` |
| update control | serial | medium_n262144 | 63.578 µs `[62.583 µs, 64.865 µs]` |
| update control | max_threads_4 | medium_n262144 | 49.509 µs `[48.686 µs, 50.386 µs]` |
| update control | serial | large_n1048576 | 252.76 µs `[249.13 µs, 257.04 µs]` |
| update control | max_threads_4 | large_n1048576 | 199.36 µs `[194.25 µs, 205.99 µs]` |

Every final performance and validity gate is **PASS**. At medium size, slice rank 4/8 serial speedups are 17.26x/29.07x and four-thread speedups are 4.56x/7.63x; non-unit/negative serial speedups are 8.97x/7.18x. Update rank 4/8 serial speedups are 16.24x/31.13x and four-thread speedups are 4.20x/6.57x; non-unit/negative serial speedups are 8.63x/7.38x. Candidate rank-8/rank-2 ratios are 1.147 (slice) and 1.048 (update), below 1.5. Every generic case improved with `p < 0.05`; the maximum rank-one control point-estimate regression was 1.62%, below 10%.

Attempt 1, attempt 2, and attempt 3 remain recorded as failed evidence; no
case, threshold, or exclusion changed.

## Verification and review

Verification on final performance candidate `75d4921`:

- default focused indexed-write/uninitialized tests: 77 passed
- parallel focused indexed-write/uninitialized/policy tests: 87 passed
- window-fusion unit test: pass
- default and `parallel` `cargo check -p strided-kernel`: pass
- allocation contract: dynamic slice/update remain allocation-free through rank
  8; the first feature-aware metadata version exposed and then fixed a one-allocation
  regression before promotion
- `cargo fmt --all -- --check`: pass
- `cargo test --workspace`: 908 passed, 9 ignored
- `cargo doc --workspace --no-deps`: pass
- deterministic repository-rules review: pass, no findings
- repository-rules review script: 83 passed

Local `cargo llvm-cov --workspace --features parallel` completed. Modified
`gather_plan.rs` reached 91.71% line coverage, above the repository's 80%
threshold. The global checker still reports the same three unmodified files as
below their configured thresholds (`reduce_view.rs`, `static_indexing_plan.rs`,
and `strided-perm/src/hptt/execute.rs`); hosted CI is authoritative for the PR
coverage gate.

Review gates completed so far:

- initial design `9fa1480`: `reviewer-flash` Correct-to-merge;
- fusion delta `a85deb3`: Correct-to-merge;
- feature-aware plan-layout delta `490a7a3`: Correct-to-merge.

The last review noted a pre-existing pathological-offset error-atomicity corner
in dynamic update (the full copy can precede a later start/base overflow). This
change does not alter that ordering or error contract; it is tracked separately
as [#243](https://github.com/tensor4all/strided-rs/issues/243) rather than hidden
in this performance PR. Exact
final-diff `reviewer-flash` review and hosted CI remain pending.
