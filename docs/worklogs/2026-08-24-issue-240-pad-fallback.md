# Issue 240 generic pad fallback replay

## Task

Remove per-element rank-length coordinate and checked-offset reconstruction from
`PadPlan` generic destination-fill and operand-copy fallbacks while preserving
padding/cropping/interior semantics, dense destination fill and contiguous
axis-0 copy fast paths, initialized/uninitialized behavior, and public APIs.
Baseline source: `37ce20b7ef3841c5a5034ae810f08f895d463342`.
Selected reviewer: read-only `reviewer-flash`, high thinking.

## Evidence reviewed

- #213/#240 and the merged P1/reduction worklogs
- repository/shared layout, materialization, threading, unsafe, and benchmark
  rules
- `PadPlan::compile`, dense fill, axis-0 run, generic serial/parallel
  initialized/uninitialized paths in `static_indexing_plan.rs`
- erased pad wrappers, ground-truth, policy, allocation, and uninitialized tests
- existing `erased_pad_fill_and_copy` threshold benchmark

## Benchmark-first experiment

Add `erased_pad_generic_rank_layout`; keep the existing dense rank-one group as
control. Generic variants target approximately N destination values and force
the fallback:

- compact rank 2, 4, and 8 with one interior element on axis 0;
- rank-2 negative low-edge cropping on axis 1 with compensating positive high
  edge, while retaining axis-0 interior padding 1 so the contiguous axis-0 run
  is provably unavailable;
- rank-2 non-unit source/destination layouts.

For compact rank `r`, the first `r-1` operand axes are extent 2, axis 0 uses
interior padding 1 (destination extent 3), and the final extent is chosen from
the requested profile size so destination work remains comparable across ranks.
All other interior/edge values are zero. Negative cropping and non-unit layouts
keep the same rank-2 logical destination size as their compact peer. Physical
buffers use validated reachable spans. Setup, allocation, compile, descriptor
construction, and fill scalar preparation stay outside timing; execute and
black-box output are timed.

Use threshold sizes `2^12`, `2^15`, `2^18`, `2^20`, serial and
`max_threads(4)`, release mode, existing 300 ms/10 sample/1 second Criterion
settings, `RAYON_NUM_THREADS=4`, and benchmark thread override 4. Runs are
sequential on AMD EPYC 7713P. Before every complete baseline/candidate run,
select four cores in one L3/CCD, require selected cores below 2% busy for four
seconds and all siblings below 20%, and pin with `taskset`; otherwise classify
the run INCONCLUSIVE.

Need gate at medium size:

- compact rank 4 or rank 8 serial must be at least 2x slower per destination
  value than compact rank 2, or exceed 1.0 ms absolute;
- and rank scaling or one crop/non-unit case must add at least 25% per-value
  cost over compact rank 2.

If unmet, retain evidence and do not change production.

Candidate gates, Criterion point estimates:

- compact rank 4/rank 8 serial at least 2x faster, four-thread at least 1.5x;
- crop/non-unit serial at least 1.5x faster;
- candidate rank-8/rank-2 per-destination ratio no more than 1.5.

Validity/non-regression gates:

- no selected generic or existing rank-one control regresses over 10%; primary
  improvements have `p < 0.05`;
- exact initialized/uninitialized output including fill, cropped elements,
  interior gaps, unreachable holes, nonzero offsets, non-unit/negative strides;
- rank0, zero source/destination, fully cropped source, below/exact/above
  threshold serial/parallel equality;
- broad dtype behavior, allocation contract, formatting, focused/default/
  parallel/workspace tests, modified-file coverage, docs, rules review,
  exact-diff review, and hosted CI pass.

Cases and gates freeze before baseline; no selective rerun/exclusion.

## Implementation design

Keep existing dense-fill and contiguous-axis0-run branches unchanged. Add one
private generic replay description to `PadPlan`:

1. a destination-fill cursor over the full destination shape;
2. a copy cursor over the rectangular subset of operand coordinates whose
   affine padded positions are in bounds.

Compile converts each axis's affine mapping
`out = edge_low + input * (interior + 1)` into a checked half-open valid input
interval. With positive `step`, compute the lower coordinate using checked i128
ceil-division of `-edge_low / step`, the upper coordinate using checked i128
floor-division of `(dest_dim - 1 - edge_low) / step`, then clamp both to
`[0, input_extent]`. Because the step is positive, the Cartesian product of
those intervals is exactly the writable source subset. The empty axis set
(rank 0) has copy total 1; any actual empty interval makes copy total 0. Compile precomputes checked source and
destination base deltas, copy shape, steps, and wrap resets. Fully cropped axes
produce an empty copy domain. This removes per-element rank scans and avoids
forming out-of-range destination offsets entirely.

Serial fill/copy decode once. Parallel fill/copy decode once per worker range.
Each cursor advances column-major offsets incrementally. Initialized and
uninitialized paths share the same private metadata and visit order; fill still
initializes every reachable destination before any copy. Existing contiguous
fast paths remain controls and do not build runtime cursor state.

Unchecked replay arithmetic requires the three-link proof: compile-time checked
spans/base deltas/steps/resets; validated raw descriptors and storage extents;
exact plan-layout equality before dispatch. Keep rank-bounded scratch
allocation-free through rank 8. Do not extract a cross-plan public/general
cursor or modify C/API semantics.

## Review and verification

No benchmark implementation starts before a Correct-to-merge design verdict.
After a valid need gate, implement and test the generic replay, run the paired
candidate suite and complete repository gates, and obtain an exact-final
`reviewer-flash` verdict before PR creation.

## Gate status

`reviewer-flash` reviewed exact design commit `b30145f` with high thinking and
a read-only boundary. Verdict: **Correct-to-merge** conditional on the benchmark
fallback and interval clarifications now incorporated above. The benchmark
negative-crop recipe retains axis-0 interior padding, and a private unit test
must assert that all benchmark recipes leave `contiguous_axis0_run` unset.
Generic non-dense fill performance is intentionally not claimed by the timing
matrix; it remains a correctness/validity target.

## Baseline evidence

Benchmark-only commit `df03ec1` ran the complete baseline on CPUs 1-4 in L3 domain 0-7 after a valid gate (selected 0.0-1.2%, sibling maximum 6.8%).

| family | variant/context | size | estimate `[low, high]` |
|---|---|---|---:|
| generic | compact_rank2_serial | n4096 | 51.580 µs `[49.332 µs, 54.127 µs]` |
| generic | compact_rank4_serial | n4096 | 105.57 µs `[103.52 µs, 109.91 µs]` |
| generic | compact_rank8_serial | n4096 | 226.90 µs `[222.69 µs, 231.24 µs]` |
| generic | rank2_negative_crop_serial | n4096 | 52.621 µs `[50.483 µs, 54.045 µs]` |
| generic | rank2_nonunit_serial | n4096 | 76.374 µs `[73.935 µs, 78.653 µs]` |
| generic | compact_rank2_max_threads_4 | n4096 | 49.314 µs `[47.375 µs, 52.041 µs]` |
| generic | compact_rank4_max_threads_4 | n4096 | 104.04 µs `[99.783 µs, 106.77 µs]` |
| generic | compact_rank8_max_threads_4 | n4096 | 221.15 µs `[215.30 µs, 227.31 µs]` |
| generic | rank2_negative_crop_max_threads_4 | n4096 | 50.588 µs `[49.204 µs, 52.910 µs]` |
| generic | rank2_nonunit_max_threads_4 | n4096 | 73.025 µs `[71.798 µs, 76.086 µs]` |
| generic | compact_rank2_serial | n32768 | 419.31 µs `[403.54 µs, 433.71 µs]` |
| generic | compact_rank4_serial | n32768 | 836.96 µs `[796.23 µs, 881.97 µs]` |
| generic | compact_rank8_serial | n32768 | 1.7168 ms `[1.6474 ms, 1.7880 ms]` |
| generic | rank2_negative_crop_serial | n32768 | 414.43 µs `[390.74 µs, 433.43 µs]` |
| generic | rank2_nonunit_serial | n32768 | 612.04 µs `[586.24 µs, 637.66 µs]` |
| generic | compact_rank2_max_threads_4 | n32768 | 392.18 µs `[383.99 µs, 410.85 µs]` |
| generic | compact_rank4_max_threads_4 | n32768 | 820.97 µs `[786.29 µs, 867.73 µs]` |
| generic | compact_rank8_max_threads_4 | n32768 | 1.7366 ms `[1.6872 ms, 1.8005 ms]` |
| generic | rank2_negative_crop_max_threads_4 | n32768 | 386.69 µs `[378.87 µs, 405.08 µs]` |
| generic | rank2_nonunit_max_threads_4 | n32768 | 608.50 µs `[581.79 µs, 632.84 µs]` |
| generic | compact_rank2_serial | n262144 | 3.1539 ms `[3.0601 ms, 3.2661 ms]` |
| generic | compact_rank4_serial | n262144 | 6.6078 ms `[6.3942 ms, 6.8712 ms]` |
| generic | compact_rank8_serial | n262144 | 13.561 ms `[13.229 ms, 14.087 ms]` |
| generic | rank2_negative_crop_serial | n262144 | 3.3040 ms `[3.2245 ms, 3.4226 ms]` |
| generic | rank2_nonunit_serial | n262144 | 4.9075 ms `[4.6598 ms, 5.1956 ms]` |
| generic | compact_rank2_max_threads_4 | n262144 | 546.61 µs `[545.90 µs, 547.29 µs]` |
| generic | compact_rank4_max_threads_4 | n262144 | 964.19 µs `[963.29 µs, 965.38 µs]` |
| generic | compact_rank8_max_threads_4 | n262144 | 1.8034 ms `[1.7988 ms, 1.8079 ms]` |
| generic | rank2_negative_crop_max_threads_4 | n262144 | 546.13 µs `[545.72 µs, 546.69 µs]` |
| generic | rank2_nonunit_max_threads_4 | n262144 | 970.48 µs `[969.25 µs, 972.14 µs]` |
| generic | compact_rank2_serial | n1048576 | 12.732 ms `[12.216 ms, 13.268 ms]` |
| generic | compact_rank4_serial | n1048576 | 26.639 ms `[26.254 ms, 27.144 ms]` |
| generic | compact_rank8_serial | n1048576 | 54.485 ms `[53.077 ms, 56.123 ms]` |
| generic | rank2_negative_crop_serial | n1048576 | 12.725 ms `[12.365 ms, 13.219 ms]` |
| generic | rank2_nonunit_serial | n1048576 | 19.389 ms `[18.751 ms, 20.330 ms]` |
| generic | compact_rank2_max_threads_4 | n1048576 | 2.0569 ms `[2.0555 ms, 2.0603 ms]` |
| generic | compact_rank4_max_threads_4 | n1048576 | 3.7116 ms `[3.7076 ms, 3.7177 ms]` |
| generic | compact_rank8_max_threads_4 | n1048576 | 7.0485 ms `[7.0395 ms, 7.0647 ms]` |
| generic | rank2_negative_crop_max_threads_4 | n1048576 | 2.0578 ms `[2.0553 ms, 2.0603 ms]` |
| generic | rank2_nonunit_max_threads_4 | n1048576 | 3.8132 ms `[3.8105 ms, 3.8187 ms]` |
| rank1 control | serial | small_n4096 | 573.82 ns `[550.88 ns, 599.21 ns]` |
| rank1 control | max_threads_4 | small_n4096 | 463.15 ns `[446.65 ns, 500.34 ns]` |
| rank1 control | serial | near_threshold_n32768 | 23.523 µs `[22.864 µs, 24.578 µs]` |
| rank1 control | max_threads_4 | near_threshold_n32768 | 23.662 µs `[22.855 µs, 24.720 µs]` |
| rank1 control | serial | medium_n262144 | 32.515 µs `[31.566 µs, 34.086 µs]` |
| rank1 control | max_threads_4 | medium_n262144 | 31.851 µs `[30.679 µs, 33.657 µs]` |
| rank1 control | serial | large_n1048576 | 132.88 µs `[126.46 µs, 138.21 µs]` |
| rank1 control | max_threads_4 | large_n1048576 | 128.73 µs `[124.40 µs, 135.89 µs]` |

The need-before-implementation gate is **PASS**. At medium target size, compact rank 2/4/8 serial measured 3.1539/6.6078/13.561 ms; rank 4/8 exceed 1.0 ms and rank 8 is 4.30x rank 2. The non-unit case is 1.56x rank 2. Cases/gates are frozen; production may proceed.

## Candidate implementation and evidence

Production commit `b141365` adds private prepared fill/copy cursor metadata. Valid per-axis source intervals are compiled with checked `i128` ceil/floor arithmetic; checked base deltas, steps, carry resets, and offset spans are prepared once. Initialized/uninitialized and serial/parallel generic replay decode once per range and advance incrementally. Dense fill and contiguous axis-0 copy remain unchanged.

The frozen candidate ran sequentially on CPUs 1-4 with four Rayon workers after a valid L3-domain gate (selected average 7.5%, other-domain-core maximum 0.7%). All 40 generic cells improved with non-overlapping intervals.

| case | context | size | baseline | candidate | speedup | interval-bound speedup |
|---|---|---:|---:|---:|---:|---:|
| compact_rank2 | serial | 262144 | 3.1539 ms | 1.2335 ms | 2.56x | 2.39-2.73x |
| compact_rank4 | serial | 262144 | 6.6078 ms | 1.3668 ms | 4.83x | 4.61-5.06x |
| compact_rank8 | serial | 262144 | 13.5610 ms | 1.6079 ms | 8.43x | 8.07-8.89x |
| rank2_negative_crop | serial | 262144 | 3.3040 ms | 1.2749 ms | 2.59x | 2.41-2.82x |
| rank2_nonunit | serial | 262144 | 4.9075 ms | 2.3353 ms | 2.10x | 1.95-2.28x |
| compact_rank2 | max_threads_4 | 262144 | 0.5466 ms | 0.4037 ms | 1.35x | 1.34-1.37x |
| compact_rank4 | max_threads_4 | 262144 | 0.9642 ms | 0.4187 ms | 2.30x | 2.22-2.36x |
| compact_rank8 | max_threads_4 | 262144 | 1.8034 ms | 0.5151 ms | 3.50x | 3.47-3.54x |
| rank2_negative_crop | max_threads_4 | 262144 | 0.5461 ms | 0.4010 ms | 1.36x | 1.34-1.39x |
| rank2_nonunit | max_threads_4 | 262144 | 0.9705 ms | 0.6823 ms | 1.42x | 1.41-1.45x |
| compact_rank2 | serial | 1048576 | 12.7320 ms | 5.0020 ms | 2.55x | 2.37-2.74x |
| compact_rank4 | serial | 1048576 | 26.6390 ms | 5.4980 ms | 4.85x | 4.67-5.01x |
| compact_rank8 | serial | 1048576 | 54.4850 ms | 6.4558 ms | 8.44x | 8.03-8.79x |
| rank2_negative_crop | serial | 1048576 | 12.7250 ms | 4.8369 ms | 2.63x | 2.50-2.76x |
| rank2_nonunit | serial | 1048576 | 19.3890 ms | 9.1039 ms | 2.13x | 2.03-2.26x |
| compact_rank2 | max_threads_4 | 1048576 | 2.0569 ms | 1.4944 ms | 1.38x | 1.37-1.40x |
| compact_rank4 | max_threads_4 | 1048576 | 3.7116 ms | 1.6353 ms | 2.27x | 2.23-2.32x |
| compact_rank8 | max_threads_4 | 1048576 | 7.0485 ms | 1.9645 ms | 3.59x | 3.56-3.63x |
| rank2_negative_crop | max_threads_4 | 1048576 | 2.0578 ms | 1.4749 ms | 1.40x | 1.37-1.42x |
| rank2_nonunit | max_threads_4 | 1048576 | 3.8132 ms | 2.6143 ms | 1.46x | 1.43-1.49x |

Across all frozen generic cells, estimate speedups were 1.35-9.10x (medium: 1.35-8.43x; large: 1.38-8.44x). The unchanged rank-one control remained stable at medium/large: serial estimates 1.04x/0.97x and four-thread estimates 1.01x/1.05x, with confidence intervals overlapping no change for the only nominal slowdown.

## Verification

- focused initialized/uninitialized default tests: 85 passed
- focused parallel/static/uninitialized/policy tests: 96 passed
- rank-8 generic initialized/uninitialized allocation test: zero allocations
- source-contract tests: 6 passed
- default workspace: 915 passed, 9 ignored
- parallel workspace: 989 passed, 9 ignored
- `cargo check -p strided-kernel --features parallel`: passed
- deterministic repository-rules review: passed
- workspace coverage: modified `static_indexing_plan.rs` 81.93%, above the repository 80% threshold. The global script still reports two unchanged baseline deficits (`reduce_view.rs` 71.5% and `hptt/execute.rs` 57.0%).

Exact-final independent review of candidate `e1f01cdd` by `reviewer-flash` (high, read-only) returned **Correct-to-merge** with no blocking findings. It confirmed the interval and pointer-safety proof, serial/parallel and initialized/uninitialized semantics, frozen fallback selection, unchanged fast paths, benchmark gates, and that the two global coverage deficits are unchanged and nonblocking. Hosted CI remains pending until the PR is opened.
