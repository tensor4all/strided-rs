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
- rank-2 negative low-edge cropping with compensating positive high edge;
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
`out = edge_low + input * (interior + 1)` into a checked valid input interval.
Because the step is positive, the Cartesian product of those intervals is
exactly the writable source subset. Compile precomputes checked source and
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

Design review, benchmark implementation, baseline, candidate, and verification
are pending.
