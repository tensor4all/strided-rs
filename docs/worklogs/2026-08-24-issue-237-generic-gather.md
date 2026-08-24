# Issue 237 generic gather replay

## Task and current contract

Issue #237 is the P1 generic-gather child of #213. PR #236 specialized compact
rank-one scalar take, but `GatherPlan` still rebuilds window, batch, and operand
coordinates and scans destination/operand/index strides inside each generic
serial or parallel output iteration. The public `GatherSpec`, clamping,
column-major output order, initialized/uninitialized writers, and compact
rank-one fast path are unchanged by this task.

Baseline source: `1679b167e1d7d665dbba9d09be2e5b38c0166d2a`, the
current `umbrella/issue-burndown-2026-08` head after the audit-rule PR.
Selected reviewer: read-only `reviewer-flash`, high thinking, for this design
and the exact final diff. A delegated implementation must not begin before this
design receives a Correct-to-merge verdict.

## Evidence and code read

- issues #213 and #237; strided-rs PRs #236/#241
- `REPOSITORY_RULES.md` layout, fast-path, threading, and benchmark sections
- shared common/Rust performance rules
- `docs/design/erased-execution-policy.md`
- `strided-kernel/src/gather_plan.rs`, especially `GatherPlan::compile`, generic
  serial/parallel replay, `index_component`, and the rank-one fast path
- `strided-kernel/tests/erased_gather_plan.rs`,
  `erased_policy_parallel.rs`, and `issue_187_uninit_indexed.rs`
- `strided-kernel/benches/erased_policy_thresholds.rs`

## Benchmark-first experiment

The first implementation commit adds only benchmark cases; production source
remains at the baseline. The cases all produce the same requested logical
output count and deliberately miss `uses_rank_one_scalar_take_path`:

- compact rank 2, 4, and 8 windowed gathers;
- rank-2 non-unit-stride operand/index layout;
- rank-2 negative-stride operand layout;
- the existing compact rank-one scalar take group remains the fast-path control.

For compact rank `r`, the first `r - 1` operand axes are extent 2 window axes;
the final axis is selected by one scalar index per batch. Output order is the
window axes followed by the batch axis, so every rank writes one compact
column-major stream; this avoids confounding rank scaling with the scattered
multi-stream destination used by the existing batch-first rank-2 correctness
test. Correctness coverage retains both output-axis orders. The physical
operand and requested output stay O(N). Non-unit and negative layouts use
validated storage extents and offsets; setup, allocation, plan compile, and
reference construction remain outside Criterion timing.

Threshold profile sizes are `2^12`, `2^15`, `2^18`, and `2^20`; contexts are
serial and `max_threads(4)`. Criterion uses the existing 300 ms warmup, 10
samples, and one-second measurement window. Runs are release mode, sequential,
with `RAYON_NUM_THREADS=4` and the benchmark thread override set to 4.

Hardware protocol: AMD EPYC 7713P, no SMT. Immediately before each complete
baseline/candidate run, choose four cores in one L3/CCD. Every selected core
must average below 2% busy over four seconds and no sibling in that L3 domain
may exceed 20%; otherwise classify the complete run INCONCLUSIVE. Pin the
process with `taskset`; do not run another timing workload concurrently.

Need-before-implementation gate, evaluated from the complete baseline:

- at `medium_n262144`, compact generic rank 4 or rank 8 serial replay must be at
  least 2x slower than the existing rank-one serial control while producing the
  same number of output values, or exceed 1.0 ms absolute; and
- at least one generic case must show increasing per-output cost with rank or a
  non-unit/negative layout penalty of at least 25%.

If neither condition holds, record the result and stop without production
implementation.

Predeclared candidate primary gates using Criterion point estimates:

- serial rank 4 and rank 8 at `medium_n262144`: at least 3x faster than baseline;
- four-thread rank 4 and rank 8 at `medium_n262144`: at least 2x faster;
- serial non-unit and negative rank-2 cases at `medium_n262144`: at least 2x
  faster;
- candidate rank-8 per-output cost at medium size no more than 1.5x candidate
  rank-2 compact cost; all compact rank cases use the same contiguous write
  pattern, and the ratio plus raw estimates will also be reported rather than
  interpreted without the memory-layout context.

Non-regression and validity gates:

- no selected generic or existing rank-one control point estimate regresses by
  more than 10%;
- all Criterion cases complete, and every primary improvement is statistically
  significant at `p < 0.05`;
- serial/parallel and initialized/uninitialized outputs exactly match ground
  truth for compact, non-unit, negative, nonzero-offset, clamped, zero/empty,
  and multidimensional-window cases;
- default and `parallel` feature tests, workspace formatting/tests, coverage,
  repository-rules review, and hosted CI pass.

No threshold or case may change after baseline results are visible. A failed
host gate invalidates the complete run rather than permitting selective retry.

## Implementation design

`GatherPlan::compile` will prepare only private, rank-bounded replay metadata:

- each output axis's destination-stride contribution;
- each output window axis's operand-stride contribution;
- each output batch axis's start-index-stride contribution;
- each index-vector component's fixed offset and mapped operand stride;
- checked reset/carry deltas needed when a column-major coordinate wraps;
- checked operand/index layout spans sufficient to prove all prepared deltas
  and range-start decodes fit `isize`, including negative strides.

Actual allocation reachability remains proven by validated `RawStridedRef` /
writer construction plus `check_call`'s exact layout match. Serial traversal and
each parallel range perform one checked initial decode; direct hot-loop
arithmetic is permitted only because those proofs cover the whole reachable
range.

Generic serial execution decodes its initial state once. Generic parallel
execution decodes once at each worker range start. Each output then:

1. reads only the data-dependent index-vector components;
2. starts a fresh source offset from the incrementally maintained window
   offset, then clamps and adds each mapped index contribution for this element
   only; index contributions are never accumulated into the next element;
3. writes the value at the incrementally maintained destination offset; and
4. advances output coordinates plus destination/window/index-batch offsets via
   prevalidated carry deltas.

The hot loop performs O(index-vector components) unavoidable data-dependent
work, not O(operand rank + output rank + index rank) static-layout scans.
Validated offsets permit direct hot-loop arithmetic only with nearby concrete
`// INVARIANT:` and `// SAFETY:` proofs. No public API, dependency, feature,
allocation boundary, clamping rule, or fallback contract changes. Keep the
rank-one fast path separate; do not create a shared cursor abstraction for the
later dynamic/reduction tasks until repeated implementations prove a stable
common boundary.

## Correctness and review plan

Add focused integration tests that compare the generic replay against explicit
expected values and, where useful, equivalent layouts:

- rank 2/4/8 windows and scalar/vector index dimensions;
- clamped lower/upper starts and nonzero raw offsets;
- non-unit and negative operand/index/destination strides;
- zero output and empty valid boundaries;
- initialized and `MaybeUninit` writers;
- below/exact/above threading threshold serial/parallel equality.

The implementer runs focused tests first, then default/parallel crate tests.
The parent reviews the full diff, runs the repository PR gates and paired
candidate benchmark, records every case and confidence interval here, and
requests an exact-final-diff `reviewer-flash` verdict before PR creation.

## Gate status

The first broad review attempt timed out before reaching a verdict and did not
clear the gate. A bounded `reviewer-flash` re-review completed with a
conditional **Correct-to-merge** verdict. Its binding findings are incorporated
above:

- index contributions are explicitly fresh/non-accumulating per element;
- operand/index span validation, checked carries, and checked serial/range-start
  decode are required before unchecked incremental arithmetic;
- compact rank cases now use the same contiguous destination pattern, removing
  the prior rank-8/rank-2 scattered-write confound.

The f64/i64 benchmark dtype is retained as the representative performance case;
existing dtype-matrix correctness tests remain required. Because the benchmark
layout and safety proof changed materially while recording these conditions,
implementation remains blocked until `reviewer-flash` re-reviews this exact
design revision. Benchmark implementation, baseline results, candidate results,
and final verification are pending.
