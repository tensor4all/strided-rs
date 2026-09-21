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
existing dtype-matrix correctness tests remain required. `reviewer-flash`
re-reviewed exact design commit `9dfdf05` and returned **Correct-to-merge**;
benchmark-first implementation may proceed. Its three non-blocking parameter
conditions are binding:

- the selected operand axis extent is the batch count (at least 2 in every
  case), and a deterministic permutation spans `[0, extent)` so starts vary;
- non-unit and negative rank-2 cases keep the same compact destination
  dims/strides as compact rank 2, varying only operand/index layout;
- the hot-loop invariant must explicitly state that window offset plus every
  clamped index contribution remains inside the validated operand span.

## Baseline evidence

Benchmark-only commit: `9bd575f`. The complete baseline ran sequentially on
CPUs 50-53 in L3 domain 48-55. The accepted four-second load gate measured
selected cores at 0.0-0.5% busy and every sibling at at most 0.5%, so the run is
valid.

Criterion point estimates and confidence intervals:

| variant | context | size | estimate `[low, high]` |
|---|---|---:|---:|
| compact rank 2 | serial | 4,096 | 155.71 us `[150.03, 162.93]` |
| compact rank 4 | serial | 4,096 | 242.02 us `[233.56, 247.17]` |
| compact rank 8 | serial | 4,096 | 434.78 us `[433.94, 435.34]` |
| non-unit rank 2 | serial | 4,096 | 158.61 us `[154.85, 162.80]` |
| negative rank 2 | serial | 4,096 | 162.74 us `[162.62, 162.89]` |
| compact rank 2 | max_threads(4) | 4,096 | 156.22 us `[149.35, 162.59]` |
| compact rank 4 | max_threads(4) | 4,096 | 239.38 us `[234.82, 244.59]` |
| compact rank 8 | max_threads(4) | 4,096 | 427.05 us `[419.67, 434.57]` |
| non-unit rank 2 | max_threads(4) | 4,096 | 160.13 us `[157.43, 163.65]` |
| negative rank 2 | max_threads(4) | 4,096 | 156.34 us `[152.19, 162.95]` |
| compact rank 2 | serial | 32,768 | 1.2649 ms `[1.2274, 1.3050]` |
| compact rank 4 | serial | 32,768 | 1.9775 ms `[1.9767, 1.9783]` |
| compact rank 8 | serial | 32,768 | 3.4039 ms `[3.3411, 3.4840]` |
| non-unit rank 2 | serial | 32,768 | 1.2902 ms `[1.2563, 1.3052]` |
| negative rank 2 | serial | 32,768 | 1.2647 ms `[1.2425, 1.3029]` |
| compact rank 2 | max_threads(4) | 32,768 | 1.2357 ms `[1.2069, 1.2762]` |
| compact rank 4 | max_threads(4) | 32,768 | 1.9665 ms `[1.9383, 1.9771]` |
| compact rank 8 | max_threads(4) | 32,768 | 3.4581 ms `[3.4132, 3.4809]` |
| non-unit rank 2 | max_threads(4) | 32,768 | 1.3026 ms `[1.2917, 1.3065]` |
| negative rank 2 | max_threads(4) | 32,768 | 1.3045 ms `[1.3009, 1.3073]` |
| compact rank 2 | serial | 262,144 | 10.457 ms `[10.442, 10.469]` |
| compact rank 4 | serial | 262,144 | 15.737 ms `[15.597, 15.821]` |
| compact rank 8 | serial | 262,144 | 27.882 ms `[27.797, 27.949]` |
| non-unit rank 2 | serial | 262,144 | 10.418 ms `[10.357, 10.461]` |
| negative rank 2 | serial | 262,144 | 10.378 ms `[10.372, 10.382]` |
| compact rank 2 | max_threads(4) | 262,144 | 1.2864 ms `[1.2857, 1.2873]` |
| compact rank 4 | max_threads(4) | 262,144 | 1.6719 ms `[1.6706, 1.6731]` |
| compact rank 8 | max_threads(4) | 262,144 | 2.6332 ms `[2.6256, 2.6373]` |
| non-unit rank 2 | max_threads(4) | 262,144 | 1.2867 ms `[1.2845, 1.2893]` |
| negative rank 2 | max_threads(4) | 262,144 | 1.2831 ms `[1.2820, 1.2836]` |
| compact rank 2 | serial | 1,048,576 | 40.147 ms `[38.931, 41.213]` |
| compact rank 4 | serial | 1,048,576 | 62.498 ms `[61.226, 63.484]` |
| compact rank 8 | serial | 1,048,576 | 108.37 ms `[106.12, 110.33]` |
| non-unit rank 2 | serial | 1,048,576 | 41.368 ms `[40.960, 41.602]` |
| negative rank 2 | serial | 1,048,576 | 41.094 ms `[40.413, 41.508]` |
| compact rank 2 | max_threads(4) | 1,048,576 | 5.1047 ms `[5.0864, 5.1131]` |
| compact rank 4 | max_threads(4) | 1,048,576 | 6.6357 ms `[6.6245, 6.6491]` |
| compact rank 8 | max_threads(4) | 1,048,576 | 10.422 ms `[10.413, 10.429]` |
| non-unit rank 2 | max_threads(4) | 1,048,576 | 5.1386 ms `[5.1339, 5.1476]` |
| negative rank 2 | max_threads(4) | 1,048,576 | 5.1339 ms `[5.0843, 5.1579]` |
| rank-one control | serial | 4,096 | 4.5166 us `[4.4982, 4.5240]` |
| rank-one control | max_threads(4) | 4,096 | 3.9043 us `[3.8332, 3.9555]` |
| rank-one control | serial | 32,768 | 36.737 us `[36.735, 36.739]` |
| rank-one control | max_threads(4) | 32,768 | 36.698 us `[36.505, 36.831]` |
| rank-one control | serial | 262,144 | 277.95 us `[267.13, 288.66]` |
| rank-one control | max_threads(4) | 262,144 | 123.51 us `[123.46, 123.57]` |
| rank-one control | serial | 1,048,576 | 1.1629 ms `[1.1495, 1.1808]` |
| rank-one control | max_threads(4) | 1,048,576 | 467.26 us `[466.45, 467.95]` |

The need-before-implementation gate is **PASS**. At medium size, serial generic
rank 4 is 56.6x and rank 8 is 100.3x slower than the rank-one control, both far
above 1.0 ms. Rank-8 per-output cost is 2.67x rank 2, exceeding the predeclared
25% scaling signal. Production implementation proceeded without changing any
case or gate.

## Candidate evidence

Production candidate commit: `d95a387`. The baseline CCD became invalid because
a sustained unrelated Julia process occupied one sibling above 20%. The
predeclared protocol permits choosing another valid same-host L3 domain; the
complete candidate ran on CPUs 9-12 in domain 8-15 after an accepted gate:
selected cores were 0.0-1.5% busy and the busiest sibling was 19.0%. No other
timing workload ran concurrently.

| variant | context | size | estimate `[low, high]` |
|---|---|---:|---:|
| compact rank 2 | serial | 4,096 | 27.867 us `[27.393, 28.401]` |
| compact rank 4 | serial | 4,096 | 30.720 us `[29.548, 31.744]` |
| compact rank 8 | serial | 4,096 | 33.712 us `[32.637, 34.773]` |
| non-unit rank 2 | serial | 4,096 | 28.444 us `[27.873, 29.148]` |
| negative rank 2 | serial | 4,096 | 28.679 us `[27.721, 29.544]` |
| compact rank 2 | max_threads(4) | 4,096 | 27.614 us `[27.255, 28.341]` |
| compact rank 4 | max_threads(4) | 4,096 | 30.243 us `[29.760, 30.666]` |
| compact rank 8 | max_threads(4) | 4,096 | 34.068 us `[32.995, 35.676]` |
| non-unit rank 2 | max_threads(4) | 4,096 | 28.907 us `[27.866, 29.578]` |
| negative rank 2 | max_threads(4) | 4,096 | 28.348 us `[27.701, 29.091]` |
| compact rank 2 | serial | 32,768 | 233.65 us `[226.30, 239.17]` |
| compact rank 4 | serial | 32,768 | 246.07 us `[242.15, 252.75]` |
| compact rank 8 | serial | 32,768 | 294.42 us `[281.94, 306.25]` |
| non-unit rank 2 | serial | 32,768 | 223.63 us `[219.22, 227.23]` |
| negative rank 2 | serial | 32,768 | 225.41 us `[222.00, 228.96]` |
| compact rank 2 | max_threads(4) | 32,768 | 225.69 us `[221.68, 230.61]` |
| compact rank 4 | max_threads(4) | 32,768 | 245.96 us `[242.03, 249.51]` |
| compact rank 8 | max_threads(4) | 32,768 | 286.91 us `[276.73, 299.29]` |
| non-unit rank 2 | max_threads(4) | 32,768 | 224.20 us `[220.08, 230.53]` |
| negative rank 2 | max_threads(4) | 32,768 | 227.28 us `[219.68, 231.77]` |
| compact rank 2 | serial | 262,144 | 1.8166 ms `[1.7647, 1.8604]` |
| compact rank 4 | serial | 262,144 | 2.0825 ms `[2.0431, 2.1322]` |
| compact rank 8 | serial | 262,144 | 2.4558 ms `[2.3820, 2.5229]` |
| non-unit rank 2 | serial | 262,144 | 1.8255 ms `[1.8030, 1.8678]` |
| negative rank 2 | serial | 262,144 | 1.8131 ms `[1.7850, 1.8544]` |
| compact rank 2 | max_threads(4) | 262,144 | 547.97 us `[547.62, 548.28]` |
| compact rank 4 | max_threads(4) | 262,144 | 630.49 us `[630.23, 630.81]` |
| compact rank 8 | max_threads(4) | 262,144 | 696.52 us `[695.70, 697.57]` |
| non-unit rank 2 | max_threads(4) | 262,144 | 549.91 us `[549.55, 550.16]` |
| negative rank 2 | max_threads(4) | 262,144 | 547.38 us `[547.27, 547.53]` |
| compact rank 2 | serial | 1,048,576 | 7.3620 ms `[7.1240, 7.5266]` |
| compact rank 4 | serial | 1,048,576 | 8.8040 ms `[8.6655, 8.9074]` |
| compact rank 8 | serial | 1,048,576 | 9.5136 ms `[9.3448, 9.7056]` |
| non-unit rank 2 | serial | 1,048,576 | 7.2503 ms `[7.1122, 7.3990]` |
| negative rank 2 | serial | 1,048,576 | 7.2727 ms `[7.0886, 7.5399]` |
| compact rank 2 | max_threads(4) | 1,048,576 | 2.1625 ms `[2.1611, 2.1638]` |
| compact rank 4 | max_threads(4) | 1,048,576 | 2.4755 ms `[2.4740, 2.4772]` |
| compact rank 8 | max_threads(4) | 1,048,576 | 2.8770 ms `[2.8721, 2.8829]` |
| non-unit rank 2 | max_threads(4) | 1,048,576 | 2.1878 ms `[2.1869, 2.1892]` |
| negative rank 2 | max_threads(4) | 1,048,576 | 2.1606 ms `[2.1575, 2.1626]` |
| rank-one control | serial | 4,096 | 3.9249 us `[3.8439, 4.0174]` |
| rank-one control | max_threads(4) | 4,096 | 3.9109 us `[3.8486, 3.9521]` |
| rank-one control | serial | 32,768 | 30.665 us `[30.003, 31.371]` |
| rank-one control | max_threads(4) | 32,768 | 31.080 us `[30.307, 31.917]` |
| rank-one control | serial | 262,144 | 241.89 us `[234.20, 256.46]` |
| rank-one control | max_threads(4) | 262,144 | 120.04 us `[119.96, 120.10]` |
| rank-one control | serial | 1,048,576 | 1.0278 ms `[1.0008, 1.0502]` |
| rank-one control | max_threads(4) | 1,048,576 | 449.04 us `[448.84, 449.25]` |

All predeclared performance gates are **PASS**:

- medium serial rank 4: 7.56x faster; rank 8: 11.35x faster;
- medium four-thread rank 4: 2.65x faster; rank 8: 3.78x faster;
- medium serial non-unit: 5.71x faster; negative: 5.72x faster;
- candidate medium rank-8/rank-2 per-output ratio: 1.35, below 1.5;
- every generic case improved with `p < 0.05`; no generic or rank-one control
  regressed.

The rank-one controls improved by up to roughly 17% on the candidate CCD,
showing a host/CCD frequency difference, but the generic improvements are
57-92% and every primary ratio remains well beyond its gate after that control
shift.

## Verification and review

Focused verification after implementation:

- default gather/uninitialized integration tests: 79 passed
- parallel gather/uninitialized/policy tests: 88 passed
- default and `parallel` `cargo check -p strided-kernel`: pass
- `cargo fmt --all -- --check`: pass
- `cargo test --workspace`: 904 passed, 9 ignored
- `cargo doc --workspace --no-deps`: pass
- deterministic repository-rules review: pass, no findings
- repository-rules review script: 83 passed

Local `cargo llvm-cov --workspace --features parallel` ran the complete test
matrix. The modified `gather_plan.rs` reached 82.95% line coverage, above this
repository's 80% file threshold. The global checker still reported three
pre-existing, unmodified files below their configured thresholds:
`reduce_view.rs` 71.5% < 80%, `static_indexing_plan.rs` 76.3% < 80%, and
`strided-perm/src/hptt/execute.rs` 57.0% < 65%. They are outside #237; hosted CI
remains authoritative for the exact PR coverage gate.

A `reviewer-flash` safety/semantic preflight of production candidate `d95a387`
found no Critical or Important issue and seven Minor observations. Disposition:

- Fixed the zero-dimension span-validation coupling by validating every sibling
  axis even when total output is zero, and added an overflow regression test.
  This changes compile-time validation only; timed replay is byte-for-byte
  unchanged, so the complete candidate benchmark carries forward.
- Existing INVARIANT/SAFETY comments already state the unchecked replay-delta
  proof; adding per-element debug span checks would recreate the audited hot-loop
  cost.
- The theoretical rank-one `usize`→`isize` and destination-base observations
  concern the unchanged #236 fast path and remain covered by validated real
  allocations.
- Clamped and explicit multi-component performance cases were not added after
  baseline because the paired protocol forbids changing the declared matrix.
  Their correctness is covered here; Phase 8's durable benchmark-suite task,
  already tracked in #213, will add those separate published cases without
  rewriting this experiment.
- The different-CCD control shift is disclosed above rather than hidden.

A current-toolchain exploratory `cargo clippy -D warnings` was also run. It is
not a repository PR gate and reported existing workspace lints across
`strided-view`/`strided-kernel`; the one new `manual_contains` lint in
`validate_layout_span` was fixed.

`reviewer-flash` reviewed exact candidate `d01a82a` and returned
**Correct-to-merge**, with no Critical or Important findings. Its suggested
error-variant assertion was already present for both overflow cases in the
submitted source, so no change was required. The only follow-up here corrects
its valid wording note from roughly 15% to roughly 17%; production, tests, and
benchmark evidence are unchanged. Hosted CI remains the final merge gate.
