# Issue #213 rank-one indexed fast paths

Date: 2026-08-23

## Scope

Reduce per-element coordinate reconstruction in the common compact rank-one
scalar gather and deterministic additive-scatter cases. Preserve all generic
rank/layout behavior as the fallback and do not add public API.

The motivating downstream report is
[tensor4all/tenferro-rs#1719](https://github.com/tensor4all/tenferro-rs/issues/1719).
Source tracing showed that direct `ErasedGatherPlan`/`ErasedScatterPlan` replay
accounts for nearly all of tenferro's indexed-operation time.

## Context read

- `AGENTS.md` and `REPOSITORY_RULES.md`
- shared common repository, performance, Rust performance, docs/tests, and
  provenance rules
- `docs/design/erased-execution-policy.md`
- `strided-kernel/src/gather_plan.rs`, `erased.rs`, and `threading.rs`
- `strided-kernel/benches/erased_policy_thresholds.rs`
- issues [#163](https://github.com/tensor4all/strided-rs/issues/163) and
  [#213](https://github.com/tensor4all/strided-rs/issues/213)

No third-party implementation is used for this change. The candidate is a
specialization of the repository's existing validated gather/scatter plans.

## Design

- Detect only the already-validated compact rank-one scalar-take/scalar-update
  layouts used by the public tenferro path.
- Gather directly walks index and destination offsets. It keeps the repository
  threshold and partitions independent outputs only above the threshold.
- Scatter copies the operand through the existing `CopyPlan`, then applies
  updates in the existing deterministic serial order. Repeated indices must
  accumulate exactly as before.
- All other ranks, strides, windows, and index-vector forms retain the existing
  generic traversal.

Rejected or deferred:

- A new public uniqueness assertion for parallel scatter: unnecessary for the
  serial fast path and a separate semantic contract.
- Tenferro's I64 index clone: measured at about 0.08 ms versus 5-7 ms in the
  lower-layer replay, so it is not the primary fix.
- General incremental-offset refactoring for every indexed layout: #213 still
  tracks that broader work; this change addresses the measured common case.

## Predeclared performance experiment

Baseline source commit: `dc0a8e03286c61a84d56446b5cc2c53295f75d76` plus
documentation-only changes. Candidate production commit: `f8997fd`.

Benchmark source:
`strided-kernel/benches/erased_policy_thresholds.rs`, release Criterion profile
`threshold`, gather and additive-scatter groups, all four sizes
`2^12`, `2^15`, `2^18`, and `2^20`, under serial and bounded four-thread
contexts. Benchmark setup remains outside timing.

Hardware/protocol:

- AMD EPYC 7713P, 64 physical cores, no SMT, one NUMA node;
- one four-core set in one L3/CCD, selected immediately before each complete
  run;
- each selected core must average below 2% busy over four seconds and no other
  process may occupy another core in that L3 domain above 20%; otherwise the
  complete run is inconclusive;
- `RAYON_NUM_THREADS=4` and
  `STRIDED_KERNEL_ERASED_POLICY_BENCH_THREADS=4`;
- no BLAS/OpenMP benchmark runs concurrently;
- Criterion's configured 300 ms warmup, 10 samples, and one-second measurement
  time; gather and scatter groups run sequentially.

Primary gates at `medium_n262144`, using Criterion point estimates:

- serial gather: at least 5x faster than baseline and no slower than 1.2 ms;
- four-thread gather: no slower than 0.8 ms;
- serial additive scatter: at least 5x faster than baseline and no slower than
  1.2 ms;
- four-thread-context additive scatter: no slower than 1.2 ms while remaining
  deterministic serial replay.

Non-regression gates:

- no selected point estimate at any measured size/thread count regresses by
  more than 10%;
- generic-layout and existing serial/parallel correctness tests pass;
- repeated indices, clamped indices, initialized output, and uninitialized
  output produce the generic contract's exact values;
- default and `parallel` feature builds pass focused tests.

## Evidence

### Baseline

Documentation commit measured: `f8672f5`. Production source is unchanged from
`dc0a8e0`. The first proposed CPU set failed the predeclared noise gate and was
discarded before timing. CPUs 44-47 then passed: each selected core was
0.0-0.3% busy over four seconds, and the complete L3 domain (CPUs 40-47) was
0.0-0.5% busy.

Criterion point estimates and reported confidence intervals:

| family | context | size | estimate |
|---|---|---:|---:|
| gather | serial | 4,096 | 96.218 us `[94.446, 98.299]` |
| gather | max_threads(4) | 4,096 | 96.436 us `[95.008, 97.760]` |
| gather | serial | 32,768 | 793.56 us `[776.95, 804.97]` |
| gather | max_threads(4) | 32,768 | 807.07 us `[803.78, 810.27]` |
| gather | serial | 262,144 | 6.2187 ms `[6.0127, 6.3665]` |
| gather | max_threads(4) | 262,144 | 1.0574 ms `[1.0571, 1.0578]` |
| gather | serial | 1,048,576 | 24.817 ms `[24.343, 25.237]` |
| gather | max_threads(4) | 1,048,576 | 4.2780 ms `[4.2626, 4.2838]` |
| additive scatter | serial | 4,096 | 96.472 us `[94.890, 99.012]` |
| additive scatter | max_threads(4) | 4,096 | 95.149 us `[92.726, 98.493]` |
| additive scatter | serial | 32,768 | 767.72 us `[746.08, 794.00]` |
| additive scatter | max_threads(4) | 32,768 | 762.24 us `[747.42, 777.36]` |
| additive scatter | serial | 262,144 | 6.2632 ms `[6.1146, 6.3923]` |
| additive scatter | max_threads(4) | 262,144 | 6.3004 ms `[6.0890, 6.4622]` |
| additive scatter | serial | 1,048,576 | 25.524 ms `[25.124, 25.787]` |
| additive scatter | max_threads(4) | 1,048,576 | 25.259 ms `[24.831, 25.746]` |

Commands used the predeclared environment and ran the gather and scatter groups
sequentially with `taskset -c 44-47`.

### Candidate

Candidate production commit: `f8997fd`. A first post-build load check failed
before timing and was discarded. The accepted check found CPUs 44-47 at
0.0-0.2% busy; the same L3 domain peaked at 18.0%, below the predeclared 20%
limit.

| family | context | size | estimate | baseline ratio |
|---|---|---:|---:|---:|
| gather | serial | 4,096 | 3.950 us `[3.913, 3.991]` | 24.4x faster |
| gather | max_threads(4) | 4,096 | 3.943 us `[3.872, 4.040]` | 24.5x faster |
| gather | serial | 32,768 | 30.957 us `[30.489, 31.542]` | 25.6x faster |
| gather | max_threads(4) | 32,768 | 31.092 us `[30.190, 32.040]` | 26.0x faster |
| gather | serial | 262,144 | 250.73 us `[247.05, 253.52]` | 24.8x faster |
| gather | max_threads(4) | 262,144 | 120.72 us `[120.61, 120.81]` | 8.8x faster |
| gather | serial | 1,048,576 | 1.0384 ms `[1.0220, 1.0508]` | 23.9x faster |
| gather | max_threads(4) | 1,048,576 | 452.87 us `[451.96, 453.60]` | 9.4x faster |
| additive scatter | serial | 4,096 | 3.851 us `[3.769, 3.911]` | 25.0x faster |
| additive scatter | max_threads(4) | 4,096 | 3.214 us `[3.007, 3.378]` | 29.6x faster |
| additive scatter | serial | 32,768 | 29.906 us `[29.760, 30.071]` | 25.7x faster |
| additive scatter | max_threads(4) | 32,768 | 29.961 us `[29.753, 30.084]` | 25.4x faster |
| additive scatter | serial | 262,144 | 1.0890 ms `[1.0751, 1.0980]` | 5.8x faster |
| additive scatter | max_threads(4) | 262,144 | 1.0576 ms `[1.0296, 1.0865]` | 6.0x faster |
| additive scatter | serial | 1,048,576 | 4.4556 ms `[4.4347, 4.4712]` | 5.7x faster |
| additive scatter | max_threads(4) | 1,048,576 | 4.3946 ms `[4.3451, 4.4482]` | 5.7x faster |

Criterion classified every case as an improvement with `p < 0.05`. All primary
and non-regression performance gates pass. The gather threshold still keeps
4,096 and exactly 32,768 elements serial; additive scatter remains ordered
serial replay for every context.

### Verification and residual risk

Verification on evidence commit `d993e72`:

- `cargo fmt --all -- --check`
- focused default-feature tests: 80 passed
- focused `parallel`-feature tests: 88 passed
- `cargo test --workspace`: 899 passed, 9 ignored
- `cargo test -p strided-kernel --features parallel`: 517 passed
- deterministic repository-rules review: pass, no findings
- repository-rules review script tests: 79 passed

An independent read-only DeepSeek review of `origin/main...d993e72` returned
**Correct-to-merge**. It found no blocking issue and three minor observations:

- The fast paths use unchecked incremental `isize` offsets after layout
  validation. This is intentional: the nearby `INVARIANT` and `SAFETY`
  comments name the proof, and restoring checked arithmetic per element would
  recreate the measured defect. Existing raw-layout validation and offset tests
  cover the boundary.
- The worklog still had a candidate placeholder; this section fixes it.
- A synthetic fast-versus-generic differential test could be added. It is
  deferred because existing ground-truth tests separately cover rank-one clamp
  and offsets, all supported dtypes, initialized and uninitialized writers,
  above-threshold parallel gather, repeated-index order, and integer wrapping;
  generic windowed and strided fallbacks retain their existing tests.

Residual scope: arbitrary-rank indexed replay still rebuilds coordinates and
checked offsets per element. Issue #213 remains the owner for that broader
incremental-offset work. This change intentionally adds no uniqueness contract
or parallel additive scatter.

### Review provenance and Phase-9 chronology

PR #236 and exact production/evidence candidate `d993e72` predate the approved
Phase-9 continuation and its selection of `reviewer-flash` for subsequent
independently mergeable tasks. The immutable PR body and this worklog record the
then-applicable read-only DeepSeek cross-model **Correct-to-merge** verdict;
commits after `d993e72` only closed documentation links. The maintainer-approved
Phase-9 ledger treated merged #236 (`39111bd7`) as an already completed input,
not as a task whose pre-implementation gate could be rerun retroactively.

Every later independently mergeable implementation task (#237, #238, #239,
#240, and measurement-gated #247) records the selected `reviewer-flash`
design-before-implementation and exact-final verdicts in its dated worklog.
PR #236: <https://github.com/tensor4all/strided-rs/pull/236>.
