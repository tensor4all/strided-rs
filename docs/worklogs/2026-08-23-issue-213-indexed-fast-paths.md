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
documentation-only changes. Candidate commit: to be recorded after
implementation.

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

Baseline results, candidate results, verification, and residual risks will be
appended after the complete runs.
