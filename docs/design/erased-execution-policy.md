# Erased execution policy and threshold evidence

This document is the entry point from an erased CPU kernel's threading decision
to its implementation, benchmark, decision record, and published results.

## Navigation

| Question | Source of truth |
|---|---|
| What selects serial versus parallel execution? | `strided-basic/src/threading.rs` (shared by ordinary and fused execution) |
| How is the threshold exercised? | `strided-kernel/benches/erased_policy_thresholds.rs` |
| Why was the current threshold selected? | This document and [strided-rs#163](https://github.com/tensor4all/strided-rs/issues/163) |
| Where does one implementation session record its evidence? | A dated file under `docs/worklogs/` |
| Where do durable cross-project tables belong? | [`strided-rs-benchmark-suite`](https://github.com/tensor4all/strided-rs-benchmark-suite) |

Repository-local Criterion targets are regression and threshold tools. Published
cross-language or cross-project tables must live in the external benchmark
suite with their source commit, command, environment, and raw results. If the
external suite has no matching family yet, add it there before publishing a
comparison; do not put a performance table in a crate README.

## Current decision

The repository uses one threshold,
`threading::MINTHREADLENGTH == 1 << 15` elements. A family enters parallel
replay only when:

- the active execution policy exposes more than one worker; and
- the family-specific independent domain contains more than
  `MINTHREADLENGTH` elements.

Small tensors, `ExecContext::serial()`, `ExecContext::max_threads(1)`, nested
fanout, and single-thread pools stay on the serial path. Exactly-threshold
inputs also stay serial.

The policy-aware parallel replay currently covers:

- full reductions, partitioned over layout traversal ranges that each run the
  serial run kernel, with partials combined in a fixed tree (both the
  one-operand `reduce`/`sum` and the two-operand `zip_reduce`);
- axis reductions, partitioned over independent outputs. Their threshold
  domain is the number of source elements read (outputs times reduced
  elements), not the output count, so shapes such as a 2048 by 2048 matrix
  reduced along one axis split; the worker count is capped by the output
  count. This measure was adopted for issue #269; its benchmark record is
  pending in that issue;
- the single-axis plans (`ErasedScanPlan`, `ErasedArgReducePlan`,
  `ErasedNormPlan`), partitioned over independent units: one line along the
  plan axis, or, when that axis is strided but a kept axis is contiguous, a
  block of up to 64 adjacent lines. The threshold domain is the number of
  source elements and the worker count is capped by the unit count. Each
  line's result is independent of the partition, so these plans are bitwise
  identical at every thread count (see the
  [worklog](../worklogs/2026-10-06-axis-scan-argreduce-norm.md));
- gather, partitioned over independent outputs;
- dynamic slice;
- the overwrite phase of dynamic update slice; and
- pad fill and input-copy phases.

Additive scatter remains serial because repeated indices are order-sensitive.
It must preserve deterministic column-major update order unless a future API
provides a separately validated non-overlap contract. Raw `CopyPlan` replay
also remains serial: the initial range-partitioned implementation regressed
large contiguous copies. Both remain benchmark targets.

### Reduction evaluation order and the deterministic route

The public reduction entries (`reduce`, `zip_reduce`, `sum`, `dot`) do not
promise a strict left-to-right fold, and none of them promises cross-machine
bitwise equality: independent lane kernels, cache-friendly traversal orders,
SIMD kernels and the parallel partial-combine tree all reorder the additions.
What they do promise is documented per entry:

- under the sequential policy — `ExecContext::serial()` or
  `with_execution_policy(ExecutionPolicy::Sequential, ..)` — the result is
  independent of the ambient Rayon pool width, because only serial branches
  run; this is the deterministic route for a caller that needs one result per
  input, and it is asserted for `reduce`, `zip_reduce`, `sum` and `dot` by
  `strided-basic/tests/zip_reduce.rs`;
- under a parallel policy the split depends on the effective thread count, so
  the result is repeatable for a fixed thread count only.

`zip_reduce` (tensor4all/strided-rs#280) is the two-input mapped reduction that
makes a widening accumulator expressible (`f32` operands accumulated in `f64`);
`dot` cannot widen because it pins the result type to `A * B`. `zip_reduce`
reuses the full-reduction threshold and the same partial-combine tree as
`reduce`.

## Changing a threshold or fast path

Before implementation, record the following in the issue or dated worklog:

- baseline and candidate commits;
- benchmark source and release profile;
- CPU model, topology, affinity, and thread/provider settings;
- complete size, shape, layout, dtype, and thread-count matrix;
- comparison statistic and acceptance thresholds;
- sample/repetition policy; and
- host-noise observations and invalidation thresholds.

Run the complete baseline and candidate matrix sequentially under the same
protocol. A failed host-noise gate makes the experiment inconclusive; do not
selectively rerun favorable cases. Preserve small serial fallback whenever
scheduler overhead erases the benefit.

A performance change is accepted only when its primary speedup gate,
correctness checks, and declared non-regression gates all pass. Record negative
and inconclusive results as evidence rather than changing the gate after seeing
the data.

## Running the repository benchmark

The benchmark compares `ExecContext::serial()` with
`ExecContext::max_threads(n)` for each selected size. Pin all participating
threads to idle cores in one cache domain and run benchmark processes
sequentially.

```bash
git rev-parse HEAD
lscpu -e=CPU,CORE,SOCKET,NODE,CACHE

RAYON_NUM_THREADS=4 \
STRIDED_KERNEL_ERASED_POLICY_BENCH_PROFILE=threshold \
STRIDED_KERNEL_ERASED_POLICY_BENCH_THREADS=4 \
taskset -c 40-43 \
  cargo bench -p strided-kernel --features parallel \
    --bench erased_policy_thresholds
```

Use `STRIDED_KERNEL_ERASED_POLICY_BENCH_PROFILE=smoke` only to check the
harness. A threshold decision requires below-threshold, near-threshold, and
clearly tensor-sized cases plus every intended bounded thread count.

## Result record

A worklog or external result record must include:

- strided-rs commit;
- CPU model and logical/physical CPU count;
- cache/NUMA topology and affinity;
- enabled features and thread environment;
- benchmark profile and complete commands;
- host load and known competing processes;
- all measured cases and uncertainty/variance;
- the proposed threshold or reason to stay serial; and
- links to the implementation issue and PR.

The initial threshold decision and exploratory measurements are preserved in
[strided-rs#163](https://github.com/tensor4all/strided-rs/issues/163). Current
indexed-loop optimization work is tracked in
[strided-rs#213](https://github.com/tensor4all/strided-rs/issues/213).
