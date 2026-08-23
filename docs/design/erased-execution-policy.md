# Erased execution policy and threshold evidence

This document is the entry point from an erased CPU kernel's threading decision
to its implementation, benchmark, decision record, and published results.

## Navigation

| Question | Source of truth |
|---|---|
| What selects serial versus parallel execution? | `strided-kernel/src/threading.rs` |
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

- axis reductions, partitioned over independent outputs;
- gather, partitioned over independent outputs;
- dynamic slice;
- the overwrite phase of dynamic update slice; and
- pad fill and input-copy phases.

Additive scatter remains serial because repeated indices are order-sensitive.
It must preserve deterministic column-major update order unless a future API
provides a separately validated non-overlap contract. Raw `CopyPlan` replay
also remains serial: the initial range-partitioned implementation regressed
large contiguous copies. Both remain benchmark targets.

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
