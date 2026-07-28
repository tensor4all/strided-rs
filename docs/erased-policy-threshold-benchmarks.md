# Erased Policy Threshold Benchmarks

This note describes how to collect evidence for issue #163. It intentionally
does not publish benchmark results. Keep durable result tables in the external
`tensor4all/strided-rs-benchmark-suite` repository.

## Scope

The `erased_policy_thresholds` benchmark compares the same erased prepared plan
under:

- `ExecContext::serial()`;
- `ExecContext::max_threads(n)`.

The benchmark covers the erased plan families that still contain serial loops
or serial raw replay paths:

- axis/multi-axis sum reduction;
- gather / indexed read;
- dynamic slice;
- dynamic update slice;
- pad;
- raw copy replay;
- additive scatter as a measurement target only.

Additive scatter remains order-sensitive when updates overlap. Treat its
benchmark as evidence for future design, not as permission for a blanket
parallel implementation.

## Running

The benchmark defaults to a conservative two-thread comparison when
`STRIDED_KERNEL_ERASED_POLICY_BENCH_THREADS` is not set. Set the thread count
explicitly for threshold work.

Run threshold-setting measurements only on a quiet machine. If other heavy
processes are running, mark the run as exploratory and do not use it to choose
thresholds.

Recommended threshold run:

```bash
git rev-parse HEAD
lscpu
RAYON_NUM_THREADS=64 \
STRIDED_KERNEL_ERASED_POLICY_BENCH_PROFILE=threshold \
STRIDED_KERNEL_ERASED_POLICY_BENCH_THREADS=64 \
cargo bench -p strided-kernel --features parallel --bench erased_policy_thresholds
```

Repeat the threshold run with each intended bounded thread count, for example
2, 8, and the high-core production budget. Each run compares that bounded
thread count against the serial context.

For a quick harness check:

```bash
STRIDED_KERNEL_ERASED_POLICY_BENCH_PROFILE=smoke \
STRIDED_KERNEL_ERASED_POLICY_BENCH_THREADS=2 \
cargo bench -p strided-kernel --features parallel --bench erased_policy_thresholds
```

Do not run threshold-setting benchmark jobs concurrently with other benchmark
jobs. Do not mix BLAS/OpenMP provider thread policies into the same run.

## Result Note Template

Record at least:

- strided-rs commit;
- CPU model and logical CPU count;
- OS and scheduler/affinity constraints, if any;
- `RAYON_NUM_THREADS`;
- `STRIDED_KERNEL_ERASED_POLICY_BENCH_THREADS`;
- benchmark profile;
- whether the machine was idle or known competing processes were present;
- per-family serial versus `max_threads(n)` results;
- variance notes;
- proposed family-specific threshold or rationale for staying serial.

Choose thresholds only from quiet or explicitly isolated runs. Keep small-size
serial fallback when scheduler overhead dominates.
