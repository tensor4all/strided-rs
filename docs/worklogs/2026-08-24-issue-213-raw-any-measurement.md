# Issue #213: `raw_any` integer preflight measurement

## Scope

This is the measurement gate for the remaining low-priority `raw_any` finding in issue #213. `raw_any` is private and is called only by `erased_zip_into` for integer divide/remainder zero-divisor preflight. Production code is out of scope until the frozen baseline proves material cost and useful headroom.

Base: `f875cc894d72187416367c26825c5cb7fca726c2`.

## Contract boundary

`erased_zip_into` must reject any reachable zero divisor before writing, including compact, permuted, non-unit, negative-stride, offset, rank-zero, and zero-extent layouts. The scan may stop at the first zero. The successful divide/remainder traversal remains separately owned by `map_view`; setup and descriptor validation stay outside timed regions.

The current private scan computes total length once, then performs flat-to-multi-index divide/remainder and a checked rank-length offset sum for every visited element. It is always serial, even when the subsequent zip replay uses a bounded parallel context.

## Frozen measurement matrix

Add a Criterion group to `strided-kernel/benches/erased_policy_thresholds.rs` without changing production:

- sizes: `2^12`, `2^15`, `2^18`, `2^20` from the existing threshold profile;
- layouts with equal reachable element counts: compact rank 1, 2, 4, and 8; rank-2 negative source with nonzero offset; rank-2 non-unit source with holes;
- all operands and divisors use `i32`; divisors are exactly `1 + index % 97`, so every scan visits the complete domain;
- benchmark-local `current_scan` duplicates the existing algorithm exactly;
- benchmark-local `incremental_scan` decodes once and advances checked prepared steps/carries, solely to measure optimization headroom;
- compact rank-1 and rank-8 public controls measure serial and `max_threads(4)` `erased_zip_into` divide and add without timing descriptor construction or allocation. Add is a same-dispatch control only: `divide - add` is not treated as preflight share because arithmetic costs differ. The only share estimate is standalone `current_scan / public_divide` for the matching compact rank/context.

Criterion settings remain 10 samples, 300 ms warmup, and 1 s measurement. Run groups sequentially on pinned idle cores in one L3 domain with `STRIDED_KERNEL_ERASED_POLICY_BENCH_PROFILE=threshold`, `STRIDED_KERNEL_ERASED_POLICY_BENCH_THREADS=4`, `RAYON_NUM_THREADS=4`, and `taskset -c 1-4`; record the pre-run load gate and exact command.

## Need-before-implementation gate

Create a focused child implementation issue only if all apply at `N=2^18`:

1. `current_scan` costs at least 0.5 ms in any rank-2/4/8 or negative/non-unit case, or its standalone estimate is at least 15% of matching compact rank-1 or rank-8 public divide time;
2. `incremental_scan` is at least 1.5x faster for compact rank 2 and at least 2x faster for compact rank 4 or 8, with non-overlapping confidence intervals;
3. negative/non-unit controls show the same direction and no semantic or allocation contract needs to change.

If the gate fails, record the evidence on #213 and do not change production. If it passes, create one child issue and a separate implementation design. The implementation must preserve early exit and exact validation-before-write semantics, stay private, allocate nothing through rank 8, retain checked compile/setup arithmetic, and add compact/generic/negative/non-unit/rank-zero/zero-extent/early-zero/late-zero correctness tests plus exact post-diff review.

## Review gate

Selected reviewer: read-only `reviewer-flash`, high thinking. Review of `c9d6cbb3` returned **Correct-to-merge** with four nonblocking wording/pinning clarifications; those clarifications are incorporated here before benchmark implementation. A later production change, if justified, requires its own reviewed design and exact-final-diff verdict.

## Frozen measurement result

Benchmark commit `ed3ee36` ran on EPYC CPUs 25-28 (one L3 domain) because the predeclared 1-4 domain had active housekeeping load. The accepted pre-run gate was selected average 0.2% and other-domain-core maximum 3.0%. Environment: `STRIDED_KERNEL_ERASED_POLICY_BENCH_PROFILE=threshold`, `STRIDED_KERNEL_ERASED_POLICY_BENCH_THREADS=4`, `RAYON_NUM_THREADS=4`, `taskset -c 25-28`; scan and public groups ran sequentially.

### Medium scan matrix (`N=262,144`)

| layout | current | incremental probe | speedup | interval-bound speedup |
|---|---:|---:|---:|---:|
| compact_rank1 | 0.4791 ms | 0.1851 ms | 2.59x | 2.51-2.73x |
| compact_rank2 | 0.8787 ms | 0.3428 ms | 2.56x | 2.46-2.69x |
| compact_rank4 | 1.7916 ms | 0.4180 ms | 4.29x | 4.06-4.43x |
| compact_rank8 | 3.6818 ms | 0.4878 ms | 7.55x | 7.24-8.02x |
| rank2_negative | 0.9265 ms | 0.3546 ms | 2.61x | 2.52-2.72x |
| rank2_nonunit | 0.9045 ms | 0.3431 ms | 2.64x | 2.51-2.74x |

### Medium public controls

| rank | context | add | divide | standalone current-scan / divide |
|---:|---|---:|---:|---:|
| 1 | serial | 0.3008 ms | 0.9349 ms | 51.2% |
| 1 | max_threads_4 | 0.1009 ms | 0.7206 ms | 66.5% |
| 8 | serial | 0.3397 ms | 4.6390 ms | 79.4% |
| 8 | max_threads_4 | 0.1022 ms | 4.3616 ms | 84.4% |

### Decision

The need-before-implementation gate is **PASS**: compact rank 2/4/8 current scans cost 0.879/1.792/3.682 ms; current scan is 51.2%/79.4% of matching rank-1/rank-8 serial divide; incremental headroom is 2.56x at rank 2, 4.29x at rank 4, and 7.55x at rank 8 with non-overlapping intervals; negative/non-unit controls improve 2.61x/2.64x. Create a focused child issue and a separately reviewed implementation design before touching production.

Exact scan command (the public-control command differed only in the final filter):

```bash
STRIDED_KERNEL_ERASED_POLICY_BENCH_PROFILE=threshold \
STRIDED_KERNEL_ERASED_POLICY_BENCH_THREADS=4 \
CARGO_TARGET_DIR=/tmp/strided213-target RAYON_NUM_THREADS=4 \
  taskset -c 25-28 cargo bench -p strided-kernel \
  --bench erased_policy_thresholds --features parallel -- erased_raw_any_scan
```

Exact review of measurement commit `72e9fdec` by read-only `reviewer-flash` (high) returned **Correct-to-merge**; its only findings were nonblocking evidence-presentation details.
