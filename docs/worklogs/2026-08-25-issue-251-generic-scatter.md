# Issue #251: generic additive scatter replay

## Scope

The final #213 audit found one remaining prohibited loop in generic
`ScatterPlan`: every window value rebuilds update/operand coordinates and runs
two checked rank scans. The rank-one scalar specialization remains correct but
does not cover windowed rank-2/4/8 layouts.

Base: umbrella commit `5009c59e37c78908e0cb76a1d3252f01210dbb6a`.
Selected reviewer: read-only `reviewer-flash`, high thinking, for the design and
exact final diff.

## Contract boundary

Additive scatter first copies operand to destination, then replays updates in
strict column-major batch order and, within each batch, strict column-major
window order. Repeated/overlapping destinations observe that exact accumulation
order. Index components are clamped per mapped operand axis. Initialized and
uninitialized destinations share the same read-modify-write replay after copy.
The replay stays serial; a bounded context may only parallelize the independent
operand copy.

Preserve public API, dtype/index dispatch, wrapping integer combine functions,
rank-one specialization, errors, and empty/rank-zero behavior.

## Minimal implementation

Reuse the existing private `WindowReplay`; do not add a public or common cursor
abstraction.

At compile time:

1. Build one batch replay over `batch_shape`. Its source strides are index-batch
   strides (index-vector axis excluded); its destination strides are update
   batch strides (update-window axes excluded).
2. Precompute checked index-component offsets from the index-vector stride.
3. Build one window replay over `window_shape_updates`. Its source strides are
   update-window strides in `spec.update_window_dims` order; destination strides
   are destination strides in `window_dims` order.
4. When `index_vector_dim == index_dims.len()`, use the single zero component
   offset; there is no physical vector-axis stride.
5. Mirror the proven dynamic-update storage disposition: with no `parallel`
   feature, store one private `ScatterReplay` containing both replays and
   component offsets; with `parallel`, validate by compiling at plan creation
   but build SmallVec-backed replay metadata only inside the generic execution
   branch after the rank-one specialization returns. This keeps the rank-one
   plan/execution footprint unchanged and generic rank<=8 execution allocation-free.
6. Retain checked spans, steps, carry resets, totals, and the existing copy plan.

At execution:

- decode batch replay once at linear zero;
- for each batch, read index components at the prepared component offsets and
  compute the clamped destination window base. This component loop is the
  genuine data-dependent indirect lookup boundary;
- decode the prepared window replay from the current update-batch base and
  destination-window base, then load/add/advance incrementally for every window
  value;
- advance the batch replay once and continue.

No full `update_idx`, `operand_idx`, or per-window checked rank scan remains.
The rank-one path is byte-for-byte unchanged.

## Frozen benchmark

Extend `erased_policy_thresholds.rs` with
`erased_scatter_generic_rank_layout`, leaving existing
`erased_scatter_additive` as the rank-one control.

For compact rank `r` in 2/4/8:

- operand/destination shape `[batch, 2, ..., 2]` (rank `r`);
- axis 0 is the inserted/scatter-mapped dimension;
- each update has a full window over the remaining axes;
- `window_elems = 2^(r-1)`, `batch = N/window_elems`, so every case replays `N`
  updates;
- indices shape `[batch, 1]`, use a deterministic permutation and include
  repeated destinations in correctness tests, not the timing fixture;
- updates shape `[batch, 2, ..., 2]`; all compact layouts are column-major.

Add exact rank-2 layout controls with shape `[batch, 2]` and compact indices
`[batch, 1]` using permutation `(5*i + 1) mod batch`:

- negative update: strides `[-1, batch]`, offset `batch-1`, compact injective
  destination;
- non-unit update/destination: strides `[2, 1]`, offset zero, physical length
  `2*batch`; this row-major-like layout is injective for destination.

Construct every raw descriptor and compile every plan before freezing baseline;
invalid layouts are a benchmark defect. Plan/descriptor/data construction
remains outside timing. Time execution and black-box destination.

Use threshold sizes `2^12`, `2^15`, `2^18`, `2^20`, serial and
`max_threads(4)`, 10 samples, 300 ms warmup, 1 s measurement,
`RAYON_NUM_THREADS=4`, thread override 4. Run groups sequentially on four pinned
cores in one EPYC L3 domain. Before each complete baseline/candidate run every
selected core must be below 2% busy over four seconds and siblings below 20%; a
failed gate produces no timing and must be recorded.

Need gate at `N=2^18`: compact rank 4 or 8 serial exceeds 1 ms or is at least
2x compact rank 2; and one rank/layout case adds at least 25% per-update cost.
If it fails, retain benchmark evidence and do not edit production.

Candidate gates when need passes:

- compact rank 4 and 8 serial >=2x faster;
- compact rank 4 and 8 max-threads(4) context >=1.5x faster;
- negative/non-unit serial >=1.5x faster;
- candidate rank-8/rank-2 per-update ratio <=1.5;
- every selected generic cell improves with non-overlapping intervals;
- existing rank-one control has no >10% regression.

## Correctness and verification

Add or extend tests for compact rank 2/4/8, negative/non-unit layouts and
nonzero offsets, clamping, repeated/overlapping windows in exact replay order,
i32/i64 wrapping, initialized/uninitialized parity, empty domains, rank-one
specialization selection, and zero execution allocations through rank 8.

Run focused default/parallel/indexed/uninitialized/allocation/source-contract
tests, default/parallel workspace tests, modified-file coverage, docs,
formatting, deterministic repository-rules review, exact-final independent
review, and hosted CI.

## Design gate

Read-only `reviewer-flash` with high thinking reviewed exact design `8ebceea9`
and returned **Correct-to-merge** for benchmark implementation. Its two
Important design pins (feature-aware replay storage and exact valid layout
recipes) plus the imaginary-vector-axis zero offset are incorporated above
before any production edit.

## Baseline evidence

Benchmark-only commit `9e00906` ran on CPUs 33-36 in L3 domain 32-39. The first gate failed because CPU 34 was 3.3% busy and produced no timing; the accepted retry had every domain core at 0.0%. Generic and rank-one control groups ran sequentially with separate setup outside timing.

| family | case | estimate `[low, high]` |
|---|---|---:|
| generic | compact_rank2_serial/n4096 | 110.84 µs `[110.43 µs, 111.96 µs]` |
| generic | compact_rank4_serial/n4096 | 168.78 µs `[163.92 µs, 177.58 µs]` |
| generic | compact_rank8_serial/n4096 | 321.33 µs `[319.09 µs, 327.21 µs]` |
| generic | rank2_negative_update_serial/n4096 | 110.76 µs `[110.32 µs, 111.54 µs]` |
| generic | rank2_nonunit_update_dest_serial/n4096 | 115.37 µs `[113.36 µs, 117.80 µs]` |
| generic | compact_rank2_max_threads_4/n4096 | 111.48 µs `[109.99 µs, 113.33 µs]` |
| generic | compact_rank4_max_threads_4/n4096 | 166.43 µs `[163.28 µs, 170.53 µs]` |
| generic | compact_rank8_max_threads_4/n4096 | 320.36 µs `[319.45 µs, 322.54 µs]` |
| generic | rank2_negative_update_max_threads_4/n4096 | 115.48 µs `[111.62 µs, 119.04 µs]` |
| generic | rank2_nonunit_update_dest_max_threads_4/n4096 | 114.82 µs `[113.44 µs, 116.49 µs]` |
| generic | compact_rank2_serial/n32768 | 890.99 µs `[881.84 µs, 905.78 µs]` |
| generic | compact_rank4_serial/n32768 | 1.3737 ms `[1.3136 ms, 1.4369 ms]` |
| generic | compact_rank8_serial/n32768 | 2.6042 ms `[2.5664 ms, 2.6410 ms]` |
| generic | rank2_negative_update_serial/n32768 | 893.90 µs `[882.68 µs, 909.40 µs]` |
| generic | rank2_nonunit_update_dest_serial/n32768 | 928.86 µs `[908.16 µs, 949.21 µs]` |
| generic | compact_rank2_max_threads_4/n32768 | 895.03 µs `[880.63 µs, 909.66 µs]` |
| generic | compact_rank4_max_threads_4/n32768 | 1.3262 ms `[1.3083 ms, 1.3508 ms]` |
| generic | compact_rank8_max_threads_4/n32768 | 2.5926 ms `[2.5638 ms, 2.6405 ms]` |
| generic | rank2_negative_update_max_threads_4/n32768 | 899.67 µs `[880.96 µs, 928.46 µs]` |
| generic | rank2_nonunit_update_dest_max_threads_4/n32768 | 917.04 µs `[904.55 µs, 934.53 µs]` |
| generic | compact_rank2_serial/n262144 | 7.1342 ms `[7.0848 ms, 7.2578 ms]` |
| generic | compact_rank4_serial/n262144 | 10.872 ms `[10.536 ms, 11.294 ms]` |
| generic | compact_rank8_serial/n262144 | 21.466 ms `[20.798 ms, 21.874 ms]` |
| generic | rank2_negative_update_serial/n262144 | 7.3089 ms `[7.0894 ms, 7.5736 ms]` |
| generic | rank2_nonunit_update_dest_serial/n262144 | 7.4482 ms `[7.2961 ms, 7.6766 ms]` |
| generic | compact_rank2_max_threads_4/n262144 | 7.1864 ms `[7.0765 ms, 7.3303 ms]` |
| generic | compact_rank4_max_threads_4/n262144 | 10.608 ms `[10.487 ms, 10.782 ms]` |
| generic | compact_rank8_max_threads_4/n262144 | 21.294 ms `[20.471 ms, 21.933 ms]` |
| generic | rank2_negative_update_max_threads_4/n262144 | 7.1282 ms `[7.0605 ms, 7.2760 ms]` |
| generic | rank2_nonunit_update_dest_max_threads_4/n262144 | 7.3069 ms `[7.2534 ms, 7.4524 ms]` |
| generic | compact_rank2_serial/n1048576 | 29.578 ms `[28.618 ms, 30.784 ms]` |
| generic | compact_rank4_serial/n1048576 | 42.301 ms `[41.849 ms, 42.945 ms]` |
| generic | compact_rank8_serial/n1048576 | 85.050 ms `[83.034 ms, 87.450 ms]` |
| generic | rank2_negative_update_serial/n1048576 | 29.062 ms `[28.531 ms, 29.860 ms]` |
| generic | rank2_nonunit_update_dest_serial/n1048576 | 29.226 ms `[28.998 ms, 29.511 ms]` |
| generic | compact_rank2_max_threads_4/n1048576 | 28.584 ms `[28.353 ms, 28.864 ms]` |
| generic | compact_rank4_max_threads_4/n1048576 | 42.405 ms `[41.905 ms, 43.040 ms]` |
| generic | compact_rank8_max_threads_4/n1048576 | 84.010 ms `[83.293 ms, 84.865 ms]` |
| generic | rank2_negative_update_max_threads_4/n1048576 | 28.885 ms `[28.313 ms, 29.355 ms]` |
| generic | rank2_nonunit_update_dest_max_threads_4/n1048576 | 29.138 ms `[28.962 ms, 29.507 ms]` |
| rank1 control | serial/small_n4096 | 14.363 µs `[14.225 µs, 14.534 µs]` |
| rank1 control | max_threads_4/small_n4096 | 14.524 µs `[14.167 µs, 15.056 µs]` |
| rank1 control | serial/near_threshold_n32768 | 25.868 µs `[25.692 µs, 26.284 µs]` |
| rank1 control | max_threads_4/near_threshold_n32768 | 24.551 µs `[24.472 µs, 24.697 µs]` |
| rank1 control | serial/medium_n262144 | 976.52 µs `[935.70 µs, 1.0109 ms]` |
| rank1 control | max_threads_4/medium_n262144 | 948.93 µs `[934.12 µs, 967.63 µs]` |
| rank1 control | serial/large_n1048576 | 3.8509 ms `[3.7950 ms, 3.8960 ms]` |
| rank1 control | max_threads_4/large_n1048576 | 3.8466 ms `[3.8049 ms, 3.8956 ms]` |

The need-before-implementation gate is **PASS**. At medium size compact rank 2/4/8 serial measured 7.134/10.872/21.466 ms; rank 4/8 exceed 1 ms, and rank 8 costs 3.01x rank 2. Frozen cases and gates may proceed to production implementation.

## Candidate implementation and corrected paired evidence

Production commits through `eba26e5` add the feature-aware batch/window replay and keep rank-one replay as a direct pointer loop; `e37611a` adds the source contract. Benchmark commit `4a3dacc` fixes a validity flaw discovered by the rank-one control: black-boxing only the erased descriptor let LLVM eliminate or retain destination writes inconsistently. The original `ce8a850` baseline and interim candidate timings are therefore reclassified INCONCLUSIVE and are not used below.

The corrected benchmark black-boxes typed output storage. Corrected baseline branch `5169eba` contains only that benchmark fix on pre-production `9e00906`; candidate `4a3dacc` contains the identical benchmark on the final production tree. Both use separate targets and CPUs 1-4 in L3 domain 0-7. Six baseline gate attempts produced no timing (CPU 0 at 60.9% once and 100% four times, then CPU 4 at 36.8%); the accepted retry had CPU 3 at 0.5% and every other domain core 0.0%. Candidate gate passed first attempt (CPU 0 12.3%, selected maximum 1.3%, all other siblings <=0.5%). Generic and control groups ran sequentially.

| case | context | size | baseline | candidate | speedup | interval-bound speedup |
|---|---|---:|---:|---:|---:|---:|
| compact_rank2 | serial | 262144 | 7.1407 ms | 4.5729 ms | 1.56x | 1.52-1.62x |
| compact_rank4 | serial | 262144 | 10.7580 ms | 1.7254 ms | 6.24x | 6.10-6.37x |
| compact_rank8 | serial | 262144 | 21.2690 ms | 1.3918 ms | 15.28x | 14.50-16.21x |
| rank2_negative_update | serial | 262144 | 7.1313 ms | 4.5494 ms | 1.57x | 1.53-1.61x |
| rank2_nonunit_update_dest | serial | 262144 | 7.4491 ms | 4.7139 ms | 1.58x | 1.52-1.63x |
| compact_rank2 | max_threads_4 | 262144 | 7.2706 ms | 4.5479 ms | 1.60x | 1.54-1.69x |
| compact_rank4 | max_threads_4 | 262144 | 10.6910 ms | 1.7290 ms | 6.18x | 6.02-6.34x |
| compact_rank8 | max_threads_4 | 262144 | 20.8700 ms | 1.3833 ms | 15.09x | 14.63-15.47x |
| rank2_negative_update | max_threads_4 | 262144 | 7.1501 ms | 4.5345 ms | 1.58x | 1.54-1.62x |
| rank2_nonunit_update_dest | max_threads_4 | 262144 | 7.5415 ms | 4.7216 ms | 1.60x | 1.53-1.65x |
| compact_rank2 | serial | 1048576 | 28.5050 ms | 18.1550 ms | 1.57x | 1.54-1.60x |
| compact_rank4 | serial | 1048576 | 43.3730 ms | 7.6843 ms | 5.64x | 5.53-5.78x |
| compact_rank8 | serial | 1048576 | 83.4460 ms | 5.8265 ms | 14.32x | 13.87-14.82x |
| rank2_negative_update | serial | 1048576 | 28.6380 ms | 18.0570 ms | 1.59x | 1.56-1.63x |
| rank2_nonunit_update_dest | serial | 1048576 | 29.3570 ms | 19.1410 ms | 1.53x | 1.44-1.59x |
| compact_rank2 | max_threads_4 | 1048576 | 28.6540 ms | 18.7970 ms | 1.52x | 1.46-1.59x |
| compact_rank4 | max_threads_4 | 1048576 | 42.8740 ms | 7.7766 ms | 5.51x | 5.34-5.70x |
| compact_rank8 | max_threads_4 | 1048576 | 82.5990 ms | 5.9352 ms | 13.92x | 13.63-14.29x |
| rank2_negative_update | max_threads_4 | 1048576 | 28.8610 ms | 18.3430 ms | 1.57x | 1.50-1.63x |
| rank2_nonunit_update_dest | max_threads_4 | 1048576 | 30.2180 ms | 19.2650 ms | 1.57x | 1.49-1.64x |

All corrected performance gates are **PASS**. At medium size compact rank 2/4/8 improves 1.56/6.24/15.28x serial and 1.60/6.18/15.09x with a four-thread context. Negative/non-unit improves 1.57/1.58x serial. Candidate rank-8/rank-2 time ratio is 0.30. Every generic cell at every frozen size/context improved with non-overlapping intervals.

With output storage black-boxed, the existing rank-one control has no regression over 10%: exact-threshold estimates are within 2.8%, medium/large estimates are stable or faster, and the largest point-estimate regression is 2.8%. The accepted implementation preserves deterministic serial update replay; four-thread gains reflect copy policy plus lower replay overhead, not parallel accumulation.

## Verification

- focused default indexed/uninitialized/source-contract: 88 passed
- focused parallel indexed/uninitialized/policy/source-contract: 99 passed
- rank-8 allocation test: zero execution allocations
- default workspace: 922 passed, 9 ignored
- parallel workspace: 996 passed, 9 ignored
- `cargo doc --workspace --no-deps`: passed
- formatting and parallel cargo check: passed
- modified coverage: `gather_plan.rs` 93.14%, `copy_plan.rs` 98.77% (threshold 80%); only unchanged `reduce_view.rs` remains below the global package threshold

Exact-final read-only `reviewer-flash` review of candidate `ed9b61ff` with high thinking returned **Correct-to-merge** with no blocking findings. It confirmed cfg-aware replay storage, arbitrary axis mapping, order/safety/wrapping preservation, rank-one behavior, corrected benchmark validity, all gates, tests, allocation, coverage, docs, and rules evidence. Hosted CI remains pending.
