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
