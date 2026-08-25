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
4. Retain checked spans, steps, carry resets, totals, and the existing copy plan.

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

Add rank-2 negative-update and non-unit update/destination variants with the
same logical shapes and validated physical holes/offsets. Plan/descriptor/data
construction remains outside timing. Time execution and black-box destination.

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
