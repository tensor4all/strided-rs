# Pad contiguous-run implementation record

Issue: #170

## Cause

`PadPlan` decoded a full destination coordinate for the fill pass and a full
operand coordinate for the copy pass. The same scalar traversal was used for
dense one-dimensional edge padding, where both passes are contiguous.

## Implementation

- Compile an axis-0 run only when operand and destination axis-0 strides are
  one and axis-0 has no interior spacing.
- Fill a standard column-major destination through one contiguous slice.
- Copy the clipped axis-0 interval as one non-overlapping run for each valid
  outer coordinate.
- Retain the coordinate fallback for arbitrary strides, interior padding, and
  layouts that do not meet the run contract.

The compile step owns all run bounds. The fast path adds no replay allocation;
`CoordScratch` retains its existing heap spill above `RAW_FUSED_RANK_LIMIT`.

## Verification

The initial contract test failed because `PadPlan` had no contiguous-run
metadata. It passes after implementation.

Focused coverage includes every `KernelDType`, signed cropping with interior
padding, empty input to non-empty output, rank zero, non-contiguous
destinations, and allocation-free erased replay.

Tenferro public API measurements used 15 runs and 3 warmups, CPU 60 for one
thread, CPUs 56-59 for four threads, and system OpenBLAS:

| Surface | Threads | Pre-adoption | Scalar plan | Run plan |
|---|---:|---:|---:|---:|
| eager | 1 | 16.103 ms | 25.948 ms | 1.078 ms |
| trace | 1 | 15.895 ms | 25.449 ms | 1.133 ms |
| eager | 4 | 16.127 ms | 4.873 ms | 1.096 ms |
| trace | 4 | 17.226 ms | 4.800 ms | 1.126 ms |

The run plan removes the tenferro +20% stop-the-line regression.
