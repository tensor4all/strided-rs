# Uninitialized static-output implementation record

Issue: #174

## Cause

Downstream buffer pools need to hand static indexing plans fresh storage
without first constructing valid typed values. The existing erased destination
descriptor requires valid bytes at construction, including the `bool` validity
invariant, so downstream code could only satisfy it by zero-initializing a
buffer that the plan immediately overwrote.

## Contract

- `ErasedRawStridedUninitMut` borrows `MaybeUninit<u8>` storage and validates
  dtype alignment, byte length, shape, strides, offset, and reachable bounds
  without reading value bytes.
- Only slice, reverse, pad, and concatenate expose `execute_uninit`; each plan
  fully initializes every reachable logical destination element on success.
  Unreachable backing-layout holes may remain uninitialized.
- Input dtype, layout, validity, arity, and input/output non-overlap are checked
  before the first destination write. Violations return `StridedError`.
- Replay keeps the caller-provided `ExecContext` and does not allocate for
  ranks up to `RAW_FUSED_RANK_LIMIT`.

## Implementation

`CopyPlan` writes through `RawStridedMut<MaybeUninit<T>>` for slice, reverse,
and concatenate. Pad writes the fill scalar to the complete logical
destination first, then copies input values through the existing contiguous
axis-0 or general traversal. No typed mutable reference to uninitialized `T`
storage is formed.

## Verification

Focused differential tests cover all `KernelDType` values, `bool`, complex
values, empty layouts, non-contiguous pad output, multi-input concatenate,
rank above the inline traversal limit, and overlap rejection before mutation.
The allocation-counting test covers all four erased full-overwrite entry
points.
