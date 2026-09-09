# CPU kernel compilation boundaries

This partition preserves the strided revision used by the current tenferro
consumer, including its incremental indexed/reduction kernels. See the
[implementation worklog](../worklogs/2026-09-09-basic-fused-boundaries.md) for the
exact baseline, preservation checks and remaining cross-repository work.

## Ownership

- `strided-view` owns typed and erased storage descriptors and their bounds,
  initialization and provenance contracts.
- `strided-basic` owns generic typed primitives, layout/planning/iteration,
  execution policy, SIMD support, and concrete copy/concatenation/reduction
  replay. Generic indexing and overwrite receipts stay with their validation.
- `strided-kernel` owns concrete dtype-erased ordinary arithmetic and indexed
  replay. It also provides the shared checked typed APIs for ordinary users.
- `strided-fused` owns the runtime-DAG interpreter, static fused specializations,
  and concrete dtype-erased fusion replay.

The two concrete operation owners depend on `strided-basic`, never on each
other. The basic package has no normal dependency on either concrete owner.
The complete `strided-rs` facade composes ordinary and fused capabilities.

This cut is about code-generation ownership, not forbidding typed arithmetic in
basic. Keeping generic code and its private helpers together avoids exposing
writer traits, initialization receipts, SIMD internals and every loop helper.
The expensive concrete instantiations remain independently compiled. Whether
this placement improves a particular build must be measured; generic source
placement alone does not centralize all downstream monomorphizations.

## Kernel-family execution contract

`strided_basic::execution` is the limited extension interface used by concrete
operation-family implementations. Implementation modules remain private.
Application APIs retain checked, safe entry points.

Prevalidated execution adapters are unsafe. Their callers establish the exact
shape/layout, allocation, overlap and initialization obligations before replay.
`ValidatedDestinationLayout` is only a local marker; possession of a marker for
another output is not a proof. Private generic helpers and initialization
receipts are not turned into safe public APIs to make imports compile.

Adapters forward the original generic callback types without erasure or a new
per-element dispatch. The initialized-input conversion uses the existing unsafe
`ErasedRawStridedPtr::try_as_ref_after_no_overlap` method directly after the
owning entry's overlap checks. No second validation pass is introduced merely
to cross a crate boundary. Checks retain their original order relative to
allocation and writes.

## Resources and features

There is one execution-policy implementation in basic. Fused execution uses the
same installed context, threshold and bounded worker policy; it does not create
an independent pool. `parallel` and `simd` features propagate from each concrete
owner to basic. No operation-family feature switches are added.

An assembling tenferro CPU backend must keep its ordinary automatic-fusion hook
and delegate using its existing context and buffers. This strided split does not
by itself complete that tenferro integration.

## Verification and publication

Tests move to their implementation owner, and cross-owner tests explicitly
import both packages. Source-contract checks inspect all three source owners,
including the private overwrite receipt and centralized execution policy.
Prevalidated adapters have executable examples; a compile-fail example ensures
that a layout marker cannot make an unchecked call safe.

New packages require explicit maintainer release handling. No package is
published by this change. Normal dependency order is traits, view, permutation,
basic, then ordinary/fused, followed by their frontend dependents; the full
publication graph must be checked at release time.
