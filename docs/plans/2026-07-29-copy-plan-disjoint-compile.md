# CopyPlan disjoint-layout compile allocation

## Context

tenferro-rs #1512 exposed two allocations per `ErasedSlicePlan` compile for a
2x2 contiguous layout. The allocations came from `CopyPlan` destination
injectivity validation: the small-layout exact check built a `HashSet` and a
coordinate vector before consulting the existing disjoint-stride proof.

## Change

`is_injective_layout` now accepts layouts proven injective by
`has_disjoint_stride_spans` before the bounded exact enumeration, after first
proving that every positive and negative cumulative offset span is
representable as `isize`. Non-disjoint layouts at or below the exact-check
limit still use the same exact algorithm; larger non-disjoint layouts remain
rejected. The fast path therefore preserves the accepted valid layout set
while rejecting previously unsafe extreme-stride layouts before fused
traversal construction. Fusion adjacency also uses checked stride
multiplication, so representable mixed-sign layouts remain valid without
overflowing while testing whether adjacent axes can be fused.

The allocation-contract integration test now also requires a 2x2 contiguous
`CopyPlan::compile` to allocate zero times when the parallel feature provides
inline rank metadata.

## Provenance

The implementation and tests are original work based on profiling this
repository and its tenferro-rs consumer. No third-party source was copied.
