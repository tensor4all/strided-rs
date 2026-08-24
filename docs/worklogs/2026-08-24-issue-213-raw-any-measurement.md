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
- all divisor fixtures are nonzero so every scan visits the complete domain;
- benchmark-local `current_scan` duplicates the existing algorithm exactly;
- benchmark-local `incremental_scan` decodes once and advances checked prepared steps/carries, solely to measure optimization headroom;
- compact rank-1 and rank-8 public controls measure serial and `max_threads(4)` `erased_zip_into` divide and add, quantifying preflight share without timing descriptor construction or allocation.

Criterion settings remain 10 samples, 300 ms warmup, and 1 s measurement. Run groups sequentially on pinned idle cores in one L3 domain.

## Need-before-implementation gate

Create a focused child implementation issue only if all apply at `N=2^18`:

1. `current_scan` costs at least 0.5 ms in one generic rank/layout case, or accounts for at least 15% of matching public divide time;
2. `incremental_scan` is at least 1.5x faster for generic rank 2 and at least 2x faster for rank 4 or 8, with non-overlapping confidence intervals;
3. negative/non-unit controls show the same direction and no semantic or allocation contract needs to change.

If the gate fails, record the evidence on #213 and do not change production. If it passes, create one child issue and a separate implementation design. The implementation must preserve early exit and exact validation-before-write semantics, stay private, allocate nothing through rank 8, retain checked compile/setup arithmetic, and add compact/generic/negative/non-unit/rank-zero/zero-extent/early-zero/late-zero correctness tests plus exact post-diff review.

## Review gate

Selected reviewer: read-only `reviewer-flash`, high thinking. Review this frozen measurement design before benchmark implementation. A later production change, if justified, requires its own reviewed design and exact-final-diff verdict.
