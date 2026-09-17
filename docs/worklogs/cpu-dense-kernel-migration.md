# Dense CPU kernel ownership

Tenferro's pinned strided branch is `umbrella/issue-burndown-2026-08` at
`fbd10fa5b70bb462b961cfd9e02faadb6bb95be0`; this change intentionally builds
on that family split, not the older main branch. The user requested moving
reusable CPU kernels out of tenferro and optimizing through the separate
strided benchmark suite.

## Design

- Move AXPBY, triangular mask and diagonal embedding bulk work to strided-basic;
  strided-kernel's existing reexport supplies the consumer API.
- The three small public APIs explicitly accept dense contiguous slices. Shape
  arguments use column-major order (matrix rows/columns are leading axes).
  This matches tenferro's current compact input contract; no hidden packing or
  general-layout fallback is introduced.
- AXPBY reads its initialized destination and keeps existing arithmetic order,
  including NaNs when either coefficient is zero. Structural operations accept
  MaybeUninit destinations and fully initialize output without reading it.
- Keep allocation, placement, dtype dispatch, error translation and execution
  entry in tenferro. Reuse strided's bounded execution policy and threshold.
- Use existing strided traversal for diagonal copies. Off-diagonal zero values
  are required output, not unnecessary scratch initialization.
- Record a migration baseline before optimizing; use the existing standalone
  benchmark suite, explicit 1T/pinning, and nonzero correctness references.
  Host contention blocks elapsed-time claims, not isolated instruction work.

## Acceptance

Representative sizes/ranks, empty and odd dimensions, all diagonal axis
positions, extreme triangular offsets, complex AXPBY, invalid metadata and
bounded parallel execution must pass. Downstream CPU tests must pass against
the migrated implementation. Any claimed optimization requires a comparison
against the recorded baseline; instruction counts are not timing speedups.
