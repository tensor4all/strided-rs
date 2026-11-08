# Dense CPU kernel ownership

2026-09-17; AMD EPYC7713P, explicit 1T diagnostic measurements.

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

## Optimization candidate

Migration baseline: `ec585b8a4bf0af96863a6136f0b1f8e9c1aeadba`.
The separate benchmark suite has a `dense_kernels` binary with nonzero
references, explicit `Sequential` policy, an excluded warmup, and the
`profile_dense` Callgrind collection boundary. Setup, view construction,
AXPBY state restoration and verification are outside that boundary.

- Triangular masks now copy only kept intervals and fill masked intervals;
  every output slot is written exactly once.
- Contiguous multiply uses pulp's ordinary full-vector slice operations,
  including its MaybeUninit output split. Partial accesses remain only for
  the tail. This stays in the existing strided SIMD implementation, including
  the existing exact complex multiply operation.
- New tests cover all lengths 0..129, unaligned inputs/outputs, sentinels,
  real/complex types and strided fallback; floating special values are checked.

Strict Clippy under Rust 1.97.1 is currently blocked by pre-existing lints in
unchanged strided-view and strided-basic code (including `uninit_vec` in
strided-view/src/view.rs). `--no-deps` still finds 33 existing basic-crate
errors; none are in the changed dense_update.rs or simd.rs code. This change
neither silences those lints nor claims a passing strict-lint gate.
