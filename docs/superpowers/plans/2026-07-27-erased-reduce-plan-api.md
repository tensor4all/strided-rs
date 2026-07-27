## Erased Reduce-Plan API

Issue: tensor4all/strided-rs#149

Goal: add the next family-sized Rust erased boundary after copy and map/zip by
exposing full-reduction replay through concrete dtype dispatch owned by
`strided-kernel`.

Scope:

- Add a small `ReduceOp` vocabulary for full reductions with unambiguous
  identities: `Sum` and `Product`.
- Add `ErasedReducePlan` for full-tensor reductions into a scalar output
  descriptor.
- Support the numeric dtype set whose `Sum` and `Product` semantics are already
  clear at this boundary: `f32`, `f64`, `i32`, `i64`, `c32`, and `c64`.
- Keep output dtype equal to input dtype. Dtype promotion remains a tensor
  runtime concern.
- Keep `ExecContext` on execute signatures. `Ambient` preserves the existing
  view-based behavior; non-ambient contexts use a serial reduce helper so this
  erased boundary does not read ambient Rayon state.
- Validate dtype equality, source layout identity, and scalar output shape
  before writing.

Out of scope:

- C ABI symbols.
- Axis and multi-axis reductions.
- `Maximum`, `Minimum`, boolean, comparison, and dtype-promotion semantics.
- New performance tuning beyond build-isolating the dtype dispatch.
- tenferro dependency migration.

Verification:

- Add integration tests for f64 sum on a transposed layout, c64 product, i32
  ambient sum, the remaining supported dtype dispatch arms, empty-input identity
  behavior, dtype/layout mismatch rejection before writing, scalar output shape
  validation, unsupported bool dtype rejection, and compile-time layout
  validation.
