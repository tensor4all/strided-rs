## Erased Fused-Plan API

Issue: tensor4all/strided-rs#149

Goal: add the next family-sized Rust erased boundary after copy by exposing
single-output map/zip elementwise replay through concrete dtype dispatch owned
by `strided-kernel`.

Scope:

- Reuse the existing `FusedPlan` / `FusedOp` runtime op-code vocabulary.
- Add `ErasedFusedPlan` for one output and one to four inputs, matching the
  existing `map_into` / `zip_map{2,3,4}_into` ownership boundary.
- Dispatch only the dtype set already covered by `FusedScalar` (`f32`, `f64`,
  `c32`, `c64`); leave integer and bool elementwise semantics for a later
  explicit design.
- Keep `ExecContext` on execute signatures so each erased family carries the
  same threading boundary, even while current execution still uses the existing
  view-based fused implementation internally.
- Treat non-ambient `ExecContext` values conservatively as serial execution for
  now; owned-pool execution is a later threading implementation detail, not a
  reason to leak ambient Rayon into this boundary.

Out of scope:

- C ABI symbols.
- Multi-output fused replay.
- Fused-DAG ABI stabilization.
- Integer, bool, comparison, and logical op semantics.
- Reduction and indexed gather/scatter/dynamic_slice families.

Verification:

- Add integration tests for f64 zip-add on a transposed output layout, c64
  multiply, f32 ternary clamp through ambient execution, c32 unary conjugation,
  a four-input DAG, dtype/count mismatch rejection before writing, plan accessor
  exposure, invalid output contracts, and unsupported dtype / arity rejection.
