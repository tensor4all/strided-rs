# Terminal pointwise pointer advances (2026-10-06)

A gpt-6.1-sol review of the tprims fresh-output prerequisite found a pre-existing
strided defect: generic pointwise loops used `offset` after the final access.
With a negative stride the unused final pointer can precede its allocation;
with positive gaps it can exceed the permitted one-past position. Not
subsequently dereferencing that pointer does not make `offset` valid.

The owning fix changes only cursor advances to `wrapping_offset` in
`map_view.rs`, `ops_view.rs`, and `update_view.rs`. Validated traversal still
proves every actual access in bounds. Unary through quaternary maps,
initialized/uninitialized multiplication, broadcast partial-contiguous leaves,
add/mul/axpy/fma/dot, and generic update siblings share the correction. No
padding, fill, numerical traversal fork, public API, dependency or thread-policy
change is introduced.

`terminal_pointer_advances.rs` exercises minimally backed reversed/gapped
inputs and outputs, both broadcast operand positions, initialized/fresh
multiplication and update siblings. It checks values and is suitable for a
future focused Miri run. Numeric success alone is not evidence that invalid
pointer arithmetic was absent before the correction.

Local checks passed: focused strided-basic regression, workspace tests and
rustdoc tests, formatting, deterministic repository-rules review, and its 83
script tests. Test environment used `RAYON_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`; these are test settings, not a
measured performance baseline. No speed claim. `cargo miri --version` reports
Miri unavailable on the installed stable toolchain; no toolchain was installed
and no interpreter/red-green safety result is claimed.

Source changes are local corrections of this repository's existing
Strided.jl-derived leaves. Existing notices remain; no third-party bodies were
copied. This commit is a prerequisite for tprims' fresh-output boundary and
cpueinsum-rs#4; downstream pins must refer to the corrected revision.
