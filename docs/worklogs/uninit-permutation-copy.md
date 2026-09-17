# Uninitialized permutation copy

2026-09-17; AMD EPYC7713P, explicit 1T instruction experiments.

Canonical packing currently uses generic map replay even for identity copies.
Reuse the existing permutation-copy engine for one-thread identity copies into
MaybeUninit destinations. Restrict permutation dispatch to exact f32/f64 types:
the existing engine reinterprets 4/8-byte storage as native floats, which is not
appropriate for arbitrary Copy types with padding or weaker alignment. Other
types retain generic map. Introduce one small strided-basic API, re-exported by
strided-kernel: copy_into_uninit. It borrows initialized source storage as
MaybeUninit<T> (same layout), never interprets unwritten destination as T, and
uses the existing copy engine without a new traversal or pool. Preserve generic
map for zero-sized types and multi-thread workloads selected by the existing
bounded execution policy; do not replace parallel copying with serial HPTT.
The initial kernel-only instruction probe found a contiguous-copy regression
(~28%) despite reductions for permutations. Preserve the existing contiguous
map path too; remeasure the corrected implementation before acceptance.
Validate shape, destination injectivity and element-count arithmetic first.
Conjugating tenferro materialization keeps its current map implementation.

Require tests for compact/transposed/permuted/negative/broadcast/offset layouts,
empty and scalar cases, destination holes, rejected overlapping destinations,
real/complex bit preservation, and sequential/bounded execution policies. Compare
kernel-only instruction costs with the previous map on representative rank/size
and fallback cases before adoption. Then require LM eager instruction reduction
>=5%, with multiply/GEMM controls no worse than 1%, and quiet-host native
validation before final performance acceptance. Publication/pin changes were
initially deferred during the experiment.

## Integration decision

The maintainer subsequently requested PRs and merging with the timing limitation
explicitly disclosed. The completed single eager instruction pair reduced LM Ir
by11.897%; additional repetitions were cancelled at the maintainer's request.
Native measurements on an initially idle L3 domain were inconclusive after
competing jobs migrated into it. Do not claim the original performance acceptance
gate passed or advertise a native speedup. Integrate this tested copy route with
that known uncertainty; no GEMV diagnostic implementation is included.
Evidence is retained in the benchmark repositories (`47f7c2e` kernel experiment,
`b127b11` eager/native experiments). The PR gate passed workspace tests, formatting
and deterministic repository-rule review.
