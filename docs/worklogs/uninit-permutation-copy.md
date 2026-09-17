# Uninitialized permutation copy

Canonical packing currently uses generic map replay even for identity copies.
Reuse the existing permutation-copy engine for one-thread identity copies into
MaybeUninit destinations. Introduce one small strided-basic API, re-exported by
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
validation before final performance acceptance. No publication/pin changes.
