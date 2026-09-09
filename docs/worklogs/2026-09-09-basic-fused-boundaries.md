# Basic/fused compilation boundaries on the consumer's current pin

## Baseline and scope

The preservation baseline is strided
`b40cd2f6d83c35ca23b24a8fb371ca061495729c`, the dependency resolved by tenferro
main `a096f280ab0dd774f92f533aa0f38b663a4c94d3`. At the start of this reapplication,
remote strided main remained `dc0a8e0`, while the consumer used the newer
`umbrella/issue-burndown-2026-08` head. The branch name alone was therefore not a
sufficient baseline check.

This branch, `build/basic-fused-boundaries-current`, starts at the exact pin.
The earlier `aca1aca` split on dc0a8e0 is retained separately but is not the
production candidate: it omitted ten commits, and two pinned plan-validation
tests failed against it. Its passing tests did not establish consumer parity.

The static-sharing tenferro change remains separately at `644e497`. It still
resolves b40cd2f and has not yet been connected to this split.

## Design and preservation

See [CPU kernel boundaries](../design/cpu-kernel-boundaries.md) for ownership.
Generic typed implementations, indexing plans and private initialization
receipts stay in basic; concrete ordinary and fused instantiations are separate.
Application entries remain checked. Prevalidated cross-crate entries are
explicit unsafe adapters with local caller proofs, not safe public proof tokens.
No generic callback erasure, independent pool, hidden copy or replacement kernel
was introduced for the partition.

The old boundary patch was reconciled with current source, not used to replace
current algorithms. In particular:

- gather/scatter rank-one fast paths, generic incremental gather/scatter,
  dynamic slice/update window replay, checked reset/span arithmetic, generic
  internal scatter callbacks, incremental axis reduction, pad cursors and the
  rank-bounded integer zero preflight are retained;
- all 27 named tests previously absent from the old candidate are present;
  existing expanded tests, benchmark cases, rules and provenance records are
  retained too;
- production fn/impl/macro token streams in moved gather_plan,
  static_indexing_plan and copy_plan match b40cd2f after normalizing only
  documentation, intended visibility and crate paths;
- erased code was partitioned from b40cd2f by dependency closure, with no
  unassigned production items. Existing cross-crate adapters were merged
  per item. The one reduction merge conflict retained both the current span
  validation and the new owner's layout-validator path;
- unit tests moved with the complete current implementation. Source-contract
  tests changed their file locations, not their assertions;
- package notices, normal dependency separation and parallel/simd forwarding
  remain part of the cut. The pin's strengthened review rules and expanded
  erased-policy benchmark were preserved byte-for-byte.

Read: both repository rule sets, shared common/Rust performance and numerical
rules, current erased-policy design, exact pin diffs, owning implementation
sources and the affected tests. This is source movement within the same project;
upstream Julia license notices remain packaged with their new owners.

## Verification

All Rust wrappers disabled; Cargo jobs=4. Focused gather/indexed tests used
`RAYON_NUM_THREADS=1`. Full parallel-correctness suites used an explicit ambient
four-worker pool and one test-harness thread; individual tests also exercise
serial, capped and nested policies. These runs are not timing benchmarks.

- Gather/indexed-write tests: **24 passed**, including both previously failing
  overflow-validation tests (and their later dynamic-update assertion).
- `cargo test --workspace -- --test-threads=1`: **947 passed, 9 existing ignored**.
- Basic/kernel/fused with all features: **571 passed**.
- The same packages without default features: **495 passed**.
- Release ordinary gather/indexed/policy/one-shot suites: **48 passed**.
- Release axis-reduction suite: **33 passed**.
- Release erased-fusion suite: **16 passed**.
- All-targets/all-features check for the three owners, formatting and staged
  diff whitespace checks passed. Deterministic repository review passed, and
  its own 83 tests passed; no independent LLM review was requested or run.
- Resolved normal dependency metadata confirms no ordinary/fused cross-edge.
  Basic's no-default normal graph contains neither Rayon nor pulp.

Logs remain at `/tmp/strided-current-*.log`; failed import-conflict attempts are
retained separately from successful runs. Reapplication/inventory material is
under `/tmp/strided-current-reapply/`. No fallback implementation was added to
make a failing test pass.

## Remaining integration

This verifies the strided partition, not the completed tenferro deliverable.
Tenferro basic/fused resource assembly, required read-into backend migration,
normal automatic fusion using the existing context/pool, final coverage and
local gate, and the actual combined dependency graph remain to be completed.
Build/runtime evidence must use the final fusion-preserving composition under
an explicit protocol; old experimental timings do not certify this branch.

No push, PR, merge or publication is part of this checkpoint. Any eventual
main-targeted PR must account for the prerequisite umbrella history rather than
silently including it as new boundary work. New-package publication remains a
separate maintainer action.
