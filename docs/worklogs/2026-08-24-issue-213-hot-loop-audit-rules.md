# Issue 213 hot-loop audit rules

## Task and evidence

This task strengthens review guardrails before the remaining #213 execution
work. The structural audit at `39111bd7` found generic gather, dynamic
slice/update, pad fallback, and erased axis-reduction loops that perform
rank-length coordinate or checked-offset reconstruction per logical element.
PR #236 fixed the measured compact rank-one gather/scatter path, but existing
benchmarks select fast paths and do not exercise the residual generic loops.

Reviewed inputs:

- `REPOSITORY_RULES.md`, especially `Unsafe And Fast-Path Boundaries`, `Layout
  And Copy Semantics`, and `Performance And Benchmark Discipline`
- `scripts/repository-rules-review.py` routing and its tests
- `ai/prompts/repository-rules-review.md`
- issues #213 and #237-#240
- `docs/design/erased-execution-policy.md`

Selected reviewer: read-only `reviewer-flash`, high thinking, for both this
design and the exact final diff.

## Design

Tighten existing rule sections rather than adding a new policy section:

1. `Layout And Copy Semantics` will prohibit avoidable full-coordinate rebuilds
   and rank-scanning checked-offset helpers inside loops whose trip counts scale
   with tensor elements, windows, or reduced elements after plan validation.
   One decode per worker range is allowed; execution should then advance
   coordinates and source/destination offsets incrementally. Data-dependent
   index reads remain allowed, while static layout mapping remains subject to
   the rule. A deliberate exception needs a concrete nearby `// INVARIANT:`
   rationale and benchmark evidence.
2. `Performance And Benchmark Discipline` will require specialized-fast-path
   work to benchmark at least one case that intentionally misses the fast path
   and representative rank scaling, or explicitly scope the claim and link a
   residual generic-path issue.
3. Kernel/permutation source routing will include the performance section, and
   added suspicious offset/decode terms will route both layout and performance
   sections.
4. The review prompt will require reviewers to distinguish per-element work
   from compile/setup and one-time worker-range decoding, identify nested loop
   multiplicity, and check generic-fallback benchmark coverage.
5. Deterministic routing/prompt tests will prove the new rules reach the
   reviewer for relevant source diffs.

## Rejected alternatives

- No new AST parser, dependency, or fail-closed lexical lint. The same tokens
  appear legitimately in plan-time injectivity checks, offset-table
  construction, and worker-range initialization; a lexical blocker would create
  false positives or a large baseline ledger.
- No standalone rule section. The defect is already governed by layout and
  benchmark policy; tighter language and routing are simpler.
- No production-loop change in this PR. Each operation family keeps its own
  benchmark-first implementation and review gate.

## Verification plan

- `python3 scripts/test-repository-rules-review.py`
- repository formatting/documentation checks required for a rules-only change
- targeted assertions that kernel paths and suspicious added content route both
  `Layout And Copy Semantics` and `Performance And Benchmark Discipline`
- targeted assertion that the prompt names loop multiplicity, range-start
  decode, and generic-fallback coverage
- final deterministic repository-rules review and hosted CI

## Gate status

Implementation must not start until `reviewer-flash` records a
Correct-to-merge verdict for this document. Final verification and review
results will be appended after implementation.
