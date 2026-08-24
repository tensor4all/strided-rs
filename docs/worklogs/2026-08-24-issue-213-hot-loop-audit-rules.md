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
   The serial path may decode once per traversal; parallel or blocked traversal
   may decode once per worker range or traversal block, then must advance
   coordinates and source/destination offsets incrementally. Plan-time
   injectivity checks, offset-table construction, and worker-range/block
   initialization are explicitly outside the per-element prohibition.
   Data-dependent index reads remain allowed, while static layout mapping
   remains subject to the rule. A deliberate replay exception needs a concrete
   nearby `// INVARIANT:` rationale plus dated worklog/benchmark evidence, or a
   narrowly scoped performance claim with a linked residual issue.
2. `Performance And Benchmark Discipline` will require work that adds or
   specializes a fast path to benchmark at least one case that intentionally
   misses that fast path plus representative rank scaling. Other performance
   claims must benchmark representative rank scaling and the relevant fallback
   or layout cases, without inventing a nonexistent fast-path miss. A change may
   instead explicitly scope the claim and link a residual generic-path issue;
   routine production refactors without a performance claim are not required to
   add benchmarks solely because they touch kernel code.
3. Kernel/permutation source routing will include the performance section.
   Added content matching the concrete terms
   `checked_strided_offset`, `flat_to_multi_index`, `multi_index`,
   `advance_col_major_index`, or `fill_col_major_index` will route both layout
   and performance sections. Generic `offset` alone is intentionally
   excluded because it would route nearly every kernel diff.
4. The review prompt will require reviewers to distinguish per-element work
   from compile/setup and one-time worker-range/block decoding, identify nested
   loop multiplicity, and check generic-fallback benchmark coverage. It will
   explicitly scope benchmark-evidence findings to fast-path specialization,
   performance claims, or diffs containing/citing benchmark evidence. Missing
   benchmark evidence for another routine kernel refactor may be at most a
   warning, not a blocking finding.
5. Deterministic routing/prompt tests will prove the new rules reach the
   reviewer for relevant source diffs, enumerate the exact content triggers,
   and assert the prompt's benchmark-scoping and non-blocking guard.

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

`reviewer-flash` reviewed design commit `b0accd0` with high thinking and a
read-only tool boundary. Verdict: **Correct-to-merge**; implementation may
begin. The implementation will also fold in its three non-blocking refinements:

- anchor `multi_index` and related identifiers with word boundaries and drop
  generic `decode` from the content trigger;
- route the performance section for `strided-view` as well as kernel and
  permutation paths so fast-path work outside the current audit files is not a
  blind spot;
- test the compiled trigger behavior, not merely the prose list.

During parent integration, the first implementation exposed an ambiguity in
point 2: its original grammar applied the fast-path-miss requirement to every
performance claim, including operations with no fast path. The revised point 2
above separates fast-path work from other performance claims. `reviewer-flash`
reviewed design-delta commit `f9ac6ee` and returned **Correct-to-merge**, with
the condition that the revised grammar be propagated to both the rule and
prompt before final review. The parent applied that correction before resuming
verification.

Implementation was delegated to `luna-implementer` with write ownership limited
to the four rule/review files named in the design. The parent reviewed the full
diff and additionally changed the prompt-presence test to normalize whitespace
instead of depending on Markdown line wrapping.

Verification before the exact candidate commit:

- `python3 scripts/test-repository-rules-review.py`: 83 passed
- `python3 -m py_compile scripts/repository-rules-review.py scripts/test-repository-rules-review.py`
- `git diff --check`

Candidate verification also passed:

- `cargo fmt --all -- --check`
- `cargo test --workspace`: 899 passed, 9 ignored
- deterministic repository-rules review: pass, no findings
- exact-diff `reviewer-flash` review at `ee6d980`: **Correct-to-merge**, no
  Critical or Important findings

Hosted CI remains the final merge gate. The exact-diff review noted that a
standalone `multi_index` identifier in an
unrelated helper or comment can over-route that diff; this is accepted because
routing only supplies applicable rules to the reviewer, while the diff-scoped
prompt and warn cap prevent it from creating a violation by itself. Removing it
would add a blind spot for renamed coordinate-reconstruction helpers without
improving correctness.
