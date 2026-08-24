# Issue #247: incremental integer zero preflight

## Scope and evidence

Issue #247 is the production follow-up to the measurement gate in `2026-08-24-issue-213-raw-any-measurement.md`. At `N=2^18`, current rank-2/4/8 scans cost 0.879/1.792/3.682 ms and the benchmark-local incremental probe showed 2.56/4.29/7.55x headroom. The scan is private and is called only before integer divide/remainder replay.

Base: measurement branch commit `57ac6bf`; upstream production base `f875cc894d72187416367c26825c5cb7fca726c2`.

## Minimal implementation

Keep `raw_any` private and its signature unchanged.

1. Compute checked total length exactly as today. Return `false` immediately for a zero-sized domain.
2. For rank at most `RAW_FUSED_RANK_LIMIT` (8), use a fixed `[usize; RAW_FUSED_RANK_LIMIT]` coordinate array. Precompute each checked carry reset into a fixed `[isize; RAW_FUSED_RANK_LIMIT]` array once. Start at `src.offset()`, test the current element, and advance axis-0-fastest with checked step/reset additions. This allocates nothing and preserves early exit and visit order.
3. For rank above 8, retain the existing flat-to-multi-index checked fallback unchanged. Do not introduce a heap cursor or shared cursor abstraction for this one private one-shot scan.

Raw descriptor construction already validates every reachable offset. Checked reset preparation plus checked cursor additions preserve the existing overflow error surface without unchecked pointer arithmetic beyond the already validated final offset dereference.

No public API, dtype dispatch, map replay, integer arithmetic, threading policy, or validation ordering changes. The scan stays serial; only the following zip replay may use the requested execution context.

## Correctness and allocation

Add focused tests through the public `erased_zip_into` boundary:

- rank 0 nonzero succeeds and rank 0 zero rejects before destination mutation;
- zero extent succeeds without reading an intentionally degenerate source;
- compact rank 1/2/4/8 late zero rejects before mutation;
- rank-2 negative stride/nonzero offset and non-unit holes find reachable zeros but ignore hole zeros;
- rank 9 exercises the unchanged fallback;
- both integer dtypes and both divide/remainder operations retain wrapping/nonzero behavior and zero rejection;
- allocation counting proves successful and rejecting rank-8 preflight allocates zero times after descriptor construction.

Prefer extending existing `erased_one_shot.rs` and `copy_plan_alloc.rs`; no new test harness.

## Frozen implementation benchmark

Before production edits, extend only the public-control group with compact rank 2/4 and rank-2 negative/non-unit layouts, then run the complete public group as the implementation baseline. Keep the already measured compact rank-1/rank-8 cells unchanged. Candidate reruns the identical group on the same pinned idle L3-domain cores with setup outside timing.

At `N=2^18`, require:

- compact rank 2 improves at least 1.3x and compact rank 4 or 8 improves at least 2x with non-overlapping intervals;
- negative/non-unit divide improve in the same direction;
- compact rank-1 divide does not regress by more than 10%;
- add controls do not regress by more than 10%;
- divide/remainder correctness remains exact. Remainder need not duplicate every Criterion cell because it shares the same preflight branch and is covered by tests.

Record rank/layout/serial/four-thread baseline and candidate evidence. The benchmark-local current/incremental probe remains frozen measurement evidence and is not itself a production candidate benchmark.

## Verification and review gates

Run focused default/parallel tests, allocation tests, default/parallel workspace tests, coverage for the modified production file, docs, formatting, and repository-rules review. Selected reviewer is read-only `reviewer-flash` with high thinking. Production implementation starts only after this design receives `Correct-to-merge`; the exact final diff requires a second `Correct-to-merge` verdict before PR creation.
