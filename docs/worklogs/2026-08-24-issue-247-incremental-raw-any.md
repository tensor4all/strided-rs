# Issue #247: incremental integer zero preflight

## Scope and evidence

Issue #247 is the production follow-up to the measurement gate in `2026-08-24-issue-213-raw-any-measurement.md`. At `N=2^18`, current rank-2/4/8 scans cost 0.879/1.792/3.682 ms and the benchmark-local incremental probe showed 2.56/4.29/7.55x headroom. The scan is private and is called only before integer divide/remainder replay.

Base: measurement branch commit `57ac6bf`; upstream production base `f875cc894d72187416367c26825c5cb7fca726c2`.

## Minimal implementation

Keep `raw_any` private and its signature unchanged.

1. Compute checked total length exactly as today. Return `false` immediately for a zero-sized domain.
2. For rank at most `RAW_FUSED_RANK_LIMIT` (8), use a fixed `[usize; RAW_FUSED_RANK_LIMIT]` coordinate array. Precompute each checked carry reset into a fixed `[isize; RAW_FUSED_RANK_LIMIT]` array once. Start at `src.offset()`, test the current element, and advance axis-0-fastest with checked step/reset additions. This allocates nothing and preserves early exit and visit order.
3. For rank above 8, retain the existing flat-to-multi-index checked fallback unchanged. Do not introduce a heap cursor or shared cursor abstraction for this one private one-shot scan.

Raw descriptor construction already validates every reachable offset. Return on zero total before computing `dim - 1`. On carry, add the prepared `-(dim - 1) * stride` reset directly; do not transiently step one-past-end and subtract `dim * stride`. Checked reset preparation plus checked cursor additions preserve the existing overflow error surface without unchecked pointer arithmetic beyond the already validated final offset dereference.

No public API, dtype dispatch, map replay, integer arithmetic, threading policy, or validation ordering changes. The scan stays serial; only the following zip replay may use the requested execution context.

## Correctness and allocation

Add focused tests through the public `erased_zip_into` boundary:

- rank 0 nonzero succeeds and rank 0 zero rejects before destination mutation;
- zero extent succeeds without reading an intentionally degenerate source;
- compact rank 1/2/4/8 late zero rejects before mutation;
- rank-2 negative stride/nonzero offset and non-unit holes find reachable zeros but ignore hole zeros;
- rank 9 exercises the unchanged fallback;
- both integer dtypes and both divide/remainder operations retain wrapping/nonzero behavior and zero rejection;
- allocation counting proves successful and rejecting rank-8 preflight allocates zero times after descriptor construction. Use compact serial replay for the successful case so the counted window does not include unrelated parallel/map allocations.

Prefer extending existing `erased_one_shot.rs` and `copy_plan_alloc.rs`; no new test harness.

## Frozen implementation benchmark

Before production edits, extend only the public-control group with compact rank 2/4 and rank-2 negative/non-unit layouts, then run the complete public group as the implementation baseline. Keep the already measured compact rank-1/rank-8 cells unchanged. Candidate reruns the identical group on the same pinned idle L3-domain cores with setup outside timing.

For negative/non-unit source variants, keep the destination compact and use the strided layout only for the divisor source; a negative-stride mutable destination would be invalid. At `N=2^18`, require:

- compact rank 2 improves at least 1.3x and compact rank 4 or 8 improves at least 2x with non-overlapping intervals;
- negative/non-unit divide improve in the same direction;
- compact rank-1 divide does not regress by more than 10%;
- add controls do not regress by more than 10%;
- divide/remainder correctness remains exact. Remainder need not duplicate every Criterion cell because it shares the same preflight branch and is covered by tests.

Record rank/layout/serial/four-thread baseline and candidate evidence. The benchmark-local current/incremental probe remains frozen measurement evidence and is not itself a production candidate benchmark.

## Verification and review gates

Run focused default/parallel tests, allocation tests, default/parallel workspace tests, coverage for the modified production file, docs, formatting, and repository-rules review. Selected reviewer is read-only `reviewer-flash` with high thinking. Pre-implementation review of `a2dd71c0` by read-only `reviewer-flash` (high) returned **Correct-to-merge** with three nonblocking benchmark/allocation/carry-construction cautions, incorporated above. The exact final diff requires a second `Correct-to-merge` verdict before PR creation.

## Candidate evidence

Production commit `b71ea74` implements the reviewed fixed-array cursor and unchanged rank-above-8 fallback. The accepted full paired run used separate Cargo target directories (to prevent cross-worktree artifact reuse) and CPUs 9-12 in L3 domain 8-15 after a valid gate (selected average 1.7%, domain-other maximum 0.3%); baseline and candidate ran sequentially with the frozen environment.

| layout | context | size | baseline divide | candidate divide | speedup | interval-bound speedup |
|---|---|---:|---:|---:|---:|---:|
| compact_rank1 | serial | 262144 | 0.9081 ms | 0.8913 ms | 1.02x | 0.98-1.06x |
| compact_rank2 | serial | 262144 | 1.3159 ms | 0.9894 ms | 1.33x | 1.29-1.38x |
| compact_rank4 | serial | 262144 | 2.2543 ms | 1.2382 ms | 1.82x | 1.74-1.88x |
| compact_rank8 | serial | 262144 | 4.1025 ms | 1.2673 ms | 3.24x | 3.16-3.35x |
| rank2_negative | serial | 262144 | 2.3155 ms | 1.7604 ms | 1.32x | 1.23-1.38x |
| rank2_nonunit | serial | 262144 | 1.3884 ms | 0.8847 ms | 1.57x | 1.46-1.69x |
| compact_rank1 | max_threads_4 | 262144 | 0.7408 ms | 0.6286 ms | 1.18x | 1.12-1.23x |
| compact_rank2 | max_threads_4 | 262144 | 1.2670 ms | 0.6901 ms | 1.84x | 1.80-1.87x |
| compact_rank4 | max_threads_4 | 262144 | 2.2491 ms | 0.9926 ms | 2.27x | 2.25-2.29x |
| compact_rank8 | max_threads_4 | 262144 | 4.3492 ms | 1.1466 ms | 3.79x | 3.72-3.86x |
| rank2_negative | max_threads_4 | 262144 | 1.4965 ms | 0.9391 ms | 1.59x | 1.57-1.62x |
| rank2_nonunit | max_threads_4 | 262144 | 1.2309 ms | 0.7190 ms | 1.71x | 1.68-1.77x |
| compact_rank1 | serial | 1048576 | 4.1410 ms | 3.7445 ms | 1.11x | 1.07-1.12x |
| compact_rank2 | serial | 1048576 | 5.4750 ms | 4.0994 ms | 1.34x | 1.31-1.35x |
| compact_rank4 | serial | 1048576 | 9.0235 ms | 5.0443 ms | 1.79x | 1.70-1.90x |
| compact_rank8 | serial | 1048576 | 16.8670 ms | 5.2414 ms | 3.22x | 3.07-3.41x |
| rank2_negative | serial | 1048576 | 9.8723 ms | 6.9330 ms | 1.42x | 1.39-1.46x |
| rank2_nonunit | serial | 1048576 | 5.9148 ms | 3.7228 ms | 1.59x | 1.52-1.67x |
| compact_rank1 | max_threads_4 | 1048576 | 2.9193 ms | 2.4148 ms | 1.21x | 1.20-1.22x |
| compact_rank2 | max_threads_4 | 1048576 | 4.7288 ms | 2.7797 ms | 1.70x | 1.66-1.73x |
| compact_rank4 | max_threads_4 | 1048576 | 8.4082 ms | 3.8841 ms | 2.16x | 2.13-2.20x |
| compact_rank8 | max_threads_4 | 1048576 | 16.4930 ms | 4.5294 ms | 3.64x | 3.58-3.71x |
| rank2_negative | max_threads_4 | 1048576 | 5.9997 ms | 3.7909 ms | 1.58x | 1.57-1.59x |
| rank2_nonunit | max_threads_4 | 1048576 | 4.7795 ms | 2.7767 ms | 1.72x | 1.71-1.73x |

At medium size, serial compact rank 2/4/8 improved 1.33/1.82/3.24x; four-thread public calls improved 1.84/2.27/3.79x. Negative/non-unit layouts improved 1.32/1.57x serial and 1.59/1.71x with four threads. Rank-one divide was non-regressed (1.02x serial, 1.18x four-thread). Large results remained directionally consistent.

The first candidate command accidentally reused the baseline worktree binary through a shared Cargo target and was discarded before interpretation. The corrected run used distinct target directories. Because the full paired run showed impossible >10% drift in several unchanged serial Add cells, Add-only baseline/candidate groups were independently rerun on CPUs 41-44 after a second accepted gate; all medium/large Add estimates were within 5.4% and every frozen 10% control gate passed.

## Verification

- focused default and parallel one-shot/allocation tests: 14 passed each
- default workspace: 916 passed, 9 ignored
- parallel workspace: 990 passed, 9 ignored
- `cargo check -p strided-kernel --features parallel`: passed
- `cargo doc --workspace --no-deps`: passed
- formatting: passed
- deterministic repository-rules preview: passed
- modified `erased.rs` coverage: 88.34% (threshold 80%); the only global package failure remains the unchanged `reduce_view.rs` baseline deficit

Exact-final independent review is pending.
