# Issue #428: the view path adopts the `slabs_share_no_line` decline

**Question ([#428](https://github.com/tensor4all/strided-rs/issues/428)).**
[#422](https://github.com/tensor4all/strided-rs/pull/422) gave plan and replay an
exit, `slabs_share_no_line`: when the tiles' kept leading slabs are disjoint,
line-multiple and nested, no visit order can reuse a line, so the flat walk is
kept. The view path (`copy_into`, `map_into` through `build_plan_fused` →
`compute_block_sizes`) still followed Strided.jl's `_computeblocks` and tiled
the same layouts, so one block model gave two answers for one layout. Adopt the
decline there, or keep the port faithful?

**Decision: adopt.** One predicate, `block::declines_tiles`, now answers both
callers; `slabs_share_no_line` moved next to it in `block.rs` instead of being
called twice inside `raw_ops::tile_blocks`. The acceptance and guard rows from
the issue are measured below.

## Routing, asserted

`block::tests::the_view_path_keeps_the_flat_walk_where_the_tiles_share_no_cache_line`
(tensor4all/strided-rs#428) checks the view path's own planner,
`kernel::pointwise_plan`, which is what `copy_into` and `map_into` call:

| fixture | model (`reduced_block_sizes`) | view path's block |
|---|---|---|
| F3 `[4096,17,33]`, dst `[1,4096,69632]`, src `[1,135168,4096]` | `[4096,16,16]` | `[4096,17,33]` (declined) |
| broadcast `[64,64,64]`, dst `[1,64,4096]`, src `[0,1,1]` | `[64,31,16]` | `[64,31,16]` (kept) |
| the bench's parity rows `[1024,1024]` transpose and `[512,512]` broadcast | model | model (kept) |

Red before the wiring: `assertion left == right failed … left: [4096, 16, 16],
right: [4096, 17, 33] … test result: FAILED. 0 passed; 1 failed; 259 filtered
out`. Green after: `test result: ok. 1 passed; 0 failed; 259 filtered out`. Both
reproduced at `6b8f3d6` with the decline call removed from `compute_block_sizes`
and then restored — one log, SHA in its header, tree clean before and after
(`/tmp/6b8f3d6…-view-decline-redgreen.log`); the gate logs are
`/tmp/6b8f3d6…-workspace-tests.log` (78 `test result:` lines, 1158 passed, 0
failed, 1158 individual `test … ok` lines, rc 0) and
`/tmp/6b8f3d6…-clippy-fmt.log`.
 `compute_block_sizes` is the view path's entry, so `reduced_block_sizes`
stays the pure model and the two sweep tests that compare it to the exact model
are untouched; `raw_ops::tests::tiled_tests` now reads the model's F3 answer
through `reduced_block_sizes` and asserts the view path's decline beside it.

## Measurement

Machine `evo_vpn` (AMD Ryzen AI 9 HX 470, 12C/24T, one L3 domain in this
container: CPUs 4-11 and 16-23). Measured cell: `taskset -c 6`, single-threaded
(`Exec::serial()`), so the thread team cannot move. Observer (this agent) pinned
away from CPU 6 (`taskset -apc 4,5,7-11,16-23 <pid>`), though it shares the one
available L3 domain — unavoidable here and stated rather than glossed. Idle
before each run: `vmstat` `id` 99-100 %, `r` 0-2. No `pinned.sh`, `aocl-bench` or
`cpu-weekend` process. Builds and tests capped at `-j8`.

`strided-basic/benches/view_decline_428.rs`,
`cargo bench -p strided-basic --features bench-hooks --bench view_decline_428`:
A is the decline (flat walk), B is the model's tiles, selected by a relaxed
atomic in `bench_hooks::DECLINE_TILES` so **both answers are timed in one
process**. The bench prints the two block vectors for F3 first, which is what
makes the two arms a routing difference and not a hope:

```text
F3 blocks, decline on : [4096, 17, 33]
F3 blocks, decline off: [4096, 16, 16]
```

Each row: warmup once per arm, then 9 samples per arm per attempt, 3 attempts,
**interleaved A/B within each attempt**, and the reported ratio is the median of
the 9 paired ratios with its MAD. Both arms write the *same* destination buffer
(the view is rebuilt per call, at equal cost on both sides) — an earlier version
gave each arm its own buffer and reported a 1.125 ratio on the broadcast row
that came from the two allocations' pages, not from routing.

Three consecutive runs at `6b8f3d6`, logs `/tmp/6b8f3d6…-bench-run{1,2,3}.log`
(SHA in the header of each):

| row | case | ratio run 1 / 2 / 3 | MAD of those ratios | A flat ns/iter (run 1) | B tiled ns/iter (run 1) |
|---|---|---|---|---|---|
| F3c | F3 `[4096,17,33]` `copy_into` | 0.998 / 1.000 / 0.999 | 0.0066 / 0.0120 / 0.0068 | 1 143 705 | 1 147 097 |
| F3m | F3 `[4096,17,33]` `map_into` | 1.003 / 1.000 / 0.995 | 0.0108 / 0.0045 / 0.0017 | 1 179 742 | 1 172 348 |
| Pc | contiguous `[2048,512]` (guard) | 1.004 / 1.005 / 1.010 | 0.0057 / 0.0068 / 0.0050 | 351 102 | 349 158 |
| Pt | transposed `[1024,1024]` (guard) | 1.000 / 1.001 / 1.001 | 0.0011 / 0.0013 / 0.0042 | 4 051 358 | 4 050 740 |
| Pb | broadcast `[512,512]` (guard) | 1.002 / 1.002 / 1.001 | 0.0021 / 0.0008 / 0.0032 | 342 217 | 341 710 |

p95 per arm sits 0.5-5 % above its median and is in the logs. The three guard
rows keep their block vector unchanged *by construction* (asserted in the test
above), so the decline cannot move them at all; their ratios are a measure of
this harness's own noise floor, not of a difference.

What the numbers do and do not say:

- **Time, not traffic.** `2 × 18 382 848` logical bytes in 1.14 ms is an
  *effective logical bandwidth* of ~32 GB/s, i.e. the bytes the mapping moves
  divided by the measured time. No traffic counter was read: this host denies
  PMU access (`/proc/sys/kernel/perf_event_paranoid` is `4`; `perf stat -e
  cache-misses true` answers `Access to performance monitoring and
  observability operations is limited`). The claim is therefore "the measured
  time is the same for both walks on F3", which is what the acceptance needs,
  and not "the traffic is the same".
- **Order effect.** A is sampled before B within every rep, and the
  second-position advantage shows up as a consistent `ratio > 1` on the
  contiguous row (1.004-1.010). Nothing else differs between the arms there, so
  that is the size of the effect: about 1 %, which is the same order as the F3
  rows' deviation from 1. Those rows straddle it (0.995-1.003), so the decline
  is inside the harness's order effect rather than measurably better or worse.
- **L3 sharing.** Only one L3 domain is visible inside this container (CPUs
  4-11 and 16-23), so pinning the observer away from the measured CPU 6 still
  leaves it on the same last-level cache. That cannot be fixed from user space
  here; it is why the acceptance is stated as "inside this harness's dispersion
  and order effect" rather than as an absolute ±X %.

## What this does not claim

- No Strided.jl cross-check was run; the guard rows are the repository's own
  parity fixtures (contiguous, transposed, broadcast), not Julia timings.
- F3 is the only declined layout measured. A layout where the decline is a clear
  win on the view path was not found and is not claimed.
- No memory-traffic, cache-miss or instruction counter underlies the F3 result;
  the PMU is inaccessible here (see above).
- The measurement scaffolding is a `bench-hooks` feature (`DECLINE_TILES`,
  `planned_blocks`), off in every normal build. If the hook itself is unwanted,
  dropping it and the bench keeps the routing change and its test intact.
