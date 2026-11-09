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

Red before the wiring: `left: [4096, 16, 16]`, `right: [4096, 17, 33]`. Green
after. `compute_block_sizes` is the view path's entry, so `reduced_block_sizes`
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
the 9 paired ratios (MAD alongside). Both arms write the *same* destination
buffer (the view is rebuilt per call, at equal cost on both sides) — an earlier
version gave each arm its own buffer and reported a 1.125 ratio on the broadcast
row that came from the two allocations' pages, not from routing.

Two consecutive runs, ratios A/B (`ns/iter` from the second run, median of 9):

| row | layout | A flat ns/iter | B tiled ns/iter | ratio run 1 | ratio run 2 |
|---|---|---:|---:|---:|---:|
| F3c `copy_into` | F3 `[4096,17,33]` | 1 192 718 | 1 191 210 | 1.001 | 1.005 |
| F3m `map_into` | F3 `[4096,17,33]` | 1 151 727 | 1 156 240 | 1.002 | 1.003 |
| Pc | contiguous `[2048,512]` | 375 217 | 372 759 | 0.998 | 1.014 |
| Pt | transposed `[1024,1024]` | 4 114 742 | 4 111 259 | 1.002 | 1.001 |
| Pb | broadcast `[512,512]` | 342 889 | 342 691 | 1.002 | 1.002 |

F3 moves 2 × 18 382 848 bytes in 1.164 ms, ~31.6 GB/s, both walks. The tiles buy
nothing there — which is the structural claim of #422, now measured on the view
path. Every row is inside ±1.5 %, and the three guard rows keep their block
vector unchanged by construction (asserted in the test above), so the decline
cannot move them at all.

Verdict: the acceptance row (F3 no worse) and the guard (Strided.jl parity rows
unchanged) both hold, so the answer to the issue's question is yes.

## What this does not claim

- No Strided.jl cross-check was run; the guard rows are the repository's own
  parity fixtures (contiguous, transposed, broadcast), not Julia timings.
- F3 is the only declined layout measured. A layout where the decline is a clear
  win on the view path was not found and is not claimed.
- The measurement scaffolding is a `bench-hooks` feature (`DECLINE_TILES`,
  `planned_blocks`), off in every normal build. If the hook itself is unwanted,
  dropping it and the bench keeps the routing change and its test intact.
