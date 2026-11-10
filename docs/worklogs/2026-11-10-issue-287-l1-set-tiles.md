# 2026-11-10 — #287: fitting destination tiles to the L1 sets

Machine: primerose, AMD EPYC (Zen 3), 64 cores in 8 L3 domains of 8; L1d 32 KiB 8-way (4 KiB way),
4 KiB pages (THP `madvise`). A shared host: measurements on the idle domain 32-39 only, agent pinned
off it. Hardware counters were not available (`perf_event_paranoid = 4`), so the mechanism below is
supported by timing controls, not by miss counts.

## Decision

`block::fit_l1_sets` shrinks the tiles of a destination-writing walk until each L1 set holds at most
`L1_WAYS - L1_WAYS / 4` = 6 lines that a tile keeps live for reuse. It runs on the view planner
(`build_plan_fused` with a destination) and on the raw pair replay (`tile_blocks`). Folds keep the
model's tiles. **The predeclared no-regression gate failed** (10 of 180 cells), see below; the change
is proposed with that result, not as a pass.

## Mechanism

On a power-of-two leading dimension every line of a transposed operand's run along the destination's
contiguous axis lands in one L1 set (`P = 4096 / gcd(stride, 4096) = 1`). The model's `[40, 32]` for
2048² `mul_into` with B row-major keeps 40 B lines live in an 8-way set. A hand scalar loop over the
same tiles (`benches/tile_probe_287.rs`, medians of 5 rounds) separates this from the walk:

| tile | 2048² walk | 2048² hand | 2047² walk | 2047² hand |
|---|---|---|---|---|
| [40,32] (model) | 20.2 ms | 17.9 ms | 15.3 ms | 9.4 ms |
| [8,64] (#287 probe) | 17.1 | 10.2 | 17.6 | 9.4 |
| [6,32] | 13.8 | 9.5 | 17.9 | 9.8 |

The aliasing penalty is real (hand loop 17.9 vs 9.4 ms on the non-power-of-two control) and the
issue's ~10 ms reproduces with the hand loop. The production walk is 4-7 ms slower than the hand loop
at every tile, which is a separate cost (per-run callback) that no tile removes.

The budget's quarter of headroom is calibrated, not derived: on the 2048² B-transposed cell a budget of
8 lines ran 0.78-0.93x the model, 6 lines 1.2x (f64) / 1.5x (f32), 4 lines 1.1x / 1.15x. Pseudo-LRU
replacement and the stride prefetcher pulling the next row's line into the same set are the plausible
reasons; they are not measured. The 2048² cells of the gate below are therefore not independent
evidence for the constant; the other shapes and 4T are.

## Gate (one paired run, protocol declared before it)

`benches/block_l1_sets_287.rs` at commit a65d56b, in-process toggle (`L1_SET_BUDGET` 0 vs 6), 45
layouts × {f64, f32} × {1T on core 33, 4T on 33-36}; 5 rounds of current / candidate / current-again,
median of 9 calls each, ratio inside a round. Noise threshold τ = max(5 %, p95 of |A/A − 1|) = 6.7 %;
a cell regresses if its median ratio < 1 − τ and all 5 rounds are slower. Every arm's output was
checked bitwise against the model's before timing.

- Tiles changed in 92 cells, none of them a non-power-of-two control. Unchanged cells: ratio 0.91-1.13.
- Changed cells: median ratio 1.44 (range 0.47-3.33). `copy_into` and `replay_copy_raw` gain
  in every changed cell (1.13-3.33x), `f32` `mul_into` 1.07-1.74x except B perm (2,0,1); `f64`
  `mul_into` 0.47-1.51x. The #287 cell (2048², B transposed): 1.22x f64 / 1.68x f32 at 1T, 1.11x / 1.55x at 4T.
- **Regressions (10):** `mul_into` with B permuted on [128,128,256]: (2,0,1) f64 (fused to [16384, 256], B row stride 2 KiB)
  and f32, (1,2,0) f64, (2,1,0) f64, each at 1T and 4T (0.47-0.80); 1024×4096 f64 at 4T, B transposed
  (0.72) and A,B transposed (0.84).

Reason, as far as the data goes: these are three-operand `f64` walks where the fit shortens the inner
run (to 6, 12 or 3 elements) on layouts whose aliasing is partial (B row stride 2 KiB, `P = 2`) or whose
per-thread share is small. The walk pays its per-run callback for every run, and in these cells that
cost exceeds what the set fit saves; the hand-loop probe shows the same walk overhead. The rule was not
narrowed to exclude them after the run.

Host: loadavg 4.7 at start, 7.1 at end; no foreign process on 32-39 at either point. Spread of the
current arm's median across rounds: median 10 %, max 20 %.

## Not done

- The walk's per-run overhead (the larger half of the #287 gap) is untouched.
- `AGENTS.md:133-134` (Key Constants) does not list `L1_WAYS` / `L1_LIVE_LINES_PER_SET`; left for the
  maintainer, as `AGENTS.md` is not edited by agents here.
