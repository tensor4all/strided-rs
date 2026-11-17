# 2026-11-17: #496 — a dispatched lane kernel and a lane reduction in `strided-segpoly`

Follow-up to #491 ([`2026-11-17-issue-491-segpoly-vs-llmlip.md`](2026-11-17-issue-491-segpoly-vs-llmlip.md)),
which measured `strided-segpoly` against llmlip on the MACE-MP-0b3 layer-1 CG
abstraction. #491 left three gaps: the `u = 128` f32 forward lane got 1.05x out of the
feature axis where llmlip's f32 path gets 2.2x, the VJP wrt the spherical harmonics
lowered to 11 008 scalar paths, and 4T scaled 3.9x where llmlip scaled 7.4x.
The issue asks for the first two as code, and this worklog reports the third.

## 1. Targets

| # | Target | Bar | Result |
| --- | --- | --- | --- |
| 1 | The `u = 128` f32 forward lane extracts real SIMD width, results unchanged | f32 1T within 1.15x of llmlip | **0.61-0.65x**, and 2.9-3.9x faster than before |
| 2 | `bar_edge_attrs` becomes a segmented partial-sum reduction, no `[E,16,128]` intermediate | 1T `bar_y` f64 <= 142 ms, f32 <= 140 ms | **37.8 ms** / **27.0 ms** at 128 atoms |
| 3 | 4T scaling | beat llmlip's 4T wall time in every cell | **in every forward cell** (0.58-0.73x); two f32 VJP cells lose (1.51x, 1.09x), see §7 |

Target 3 is reported as a wall time, not as a ratio of our own 1T, for the reason in
§7: the ratio is dominated by how slow a side's 1T is, and llmlip's 1T is 2.7x slower
than ours.

## 2. What changed

### 2.1 The lane kernel (`strided-segpoly/src/simd.rs`)

Operations with one to three operands no longer run the scalar per-lane loop. A new
`simd` module holds one `pulp::WithSimd` body per element type,
`AxpyF64<'a, N, SLICES>` / `AxpyF32<'a, N, SLICES>`, and the executor reaches it through
`pulp::Arch::new().dispatch(...)`. The dispatch compiles the body once per SIMD level
and picks one at run time, so the crate has no `target_arch` case, no width constant and
no `-C target-cpu` requirement: a baseline x86-64 build reaches AVX2 and AVX-512, and a
target with neither runs pulp's scalar arm (`F64_LANES == 1`), which is the scalar loop.

Three properties keep it a pure speedup rather than a numerics change:

* **No FMA.** The body multiplies and adds as two operations, so the rounding sequence
  is the one the scalar `accumulate` performs. `mul_add` is deliberately not used.
* **Fixed operand order.** The body computes `((c · x_0) · x_1) · x_2` left to right
  from the coefficient, whatever the operand count.
* **Whole lanes, then the tail in source order.** The vector part covers
  `lanes / V * V` lanes and the remainder is the scalar loop.

`op_row_lanes` resolves one row base per operand and passes it with the path's offsets;
each operand is either a slice of `u` lanes or one splatted scalar (a lane extent of 1 on
an input), decided by the const `SLICES` mask, so a broadcast operand costs no loads.
An operand whose lane stride is neither 0 nor 1 is copied into a packed tile once per
(block, input) and then read as whole vectors -- the arithmetic stays per-lane, so the
copy cannot move a bit. Operations with four or more operands keep the old loop and its
`shortcut:` note.

`src/simd/tests.rs` runs every entry point over every mask of its arity against the scalar
statement of the contract -- the mask is the only thing the dispatch selects by value -- on
19 lanes, so the vector loop and its remainder both run (the crate's per-file coverage gate
made that gap visible: `src/simd.rs` was at 73.9 % before the test, 98.9 % after).

`lane_width::<T>()` reports the width the dispatch selected (8 lanes f64 / 16 lanes f32
on this host, i.e. AVX-512); the campaign records it per cell (`"simd"` in every record),
and `examples/simd_probe.rs` runs an f64 contiguous and an f32 lane-strided body so
`scripts/check-runtime-simd-dispatch.py` can prove both levels exist in a baseline
build: `ok: 9 dispatched bodies reach AVX2 and AVX-512`.

### 2.2 The lane reduction (target 2)

`bar_edge_attrs` is `Σ_paths c · x · w · bar`, i.e. 86 CG entries × 128 features. #491's
harness lowered it as 11 008 scalar paths (`u = 1`), so the executor ran a
one-element-per-row plan over `N × 5120` rows. The descriptor was always available in
the other shape: the feature axis is the *lane* axis (lane extent 128) and the
destination is `[E, 16]`, i.e. lane extent **1**.

That shape used to be rejected (`OutputLaneBroadcast`). It is now the lane reduction it
reads as: an output whose lane extent is 1 while `u > 1` accumulates `u` lanes per
destination segment and writes their sum. The sum is taken in increasing lane order
after the block partials are combined, so the documented order extends by one rule and
nothing else moves:

1. rows in unit order, 2. operations in descriptor order, 3. paths in plan order within
their destination segment, 4. **the `u` lanes of a lane-reduced destination, increasing**.

Everything else -- `BLOCK_ROWS`, the combine, the scatter order, `execute_into` seeding
by zeroing -- is unchanged, and no `[E, 16, 128]` array is ever allocated: the plan
materialises one 2048-element tile per worker in the caller's scratch (16 slots × 128
lanes) and reduces it in place.

The harness's VJP-y plan is now that plan: 86 paths, two operations (the narrow `x0e`
block and the wide `x1o` block, whose lane stride is 3 and is packed), two `Gather`
index lists and one output. `"vjp_y_paths":11008` is kept in every record as the count
the same reduction takes in the `u = 1` form, next to `"vjp_y_plan_paths":86`.

### 2.3 Scratch

`scratch_len(budget)` is `block_partials + budget · (tile + pack_area)`, where
`block_slots · tile_elems` is the compile-time block-partial floor and `pack_area` is
one tile per packed operand of the costliest operation (`0` when nothing is packed).
`budget.max(1)` keeps a zero budget at the one-worker floor. A caller that passes less
than the floor still gets `ScratchTooSmall { need }` rather than a truncated sum; the
new floor is what `need` reports. `tests/scratch.rs` prints the four plans' numbers:
fwd 5504 elements at budget 1 and 44 032 at budget 8, vjp_x 384 / 3072, vjp_w
1664 / 13 312, vjp_y 2432 / 19 456 -- 19.5 KiB per call for the lane-reduced plan,
against buffers of hundreds of MiB.

## 3. Method

Unchanged from #491, and the two raw directories were produced by the same script:
`OUT=raw/496-before ./campaign.sh` before any crate change, `OUT=raw/496-after` after.
Pins: measured cells on `4` (1T) and `4,5,6,7` (4T) inside `index3/cache0 = 0-7,16-23`,
observer on `8-15,24-31`, governor `performance`, `--samples 15 --target-ms 20`, llmlip's
global rayon pool fixed to the cell's thread count. The anchor cell (fwd f64 1T 128) is
measured at both ends of each campaign: strided 69.6 → 69.0 ms before and 29.0 → 29.7 ms
after (2.7 %), llmlip 77.9 → 78.1 and 77.7 → 78.2 ms (0.7 %) -- inside the 15 % guard.

`raw/496-scaling/` holds the 1T-4T sweep of §7; `raw/campaign.txt` holds the #491 run and
`raw/496-before/campaign.txt` / `raw/496-after/campaign.txt` the two above.

## 4. Forward: `conv_tp`

Median wall time over 15 interleaved samples; ratio = strided / llmlip.

| dtype | T | atoms | llmlip (before) | strided (before) | ratio | llmlip (after) | strided (after) | ratio | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 | 1 | 128 | 78.3 ms | 69.0 ms | **0.88** | 78.0 ms | 28.1 ms | **0.36** | 2.45x |
| f64 | 1 | 1024 | 709.6 ms | 569.0 ms | **0.80** | 635.4 ms | 231.4 ms | **0.36** | 2.46x |
| f64 | 4 | 128 | 11.2 ms | 21.9 ms | **1.95** | 10.3 ms | 7.2 ms | **0.71** | 3.02x |
| f64 | 4 | 1024 | 87.9 ms | 142.6 ms | **1.62** | 88.1 ms | 64.7 ms | **0.73** | 2.20x |
| f32 | 1 | 128 | 48.2 ms | 82.9 ms | **1.72** | 35.2 ms | 21.6 ms | **0.61** | 3.85x |
| f32 | 1 | 1024 | 281.2 ms | 533.9 ms | **1.90** | 279.7 ms | 182.7 ms | **0.65** | 2.92x |
| f32 | 4 | 128 | 17.7 ms | 35.9 ms | **2.02** | 8.9 ms | 5.2 ms | **0.58** | 6.95x |
| f32 | 4 | 1024 | 71.8 ms | 133.5 ms | **1.86** | 71.9 ms | 49.0 ms | **0.68** | 2.73x |

Every cell improves by 2.2-7.0x, and the forward is now 1.4-2.8x faster than llmlip
where six of the eight cells had us 1.1-2.0x behind (the two f64 1T cells were already
ahead). Target 1 specifically: the f32 lane went from 1.7-1.9x *slower* to 1.5-1.7x
*faster* than llmlip, i.e. from 1.05x of the f64 rate to 1.3x.

### f32 vs f64 on the same cell

| cell | llmlip f32/f64 | strided f32/f64 |
| --- | --- | --- |
| 1T 128 | 0.45 | 0.77 |
| 1T 1024 | 0.44 | 0.79 |
| 4T 128 | 0.87 | 0.71 |
| 4T 1024 | 0.82 | 0.76 |

Target 1's second bar was "f32 within 0.6x of the same cell's f64 wall time", and the
measured 0.77-0.79 at 1T **misses it** (llmlip reaches 0.44-0.45 there). What the f32
cells do deliver is the target that matters: 1.5-1.7x ahead of llmlip where they were
1.7-1.9x behind, i.e. 2.9-3.9x faster than before, and at 4T both sides' dtypes
converge at 0.71-0.87. The residual is dtype-independent work per path and per row --
the coefficient, three offset loads, the mask select, the dispatch and the call do not
halve with the element size, and a lane-strided operand is copied at element count, not
byte count. Which share of the 20-25 % is which is **not measured**: no profile of the
f32 cell was taken, and the honest statement is that the lane loop is no longer the
bottleneck there. Cutting it further means fewer, fatter paths, which is a descriptor
shape question, not a width question. At 1T/1024 llmlip's 0.44 is also a slow-1T
artifact (§7): its per-core rate rises 1.72x from 1T to 4T, so its f32/f64 ratio at 1T
flatters a path that is 2.75x slower than ours in absolute time.

## 5. VJP: `bar_x`, `bar_w`, `bar_y`

llmlip returns the three gradients in one fused pass, so the comparable number is the
sum of the three plans (ms).

| dtype | T | atoms | `bar_y` before | sum before | ratio | `bar_y` after | sum after | ratio | `bar_y` speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 | 1 | 128 | 527.8 | 664.4 | 3.02 | 37.8 | 94.4 | 0.48 | 14.0x |
| f64 | 1 | 1024 | 3501.5 | 4608.7 | 2.87 | 349.6 | 859.9 | 0.55 | 10.0x |
| f64 | 4 | 128 | 114.6 | 158.0 | 4.16 | 9.6 | 28.9 | 0.77 | 12.0x |
| f64 | 4 | 1024 | 909.0 | 1263.8 | 4.41 | 89.2 | 257.8 | 0.96 | 10.2x |
| f32 | 1 | 128 | 418.7 | 542.9 | 11.82 | 27.0 | 69.6 | 1.53 | 15.5x |
| f32 | 1 | 1024 | 3379.8 | 4404.6 | 9.92 | 231.6 | 596.4 | 1.38 | 14.6x |
| f32 | 4 | 128 | 107.7 | 151.3 | 9.40 | 7.1 | 18.2 | 1.51 | 15.1x |
| f32 | 4 | 1024 | 924.8 | 1257.2 | 7.41 | 58.9 | 179.4 | 1.09 | 15.7x |

`bar_y` is 10-16x faster and the VJP sum is 7.0x faster at 128 atoms (f64: 664.4 →
94.4 ms) and 7.4x at 1024 atoms (f32: 4404.6 → 596.4 ms). `bar_x` and `bar_w` improved
too (e.g. f64 1T 1024: 581.2 → 288.2 and 525.9 → 222.2 ms) because they also read `x1o` at lane stride 3
-- the packed tile replaced their fallback to the scalar loop.

Target 2 asked for 1T `bar_edge_attrs` <= 142 ms f64 / <= 140 ms f32 at 128 atoms:
**37.8 ms** and **27.0 ms**. The VJP-sum ratios (1.53 f32, 0.48 f64 at 1T/128) are
within the 2.5x / 1.6x bars of the plan.

## 6. Allocation, scratch, memory

`alloc_bytes_per_call` and `alloc_calls_per_call` are **0 and 0** for every strided
entry at 1T in both campaigns, and `the_executor_allocates_nothing_per_call` (serial
token, warmed pool) still passes. Some 4T cells record a 1 520-byte allocation at a low
rate: the campaign's 15-sample cells show 0-3 events, and `--samples 240` on the f32 4T
forward gives 15 events in 960 calls, i.e. 25 B/call amortised (0.008 % of one call's
scratch). A backtrace probe on the counting allocator (`std::backtrace` behind a
1520-byte size filter, guarded against recursion, reverted afterwards) attributed
**every one** of those events to rayon worker-thread start-up --
`rayon_core::registry::WorkerThread::from` under `ThreadBuilder::run`, and
`set_global_registry` under `fix_llmlip_threads` -- and none to `execute_into`. The
counter is process-wide, so a worker spawned by either pool while our window is open is
charged to us. #491 read the same 1 520-byte events as a per-plan first touch on the
pool's workers; the 15-in-960 rate refutes that, and the backtraces say rayon, not the
executor. llmlip allocates 5 475 056 B in 43 calls (f64 1T 128) and 250 507 344 B in
210 calls (f32 4T 1024) per call -- unchanged, and not ours to fix.

Peak RSS is unchanged to within 1.5 MiB: fwd f64 1T 128 85.1 → 84.1 MiB, VJP f64 1T 1024
1510.5 → 1508.9 MiB, dominated by the buffers both sides share. The lane reduction needs
no `[E, 16, 128]` intermediate; the plan's own footprint is the 2432-element scratch
(19 KiB) printed in §2.3.

## 7. Thread scaling (target 3)

Same campaign data, 1T vs 4T, plus a 1/2/3/4T sweep of one cell (`raw/496-scaling/`)
to separate decomposition from clock.

| cell | dtype | atoms | 1T | 4T | strided scaling | llmlip scaling |
| --- | --- | --- | --- | --- | --- | --- |
| cg-fwd | f64 | 128 | 28.1 ms | 7.2 ms | **3.88x** | 7.61x |
| cg-vjp | f64 | 128 | 94.4 ms | 28.9 ms | **3.26x** | 5.20x |
| cg-fwd | f64 | 1024 | 231.4 ms | 64.7 ms | **3.58x** | 7.21x |
| cg-vjp | f64 | 1024 | 859.9 ms | 257.8 ms | **3.34x** | 5.75x |
| cg-fwd | f32 | 128 | 21.6 ms | 5.2 ms | **4.18x** | 3.95x |
| cg-vjp | f32 | 128 | 69.6 ms | 18.2 ms | **3.82x** | 3.78x |
| cg-fwd | f32 | 1024 | 182.7 ms | 49.0 ms | **3.73x** | 3.89x |
| cg-vjp | f32 | 1024 | 596.4 ms | 179.4 ms | **3.33x** | 2.62x |

| threads (cores) | strided fwd f64 1024 | scaling | efficiency | llmlip | scaling |
| --- | --- | --- | --- | --- | --- |
| 1 (4) | 233.0 ms | 1.00x | -- | 638.6 ms | 1.00x |
| 2 (4,5) | 129.7 ms | 1.80x | 0.90 | 171.3 ms | 3.73x |
| 3 (4,5,6) | 88.5 ms | 2.63x | 0.88 | 122.4 ms | 5.22x |
| 4 (4,5,6,7) | 63.9 ms | 3.65x | 0.91 | 92.9 ms | 6.87x |

Efficiency is `scaling / threads`.

The executor already splits by *rows within a run*, one task family per destination run,
each split by cumulative edge count, so 128 nodes are not the granularity. The sweep
shows what is left: 0.90 / 0.88 / 0.91 efficiency per added thread -- a 4-thread
all-core clock below the 1T boost, not a tail. A 4x ceiling minus that is 3.5-3.6x,
which is what all four 1024-atom cells show (3.3-3.7x); the 128-atom f32 forward reaches
4.18x.

llmlip's 6.87x is the other direction: its 1T is 2.75x slower than ours and its per-core
rate rises 1.72x from 1T to 4T, which no decomposition of a fixed kernel can produce.
Judged by absolute 4T wall time -- what a user waits for -- the strided side wins every
**forward** cell: 64.7 vs 88.1 ms (f64 1024), 49.0 vs 71.9 (f32 1024), 7.2 vs 10.3 (f64
128), 5.2 vs 8.9 (f32 128). It does **not** win the two f32 4T VJP cells: the three-plan
sum is 18.2 vs 12.1 ms at 128 atoms (1.51x slower) and 179.4 vs 164.6 ms at 1024 (1.09x),
even though the sum is 7.0-8.3x faster than before. The f64 4T VJP cells are 0.77x and
0.96x. The target "4T >= 6x of our own 1T" is not reachable on 4 physical
cores without a kernel whose 1T is 1.7x slower; the plan recorded that before the work
started and it is not a decomposition failure.

## 8. Numerics

Exact bit identity is what the crate's tests check: the dispatched kernel's output is
compared element by element (`assert_eq!` on `f64`/`f32` slices) against a scalar
reference in the documented order, for lane strides 1, 2, 3, 5, -1 and -2 at 19 lanes (so
the vector loop and its scalar remainder both run) and 517 `Batched` rows (one unit each).
Block partials are a `Shared` output's business, so the "combine before the lane sum" rule
has its own cancellation-sensitive case,
`a_shared_lane_reduction_combines_block_partials_first`: two blocks whose lane partials are
`[1e16, 1]` and `[-1e16, 1]` reduce to 2 when the partials are combined lane-wise first,
and to 0 if each block's lanes are reduced first. Budget and thread invariance are checked
by the pooled `a_lane_reduction_is_bitwise_across_budgets_and_block_partials` (2 and 4
threads x budgets 1/2/4 on four block partials, `tests/kernels.rs`): bit equality between
executions, with the order itself pinned by the two cases above. Both scalars are `f64`;
the `f32` kernel is covered by the campaign's own check and by the crate's `f32`
conversions, not by a second hand-written oracle. What the two campaigns add is corroboration, not
proof: every forward cell's `rel_to_scale` is unchanged in the printed digits
(f64 6.362e-16 at 128 atoms / 4.626e-16 at 1024; f32 2.562e-7 / 2.981e-7), and `bar_x` and
`bar_w` are unchanged too (f64 1T 128: 3.20e-16 and 0.00e+00; 4T 1024: 5.54e-16 and
6.06e-16). Those are aggregates -- many outputs could move a bit without moving a maximum
-- so they are reported as "the aggregate error statistics did not change", and the exact
claim rests on the tests.

`bar_y` moves, because its association changed from 11 008 scalar path additions to 86
path additions of 128-lane sums followed by a lane reduction. It stays at the rounding
level of the sum: f64 128 atoms 1.69e-15 → 2.20e-15, 1024 atoms 2.49e-15 → 2.80e-15
(bar: <= 3.6e-15); f32 128 atoms 3.40e-6 → 9.97e-7 and 1024 atoms 1.29e-6 → 1.17e-6
(bar: <= 1e-5, and f32 got *better*). 1T and 4T agree bit for bit in every cell: the
harness's `bits_are_thread_invariant_f32` / `_f64` tests pass, and the lane reduction is
budget-invariant because the partials are combined before the lanes are summed
(`a_lane_reduction_is_budget_invariant`).

## 9. Limits

* Four-or-more-operand operations still take the scalar loop (`shortcut:` note kept in
  `exec.rs`); none of the layer-1 plans has one.
* The lane reduction reads the destination as the accumulator's first term, which is
  what `execute_into`'s zeroing and `execute_acc`'s seeding mean for a lane unit output.
  A caller that seeds a lane-reduced destination gets "seed + Σ lanes"
  (`a_seeded_lane_reduction_adds_the_seed_once`).
* `pack_area` is the sum over the packed operands of the *costliest* operation, not of
  all operations; a plan whose packed operands differ per operation is sized by the
  maximum, which is correct because operations run one at a time.
* f32 does not reach llmlip's f32/f64 ratio (§4); the absolute numbers are 1.5-1.7x
  ahead of llmlip in every f32 *forward* cell (the two f32 VJP cells are still behind,
  §7), so this is a reporting limit, not a target met by weakening the bar.
* No CUDA, no `-C target-cpu=native`, no change to `BLOCK_ROWS` or to the other four
  plans' shapes.

## 10. Reproduction

```bash
# baseline (a worktree before the crate change) and the result, same script
cd benchmarks/strided-491-mace-ab
OUT=raw/496-before ./campaign.sh          # 18 records, 0 FAILED
OUT=raw/496-after  ./campaign.sh

# the numbers in the tables
python3 check_report_496.py

# the dispatch proof (baseline x86-64 build, no target-cpu)
python3 ../../scripts/check-runtime-simd-dispatch.py

# the crate gates
cargo test --release -p strided-segpoly
cargo test --release -p mace-ab
```

`raw/496-before/` and `raw/496-after/` hold 16 cell records each plus the
two-line-per-cell `campaign.txt` (cell header with its timestamp, then the record);
`raw/496-scaling/` holds the 1/2/3/4T sweep. The host is the one `raw/host.txt` of the
#491 campaign describes -- same machine, same pins -- and this run did not re-capture
`host.txt` / `load.txt` / `commands.txt`.

## 11. Review

**Plan gate.** The plan was reviewed before any code was written by
`ollama-cloud/glm-5.3-flash` (fresh read-only `pi --print` process). Round 1:
**`VERDICT: CHANGES`** -- the scratch formula did not cover the pack area, the probe could
not reach the widths it claimed, the plan said both "pack" and "splat" for a unit-lane
operand, the overflow checks were missing for a u-lane product, and two pinned tests would
change meaning. Round 2: **`VERDICT: APPROVE`** after those five were folded in.

**Post-implementation review.** The finished diff went to `openai/gpt-6.1-sol` (same
harness, read-only, whole-request-in-prompt). Round 1: **`VERDICT: CHANGES`**.

| finding | disposition |
| --- | --- |
| BLOCKER: a lane stride of 0 with lane extent `u` was compiled as "packed" while the replay read it in place -- out of bounds for `[1,1,4]` / `[1,1,0]` | real; fixed by one three-way classification (`splat` = extent 1 **or** stride 0, else `packed` when stride != 1) used by both the sizing and the replay; `a_stride_zero_lane_is_a_broadcast_not_a_pack` |
| BLOCKER: offsets were rewritten into tile space for every operation, but only arity <= 3 packs, so an arity >= 4 fallback read tile offsets from an unpacked buffer | real; `tiled = lanes && packed[b]`; `a_deep_operation_reads_the_buffer_offsets` |
| BLOCKER: a pathless operation resolved base pointers from a dangling empty view before noticing it had no paths (pre-existing) | real; skipped before any base is resolved |
| IMPORTANT: the reduction tests did not establish the claimed association -- the reference summed lanes inside a path, `budget` only changed the scratch length, and 5 lanes never reached the vector loop | real; the reference now sums paths within a lane and lanes last on cancellation-sensitive payloads, and the cases run 19 lanes |
| IMPORTANT: "wins every 4T cell" was false -- the two f32 4T VJP sums are 1.51x and 1.09x *slower* than llmlip | real; §7 and the PR body say "forward" and name the losing cells |
| IMPORTANT: unchanged rounded error statistics do not prove bit-identical results | real; §8 now rests the exact claim on the element-wise tests and labels the campaign figures as aggregates |
| MINOR: the `OutputLaneBroadcast` doc still forbade lane-extent-1 outputs | fixed |
| delete: a second mask-selection layer between the executor and the dispatch | removed; `lane_axpy` takes a runtime `slices` count |
| delete: the public dispatch hooks on `SegPolyElem` (a caller could invoke a panicking arity-0 body) | partly; see round 2 |
| delete: `Box::leak` in the probe example | removed |

Round 2 (same model) accepted fixes 1-3, 7 and the deletions, and raised two IMPORTANTs
about what the *tests* and the *records* still failed to establish -- both accepted:

| finding (round 2) | disposition |
| --- | --- |
| `lane_axpy` is still callable downstream through the supertrait bound, so `lane_axpy::<0>` panics -- the comment claimed a boundary that does not exist | real. The comment now says what sealing does and does not do, and an arity outside 1..=3 is a `debug_assert!` plus a no-op in release instead of an out-of-bounds index. Hiding it completely would need a private token threaded through the executor's hot loop or a function-pointer indirection: more machinery than a can't-happen call in someone else's code is worth. |
| the reduction tests still did not establish "combine block partials before reducing lanes": their outputs are `Batched`, so each row is a unit, and the pooled case only compares the implementation with itself | real. `a_shared_lane_reduction_combines_block_partials_first` adds the independent, cancellation-sensitive `Shared` oracle (2 with the documented order, 0 with the per-block order); the worklog's §8 claim was corrected to match. |
| the PR draft still inferred bit identity from unchanged aggregate errors | fixed |
| the before-campaign anchor was stale (#491's numbers), the §9 seeded-reduction limitation was obsolete, and "1.5-1.7x ahead in every f32 cell" included the losing VJP cells | all three fixed |
| delete: the one-, two- and three-operand arms of `accumulate` are dead -- arity <= 3 always leaves through the lane kernel | removed, with the `unrolled!` macro |
| delete: the serial-only budget sweep in `a_lane_reduction_is_budget_invariant` only changes the scratch length | removed; the pooled test owns budget coverage |

The regression tests were checked against the round-1 code (the classification reverted in
place, nothing else): `a_stride_zero_lane_is_a_broadcast_not_a_pack` and
`a_deep_operation_reads_the_buffer_offsets` fail there and pass on the fix, so they pin a
real defect rather than the current behaviour. `a_unit_lane_plan_keeps_the_buffer_strides`
passes both ways -- round 1's splat rule already covered extent 1, and the test pins that
reading so a later change to the classification cannot silently pack it.

**Round 3** (same model, on the diff with the round-2 fixes): **`VERDICT: APPROVE`** -- no
new BLOCKER or IMPORTANT. Its three MINORs were stale comments, now fixed: §8 still named
the deleted serial budget test, the `Batched` lane-reduction comments still described block
combination, and the arity-six fallback test called its five operands four. It also noted
that an operation with *no* inputs still reaches the scalar fallback (`op.lanes` needs at
least one), which the `shortcut:` note now says.

So: `review: gpt-6.1-sol APPROVE` after three rounds, and the plan gate
`ollama-cloud/glm-5.3-flash APPROVE` after one `CHANGES` round.

**Not fixed, deliberately** (out of this change's scope, all pre-existing): the other three
layer-1 plans keep their shapes, `benchmarks/strided-491-mace-ab` -- not a workspace member
-- has three clippy findings that predate this branch (`src/cg.rs:55`,
`src/measure.rs:203`, `src/lib.rs:133`), and four-or-more-operand operations keep the
scalar loop.
