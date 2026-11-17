# 2026-11-18: #498 — one traversal for the three `strided-segpoly` VJP plans

Follow-up to #491 ([`2026-11-17-issue-491-segpoly-vs-llmlip.md`](2026-11-17-issue-491-segpoly-vs-llmlip.md))
and #496 ([`2026-11-17-issue-496-segpoly-simd-reduce.md`](2026-11-17-issue-496-segpoly-simd-reduce.md)).
After #497 the f32 VJP on the MACE-MP-0b3 layer-1 block was still 1.09-1.53x slower than
llmlip at 4T, because the VJP ran as three separate plans (`bar_x`, `bar_w`, `bar_y`) over
the same edges. #498 asks for one traversal — without materialising the
`[E, 16, 128]` intermediate — and for f32 4T `VJP / llmlip <= 1.0`.

## 1. Targets

| # | Target | Bar | Result |
| --- | --- | --- | --- |
| 1 | f32 4T VJP wall time / llmlip, 128 atoms | <= 1.00 | **0.741** (8.98 ms vs 12.11) |
| 2 | f32 4T VJP wall time / llmlip, 1024 atoms | <= 1.00 | **0.520** (84.05 ms vs 161.70) |
| 3 | One traversal, no `[E, 16, 128]` intermediate | by construction | one call, one row loop, one `Exec`; the plan's own footprint is its scratch: 23 552 elements for eight workers (92 KiB f32 / 184 KiB f64), 2944 at budget 1 |
| 4 | Bit-identical to the three split plans | both entry points, 1T and 4T, every budget | `a_fused_plan_matches_the_split_plans_bit_for_bit`, `a_fused_plan_is_budget_and_thread_invariant`, `the_fused_vjp_matches_the_split_plans_{f32,f64}` |
| 5 | No per-call allocation | 0 bytes, 0 calls | **0 / 0** in all 8 VJP cells, plus the harness and crate allocation gates |
| 6 | No f64 regression | f64 4T VJP <= before | 28.87 -> **15.48 ms** at 128, 267.37 -> **129.21 ms** at 1024 |

Target 1 and 2 are `strided_total_ns / llmlip` from `raw/498-after`, which is the fused
call; `raw/498-before` is the same harness on the baseline tree for the same cell.

## 2. What changed

### 2.1 One traversal: output groups (`strided-segpoly/src/plan.rs`, `src/plan/exec.rs`)

A compiled plan used to have one unit per output row and a task list that walked every
output separately. It now groups outputs that can be evaluated in one pass over the batch
points:

* **Scattered outputs with the same compiled run topology** (`RunIndex::ptr()` and
  `perm()` equal by value) form one group. They share the traversal and keep that
  topology's permutation order, which is what their documented summation order says: a
  run's rows in increasing `j`, i.e. sorted-space order.
* **Batched outputs join the first group.** A batched row is one batch point, so the
  visit order does not enter its sum; it is written once, when the traversal reaches its
  point.
* **A shared output keeps its own group** and its natural `0..batch` reduction. Fusing a
  batch reduction into a permuted traversal would change its order, so it does not.

Each group owns an accumulator region in the caller's scratch sized
`Σ segments × u` over its accumulating outputs, so the two `bar_x` outputs write their
`[segments, u]` blocks side by side instead of each re-walking the edge list. A unit is
one destination block of one group; the existing row-balanced task split, `BLOCK_ROWS`
block partials and left-to-right combine are unchanged, and a streamed (batched) output
writes its point inside whichever task holds it, so no two tasks write one row.

`bar_y` (`u = 1` while the inputs have 128 lanes) is a lane reduction and stays one; it is
streamed per point like any batched output, and its `[16, 128]` tile is the child's
`acc_stride` region, not an `[E, 16, 128]` array.

One behaviour tightens. Until now the replay read the call's own `perm` and traversed **that**
order, so a `Scatter` call could pass the compiled windows in any row order and the sum
followed the argument -- correct per seed, but the result was a function of the call, not of
the plan. A grouped output shares one traversal, so the order must be the plan's; a call
whose `RunIndex` has the compiled windows but a different `perm` is now
`StridedError::PlanLayoutMismatch` (checked before anything is written), and the documented
order is the compiled one. The error lists in `execute_into` / `execute_acc` say so, and
`a_scatter_call_must_carry_the_compiled_run_order` pins it for both entry points.

### 2.2 The dispatch hoist (`strided-segpoly/src/simd.rs`)

Fusing the three plans was necessary but not sufficient: with only the groups in place the
f32 4T 128 cell was 1.26x (15.17 ms vs 12.07 ms), because one row of the fused plan runs
3 gradients × 86 paths = 258 lane-kernel calls, and each call resolved the CPU features
(`pulp::Arch::new()`), matched the splat mask and dispatched. The plan's contingency was
to hoist that out of the per-path call; the measurement said it was the whole remaining
gap, so it was done:

* the replay resolves `pulp::Arch` once per operation row instead of once per path;
* the path loop now lives **inside** the dispatched body: `simd::OpRow` carries the row's
  segment bounds, coefficients, offsets and operand bases, and `RowF64` / `RowF32`
  (one `pulp::WithSimd` body per element type, one const `(arity, slices-mask)`
  instantiation each) runs every path of the row under one dispatch.

The arithmetic is unchanged: for each segment, for each of its paths, for each lane,
`acc = acc + ((c · x_0) · x_1) · x_2`, the vector loop and the scalar remainder doing the
same two roundings in the same order (still no FMA). `src/simd/tests.rs` checks every
`(arity, mask)` of the row entry against the scalar statement on 19 lanes, and the crate's
bitwise tests compare plans with a scalar reference and against each other.

This is where most of the win is, and it is not VJP-specific: every `strided-segpoly`
plan runs its operations through this kernel, so the forward cells improved by 1.3-1.6x as
well (§4). No profile was available (`perf_event_paranoid = 4` on this host); the change was
selected from the plan's contingency and validated by measurement, not by a profile.

`scripts/check-runtime-simd-dispatch.py` was updated to the new body names and still
reports `ok: 9 dispatched bodies reach AVX2 and AVX-512` on a baseline build, so the
runtime-ISA portability the crate documents is intact.

### 2.3 Scratch

`scratch_len(budget)` is now

```text
block_slots · acc_stride  +  max(budget, 1) · (acc_stride + row_elems + pack_elems)
```

where `acc_stride` is the widest group's `Σ segments × u` (the block partials need one
such region per slot), `row_elems` is one `tile_elems` when some group streams a batched
row (`0` otherwise), and `pack_elems` is unchanged. The floor `execute` enforces is
`block_slots · acc_stride + child_elems`; one element less is still
`ScratchTooSmall { need }`. The harness prints the new numbers: the fused plan needs 2944
elements at budget 1 and 23 552 at budget 8, against 512/4096 for `bar_x`,
1664/13 312 for `bar_w` and 2432/19 456 for `bar_y` — one call's scratch instead of three
plans' buffers, and still four orders of magnitude below the buffers.

### 2.4 The harness (`benchmarks/strided-491-mace-ab`)

`cg.rs` builds a fifth plan: the same six operations (`bar_x` narrow/wide, `bar_w`
narrow/wide, `bar_y` narrow/wide) over five inputs and four outputs, `bar_x`'s two outputs
scattered over one `RunIndex`, `bar_w` and `bar_y` batched. `main.rs` times it as
`strided_vjp_fused` and reports it as `strided_total_ns` (the number the target is stated
against), keeps the three split timings, and adds `strided_split_total_ns` and
`ratio_split` so the fusion's own share is visible in every record. The `check` block's
numerics now come from the fused call.

`tests/fidelity.rs` gates the fused call against the split plans bit for bit (1T and 4T)
and against llmlip within the dtype tolerance; `tests/alloc.rs` adds the fused call to the
zero-allocation gate; `tests/scratch.rs` prints the fused plan's scratch.

## 3. Method

Same protocol as #496. Pins: measured cells on `4` (1T) and `4,5,6,7` (4T) inside
`index3/cache0 = 0-7,16-23`; observer, campaign shell and this agent process on
`8-15,24-31` (`taskset -apc` on the agent's pid for the run); governor `performance`;
`--samples 15 --target-ms 20`; llmlip's global rayon pool fixed to the cell's thread
count. `OUT=raw/498-before ./campaign.sh` ran on a pristine worktree of the baseline
commit `e8d3cdf`, `OUT=raw/498-after ./campaign.sh` on this branch, back to back.

Anchors (fwd f64 1T 128), measured at both ends of each campaign:

| campaign | strided start -> end | drift | llmlip start -> end | drift |
| --- | --- | --- | --- | --- |
| before | 29.0 -> 28.8 ms | 0.7 % | 78.0 -> 77.9 ms | 0.1 % |
| after | 21.1 -> 19.6 ms | 7.0 % | 77.9 -> 78.0 ms | 0.2 % |

Both campaigns are inside the 15 % guard `campaign.sh` enforces; the after campaign's
strided anchor moved 7 %, which is the noise floor to read the smaller f32 4T cells
against. The before campaign's own anchor (29.8 ms in its records, 29.0/28.8 in the guard)
is 6 % slower than #496's 28.1 ms on the same build, i.e. the host was slightly slower
today; the two campaigns ran in one window, so the before/after pairs are comparable.

No attempt was discarded and no cell failed.

## 4. Forward: `conv_tp`

Median wall time over 15 interleaved samples; ratio = strided / llmlip.

| dtype | T | atoms | llmlip (before) | strided (before) | ratio | llmlip (after) | strided (after) | ratio | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 | 1 | 128 | 78.01 ms | 29.78 ms | **0.382** | 77.91 ms | 21.31 ms | **0.274** | 1.40x |
| f64 | 1 | 1024 | 636.81 ms | 225.21 ms | **0.354** | 636.45 ms | 168.60 ms | **0.265** | 1.34x |
| f64 | 4 | 128 | 10.14 ms | 7.29 ms | **0.718** | 10.14 ms | 5.72 ms | **0.564** | 1.27x |
| f64 | 4 | 1024 | 86.71 ms | 60.91 ms | **0.703** | 85.88 ms | 43.22 ms | **0.503** | 1.41x |
| f32 | 1 | 128 | 35.09 ms | 19.83 ms | **0.565** | 34.72 ms | 12.46 ms | **0.359** | 1.59x |
| f32 | 1 | 1024 | 280.09 ms | 167.78 ms | **0.599** | 279.14 ms | 120.95 ms | **0.433** | 1.39x |
| f32 | 4 | 128 | 8.85 ms | 4.57 ms | **0.516** | 8.83 ms | 2.98 ms | **0.337** | 1.53x |
| f32 | 4 | 1024 | 71.86 ms | 49.52 ms | **0.689** | 72.24 ms | 30.18 ms | **0.418** | 1.64x |

The forward did not change shape: it is the same single-output plan, and it improved
1.27-1.64x because it runs through the same lane kernel (§2.2). No forward cell regressed.

## 5. VJP: `bar_x`, `bar_w`, `bar_y`

llmlip returns the three gradients in one fused pass, so the comparable number is one
call: the fused `strided_total_ns` after, the three-plan sum before. `split after` is the
same branch's three-plan sum, i.e. the fusion isolated from the dispatch hoist.

| dtype | T | atoms | llmlip (before) | sum before | ratio | llmlip (after) | fused after | split after | ratio | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f64 | 1 | 128 | 195.93 ms | 90.28 ms | **0.461** | 195.09 ms | 63.49 ms | 69.56 ms | **0.325** | 1.42x |
| f64 | 1 | 1024 | 1556.26 ms | 827.37 ms | **0.532** | 1554.57 ms | 504.95 ms | 554.76 ms | **0.325** | 1.64x |
| f64 | 4 | 128 | 35.25 ms | 28.87 ms | **0.819** | 34.54 ms | 15.48 ms | 17.46 ms | **0.448** | 1.87x |
| f64 | 4 | 1024 | 256.80 ms | 267.37 ms | **1.041** | 258.41 ms | 129.21 ms | 143.74 ms | **0.500** | 2.07x |
| f32 | 1 | 128 | 45.79 ms | 67.54 ms | **1.475** | 45.76 ms | 40.72 ms | 47.76 ms | **0.890** | 1.66x |
| f32 | 1 | 1024 | 433.15 ms | 582.57 ms | **1.345** | 426.64 ms | 331.32 ms | 367.29 ms | **0.777** | 1.76x |
| f32 | 4 | 128 | 12.02 ms | 20.00 ms | **1.664** | 12.11 ms | 8.98 ms | 10.79 ms | **0.741** | 2.23x |
| f32 | 4 | 1024 | 161.50 ms | 177.18 ms | **1.097** | 161.70 ms | 84.05 ms | 92.86 ms | **0.520** | 2.11x |

Where the 2.1-2.2x comes from, at f32 4T:

| step | 128 atoms | ratio | 1024 atoms | ratio |
| --- | --- | --- | --- | --- |
| three plans (baseline) | 20.00 ms | 1.664 | 177.18 ms | 1.097 |
| fused groups only, before the dispatch hoist | 15.17 ms | 1.256 | 127.57 ms | 0.777 |
| fused groups + row dispatch (campaign) | 8.98 ms | **0.741** | 84.05 ms | **0.520** |

The middle row is a **single-cell probe on an intermediate tree, not a campaign record**:
`taskset -c 4,5,6,7 ./target/release/mace-ab --cell cg-vjp --dtype f32 --threads 4 --atoms N
--samples 15 --target-ms 20`, with that probe's own llmlip side (12.07 and 164.20 ms). The
first step is 1.32-1.39x, i.e. the fusion alone was *not* enough at 128 atoms; the second is
another 1.52-1.69x. Both are in this PR because the second is what a 258-path traversal
costs.

Per-plan times show where the remaining time goes (after, ms): f32 4T 128 `bar_x` 2.70,
`bar_w` 3.00, `bar_y` 5.10; the fused call is 8.98 rather than their 10.79 ms sum, i.e.
the three gradients share 17 % of their work by construction. `bar_y` is still the
most expensive of the three: its accumulator is `[16 segments × 128 lanes]` for a
16-element destination, so it reads and writes 2048 elements per path where `bar_x` writes
512 — that is the lane form of the reduction #496 introduced, not new work here.

## 6. Allocation, scratch, memory

`alloc_calls_per_call` and `alloc_bytes_per_call` are **0 and 0** for the fused VJP in all
8 cells of `raw/498-after` (and for the forward at 1T; four 4T cells record the #496-noted
1 520-byte rayon worker start-up at 0.017 calls/call). The harness gate
`the_executor_allocates_nothing_per_call` (serial, warmed pool, fused call included) and
the crate's `execute_allocates_nothing` (now a fused plan with two scattered outputs, one
batched output and one shared output, both entry points, five shapes including 600 rows)
both pass.

The fused plan's scratch is 2944 elements at budget 1 and 23 552 at budget 8 — the same at
128 and 1024 atoms, because no run in this descriptor exceeds `BLOCK_ROWS = 256` rows, so
`block_slots` is 0 and the scratch is the per-worker `acc_stride + row_elems + pack_elems`.
For f32 that is 11.5 KiB per worker. No `[E, 16, 128]` array exists anywhere: `bar_y`'s
accumulator is the child's `[16, 128]` region.

Peak RSS is unchanged within 1 MiB of the before campaign in every one of the 16 cells — the
process peak is dominated by the buffers both sides share (f64 1T 1024 VJP 1509.7 vs 1509.5
MiB). `grad_x`, and the `bar_y` intermediate the comparison excludes:

| dtype | atoms | grad_x | bar_y intermediate |
| --- | --- | --- | --- |
| f64 | 128 | 0.5 MiB | 92.0 MiB |
| f64 | 1024 | 4.0 MiB | 736.0 MiB |
| f32 | 128 | 0.2 MiB | 46.0 MiB |
| f32 | 1024 | 2.0 MiB | 368.0 MiB |

The intermediate is an array this plan never allocates: `bar_y`'s accumulator is the child's
`[16, 128]` region, and the fused plan's own state is the scratch above, one `Exec` and a
stack `Call`.

## 7. Numerics and determinism

* **Fused == split, bit for bit.** `a_fused_plan_matches_the_split_plans_bit_for_bit`
  builds the same split plans and compares every output by bit pattern (`to_bits`, not
  `==`, which would accept `-0.0`), for `execute_into` and for a *seeded* `execute_acc`,
  in three shapes: 5 atoms over 3 runs, 300 rows in one run (two block partials per
  scattered output), the same 300 rows with every lane at stride 2 (packed inputs and
  non-contiguous accumulators in a block slot), and 5 atoms with the same strides (the
  same in a `Child` slot) — so one child holds accumulators, a streamed row and a pack
  tile at once, with and without block partials.
* **The rest of the layout.** `a_fused_plan_is_budget_and_thread_invariant` sweeps budgets
  1/2/4 on a 4-thread pool for both entry points in two shapes: 300 rows in one run (two
  block partials, two tasks) and 600 rows over two sources (two 300-row runs, four block
  tasks). It seeds `execute_acc` destinations with a non-zero payload, and it does *not*
  assert a child width: how far the executor forks is its cost model's and the pool's
  decision, not the plan's (its 9 600-mac case opens to four children on x86 and to two on a
  baseline aarch64 host); the plan's own task count is pinned in `src/plan/tests.rs`;
  `one_group_accumulator_covers_every_output_of_one_traversal` pins the block-slot and
  child-region sizing; `scatter_outputs_with_different_topologies_keep_their_own_order`
  and `scatter_outputs_with_different_run_orders_get_their_own_group` pin the two cases
  that must *not* share a traversal (different windows, same windows in another row
  order); `a_scatter_call_must_carry_the_compiled_run_order` pins the rejection at both
  entry points and that a refused call writes nothing.
* **The row loop.** `a_row_visits_every_segment_and_path_at_its_own_offset`
  (`src/simd/tests.rs`) runs 2 segments and 5 paths at distinct non-zero offsets with
  non-dyadic payloads and coefficients against the scalar statement, so the moved path
  loop, its offset stride (`arity + 1` across the inputs' entries) and the two roundings
  are what it pins — an FMA contraction or a re-association changes the bits. It fails
  when the stride is mutated (checked by reverting the kernel's `local * (N + 1)` to
  `local * N`).
* **The harness gates the same thing on the real descriptor**: both dtypes, 1T and 4T,
  fused output equal to the three split plans' output element for element.
* **llmlip agreement** (scale relative, worst of the four cells per dtype in
  `raw/498-after`): f64 VJP `bar_x` 5.54e-16, `bar_w` 6.06e-16, `bar_y` 2.80e-15; f32 VJP
  `bar_x` 2.97e-07, `bar_w` 2.68e-07, `bar_y` 1.17e-06 — inside the 1e-14 / 1e-5 bars the
  harness gates at 1T.
* **Order preserved, one pass.** Each group's own order is the one it had: a scattered
  group's runs in the compiled order, a shared output's batch in `0..batch`, operations in
  descriptor order, paths in plan order, `BLOCK_ROWS` partials combined left to right. The
  fusion only interleaves outputs, so no output's association changed; the tests above are
  what establishes it rather than an aggregate.

## 8. Thread scaling

Same campaign data, 1T vs 4T, after: f64 fwd 21.31 -> 5.72 ms (3.7x), f64 VJP 63.49 ->
15.48 (4.1x), f32 fwd 12.46 -> 2.98 (4.2x), f32 VJP 40.72 -> 8.98 (4.5x); at 1024 atoms
f64 fwd 168.60 -> 43.22 (3.9x), f64 VJP 504.95 -> 129.21 (3.9x), f32 fwd 120.95 -> 30.18
(4.0x), f32 VJP 331.32 -> 84.05 (3.9x). Efficiency is 0.93-1.13 of linear, i.e. the
executor is not the limit; the fused plan's task list is unchanged in shape from the split
plans', so the scaling did not move.

## 9. Review record

* Plan gate: `plan-review: glm-5.3-flash APPROVE` (`ollama-cloud/glm-5.3-flash:high`,
  before any edit). Its two IMPORTANTs are in the change: `check_call` rejects a
  same-windows/different-`perm` `RunIndex` with `PlanLayoutMismatch`, and the
  fused-vs-split bitwise test covers `execute_acc` as well as `execute_into`.
* Post-implementation review: `review: gpt-6.1-sol`, read-only process on the committed
  diff (`git diff origin/main...HEAD`), no access to this session. Round 1 returned
  `VERDICT: CHANGES` with two IMPORTANT and five MINOR findings and six deletions. All
  eight findings were checked against the source; all were real and are fixed here:

| finding | disposition |
| --- | --- |
| IMPORTANT: the new `PlanLayoutMismatch` had no regression test -- the neighbouring tests vary the run *windows*, not the row order | real; `a_scatter_call_must_carry_the_compiled_run_order` compiles `[0,1,0,1]`, calls with `[1,0,1,0]` (same `ptr`, different `perm`) at both entry points, asserts the typed error and that nothing was written, then runs the compiled order successfully; `scatter_outputs_with_different_run_orders_get_their_own_group` pins that the two orders get separate groups (`groups == 2`, `units == 4`, `tasks == 4`). Each fails on the un-fixed code it pins: dropping `perm` from the grouping key gives one group, and removing the `check_call` guard lets the wrong order through |
| IMPORTANT: the budget/thread-invariance case had two tasks, so `threads = 4` never forked four children, and its accumulating run started from zero | real; the test now sweeps two shapes -- 300 rows in one run (two block tasks) and 600 rows over two sources (two 300-row runs, four block tasks) -- and seeds the `execute_acc` destinations with a non-zero payload. Round 3 below removed the width assertion it first carried; the four-task property is pinned in `src/plan/tests.rs` instead |
| MINOR: the SIMD arithmetic-order test used dyadic values, which an FMA or a re-association would reproduce exactly, and it never used a non-zero offset | real; `a_row_visits_every_segment_and_path_at_its_own_offset` adds 2 segments x 5 paths at distinct non-zero offsets with non-dyadic payloads and coefficients, compared by bit pattern. It fails when the kernel's offset stride is mutated from `local * (N + 1)` to `local * N` |
| MINOR: `assert_eq!` on floats is numerical equality (`-0.0 == 0.0`), not the bit identity the report claims | real; `assert_same_bits` in `tests/kernels.rs` and in the harness's `fidelity.rs` compares `to_bits()`, so the claim is now literal |
| MINOR: `check_report_498.py` advertised buffer and RSS checks it did not make, and did not require every table row | real; it now requires the complete 16-cell key set, parses and verifies §6's buffer table from the records, checks the "peak RSS within 1 MiB" claim against both campaigns, and its docstring lists exactly what it checks. Checked against a corrupted median, a deleted row and a corrupted buffer row: all three are refused |
| MINOR: §1 understated the plan's own scratch eightfold (2944 is the budget-1 figure) | fixed; §1 now says 23 552 elements for eight workers (92 KiB f32 / 184 KiB f64), and §2.3/§6 agree |
| MINOR: the baseline did not ignore the call's permutation -- it stored and traversed `runs.perm()` | real and important for a reader: §2.1 now says the replay used to follow the *call's* order (so the sum was a function of the call, not of the plan), and that grouping is what forces the compiled order. §7's determinism paragraph matches |
| MINOR: `OpRow` documented `arity` offsets per path while the stride is `arity + 1` (the output's entry is last) | fixed; the field doc states the `arity + 1` layout, the inputs-first order and the stride |

Deletions from the same report, all done: the duplicated opening paragraph of
`scratch_len`'s rustdoc; the second owned permutation copy in the compile-time
`topologies` (it now holds the run offsets only and compares `Group::perm`, mutation-checked
above); the identity path map in `scatter_outputs_with_different_topologies_keep_their_own_order`;
the stale `lane_axpy`/release-no-op paragraph on `Sealed`; the stale "the dispatch itself
stays per path" comment in `accumulate_output`; and the placeholder in this section.

**Round 2** (same model, on the fixed diff): **`VERDICT: APPROVE`**. Its two MINORs were
errors in *this report*, both fixed here: the 600-row/two-source shape is two 300-row runs
with four block tasks, not 300 one-row runs, and the mutation sentence above conflated the
two regression tests (the grouping key is pinned by the two-group test, the replay guard by
the rejection test -- each was mutated separately, and each fails only on its own). Its
"not verifiable by reading" note that the packed case ran only with block-partial
accumulators was closed by one more call of the same helper: `5, 3, 2, 3, true` puts packed
inputs and non-contiguous accumulators in a `Child` slot, with no source change.

**Round 3** (hosted CI, not a review): the width assertion added above
(`tree_width(...).get() == budget`) failed on `macos-latest` with `left = 2`. The value is
the executor's, not the plan's: a baseline `aarch64` host's in-pool dispatch cost does not
amortise four children over this plan's 9 600 macs, and a pool smaller than the budget caps
it too. Relaxed to no assertion at all -- the bit-identity body runs at whatever width the
host grants, and the plan's own task count stays pinned white-box. (The older
`budget_split_is_bitwise_for_a_batched_and_a_shared_output` keeps its exact assertion: its
plan carries more than 65 536 macs, which opens the gate to four on every class.)

So: `review: gpt-6.1-sol APPROVE` after one `CHANGES` round, and the plan gate
`ollama-cloud/glm-5.3-flash APPROVE` after one `CHANGES` round.

## 10. Not changed

The other plans' shapes, `BLOCK_ROWS`, the public API (`Group`, `AccOut`, `OpRow` and the
`Sealed::op_row_lanes` method are private), the descriptor format, the docs' summation
guarantees, and operations with four or more operands, which still take the scalar loop
(`shortcut:` in `plan/exec.rs`). `strided-segpoly` was not the only crate with per-path
dispatches to consider: the other runtime-dispatched bodies listed in
`docs/design/runtime-simd-dispatch.md` were already dispatched once per leaf call.
