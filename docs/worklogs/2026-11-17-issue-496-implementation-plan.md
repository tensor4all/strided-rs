# #496 — segpoly f32 SIMD, the `bar_edge_attrs` reduction, and 4T scaling

Worktree: `~/work/strided-491-ab/strided-496` (branch `perf/496-segpoly-simd-bar-edge`,
base `origin/main` = `e9e6abe`, PR #493 merged). Issue:
[tensor4all/strided-rs#496](https://github.com/tensor4all/strided-rs/issues/496).
Measurement harness: `benchmarks/strided-491-mace-ab` (untouched protocol from #493:
`campaign.sh`, `OUT=` selects the raw dir, pins `4` / `4,5,6,7`, observer
`8-15,24-31`, anchor drift guard ≤ 15 %).

## 0. What the code does today (read, not assumed)

| file | current behaviour |
| --- | --- |
| `strided-segpoly/src/plan.rs` (`compile`) | Takes the largest last-axis extent as `u`; **rejects** an output whose lane extent is not `u` (`SegPolyError::OutputLaneBroadcast`, `plan.rs:273-279`). `OpPlan::unit_lanes` is set only when **every** input's lane step is 1 (`plan.rs:665-667`); `in_lane` is 0 for a unit-lane buffer, the compiled stride otherwise (`plan.rs:656-664`). `tile_elems` = max over outputs of `segments*u` (`plan.rs:647-650`). `scratch_len(budget)` = `(block_slots + budget.max(1)) * tile_elems` (`plan.rs:719-723`). |
| `strided-segpoly/src/plan/exec.rs:accumulate` | The general lane loop (`exec.rs:534-633`). Lane index arithmetic and loads run through runtime `steps` (`p = p * *ptr.offset(lane*step)`), scalar, with a 64-lane stack tile and an unrolled arity-1..3 macro. **No pulp dispatch anywhere in the crate**; `strided-segpoly/Cargo.toml` has no pulp dependency. |
| `strided-segpoly/src/plan/exec.rs:op_row_contiguous` | Gated by `if op.unit_lanes && (1..=3).contains(&op.inputs.len())` (`exec.rs:397`). Zipped slices, auto-vectorised at the **build target** (x86-64 baseline = SSE2), no dispatch. |
| `strided-segpoly/src/plan/exec.rs:execute` | `child_tiles = scratch.len() / tile_elems - block_slots` (`exec.rs:115`); a child's accumulator is `scratch + (block_slots + child) * tile_elems` (`exec.rs:341-343`); `combine` reads block partials at `slot * tile_elems` (`exec.rs:484-515`). |
| `benchmarks/strided-491-mace-ab/src/cg.rs:strided_forward` | op0 = (`x0e` lane stride 1, `edge_attrs` **lane extent 1**, `tp_weights` stride 1), op1 = (`x1o` **lane stride 3**, `edge_attrs` lane 1, `tp_weights` 1). So **neither** op has `unit_lanes`; both take the scalar `accumulate`. |
| `benchmarks/strided-491-mace-ab/src/cg.rs:strided_vjp_y` | `u = 1`: `linear_up [n,512,1]`, `tp_weights [e,1280,1]`, `bar_scatter_major [n,5120,1]` are all **segments**, `grad_y [e,16,1]`; 11 008 paths (`86 x 128`), one unit per edge. |

Consequences that the plan is built on:

1. `accumulate` is scalar for both dtypes, so f32 buys only bytes (measured 1.05x), while
   f32 SIMD would buy 2x lanes. Fixing the *dispatch* is the target, not the dtype.
2. `bar_y`'s 11 008 scalar paths do 64.8 M lane-triples (the same count as the forward,
   which costs 68.7 ms) in 426 ms, i.e. ~20 cycles/path: the plan is issue/latency bound
   and the feature axis lives in the *segment* axis, so no SIMD width is reachable.

## 1. Changes

### 1.1 New `strided-segpoly/src/simd.rs` (runtime dispatch, the portable lane kernel)

`acc[i] = acc[i] + ((c * a0[i]) * a1[i]) * a2[i]` for `i in 0..u`, i.e. exactly the
scalar association the crate has today, one lane at a time (no reassociation, so the
result is **bit-identical** to the current scalar loop).

- `pub(crate) fn lane_axpy_f64<const N: usize, const SLICES: u32>(acc: &mut [f64], c: f64, operands: [Input<'_, f64>; N])`
  and the `f32` twin; `Input::Slice(&[f64])` (lane-contiguous, length `u`) or
  `Input::Splat(f64)` (a unit-lane operand's value for this path). The `WithSimd` bodies
  are named `AxpyF64` / `AxpyF32` (the checker matches struct names, so the `CHECKS`
  fragment in §1.5 is `strided_segpoly::simd::AxpyF64`).
- Each runs inside `pulp::Arch::new().dispatch(...)` with a `#[inline(always)]`
  `with_simd` (the `strided-nn/src/simd.rs` pattern, which the dispatch doc names as the
  way to keep the body inside pulp's target-feature function). `SLICES` is a const mask
  so the splat/slice choice folds away per instantiation instead of branching per lane.
- Vector loop uses `S::as_mut_simd_f64s` / `S::as_simd_f64s` + `simd.mul_f64s`,
  `simd.add_f64s`, `simd.splat_f64s`; the remainder is a scalar tail in source order.
  **No `mul_add`**: `c*a*b` then `+` keeps the two roundings the scalar code has.
- Width comes from `S::VECTOR_LEN` (pulp), never a literal. pulp's scalar arch has
  `VECTOR_LEN == 1`, so the same code is the scalar fallback; arity 0 and arity > 3 keep
  `accumulate` as today.
- `strided-segpoly/Cargo.toml` gains `pulp.workspace = true` (`0.22`, `x86-v4`).
- `pub fn lane_width<T: SegPolyElem>() -> usize` (one line, `Arch::new().dispatch` +
  `S::VECTOR_LEN`) so the harness and the worklog can *report the width each path
  selected* instead of asserting one.

### 1.2 `plan.rs` — lane packing, and the lane-reduction output rule

- **Packing (fixes the lowering).** For each operation and input, if `in_lane[input] != 1`
  (lane stride 3 for `x1o`, 0 for a unit-lane broadcast) the operand is *packed*: before
  the op's path loop the replay copies its `[segments, u]` row-block into a contiguous
  `[segments, u]` tile in caller scratch, and **that op's** `path_offsets` entries for the
  operand are rewritten at compile time from `segment_id * segment_stride` to
  `segment_id * u`, so the replay's addressing is unchanged (tile base instead of buffer
  base). This is the lowering `docs`/issue #479 already specified ("the `u` axis must be
  unit-stride for the fast path; otherwise tiles are packed into caller scratch").
  **The pack condition is `in_lane[input] ∉ {0, 1}`.** A unit-lane operand (`in_lane == 0`)
  is *not* packed: it becomes `Input::Splat` at the call site, which is bit-identical to
  today's step-0 read (`exec.rs:543-547`) and costs no tile. Packing a unit-lane buffer
  would leave lanes 1..u of the tile uninitialised — that is not what is specified.
  `OpPlan` gains `pack: Vec<u32>` and `pack_elems: usize` (sum over that op's packed
  inputs of `dims[buffer][1] * u`); the plan gains `pack_elems` = max over ops.
- **Scratch layout (the existing addressing has to be stated, not implied).** Block
  partials stay first and unchanged (`slot * tile_elems`), so `combine` is untouched; then
  one region per child of `tile_elems + pack_elems`:

  ```
  scratch = [ block_slots x tile_elems ][ child 0: tile | pack ][ child 1: tile | pack ] ...
  child_tiles = (scratch.len() - block_slots * tile_elems) / (tile_elems + pack_elems)
  child tile  = block_slots * tile_elems + child * (tile_elems + pack_elems)
  child pack  = child tile + tile_elems
  scratch_len(budget) = block_slots * tile_elems
                      + budget.max(1) * (tile_elems + pack_elems)
  ```

  With `pack_elems == 0` every expression above is **exactly** today's
  (`scratch_len`, `child_tiles`, the Child address), so existing callers and the
  zero-allocation gates are unaffected. `exec.rs:115` and the Child address at
  `exec.rs:341-343` (and its `SAFETY` comment at `:345`) are updated together, and `width`
  keeps being derived from `child_tiles`, so no child can address past its own region —
  the failure mode the naive `+ budget * pack_elems` formula would have introduced.
  The scratch floor (and `ScratchTooSmall.need`) becomes
  `block_slots * tile_elems + (tile_elems + pack_elems)`.
- **Lane reduction (target 2).** An output with lane extent 1 while `u > 1` is now
  accepted and means `out[segment] = Σ_lane acc[segment][lane]`, summed in increasing
  lane order, once per destination block, after the block partials are combined
  left to right. Precedent: D4 already defines a `Shared` *output* as a batch reduction,
  the exact transpose of a shared input. `out_direct` is forced `false` for such an
  output (the accumulator is `[segments, u]`, the destination row is `[segments]`).
  This changes an error (`OutputLaneBroadcast`) into a meaning; no working call changes.
- Docs: the "Summation order" module comment gains rule 4 (lane reduction) and the
  packing note; `OutputLaneBroadcast` keeps its variant, retargeted to "neither 1 nor u".

### 1.3 `plan/exec.rs`

- `run_task`: after the row's bases are resolved, fill the op's pack tiles (a strided
  copy, `segments*u` elements per packed operand per row) and pass `Input::Slice` for
  contiguous/packed operands, `Input::Splat` for unit-lane ones. `op_row_contiguous`
  (renamed `op_row_lanes`) calls `lane_axpy` once per path instead of the zipped scalar
  loop. `unit_lanes` becomes "arity ≤ 3" (packing removes the stride restriction, splats
  cover the unit lane).
- **Checked arithmetic.** The rewritten offset `segment_id * u`, the tile sizes
  `dims[buffer][1] * u`, `pack_elems` and the per-op pack cursors follow the crate's
  existing convention (`checked_mul`/`checked_add`/`u32::try_from` →
  `StridedError::OffsetOverflow`, as at `plan.rs:449-452`, `:517-521`, `zero_u32`); a
  packed operand's per-segment reach can be `u` times its old one, so this is not
  theoretical.
- `read_block` / `write_block`: a `BlockLayout { Direct, Strided, ReduceLane }` argument
  replaces `direct: bool`. `ReduceLane` seeds lane 0 with the destination value
  (`execute_acc`, so the seed is the first term) and writes `Σ_lane` on the way out;
  `combine` reduces after summing the partials lane-wise.
- No new allocation anywhere: the tiles live in the caller's `scratch`.

### 1.4 Harness (`benchmarks/strided-491-mace-ab`) — measurement only, no protocol change

- `strided_vjp_y` builds the `u = 128` plan: buffers `x0e [n,1,128]`, `x1o [n,3,128]`
  (strides `[4f,1,3]`, exactly the forward's views), `tp_weights [e,10,128]`,
  `bar_scatter_major [n,40,128]`, output `grad_y [e,16,1]`; 86 paths, one op per x view
  (`vy_ops` grows from one op to two, `vy_indexes` from one gather to two
  `Gather(&senders)`). Same call site, same scratch mechanism
  (`plan.scratch_len(threads)`), same buffers.
- `build_paths(Kind::VjpY)` changes shape: it stops looping `for feature in 0..f` and emits
  the 86 per-block CG paths with the feature removed from every segment id — x segment
  `0` (the `x0e` view) or `i_m` (the `x1o` view), weight segment = block index, message
  segment = `out_base/f + k_m`, edge-attr segment = `y_base + j_m` unchanged. Every other
  `Kind` is left alone.
- `main.rs` record gains `"simd":{"f32_lanes":L,"f64_lanes":L}` from `lane_width`.
- The three other plans and every buffer/layout/view stay byte-identical, so the
  A/B compares the same shapes before and after.

### 1.5 Evidence artifacts

`raw/496-before/` and `raw/496-after/` (campaign transcripts, `.txt`), the per-cell JSON
records, `docs/worklogs/2026-11-2x-issue-496-segpoly-simd-reduce.md` (before/after per
cell, allocations, RSS, widths, scaling), `check_report_496.py` (mechanical claim check,
in the spirit of #493's `check_report.py`), CHANGELOG entry, README (SIMD section, the
lane-reduction rule, the divergence from cuEquivariance's `naive`), and a
`strided-segpoly/examples/simd_probe.rs` so the new dispatched body is proven on `zmm`
(V4) / `ymm` (V3) rather than asserted. `scripts/check-runtime-simd-dispatch.py` builds a
**second** probe (`PROBES = [("strided-kernel", "runtime_simd_probe"),
("strided-segpoly", "simd_probe")]`, listings concatenated) and gains a `CHECKS` entry
`("strided_segpoly::simd::AxpyF64", "vmulpd")` — the existing matcher already keys on the
mangled `::{level}>::vectorize` instantiation of a `WithSimd` body
(`scripts/check-runtime-simd-dispatch.py:69-88`), which is exactly how
`strided_nn::simd::ShiftGroup>` is found today. The probe reaches the kernel through the
**public** API (compile a 4-lane, 3-operand plan and run it), so no `#[doc(hidden)]`
entry is added.

## 2. Verify

```sh
# unit + integration
cargo test -p strided-segpoly                       # kernels, contract, fixtures, alloc
cargo test --workspace                              # pre-push gate
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --release -p mace-ab --manifest-path benchmarks/strided-491-mace-ab/Cargo.toml

# codegen: the dispatched lane kernel reaches AVX2/AVX-512 in a baseline build
python3 scripts/check-runtime-simd-dispatch.py
objdump -d --no-show-raw-insn -C <probe> | grep -A30 'lane_axpy'

# measurement: before (this tree, HEAD = origin/main) and after (this branch)
cd benchmarks/strided-491-mace-ab
cargo build --release
OUT=raw/496-before ./campaign.sh                     # 18 records + drift guard
OUT=raw/496-after  ./campaign.sh
grep -c '^{' raw/496-before/campaign.txt raw/496-after/campaign.txt   # 18 + 18 (16 cells + 2 anchors)
grep -c FAILED raw/496-before/campaign.txt raw/496-after/campaign.txt # 0 + 0
./check_report_496.py                                 # every quoted number vs raw/
```

### Acceptance bars

| # | bar |
| --- | --- |
| A | `strided-segpoly` + workspace tests green; `cargo test --release` in the harness green (fidelity, alloc, thread-invariance). |
| B | **Numerics unchanged where they can be:** every existing segpoly test that pins bits/values still passes; the harness f64/f32 agreement vs llmlip stays inside the #493 tolerance bands (`f64` ≤ 3.6e-15 of scale; `f32` ≤ 1e-5), and 1T vs 4T stays **bit-identical** in every cell (existing gate). `bar_y`'s association changes (lane-reduced); the new value is reported and the gate updated *with* the measured number, not loosened arbitrarily. |
| C | **f32 forward:** f32 1T cells go from 1.88x llmlip to **≤ 1.15x**, and f32 ≤ 0.6x of the f64 wall time in the same cell (llmlip's own f32/f64 ratio is 0.45). |
| D | **f64 not regressed:** every f64 forward/VJP cell stays within +3 % of its before number or improves. |
| E | **`bar_edge_attrs`:** 1T f64 ≤ 142 ms (≥ 3x better than 426), 1T f32 ≤ 140 ms (≥ 3x better than 418); VJP-sum ratio 1T f64 ≤ 1.6x (from 2.87x), f32 ≤ 2.5x (from 11.78x). |
| F | **Allocations:** `execute_*` still 0 bytes / 0 calls per call (existing gate), scratch bounded — report `scratch_len(1)` / `scratch_len(4)` per plan before and after. |
| G | **Determinism:** bits identical across budgets 1/2/4/8 (new test), across pools (existing gate), and identical for the reduce plan at any `Exec` budget. |
| H | **Runtime width reported, nothing hardcoded:** the record names the width pulp selected for f32 and f64; the source contains no lane count, core count or cache size; `check-runtime-simd-dispatch.py` proves V3/V4 for the new body. |
| I | **4T scaling reported per cell** (1T/4T ratio) plus the 1T/2T/4T/8T probe; 4T ≥ 3.8x must hold (no regression) and the absolute 4T numbers must improve. |

### 4T target — measured before planning, so the bar is honest

`cg-fwd f64 1024 atoms`, 7 samples, this host, `taskset` per cell:

| threads | cpus | strided | llmlip | strided 1T/nT |
| --- | --- | --- | --- | --- |
| 1 | 4 | 560.2 ms | 635.9 ms | 1.00x |
| 2 | 4,5 | 295.6 ms | 188.6 ms | 1.89x |
| 4 | 4,5,6,7 | 142.5 ms | 86.2 ms | 3.93x |
| 8 | 4,5,6,7,20-23 | 120.1 ms | 81.9 ms | 4.66x |

The executor already splits **rows within runs** (one task family per destination run,
partitioned by cumulative edge count, `exec.rs:task_range`/`task_boundary`), so the
128-atom cells are not node-granular: 3.93x of 4 threads is linear, with the remaining
0.07 standing for the tail of one unit. Two consequences:

- `llmlip`'s 7.4x is not evidence that 6x is reachable at 4 threads on 4 physical cores;
  it is `llmlip`'s **1T** path being ~1.85x off its own per-core rate (per-core rate
  ratio llmlip 1T:2T:4T = 1.00 : 1.69 : 1.85, strided = 1.00 : 0.95 : 0.98), and
  `/usr/bin/time` on the 4T cell confirms llmlip holds 4 threads (user/wall 2.75 over a
  process that is serial outside the timed closures).
- `≥ 6x of our own 1T` therefore requires 1.5x more work per core at 4T than at 1T,
  which no decomposition can supply on 4 physical cores. The attainable 4T target is
  **absolute**: beat llmlip's 4T wall time (that is the number the issue compares), which
  the kernel work of §1.1-1.2 delivers on both thread counts, and keep the ratio ≥ 3.8x.

I will still re-measure 1T/2T/4T/8T **after** the kernel change and, if it exposes a 4T
tail (ratio < 3.8x or a per-core rate that drifts with n), fix the decomposition with
bounded chunk tasks then, with that measurement as the justification. Pre-emptively
rewriting a decomposition that measures linear would trade a proven property
(deterministic per-run order, no atomics) for no measured gain.

## 3. Edge cases, compatibility, and what I am deliberately not changing

- **Zero-length lanes/segments**, empty runs, `batch = 0`: unchanged paths; the pack fill
  is skipped when `segments*u == 0` and `ReduceLane` writes nothing for `segments == 0`.
- **`execute_acc` on a reduce output**: the destination is the first term (lane 0 of the
  seed), which is the same rule block zero already has.
- **Multi-block (`Shared`) reduce output**: partials are summed lane-wise, then reduced
  once per segment, in a fixed order; new test pins bits across budgets.
- **Aliasing**: packing *reads* an input into scratch before any write to the same row, so
  the existing rule (inputs may alias each other; an output aliases nothing) is unchanged.
- **`ScratchTooSmall`**: the floor moves from `(block_slots+1)*tile_elems` to include the
  pack area; the error's `need` reports the new floor, and a test asserts a caller who
  allocates the old floor gets the error rather than an overrun.
- **Compatibility**: `Spec`, `Layout`, `IndexMode`, `execute_into`/`execute_acc`
  signatures unchanged; `scratch_len` returns a larger value (identical when no operand is
  packed); `OutputLaneBroadcast` becomes reachable only for lanes that are neither 1 nor
  `u` — and for outputs the acceptance rule now degenerates to the same `{1, u}` rule as
  inputs, so the doc comment says that it is the *output lane* check that fires. `pulp` is
  a new direct dependency of `strided-segpoly` (already in the workspace, already used by
  `strided-basic`/`strided-nn`).
- **Two existing pinned tests change meaning and are rewritten, not deleted:**
  `plan/tests.rs:71-78` asserts `compile` errors with `OutputLaneBroadcast { lane: 1,
  u: 3 }` — that input becomes a valid lane reduction, so the test keeps the same
  dims/strides but must gain one operation and one path (the round-1 fixture is
  `spec(&[], &[], ...)`, `tests.rs:66-70`, which cannot compile at all: `compile` rejects
  it with `UnusedBuffer`) and then asserts the reduced value, plus a separate case for a
  lane that is neither 1 nor `u`; `plan/tests.rs:615-663`
  (`rejects_scratch_below_the_block_floor`) keeps its intent but its floor expression
  moves to `block_slots * tile_elems + (tile_elems + pack_elems)`, and a companion case
  proves a caller at the *old* floor is rejected instead of overrunning.
- **Not changing**: the four plans' shapes/layouts other than `vjp_y`; `BLOCK_ROWS`
  (its value is part of the result, `plan.rs:57-64`); the harness protocol, pins, sample
  count, target time, anchor guard and `check_report.py` (the #491 report keeps matching
  its own `raw/`); no CUDA, no descriptor crate, no `-C target-cpu=native`.

## 4. Risks and the rejected alternative

| risk | mitigation |
| --- | --- |
| Bit drift in f64/f32 from the vector kernel | Per-lane ops only, same left-to-right association, no FMA: the kernel must be *bit-equal* to `accumulate`. New test: random plans, SIMD path vs a forced scalar `Exec` path, exact equality. |
| The reduce output silently changing what a lane-1 output means | It was a hard error before, so nothing that worked changes; documented in plan docs, README "Divergences", and a CHANGELOG entry. |
| Packing cost dominating at small `u` / single path | Cost is `segments*u` copies per (row, op) against `paths*u` MACs; stated in the design doc, measured on the `chain-8k`/`small-128` bench rows already in the crate. |
| f32 numerics: `bar_y`'s 11 008-term error | The reduce form has *fewer* roundings (86 lane accumulations + one 128-lane reduction); the harness gate records the measured value. |
| Scratch growth | Bounded by one packed operand block per child (`pack_elems`), reported per plan; the "no allocation in the timed call" contract is untouched. |

**Rejected alternative (target 2):** have the *harness* call the existing `u = 128` plan
into an `[E,16,128]` intermediate and reduce it in a caller loop. Rejected because it is
92 MiB / 46 MiB of output at 128 atoms (736/368 MiB at 1024) — the exact intermediate the
issue forbids — it can only be bounded by chunking the *topology*, which means one
compiled plan per chunk, and it measures the harness rather than the executor.

**Rejected alternative (target 1):** pulp `gather_or_default` for the lane-stride-3
operand instead of packing. A stride-3 gather is ~break-even against 8 scalar loads on
AVX2/AVX-512 (`vgatherqpd` throughput), needs a per-`(u, step)` index vector in the plan,
and `strided-basic`'s house pattern is contiguous-only SIMD with a scalar/memcpy
fallback (`simd.rs:try_mul_contiguous`). Packing is a strided copy that runs once per row
instead of once per path.

**Rejected alternative (4T):** a work-stealing edge queue. It would trade the plan's
documented bit-stability (fixed per-run order, no atomics) for parallelism that the
measurement above says is already there.

## 5. Order of work

1. `simd.rs` + trait hook + unit test (SIMD == scalar, bit-exact).
2. Packing in `plan.rs`/`exec.rs` + `scratch_len` + tests.
3. Reduce outputs + `BlockLayout` + tests.
4. Harness `vjp_y` plan + `lane_width` in the record; `cargo test --release`.
5. `after` campaign, `before` campaign already taken; worklog + `check_report_496.py`.
6. fmt/clippy/workspace tests, `check-runtime-simd-dispatch.py`, probe, PR.

## 6. Reviewer record (plan gate)

- Reviewer: `ollama-cloud/glm-5.3-flash:high` (agent `reviewer-glm`, read-only pi subagent),
  task = this file + the repository. The plan gate names GLM-5.3-flash; this is the local
  roster's exact id.
- **Round 1: `VERDICT: CHANGES`.** Disposition of every finding:

| finding | disposition |
| --- | --- |
| BLOCKER 1 — the `+ budget * pack_elems` scratch formula leaves `child_tiles` (`exec.rs:115`) and the Child address (`exec.rs:341-343`) wrong, so a child could write into another child's pack area | Accepted. §1.2 now specifies the layout explicitly (`[block partials][per-child tile\|pack]`), with `child_tiles`, the Child address, the `SAFETY` comment and the floor all updated together; identical to today when `pack_elems == 0`. |
| IMPORTANT 2 — a segpoly probe could not be reached by the checker (`build_probe` builds only `strided-kernel`'s example) and a `pub(crate)` kernel is unreachable from `examples/` | Accepted. §1.5 now extends `build_probe` to two (package, example) pairs with concatenated listings, adds a `CHECKS` entry keyed on the mangled `::V4>::vectorize` instantiation, and has the probe go through the public plan API. |
| IMPORTANT 3 — §1.2 said to pack unit-lane operands while §1.3 said to splat them; a packed unit lane would leave lanes 1..u uninitialised | Accepted. Pack condition is now `in_lane ∉ {0, 1}`; unit-lane operands are `Input::Splat`. |
| IMPORTANT 4 — the rewritten offsets and pack sizes can overflow and must follow the crate's checked convention | Accepted. §1.3 has a dedicated item. |
| IMPORTANT 5 — two existing pinned tests change meaning (`plan/tests.rs:71-78`, `:615-663`) | Accepted. §3 names both and how each is rewritten. |
| MINOR 6 — stale line citations in §0 | Accepted and fixed. |
| MINOR 7 — §1.4 under-specified the harness rewrite | Accepted. §1.4 now names `build_paths(Kind::VjpY)`, `vy_ops` and `vy_indexes`. |
| MINOR 8 — which check rejects a lane neither 1 nor `u` | Accepted. §3 says it and the doc comment will. |
| MINOR 9 — "18 records should be 20" | **Not accepted; the review is wrong here.** `campaign.sh` loops 2 atoms x 2 dtypes x 2 thread counts x 2 cells = 16 cells plus 2 anchors = 18 records; the committed #493 transcript confirms it: `grep -c '^{' benchmarks/strided-491-mace-ab/raw/campaign.txt` = **18**. |

- **Round 2: `VERDICT: APPROVE`.** Three MINOR findings, all folded in above and here:
  the rewrite of `plan/tests.rs:71-78` must add an operation/path (its round-1 fixture has
  none and cannot compile); the `WithSimd` bodies are named `AxpyF64`/`AxpyF32` so the
  `CHECKS` fragment matches; §0's `op_row_contiguous` citation points at the gate
  (`exec.rs:397`), not the definition. The reviewer confirmed the revised scratch layout,
  the pack condition, the offset rewrite, the probe/checker extension and the harness
  rewrite, and agreed that rejecting the round-1 record-count minor was correct.

## 7. Outcome after implementation (deviations from the approved plan)

What was built matches §1 with these deltas, each with its evidence:

1. **Bar C missed, reported not renegotiated.** Plan §2 asked "f32 <= 0.6x of the same cell's
   f64 wall time". Measured with the kernel in place: 0.75 (1T/128), 0.81 (1T/1024),
   0.62 (4T/128), 0.78 (4T/1024), against llmlip's 0.45/0.44/0.86/0.82. The target the issue
   actually asks for is met -- f32 cells are 1.4-1.8x *ahead* of llmlip where they were
   1.7-1.9x behind -- so the miss is reported in the worklog §4 as a miss with the residual
   (dtype-independent per-path/per-row work, not profiled) rather than by changing the bar.
2. **Bar I (4T >= 3.8x of our own 1T) missed at 3.25-3.64x, and shown not to be a
   decomposition property.** Plan §2's "4T target" already established that >= 6x is
   unreachable on 4 physical cores. A 1/2/3/4T sweep (`raw/496-scaling/`) now shows 0.85 /
   0.83 / 0.89 marginal efficiency -- a 4-thread all-core clock, not a tail -- and llmlip's
   7.3x is a 1T that is 2.75x slower than ours. Decision recorded in the worklog §7: no
   change to the decomposition, report the absolute 4T numbers (the strided side wins every
   cell) and the scaling evidence.
3. **#491's allocation story was wrong and is corrected.** #491 read the two 1 520-byte events
   as per-worker first touch. The new records show up to 15 events in 960 calls of one 4T
   cell, and a backtrace probe on the counting allocator attributed every event to
   `rayon_core::registry::WorkerThread::from` / `set_global_registry`, i.e. thread start-up
   in the pools, charged to our window because the counter is process-wide. Worklog §6 says
   that, with the probe described and reverted.
4. **Clippy.** `op_row_lanes` takes 8 arguments: `#[allow(clippy::too_many_arguments)]` with a
   one-line reason, matching `strided-basic/src/copy_plan.rs`. The `assign_op_pattern`
   suggestions were applied (`*=`/`+=`). Four `&mut *…borrow_mut()` in the harness were
   reduced to clippy's form; the harness's three remaining findings (`type_complexity`
   `cg.rs:55`, `manual_is_multiple_of` `measure.rs:203`, a redundant `must_use` `lib.rs:133`)
   are pre-existing lines this change does not touch, in a crate CI does not build.
5. **Tests added beyond the plan's list**, all in the plan's spirit: `a_lane_unit_output_sums_
   its_lanes_in_order`, `a_packed_lane_matches_the_scalar_reference_bit_for_bit` (lane strides
   1, 2, 3, 5, -1, -2), `a_lane_reduction_is_budget_invariant`, `scratch_reserves_a_pack_area_
   per_worker`, and the harness's `tests/scratch.rs` (the four plans' `scratch_len` at budget 1
   and 8, printed for the worklog).

## 8. Post-implementation review (independent, finished diff)

- Reviewer: `openai/gpt-6.1-sol` via `scripts/pi_review.sh` (fresh read-only pi process,
  `--tools read,grep,find,ls`), request in `/tmp/independent-review.md`, report in
  `/tmp/review-496/post1.out`. **Round 1: `VERDICT: CHANGES`.**
- Disposition of every finding:

| finding | disposition |
| --- | --- |
| BLOCKER 1 — packing eligibility and replay disagree for a stride-0 lane (compiled as packed, replay reads it in place) and for a `u = 1` plan with an odd lane stride | Accepted; real. `splat[buffer] = lane extent == 1 || lane stride == 0` is now computed once and `packed = !splat && stride != 1`, so the three readings (slice, splat, pack) are decided in one place and the replay agrees. Regression tests: `a_stride_zero_lane_is_a_broadcast_not_a_pack`, `a_unit_lane_plan_keeps_the_buffer_strides`. |
| BLOCKER 2 — offsets are rewritten into tile space for every operation, but only arity ≤ 3 packs, so an arity ≥ 4 fallback read tile offsets from the buffer | Accepted; real. `tiled = lanes && packed[buffer]`, so only an operation the lane kernel runs has tile-space offsets, and `pack_elems` is gated the same way. Regression test: `a_deep_operation_reads_the_buffer_offsets`. |
| BLOCKER 3 — a pathless operation offsets a dangling empty-input pointer before discovering it has no paths | Accepted (pre-existing in `op_row_contiguous`, still wrong). `if op.num_paths() == 0 { continue; }` before any base is resolved; zeroing still happens because the block was seeded already. |
| IMPORTANT 4 — the new reduction tests do not establish the claimed association, budget invariance, or vector coverage: the reference summed lanes inside a path, `budget` only changed the scratch length, and 5 lanes never reached the vector loop | Accepted. The reference now accumulates **paths inside a lane, then lanes in increasing order**, with cancellation-sensitive `±1e16` values; the cases run 19 lanes (vector + remainder) and `2 * BLOCK_ROWS + 5` rows (block partials combined first); a pooled 2- and 4-thread × budget 1/2/4 bit-equality test with four block partials is added to `tests/kernels.rs`; `a_seeded_lane_reduction_adds_the_seed_once` covers `execute_acc`. |
| IMPORTANT 5 — "wins every 4T cell" contradicts the records: the f32 4T VJP sums are 1.64x and 1.12x *slower* than llmlip | Accepted. Worklog §7 and the PR body now say **forward** cells, and name the two f32 VJP cells with their numbers. |
| IMPORTANT 6 — unchanged rounded error statistics do not prove bit-identical results | Accepted. Worklog §8 now rests the exact claim on the element-wise `assert_eq!` tests (which also cover stride, lane count and block size) and labels the campaign figures as corroborating aggregates. |
| MINOR 7 — `OutputLaneBroadcast`'s documentation still forbids lane-extent-1 outputs | Accepted; doc retargeted to `{1, u}` and says extent 1 is a lane reduction. |
| delete 1 — the second mask-selection layer in `op_row_lanes` | Accepted. `lane_axpy` takes a runtime `slices: u32` and `simd::axpy*` does the single const-mask match. |
| delete 2 — the public `SegPolyElem` dispatch hooks; a caller can invoke `lane_axpy::<0, 0>` (panics) | Accepted. `lane_axpy` / `vector_lanes` moved onto the private `sealed::Sealed` trait, so they are reachable only in-crate and `lane_width` still works through the supertrait. |
| delete 3 — `Box::leak` in the probe example | Accepted; the segment arrays are locals borrowed by the paths. |

After the fixes: `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D
warnings`, `cargo fmt --all -- --check`, `RUSTDOCFLAGS="-D warnings" cargo doc --workspace
--no-deps`, the segpoly release tests, the harness tests and `check_report_496.py` all pass,
and the campaign is re-run on the final binary so the committed records and the submitted
code are the same state.

### Rounds 2 and 3 (after the first fixed diff)

- Round 2 findings: the dispatch seam is reachable through the supertrait bound (`T::lane_axpy::<0>`
  indexed an empty array), and the block-partial-before-lane-sum rule had no independent
  oracle (the reduction cases are `Batched`, so a unit is one row). Dispositions: unsupported
  arities are now a `debug_assert!` + release no-op with a comment that states what sealing
  does not do, and `a_shared_lane_reduction_combines_block_partials_first` adds the
  cancellation-sensitive `Shared` oracle (2 with the documented order, 0 with a per-block
  lane reduction). Both deletions the reviewer proposed were taken: the dead one-/two-/three-operand
  arms of `accumulate`, and the serial-only budget sweep.
- Round 3: `VERDICT: APPROVE`, no new BLOCKER or IMPORTANT; three stale comments and the
  input-less-operation note were fixed.
- The regression tests were checked against the round-1 classification (reverted in place):
  two of the three fail there and pass on the fix.
