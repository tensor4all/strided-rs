# 2026-11-16: #479 Part 2 — the segmented-polynomial executor

Issue: [tensor4all/strided-rs#479](https://github.com/tensor4all/strided-rs/issues/479).
Scope: Part 2 only, the new unpublished crate `strided-segpoly`. Part 1
(`RunIndex` / `RunSumPlan` in `strided-basic`) merged as #489 and is consumed,
not changed; the dtype-erased replay in `strided-kernel` and the descriptor
crate are **not** in this change (D6, and the scope note in §6 of
`IMPLEMENTATION-PLAN.md`).

Decisions D1–D11 were accepted before implementation and are the contract this
crate implements: at most one batch axis, `execute_into`/`execute_acc` without a
scaling factor, typed errors for bad indices, a deterministic batch reduction for
a shared output, `f32`/`f64` only, no `REPEATED`/`indexed_linear`, an empty run
is zero, budget-invariant bits, generated fixtures from a pinned cuEquivariance
checkout, no forced 1.89 MSRV, and evo2 numbers as preliminary until the
aarch64 M5 host closes the performance gate.

## 1. What the kernel does

A lowered segmented polynomial is

```text
out[out_row, s, t] (=|+=) Σ_path coeff_path · Π_operand operand[operand_row, s_path, t]
```

where the segment and lane indices come from the path, and the *row* comes from
the buffer's index mode instead of the batch axis: `Batched` (the row is the
batch point), `Shared` (row 0, one row read by every batch point), `Gather(&[i32])`
(the row is a caller-supplied source row) and `Scatter(&RunIndex)` (the batch
points are the rows of one destination run, resolved through the sorted-run
topology of Part 1). The point of the crate is that a message-passing layer's
whole per-edge computation runs in one pass: no `edges × segments × u`
intermediate, no atomics, no per-edge read-modify-write on the destination.

`SegPolyPlan::compile` takes the descriptor plus the inputs' and outputs' layouts
and proves everything it can without the buffers: one batch axis, arity and
extent agreement, path ranges into the segment lists, per-buffer role against its
index mode, the `Shared` row extent, the scatter destination's rows against the
run count, an injective output layout, the uniform lane extent `u`, and a checked
element count. Coefficients are rounded to `T` at compile time, so a `f32` plan
rounds once and the replay is a pure multiply-add.

## 2. Determinism is an association, not a tolerance

The summation order is part of the result:

1. rows inside a unit in run order (`RunIndex` order for a scatter, `0..batch`
   otherwise),
2. operations in descriptor order,
3. paths in descriptor order within their operation,
4. rows grouped into fixed blocks of `BLOCK_ROWS = 256`, partials combined left
   to right after the fork joins.

So the bits depend on the descriptor, the layouts and `BLOCK_ROWS` — and on
nothing else. Not on the `Exec` budget, not on the worker count, not on which
buffer the caller reaches the same elements through. `BLOCK_ROWS` is a fixed
constant for exactly that reason; changing it changes results, which is stated
where it is defined, in the rustdoc and in the README. The documented exception
is the sign of zero in a later block's partial (a `+0.0` and a `-0.0` sum
differently depending on the association), which no test asserts.

The tests that hold this: `tests/kernels.rs::budget_split_is_bitwise_for_a_batched_and_a_shared_output`
on a `4096 × 8 × 8` batched and a shared output (it also asserts the row count
actually exceeds one block, so the test cannot silently degenerate to a
single-block comparison), and `src/plan/tests.rs::block_partials_combine_in_the_documented_order`,
which pins the association with values whose left-to-right and right-to-left sums
differ by construction.

Part 1's `RunSumPlan` gets the same property from a different argument (one
worker per run, rows in increasing order, disjoint destinations). Here a
destination row can be reached by many batch points, so the blocks are what make
the order budget-independent: a fixed partition of a deterministic sequence is
deterministic, a budget-dependent one is not.

## 3. Dispatch

One `tree_width(&exec, KernelClass::Map, macs)` fork level per call, children
partitioned by cumulative row count so units of very different sizes stay
balanced, no nested forks, and the combine runs sequentially after the join.
Nothing is atomic and nothing locks: a unit owns its destination rows, and rows
are disjoint between units (compile requires the injective destination layout).
Child tiles are capped by the scratch the caller supplied, so a narrower token
runs fewer children instead of allocating.

The per-call allocation count is asserted to be zero by
`tests/alloc.rs` (a counting allocator, `execute_into` and `execute_acc`, from
`u = 1` through `u = 128`, and past one block). The per-call state is a stack
`Call`.

`execute_into` zero-initialises then accumulates; `execute_acc` adds to the
caller's values. For an output block whose strides are contiguous, the
destination *is* the accumulator and is never read into a temporary first — the
second path would be an aliasing copy with no purpose, and `execute_acc` needs no
seed copy for exactly that reason.

## 4. Fixtures come from the pinned upstream

`tests/fixtures/generate.py` drives a pinned cuEquivariance checkout
(`9ed8c7613a79861415071027249cc458e423be30`, Apache-2.0) to lower each descriptor
and to compute the expectation with its own `execute_naive`, then writes the
inputs as `.npy` **seeds** (so a fixture is a case description plus the data a
caller would allocate) and the expectation as a separate image. The Rust side
reads those images and checks the executor against them; `cases.rs` is generated
too. Nothing is hand-written and nothing is a value this crate produced itself.

Seven cases: symmetric contractions of degree 1, 2 and 3, spherical harmonics
(the `u = 1` coverage), a jvp and the backward pass of a degree-2 case (four
buffers), and a `gather + scatter` message-passing case with a run of six rows.
The generator had to replay `_remap_indices_and_config` itself because the
upstream `uniform_1d` lowering only accepts a subset of primitives: it rejects
`channelwise_tensor_product` and `linear` (multi-mode) and requires a uniform
multiplier for `symmetric_contraction` (hence `scaled()` adds a `same_mul=4` term
per path), and a non-linear `transpose` is rejected ("at most one undefined
primal") so the backward case transposes the jvp polynomial.

On top of the fixtures: finite differences against the analytic derivative, the
transpose as a discrete adjoint of `execute_into` (per input, `1e-10`), the
rejection table (D1, D3, D5, D6, scratch floor, layout mismatch, foreign scatter
topology, gather out of range), and the allocation windows.

## 5. Performance, and what it does not say

`benches/executor.rs` compares the fused path with the two-pass alternative a
fused executor has to beat: materialise every operand product, then reduce it
with `RunSumPlan`. Both rows compute the same numbers from the same buffers.
evo2, 32-core Zen 5, `cargo bench --profile release -p strided-segpoly --bench
executor`, 4 worker slots for the pooled rows:

```text
row                           ns/call    GFLOP/s   MiB extra  vs fused
chain-8k/fused/serial         6491853       0.65        32.0      1.00
chain-8k/fused/pooled         1669975       2.51        32.0      0.26
chain-8k/two-pass/serial      3775238       1.11        32.0      0.58
chain-8k/two-pass/pooled      3138258       1.34        32.0      0.48
small-128/fused/serial          10215       0.40         0.0      1.00
small-128/fused/pooled          10276       0.40         0.0      1.01
small-128/two-pass/serial        6993       0.59         0.0      0.68
small-128/two-pass/pooled        7016       0.58         0.0      0.69
```

`chain-8k` is 8192 edges × 16 segments × 32 lanes into 2048 runs: the fused
pooled row is 1.9x the two-pass pooled row and never materialises the 32 MiB
intermediate. `small-128` is 128 edges × 4 segments × 8 lanes: there the fused row
is 1.4x *slower*, because per-path setup and the call's fixed costs dominate 4096
multiply-adds. The crossover is real and is the honest statement of where this
executor pays off; the small case was not tuned further. Both rows apply the same
per-segment coefficient and build one view per call, and the bench asserts before
timing that they agree to 1e-12 (they are not bitwise equal: the fused row combines
fixed block partials).

Two things this measurement is not. It is not the D11 gate — that needs the
aarch64 M5 host, and these numbers are preliminary by decision. And it is not
pinned to one L3 domain as `AGENTS.md` requires for a recorded measurement: the
host had other load during the run (the two-pass pooled row moved 4.4 ms → 6.9 ms
between two runs of the same binary), so read the ratios as approximate and the
"vs fused" column as the only stable statement.

Two changes came out of it and both are in:

- the lane tile is uninitialised rather than zero-filled (`MaybeUninit`): the
  zero-fill of 64 lanes dominated a narrow lane axis, `small-128` went from 13.8 µs
  to 12.3 µs on that alone;
- the lane loop is unrolled for arity 0–3, which is every arity a lowered
  polynomial uses here (`chain-8k` pooled 2.5 → 1.43 ms, `small-128` 12.3 →
  9.7 µs).

The kernel is not blocked and not SIMD-dispatched: the lane loop autovectorises,
and a `pulp` dispatch is a separate change with its own measurement (the same
non-goal as Part 1's payload walk).

## 6. Deliberate non-changes

- **No `beta`.** `execute_into` (zero-init + accumulate) and `execute_acc`
  (accumulate on top of the caller's values) cover the two behaviours; a scaling
  factor would be a third parameter on every call for a case the caller can do.
- **No descriptor crate, no `REPEATED`/`indexed_linear`.** Producing the
  descriptor is a front end's job, and the two primitives Part 1's contract
  excludes stay excluded.
- **No `PartialEq` on `SegPolyError`.** Nothing compares errors; a derive would
  only invite a test to assert on a variant instead of on behaviour.
- **No MSRV bump.** D10: nothing here needs a newer rustc than the workspace
  already requires.

## 7. Review record

- Plan gate (before any code): agent definition `reviewer-glm`, model
  `ollama-cloud/glm-5.3-flash` at thinking level `high`, read-only, launched from
  the implementation session. **`VERDICT: CHANGES`** with 2 BLOCKER, 3 IMPORTANT
  and 5 MINOR findings, all dispositioned in `IMPLEMENTATION-PLAN.md` §8 before
  implementation started: B1 the scratch contract disagreed with itself, B2
  "blocked == unblocked bitwise" was unachievable and was replaced by the
  scalar-reference association test, I1 the now-false erased-replay sentence in
  `strided-basic/src/run_sum.rs` (amended in this change), I2 the
  `required-features` wiring for the benches, I3 the aliasing contract stated in
  the rustdoc, M1 the `AGENTS.md` layout row, M2 the coverage gate in the local
  check list, M3 checked `u32` slot conversions, M4 `## Verification` in the PR
  body, M5 the `BLOCK_ROWS` wording.
- Post-implementation review of the finished diff: agent definition
  `reviewer-gpt` semantics through `scripts/pi_review.sh`, model
  `openai/gpt-6.1-sol`, read-only, on `git diff origin/main..HEAD` plus the
  untracked plan. **`VERDICT: CHANGES`** with 3 BLOCKER, 9 IMPORTANT and 4 MINOR
  findings plus a delete list. Every finding was checked against the source; all
  were real and all are fixed, each with a regression test in
  `tests/contract.rs` where the public API can reach it:
  - **BLOCKER** a gather shorter than the batch compiled and was then read past
    its end (`exec.rs:219`). Fixed: compilation and the call both require
    `len == batch`, and a length-1 gather is no longer accepted as a broadcast
    (`Shared` is that case). Test: `a_gather_must_cover_the_batch`.
  - **BLOCKER** the input lane step was hardcoded to `1` (`plan.rs:598`), so any
    input whose lanes are separated read the wrong elements. Fixed: the step is the
    compiled lane stride. Test: `an_input_reads_its_lanes_at_the_compiled_stride`.
  - **BLOCKER** an empty output beside a non-empty one offset a dangling empty
    view's pointer (`exec.rs:314`). Fixed: a zero-element output is skipped before
    any pointer arithmetic. Test: `an_empty_output_beside_a_full_one_executes`.
  - **IMPORTANT** `execute_acc` seeded every block partial from the destination, so
    a multi-block unit added the caller's value once per block (`exec.rs:335`).
    Fixed: only block zero (and a single-block child) seeds from the destination.
    Test: `execute_acc_adds_the_destination_once_per_unit`, which was confirmed to
    fail against the old line before the fix.
  - **IMPORTANT** the per-output operation list was one bounding range, so
    descriptor order A, B, A ran B for A (`plan.rs:451`). Fixed: per-output
    operation indices, still in descriptor order. Test:
    `an_output_written_by_two_operations_runs_both_in_descriptor_order`.
  - **IMPORTANT** `MAX_INPUTS` bounds distinct buffers but the replay tables are
    per operand, so an operation repeating an input past the capacity panicked
    (`exec.rs:494`). Fixed: `TooManyOperands` at compile. Test:
    `an_operation_may_not_hold_more_operands_than_the_replay_tables`.
  - **IMPORTANT** a segment axis of `usize::MAX` beside a zero lane axis reached
    `vec![0u32; segments]`, which aborts (`plan.rs:430`). Fixed: the segment tables
    are allocated with `try_reserve` and overflow to `OffsetOverflow`. Test:
    `a_segment_axis_whose_table_cannot_be_allocated_is_a_typed_error`.
  - **IMPORTANT** unit ids were cast to `u32` unchecked (`plan.rs:568`). Fixed:
    `u32::try_from` (and the same for a path count that indexes the `u32` segment
    tables).
  - **IMPORTANT** `check_layout` summed the reach with signs, rejecting a
    representable reversed view (`plan.rs:723`). Fixed: the reach is summed in
    absolute stride. Test: `a_reversed_batch_stride_is_a_representable_layout`.
  - **IMPORTANT** `SegPolyElem` promised to be sealed and was not (`plan.rs:63`).
    Fixed: a private `Sealed` supertrait.
  - **IMPORTANT** the bench's two-pass row dropped the per-segment coefficient and
    built its views inside the timed window, so the "same numbers" claim was
    false. Fixed: one `build_product` shared by both rows, views per call in both,
    and a pre-timing 1e-12 cross-check.
  - **IMPORTANT** determinism coverage was one dense `f64` case at one pool size.
    Extended: two pool sizes (2 and 4 threads), budgets 1/2/4, and a new
    lane-strided layout case; a separate test covers the `f32` rounding path
    (`an_f32_plan_rounds_the_coefficient_at_compile_time`).
  - **MINOR** the task splitter's `partition_point(.. <= target)` handed an exact
    boundary's tasks to the earlier child (`exec.rs:270`). Fixed to `<`, which
    keeps the ranges contiguous and balanced.
  - **MINOR** the fixture generator derived its output seeds from the number of
    existing fixture directories, so a regeneration into a partial tree differed.
    Fixed: a fixed seed per case, and `require_pinned_upstream()` refuses a checkout
    whose `HEAD` is not the pinned commit. Regenerating twice is now byte-identical
    (checked with `md5sum -c`).
  - **MINOR** the changelog claimed `compile` detects overlapping outputs, which a
    layout cannot show. Fixed: the entry states the real obligation.
  - **MINOR** the `AGENTS.md` row advertised gathered *outputs*, which compile
    rejects. Fixed.
  - **Delete list**: the unused `Case::naive` reference and its indexing helpers,
    the `Npy<T>` container (readers return vectors), and the unused `CALL` global in
    the generator are deleted. No test reached the second reference implementation,
    so it was dead weight rather than an independent check.
  Nothing in the report was dismissed.

## 8. Verification

Local gates, all green:

```text
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                       # 93 test binaries, 0 failures
cargo test -p strided-segpoly                # 20 lib + 3 alloc + 11 contract + 6 fixtures + 7 kernels + 1 doc
RUSTDOCFLAGS="-D warnings" cargo doc -p strided-segpoly --no-deps
cargo llvm-cov --workspace --json --output-path /tmp/cov-workspace.json
python3 scripts/check-coverage.py /tmp/cov-workspace.json
cargo bench --profile release -p strided-segpoly --bench executor
```

The workspace run passes the per-file gate (`51/51 files passed`); the three files
this crate has are `src/plan.rs` 85.2 % (392/460), `src/plan/exec.rs` 86.9 %
(339/390) and `src/spec.rs` 100 %, all over the 80 % default in
`coverage-thresholds.json`.
