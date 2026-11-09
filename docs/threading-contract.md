# Threading contract: opt-in only

This page records the maintainer decision on
[issue #6](https://github.com/tensor4all/strided-rs/issues/6) — the comment of
2026-11-07, **"(a) opt-in を維持"** — and what it means for the code. The
mechanism behind the token (pools, `fork`, `for_ranges`, thresholds) is in
[`design/cpu-exec-placement.md`](design/cpu-exec-placement.md) and in
threadteam-rs; this page only states the contract.

## The decision

**Placement is opt-in. The default is serial. Work fans out only when the caller
hands in an explicit `Exec` token.**

The maintainer's reasons, in the words of the decision:

- Parallel by default oversubscribes a caller that already runs on its own pool.
- The parallel API of this workspace carries the *team's lifetime* in the token,
  so **"whose team the work runs on" is the caller's decision** — that is the
  consistent reading of the design.
- If automatic parallelism is ever wanted, it is added as an explicit new value
  (a `Par::Auto` equivalent), not by changing the default.

Julia's Strided.jl makes the same choice for threaded matmul
(`_use_threaded_mul = Ref(false)`); see the issue for the comparison.

## The rules

1. **No automatic parallelism.** No kernel reads a global thread count, no
   thread-local scope exists, and no strided crate starts a process-global pool.
   The only token a caller gets without a `Pool` is `Exec::serial()`
   (`threadteam`'s `Exec::serial`, re-exported by `strided-basic`), which runs
   everything inline on the calling thread.
2. **Every public entry that can fan out takes its token last.**
   `exec: Exec<'_>` is the final parameter, by value, never stored; plans take it
   per replay. Two kinds of entry sit outside that rule:
   - **The low-level range helper.** `strided_basic::execution::parallel_map_reduce`
     (`strided-basic/src/threading.rs:134-138`, re-exported at
     `strided-basic/src/execution.rs:40`) takes `exec` second: it is not an
     operation entry but the helper the tree-shaped entries call.
   - **Serial-only entries.** An entry that cannot fan out takes no token at all
     and always runs on the calling thread, for example `dot`
     (`strided-basic/src/ops_view.rs:935`). `strided_basic::copy_into_col_major`
     used to sit here: it took no token and handed `Exec::serial()` to the
     permutation engine, so a caller could not see that a large transpose ran on
     one core. It now takes its token like `copy_into` (issue #212).

   `REPOSITORY_RULES.md` ("CPU Threading Contract") makes the fan-out half a
   review rule, and the crate docs state it
   (`strided-basic/src/lib.rs:42-51`, `strided-perm/src/lib.rs:14-22`).
3. **The token decides the width, never the entry.** A serial token runs the
   serial kernel directly: `parallel_width` and `tree_width`
   (`strided-basic/src/cost.rs:149`, `:168`) answer `1` before consulting any
   cost, and the entries read them through `for_len` (`cost.rs:190`) or repeat the
   serial check themselves (`is_parallel` at `cost.rs:203`, the `exec.is_serial()`
   branches such as `map_view.rs:880`). A pooled token is bounded by its budget
   (see [`design/cpu-exec-placement.md`](design/cpu-exec-placement.md)).
4. **The caller owns every other thread pool.** strided-rs cannot see an
   external BLAS or faer's internal parallelism. A caller that combines its own
   fan-out with a provider that threads internally must set that provider to one
   thread itself (`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, faer with
   `Par::Seq`). Strided does not set those variables, and it does not reduce a
   provider's thread count behind the caller's back. When a provider route is
   added, the token's width reaches it through `Exec::external`, which hands the
   library a plain width: the caller's choice, forwarded unchanged and never
   widened or narrowed.

## Why the fan-out shares rayon instead of owning a pool

Only `threadteam` depends on `rayon` as a scheduling dependency in this
workspace (`deny.toml`; `cargo deny check bans` in CI). The one other crate that
brings rayon in is `criterion`, a benchmark harness, and `deny.toml` names it as
a wrapper exception. Every other crate fans out through the token it was handed.
Because the token carries the pool, nested fan-out of two strided calls lands on
the same work-stealing pool instead of a second pool competing for the same
cores. That is also why the contract is a *token* and not a global setting: two
callers can hold different pools at the same time, and neither affects the other.
A caller that wants its own thread budget builds a `Pool`, mints
`Exec::new(&pool, n)` and passes it down.

## State of the code

- The typed entries of `strided-basic` and `strided-perm` that can fan out are
  pinned by type in `strided-basic/tests/threading_contract.rs`: 48 entries, of
  which 47 take the token last and one (`parallel_map_reduce`) takes it second.
  An entry that drops its token stops compiling. The pinned list is:
  - element-wise: `map_into`, `map_into_uninit_with_kernel`, `zip_map2_into`,
    `zip_map3_into`, `zip_map4_into`, `mul_into`, `mul_into_uninit`,
    `broadcast_mul_into`, `broadcast_mul_into_uninit`, `compare_into`,
    `compare_into_uninit`, `batched_outer_product_into`,
    `batched_outer_product_into_uninit`;
  - in-place update: `map_update_into`, `zip_update2_into`, `zip_update3_into`;
  - reduction: `reduce`, `zip_reduce`, `reduce_axis`, `reduce_axis_into`, `sum`;
  - copy and scale: `copy_into`, `copy_into_col_major`, `copy_into_uninit`,
    `copy_scale_into`, `copy_conj_into`,
    `copy_transpose_scale_into`, and `strided-perm`'s `copy_into`,
    `copy_into_uninit`, `copy_into_col_major`;
  - arithmetic and layout: `add_acc`, `mul_acc`, `axpy_acc`, `fma_acc`, `axpby_accum`,
    `symmetrize_into`, `symmetrize_conj_into`, `triangular_mask_into_uninit`,
    `triangular_mask_view_into_uninit`, `embed_diagonal_into_uninit`,
    `embed_diagonal_view_into_uninit`;
  - plan replay: `CopyPlan::execute`, `execute_uninit`, `execute_scale`,
    `execute_conj`, `UpdatePlan::execute`, `execute_uninit`;
  - the range-helper exception: `parallel_map_reduce`, token second.

  Entries that cannot fan out are not pinned because they carry no token:
  `dot` is the current example.
- The remaining plan replays (`GatherPlan`, `ScatterPlan`, `SlicePlan`,
  `ReversePlan`, `ConcatenatePlan`, `PadPlan`, `DynamicSlicePlan`,
  `DynamicUpdateSlicePlan`) and the erased entries of `strided-kernel`,
  `strided-fused` and `strided-nn` are replayed with a token by the tests that own
  them, so dropping the parameter fails those tests instead.
- **There is no faer matmul path yet.** No crate in the workspace depends on
  faer — `deny.toml` bans it outright, together with `spindle`, so a GEMM
  provider is always the consumer's own dependency — and `matmul` is still
  future work ([`future.md`](future.md) §3). This page records decision (a) of
  issue #6 (2026-11-08, maintainer judgement) as the contract; a future faer
  route is *planned* to follow the same contract, forwarding the caller's choice
  unchanged, and gets no rules of its own.

## Regression tests

- `strided-basic/tests/threading_contract.rs` — a serial token never justifies a
  second worker for any kernel class; the two observed entries (`map_into`,
  `zip_map2_into`) run on the calling thread under `Exec::serial()` and off it
  under a pooled token (when the `parallel` feature is on); and the 48 entries
  above are pinned by their types.
- `strided-basic/tests/copy_into_col_major_policy.rs` — `copy_into_col_major`
  must use the token it is handed: a large permutation copy under a four-slot
  token shows more than one live task under that token's ledger root, which the
  hard-coded `Exec::serial()` of #212 failed.
- `strided-basic/tests/serial_exec_global_pool.rs` — serial work never starts
  the process-global pool.
- `strided-basic/tests/execution_pool_*.rs`, `strided-fused`,
  `strided-perm` — a pooled token stays within its budget.
