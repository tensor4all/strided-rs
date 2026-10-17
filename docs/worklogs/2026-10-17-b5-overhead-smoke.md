# 2026-10-17: B5 — NaN-propagating fused max/min, single-op dispatch, overhead smoke

Scope: the last 0.6.0 batch unit on `wip/0.6-b5`. Two source fixes and the
per-repository overhead smoke benchmark from the `#306` comment.

Machine for the numbers: AMD EPYC 7713P, Linux, rustc 1.97.1, `cargo bench
--profile release`, `RAYON_NUM_THREADS` pinned through a one-thread pool (fused)
or `ExecutionPolicy::Sequential` (basic).

## #315: fused `maximum`/`minimum`/`clamp` propagate NaN

`strided-fused` used Rust `f32`/`f64` `max`/`min`, which return the non-NaN
operand. It now calls the shared
`strided_basic::nan_propagating_maximum`/`nan_propagating_minimum` select, and
`strided-kernel`'s ordinary binary `ErasedZipOp::Maximum`/`Minimum` calls the
same helpers, so the two definitions cannot drift. `fused_clamp` is derived as
`minimum(hi, maximum(lo, x))`, which keeps the ordinary clamp's signed-zero ties
(`clamp(-0.0, 0.0, 1.0) == +0.0`).

Differential tests compare the fused result bit-for-bit against the ordinary
erased kernel for NaN in either operand and signed-zero ties, f32 and f64
(`strided-kernel/tests/erased_one_shot.rs`); the fused tests cover the
per-element interpreter (stride-2 input), the chunked replay above
`FUSED_CHUNK_ELEMS`, and the static single-op and ternary specializations.

## #305 item 2: single-op specializations dispatch per element

The `Unary`/`Binary`/`Ternary` static specializations passed
`eval_unary/eval_binary/eval_ternary` into the element loop, matching on the
runtime `FusedOp` per element and preventing vectorization. They now match the
op once and call the monomorphized `FusedScalar` method, so each arm is a
distinct vectorizable kernel. Behavior is unchanged (the same trait methods the
interpreter calls).

## Overhead smoke benches

`strided-basic/benches/overhead_smoke.rs` (S3, S4) and
`strided-fused/benches/overhead_smoke.rs` (S1, S2, S5), `harness = false`,
counting `#[global_allocator]`. Each gated row measures both paths
interleaved round by round (minimum of 5 rounds), three attempts, and fails only
when all three attempts exceed the threshold. `OVERHEAD_GATE=1` makes a failure
exit non-zero; the table also goes to `$GITHUB_STEP_SUMMARY`.

Representative run (this machine):

| id | ratio | gate | allocs/op |
|---|---|---|---|
| S1 serial vs 1-thread parallel fused two-op DAG, 8^3 | 0.99 | <= 1.5 | 0 |
| S2 fused single add vs `zip_map2_into`, 8^3 | 1.43 to 1.48 | <= 1.5 | 0 |
| S3a `map_into` vs compiled `ErasedUpdatePlan`, n=16 | 1.06 | <= 2 | 0 |
| S3b `map_into` vs compiled `ErasedUpdatePlan`, 4^3 | 1.62 | <= 2 | 0 |
| S4 permuted vs contiguous `copy_into`, 8^3 / 32^3 | 0.78 / 15.3 | report | 0 |
| S5 unspecialized vs specialized add*mul, 8^3 | 7.6 to 8.1 | report | 0 |

S2 was 6.5 before the item-2 fix (the per-element `eval_binary` dispatch) and is
1.4 to 1.5 after it; at 16^3 it is 1.12, so the residual is the fused per-call
validation, not the kernel. S3 uses the compiled erased update plan because that
is the plan a downstream runtime compiles and executes per call (P4); against
the typed `UpdatePlan` the 4^3 row is about 2.2 and the threshold would not hold.
