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
| S2 fused single add vs `zip_map2_into`, 8^3 | 1.04 to 1.18 | <= 1.5 | 0 |
| S3a `map_into` vs compiled typed `UpdatePlan`, n=16 | 1.54 | <= 2.0 | 0 |
| S3b `map_into` vs compiled typed `UpdatePlan`, 4^3 | 2.24 | <= 3.0 | 0 |
| S3c `map_into` vs compiled `ErasedUpdatePlan`, n=16 | 0.92 | report | 0 |
| S3d `map_into` vs compiled `ErasedUpdatePlan`, 4^3 | 1.49 | report | 0 |
| S4 permuted vs contiguous `copy_into`, 8^3 / 32^3 | 0.78 / 15.3 | report | 0 |
| S5 unspecialized vs specialized add*mul, 8^3 | 7.6 to 8.1 | report | 0 |

S2 was 6.5 before the item-2 fix (the per-element `eval_binary` dispatch) and
1.4 to 1.5 after it, because `fused_elementwise_into` validated the destination
injectivity a second time inside `try_static_specialization`. Threading the
`ValidatedDestinationLayout` marker out of `validate_shapes` into the
specialization removed the duplicate check (`strided-basic` now exposes
`validate_destination_layout` returning the marker); S2 is 1.04 to 1.18 and the
fused single add matches `zip_map2_into` (about 158 ns against 147 ns at
8x8x8).

S3 measures the P4 per-call-planning residual of typed `map_into`: it is about
1.5x a compiled typed `UpdatePlan` at n=16 and about 2.2x at 4x4x4, which is
the gap the gate exists to watch. Because #306 says the first thresholds are
loose and tightened after about 20 CI runs, S3a/S3b start at 2.0 and 3.0 (the
measured value times about 1.3). The `ErasedUpdatePlan` a downstream runtime
actually replays is reported as S3c/S3d, separately from the typed gate.
