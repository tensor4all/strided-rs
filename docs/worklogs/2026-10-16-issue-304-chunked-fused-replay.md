# strided-rs#304 — a chunked, vectorized fused replay

## Scope

`interpret_inner_loop` / `interpret_inner_loop_uninit` replayed a fused region
one element at a time through runtime strides and `eval_op`, which is why
tenferro-rs declines to select a fused region (tenferro-rs#2140 item 3,
#2161). A contiguous block — every operand unit-stride along the block's
innermost axis — now goes through `strided-fused/src/fused/chunk.rs`: load each
input chunk, run every instruction over the chunk with a monomorphized kernel,
store the output chunks. Non-contiguous blocks, rank-zero blocks and non-real
scalars keep the per-element interpreter. The planner and the static
specialization path are unchanged, and no public item was added.

The design and the independent pre-implementation review are recorded in
`docs/design/fused-chunked-replay-304.md`.

## Decisions

- **Analytic kernels come from `elem-simd-math 0.1.0`** (1.0 ULP) for `exp`,
  `log`, `sin`, `cos`, `tanh`, `expm1`, `log1p`, behind a `TypeId` guard. This is
  the same kernel set a per-instruction dispatch uses, so a fused region and the
  non-fused path it replaces agree bitwise; against std's scalar libm the fused
  analytic values move by at most one ULP. Arithmetic, `Maximum`, `Minimum`,
  `Clamp`, `sqrt`, `rsqrt`, `abs`, `negate` and `conj` stay bit-identical.
- One register file per traversal invocation (per partition on the parallel
  walk, which `mapreduce_threaded` requires to be `Fn + Sync`), plus a
  `Vec<bool>` of single-use inputs. Nothing allocates inside the chunk loop.
- Inputs read by exactly one instruction are read straight from the operand
  instead of being copied into the register file.

## Verification

- `strided-fused/tests/chunked_replay.rs`: the per-element replay is produced by
  giving one operand a stride of 2, so the reference is the real scalar path.
  Arithmetic and exact unary chains are bitwise equal at lengths 63/64/65/1000;
  analytic chains stay inside a ULP budget; the ten-`tanh` f32 chain of the
  reported workload is **bitwise equal to ten `elem_simd_math::f32::tanh` passes
  over the whole array**; `NaN`/`±inf`/signed zero/domain boundaries are covered
  for the analytic and the exact ops; multi-output plans, reused inputs and a
  stride-0 broadcast fallback are covered.
- The existing suite passes; `specializes_exp_of_multiply_add_chain` compares
  the static path against the interpreter, which is now worth one ULP of an
  `exp(13.5)` result, so that one comparison is ULP-based (`2.0` ULP) instead of
  a fixed `1e-12` absolute.
- `cargo fmt --all -- --check`, `cargo test --workspace` and
  `python3 scripts/repository-rules-review.py --base main --worktree --dry-run`
  pass.

## Measured

`RAYON_NUM_THREADS=1 cargo bench -p strided-fused --bench fused_chunked_replay`,
AMD EPYC 7713P (shared host), release, criterion median of 100 samples, ten
instructions per region. The `mul` baseline is the workspace's own SIMD
multiply (`mul_into_uninit`), the `tanh` baseline is
`map_into_uninit_with_kernel` with the `elem-simd-math` kernel: the same kernels
the fused replay uses.

| chain | size | interpreter (before) | fused erased | per-instruction dispatch | fused / dispatch |
|:-|:-|-:|-:|-:|-:|
| f32 tanh x10 | 1024x64 | 23.4 ms | 1.446 ms | 1.418 ms | 1.020 |
| f64 tanh x10 | 1024x64 | — | 3.469 ms | 3.520 ms | 0.985 |
| f32 mul x10 | 1024x64 | 6.0 ms | 88.2 µs | 105.1 µs | 0.839 |
| f64 mul x10 | 1024x64 | — | 165.2 µs | 190.4 µs | 0.867 |
| f32 tanh x10 | 4096x256 | — | 23.29 ms | 22.47 ms | 1.036 |
| f64 tanh x10 | 4096x256 | — | 55.69 ms | 55.37 ms | 1.006 |
| f32 mul x10 | 4096x256 | — | 1.327 ms | 1.535 ms | 0.864 |
| f64 mul x10 | 4096x256 | — | 2.907 ms | 3.461 ms | 0.840 |

The interpreter numbers are the ones #304 quotes for the same workload; the
chunked replay is 13x to 16x faster than it, which is what tenferro needs.

Against per-instruction dispatch the result is split by what the chain is
bound by. The memory-bound `mul` chains are 13 to 16% faster at every size,
which is the fusion benefit: one read and one write of the large arrays instead
of ten. The compute-bound `tanh` chains are within ±3.6%: the kernel dominates
(2.2 ns per element per pass) and the chunked replay pays the register-file
traffic that fusion adds — two chunk moves per instruction plus one input load
and one output store, against the dispatcher's two large-array moves per
instruction. So **the acceptance is met on the `mul` chains and the f64 `tanh`
case at 1024x64, and is not met on the f32 `tanh` chains (2.0% and 3.6%
behind)**; the remaining lever is to write the final instruction's result
straight into the destination instead of a register, which removes one of those
moves.

`FUSED_CHUNK_ELEMS` was tuned over 256, 1024, 2048, 4096 and 8192: the ratio
moved by less than 4% across that range, so 1024 was kept (two registers per
instruction stay in L1 with room to spare).
