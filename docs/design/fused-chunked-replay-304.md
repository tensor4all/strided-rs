# Vectorized chunked replay for fused elementwise plans (#304)

Status: design for review, before implementation. Repo: `tensor4all/strided-rs`,
crate `strided-fused`.

## Problem

`interpret_inner_loop` / `interpret_inner_loop_uninit` replay a fused plan one
element at a time: gather each operand through runtime strides, evaluate every
instruction through `eval_op`, store one output. Measured on AMD EPYC 7713P at
one thread (tenferro-rs#2140/#2161, quoted in #304): a 1024x64 f32 region of ten
`tanh` costs 23.4 ms fused against 1.68 ms dispatching the same instructions one
at a time through the workspace's vectorized kernels; ten `mul` costs 6.0 ms
against 0.11 ms. tenferro declines to select a fused region for this reason.

## Goal

Make the fused replay at least as fast as per-instruction dispatch, without
changing the planner or the plan semantics. Only the execution path changes.

## Design

### One chunked register machine, interpreter as fallback

Both interpreter entries keep their shape: `for_each_inner_block_preordered`
walks blocks and a `run_block` callback handles one block. The callback gets
`(offsets, len, strides)` where `strides[i]` is operand `i`'s stride along the
block's innermost axis (outputs first, then inputs). A block is **contiguous for
every operand** when `strides` is non-empty and all `strides[i] == 1`; that is
the chunked path's eligibility test, and it costs one comparison over `strides`.
A rank-zero block hands the callback `len = 1` and an empty `strides`, so it
falls back rather than reading a stride that does not exist.

Both traversals are changed: `fused_elementwise_into_uninit` owns its own
`interpret_inner_loop_uninit` walk and is tenferro's entry
(`ErasedFusedPlan::execute_uninit`), so it takes the same chunked path.

- Contiguous block, `T` is `f32` or `f64`: chunked replay.
- Anything else (any stride != 1, including stride-0 broadcasts, or `T` is
  `i32`/`i64`/`bool`/`Complex32`/`Complex64`): the existing per-element
  interpreter, unchanged.

Non-contiguous regions keep the interpreter deliberately: they are not the
reported workload, and a gather/scatter chunk machine is a second design. #304
records this option as acceptable.

### Register file

`(plan.input_count + plan.ops.len()) * CHUNK` elements of `T`, allocated **once
per traversal invocation**: in the interpreter entry for the serial walk, and
inside the `run_partition` closure body for the parallel walk, which
`mapreduce_threaded` requires to be `Fn + Sync` and therefore cannot capture a
`&mut` register file. Registers are passed to the block callback as `&mut`;
register `r` occupies `regs[r * CHUNK .. (r + 1) * CHUNK]`; registers
`0..input_count` are the loaded inputs and `input_count + k` is instruction
`k`'s result. Nothing allocates inside the chunk loop, and non-real dtypes pay
one allocation per walk where they used to pay one per block.

Per chunk (a tail chunk is shorter):

1. Load each input's chunk from its operand base pointer (`ptr + offsets[i]`),
   contiguous by the eligibility test.
2. Run every instruction over the chunk in order.
3. Store each output register's chunk into its destination.

No allocation inside the chunk loop. CHUNK = 1024 elements for both real dtypes: two registers per instruction (8 KiB
of f64, 4 KiB of f32) stay in L1, and the measured fused-to-dispatch ratio moved
by less than 4% over 256 to 8192, so the size only has to keep the per-call
kernel count low.

### Instruction kernels

Per-op dispatch happens **once per instruction per chunk**, not once per element
per instruction. The scalar fallback for an op is a monomorphized loop (op
matched outside the loop, per `REPOSITORY_RULES.md` "Unsafe And Fast-Path
Boundaries"), so LLVM can vectorize it.

| op class | ops | chunk kernel |
|---|---|---|
| analytic unary | `Exp`, `Log`, `Sin`, `Cos`, `Tanh`, `Expm1` | `elem_simd_math::{f32,f64}` slice kernel |
| exact unary | `Negate`, `Conj`, `Abs`, `Sqrt`, `Rsqrt` | monomorphized chunk loop |
| binary | `Add`, `Multiply`, `Divide`, `Maximum`, `Minimum`, `Pow` | monomorphized chunk loop |
| ternary | `Clamp` | monomorphized chunk loop |

`Log1p` is excluded after measuring the kernel: `elem-simd-math 0.1.0` returns
`inf` at `f64::MAX` where `f64::ln_1p` returns `709.78` (reported upstream as
elem-simd-math-rs#6), so it keeps the scalar chunk loop and stays bit-identical.
`Maximum`/`Minimum` keep `f32`/`f64` `max`/`min` semantics by calling those
methods; the loop only needs to be vectorizable, not reassociated. `Sqrt` uses
std's correctly rounded `sqrt`, not the elem-simd kernel (whose documented bound
is 0.5001 ULP), so it stays bit-identical to the scalar replay.

**Accuracy decision (the one semantic change).** The seven analytic kernels come
from `elem-simd-math 0.1.0`, documented at 1.0 ULP. This is the same kernel set
tenferro's per-instruction path uses through
`strided_basic::map_into_uninit_with_kernel`, so a fused region and the
non-fused path it is compared against agree **bitwise with each other**; what
changes is only the relation to std's scalar `tanh`/`exp`/… inside
`strided-fused` itself, whose own suite compares against std at a relative
tolerance. `fused_elementwise_into` does not document bit-exactness against
libm. Tests therefore pin bit-exact equality for the exact ops and a
dtype-aware bound for the analytic ones (1 ULP, not a fixed `1e-12`: one f32 ULP
is about `1e-7`), and special values are checked by classification, not by bit
pattern.

Arithmetic, `Maximum`, `Minimum`, `Clamp`, `sqrt`, `rsqrt`, `abs`, `negate` and
`conj` stay bit-identical to the scalar replay. `Sqrt` uses std's correctly
rounded `sqrt` — elem-simd 0.1.0's slice `sqrt` is a std `sqrt` call behind its
own tail padding, so there is nothing to gain by routing it.

### Plumbing

- New dependency `elem-simd-math = "0.1.0"` (already in tenferro's graph, so no
  new crate downstream).
- **No public-surface change.** The f32/f64 kernel entry is a private function
  that guards on `TypeId` and reinterprets the chunk pointers, exactly like
  `strided-basic/src/simd.rs::try_mul_contiguous_ptr`; complex/`bool`/integer
  types simply fail the guard and use the interpreter. The chunked executor
  itself is generic over the existing `FusedScalar`.
- `fused_elementwise_into` and `fused_elementwise_into_serial` keep taking the
  static specialization first; this changes only what the interpreter fallback
  does, and the planner is untouched.

## Tests

Every case runs through the four entries: `fused_elementwise_into`, the serial
`fused_elementwise_into_serial`, `fused_elementwise_into_uninit`, and
`ErasedFusedPlan::execute_uninit`, in a serial and a bounded context, against the
per-element interpreter as the reference. The plans are a ten-instruction chain
(no static specialization kind, so the interpreter is the only path) and a
multi-output plan.

- Equivalence: every op and the ten-op chain, f32 and f64, at lengths 1, 63, 64,
  65, 1000 and a gapped 2-D layout (outer strides unequal and non-contiguous,
  nonzero view offset) so several callbacks with different bases run, including
  a tail chunk per callback.
- Exact ops: bitwise equality with the interpreter. Analytic ops: within one ULP
  of the interpreter, with the dtype's `EPSILON` as the unit.
- Special values on the analytic kernels (NaN, `+inf`, `-inf`, signed zero,
  domain boundaries of `tanh`/`expm1`/`log1p`) and on `Maximum`, `Minimum`,
  `Clamp`, `Divide`, `Rsqrt`, `Sqrt`; NaN results compared by classification.
- Fallback: a permuted operand, a stride-0 broadcast, a rank-zero view and a
  non-real dtype still produce the interpreter's results, and the chunked
  executor never runs for them.
- Allocation: one register file per walk; `serial_specialization_alloc.rs` counts
  stay valid (its plans take the static specialization, not the interpreter).

## Benchmark (acceptance)

`strided-fused/benches/fused_chunked_replay.rs` (criterion dev-dependency
present; `elem-simd-math` for the baseline kernels): ten-`tanh` and ten-`mul`
chains, f32 and f64, at 1024x64 and 4096x256, serial, one thread asserted.

The baseline is per-instruction dispatch through
`strided_basic::map_into_uninit_with_kernel` with the elem-simd kernels and
preallocated intermediates — not `map_into`, which passes `NoKernel` and would
compare against scalar `tanh`. Both the typed entry and
`ErasedFusedPlan::execute_uninit` are measured. Acceptance: the fused replay is
not slower than the baseline at both sizes, and the numbers go into the PR body.

## Out of scope

Planner behaviour, the static specialization path, complex/int/bool vector
kernels, gather/scatter for non-contiguous regions, and the IR shape of #233.
