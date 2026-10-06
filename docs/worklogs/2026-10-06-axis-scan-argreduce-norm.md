# 2026-10-06: single-axis scan, arg-reduction and fused norm plans

Consumer: tensor4all/tenferro-rs#2010 PR-B2 (#1974 `cumsum`/`cumprod`,
#1976 `argmax`/`argmin`, #2006 fused `layer_norm`/`rms_norm`). The tenferro
wiring lands after tenferro-rs#2004 replaces its CPU strided adapter.

## API

All three are dtype-erased prepared plans in `strided-basic` (re-exported by
`strided-kernel`), with the `ErasedReducePlan` shape: `compile` validates and
stores fixed layouts, `execute` / `execute_uninit` take an `ExecContext` and
erased descriptors and check exact layout equality before any write.

| Plan | Ops | Dtypes | Output |
|---|---|---|---|
| `ErasedScanPlan` | `ScanOp::{Sum, Product}`, `ScanOptions` exclusive / reverse | f32 f64 i32 i64 c32 c64 | source dims, own strides |
| `ErasedArgReducePlan` | `ArgReduceOp::{Max, Min, MaxAbs, MinAbs}` | real all ops; complex `*Abs` only | source dims without the axis, i32 or i64 |
| `ErasedNormPlan` | `NormSpec::{layer_norm, rms_norm}(eps)`, optional strided weight / bias | f32 f64 | source dims, own strides |

## Semantics decisions

- Scan: each output is the sequential fold in scan order, no reassociation,
  so the result is bitwise independent of layout and thread count. Integers
  wrap. An empty axis writes nothing.
- Arg-reduction: lowest index on ties (`-0.0` ties `+0.0`); the first NaN wins
  for both max and min (NumPy / PyTorch / JAX). Complex magnitude is the
  `hypot` modulus, not `|z|^2`, which overflows near `1e154` and would tie
  large pivots; a NaN component makes the element NaN even when the other is
  infinite. An empty axis is rejected at compile (`UnsupportedOp`); an empty
  set of lines is a no-op. `StridedError` is exhaustive, so no new variant was
  added (that would be a breaking 0.5 change).
- Norm: biased variance (PyTorch `layer_norm`), `rsqrt` once per line,
  accumulation in the element dtype. The layer-norm variance uses the
  shifted-data two-pass algorithm (shift by the line's first element), so a
  constant line has exactly zero variance and normalizes to exactly `bias`
  for `eps > 0` (`NaN` for `eps == 0`). The plain two-pass form left a
  rounding residue of the mean (an `f32` line of 70 equal values normalized to
  `-6e-5 * rsqrt(eps)` instead of zero). `eps` must be finite and `>= 0`.

## Kernel structure

`erased/line.rs` compiles the non-axis (outer) axes once into the reduction
cursor. A unit is one line, or, when the plan axis is strided and the leading
outer axis is contiguous, a block of up to 64 lines whose kernels loop over the
axis outermost and over the contiguous block innermost. Each serial call or
parallel worker range runs inside one `pulp` runtime dispatch, with the kernel
passed as a `UnitKernel` struct so its loops inline into the target-feature
function. A closure passed to `pulp` was not inlined and its loops stayed
SSE2; this mattered for the default (non `target-cpu=native`) build that
tenferro uses.

The contiguous norm sums use sixteen partial sums combined by a left fold.
A pairwise tree over the partial sums made LLVM's SLP vectorizer split the
accumulators into two-lane groups (`vmovsd` loads), about 2x slower. The
contiguous arg-reduction runs a lane-parallel search for the winning key and
then a scan for its first occurrence, which keeps the lowest-index rule.

### Codegen evidence

`perf annotate` of the decode case (`d = 1024, len = 64`, f32, layer norm with
weight and bias), default target. Before the change, the hottest symbol was
the variance pass outside the dispatched function, two `f32` lanes per
instruction:

```
vmovsd 0x60(%rsi,%r9,4),%xmm11
vsubps %xmm1,%xmm11,%xmm11
vsubps %xmm2,%xmm11,%xmm11
vmulps %xmm11,%xmm11,%xmm11
vaddps %xmm7,%xmm11,%xmm7
```

After it, the output pass (the hottest loop, 45% of samples) and the sums run
on 256-bit registers inside the `pulp` V3 function:

```
vmovups -0x60(%r8,%r9,4),%ymm4
vsubps  %ymm1,%ymm4,%ymm4
vsubps  %ymm2,%ymm4,%ymm4
vmulps  %ymm4,%ymm3,%ymm4
vmulps  (%rax,%r9,4),%ymm4,%ymm4
vaddps  (%rcx,%r9,4),%ymm4,%ymm4
vmovups %ymm4,-0x60(%rdi,%r9,4)
```

The same case went from 45 us to 19 us per call.

## Measurements

`cargo bench -p strided-kernel --features parallel --bench axis_ops`, default
target (no `target-cpu=native`), AMD EPYC 7713P, shared host with load
average 59-79 during the run (indicative, not a controlled campaign). Medians
in microseconds; 4T is `ExecContext::max_threads(4)` in a four-worker pool.
The composed layer norm is the sum, scale, subtract, sum of squares, rsqrt,
multiply, weight and bias sequence of existing erased plans with
preallocated outputs, so it excludes tenferro's per-op allocation and
dispatch.

| layer_norm f32 case | fused 1T | composed 1T | fused 4T | composed 4T |
|---|---:|---:|---:|---:|
| d=1024 len=8 batch=1 | 2.5 | 35.2 | 2.5 | 35.0 |
| d=1024 len=64 batch=1 | 19.3 | 66.4 | 11.0 | 86.0 |
| d=1024 len=8 batch=8 | 20.0 | 240.6 | 14.0 | 277.9 |
| d=1024 len=64 batch=8 | 153.0 | 492.7 | 79.5 | 585.9 |
| d=1024 len=16384 (64 MiB) | 9227 | 39437 | 3539 | 28080 |
| strided axis, 512 x 1024 | 425.8 | 582.8 | 133.8 | 495.1 |

| f32 case | cumsum 1T | naive 1T | argmax 1T | naive 1T | cumsum 4T | argmax 4T |
|---|---:|---:|---:|---:|---:|---:|
| 1024 x 64, unit-stride axis | 62.8 | 64.2 | 34.8 | 32.6 | 18.3 | 13.6 |
| 1024 x 64, strided axis | 23.1 | 496.2 | 44.0 | 218.2 | 9.0 | 13.9 |
| 4096 x 4096, unit-stride axis | 16491 | 16980 | 5147 | 8717 | 4268 | 1536 |

The naive scan and argmax loops are raw pointer loops line by line; the naive
argmax ignores NaN. A unit-stride cumsum is bound by the add latency of the
sequential fold, which the fixed fold order rules out shortening; interleaving
independent lines is a possible follow-up.
