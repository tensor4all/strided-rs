# Permutation Optimization in strided-rs

This document summarizes the permutation-related optimizations introduced in
strided-rs (February 2026), their motivation, and remaining design questions.

## Background: Lazy vs Eager Permutation

When a binary einsum step produces output whose index order differs from the
internal `[lo, ro, batch]` GEMM layout, a permutation is needed. Two strategies
exist:

### OMEinsum.jl (eager)

Julia always materializes the output permutation immediately after GEMM via
`permutedims!`. Every intermediate tensor is contiguous col-major, so
subsequent steps read from sequential memory.

### strided-rs original (lazy)

The original design applied the output permutation as **metadata-only**: it
reordered the dims/strides arrays without copying data. This was zero-cost at
the current step, but left the tensor with non-contiguous (scattered) strides.

When a subsequent step needed to copy this tensor to contiguous layout for GEMM
input preparation, it had to read data with scattered access patterns — for
example strides like `[1, 2, 4, 8, 4194304, 16, 8388608, ...]`. This is
extremely cache-unfriendly:

| Method | Bandwidth (16M f64 elems) |
|--------|--------------------------|
| memcpy (sequential) | 63 GB/s |
| HPTT-style blocked copy | 37 GB/s |
| **Naive scattered copy** | **4 GB/s** |

The 16x bandwidth gap made the lazy approach a net loss for tensor networks with
many small dimensions (e.g., 24 dims of size 2), where scattered strides cause
massive cache line waste.

## Solution: HPTT-Inspired Permutation + Direct `strided-perm` Integration

The fix has two parts:

### 1. HPTT-faithful cache-efficient permutation (`strided-perm` crate)

Extracted permutation logic into a dedicated `strided-perm` crate implementing
the key techniques from HPTT (High-Performance Tensor Transpose, Springer et
al. 2017):

1. **Bilateral dimension fusion** — fuse consecutive dimensions contiguous in
   both source and destination
2. **2D micro-kernel transpose** — 4x4 scalar for f64, 8x8 for f32
3. **Macro-kernel blocking** — BLOCK x BLOCK tile (16 for f64, 32 for f32)
4. **Recursive ComputeNode loop nest** — mirrors HPTT's loop structure; only
   stride-1 dims get blocked
5. **ConstStride1 fast path** — when src and dst stride-1 dims coincide, uses
   memcpy/strided-copy
6. **Optional Rayon parallelism** — parallelize the outermost ComputeNode
   dimension

This reduced scattered-stride permutation from 67 ms to ~11 ms (1T) on the
benchmark case.

### 2. Direct `strided-perm` in einsum2 writeback path

Changed `strided-einsum2`'s `finalize_into` to call `strided_perm::copy_into`
directly instead of going through `strided_kernel::copy_into` (which fell back
to the non-HPTT `map_into` path). This ensures the HPTT-optimized permutation
is used when copying GEMM results back to non-contiguous destinations.

## Current Strategy: Lazy Permutation + Source-Stride-Order Copy

strided-rs keeps the lazy (metadata-only) permutation but uses two optimized
copy strategies when a subsequent step needs contiguous data:

### 1. Source-stride-order copy in `prepare_input_owned`

Since einsum2 always produces col-major output, the source of
`prepare_input_owned` is physically contiguous in memory — only the
dims/strides metadata is permuted. The function `copy_strided_src_order`
iterates in **source-stride order** (smallest source stride innermost), giving
sequential reads that exploit the hardware prefetcher on cold-cache data.
Scattered writes are absorbed by hardware write-combining buffers.

This replaces HPTT for this specific path because HPTT iterates in
*destination*-stride order — sequential writes, scattered reads. For large
cold-cache data with many small dimensions (e.g. 24 binary dims), the scattered
reads dominate performance. Additionally, HPTT's bilateral fusion can only merge
consecutive dimensions; for 24 binary dims with scattered strides this leaves
~17 fused dims with a 2×2 inner tile and 15 recursion levels — high per-element
overhead.

With `--features parallel`, a rayon-parallelized variant
(`copy_strided_src_order_par`) splits the outer source-stride dimensions across
threads, with automatic fallback to single-threaded when `RAYON_NUM_THREADS=1`
or the tensor is small (< 1M elements).

### 2. HPTT for other copy paths

The rest of the pipeline (`finalize_into`, `bgemm_faer` pack, `single_tensor`,
`operand`) still uses HPTT via `strided_kernel::copy_into` /
`strided_perm::copy_into`. These paths typically operate on warm-cache data or
have different stride patterns where HPTT's blocked approach remains effective.

### Why not always-materialize?

- **No extra copy when not needed** — if the next step's canonical order aligns,
  `try_fuse_group` succeeds and no copy occurs (truly zero cost)
- **Source-order copy is fast enough** — sequential reads on contiguous source
  achieve near-memcpy bandwidth

## Benchmark Results

`tensornetwork_permutation_light_415` (415 tensors, 24 binary dims, AMD EPYC
7713P):

| Configuration | opt_flops (ms) | vs OMEinsum.jl (388 ms) |
|---------------|---------------:|------------------------:|
| Original (HPTT) 1T | 455 | 1.17x slower |
| Source-order copy 1T | 298 | **1.30x faster** |
| Source-order copy + parallel 4T | 228 | **1.70x faster** |

The source-order copy alone yields a 34% improvement over HPTT. Adding
parallel copy with 4 threads provides a further 24% speedup.

## Open Questions

### Extending source-order copy to other paths

Currently only `prepare_input_owned` uses source-stride-order copy. Other copy
paths (`finalize_into`, etc.) could also benefit when source data is contiguous
but strides are scattered. This would require detecting contiguous-source
patterns at each call site.

### Thread scaling

With 4 threads the parallel copy helps significantly, but the improvement may
plateau with more threads as memory bandwidth saturates. Benchmarking with
higher thread counts on different architectures would clarify the scaling
characteristics.

## Related Issues and PRs

- [#103](https://github.com/tensor4all/strided-rs/issues/103) — Analysis of
  eager materialization vs lazy permutation
- [#104](https://github.com/tensor4all/strided-rs/issues/104) — Extract
  `strided-perm` crate
- [#105](https://github.com/tensor4all/strided-rs/issues/105) — Umbrella issue
  for HPTT-inspired permutation
- [#109](https://github.com/tensor4all/strided-rs/issues/109) — Always-
  materialize proposal
- [#111](https://github.com/tensor4all/strided-rs/pulls/111) — HPTT
  implementation PR
- [#112](https://github.com/tensor4all/strided-rs/pulls/112) — HPTT module
  cleanup and rewrite
