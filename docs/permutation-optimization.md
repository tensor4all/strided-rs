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

## Current Strategy: Still Lazy, but with Fast Permutation

strided-rs currently **does NOT** always materialize like OMEinsum.jl. Instead,
it keeps the lazy (metadata-only) permutation but ensures that when a subsequent
step needs to copy the scattered tensor, it uses the HPTT-optimized path.

This is a pragmatic middle ground:

- **No extra copy when not needed** — if the next step's canonical order aligns,
  `try_fuse_group` succeeds and no copy occurs (truly zero cost)
- **Fast copy when needed** — HPTT permutation runs at ~25 GB/s instead of
  4 GB/s for the scattered case

### When lazy permutation wins

- Few dimensions with large sizes (e.g., `[1000, 1000, 1000]`) — even scattered
  reads have good cache line utilization
- Next step's access pattern aligns with current strides — no copy at all
- Final output — no subsequent step pays the deferred cost

### When eager materialization would win

- Many small dimensions (e.g., 24 dims of size 2) where the scattered copy,
  even with HPTT, is slower than two contiguous-to-contiguous copies
- Long chains of steps where the deferred cost propagates

## Benchmark Results

`tensornetwork_permutation_light_415` (415 tensors, 24 binary dims, Apple M2):

| Threads | strided-rs faer | OMEinsum.jl | Ratio |
|---------|----------------:|------------:|------:|
| 1T | 208 ms | 166 ms (IQR 83) | 1.25x |
| 4T | 142 ms | 172 ms (IQR 40) | **0.83x** |

strided-rs is now competitive (faster at 4T), with dramatically lower variance
(IQR < 4 ms vs 40-84 ms for OMEinsum.jl).

## Open Questions

### Always-materialize as a future option

Issue #109 proposed always materializing output permutations (matching
OMEinsum.jl). With HPTT-style permutation, the cost of eager materialization is
low (~7 ms for 16M elements). The remaining question is whether the benefit
outweighs the cost across all workload types:

- For tensor networks with many small dimensions: likely beneficial
- For workloads with large contiguous dimensions: likely wasteful
- A heuristic based on dimension count/sizes could be added

### Two-stage permutation

When a lazy permutation is followed by another permutation (e.g., from the next
step's input preparation), and the combined result is still non-contiguous,
strided-rs currently performs a single scattered-to-contiguous copy. An
alternative would be to split this into two stages:

1. First: permute the lazy tensor to contiguous (HPTT, fast)
2. Second: permute the contiguous result to the target layout (HPTT, fast)

Two contiguous-to-contiguous permutations can be faster than one
scattered-to-contiguous permutation because sequential reads have full cache
line utilization and the hardware prefetcher works effectively. This is exactly
the pattern that makes OMEinsum.jl's eager strategy work well.

This two-stage approach could be implemented as:
- Detect when the source has non-contiguous strides before calling
  `copy_into_col_major` or `strided_perm::copy_into`
- If so, first materialize to a temporary contiguous buffer, then permute from
  the temporary to the destination
- Only apply when the total element count exceeds a threshold (to avoid overhead
  for small tensors)

This would give the benefits of eager materialization without changing the
overall lazy architecture.

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
