# Performance Comparison: strided-opteinsum vs OMEinsum.jl

This document summarizes the key optimizations that make strided-opteinsum
faster than OMEinsum.jl on the
[einsum benchmark](https://benchmark.einsum.org/) suite.

Both systems use the same pre-computed contraction paths for fair comparison
(omeinsum_path mode). Benchmarks are run on AMD EPYC 7713P (Zen 3) and
Apple Silicon M2. See the
[benchmark suite](https://github.com/tensor4all/strided-rs-benchmark-suite)
for full results.

## Summary of Advantages

| Optimization | Biggest impact | Typical speedup |
|---|---|---|
| Lazy permutation + source-order copy | Tensor networks (many small dims) | 1.3–1.7x |
| Parallel copy/permutation (rayon) | 4T tensor networks | 1.7x vs Julia |
| HPTT-based permutation | Writeback and finalize paths | 6x vs naive |
| Buffer reuse | All instances | reduced allocation overhead |
| faer GEMM | 1T small-to-medium matrices | competitive with OpenBLAS |

## 1. Lazy Permutation + Source-Order Copy

**The single largest source of speedup**, especially on tensor network
instances with many small dimensions (e.g. 24 binary dims of size 2).

### Design

OMEinsum.jl eagerly materializes every intermediate permutation via
`permutedims!`. strided-rs uses **metadata-only permutation**: between
contraction steps, only the dims/strides arrays are reordered — no data copy.

When a subsequent step needs contiguous input for GEMM, strided-rs copies
using **source-stride-order iteration** (`copy_strided_src_order`). Because
the source data is physically contiguous (col-major output from the previous
GEMM), iterating in source-stride order gives sequential reads that exploit
the hardware prefetcher. Scattered writes are absorbed by write-combining
buffers.

This is faster than HPTT (destination-stride-order) for this case because:
- HPTT's dst-order causes scattered reads from cold L3 cache
- For 24 binary dims, HPTT's bilateral fusion yields ~17 fused dims with a
  2x2 inner tile and 15 recursion levels — high per-element overhead

See [permutation-optimization.md](permutation-optimization.md) for detailed
analysis and bandwidth measurements.

### Impact

`tensornetwork_permutation_light_415` (AMD EPYC, 1T):
- strided-rs (faer): **283 ms**
- OMEinsum.jl: 476 ms
- Speedup: **1.68x**

## 2. Parallel Copy/Permutation

Copy and permutation operations are parallelized via rayon when the
`parallel` feature is enabled:

- `copy_strided_src_order_par` in strided-einsum2: splits outer source-stride
  dimensions across rayon threads (threshold: > 1M elements)
- `strided-kernel/parallel`: parallelizes `copy_into`, `map_into`, and other
  kernel operations via rayon
- `strided-perm/parallel`: parallelizes the outermost HPTT ComputeNode loop

**Important**: The `parallel` feature must propagate through the full
dependency chain: `strided-opteinsum/parallel` → `strided-einsum2/parallel`
→ `strided-kernel/parallel` → `strided-perm/parallel`. A bug where
`strided-kernel/parallel` was not enabled was fixed in
[840d9b8](https://github.com/tensor4all/strided-rs/commit/840d9b8).

### Impact

`tensornetwork_permutation_light_415` (AMD EPYC, 4T):
- strided-rs (faer): **222 ms**
- OMEinsum.jl: 388 ms
- Speedup: **1.75x**

## 3. HPTT-Based Permutation

The `strided-perm` crate implements cache-efficient tensor transpose based on
HPTT (Springer et al. 2017):

- Bilateral dimension fusion
- 4x4 (f64) / 8x8 (f32) micro-kernel transpose
- Macro-kernel blocking (BLOCK=16 for f64)
- Recursive loop nest with stride-1 fast path

Used in writeback paths (`finalize_into`), BGEMM packing, and single-tensor
operations where HPTT's blocked approach is effective (warm cache, regular
stride patterns).

See [permutation-optimization.md](permutation-optimization.md) for details.

## 4. Buffer Reuse

strided-opteinsum reuses intermediate buffers across contraction steps via a
thread-local pool (`alloc_col_major_uninit_with_pool`). This avoids repeated
allocation/deallocation of large temporary arrays during multi-step
contractions.

OMEinsum.jl allocates fresh arrays at each step.

## 5. faer GEMM

The [faer](https://github.com/sarah-quinones/faer-rs) backend provides a
pure-Rust GEMM that is competitive with OpenBLAS, especially for
small-to-medium matrices at 1 thread. On the benchmark suite, faer often
matches or beats OpenBLAS at 1T, and significantly outperforms at 4T when
combined with rayon-parallelized copy operations (since faer and rayon share
the same thread pool, avoiding the OMP/rayon dual-pool overhead).

## Where OMEinsum.jl Still Wins

- `str_nw_mera_closed_120` (opt_size, 1T): Julia 1296 ms vs strided-rs
  1363 ms. MERA networks are GEMM-dominated with large matrices where
  Julia's OpenBLAS 0.3.29 via libblastrampoline is slightly faster.
- Some 4T instances show Julia winning on small-tensor workloads where
  strided-rs threading overhead exceeds the parallelism benefit.
