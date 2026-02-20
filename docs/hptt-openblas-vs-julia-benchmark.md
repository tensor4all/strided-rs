# strided-rs (HPTT + OpenBLAS) vs OMEinsum.jl Benchmark

Branch: `perf/src-vs-dst-order`

## Purpose

Verify that strided-rs with HPTT as the default copy strategy (instead of
source-stride-order) remains competitive with OMEinsum.jl. If so, there is no
need for an adaptive copy strategy — HPTT can be the universal default.

## Setup

- **Rust**: strided-opteinsum with `blas` + `hptt-input-copy` features.
  Copy elision (`try_fuse_group`) is enabled; when a copy is needed, HPTT
  (destination-stride-order) is used. GEMM backend: OpenBLAS 0.3.29.
- **Julia**: OMEinsum.jl v0.9.3 with pre-computed contraction paths
  (`omeinsum_path` mode). Julia 1.10.0, BLAS vendor: lbt (OpenBLAS).
- **Hardware**: AMD EPYC 7713P
- **Timing**: median of 15 runs, 3 warmup

## Results

### 1T — opt_flops

| Instance | Rust (ms) | Julia (ms) | Ratio |
|---|---|---|---|
| gm_queen5_5_3 | 5209 | SKIP | — |
| lm_brackets_4_4d | 19 | 30 | **0.62x** |
| lm_sentence_3_12d | 76 | 80 | **0.95x** |
| lm_sentence_4_4d | 22 | 33 | **0.66x** |
| str_matrix_chain_100 | 14 | 16 | **0.84x** |
| str_mps_varying_200 | 15 | 37 | **0.42x** |
| mera_closed | 1361 | 1386 | **0.98x** |
| mera_open | 880 | 1251 | **0.70x** |
| tn_focus | 455 | 491 | **0.93x** |
| tn_light | 450 | 495 | **0.91x** |

### 1T — opt_size

| Instance | Rust (ms) | Julia (ms) | Ratio |
|---|---|---|---|
| gm_queen5_5_3 | 1632 | SKIP | — |
| lm_brackets_4_4d | 20 | 32 | **0.60x** |
| lm_sentence_3_12d | 60 | 92 | **0.66x** |
| lm_sentence_4_4d | 26 | 33 | **0.77x** |
| str_matrix_chain_100 | 12 | 17 | **0.69x** |
| str_mps_varying_200 | 17 | 35 | **0.48x** |
| mera_closed | 1173 | 1286 | **0.91x** |
| mera_open | 914 | 1298 | **0.70x** |
| tn_focus | 449 | 495 | **0.91x** |
| tn_light | 451 | 500 | **0.90x** |

### 4T — opt_flops

| Instance | Rust (ms) | Julia (ms) | Ratio |
|---|---|---|---|
| gm_queen5_5_3 | 4092 | SKIP | — |
| lm_brackets_4_4d | 20 | 42 | **0.49x** |
| lm_sentence_3_12d | 53 | 59 | **0.89x** |
| lm_sentence_4_4d | 23 | 46 | **0.51x** |
| str_matrix_chain_100 | 14 | 17 | **0.81x** |
| str_mps_varying_200 | 23 | 59 | **0.39x** |
| mera_closed | 587 | 931 | **0.63x** |
| mera_open | 353 | 798 | **0.44x** |
| tn_focus | 352 | 444 | **0.79x** |
| tn_light | 358 | 446 | **0.80x** |

### 4T — opt_size

| Instance | Rust (ms) | Julia (ms) | Ratio |
|---|---|---|---|
| gm_queen5_5_3 | 1202 | SKIP | — |
| lm_brackets_4_4d | 22 | 40 | **0.55x** |
| lm_sentence_3_12d | 39 | 63 | **0.63x** |
| lm_sentence_4_4d | 29 | 49 | **0.59x** |
| str_matrix_chain_100 | 8 | 16 | **0.48x** |
| str_mps_varying_200 | 20 | 44 | **0.45x** |
| mera_closed | 457 | 704 | **0.65x** |
| mera_open | 356 | 796 | **0.45x** |
| tn_focus | 353 | 457 | **0.77x** |
| tn_light | 356 | 464 | **0.77x** |

## Analysis

Rust with HPTT is **equal or faster than Julia on every instance**, in both
1T and 4T configurations.

- **1T**: Rust is 0.42x–0.98x of Julia (2–58% faster across instances)
- **4T**: Rust is 0.39x–0.89x of Julia (11–61% faster across instances)
- The 4T advantage is larger because strided-rs parallelizes both
  permutation copies (via rayon) and GEMM (via OpenBLAS threads), while
  Julia's OMEinsum only parallelizes GEMM

Even `tn_focus` and `tn_light` — the instances where HPTT is slower than
source-stride-order in isolation (see `src-vs-dst-order-experiment.md`) —
still outperform Julia. The copy elision (`try_fuse_group`) compensates for
HPTT's overhead on these degenerate many-small-dims cases.

## Conclusion

**HPTT can be the default copy strategy** for the `Contract` CPU backend.
There is no need for an adaptive strategy that switches between HPTT and
source-stride-order based on tensor shape. The combination of copy elision +
HPTT is sufficient to match or exceed Julia's OMEinsum.jl on all tested
workloads.

## Notes

- `gm_queen5_5_3` is skipped by Julia due to a `MethodError` (3D+ array
  incompatibility in OMEinsum.jl)
- Julia's IQR is generally higher than Rust's, suggesting more variance
  (likely due to GC pressure)
