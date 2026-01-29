# BLAS Benchmark Report: strided-rs v0.1

**Date:** January 23, 2026
**Platform:** macOS (darwin 25.2.0, Apple Silicon)
**BLAS Backend:** Apple Accelerate framework

This report compares generic implementations vs BLAS-backed implementations to measure the benefit of the `blas` feature.

## Executive Summary

| Operation | Size | BLAS Speedup | Recommendation |
|-----------|------|--------------|----------------|
| **dot** | 1K | **7x faster** | Use BLAS |
| **dot** | 10K | **26x faster** | Use BLAS |
| **dot** | 100K | **35x faster** | Use BLAS |
| **dot** | 1M | **6x faster** | Use BLAS |
| **axpy** | 1K | 0.75x (slower) | Use generic |
| **axpy** | 10K | **2.4x faster** | Use BLAS |
| **axpy** | 100K | **2.9x faster** | Use BLAS |
| **axpy** | 1M | **1.9x faster** | Use BLAS |
| **gemm** | 64×64 | **42x faster** | Use BLAS |
| **gemm** | 128×128 | **77x faster** | Use BLAS |
| **gemm** | 256×256 | **118x faster** | Use BLAS |

## Key Findings

### 1. GEMM (Matrix Multiplication) Benefits Most

BLAS gemm provides **40-120x speedup** over the generic O(n³) implementation:

```
gemm/64×64 (4K elements):
  generic: 112.48 µs  (36 Melem/s)
  blas:    2.77 µs    (1.48 Gelem/s)  → 42x faster

gemm/128×128 (16K elements):
  generic: 1.09 ms    (15 Melem/s)
  blas:    14.0 µs    (1.17 Gelem/s)  → 77x faster

gemm/256×256 (65K elements):
  generic: 10.70 ms   (6.1 Melem/s)
  blas:    90.8 µs    (722 Melem/s)   → 118x faster

gemm/512×512 (BLAS only):
  blas:    784 µs     (334 Melem/s)

gemm/1024×1024 (BLAS only):
  blas:    6.32 ms    (166 Melem/s)
```

### 2. Dot Product Shows Excellent Scaling

BLAS dot product achieves up to **35x speedup**:

```
dot_product/1K:
  generic: 655 ns  (1.53 Gelem/s)
  blas:    96 ns   (10.4 Gelem/s)   → 6.8x faster

dot_product/10K:
  generic: 6.81 µs (1.47 Gelem/s)
  blas:    260 ns  (38.5 Gelem/s)   → 26x faster

dot_product/100K:
  generic: 73.3 µs (1.36 Gelem/s)
  blas:    2.11 µs (47.3 Gelem/s)   → 35x faster

dot_product/1M:
  generic: 684 µs  (1.46 Gelem/s)
  blas:    119 µs  (8.4 Gelem/s)    → 5.7x faster
```

### 3. AXPY Has Overhead for Small Arrays

For small arrays, BLAS overhead dominates:

```
axpy/1K:
  generic: 494 ns  (2.03 Gelem/s)
  blas:    653 ns  (1.53 Gelem/s)   → 0.76x (SLOWER)

axpy/10K:
  generic: 5.94 µs (1.68 Gelem/s)
  blas:    2.43 µs (4.11 Gelem/s)   → 2.4x faster

axpy/100K:
  generic: 49.0 µs (2.04 Gelem/s)
  blas:    16.9 µs (5.91 Gelem/s)   → 2.9x faster

axpy/1M:
  generic: 551 µs  (1.82 Gelem/s)
  blas:    297 µs  (3.36 Gelem/s)   → 1.9x faster
```

### 4. f32 is Faster than f64

Single-precision achieves higher throughput:

```
gemm/256×256:
  f32: 28.6 µs  (2.29 Gelem/s)
  f64: 92.1 µs  (711 Melem/s)   → f32 is 3.2x faster

gemm/512×512:
  f32: 185 µs   (1.42 Gelem/s)
  f64: 780 µs   (336 Melem/s)   → f32 is 4.2x faster

gemm/1024×1024:
  f32: 1.56 ms  (672 Melem/s)
  f64: 6.18 ms  (170 Melem/s)   → f32 is 4.0x faster
```

### 5. Column-Major Handled Efficiently

BLAS handles column-major input via transpose flags with minimal overhead:

```
gemm_column_major/128×128:
  row-major blas: 14.0 µs
  col-major blas: 14.1 µs  → ~same performance
```

## Detailed Results

### Dot Product

| Size | Generic | BLAS | Speedup |
|------|---------|------|---------|
| 1K | 655 ns | 96 ns | **6.8x** |
| 10K | 6.81 µs | 260 ns | **26x** |
| 100K | 73.3 µs | 2.11 µs | **35x** |
| 1M | 684 µs | 119 µs | **5.7x** |

### AXPY (y = αx + y)

| Size | Generic | BLAS | Speedup |
|------|---------|------|---------|
| 1K | 494 ns | 653 ns | **0.76x** (slower) |
| 10K | 5.94 µs | 2.43 µs | **2.4x** |
| 100K | 49.0 µs | 16.9 µs | **2.9x** |
| 1M | 551 µs | 297 µs | **1.9x** |

### GEMM (C = αAB + βC)

| Size | Generic | BLAS | Speedup |
|------|---------|------|---------|
| 64×64 | 112 µs | 2.77 µs | **42x** |
| 128×128 | 1.09 ms | 14.0 µs | **77x** |
| 256×256 | 10.7 ms | 90.8 µs | **118x** |
| 512×512 | - | 784 µs | N/A |
| 1024×1024 | - | 6.32 ms | N/A |

### GEMM f32 vs f64

| Size | f32 BLAS | f64 BLAS | f32 Speedup |
|------|----------|----------|-------------|
| 256×256 | 28.6 µs | 92.1 µs | **3.2x** |
| 512×512 | 185 µs | 780 µs | **4.2x** |
| 1024×1024 | 1.56 ms | 6.18 ms | **4.0x** |

## Recommendations

### Use `blas_*` Functions When:

1. **Matrix multiplication** (always use BLAS for any size)
2. **Dot products** (always faster)
3. **AXPY with >5K elements**
4. **Need maximum performance** with f32 precision
5. **BLAS library is available** (Accelerate, OpenBLAS, MKL)

### Use `generic_*` Functions When:

1. **AXPY with <5K elements** (BLAS overhead dominates)
2. **Non-contiguous/strided arrays** (BLAS requires contiguous data)
3. **Complex element operations** (Conj, Transpose) that BLAS doesn't support
4. **Portability** without BLAS dependency

### Performance Guidelines

| Operation | Use BLAS When |
|-----------|---------------|
| dot | Always (any size) |
| axpy | >5,000 elements |
| gemm | Always (any size) |

## System Configuration

```
OS:           macOS 14.2 (darwin 25.2.0)
CPU:          Apple Silicon (M-series)
BLAS:         Apple Accelerate framework
Compiler:     rustc (release build)
```

## Methodology

All benchmarks use:
- `criterion` 0.5.1
- 10-20 samples per benchmark
- 1-2 second warmup
- 3-5 second measurement time
- Random data seeded with `StdRng` for reproducibility
- Built with `--features blas`

Run benchmarks: `cargo bench --features blas --bench blas_bench`

## Notes

- Generic implementations use simple O(n³) triple loop for gemm
- BLAS implementations use Apple's Accelerate framework (highly optimized for Apple Silicon)
- Results may vary significantly with different BLAS backends (OpenBLAS, MKL, etc.)
- Generic implementations work with any strided view; BLAS requires contiguous or stride-1 data
