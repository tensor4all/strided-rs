# Benchmark Report: mdarray-strided v0.1

**Date:** January 23, 2026
**Platform:** macOS (darwin 25.2.0)

This report summarizes performance benchmarks comparing strided array operations against naive implementations.

## Executive Summary

The strided kernels show significant performance improvements in most operations:

| Operation | Size | Speedup | Notes |
|-----------|------|---------|-------|
| `copy_permuted` | 1000×1000 | **0.75x slower** | Simple permutation copy |
| `zip_map_mixed` | 1000×1000 | **1.46x faster** | Mixed stride access (transpose + add) |
| `reduce_transposed` | 1000×1000 | **1.07x faster** | Sum of transposed array |
| `symmetrize_aat` | 4000×4000 | **1.55x faster** | `B = (A + A') / 2` |
| `scale_transpose` | 1000×1000 | **1.46x faster** (median) | `B = 3.0 * A'` |
| `nonlinear_map` | 1000×1000 | **≈same** | element-wise ops |
| `permutedims_4d` | 32×32×32×32 | **≈same** | Full 4D transpose |
| `multi_permute_sum` | 32×32×32×32 | **2.02x faster** | Sum of 4 permutations |

## Detailed Results

### 1. Copy Permuted (1000×1000)

**Operation:** `B = permutedims(A, (2,1))` (transpose copy)

```
naive:    503.77 µs - 513.88 µs  (1.95-1.99 Gelem/s)
strided:  624.74 µs - 734.63 µs  (1.36-1.60 Gelem/s)
```

**Result:** Strided is **0.75x slower** (median: 675.84 µs vs 508.65 µs)

**Analysis:** Naive `to_tensor()` benefits from highly optimized bulk copy routines. The strided implementation's blocking overhead dominates for simple transpose operations. This is acceptable as transpose is typically composed with other operations.

---

### 2. Zip Map with Mixed Strides (1000×1000)

**Operation:** `out = A' + B` (add transposed array to contiguous array)

```
naive:    957.45 µs - 987.10 µs
strided:  657.02 µs - 668.78 µs
```

**Result:** Strided is **1.46x faster** (969.41 µs → 662.33 µs)

**Analysis:** Cache-optimized blocking strategy significantly improves performance when mixing strided and contiguous access patterns. This is the primary use case for the library.

---

### 3. Reduce Transposed (1000×1000)

**Operation:** `sum(A')` (sum elements of transposed array)

```
naive:    606.25 µs - 623.13 µs
strided:  573.96 µs - 578.03 µs
```

**Result:** Strided is **1.07x faster** (613.41 µs → 575.84 µs)

**Analysis:** For larger arrays (1000×1000), cache optimization provides measurable benefit (~7% faster). The dimension reordering and blocking strategy improves cache locality during reduction.

---

### 4. Symmetrize AAT (4000×4000)

**Operation:** `B = (A + A') / 2` (symmetrize matrix)

```
naive:    47.543 ms - 48.082 ms  (333-337 Melem/s)
strided:  30.639 ms - 31.216 ms  (513-522 Melem/s)
```

**Result:** Strided is **1.55x faster** (47.83 ms → 30.91 ms)

**Analysis:** Large arrays benefit significantly from blocked iteration that fits in L1 cache (32KB blocks). This is one of the best showcases for the library.

---

### 5. Scale Transpose (1000×1000)

**Operation:** `B = 3.0 * A'` (scale and transpose)

```
naive:    633.20 µs - 728.41 µs  (1.37-1.58 Gelem/s)
strided:  361.79 µs - 548.14 µs  (1.82-2.76 Gelem/s)
```

**Result:** Strided is **1.46x faster** (median: 676.41 µs → 464.28 µs)

**Analysis:** Specialized `copy_transpose_scale_into_fast` kernel outperforms naive nested loops. Note the higher variance in strided results likely due to cache state sensitivity.

---

### 6. Nonlinear Map (1000×1000)

**Operation:** `B = A * exp(-2*A) + sin(A*A)` (complex element-wise function)

```
naive:    10.330 ms - 11.026 ms  (90.7-96.8 Melem/s)
strided:  10.650 ms - 10.873 ms  (92.0-93.9 Melem/s)
```

**Result:** **≈same performance** (10.64 ms vs 10.72 ms)

**Analysis:** When computation cost dominates memory access, cache optimization provides minimal benefit. Both approaches are compute-bound, spending time in `exp` and `sin` rather than memory operations.

---

### 7. Permutedims 4D (32×32×32×32)

**Operation:** `permutedims!(B, A, (4,3,2,1))` (full 4D transpose)

```
naive:    953.29 µs - 984.78 µs  (1.06-1.10 Gelem/s)
strided:  950.73 µs - 978.79 µs  (1.07-1.10 Gelem/s)
```

**Result:** **≈same performance** (964.98 µs vs 963.01 µs)

**Analysis:** For 4D arrays with full dimension reversal, current blocking strategy does not provide advantage. See [Issue #5](https://github.com/AtelierArith/strided-rs-private/issues/5) for planned optimizations.

---

### 8. Multi-Permute Sum (32×32×32×32)

**Operation:** `B = A + permute(A, p1) + permute(A, p2) + permute(A, p3)` (sum 4 permutations)

```
naive:           3.5791 ms - 3.9922 ms  (263-293 Melem/s)
strided_fused:   1.8844 ms - 1.9637 ms  (534-556 Melem/s)
```

**Result:** Strided is **2.02x faster** (3.81 ms → 1.91 ms)

**Analysis:** `zip_map4_into` demonstrates the power of lazy evaluation and single-pass fusion. Instead of materializing 4 intermediate arrays, strided performs a single-pass computation with cache-optimized access. This represents **best-case performance** for the library's design.

---

## Performance Patterns

### When Strided Wins (✅)

1. **Mixed stride patterns** (`zip_map_mixed`): 1.46x faster
2. **Large arrays with transpose** (`symmetrize_aat`): 1.55x faster
3. **Fused multi-array operations** (`multi_permute_sum`): 2.02x faster
4. **Scale + transpose** (`scale_transpose`): 1.46x faster

### When Strided Breaks Even (⚠️)

1. **Compute-bound operations** (`nonlinear_map`): ≈same
2. **4D permutations** (`permutedims_4d`): ≈same (needs optimization)

### When Strided Loses (❌)

1. **Simple transpose copy** (`copy_permuted`): 0.75x slower
   *Reason:* Naive `to_tensor()` uses bulk copy; blocking overhead not justified

---

## Recommendations

### Use Strided When:
- Combining multiple strided arrays in single operation
- Working with large arrays (>1MB)
- Mixing transposed/permuted views with contiguous data
- Fusing operations to avoid intermediate allocations

### Use Naive When:
- Simple single-array transpose/copy
- Small arrays (<10KB)
- Already compute-bound (complex math functions)

### Future Work:
- Optimize 4D blocking strategy ([Issue #5](https://github.com/AtelierArith/strided-rs-private/issues/5))
- Add fast-path detection for simple transpose
- Dynamic thread count tuning

---

## System Configuration

```
OS:         macOS 14.2 (darwin 25.2.0)
Compiler:   rustc (release build)
CPU:        (detected from BLOCK_MEMORY_SIZE = 32KB L1 cache)
Threading:  Rayon (parallel feature enabled)
```

## Methodology

All benchmarks use:
- `criterion` 0.5.1 with default settings
- 100 samples for fast benchmarks (<1ms)
- 10 samples for slow benchmarks (>1ms)
- 3-second warmup period
- Random data seeded with `StdRng` for reproducibility
