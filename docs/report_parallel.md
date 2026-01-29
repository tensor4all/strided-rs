# Parallel Benchmark Report: strided-rs v0.1

**Date:** January 23, 2026
**Platform:** macOS (darwin 25.2.0)
**Features:** `--features parallel`

This report compares sequential (`zip_map2_into`) vs parallel (`par_zip_map2_into`) implementations to measure the benefit of the `parallel` feature.

## Executive Summary

| Scenario | Size | Parallel Speedup | Recommendation |
|----------|------|------------------|----------------|
| **Compute-heavy** | 500×500 | **4.6x faster** | Use parallel |
| **Compute-heavy** | 1000×1000 | **5.7x faster** | Use parallel |
| **Compute-heavy** | 2000×2000 | **5.6x faster** | Use parallel |
| **Both transposed** | 2000×2000 | **4.5x faster** | Use parallel |
| **Both transposed** | 4000×4000 | **4.6x faster** | Use parallel |
| **4D arrays** | 32⁴ | **3.2x faster** | Use parallel |
| **4D arrays** | 48⁴ | **4.1x faster** | Use parallel |
| **Mixed strides** | 1000×1000 | 0.97x (≈same) | Use sequential |
| **Contiguous** | Any | 0.9-1.0x | Use sequential |

## Key Findings

### 1. Compute-Heavy Operations Benefit Most

When the per-element computation is expensive (exp, sin, cos), parallel execution provides **4.6-5.7x speedup**.

```
par_zip_map2_compute_heavy/500:
  sequential:  3.51 ms  (71.2 Melem/s)
  parallel:    756 µs   (330.7 Melem/s)  → 4.6x faster

par_zip_map2_compute_heavy/1000:
  sequential:  15.46 ms (64.7 Melem/s)
  parallel:    2.72 ms  (367.8 Melem/s) → 5.7x faster

par_zip_map2_compute_heavy/2000:
  sequential:  58.30 ms (68.6 Melem/s)
  parallel:    10.48 ms (381.7 Melem/s) → 5.6x faster
```

### 2. Poor Memory Access Patterns Benefit from Parallel

When both arrays are transposed (worst-case memory access), parallelism helps by distributing cache misses:

```
par_zip_map2_both_transposed/2000:
  sequential:  18.15 ms (220.4 Melem/s)
  parallel:    4.03 ms  (992.9 Melem/s) → 4.5x faster

par_zip_map2_both_transposed/4000:
  sequential:  93.32 ms (171.5 Melem/s)
  parallel:    20.15 ms (794.0 Melem/s) → 4.6x faster
```

### 3. 4D Arrays Show Good Parallel Scaling

```
par_zip_map2_4d/32 (1M elements):
  sequential:  5.22 ms  (201.0 Melem/s)
  parallel:    1.63 ms  (643.1 Melem/s) → 3.2x faster

par_zip_map2_4d/48 (5.3M elements):
  sequential:  38.81 ms (136.8 Melem/s)
  parallel:    9.50 ms  (558.6 Melem/s) → 4.1x faster
```

### 4. Contiguous Arrays Don't Benefit from Parallel

For already cache-friendly access patterns, parallel overhead hurts performance:

```
par_zip_map2_contiguous/4000:
  sequential:  5.87 ms  (2.73 Gelem/s)
  parallel:    6.20 ms  (2.58 Gelem/s) → 0.95x (slower)
```

### 5. Small Arrays: Parallel Overhead Dominates

```
par_zip_map2_both_transposed/500:
  sequential:  112.5 µs (2.22 Gelem/s)
  parallel:    295.4 µs (846.3 Melem/s) → 0.38x (2.6x slower!)
```

## Detailed Results

### Mixed Strides (A' + B)

| Size | Sequential | Parallel | Speedup |
|------|------------|----------|---------|
| 500×500 | 178 µs | 388 µs | **0.46x** (slower) |
| 1000×1000 | 661 µs | 708 µs | **0.93x** (≈same) |
| 2000×2000 | 2.73 ms | 2.39 ms | **1.14x** |
| 4000×4000 | 11.0 ms | 9.66 ms | **1.14x** |

### Contiguous (A + B)

| Size | Sequential | Parallel | Speedup |
|------|------------|----------|---------|
| 500×500 | 90 µs | 154 µs | **0.58x** (slower) |
| 1000×1000 | 362 µs | 450 µs | **0.80x** (slower) |
| 2000×2000 | 1.46 ms | 1.57 ms | **0.93x** (≈same) |
| 4000×4000 | 5.87 ms | 6.20 ms | **0.95x** (≈same) |

### Compute-Heavy (exp + sin + cos)

| Size | Sequential | Parallel | Speedup |
|------|------------|----------|---------|
| 500×500 | 3.51 ms | 756 µs | **4.6x** |
| 1000×1000 | 15.46 ms | 2.72 ms | **5.7x** |
| 2000×2000 | 58.30 ms | 10.48 ms | **5.6x** |

### Both Transposed (A' + B')

| Size | Sequential | Parallel | Speedup |
|------|------------|----------|---------|
| 500×500 | 112 µs | 295 µs | **0.38x** (slower) |
| 1000×1000 | 898 µs | 925 µs | **0.97x** (≈same) |
| 2000×2000 | 18.15 ms | 4.03 ms | **4.5x** |
| 4000×4000 | 93.32 ms | 20.15 ms | **4.6x** |

### 4D Permuted Arrays

| Size | Elements | Sequential | Parallel | Speedup |
|------|----------|------------|----------|---------|
| 20⁴ | 160K | 671 µs | 246 µs | **2.7x** |
| 32⁴ | 1.05M | 5.22 ms | 1.63 ms | **3.2x** |
| 48⁴ | 5.31M | 38.81 ms | 9.50 ms | **4.1x** |

## Recommendations

### Use `par_zip_map2_into` When:

1. **Computation is expensive** (transcendental functions, complex math)
2. **Arrays are large** (>1M elements for memory-bound, >100K for compute-bound)
3. **Memory access is poor** (multiple transposed arrays, 4D permutations)
4. **Working with 4D+ arrays** with permuted strides

### Use `zip_map2_into` When:

1. **Arrays are small** (<500×500)
2. **Arrays are contiguous** (optimal cache access)
3. **Operation is simple** (addition, multiplication)
4. **Memory bandwidth is the bottleneck** (parallel adds synchronization overhead)

### Threshold Guidelines

| Operation Type | Use Parallel When |
|----------------|-------------------|
| Simple (add, mul) | >4000×4000 and non-contiguous |
| Compute-heavy | >250K elements |
| 4D arrays | >160K elements |
| Both transposed | >1M elements |

## System Configuration

```
OS:         macOS 14.2 (darwin 25.2.0)
Compiler:   rustc (release build)
CPU:        Apple Silicon (detected via rayon)
Threads:    Available via rayon::current_num_threads()
```

## Methodology

All benchmarks use:
- `criterion` 0.5.1
- 10 samples per benchmark
- 2-second warmup
- 5-second measurement time
- Random data seeded with `StdRng` for reproducibility
- Built with `--features parallel`

Run benchmarks: `cargo bench --features parallel --bench parallel_bench`
