# strided-kernel

Cache-optimized compute kernels over `strided-view` tensors.

## Scope

- Unary/Binary/N-ary map kernels (`map_into`, `zip_map*_into`)
- Reductions (`reduce`, `reduce_axis`)
- Utility ops (`copy_into`, `add`, `dot`, `sum`, `symmetrize_into`)
- Optional parallel execution via `parallel` feature (Rayon)

## Quick Example

```rust
use strided_kernel::{map_into, StridedArray};

let src = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1]) as f64);
let mut dst = StridedArray::<f64>::row_major(&[2, 3]);
map_into(&mut dst.view_mut(), &src.view(), |x| 2.0 * x).unwrap();
assert_eq!(dst.get(&[1, 2]), 10.0);
```

## Parallel Feature

```toml
[dependencies]
strided-kernel = { path = "../strided-kernel", features = ["parallel"] }
```

## Benchmarks

Run all benchmarks (single-threaded + multi-threaded, Rust + Julia):

```bash
bash strided-kernel/benches/run_all.sh        # default thread counts: 1 2 4
bash strided-kernel/benches/run_all.sh 1 2 4 8  # custom thread counts
```

Or individually:

```bash
# Single-threaded Rust
cargo bench --bench rust_compare --manifest-path strided-kernel/Cargo.toml

# Single-threaded Julia
JULIA_NUM_THREADS=1 julia --project=strided-kernel/benches strided-kernel/benches/julia_compare.jl

# Multi-threaded Rust (N threads)
RAYON_NUM_THREADS=N cargo bench --features parallel --bench threaded_compare --manifest-path strided-kernel/Cargo.toml

# Multi-threaded comparison script
bash strided-kernel/benches/run_threaded.sh 1 2 4

# Scaling benchmarks (sum + permute, 1/2/4 threads)
bash strided-kernel/benches/run_scaling.sh
bash strided-kernel/benches/run_scaling.sh 1 2 4 8  # custom thread counts

# Rank-25 tensor permutation (quantum circuit simulation workload)
RAYON_NUM_THREADS=1 cargo bench --bench rank25_permute --manifest-path strided-kernel/Cargo.toml

# Rank-25 Julia comparison
JULIA_NUM_THREADS=1 julia --project=strided-kernel/benches strided-kernel/benches/julia_rank25_compare.jl
```

### Single-Threaded Results

Environment: Apple Silicon M2, single-threaded.

| Case | Julia Strided (ms) | Rust strided (ms) | Rust naive (ms) |
|---|---:|---:|---:|
| symmetrize_4000 | 17.75 | 20.56 | 41.38 |
| scale_transpose_1000 | 0.53 | 0.79 | 0.42 |
| mwe_stridedview_scale_transpose_1000 | 0.51 | 0.81 | 0.40 |
| complex_elementwise_1000 | 7.75 | 12.91 | 12.27 |
| permute_32_4d | 0.90 | 1.23 | 2.03 |
| multiple_permute_sum_32_4d | 2.26 | 3.11 | 2.21 |

Notes:
- Julia results from `strided-kernel/benches/julia_compare.jl` (mean time). Rust results from `strided-kernel/benches/rust_compare.rs` (mean time).
- All benchmarks use column-major layout for parity with Julia.
- The Rust naive baseline uses raw pointer arithmetic with `unsafe` and precomputed strides (no bounds checks, no library overhead).
- `scale_transpose` and `multiple_permute_sum`: the naive baseline is faster because the ordering/blocking pipeline overhead is not recovered on these relatively small, simple access patterns. Julia Strided shows the same trend.

### Multi-Threaded Scaling (Rust, `parallel` feature)

Environment: Apple Silicon M2 (4 performance + 4 efficiency cores). Mean time.

| Case | 1T (ms) | 2T (ms) | 4T (ms) | Speedup (4T) |
|---|---:|---:|---:|---:|
| symmetrize_4000 | 20.88 | 17.78 | 11.33 | 1.8x |
| scale_transpose_1000 | 0.76 | 0.48 | 0.37 | 2.1x |
| mwe_scale_transpose_1000 | 0.69 | 0.53 | 0.41 | 1.7x |
| complex_elementwise_1000 | 12.85 | 6.61 | 3.52 | 3.7x |
| permute_32_4d | 1.09 | 0.64 | 0.48 | 2.3x |
| multiple_permute_sum_32_4d | 3.03 | 1.86 | 1.36 | 2.2x |
| sum_1m | 0.89 | 0.49 | 0.39 | 2.3x |

### Algorithm Comparison: Julia Strided.jl vs Rust strided-rs

Both implementations share the same core algorithm ported from Strided.jl:
1. **Dimension fusion** — merge contiguous dimensions to reduce loop depth
2. **Importance-weighted ordering** — bit-pack stride orders with output array weighted 2× to determine optimal iteration order
3. **L1 cache blocking** — iteratively halve block sizes until the working set fits in 32 KB
4. **Reversed loop nesting** — innermost loop operates on the highest-importance dimension (smallest stride) for optimal cache access

The key architectural differences are:

| Feature | Julia | Rust |
|---------|-------|------|
| **Kernel generation** | `@generated` unrolls loops per (rank, num\_arrays) at compile time | Handwritten 1D/2D/3D/4D specializations + generic N-D fallback |
| **Inner-loop SIMD** | Explicit `@simd` pragma on innermost loop | Stride-specialized inner loops: slice-based when stride=1, raw pointer otherwise; relies on LLVM auto-vectorization |
| **Threading** | Recursive dimension-splitting via `Threads.@spawn` | Recursive dimension-splitting via `rayon::join`; order-before-fuse pipeline enables layout-agnostic parallelization |

> **Note: Strided.jl threading bug for non-column-major views.**
> Julia's pipeline fuses before ordering (`fuse → order → block`), so
> `_mapreduce_fuse!` only detects column-major contiguity. Permuted views
> (e.g. `PermutedDimsArray(A, (2,1))`) with row-major strides are never fused,
> causing `_mapreduce_threaded!` to fall through to the single-threaded kernel.
> strided-rs fixes this by simply reordering the pipeline to `order → fuse →
> block`: ordering first puts smallest-stride dimensions adjacent, and a single
> fusion pass then catches contiguity regardless of memory layout. See
> [docs/strided\_jl\_threading\_bug.md](../docs/strided_jl_threading_bug.md) for a
> minimal reproduction and root cause analysis.

### Scaling Benchmarks (Strided.jl benchtests.jl suite)

These benchmarks measure performance across exponentially-scaled array sizes, showing the
crossover point where the ordering/blocking pipeline overhead pays off.

Run with: `bash strided-kernel/benches/run_scaling.sh` (default: 1 2 4 threads)

Environment: Apple Silicon M2 (4P + 4E cores). Median timing, adaptive iteration count.

#### 1D Sum

| size | Rust naive (μs) | Rust strided 1T | 2T | 4T | Julia base (μs) | Julia Strided 1T | 2T | 4T |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.04 | 0.04 | 0.00 | 0.00 | 0.00 | 0.04 | 0.04 | 0.04 |
| 12 | 0.04 | 0.04 | 0.00 | 0.00 | 0.00 | 0.04 | 0.04 | 0.04 |
| 32 | 0.04 | 0.04 | 0.00 | 0.04 | 0.00 | 0.04 | 0.04 | 0.04 |
| 91 | 0.08 | 0.04 | 0.04 | 0.04 | 0.00 | 0.08 | 0.08 | 0.08 |
| 256 | 0.25 | 0.08 | 0.04 | 0.04 | 0.04 | 0.17 | 0.17 | 0.17 |
| 725 | 0.92 | 0.17 | 0.08 | 0.08 | 0.08 | 0.33 | 0.33 | 0.38 |
| 2048 | 2.25 | 0.29 | 0.25 | 0.25 | 0.21 | 0.92 | 0.96 | 0.96 |
| 5793 | 5.29 | 0.63 | 0.67 | 0.67 | 0.50 | 2.58 | 2.54 | 2.63 |
| 16384 | 13.96 | 1.83 | 1.83 | 1.83 | 1.54 | 7.08 | 7.25 | 7.29 |
| 46341 | 39.58 | 40.67 | 35.67 | 37.75 | 4.42 | 19.96 | 26.67 | 16.67 |
| 131072 | 112.13 | 113.04 | 75.29 | 76.63 | 12.58 | 57.75 | 46.67 | 24.21 |
| 370728 | 317.33 | 318.29 | 185.50 | 188.13 | 35.25 | 163.13 | 101.73 | 61.54 |
| 1048576 | 939.67 | 935.33 | 497.04 | 430.04 | 103.25 | 461.63 | 258.71 | 162.83 |

Notes:
- For 1D contiguous sum, Rust strided can beat the naive loop at small/medium sizes due to an explicit SIMD reduction kernel.
- With the Rust `parallel` feature enabled, sizes above the threading threshold route through the threaded kernel path (even at 1T), so 1T can be slightly slower than the naive loop until multi-threading is enabled.
- Julia's `Base.sum` uses hand-tuned SIMD reduction; Julia's `@strided sum` includes kernel-dispatch overhead compared to `Base.sum`.

#### 4D Permute: (4,3,2,1) — full reversal

| s | s⁴ | Rust naive (μs) | Rust strided 1T | 2T | 4T | Julia copy (μs) | Julia Strided 1T | 2T | 4T |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 256 | 0.21 | 1.29 | 1.29 | 1.33 | 0.04 | 0.29 | 0.29 | 0.29 |
| 8 | 4096 | 2.04 | 3.92 | 3.92 | 3.92 | 0.46 | 2.00 | 1.79 | 1.83 |
| 12 | 20736 | 14.33 | 14.04 | 14.00 | 14.00 | 2.38 | 8.71 | 8.75 | 8.54 |
| 16 | 65536 | 54.71 | 50.46 | 48.88 | 48.83 | 7.92 | 31.92 | 34.63 | 40.21 |
| 24 | 331776 | 275.50 | 190.79 | 139.29 | 131.54 | 40.29 | 127.13 | 84.02 | 84.75 |
| 32 | 1048576 | 1119.75 | 1152.58 | 715.42 | 502.50 | 164.17 | 899.54 | 477.90 | 329.10 |
| 48 | 5308416 | 16857.92 | 8614.17 | 5567.96 | 3687.67 | 1077.00 | 7558.15 | 4879.65 | 3206.54 |
| 64 | 16777216 | 85412.17 | 50799.25 | 27581.00 | 23729.00 | 3593.38 | 47799.17 | 25965.08 | 15537.29 |

#### 4D Permute: (2,3,4,1)

| s | s⁴ | Rust naive (μs) | Rust strided 1T | 2T | 4T | Julia copy (μs) | Julia Strided 1T | 2T | 4T |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 256 | 0.21 | 1.17 | 1.17 | 1.21 | 0.04 | 0.25 | 0.25 | 0.29 |
| 8 | 4096 | 2.04 | 3.79 | 3.83 | 3.96 | 0.46 | 1.63 | 1.63 | 1.63 |
| 12 | 20736 | 17.33 | 13.83 | 13.79 | 13.83 | 2.42 | 8.96 | 8.88 | 8.88 |
| 16 | 65536 | 46.13 | 38.04 | 39.04 | 39.42 | 7.75 | 25.79 | 29.33 | 30.50 |
| 24 | 331776 | 231.71 | 171.88 | 114.33 | 120.88 | 42.38 | 129.17 | 85.29 | 77.71 |
| 32 | 1048576 | 762.08 | 784.88 | 498.58 | 356.42 | 154.79 | 671.00 | 407.04 | 226.33 |
| 48 | 5308416 | 16466.67 | 6234.46 | 4795.38 | 3023.21 | 1089.81 | 6168.71 | 4060.40 | 2800.46 |
| 64 | 16777216 | 72763.71 | 21973.96 | 15040.88 | 9475.75 | 3552.79 | 18350.92 | 13969.13 | 8989.65 |

#### 4D Permute: (3,4,1,2)

| s | s⁴ | Rust naive (μs) | Rust strided 1T | 2T | 4T | Julia copy (μs) | Julia Strided 1T | 2T | 4T |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 256 | 0.21 | 1.21 | 1.17 | 1.17 | 0.04 | 0.29 | 0.29 | 0.29 |
| 8 | 4096 | 2.08 | 3.88 | 3.88 | 3.88 | 0.46 | 1.50 | 1.46 | 1.46 |
| 12 | 20736 | 10.04 | 13.79 | 13.79 | 13.79 | 2.42 | 7.42 | 7.50 | 7.46 |
| 16 | 65536 | 47.75 | 39.00 | 41.96 | 34.92 | 7.67 | 26.58 | 30.75 | 46.00 |
| 24 | 331776 | 231.58 | 198.50 | 138.83 | 129.25 | 39.71 | 116.58 | 78.67 | 91.96 |
| 32 | 1048576 | 1118.71 | 888.75 | 569.04 | 413.58 | 161.50 | 1102.48 | 606.48 | 393.83 |
| 48 | 5308416 | 8661.25 | 6699.79 | 4990.75 | 2955.79 | 1111.21 | 5649.42 | 4040.88 | 2726.33 |
| 64 | 16777216 | 64188.00 | 28535.00 | 19057.08 | 12836.33 | 3574.73 | 24472.02 | 17239.00 | 11821.58 |

#### Scaling observations

- **Crossover point**: Strided's ordering/blocking overhead is recovered at ~20K elements (s≥12 for 4D permute). Below this, the naive loop is faster.
- **Large arrays (s≥48)**: Rust strided achieves 0.3-0.6x of naive single-threaded, and further improves with threading.
- **Rust vs Julia strided (1T, s=64)**:
  - (4,3,2,1): Rust 50.8ms vs Julia 47.8ms — Rust 1.1x slower
  - (2,3,4,1): Rust 22.0ms vs Julia 18.4ms — Rust 1.2x slower
  - (3,4,1,2): Rust 28.5ms vs Julia 24.5ms — Rust 1.2x slower
- **Multi-threaded (4T, s=64)**: Both achieve ~2-4x speedup over single-threaded strided. Rust and Julia reach comparable absolute performance at large sizes.
- **Julia copy vs Rust naive**: Julia's `copy!` is ~10-25x faster than Rust's raw pointer loop for large s because Julia uses optimized `memcpy`; Rust's naive loop does element-by-element copy for parity with the permute benchmark.

#### Per-case analysis

**symmetrize\_4000** (Julia 17.7 ms, Rust 20.6 ms) —
Both use the general mapreduce kernel: dimension fusion → importance ordering → L1 cache blocking. Julia applies `@simd` on the innermost loop. Rust uses stride-specialized inner loops (slice-based when stride=1). Rust is ~1.2x slower.

**scale\_transpose\_1000** (Julia 0.53 ms, Rust 0.79 ms) —
Both follow the same importance-weighted ordering for a 2-array (dest + transposed src) operation. The naive baseline (0.42 ms) is faster because it writes contiguously without blocking overhead; the strided version pays for the ordering/blocking pipeline on a small array.

**mwe\_stridedview\_scale\_transpose\_1000** (Julia 0.51 ms, Rust 0.81 ms) —
Same operation as scale\_transpose\_1000 using `map_into` with a transposed view.

**complex\_elementwise\_1000** (Julia 7.8 ms, Rust 12.9 ms) —
Both arrays are contiguous, so the operation is compute-bound. The gap comes from Julia's `@simd` enabling aggressive auto-vectorization of transcendental functions (`exp`, `sin`), while Rust's LLVM generates more conservative code for the same operations.

**permute\_32\_4d** (Julia 0.90 ms, Rust 1.23 ms) —
Both nest loops with the highest-importance dimension innermost. The stride=1 specialization allows LLVM to vectorize the contiguous inner dimension effectively, but Rust is still ~1.4x slower here.

**multiple\_permute\_sum\_32\_4d** (Julia 2.3 ms, Rust 3.1 ms) —
Both compute a combined importance score over all 5 arrays (output + 4 inputs) and iterate in the optimal compromise order, but Rust is still ~1.4x slower.
