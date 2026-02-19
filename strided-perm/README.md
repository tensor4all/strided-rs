# strided-perm

Cache-efficient tensor permutation / transpose, inspired by
[HPTT](https://github.com/springer13/hptt) (Springer et al., 2017).

## Techniques

1. **Bilateral dimension fusion** -- fuse consecutive dimensions that are
   contiguous in *both* source and destination stride patterns
   (equivalent to HPTT's `fuseIndices`).
2. **2D micro-kernel transpose** -- 4×4 scalar kernel for f64, 8×8 for f32.
3. **Macro-kernel blocking** -- BLOCK × BLOCK tile (16 for f64, 32 for f32)
   processed as a grid of micro-kernel calls, with scalar edge handling.
4. **Recursive ComputeNode loop nest** -- mirrors HPTT's linked-list loop
   structure; only stride-1 dims get blocked.
5. **ConstStride1 fast path** -- when src and dst stride-1 dims coincide,
   uses memcpy/strided-copy instead of the 2D transpose kernel.
6. **Optional Rayon parallelism** (`parallel` feature) -- parallelize the
   outermost ComputeNode dimension via `rayon::par_iter`.

### TODO

- **SIMD micro-kernels** -- the current scalar 4×4/8×8 kernels rely on LLVM
  auto-vectorization. Dedicated AVX2/NEON intrinsic kernels could further
  close the gap with HPTT C++.

## Benchmark Results

Environment: Apple M2, 8 cores, macOS.

All tensors use `f64` (8 bytes). "16M elements" = 128 MB read + 128 MB write.

### Single-threaded (1T)

| Scenario | strided-perm | naive | Speedup |
|---|---:|---:|---:|
| Scattered 24d (16M elems) | 11.0 ms (24 GB/s) | 38 ms (7.0 GB/s) | 3.5x |
| Contig→contig perm (24d) | 6.0 ms (45 GB/s) | 30 ms (9.1 GB/s) | 5.0x |
| Small tensor reverse (13d, 8K) | 0.035 ms (3.7 GB/s) | 0.015 ms (8.9 GB/s) | 0.4x |
| Small tensor cyclic (13d, 8K) | 0.004 ms (29 GB/s) | -- | -- |
| 256^3 transpose [2,0,1] | 17.1 ms (16 GB/s) | 45 ms (6.0 GB/s) | 2.6x |
| 256^3 transpose [1,0,2] | 15.0 ms (18 GB/s) | -- | -- |
| memcpy baseline | 4.5 ms (59 GB/s) | -- | -- |

### Multi-threaded (8T, `parallel` feature)

| Scenario | 1T | 8T | Speedup |
|---|---:|---:|---:|
| Scattered 24d (16M elems) | 15.7 ms (17 GB/s) | 7.8 ms (35 GB/s) | 2.0x |
| Contig→contig perm (24d) | 6.3 ms (43 GB/s) | 6.5 ms (42 GB/s) | ~1x |
| Small tensor reverse (13d, 8K) | 0.033 ms | 0.033 ms | 1.0x (below threshold) |
| 256^3 transpose [2,0,1] | 17.0 ms (16 GB/s) | 17.5 ms (15 GB/s) | ~1x |
| 256^3 transpose [1,0,2] | 15.8 ms (17 GB/s) | 6.3 ms (42 GB/s) | 2.5x |

### Notes

- **Scattered 24d**: 24 binary dimensions with non-contiguous strides from a
  real tensor-network workload. Parallel improvement is modest because bilateral
  fusion leaves few outer blocks to distribute.
- **Small tensor reverse**: Slower than naive because plan construction overhead
  dominates at 8K elements. The cyclic permutation fuses to fewer dims and is
  much faster.
- **256^3 transpose [2,0,1]**: Parallel speedup is limited because the outermost
  ComputeNode dimension is small after bilateral fusion.
- **Small tensor**: Below `MINTHREADLENGTH` (32K elements), the parallel path
  falls back to single-threaded, incurring no overhead.

## Running Benchmarks

```bash
# Single-threaded
RUSTFLAGS="-C target-cpu=native" cargo bench -p strided-perm

# With Rayon parallelism
RUSTFLAGS="-C target-cpu=native" cargo bench -p strided-perm --features parallel
```
