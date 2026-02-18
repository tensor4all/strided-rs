# strided-perm

Cache-efficient tensor permutation / transpose, inspired by
[HPTT](https://github.com/springer13/hptt) (Springer et al., 2017).

## Techniques

1. **Bilateral dimension fusion** -- fuse consecutive dimensions that are
   contiguous in *both* source and destination stride patterns.
2. **Cache-aware blocking** -- tile iterations to fit in L1 cache (32 KB).
3. **Optimal loop ordering** -- place the stride-1 dimension innermost for
   sequential memory access; sort outer dimensions by descending stride.
4. **Rank-specialized kernels** -- tight 1D/2D/3D blocked loops with no
   allocation overhead; generic N-D fallback with pre-allocated odometer.
5. **Optional Rayon parallelism** (`parallel` feature) -- parallelize the
   outermost block loop via `rayon::par_iter`.

## Benchmark Results

Environment: Linux, AMD 64-core server, `RUSTFLAGS="-C target-cpu=native"`.

All tensors use `f64` (8 bytes). "16M elements" = 128 MB read + 128 MB write.

### Single-threaded (1T)

| Scenario | strided-perm | naive | Speedup |
|---|---:|---:|---:|
| Scattered 24d (16M elems) | 30 ms (9.0 GB/s) | 84 ms (3.2 GB/s) | 2.8x |
| Contig->contig perm (24d) | 30 ms (8.9 GB/s) | 84 ms (3.2 GB/s) | 2.8x |
| Small tensor (13d, 8K elems) | 0.023 ms (5.7 GB/s) | 0.039 ms (3.4 GB/s) | 1.7x |
| 256^3 transpose [2,0,1] | 76 ms (3.6 GB/s) | 73 ms (3.7 GB/s) | ~1x |
| 256^3 transpose [1,0,2] | 37 ms (7.3 GB/s) | -- | -- |
| memcpy baseline | 5.8 ms (46 GB/s) | -- | -- |

### Multi-threaded (64T, `parallel` feature)

| Scenario | 1T | 64T | Speedup |
|---|---:|---:|---:|
| Scattered 24d (16M elems) | 30 ms (9.0 GB/s) | 23 ms (11.7 GB/s) | 1.3x |
| Contig->contig perm (24d) | 30 ms (8.9 GB/s) | 24 ms (11.4 GB/s) | 1.3x |
| Small tensor (13d, 8K elems) | 0.023 ms | 0.023 ms | 1.0x (below threshold) |
| 256^3 transpose [2,0,1] | 76 ms (3.6 GB/s) | 4.7 ms (56.8 GB/s) | 16x |
| 256^3 transpose [1,0,2] | 37 ms (7.3 GB/s) | 4.2 ms (64.1 GB/s) | 8.8x |

### Notes

- **Scattered 24d**: 24 binary dimensions with non-contiguous strides from a
  real tensor-network workload. Parallel improvement is modest because bilateral
  fusion leaves few outer blocks to distribute.
- **256^3 transpose**: Parallel execution yields dramatic speedup (16x) by
  exploiting the large L3 cache and memory bandwidth of the 64-core machine.
  Single-threaded performance is TLB-limited due to stride-65536 access.
- **Small tensor**: Below `MINTHREADLENGTH` (32K elements), the parallel path
  falls back to single-threaded, incurring no overhead.

## Running Benchmarks

```bash
# Single-threaded
RUSTFLAGS="-C target-cpu=native" cargo bench -p strided-perm

# With Rayon parallelism
RUSTFLAGS="-C target-cpu=native" cargo bench -p strided-perm --features parallel
```
