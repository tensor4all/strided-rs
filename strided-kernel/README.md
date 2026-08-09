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

## Map and Reduce Operations

```rust
use strided_kernel::{StridedArray, map_into, zip_map2_into, reduce};

let a = StridedArray::<f64>::from_fn_row_major(&[4, 5], |idx| idx[0] as f64);
let b = StridedArray::<f64>::from_fn_row_major(&[4, 5], |idx| idx[1] as f64);
let mut out = StridedArray::<f64>::row_major(&[4, 5]);

// Unary map: dest[i] = f(src[i])
map_into(&mut out.view_mut(), &a.view(), |x| x * 2.0).unwrap();

// Binary zip map: dest[i] = f(a[i], b[i])
zip_map2_into(&mut out.view_mut(), &a.view(), &b.view(), |x, y| x + y).unwrap();

// Full reduction
let total = reduce(&a.view(), |x| x, |a, b| a + b, 0.0).unwrap();
```

### Built-in Erased Reduction Order

The built-in erased `Sum`, `Product`, and `SumSquares` reductions have a
narrower, explicit determinism contract than the generic closure-based
`reduce` API:

- Compact full reductions traverse consecutive physical storage from the
  logical origin. Other full reductions use the stable fused/block traversal.
  Axis reductions enumerate reduced coordinates in declared axis order, with
  the first reduced axis varying fastest.
- Serial compact `f32`/`f64` full reductions use four SIMD accumulators when
  the `simd` feature is enabled. Other dtype/feature combinations use a fixed
  eight-lane scalar accumulator. Accumulator lanes are merged in stable order.
  Generic closure reductions retain their existing association.
- `SumSquares` accepts `f32` and `f64`. Each value is multiplied by itself and
  rounded in that dtype before the result enters the `Sum` accumulator. The
  multiply and add are not contracted into FMA, so overflow and underflow are
  classified before accumulation. The kernel does not materialize a squared
  tensor.
- `i32` and `i64` use wrapping sum/product. Floating and complex reductions use
  same-dtype arithmetic without widening, compensation, conjugation, or
  fast-math. Reassociation can change roundoff, signed zero, NaN details, and
  the point at which an intermediate overflows or underflows. Consequently,
  floating product may differ in finite/zero/infinite/NaN classification from
  a strict left fold.
- `ExecContext::serial()` and `ExecContext::max_threads(1)` use the same serial
  algorithm. A fixed nonzero thread budget has deterministic partition and
  merge order for a fixed executable, target, input, and layout; worker
  completion order does not affect the result. Different budgets may produce
  different floating or complex roundoff.
- Empty sum, product, and sum-of-squares return zero, one, and zero
  respectively. No tolerance or correctness gate is relaxed for the optimized
  association.

## High-Level Operations

```rust
use strided_kernel::{StridedArray, copy_into, add, dot, symmetrize_into};

let a = StridedArray::<f64>::from_fn_row_major(&[4, 4], |idx| (idx[0] * 10 + idx[1]) as f64);
let mut out = StridedArray::<f64>::row_major(&[4, 4]);

// Copy
copy_into(&mut out.view_mut(), &a.view()).unwrap();

// Element-wise add: dest[i] += src[i]
add(&mut out.view_mut(), &a.view()).unwrap();

// Dot product
let d = dot(&a.view(), &a.view()).unwrap();

// Symmetrize: dest = (src + src^T) / 2
symmetrize_into(&mut out.view_mut(), &a.view()).unwrap();
```

## Cache Optimization

The library automatically optimizes iteration order for cache efficiency:

1. **Dimension Fusion**: Contiguous dimensions are fused to reduce loop overhead
2. **Dimension Reordering**: Dimensions are sorted by stride magnitude for optimal memory access
3. **Tiled Iteration**: Operations are blocked to fit in L1 cache (32KB)
4. **Contiguous Fast Paths**: Contiguous arrays bypass blocking for direct iteration

## Parallel Feature

```toml
[dependencies]
strided-kernel = { version = "0.4", features = ["parallel"] }
```

The default `ExecutionPolicy::AmbientRayon` behavior uses the current installed
Rayon pool (or the global pool). Explicit runtimes can bound strided-owned
fanout without creating another pool:

```rust
use std::num::NonZeroUsize;
use strided_kernel::{
    map_into, with_execution_policy, ExecutionPolicy, StridedArray,
};

let source = StridedArray::<f64>::from_fn_col_major(&[4], |index| index[0] as f64);
let mut destination = StridedArray::<f64>::col_major(&[4]);
let max_threads = NonZeroUsize::new(2).unwrap();

with_execution_policy(ExecutionPolicy::Rayon { max_threads }, || {
    map_into(&mut destination.view_mut(), &source.view(), |value| value + 1.0).unwrap();
});
assert_eq!(destination.into_data(), vec![1.0, 2.0, 3.0, 4.0]);
```

`ExecutionPolicy` controls fanout, not CPU placement or pool construction.
Nested strided operations inside a bounded worker partition run sequentially.
The opaque parallel permutation-copy path is used under an explicit Rayon
policy only when the installed pool size fits the requested budget; otherwise
that copy falls back to its serial implementation because the external copy
cannot be capped per call.
The policy is worker-local; callbacks that depend on isolation from unrelated
work must not invoke their own Rayon scheduling or yielding APIs.

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

# Mul kernel comparison against PyTorch CPU at 1T and 4T.
# Requires uv; the script uses uv run --with torch --with numpy so PyTorch
# is installed only in the benchmark environment.
# The PyTorch runner uses torch.mul(..., out=...) to avoid allocator/autograd overhead.
# Noncompact batched outer product includes both compact output and a
# torchlike_output case whose non-contiguous output strides match torch.einsum.
uv --version
STRIDED_KERNEL_MUL_BENCH_PROFILE=full bash strided-kernel/benches/run_mul_pytorch_compare.sh 1 4
```

Published benchmark programs and current measured results live in
[`strided-rs-benchmark-suite`](https://github.com/tensor4all/strided-rs-benchmark-suite).

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
