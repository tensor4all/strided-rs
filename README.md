# strided-rs

Cache-optimized kernels for strided multidimensional array operations in Rust.

This crate is a port of Julia's [Strided.jl](https://github.com/Jutho/Strided.jl) and [StridedViews.jl](https://github.com/Jutho/StridedViews.jl) libraries.

This crate is currently built on top of the `mdarray` crate, but the long-term goal is to remove the `mdarray` dependency.

## Features

- **Zero-copy strided views** over contiguous memory
- **Lazy element operations** (conjugate, transpose, adjoint) with type-level composition
- **Zero-copy transformations**: slicing, reshaping, permuting, transposing
- **Broadcasting** with stride-0 for size-1 dimensions
- **Cache-optimized iteration** with automatic blocking and loop reordering
- **Parallel iteration** with rayon (optional)
- **BLAS integration** for optimized linear algebra (optional)

## Installation

This crate is currently **not published to crates.io** (`publish = false` in `Cargo.toml`).

For local development, add a path dependency:

```toml
[dependencies]
strided-rs = { path = "../strided-rs" }
```

When this crate is published, you will be able to add it to your `Cargo.toml` as:

```toml
[dependencies]
strided-rs = "0.1"
```

### Optional Features

```toml
[dependencies]
strided-rs = { version = "0.1", features = ["parallel", "blas"] }
```

- `parallel`: Enable rayon-based parallel iteration (`par_iter()`)
- `blas`: Enable BLAS-backed linear algebra operations (`blas_axpy`, `blas_dot`, `blas_gemm`)

## Quick Start

```rust
use strided_rs::{StridedArrayView, Identity};

// Create a 2D view over existing data
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let view: StridedArrayView<'_, f64, 2, Identity> =
    StridedArrayView::new(&data, [2, 3], [3, 1], 0).unwrap();

// Access elements
assert_eq!(view.get([0, 0]), 1.0);
assert_eq!(view.get([1, 2]), 6.0);

// Transpose (zero-copy)
let transposed = view.t();
assert_eq!(transposed.size(), &[3, 2]);
assert_eq!(transposed.get([0, 1]), 4.0);
```

## Core Types

### `StridedArrayView<'a, T, N, Op>`

An immutable view over strided data with:
- `T`: Element type
- `N`: Number of dimensions (const generic)
- `Op`: Element operation (Identity, Conj, Transpose, Adjoint)

### `StridedArrayViewMut<'a, T, N, Op>`

A mutable version of `StridedArrayView`.

## View Operations

### Slicing

```rust
use strided_rs::{StridedArrayView, Identity, StridedRange};

let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
let view: StridedArrayView<'_, f64, 3, Identity> =
    StridedArrayView::new(&data, [2, 3, 4], [12, 4, 1], 0).unwrap();

// Slice with range
let sliced = view.slice([0..1, 1..3, 0..4]).unwrap();
assert_eq!(sliced.size(), &[1, 2, 4]);

// Slice with stride
let strided = view.slice([
    StridedRange::new(0, 2, 1),   // all rows
    StridedRange::new(0, 3, 2),   // every other column
    StridedRange::new(0, 4, 1),   // all depth
]).unwrap();
```

### Transpose and Permute

```rust
// 2D transpose
let transposed = view_2d.t();

// Hermitian adjoint (transpose + conjugate)
let adjoint = complex_view.h();

// General permutation
let permuted = view_3d.permute([2, 0, 1]);
```

### Reshape

```rust
let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
let view_1d: StridedArrayView<'_, f64, 1, Identity> =
    StridedArrayView::new(&data, [12], [1], 0).unwrap();

// Reshape to 2D (only works if contiguous)
let view_2d = view_1d.reshape_2d([3, 4]).unwrap();
```

### Broadcasting

```rust
// Broadcast a row vector [3] to a matrix [4, 3]
let row: StridedArrayView<'_, f64, 1, Identity> =
    StridedArrayView::new(&data, [3], [1], 0).unwrap();
let matrix = row.broadcast([4, 3]).unwrap();
// row is repeated 4 times (stride-0 broadcasting)
```

## Element Operations

Element operations are applied lazily and compose at the type level:

```rust
use strided_rs::{StridedArrayView, Identity, Conj};
use num_complex::Complex64;

let data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
let view: StridedArrayView<'_, Complex64, 1, Identity> =
    StridedArrayView::new(&data, [2], [1], 0).unwrap();

// Apply conjugate (lazy)
let conj_view = view.conj();
assert_eq!(conj_view.get([0]), Complex64::new(1.0, -2.0));

// Double conjugate returns to identity (type-level optimization)
let double_conj = conj_view.conj(); // type: StridedArrayView<..., Identity>
```

## Iteration

```rust
// Sequential iteration
for value in view.iter() {
    println!("{}", value);
}

// Enumerated iteration
for (indices, value) in view.enumerate() {
    println!("{:?}: {}", indices, value);
}
```

### Parallel Iteration (requires `parallel` feature)

```rust
use rayon::prelude::*;

let sum: f64 = view.par_iter().sum();
let max: f64 = view.par_iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
```

## Map and Reduce Operations

```rust
use strided_rs::{map_into, zip_map2_into, zip_map3_into, zip_map4_into, reduce};

// Unary map: dest[i] = f(src[i])
map_into(&mut dest, &src, |x| x * 2.0).unwrap();

// Binary zip map: dest[i] = f(a[i], b[i])
zip_map2_into(&mut dest, &a, &b, |x, y| x + y).unwrap();

// Ternary zip map: dest[i] = f(a[i], b[i], c[i])
zip_map3_into(&mut dest, &a, &b, &c, |x, y, z| x * y + z).unwrap();

// Quaternary zip map: dest[i] = f(a[i], b[i], c[i], d[i])
zip_map4_into(&mut dest, &a, &b, &c, &d, |w, x, y, z| w * x + y * z).unwrap();

// Reduce with map
let total = reduce(&src, |x| *x, |a, b| a + b, 0.0).unwrap();
```

## High-Level Operations

```rust
use strided_rs::{copy_into, copy_scale, copy_conj, add, mul, axpy, fma, sum, dot};

// Copy operations
copy_into(&mut dest, &src).unwrap();           // dest = src
copy_scale(&mut dest, &src, 2.0).unwrap();     // dest = 2.0 * src
copy_conj(&mut dest, &src).unwrap();           // dest = conj(src)

// Element-wise arithmetic
add(&mut dest, &a, &b).unwrap();               // dest = a + b
mul(&mut dest, &a, &b).unwrap();               // dest = a * b

// BLAS-like operations
axpy(&mut y, 2.0, &x).unwrap();                // y = 2.0 * x + y
fma(&mut dest, &a, &b, &c).unwrap();           // dest = a * b + c

// Reductions
let s = sum(&array).unwrap();                  // sum of all elements
let d = dot(&x, &y).unwrap();                  // dot product
```

## BLAS Integration (requires `blas` feature)

```rust
use strided_rs::{blas_axpy, blas_dot, blas_gemm, is_blas_matrix};

// Check if a matrix is BLAS-compatible
if let Some(info) = is_blas_matrix(&matrix_view) {
    println!("Layout: {:?}, ld: {}", info.layout, info.ld);
}

// BLAS axpy: y = alpha * x + y
blas_axpy(2.0, &x, &mut y).unwrap();

// BLAS dot: result = x · y
let dot_product = blas_dot(&x, &y).unwrap();

// BLAS gemm: C = alpha * A * B + beta * C
blas_gemm(1.0, &a, &b, 0.0, &mut c).unwrap();
```

## Cache Optimization

The library automatically optimizes iteration order for cache efficiency:

1. **Dimension Reordering**: Dimensions are sorted by stride magnitude
2. **Tiled Iteration**: Operations are blocked to fit in L1 cache (32KB)
3. **Contiguous Fast Paths**: Contiguous arrays bypass blocking for direct iteration

```rust
// Check contiguity information
let inner_dims = view.contiguous_inner_dims();
let inner_len = view.contiguous_inner_len();

// Get contiguous slice (if applicable)
if let Some(slice) = view_1d.as_slice() {
    // Direct slice access for SIMD/BLAS
}
```

## Benchmarks

Run benchmarks to measure performance:

```bash
# Basic strided operations (strided vs naive implementations)
cargo bench --bench strided_bench

# Parallel benchmarks (sequential vs parallel)
cargo bench --features parallel --bench parallel_bench

# BLAS benchmarks (generic vs BLAS-backed)
cargo bench --features blas --bench blas_bench

# Run all benchmarks
cargo bench --all-features
```

**Latest Benchmark (2026-01-28)**

- Environment: macOS, single-threaded runs (`RAYON_NUM_THREADS=1` for Rust and `JULIA_NUM_THREADS=1` with `julia --project=.` for Julia).
- Rust (Criterion) representative results:
    - `permute_32_4d`:
        - `mdarray_assign` ≈ 0.91 ms
        - `strided::copy_into` ≈ 0.93 ms
    - `transpose_2000`:
        - `mdarray_assign` ≈ 3.33 ms
        - `strided::copy_into` varied (runs showed ~5.0–8.4 ms; micro-kernel overhead observed)
    - `transpose_4000`:
        - `mdarray_assign` ≈ 43.2 ms
        - `strided::copy_into` ≈ 20.4 ms

- Julia (BenchmarkTools) representative results (single-threaded):
    - `symmetrize_4000` (Strided.jl): ~17.38 ms
    - `scale_transpose_1000` (Strided.jl): ~0.379 ms
    - `complex_elementwise_1000` (Strided.jl): ~7.56 ms
    - `permute_32_4d` (Strided.jl): ~0.844 ms
    - `multiple_permute_sum_32_4d` (Strided.jl): ~2.26 ms

Notes:
- Two-level blocking (Julia-style outer block sizing + inner micro-tiles) and a POD byte-copy fast path were implemented to reduce large-transpose overhead.
- Rust now shows large-matrix transpose improvement (see `transpose_4000`), but medium-size variance indicates micro-kernel overhead and branching costs.

Next steps:
- Add unrolled/SIMD micro-kernels for `f64`, `f32`, `Complex{f64}`, `Complex{f32}` (platform-aware NEON/AVX).
- Tune micro-tile sizes and outer block heuristics per-target CPU.
- Add tests for correctness of specialized fast paths and POD trait bounds.

**Julia-compatible Benchmarks**

Purpose: Provide a set of Rust benchmarks that match the cases listed in the Strided.jl README so results can be compared under equivalent conditions.

- Single-threaded comparison (for reproducibility):
  - Julia: set `JULIA_NUM_THREADS=1` before running
  - Rust: set `RAYON_NUM_THREADS=1` before running

- Example: run the Julia script included in this repository:

```bash
export JULIA_NUM_THREADS=1
julia --project=. benches/julia_readme_compare.jl
```

- Example: run the Rust runner that prints the same cases:

```bash
export RAYON_NUM_THREADS=1
cargo bench --bench rust_readme_compare
```

- The five matched cases are: `symmetrize_4000`, `scale_transpose_1000`, `complex_elementwise_1000`, `permute_32_4d`, and `multiple_permute_sum_32_4d`.

**Comparison results (single-threaded, 2026-01-28)**

| Case | Julia Strided (ms) | Rust strided (ms) | Rust mdarray/naive (ms) |
|---|---:|---:|---:|
| symmetrize_4000 | 17.376 | 25.796 | 59.404 (naive) |
| scale_transpose_1000 | 0.379 | 0.596 | 1.327 (naive) |
| complex_elementwise_1000 | 7.562 | 13.164 | 14.301 (naive) |
| permute_32_4d | 0.844 | 0.949 | 0.946 (mdarray_assign) |
| multiple_permute_sum_32_4d | 2.257 | 2.865 | 4.559 (mdarray_alloc4) |

Notes:
- The Rust measurements are produced by [benches/rust_readme_compare.rs](benches/rust_readme_compare.rs), which uses `Instant` and reports the mean duration after warm-up and repeated iterations.
- The `mdarray_alloc4` entry for `multiple_permute_sum_32_4d` represents the Julia "Base" approach (materialize four permuted arrays, then add).
- The `complex_elementwise_1000` case in the Julia script uses `Float64` inputs (the name is historical), so the Rust runner also uses `f64` for parity.

File mapping (reference):
- Julia script: [benches/julia_readme_compare.jl](benches/julia_readme_compare.jl)
- Rust benches: [benches/strided_bench.rs](benches/strided_bench.rs), [benches/mdarray_compare.rs](benches/mdarray_compare.rs), and the new runner [benches/rust_readme_compare.rs](benches/rust_readme_compare.rs)
- Example logs: `bench_readme_single_thread.log`, `bench_permute_1000.log`

Comparison procedure:
- Run both Julia and Rust on the same hardware with single-thread settings and compare the named cases (e.g., `permute_32_4d`, `transpose_4000`).
- Use Criterion summaries for Rust and BenchmarkTools output for Julia.
- Record environment details (OS, CPU, compiler flags) when taking measurements.

Note:
- Using the existing `benches/` and `benches/julia_readme_compare.jl` script allows reproducing the README numbers. If discrepancies appear, check `RAYON_NUM_THREADS`/`JULIA_NUM_THREADS`, compiler optimizations, and whether type-specialized micro-kernels are enabled.

**Copy vs Pod and static POD gating**

Summary:
- The Rust `Copy` trait allows bitwise copying at the language level but does not by itself guarantee the safest or fastest memory-level byte-copy semantics for all optimization paths. Relying only on `Copy` can leave runtime branching, alignment concerns, or missed SIMD/vectorization opportunities.

Recommendation:
- Use a static POD trait (for example `bytemuck::Pod` or `zerocopy::AsBytes`/`FromBytes`) to mark element types that are safe for raw byte-wise copies. This removes runtime checks (such as `needs_drop`) and enables `memcpy`/`copy_nonoverlapping` fast paths and more aggressive micro-kernel specialization.

Practical notes and steps to adopt in this crate:
1. Add an explicit dependency in `Cargo.toml`: e.g. `bytemuck = "1"` or `zerocopy = "0.6"`.
2. Replace runtime gating (e.g. `if !needs_drop::<T>() { ... }`) with static bounds: `T: bytemuck::Pod` or `T: zerocopy::AsBytes + zerocopy::FromBytes` for POD fast paths.
3. Implement a `copy_2d_transpose_pod<T: Pod>` path that uses `std::ptr::copy_nonoverlapping` when safe and falls back to the generic path otherwise.
4. Add hand-optimized micro-kernels for common numeric types (`f64`, `f32`, `Complex<f64>`, `Complex<f32>`) that use unrolling and platform SIMD intrinsics where appropriate.
5. Tune outer-block and micro-tile sizes per target CPU and re-run the `benches/` suite to verify improvements.

Caveats:
- Complex or user-defined types may not be `Pod` by default; they may require `bytemuck::Pod` derives or manual verification of layout and padding.
- Alignment requirements must be respected when using raw pointer copies and SIMD loads/stores.
- Static POD gating improves performance by enabling specialized codegen, but correctness must be validated with tests for all specialized paths.

If you want, I can start by adding `bytemuck` to `Cargo.toml`, update the POD gating in `src/ops.rs` to use `T: bytemuck::Pod`, and run the README benchmark runner to show the before/after. Proceed? 
```

### Benchmark Reports

Detailed benchmark results are available in the `docs/` directory:

- [`docs/report.md`](docs/report.md) - Strided vs naive implementation comparison
- [`docs/report_parallel.md`](docs/report_parallel.md) - Sequential vs parallel comparison
- [`docs/report_blas.md`](docs/report_blas.md) - Generic vs BLAS comparison

### Performance Highlights

| Feature | Speedup | Best Use Case |
|---------|---------|---------------|
| **Strided kernels** | 2-4x | Mixed stride patterns |
| **Parallel** (`par_zip_map2_into`) | 4-6x | Compute-heavy ops, large arrays |
| **BLAS** (`blas_gemm`) | 40-120x | Matrix multiplication |

## Performance Tips

1. **Use contiguous arrays when possible** - they get fast-path optimization
2. **Enable the `parallel` feature** for large arrays (>1M elements) or compute-heavy operations
3. **Use `blas_*` functions** for linear algebra when BLAS is available
4. **Use `is_blas_matrix`** to check BLAS compatibility before calling BLAS functions
5. **Prefer row-major layout** (stride[N-1] == 1) for better cache performance

## Linear Algebra

## Development status (2026-01-28)

- Summary:
  - Added `bytemuck` and a POD-only fast path.
  - New public APIs:
    - `copy_into_pod<T: Pod>(...)`: fast byte-copy path for POD element types.
    - `copy_into_pod_complex_f32` / `copy_into_pod_complex_f64`: helpers that reuse the POD path for `Complex` via an internal POD representation with runtime checks.
  - Added `src/pod_complex.rs` providing `PodComplexF32` / `PodComplexF64` and casting utilities.

- Safety / implementation notes:
  - `unsafe` is kept internal; the public API remains safe to call.
  - Cast compatibility is checked at runtime (size/alignment); on mismatch it returns `StridedError::PodCastUnsupported`.

- Benchmarks (single-thread representative values):
  - symmetrize_4000 — Julia: ~17.38 ms | Rust: ~25.80 ms
  - scale_transpose_1000 — Julia: ~0.379 ms | Rust: ~0.596 ms
  - complex_elementwise_1000 — Julia: ~7.56 ms | Rust: ~13.16 ms
  - permute_32_4d — Julia: ~0.844 ms | Rust: ~0.949 ms
  - multiple_permute_sum_32_4d — Julia: ~2.26 ms | Rust: ~2.87 ms

- Next steps:
  1. Add dedicated `Complex` benchmarks to measure `copy_into_pod_complex_*`.
  2. Implement Complex micro-kernels (unrolling / SIMD).
  3. Tune outer-block and micro-tile parameters per target CPU.

```rust
use strided_rs::{matmul, generic_matmul, linalg_axpy, axpby, lmul, rmul};

// Matrix multiplication: C = alpha * A * B + beta * C
matmul(&mut c, &a, &b, 1.0, 0.0).unwrap();

// Generic matmul (without BLAS)
generic_matmul(&mut c, &a, &b, 1.0, 0.0).unwrap();

// Vector operations (1D arrays)
linalg_axpy(&mut y, 2.0, &x).unwrap();       // y = 2.0 * x + y
axpby(&mut y, 2.0, &x, 0.5).unwrap();        // y = 2.0 * x + 0.5 * y
lmul(&mut x, 2.0).unwrap();                  // x = 2.0 * x
rmul(&mut x, 2.0).unwrap();                  // x = x * 2.0
```

## Broadcasting with CaptureArgs

```rust
use strided_rs::{broadcast_into, promoteshape, CaptureArgs, Arg, Scalar};

// Broadcast [1, 3] and [4, 1] to [4, 3] and add
let target = [4, 3];
let a_promoted = promoteshape(&target, &row_vec).unwrap();
let b_promoted = promoteshape(&target, &col_vec).unwrap();
broadcast_into(&mut dest, |x, y| x + y, &a_promoted, &b_promoted).unwrap();

// CaptureArgs for lazy evaluation
let capture = CaptureArgs::new(|a, b, c| a + b * c, (Arg, Arg, Scalar(2.0)));
```

## Dimension Reduction

```rust
use strided_rs::mapreducedim_into;

// Sum along axis 0: [3, 4] -> [1, 4]
let mut dest = Tensor::from_fn([1, 4], |_| 0.0);
mapreducedim_into(&mut dest, &src, |&x| x, |a, b| a + b, None).unwrap();

// Sum of squares with init_op
fn zero_init(_: &f64) -> f64 { 0.0 }
mapreducedim_into(&mut dest, &src, |&x| x * x, |a, b| a + b, Some(zero_init)).unwrap();
```

## Julia Port Status

This crate is a **~98% complete port** of Julia's Strided.jl/StridedViews.jl:

| Julia Module | Rust Module | Status |
|--------------|-------------|--------|
| `stridedview.jl` | `view.rs` | ✅ Complete |
| `mapreduce.jl` | `kernel.rs`, `map.rs`, `reduce.rs` | ✅ Complete |
| `broadcast.jl` | `broadcast.rs` | ✅ Complete |
| `linalg.jl` | `linalg.rs` | ✅ Complete |

## Acknowledgments

This crate is inspired by and ports functionality from:
- [Strided.jl](https://github.com/Jutho/Strided.jl) by Jutho
- [StridedViews.jl](https://github.com/Jutho/StridedViews.jl) by Jutho

## License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

See `NOTICE` for upstream attribution (Strided.jl / StridedViews.jl are MIT-licensed).
