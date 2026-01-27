# mdarray-strided

Cache-optimized kernels for strided multidimensional array operations in Rust.

This crate is a port of Julia's [Strided.jl](https://github.com/Jutho/Strided.jl) and [StridedViews.jl](https://github.com/Jutho/StridedViews.jl) libraries.

## Features

- **Zero-copy strided views** over contiguous memory
- **Lazy element operations** (conjugate, transpose, adjoint) with type-level composition
- **Zero-copy transformations**: slicing, reshaping, permuting, transposing
- **Broadcasting** with stride-0 for size-1 dimensions
- **Cache-optimized iteration** with automatic blocking and loop reordering
- **Parallel iteration** with rayon (optional)
- **BLAS integration** for optimized linear algebra (optional)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mdarray-strided = "0.1"
```

### Optional Features

```toml
[dependencies]
mdarray-strided = { version = "0.1", features = ["parallel", "blas"] }
```

- `parallel`: Enable rayon-based parallel iteration (`par_iter()`)
- `blas`: Enable BLAS-backed linear algebra operations (`blas_axpy`, `blas_dot`, `blas_gemm`)

## Quick Start

```rust
use mdarray_strided::{StridedArrayView, Identity};

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
use mdarray_strided::{StridedArrayView, Identity, StridedRange};

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
use mdarray_strided::{StridedArrayView, Identity, Conj};
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
use mdarray_strided::{map_into, zip_map2_into, zip_map3_into, zip_map4_into, reduce};

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
use mdarray_strided::{copy_into, copy_scale, copy_conj, add, mul, axpy, fma, sum, dot};

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
use mdarray_strided::{blas_axpy, blas_dot, blas_gemm, is_blas_matrix};

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

- Environment: macOS, single-threaded runs (JULIA_NUM_THREADS=1 for Julia).
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
    - `symmetrize_4000` (Strided.jl): ~16.63 ms
    - `scale_transpose_1000` (Strided.jl): ~0.393 ms
    - `complex_elementwise_1000` (Strided.jl): ~7.55 ms
    - `permute_32_4d` (Strided.jl): ~0.748 ms
    - `multiple_permute_sum_32_4d` (Strided.jl): ~2.20 ms

Notes:
- Two-level blocking (Julia-style outer block sizing + inner micro-tiles) and a POD byte-copy fast path were implemented to reduce large-transpose overhead.
- Rust now shows large-matrix transpose improvement (see `transpose_4000`), but medium-size variance indicates micro-kernel overhead and branching costs.

Next steps:
- Implement static POD gating (e.g. `bytemuck::Pod`) instead of runtime `needs_drop` checks.
- Add unrolled/SIMD micro-kernels for `f64`, `f32`, `Complex{f64}`, `Complex{f32}` (platform-aware NEON/AVX).
- Tune micro-tile sizes and outer block heuristics per-target CPU.
- Add tests for correctness of specialized fast paths and POD trait bounds.

**Julia 対応ベンチマーク**

目的: Julia の `Strided.jl` README に載っているベンチマークと同等条件で Rust 側のベンチを実行・比較できるようにする。

- 単一スレッドでの比較（再現性のため）:
    - Julia: `export JULIA_NUM_THREADS=1` を設定して実行
    - Rust: `export RAYON_NUM_THREADS=1` を設定して実行

- Julia の実行例 (リポジトリルートにあるスクリプトを使う):

```bash
export JULIA_NUM_THREADS=1
julia --project=. julia_readme_bench.jl
```

- Rust の実行例 (Criterion ベンチ):

```bash
export RAYON_NUM_THREADS=1
cargo bench --bench strided_bench
```

- Julia README（相当）と同一ケースの比較を出力する Rust ランナー:

```bash
export RAYON_NUM_THREADS=1
cargo bench --bench rust_readme_compare
```

- Julia 側（同一ケース）:

```bash
export JULIA_NUM_THREADS=1
julia --project=. benches/julia_readme_compare.jl
```

**比較結果（単一スレッド, 2026-01-28）**

| Case | Julia Strided (ms) | Rust strided (ms) | Rust mdarray/naive (ms) |
|---|---:|---:|---:|
| symmetrize_4000 | 16.425 | 32.300 | 58.165 (naive) |
| scale_transpose_1000 | 0.408 | 0.405 | 1.030 (naive) |
| complex_elementwise_1000 | 7.517 | 12.457 | 13.903 (naive) |
| permute_32_4d | 0.737 | 0.939 | 0.958 (mdarray_assign) |
| multiple_permute_sum_32_4d | 2.200 | 2.097 | 4.192 (mdarray_alloc4) |

注記:
- Rust の計測は [benches/rust_readme_compare.rs](benches/rust_readme_compare.rs) が `Instant` で平均時間を表示（ウォームアップ後に複数回実行して平均）。
- `multiple_permute_sum_32_4d` の `mdarray_alloc4` は Julia の Base 相当（4つの permute を materialize してから加算）を意図。
- `complex_elementwise_1000` は Julia スクリプト上は Float64 入力（名前だけ complex）なので、Rust 側も `f64` で揃えています。

- ファイル対応（参考）:
    - Juliaスクリプト: [julia_readme_bench.jl](julia_readme_bench.jl)
    - Rustベンチ: [benches/strided_bench.rs](benches/strided_bench.rs) および [benches/mdarray_compare.rs](benches/mdarray_compare.rs)
    - 追加ログ: `bench_readme_single_thread.log`, `bench_permute_1000.log`

- 比較方法:
    - 同一ハードウェア、単一スレッド設定で両方実行して各ケース名（例: `permute_32_4d`, `transpose_4000`）の出力を照合する。
    - Rust の出力は Criterion のサマリ（`cargo bench` の結果）を、Julia は BenchmarkTools の出力を使う。
    - ベンチ実行時の環境（OS, CPU, コンパイラ最適化フラグ）を記録しておくこと。

注記:
- 既存の `benches/` と `julia_readme_bench.jl` を使えば、README の数値と同等の比較が可能です。ベンチの差異が出た場合は、`RAYON_NUM_THREADS`/`JULIA_NUM_THREADS`、コンパイラ最適化、またはマイクロカーネルの型特化の有無を確認してください。
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

```rust
use mdarray_strided::{matmul, generic_matmul, linalg_axpy, axpby, lmul, rmul};

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
use mdarray_strided::{broadcast_into, promoteshape, CaptureArgs, Arg, Scalar};

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
use mdarray_strided::mapreducedim_into;

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
