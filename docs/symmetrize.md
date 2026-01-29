# Symmetrize (4000x4000) Calculation Path: Rust vs Julia (Strided.jl)

This note documents what happens in the `symmetrize_4000` benchmark and how the
Rust implementation differs from (and is inspired by) the Julia Strided.jl path.

Benchmark definitions:
- Rust driver: `benches/rust_compare.rs` (runs `zip_map2_into` over `A` and `A^T`)
- Julia driver: `benches/julia_compare.jl` (runs `@strided B .= (A .+ A') ./ 2`)

The operation being benchmarked is:

```text
B[i,j] = (A[i,j] + A[j,i]) / 2
```

For real `Float64` this is a pure transpose-average (no conjugation).

## Julia (extern/Strided.jl) Execution Path

In Julia, the benchmark uses a normal broadcast expression, but wrapped with
`@strided`:

```julia
@strided B .= (A .+ A') ./ 2
```

The key steps in Strided.jl are:

1) `@strided` turns arrays into `StridedView` (from the external `StridedViews`
   package) so that Strided.jl can control strides and offsets.
   See `extern/Strided.jl/src/macros.jl`.

2) Broadcasting with `StridedView` dispatches to Strided.jl's broadcast
   implementation:
   - `Base.copyto!(dest::StridedView, bc::Broadcasted{StridedArrayStyle})`
   - It converts the broadcast into a map-like kernel by capturing the broadcast
     function (`CaptureArgs`) and extracting the `StridedView` arguments.
   See `extern/Strided.jl/src/broadcast.jl`.

3) The work is executed by the mapreduce engine:
   - `_mapreduce_fuse!` attempts to fuse contiguous dimensions across *all*
     arrays (output and inputs) to reduce loop nesting.
   - `_mapreduce_order!` chooses a loop permutation based on stride "importance"
     across the arrays (with output weighted 2x), so the innermost loop tends to
     correspond to smaller (cheaper) strides.
   - `_computeblocks` chooses per-dimension block sizes to fit a cache model
     based on element sizes and stride costs.
   - `_mapreduce_kernel!` is a `@generated` nested-loop kernel that advances
     offsets by adding strides (avoids per-element multiplications), and uses
     `@simd` on the innermost loop.
   See `extern/Strided.jl/src/mapreduce.jl`.

Important observation for `symmetrize_4000`:
- Strided.jl treats this as a generic broadcast over the full `N x N` domain.
  There is no symmetry-aware shortcut; every `B[i,j]` is computed independently
  from `A[i,j]` and `A[j,i]`.

## Rust (strided-rs) Execution Path

The Rust benchmark in `benches/rust_compare.rs` uses:

```rust
let a_t = a_view.permute([1, 0]);
zip_map2_into(&mut b, a_view, &a_t, |&x, &y| (x + y) * 0.5)
```

That call goes through the generic map kernel machinery (`zip_map2_into`),
which builds a Strided.jl-style iteration plan and runs the cache-blocked
strided kernel (see `src/map.rs`, `src/kernel.rs`).

This is intentionally the same execution style as the Julia benchmark: treat
symmetrization as a broadcast/map over the full `N x N` output domain.

## Rust Notes: Why This Matches Julia Better

An earlier version of the benchmark used a dedicated Rust helper for `f64`
symmetrization. That was fast, but it did not follow the same path as the Julia
benchmark (which is lowered through Strided.jl's broadcast -> mapreduce kernel).

The current benchmark uses `zip_map2_into` to match Julia's execution style:
- Both treat the operation as a full-domain map over `(dest, A, A^T)`.
- Both rely on the shared Strided.jl-style iteration planning (order + blocks)
  and stride-increment inner loops, rather than a symmetry-aware triangular fill.

## Key Differences (Julia vs Rust)

1) Abstraction level
   - Julia/Strided.jl: generic broadcast lowered to a generic mapreduce kernel.
   - Rust: `zip_map2_into` executes the same kind of generic map kernel as Julia.

2) Symmetry exploitation
   - Julia/Strided.jl broadcast: computes the full `N x N` domain; no special
     handling for symmetry.
   - Rust (benchmark path): same as Julia (full-domain map). A separate
     `symmetrize_into<T>` API may still choose a symmetry-aware strategy.

3) Inner-loop strategy
   - Julia: `_mapreduce_kernel!` uses stride-increment indexing and `@simd` on the
     innermost loop (dimension order chosen by `_mapreduce_order!`).
   - Rust: micro-tiling + explicit small buffering; relies on compiler
     optimizations (autovectorization, inlining) rather than explicit SIMD intrinsics.

4) Memory layout assumptions in the benchmarks
   - Julia arrays are column-major; the transpose view swaps strides, so one of
     `(A, A')` is contiguous depending on loop order.
   - mdarray (used by Rust benches) is row-major; the same transpose tradeoff
     exists, but "which side is contiguous" is reversed vs Julia.
   - Rust includes both row-major and column-major contiguous fast paths to keep
     the algorithmic comparison meaningful.
