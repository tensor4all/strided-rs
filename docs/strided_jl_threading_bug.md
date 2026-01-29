# Strided.jl: Threading not activated for permuted (non-column-major) views

## Summary

`_mapreduce_threaded!` fails to split dimensions for arrays whose strides are
not in column-major order. Contiguous row-major views (e.g. `permutedims(A, (2,1))`)
cannot be parallelized even when the total element count far exceeds
`MINTHREADLENGTH`.

## Minimal reproduction

```julia
using Strided

function bench_threaded(n, niter)
    A = rand(n, n)                    # col-major  (strides = (1, n))
    B = similar(A)
    At = StridedView(PermutedDimsArray(A, (2,1)))  # row-major-ish (strides = (n, 1))
    Bt = StridedView(PermutedDimsArray(B, (2,1)))  # row-major-ish (strides = (n, 1))

    f(x) = x * exp(-2x) + sin(x*x)

    # Warm up
    @strided B  .= f.(A)
    @strided Bt .= f.(At)

    # Benchmark column-major
    t_col = @elapsed for _ in 1:niter
        @strided B .= f.(A)
    end

    # Benchmark row-major (permuted) – same data, same total elements
    t_row = @elapsed for _ in 1:niter
        @strided Bt .= f.(At)
    end

    println("Threads : ", Threads.nthreads())
    println("Elements: ", n*n)
    println("Col-major: $(1e3 * t_col / niter) ms")
    println("Row-major: $(1e3 * t_row / niter) ms")
end

bench_threaded(1000, 20)
```

### Expected (4 threads)

Both cases should scale roughly equally because the data is contiguous in
memory; only the stride order differs.

```
Col-major: ~3 ms
Row-major: ~3 ms
```

### Actual (4 threads)

```
Col-major: ~3 ms     ← parallelized ✓
Row-major: ~12 ms    ← sequential   ✗
```

## Root cause

The pipeline is: `_mapreduce_fuse!` → `_mapreduce_order!` → `_computeblocks` → `_mapreduce_threaded!`.

1. **`_mapreduce_fuse!`** checks `s[i] == dims[i-1] * s[i-1]` for consecutive
   dimensions. This only detects column-major contiguity.

   - Col-major `(1, 1000)`: `s[2] = 1000 == 1000 * 1 = dims[1] * s[1]` → **fuses** to `(1_000_000,)`
   - Row-major `(1000, 1)`: `s[2] = 1 ≠ 1000 * 1000` → **no fusion**, stays `(1000, 1000)`

2. **`_computeblocks`** preserves dimensions whose stride-order position is minimal
   (the "first dim has smallest stride" path). For row-major after ordering,
   both dimensions are preserved: `blocks = (1000, 1000) = dims`.

3. **`_mapreduce_threaded!`** guard: `dims[i] ≤ min(blocks[i], 1024)` →
   `1000 ≤ min(1000, 1024) = 1000` → **true** for all dimensions → falls
   through to single-threaded `_mapreduce_kernel!`.

For column-major, fusion produces a single `1_000_000`-element dimension that
exceeds the 1024 guard and is split normally.

## Suggested fix

Apply a **second fusion pass after ordering**. Once `_mapreduce_order!` has
sorted dimensions by stride importance (smallest stride first), consecutive
dimensions may now be contiguous regardless of the original memory layout:

```julia
# After ordering (pseudocode)
ordered_dims   = permute(dims, p)
ordered_strides = map(s -> permute(s, p), strides)

# Second fuse on ordered representation
ordered_dims = _mapreduce_fuse_ordered!(ordered_dims, ordered_strides)

# Then compute blocks and proceed to threading
blocks = _computeblocks(ordered_dims, ...)
```

For the row-major example after ordering:
- `ordered_strides = (1, 1000)` → `s[2] = 1000 == 1000 * 1` → fuses to `(1_000_000, 1)`
- Now `dims[1] = 1_000_000 > 1024` → threading guard passes → parallelized

This fix is layout-agnostic and preserves the existing algorithm structure.
