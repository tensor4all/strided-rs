# strided-opteinsum Performance: Closing the Gap with Julia OMEinsum

Analysis of benchmarks where Rust strided-opteinsum is abnormally slower than Julia OMEinsum, root causes, and proposed fixes.

## Benchmark Environment

Apple Silicon M2, single-threaded. Mean time (ms).

## Abnormally Slow Cases

| Case | Julia (ms) | Rust (ms) | Ratio | Category |
|---|---:|---:|---:|---|
| hadamard (100,100,100) Float64 | 0.196 | 45.906 | **234x** | Binary element-wise |
| hadamard (100,100,100) ComplexF64 | 0.755 | 65.349 | **87x** | Binary element-wise |
| trace 1000x1000 Float64 | 0.003 | 0.356 | **119x** | Single-tensor (diag+reduce) |
| trace 1000x1000 ComplexF64 | 0.004 | 0.772 | **193x** | Single-tensor (diag+reduce) |
| diag (100,100,100) Float64 | 0.016 | 0.495 | **31x** | Single-tensor (diag extract) |
| diag (100,100,100) ComplexF64 | 0.023 | 0.684 | **30x** | Single-tensor (diag extract) |
| ptrace (100,100,100) Float64 | 0.020 | 0.327 | **16x** | Single-tensor (diag+reduce) |
| ptrace (100,100,100) ComplexF64 | 0.029 | 0.700 | **24x** | Single-tensor (diag+reduce) |
| indexsum (100,100,100) Float64 | 0.163 | 0.747 | **4.6x** | Single-tensor (reduce) |
| perm (30,30,30,30) ComplexF64 | 0.835 | 2.540 | **3.0x** | Single-tensor (permute+copy) |

---

## Root Cause 1: Hadamard — GEMM Dispatch for Element-wise Multiply

### Current code path

`"ijk,ijk->ijk"` is a binary operation routed through `einsum2_into`. The `Einsum2Plan` classifies all 3 axes as **batch** (present in both inputs and output):

```
batch = [i, j, k],  lo = [],  ro = [],  sum = []
→ m = 1, k = 1, n = 1
```

The faer backend (`bgemm_faer.rs`) then iterates over 100^3 = 1,000,000 batch elements, calling `faer::matmul_with_conj` on a 1x1 matrix multiply per element:

```
for each (i,j,k) in 100x100x100:           // 1M iterations
    create 1x1 MatRef for A, B
    create 1x1 MatMut for C
    faer::matmul_with_conj(c, a, b, ...)    // full GEMM dispatch overhead
```

Each faer matmul call has significant dispatch overhead (size checks, SIMD capability detection, microkernel selection). For 1M scalar multiplications, this overhead dominates completely.

### What Julia does

OMEinsum recognizes `ijk,ijk->ijk` as a `SimpleBinaryRule` where all indices are shared and all appear in output. It emits `A .* B` — Julia's broadcasting element-wise multiply — which compiles to a single SIMD-vectorized loop over contiguous memory. Cost: ~0.2ms for 1M elements.

---

## Root Cause 2: Single-tensor Ops — Unnecessary Materialization + Slow Reduce

All single-tensor operations go through `single_tensor_einsum` (`single_tensor.rs`):

```
1. diagonal_view()     → zero-copy stride trick (fast, ~0)
2. copy_into()         → MATERIALIZE into new owned array (slow, unnecessary)
3. reduce_axis()       → sum over axis (slow: loop-carried dep, Option overhead)
4. permute + copy_into → reorder to output layout (if needed)
```

### trace `"ii->"` (119-193x slower)

Steps: `diagonal_view` (stride=1001) → `copy_into` (1000 elements) → `reduce_axis` (sum 1000 elements).

Julia: `tr(x)` — a single loop `sum += A[i,i]`. No allocation, no copy. 0.003ms.

The materialization is completely unnecessary — `reduce_axis` already handles strided views. Even without materialization, `reduce_axis` itself is slow due to:

1. Loop-carried dependency: `acc = reduce_fn(acc, val)` — LLVM cannot auto-vectorize because it cannot assume floating-point addition is associative.
2. `Option` wrapping: `acc.take().ok_or(StridedError::OffsetOverflow)?` per element adds branch + write overhead.
3. Kernel setup overhead: blocking/ordering machinery is expensive for 1000 elements.

### ptrace `"iij->j"` (16-24x slower)

Same pattern: `diagonal_view` → materialize → `reduce_axis`. Julia uses `compactify!` with a tight loop.

### diag `"ijj->ij"` (30-31x slower)

`diagonal_view` → materialize via `copy_into`. No reduction. The entire cost is copying 10,000 elements from a non-contiguous diagonal view through the general strided kernel, whose blocking/ordering setup overhead dominates for this small size.

### indexsum `"ijk->ik"` (4.6x slower)

No diagonal. Just `reduce_axis(src, 1, ...)`. Pure `reduce_axis` performance gap — loop-carried dependency prevents vectorization. Julia's `sum(x, dims=dims)` uses optimized reduction with multi-accumulator SIMD.

### perm `"ijkl->ljki"` (3.0x slower for ComplexF64)

`permute` + `copy_into`. The strided kernel copy performance for ComplexF64 with non-trivial 4D strides. Documented in `faer_design.md` as Gap 2.

---

## Proposed Solutions

All solutions add fast paths that return early. Existing general code paths are never modified, only bypassed when a simpler operation suffices. This matches Julia OMEinsum's strategy of pattern-matching simple cases before the general machinery.

### Solution A: Element-wise fast path in `einsum2_into` for `n_sum == 0`

**Where**: `strided-einsum2/src/lib.rs`, before the GEMM call

**Change**: When `plan.sum.is_empty() && plan.lo.is_empty() && plan.ro.is_empty()` (all indices are batch = element-wise operation), bypass GEMM and use strided-kernel:

```rust
// In einsum2_into(), after plan construction and validation:
if plan.sum.is_empty() && plan.lo.is_empty() && plan.ro.is_empty() {
    // Pure element-wise: C[batch] = alpha * A[batch] * B[batch] + beta * C[batch]
    // Use strided-kernel which handles strides, cache blocking, auto-vectorization
    if beta == T::zero() {
        if alpha == T::one() {
            zip_map2_into(&mut c_perm, &a_perm, &b_perm, |a, b| a * b)?;
        } else {
            zip_map2_into(&mut c_perm, &a_perm, &b_perm, |a, b| alpha * a * b)?;
        }
    } else {
        // Need to handle beta * c_old — read-modify-write via zip_map3_into
        // or a two-pass approach: scale C, then add product
    }
    return Ok(());
}
```

Conjugation handling: when `conj_a` or `conj_b` is true, apply `Conj::apply()` inside the closure.

**Fixes**: hadamard (234x → ~1-2x).

**Risk**: Zero. Only intercepts the degenerate no-contraction case. All GEMM paths unchanged.

### Solution B: Skip materialization before reduce in `single_tensor_einsum`

**Where**: `strided-opteinsum/src/single_tensor.rs`, lines 36-62

**Change**: When diagonal axes need reduction, pass the diagonal view directly to `reduce_axis` instead of materializing first:

```rust
if pairs.is_empty() {
    diag_arr = None;
    unique_ids = input_ids.to_vec();
} else {
    let dv = src.diagonal_view(&pairs)?;
    unique_ids = /* same as before */;

    // Compute axes_to_reduce early to decide whether to materialize
    let axes_to_reduce: Vec<usize> = unique_ids.iter().enumerate()
        .filter(|(_, ch)| !output_ids.contains(ch))
        .map(|(i, _)| i)
        .collect();

    if axes_to_reduce.is_empty() {
        // No reduction follows — must materialize for output
        let mut owned = StridedArray::<T>::col_major(&dims);
        copy_into(&mut owned.view_mut(), &dv)?;
        diag_arr = Some(owned);
    } else {
        // Reduction follows — reduce directly from strided diagonal view
        // (reduce_axis already handles arbitrary strides)
        // ... reduce from dv, skipping the copy ...
    }
}
```

**Fixes**: trace (removes copy before reduce), ptrace (same).

**Risk**: Zero. `reduce_axis` already accepts views with arbitrary strides.

### Solution C: Specialized loops for trace/ptrace patterns

**Where**: `strided-opteinsum/src/single_tensor.rs`, at the top of `single_tensor_einsum`

**Change**: Detect common patterns and use direct loops before falling through to the general path.

#### Full trace `"ii->"`

```rust
if input_ids.len() == 2 && input_ids[0] == input_ids[1] && output_ids.is_empty() {
    let n = src.dims()[0];
    let stride = src.strides()[0] + src.strides()[1];
    let ptr = src.data().as_ptr();
    let mut offset = src.offset() as isize;
    let mut acc = T::zero();
    for _ in 0..n {
        acc = acc + unsafe { *ptr.offset(offset) };
        offset += stride;
    }
    let mut out = StridedArray::<T>::col_major(&[]);
    out.data_mut()[0] = acc;
    return Ok(out);
}
```

#### Partial trace `"iij->j"` / `"iji->j"` etc.

Generalize: when there is exactly one repeated index pair and one or more output axes, use nested loops. The outer loop iterates output axes, the inner loop sums over the diagonal:

```rust
// Detect: exactly 1 repeated pair, remaining axes go to output
if pairs.len() == 1 {
    let (p0, p1) = pairs[0];
    let diag_stride = src.strides()[p0] + src.strides()[p1];
    let diag_len = src.dims()[p0];
    // remaining axes (not p0, not p1) are output axes
    // iterate them, sum diagonal for each
    // ... direct nested loop ...
}
```

**Fixes**: trace (119x → ~1x), ptrace (16x → ~1-2x).

**Risk**: Zero. Early return before the general path. Pattern detection is exact.

### Solution D: Remove `Option` wrapping in `reduce_axis` hot loop

**Where**: `strided-kernel/src/reduce_view.rs`

**Change**: Replace `Option<T>` accumulator with plain `T`:

```rust
// Before:
let mut acc = Some(init);
for _ in 0..len {
    let current = acc.take().ok_or(StridedError::OffsetOverflow)?;
    acc = Some(reduce_fn(current, mapped));
}

// After:
let mut acc = init;
for _ in 0..len {
    acc = reduce_fn(acc, map_fn(Op::apply(unsafe { *ptr })));
    ptr = ptr.offset(stride);
}
```

**Fixes**: All non-contiguous reductions across the entire library (10-30% improvement). Directly benefits indexsum (4.6x → ~3-4x).

**Risk**: Zero. The `Option` was never `None` during execution — the `OffsetOverflow` error was unreachable.

### Solution E: Direct loop for small diagonal extraction (diag)

**Where**: `strided-opteinsum/src/single_tensor.rs`

**Change**: When materializing a diagonal view and total element count is below a threshold, use a direct index-based loop instead of routing through the general strided kernel:

```rust
if total_elements < DIRECT_LOOP_THRESHOLD {
    let mut out = StridedArray::<T>::col_major(&dims);
    // Direct iteration — avoids kernel setup overhead
    for idx in CartesianIndices::new(&dims) {
        out.set(&idx, dv.get(&idx));
    }
    diag_arr = Some(out);
}
```

**Fixes**: diag (31x → ~5-10x). Also helps any small single-tensor operation.

**Risk**: Low. Falls back to existing path for large arrays.

---

## Priority and Implementation Order

| Priority | Solution | Fixes | Expected | Effort |
|---|---|---|---|---|
| **P0** | A: Element-wise fast path | hadamard (234x) | → ~1-2x | Small |
| **P0** | C: Specialized trace/ptrace | trace (119x), ptrace (16x) | → ~1x | Small |
| **P1** | B: Skip materialization | trace, ptrace | Complements C | Small |
| **P1** | D: Remove Option in reduce | indexsum (4.6x), all reduce | 10-30% | Tiny |
| **P2** | E: Direct loop for small diag | diag (31x) | → ~5-10x | Small |

### Dependency graph

```
A (element-wise fast path)         — independent, fixes hadamard
C (specialized trace/ptrace loops) — independent, fixes trace/ptrace
B (skip materialization)           — complements C for edge cases
D (remove Option in reduce)        — independent, fixes indexsum
E (direct loop for small diag)     — independent, fixes diag
```

All solutions are independent of each other and can be implemented in any order.

### What is NOT changed

- GEMM path (faer backend) — untouched for all cases with contraction indices
- `strided-kernel` map/copy kernels — not modified (only bypassed for small cases)
- `einsum2_into` general logic — only a new early-return branch added
- Threading/parallelism — no changes

---

## Comparison with Julia OMEinsum Strategy

Julia OMEinsum achieves its speed through rule-based dispatch:

| Julia rule | Rust equivalent (proposed) |
|---|---|
| `Tr()` → `tr(x)` | Solution C: direct trace loop |
| `Sum()` → `sum(x, dims=dims)` | Solution D: faster reduce_axis |
| `Diag()` → `compactify!()` | Solution E: direct loop for small diag |
| `Permutedims()` → `permutedims!` | Existing `permute + copy_into` (adequate) |
| `SimpleBinaryRule` (element-wise) → `A .* B` | Solution A: element-wise fast path |

The core principle is identical: **pattern-match simple operations before falling through to the general GEMM/kernel machinery**.
