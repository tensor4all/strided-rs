# Future Work: strided-rs

This document summarizes the remaining gaps between strided-rs and Julia's Strided.jl,
along with design directions discussed for each.

## 1. Mixed Scalar Types (Issue #5)

**Problem:** All map/broadcast/ops operations require every operand to share the same
element type `T`. This prevents operations like `Array<f64> + Array<Complex64>`.

**Design:** Generalize type parameters per operand without bulk pre-promotion.
The user-supplied closure handles per-element type conversion at zero extra allocation cost.

```rust
// Before: single type T
zip_map2_into<T>(dest: &mut StridedViewMut<T>, a: &StridedView<T>, b: &StridedView<T>, f: Fn(T,T)->T)

// After: independent types per operand
zip_map2_into<D, A, B>(dest: &mut StridedViewMut<D>, a: &StridedView<A>, b: &StridedView<B>, f: Fn(A,B)->D)
```

### Changes required by layer

| Layer | Change | Complexity |
|-------|--------|------------|
| `kernel.rs` | None (already type-agnostic: offsets + strides only) | None |
| `block.rs` | `elem_size: usize` → `elem_sizes: &[usize]` (per-array byte strides) | Small |
| `map_view.rs` | Separate type params on inner loops and public APIs | Medium |
| `ops_view.rs` | Trait bounds express promotion (e.g. `D: Add<S, Output=D>`) | Medium |
| `threading.rs` | `SendPtr<T>` already generic; just use per-type pointers | Small |

Backward compatible: when `D = A = B = T`, Rust infers the same-type case and
existing call sites compile unchanged.

Tracked as: https://github.com/tensor4all/strided-rs/issues/5

---

## 2. Multi-Input `_mapreducedim!` (Highest Priority for matmul)

**Problem:** Julia's `_mapreducedim!(f, op, initop, dims, (dest, src1, src2, ...))` supports
an arbitrary number of input arrays with broadcasting, reduction, and destination initialization.
Rust's `reduce` / `reduce_axis` accept only a **single** input array.

This is the core building block for:
- Generic matrix multiplication (`__mul!`)
- Batched GEMM (by adding batch dimensions)
- Any fused map-reduce-broadcast pattern

### Julia's `__mul!` depends on this

```julia
# __mul!(C, A, B, α, β)  — linalg.jl:130-162
A2 = sreshape(A, (m, 1, k))
B2 = sreshape(permutedims(B, (2, 1)), (1, n, k))
C2 = sreshape(C, (m, n, 1))
_mapreducedim!(*, +, zero, (m, n, k), (C2, A2, B2))
```

The output C2 has stride 0 in the k-dimension, so the kernel reduces over k while
broadcasting A2 and B2. Batched GEMM works identically by adding a batch dimension
with nonzero stride.

### `initop` semantics

Julia's `_mapreducedim!` takes an `initop` that pre-processes the destination before accumulation:

| `initop` | Meaning | Use case |
|----------|---------|----------|
| `nothing` | No initialization (accumulate onto existing values) | `β = 1`: `C += A * B` |
| `zero` | Zero out destination first | `β = 0`: `C = A * B` |
| `x -> x * β` | Scale destination | General `C = α*A*B + β*C` |

The kernel applies `initop` only on the **first block visit** to each output element,
tracked via a per-dimension flag (`initvars[i]`) that becomes false after the first
block-loop iteration for dimensions where `strides[1] > 0`.

### Design direction

Add a `mapreducedim_into` that accepts:
- Destination `StridedViewMut<D>` (may have stride-0 dims for reduction)
- Multiple source views (at least 2 for matmul; variadic or fixed-arity)
- `f: Fn(A, B) -> R` (map), `op: Fn(D, R) -> D` (reduce)
- `initop: Option<Fn(D) -> D>` (destination initialization)

The existing fuse → order → block → kernel pipeline already supports stride-0 dimensions;
the missing piece is wiring multiple typed inputs through the kernel callback with
initop logic.

---

## 3. Matrix Multiplication (`mul!`)

**Problem:** Julia's `mul!(C, A, B, α, β)` implements `C = α*A*B + β*C` with two paths.

### Path 1: faer GEMM (replaces Julia's BLAS path)

Instead of Julia's `BLAS.gemm!` path, use `faer`'s pure-Rust SIMD matmul.
faer is already available in `extern/faer` and uses rayon for parallelism.

**faer's matmul dispatch chain:**
```
faer::matmul(dst, beta, lhs, rhs, alpha, par: Par)
├── matvec fast paths (m==1 or n==1)
├── rank-1 update (k==1)
├── nano_gemm (m*n*k <= 16³) — branchless small matrix
├── private_gemm_x86 (AVX512/AVX2+FMA) — x86 SIMD microkernel
├── gemm crate (other architectures)
└── matmul_vertical / matmul_horizontal (non-native types: custom SIMD)
```

**Key advantage over external BLAS:** faer's entire parallelism stack runs on rayon.
No foreign thread pools (OpenBLAS pthreads, MKL OpenMP). This eliminates the
nested parallelism / oversubscription problem entirely (Issue #6).

**faer's `Par` enum controls parallelism explicitly:**
```rust
pub enum Par {
    Seq,
    Rayon(NonZeroUsize),  // explicit thread count
}
```

Every faer public API takes `par: Par` as a parameter, giving the caller full control.

### Path 2: Generic (linalg.jl:130-162)

For non-BLAS types or non-contiguous layouts:
- Reshape A→(m,1,k), B→(1,n,k), C→(m,n,1)
- Call `_mapreducedim!(*, +, initop, (m,n,k), (C2, A2, B2))`
- Parallelized by `_mapreduce_threaded!` (reduction dim k excluded from splitting)

### Path 2 extends to batched GEMM

```
# Batched: C[b,i,j] = Σ_k A[b,i,k] * B[b,k,j]
A2 = reshape(A, (batch, m, 1, k))
B2 = reshape(B, (batch, 1, n, k))
C2 = reshape(C, (batch, m, n, 1))
_mapreducedim!(*, +, initop, (batch, m, n, k), (C2, A2, B2))
```

Batch dimensions have nonzero strides in all arrays, so they are iterated (not reduced)
and can be split across threads.

### Dispatch logic (linalg.jl:44-63)

Handles `C.op` (conj) and stride orientation:
- `C.op == conj` + col-major → `_mul!(conj(C), conj(A), conj(B), conj(α), conj(β))`
- `C.op == conj` + row-major → `_mul!(C', B', A', conj(α), conj(β))`
- row-major C → `_mul!(C^T, B^T, A^T, α, β)`

### Proposed strided-rs matmul API

```rust
pub fn matmul<T: ComplexField>(
    c: &mut StridedViewMut<T>,
    a: &StridedView<T>,
    b: &StridedView<T>,
    alpha: T,
    beta: T,
    par: Par,
) -> Result<()> {
    if is_faer_compatible(c, a, b) {
        // faer path: convert StridedView → MatRef, call faer::matmul
        // Pass `par` directly → faer uses same rayon pool
        faer_matmul(c, a, b, alpha, beta, par)
    } else {
        // generic path: reshape → _mapreducedim!
        // `par` controls strided's own mapreduce_threaded
        generic_matmul(c, a, b, alpha, beta, par)
    }
}
```

**faer compatibility check:** `stride(C,1)==1 || stride(C,2)==1` (at least one contiguous
dimension). faer's `MatRef` accepts arbitrary row/col strides and handles negative strides
internally via `reverse_rows` / `reverse_cols`.

### Implementation order

1. **Multi-input `_mapreducedim!`** (§2) — prerequisite for generic path
2. **Generic `__mul!`** — reshape + permute + `_mapreducedim!`
3. **faer fast path** — `StridedView` ↔ `MatRef` conversion + dispatch

---

## 4. Parallel Complete Reduction

**Problem:** `reduce` and `reduce_axis` run single-threaded even when `parallel` feature
is enabled. This directly affects the `reduce_transposed` benchmark (currently only 1.04x).

### Julia's approach (mapreduce.jl:153-170)

When `op !== nothing && _length(dims, strides[1]) == 1` (all output strides are zero = complete reduction):

1. Allocate thread-local output slots with cache-line spacing to avoid false sharing:
   ```julia
   spacing = max(1, div(64, sizeof(T)))
   threadedout = similar(arrays[1], spacing * nthreads)
   _init_reduction!(threadedout, f, op, a)
   ```
2. Each thread reduces into its own slot via `spacing * (taskindex - 1)` offset
3. After all threads finish, merge serially:
   ```julia
   for i in 1:nthreads
       a = op(a, threadedout[(i-1) * spacing + 1])
   end
   ```

### Rust status

`mapreduce_threaded` already accepts `spacing` and `taskindex` parameters, but
`reduce` / `reduce_axis` in `reduce_view.rs` never invoke the threaded path.
The infrastructure exists; wiring it up is the remaining work.

---

## 5. Missing LinearAlgebra Operations

These are straightforward to add once the building blocks exist.

| Julia function | Description | Depends on |
|---------------|-------------|------------|
| `rmul!(dst, α)` / `lmul!(α, dst)` | In-place scalar multiply: `dst .*= α` | `map_into(dst, dst, \|x\| α * x)` |
| `axpby!(a, X, b, Y)` | `Y = a*X + b*Y` | `zip_map2_into` or broadcast |
| `conj!(a)` | In-place conjugation | `map_into(a, a, conj)` |
| `adjoint!(dst, src)` | `copy!(dst, adjoint(src))` | `src.adjoint_2d()` + `copy_into` |
| `permutedims!(dst, src, p)` | `copy!(dst, permutedims(src, p))` | `src.permute(p)` + `copy_into` |
| `mul!(dst, α, src)` | `dst = α * src` (with `α==1` fast path) | `copy_into` / `copy_scale` |

---

## 6. Nested Parallelism Strategy (Issue #6)

### The problem

When matmul uses an external GEMM implementation, two levels of parallelism can conflict:
1. **Strided-level:** Recursive dimension splitting via `rayon::join`
2. **GEMM-level:** The GEMM library's own thread pool

With external BLAS (OpenBLAS/MKL), this causes oversubscription because they use
foreign thread pools (pthreads/OpenMP) that don't coordinate with rayon.

### The solution: faer + `Par` parameter

faer's entire stack runs on rayon. By passing `Par` through the call chain, nesting
is handled naturally:

```
strided matmul(c, a, b, alpha, beta, par)
│
├── faer path (2D, compatible strides):
│   └── faer::matmul(dst, beta, lhs, rhs, alpha, par)
│       └── All internal parallelism uses same rayon pool
│           (nano_gemm, private_gemm_x86, gemm crate, spindle::for_each)
│
└── generic path (arbitrary rank / non-contiguous):
    └── _mapreducedim! → mapreduce_threaded (rayon::join)
        └── Leaf: sequential kernel (no nested parallelism)
```

**When strided splits work and calls faer per-tile:**
```rust
// strided splits along m/n dimensions via rayon::join
// Each leaf calls faer with Par::Seq → faer runs single-threaded
faer::matmul(tile_dst, beta, tile_lhs, rhs, alpha, Par::Seq);
```

**When faer handles all parallelism:**
```rust
// No strided splitting; faer manages everything
faer::matmul(dst, beta, lhs, rhs, alpha, Par::Rayon(nthreads));
```

Both modes use the same rayon pool. No oversubscription. No `OPENBLAS_NUM_THREADS`
hacks. No `enable_threaded_mul()` / `disable_threaded_mul()` complexity.

### Comparison with Julia's approach

| Aspect | Julia Strided.jl | strided-rs + faer |
|--------|-----------------|-------------------|
| GEMM backend | External BLAS (OpenBLAS/MKL) | faer (pure Rust) |
| GEMM thread pool | Foreign (pthreads/OpenMP) | rayon (shared) |
| Nesting conflict | Yes → manual `BLAS.set_num_threads(1)` | None (same pool) |
| Default matmul threading | BLAS-only (`threaded_mul = false`) | `Par` controls everything |
| Strided splitting | Opt-in via `enable_threaded_mul()` | Natural via `Par` dispatch |

Tracked as: https://github.com/tensor4all/strided-rs/issues/6

---

## 7. Thread Control API

Julia exposes explicit controls:

```julia
get_num_threads() / set_num_threads(n)
enable_threads() / disable_threads()
use_threaded_mul() / enable_threaded_mul() / disable_threaded_mul()
```

With the faer approach, most of this complexity disappears. The `Par` parameter
at the API level replaces the global mutable state:

```rust
// User controls parallelism per-call
matmul(c, a, b, alpha, beta, Par::Seq);           // single-threaded
matmul(c, a, b, alpha, beta, Par::Rayon(4));       // 4 threads
matmul(c, a, b, alpha, beta, Par::rayon(0));       // all available threads

// Same for map/reduce operations
map_into(dest, src, f, Par::Rayon(2));
reduce(src, map_fn, reduce_fn, init, Par::Seq);
```

This is more ergonomic and less error-prone than Julia's global toggles.
No hidden state, no need for `enable_threaded_mul()` / `disable_threaded_mul()`.

**Open question:** Should existing map/reduce APIs also take `Par`?
Currently they auto-detect via `rayon::current_num_threads()` when the `parallel`
feature is enabled. Adding `Par` would give users explicit control but changes
the API surface. This could be a v0.2 change.

---

## Priority Summary

| Priority | Feature | Rationale |
|----------|---------|-----------|
| **P0** | Multi-input `_mapreducedim!` | Prerequisite for matmul (generic + batched) |
| **P0** | Parallel complete reduction | Directly affects `reduce_transposed` benchmark (1.04x → Nx) |
| **P1** | Generic `__mul!` (matmul via mapreducedim) | Core linalg operation |
| **P1** | faer fast path for matmul | SIMD-optimized 2D GEMM, no nesting issues |
| **P1** | Mixed scalar types (#5) | Needed for real + complex interop |
| **P1** | `rmul!`, `lmul!`, `axpby!` | Used by `__mul!` for β-scaling |
| **P2** | `Par` parameter on map/reduce APIs | Explicit user control over parallelism |
| **P2** | Convenience wrappers (`conj!`, `adjoint!`, `permutedims!`) | Small, easy wins |

---

## Dependency Graph

```
Mixed scalar types (#5)
    │
    ▼
Multi-input _mapreducedim!  ←──── Parallel complete reduction
    │
    ├──► Generic __mul! (2D matmul)
    │       │
    │       ├──► rmul!/lmul! (β-scaling)
    │       │
    │       └──► Batched GEMM (just add batch dim)
    │
    └──► faer fast path (2D GEMM)
            │
            └──► Par parameter propagation
                 (unified parallelism control, no nesting issues)
```
