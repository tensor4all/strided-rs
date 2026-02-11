# Merge Analysis: strided-rs + omeinsum-rs

This document summarizes a detailed comparison of the einsum subsystems in
**strided-rs** (`strided-einsum2` + `strided-opteinsum`) and
**omeinsum-rs**, and proposes a merge strategy that combines the strengths
of both projects.

---

## 1. Architecture Overview

### strided-rs (einsum crates)

Two-crate architecture built on top of `strided-view` and `strided-kernel`:

| Crate | Role |
|-------|------|
| `strided-einsum2` | Binary (pairwise) tensor contractions with pluggable GEMM backends |
| `strided-opteinsum` | N-ary einsum frontend with contraction tree optimization (omeco) |

Data representation:
- `StridedArrayView<'a, T, N, Op>` — borrow-based (`&'a [T]`), const-generic rank `N`
- `StridedArray<T>` — owned (`Vec<T>`)
- Arbitrary strides (row-major, col-major, or non-contiguous)

### omeinsum-rs

Single-crate design:

| Component | Role |
|-----------|------|
| `Tensor<T, B: Backend>` | Multi-dimensional tensor with `Arc<B::Storage<T>>` shared storage |
| `Algebra` trait hierarchy | Semiring abstraction (Standard, MaxPlus, MinPlus, MaxMul) |
| `Backend` trait | CPU / CUDA dispatch |
| `Einsum` engine | Contraction tree evaluation with omeco optimization |

Data representation:
- `Arc<Storage>`-based reference counting
- Column-major layout throughout
- Dynamic rank (`Vec<usize>`)

---

## 2. Feature Comparison

### 2.1 Scalar Type Support

| Type | strided-rs | omeinsum-rs |
|------|-----------|-------------|
| f64 | Yes | Yes |
| f32 | No (einsum2 could work; opteinsum enum lacks variant) | Yes |
| Complex64 | Yes | Yes |
| Complex32 | No | Yes |
| Integer types (i32, i64, u32, u64) | No | Yes |
| Custom scalar types | Yes (pluggable `BgemmBackend<T>`) | Yes (`Scalar` trait) |
| Mixed-type auto-promotion (f64 -> C64) | Yes | No |

### 2.2 Algebraic Structures

| Feature | strided-rs | omeinsum-rs |
|---------|-----------|-------------|
| Standard linear algebra (+, ×) | Yes | Yes |
| Tropical algebras (MaxPlus, MinPlus, MaxMul) | Ad hoc (custom backend example) | **Native** (Semiring/Algebra traits) |
| Semiring trait abstraction | No | Yes |
| Argmax tracking (for tropical backward) | No | Yes |
| Automatic differentiation | No | Yes (`einsum_with_grad`) |

omeinsum-rs separates the scalar type from the algebraic operations:
```rust
Standard<f64>  // add = +,   mul = ×
MaxPlus<f64>   // add = max, mul = +
MinPlus<f64>   // add = min, mul = +
```

strided-rs ties operations to the type itself via `ScalarBase: Zero + One + Add + Mul`.

### 2.3 GEMM Backends

| Backend | strided-rs | omeinsum-rs |
|---------|-----------|-------------|
| faer (f32/f64) | Yes (default) | Yes |
| CBLAS | Yes (feature flag) | No |
| Naive loop fallback | Yes | Yes (generic) |
| Tropical GEMM (SIMD) | No | Yes (`tropical-gemm`) |
| Pluggable backend trait | Yes (`BgemmBackend<T>`) | No (fixed dispatch) |
| CUDA (cuTENSOR) | No | Yes (optional feature) |

**Current limitation in strided-rs:** GEMM backends are selected at
**compile time** via mutually exclusive Cargo features (`faer`, `blas`,
`blas-inject`). The `ActiveBackend` type alias resolves to exactly one
backend, and enabling multiple features simultaneously triggers
`compile_error!`. This prevents runtime backend selection and complicates
downstream integration (see [issue #86](https://github.com/tensor4all/strided-rs/issues/86)).

The `einsum2_with_backend_into` function already accepts an explicit
backend type parameter `B: BgemmBackend<T>`, which is the foundation for
runtime selection — but the compile-time exclusivity of feature flags
prevents multiple backends from coexisting in the same binary.

### 2.4 Contraction Tree Optimization

Both use **omeco** for contraction order optimization. The difference is
purely in which optimizer options are exposed:

| Option | strided-rs | omeinsum-rs |
|--------|-----------|-------------|
| Greedy | Yes | Yes |
| TreeSA (simulated annealing) | Not exposed | Yes |

This is **not a fundamental difference** — enabling TreeSA in strided-rs
requires only changing the omeco call-site options.

---

## 3. Performance Optimizations

### 3.1 strided-rs Advantages

These optimizations exist in strided-rs and are absent from omeinsum-rs:

**Cache-optimized CPU kernels (strided-kernel):**
- Dimension fusion: contiguous dimensions are fused to reduce loop overhead
- Dimension reordering: sorted by stride magnitude for optimal cache access
- L1 tiled iteration: operations blocked into tiles fitting L1 cache (32 KB)
- Contiguous fast paths: bypass blocking for direct iteration

**Einsum-specific fast paths:**
- **Element-wise bypass:** When all axes are batch (no contraction), GEMM is
  completely bypassed in favor of `zip_map2_into` — avoids GEMM dispatch
  overhead for Hadamard products
- **Single-tensor fast paths:** Direct trace loop (diagonal stride trick),
  partial trace optimization, zero-copy permutation
- **Fusability-aware Reshape-to-GEMM:** `try_fuse_group` checks whether
  dimension groups within an operand are contiguous. If fusable, the
  existing strides are used directly for GEMM (no copy). Only non-fusable
  groups trigger a copy to col-major buffer.
- **Buffer pool:** Intermediate tensor buffers are recycled across
  contraction steps, reducing allocation pressure
- **Owned-input optimization:** `einsum2_into_owned` transfers ownership
  of input arrays to avoid buffer copies for non-contiguous operands

### 3.2 omeinsum-rs Advantages

- **Tropical GEMM with SIMD:** `tropical-gemm` crate provides optimized
  SIMD kernels for tropical algebra contractions
- **Argmax tracking:** Batched contraction can track argmax indices for
  tropical backward pass
- **GPU dispatch:** cuTENSOR handles reshape/permute/contraction on GPU
  in a single call

### 3.3 Shared Strategies (Both Implement)

- **Reshape-to-GEMM:** Both reshape multi-dimensional tensors to
  `[batch, left, contract]` × `[batch, contract, right]` for GEMM.
  strided-rs is more sophisticated (fusability check avoids unnecessary
  copies); omeinsum-rs always materializes the reshape.
- **Batched GEMM:** Both handle batch dimensions (strided-rs via
  `batch_strides` in `ContiguousOperand`; omeinsum-rs via batch slicing)
- **Zero-copy views:** permute/reshape are metadata-only in both

### 3.4 Apparent Differences That Are Not Fundamental

| Claimed difference | Reality |
|--------------------|---------|
| Arc shared storage (cheap clone + zero-copy view) | Borrowing (`&'a [T]`) achieves the same zero-copy semantics without atomic refcount overhead. Arc's benefit is ergonomic (no lifetime annotations), not performance. |
| TreeSA optimization | Both use omeco; TreeSA is an option, not a separate implementation. |
| Batched GEMM | Both support it with different representations. |

---

## 4. GPU Backend Considerations

### Why strided-kernel Is Irrelevant for GPU

`strided-kernel` operates via raw CPU pointer arithmetic:
```rust
let val = unsafe { *src_ptr.offset(idx) };  // CPU deref
*dst_ptr.offset(out_idx) = result;           // CPU write
```

GPU operations (cuBLAS, cuTENSOR) handle the entire
reshape → contract → permute pipeline on-device. The CPU side only needs:

1. **Contraction tree optimization** (omeco — pure graph computation, shared)
2. **GPU kernel dispatch** (cuTENSOR call, entirely separate from strided-kernel)
3. **Device memory management** (allocate/free/transfer)

Therefore, adding GPU support does **not** require modifying `strided-kernel`
or `strided-view`. It requires a new GPU backend at the einsum layer only.

### Arc vs Borrowing for GPU

omeinsum-rs uses `Arc<B::Storage<T>>` where `Storage` is a trait that can
be implemented for GPU memory. This is a reasonable design for GPU memory
management (reference-counted device buffers).

However, for the CPU path, `Arc` adds unnecessary atomic overhead compared
to strided-rs's borrow-based approach. The two concerns are orthogonal:
- CPU tensors: borrowing is optimal
- GPU tensors: reference counting (or arena allocation) is practical

A merged design can use different storage strategies per backend without
forcing Arc onto the CPU path.

---

## 5. Proposed Merge Strategy

### 5.1 High-Level Architecture

```
                    User API (einsum, einsum_into)
                              │
                    ┌─────────┴─────────┐
                    │   strided-opteinsum │  ← Algebra<T> trait
                    │   (contraction tree,│     (Standard, MaxPlus, ...)
                    │    omeco optimizer) │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴──────────────────┐
                    │   strided-einsum2           │
                    │   einsum2_with_backend_into │  ← primary API
                    │   (binary contract,         │
                    │    fast paths)              │
                    └────┬─────────┬─────────┬───┘
                         │         │         │
              ┌──────────┴──┐  ┌───┴───┐  ┌──┴───────────┐
              │ CPU path    │  │  ...   │  │ GPU path     │
              │             │  │       │  │              │
              │ strided-    │  │ other │  │ cuTENSOR /   │
              │ cpu-kernel  │  │ BgemmB│  │ cuBLAS       │
              │ (map/reduce/│  │ impls │  │ (device-side │
              │  broadcast) │  │       │  │  contraction)│
              │      +      │  │       │  │              │
              │  ┌────────┐ │  └───────┘  └──────────────┘
              │  │Runtime │ │
              │  │backend │ │
              │  │select: │ │
              │  │ faer   │ │ (#86: coexist in same binary,
              │  │ BLAS   │ │  runtime match at call site)
              │  │tropical│ │
              │  │ naive  │ │
              │  └────────┘ │
              └─────────────┘
```

### 5.2 Specific Changes

#### Phase 0: Runtime GEMM Backend Selection ([#86](https://github.com/tensor4all/strided-rs/issues/86)) (Small-Medium effort)

**Problem:** strided-einsum2 currently enforces mutually exclusive Cargo
features (`faer` vs `blas` vs `blas-inject`). This prevents multiple GEMM
backends from coexisting in the same binary and blocks runtime backend
selection.

**Changes:**

1. **Allow `faer` + `cblas-inject` simultaneous compilation.** Remove the
   `compile_error!` for this combination. Both `FaerBackend` and
   `BlasBackend` (with their `BgemmBackend<T>` impls) coexist in the same
   binary. Default features become `faer` + `cblas-inject`.

   ```toml
   [features]
   default = ["faer", "cblas-inject"]
   faer = ["dep:faer", "dep:faer-traits"]
   cblas-inject = ["dep:cblas-inject", "dep:num-complex"]
   blas = ["dep:cblas-sys", "dep:num-complex"]  # exclusive with cblas-inject
   ```

   `blas` + `cblas-inject` remains `compile_error!` (symbol conflict).

2. **Deprecate `ActiveBackend` type alias.** `ActiveBackend` is a
   compile-time-only concept that selects a single backend. Keep it for
   backward compatibility but mark as `#[deprecated]`. Downstream users
   should migrate to `einsum2_with_backend_into` with explicit backend type.

3. **`einsum2_with_backend_into` becomes the primary API.** It already
   accepts `B: BgemmBackend<T> + BackendConfig` as a type parameter.
   Downstream libraries perform runtime dispatch at the call site via match:

   ```rust
   match preferred_backend {
       EinsumBackend::Faer => einsum2_with_backend_into::<T, FaerBackend, _>(...),
       EinsumBackend::Blas => einsum2_with_backend_into::<T, BlasBackend, _>(...),
       EinsumBackend::Naive => einsum2_naive_into(...),
   }
   ```

   Each branch is monomorphized — no dynamic dispatch overhead in the
   GEMM hot loop.

**Impact on later phases:** This change is a prerequisite for the merged
architecture. When tropical-gemm is added (Phase 1), it becomes another
`BgemmBackend` impl that coexists with faer and BLAS. The runtime dispatch
pattern naturally extends to new backends.

#### Phase 1: Algebra Trait Introduction (Medium effort)

Import the Semiring/Algebra abstraction from omeinsum-rs into the einsum
layer:

```rust
pub trait Semiring: Copy + Send + Sync {
    type Scalar;
    fn zero() -> Self;
    fn one() -> Self;
    fn add(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
}

pub trait Algebra: Semiring {
    // Optional: backward pass, argmax tracking
    fn needs_argmax() -> bool { false }
    // ...
}
```

**Key design point:** The Algebra trait lives at the einsum layer only.
`strided-kernel` remains unchanged — it already uses closures for
map/reduce operations, making it algebra-agnostic.

Extend `BgemmBackend` to dispatch based on algebra:
- `Standard<f64>` / `Standard<f32>` → faer GEMM
- `MaxPlus<f64>` → tropical-gemm
- `Standard<i32>` → naive loop
- Custom algebras → user-provided `BgemmBackend` implementation

#### Phase 2: Generalize EinsumOperand (Medium effort)

The current `EinsumOperand` enum has hardcoded variants:
```rust
pub enum EinsumOperand<'a> {
    F64(StridedData<'a, f64>),
    C64(StridedData<'a, Complex64>),
}
```

Adding f32, i32, etc. causes **variant explosion**. Replace with a generic
design:
```rust
pub struct EinsumOperand<'a, T> {
    data: StridedData<'a, T>,
}
```

**Challenge:** The current enum enables mixed-type promotion (f64 + C64 →
C64) via runtime dispatch. A fully generic design loses this. Options:
1. Require homogeneous types (caller promotes beforehand)
2. Use a trait-based promotion system
3. Keep a small enum for the common f64/C64 case, with generic escape hatch

This is the **highest-risk design decision** and should be resolved before
implementation begins.

#### Phase 3: f32 and Integer Support (Small effort)

- f32/Complex32: faer already supports these via `ComplexField`. Mainly
  requires extending opteinsum's operand handling.
- Integer types: Route through naive backend (no BLAS/faer GEMM for
  integers). The `BgemmBackend` pluggable trait already supports this.

#### Phase 4: Rename strided-kernel (Small effort, do with Phase 6)

Rename `strided-kernel` → `strided-cpu-kernel` to clarify the architectural
boundary. Only necessary when a GPU kernel crate is actually introduced.

#### Phase 5: GPU Backend (Large effort)

Add a GPU einsum backend trait:
```rust
pub trait GpuEinsumBackend<A: Algebra> {
    type DeviceTensor;
    fn contract_binary(
        a: &Self::DeviceTensor,
        b: &Self::DeviceTensor,
        plan: &Einsum2Plan<impl AxisId>,
        alpha: A::Scalar,
        beta: A::Scalar,
    ) -> Result<Self::DeviceTensor>;
}
```

This is entirely independent of the CPU path. The contraction tree
optimizer (omeco) is shared; only the leaf-level contraction dispatch
differs.

#### Phase 6: Gradient Support (Medium effort, optional)


Import `einsum_with_grad` from omeinsum-rs as an optional feature. This
depends on the Algebra trait's `add_backward`/`mul_backward` methods and
argmax tracking. For standard linear algebra, gradient is not needed, so
it should remain opt-in.

### 5.3 What Does NOT Change

- `strided-kernel` internals (cache-optimized map/reduce/broadcast)
- `strided-view` (borrow-based `StridedArrayView`)
- Element-wise fast path in einsum2
- Single-tensor fast paths in opteinsum (trace, diagonal, permutation)
- Buffer pool in opteinsum
- Fusability-aware contiguous operand preparation

These are strided-rs's core performance advantages and are unaffected by
the merge.

---

## 6. Risk Assessment

| Phase | Change | Risk | Mitigation |
|-------|--------|------|------------|
| 0 | Runtime backend selection ([#86](https://github.com/tensor4all/strided-rs/issues/86)) | Low-Medium | `einsum2_with_backend_into` already exists; main work is removing `compile_error!` and testing coexistence |
| 1 | Algebra trait introduction | Low | Additive change; existing `Scalar`/`ScalarBase` paths continue to work |
| 2 | EinsumOperand generalization | **High** | Mixed-type promotion design must be resolved first |
| 3 | f32 / integer support | Low | faer already supports f32; integers route through naive backend |
| 4 | Kernel rename | Low | Mechanical rename; defer until Phase 5 |
| 5 | GPU backend | Medium | Independent from CPU path; risk is in cuTENSOR integration |
| 6 | Gradient support | Medium | Optional feature; can be deferred |

---

## 7. Summary

The two projects have complementary strengths:

- **strided-rs** excels at CPU performance: cache-optimized kernels,
  fusability-aware GEMM preparation, element-wise/trace fast paths,
  buffer pooling, and pluggable GEMM backends.

- **omeinsum-rs** excels at algebraic generality: systematic Semiring/Algebra
  abstraction, tropical GEMM with SIMD, gradient computation, GPU dispatch,
  and broad scalar type support.

The proposed merge introduces omeinsum-rs's Algebra layer into strided-rs's
einsum crates while preserving all existing CPU optimizations. The GPU path
is added as an independent backend, and `strided-kernel` remains a
CPU-specific, high-performance foundation that requires no changes.

The highest-priority design decision is **how to generalize `EinsumOperand`**
without losing mixed-type promotion capabilities. This should be designed
and agreed upon before implementation begins.
