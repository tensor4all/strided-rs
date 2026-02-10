# tenet: Physics-Oriented Tensor Computing Framework

**Version:** 2.0 (Feb 2026)
**Status:** Design Draft

---

## 1. Executive Summary

**tenet** is a pure-Rust tensor computing framework for physics and tensor network (TN) computation. It combines neural network (NN) capabilities via Burn integration with native complex number support, automatic differentiation (Wirtinger-aware), and GPU acceleration — filling a gap that no existing Rust library covers.

**Vision:** A Rust-native alternative to the subset of libtorch used in scientific computing, built incrementally on proven components.

```
libtorch (C++)        →  tenet (Rust)
─────────────────────────────────────
c10 (core infra)         strided-rs (Layer 0-1)
ATen (tensor ops)        strided-kernel/einsum/linalg
autograd                 tenet-ad (Wirtinger-aware)
torch.nn                 Burn (via tenet-burn bridge)
CUDA/ROCm backend        CubeCL + cuBLAS/hipBLAS dlopen
complex tensor           strided-rs (native, day one)
```

**Primary types:** f64, Complex128 (scientific computing).
**Future types:** f32, Complex64, f16/bf16 (ML integration).

## 2. Motivation

### 2.1 The Problem

Scientific computing — particularly tensor network methods (DMRG, TEBD, TCI) and physics-informed neural networks — requires:

- **Complex128 tensors** with full autodiff (Wirtinger derivatives)
- **Einsum** with contraction tree optimization
- **Dense linear algebra** (SVD, QR, eigendecomposition) on GPU
- **NN layers** (Linear, Attention) interoperable with TN operations
- **GPU acceleration** (CUDA, ROCm) without build-time SDK dependencies

No existing Rust framework provides all of these. The ecosystem is fragmented:

| Library | Tensor ops | Complex | Autodiff | GPU | NN |
|---------|:---:|:---:|:---:|:---:|:---:|
| strided-rs | Yes | Yes | No | No | No |
| ndtensors | Yes | Yes | Yes | No | No |
| Burn | Yes | No | Yes | Yes | Yes |
| Candle | Yes | No | No | Yes | Yes |

### 2.2 Why Not Just Use libtorch?

libtorch covers everything above. However:

- **C++ build chain** is a major pain point (CMake, nvcc, 2GB+ binary)
- **No `cargo add`** — manual linking, platform-specific setup
- **License:** BSD-3-Clause (permissive), but the C++ dependency itself is the barrier
- **Deployment:** Cannot cross-compile easily, no Wasm target

A pure Rust stack eliminates these issues while maintaining comparable performance for the operation classes that physics needs.

### 2.3 What We Already Have

| libtorch equivalent | Existing Rust asset | Status |
|---------------------|---------------------|--------|
| c10 (core infra) | strided-kernel, strided-view | Complete |
| ATen (tensor ops) | strided-einsum2, strided-opteinsum | Complete |
| autograd | ndtensors (proven prototype) | Needs refinement |
| complex tensor | strided-rs | Complete |
| einsum | strided-opteinsum (with contraction tree) | Complete |

The remaining gaps are: **GPU dispatch**, **linear algebra**, **Burn bridge**, and **production AD**.

## 3. Architecture Overview

```
Layer 3: Domain Applications
  ├── Symmetry tensors (U(1), SU(2) block selection rules)
  ├── Named index / ITensor-like API
  ├── TN algorithms (DMRG, TEBD, TCI, ...)
  └── Physics-informed NN

Layer 2: tenet (this framework)
  ├── tenet-core         GPU/CPU dispatch, device abstraction (CubeCL)
  ├── tenet-ad           Automatic differentiation (Wirtinger-aware)
  ├── tenet-block        Block-sparse tensor (generic, symmetry-agnostic)
  └── tenet-burn         Burn interop (NN modules, real tensor bridge)

Layer 1: strided-rs (device-aware operations)
  ├── strided-linalg     SVD, QR, LU, Cholesky, eigen
  ├── strided-einsum2    Binary einsum (GEMM backend)
  └── strided-opteinsum  N-ary einsum with contraction tree

Layer 0: strided-rs foundation (pure CPU)
  ├── strided-view       StridedArrayView types (zero-copy)
  └── strided-kernel     CPU cache-optimized map/reduce/broadcast
```

**Key principle:** strided-rs (Layer 0-1) remains a **pure CPU library** with no GPU dependencies. All GPU functionality lives in tenet-core (Layer 2), which depends on CubeCL for JIT kernels and dlopen for vendor BLAS/LAPACK.

## 4. Layer 0-1: strided-rs (CPU Foundation)

### 4.1 Existing Assets

| Crate | Purpose | Status |
|-------|---------|--------|
| strided-view | StridedArrayView/Mut with zero-copy slice, permute, reshape, broadcast | Complete |
| strided-kernel | Cache-optimized map/reduce/broadcast (port of Strided.jl) | Complete |
| strided-einsum2 | Binary einsum with GEMM backend (faer) | Complete |
| strided-opteinsum | N-ary einsum with contraction tree optimization (opt_einsum) | Complete |

**Cache optimization strategy** (faithful to Strided.jl):
1. Dimension fusion for contiguous dims
2. Stride-based dimension reordering
3. L1-cache-fitted tiled iteration (32KB blocks)
4. Contiguous fast paths bypass blocking

### 4.2 strided-linalg (Planned)

Dense linear algebra decompositions. CPU-only initially, GPU dispatch added via tenet-core later.

| Operation | CPU Backend | GPU Backend (via tenet-core) |
|-----------|-------------|------------------------------|
| SVD | faer / Accelerate | cuSOLVER `Zgesvd` / hipSOLVER |
| QR | faer / Accelerate | cuSOLVER `Zgeqrf` / hipSOLVER |
| LU | faer / Accelerate | cuSOLVER `Zgetrf` / hipSOLVER |
| Cholesky | faer / Accelerate | cuSOLVER `Zpotrf` / hipSOLVER |
| Eigen | faer / Accelerate | cuSOLVER `Zheevd` / hipSOLVER |

## 5. Layer 2: tenet

### 5.1 tenet-core: Device Abstraction & GPU Dispatch

Central device management layer. All GPU-related functionality lives here — strided-rs remains pure CPU.

**Responsibilities:**
- `Device` enum and runtime GPU detection
- Unified `Stream` abstraction (Rayon / CUDA stream / HIP stream)
- GPU memory pool (slab allocation, async-safe recycling)
- BLAS dispatch: cuBLAS/hipBLAS via dlopen
- LAPACK dispatch: cuSOLVER/hipSOLVER via dlopen
- GPU element-wise kernels: CubeCL JIT
- `TensorPromise` for async execution

**Device & Backend Matrix:**

| Platform | Backend | GEMM Provider | f64 | Complex128 |
|----------|---------|---------------|:---:|:----------:|
| CPU (generic) | faer / OpenBLAS | faer | Yes | Yes |
| CPU (Apple Silicon) | Apple Accelerate | AMX coprocessor | Yes | Yes |
| NVIDIA GPU | CUDA | cuBLAS via dlopen | Yes | Yes |
| AMD GPU | ROCm | hipBLAS via dlopen | Yes | Yes |

**Excluded: Apple Metal GPU** — MSL has no `double` (f64) type. No FP64 ALUs on Apple GPU cores. On Apple Silicon, the CPU path via Accelerate/AMX is the correct backend for scientific computing (~1 TFLOPS FP64 on M4).

```rust
enum Device {
    Cpu(usize),          // CPU socket index (NUMA node)
    Cuda(usize),         // NVIDIA GPU index
    Rocm(usize),         // AMD GPU index
}
// No Metal variant — Apple Silicon uses Cpu(0) with Accelerate auto-detected
```

**Runtime Loading (Zero Build-Time GPU Dependencies):**

The tenet binary compiles with only a Rust toolchain. All GPU libraries loaded at runtime via `dlopen`:

| Library | Shared Object | Purpose |
|---------|---------------|---------|
| CUDA Driver | `libcuda.so` | Device management, memory, kernel launch |
| cuBLAS | `libcublas.so` | GEMM (Zgemm for complex128) |
| cuSOLVER | `libcusolver.so` | SVD, QR, eigen |
| NVRTC | `libnvrtc.so` | Runtime kernel compilation (fallback) |
| HIP Runtime | `libamdhip64.so` | ROCm device management |
| hipBLAS | `libhipblas.so` | ROCm GEMM |
| hipSOLVER | `libhipsolver.so` | ROCm SVD, QR, eigen |

CUDA: via `cudarc` crate (proven dlopen approach).
ROCm: via custom `hiparc` crate (HIP API parallels CUDA — `cu*` → `hip*`). Feasibility proven by AMD Orochi, Blender, TensorFlow-ROCm.

**Feature Flags:**

```toml
[features]
default = ["cpu-faer"]
cpu-faer = ["faer"]
cpu-accelerate = ["accelerate-src"]
cuda = ["cudarc"]
rocm = ["libloading"]
```

### 5.2 tenet-ad: Automatic Differentiation

Wirtinger-aware reverse-mode AD for complex tensor operations. Based on proven ndtensors-rs approach.

**Why not Burn's autodiff?**
Burn's `TensorKind` has only `Float`, `Int`, `Bool` — no `Complex`. Complex AD requires Wirtinger derivatives (`∂f/∂z` and `∂f/∂z̄`), which Burn's real-valued chain rule cannot provide. Operations like `conj`, `abs` would produce incorrect gradients.

**Scope:** Differentiate operations that tensor networks actually use:

| Operation | Forward | Backward |
|-----------|---------|----------|
| einsum | strided-opteinsum | Transpose subscripts + einsum |
| SVD | strided-linalg | Seeger et al. 2017 |
| QR | strided-linalg | Known formula |
| exp(z) | element-wise | exp(z) * grad |
| Element-wise (+, *, conj) | strided-kernel | Standard + Wirtinger rules |

```rust
pub struct TrackedTensor<T> {
    data: StridedArray<T>,       // f64 or Complex64
    node: Option<NodeRef>,       // computation graph node
}

pub trait WirtingerBackward {
    fn backward_holomorphic(&self, grad: &Tensor) -> Tensor;   // ∂f/∂z
    fn backward_conjugate(&self, grad: &Tensor) -> Tensor;     // ∂f/∂z̄
}
```

### 5.3 tenet-block: Block-Sparse Tensor

Generic block-sparse tensor. **No symmetry concepts** — block selection logic is the caller's responsibility (Layer 3).

```rust
pub struct BlockTensor<T> {
    blocks: HashMap<BlockIndex, StridedArray<T>>,
    structure: BlockStructure,
}

// Block-level einsum — caller provides contraction pairs
fn block_einsum<T>(
    subscripts: &str,
    a: &BlockTensor<T>,
    b: &BlockTensor<T>,
    pairs: &[(BlockIndex, BlockIndex, BlockIndex)],
) -> BlockTensor<T>;
```

Layer 3 drives block selection:

```rust
// Layer 3: physics-specific
let pairs = u1_contraction_pairs(&a_structure, &b_structure);
let result = block_einsum("ij,jk->ik", &a, &b, &pairs);
```

### 5.4 tenet-burn: Burn Integration

Bridge between tenet and Burn, enabling NN module reuse without reimplementation.

**Strategy:** Use Burn's existing NN modules (Linear, LayerNorm, Attention, etc.) for real-valued operations, supplemented by tenet-ad for complex-valued operations.

**Short-term (Burn has no Complex):**

```
Burn Autodiff (real only)      tenet-ad (complex + real)
     │                               │
     │  real Tensor ←→ real Tensor   │
     │  via TensorData (CPU copy)    │
     │                               │
     NN layers (Burn)    ←→    TN ops (tenet-ad)
```

- NN layers: Burn's autodiff (mature, GPU-accelerated)
- TN operations on complex tensors: tenet-ad
- Interface: Exchange real-valued tensors via `TensorData`
- Real-valued TN ops: Register as Burn Custom Backward Ops

**Long-term (Burn Complex Kind contribution):**

```rust
// Unified autodiff graph — NN + TN in one graph
let nn_output: Tensor<Autodiff<B>, 2, Float> = model.forward(input);
let tn_input: Tensor<Autodiff<B>, 2, Complex> = nn_output.to_complex();
let result = einsum_burn("ij,jk->ik", &tn_input, &mps_tensor);
let grads = result.backward();  // gradient flows through both NN and TN
```

Prerequisites for upstream Burn contribution:
1. Add `Complex` to `TensorKind` (with `ComplexTensorOps` trait)
2. Add Wirtinger derivative support to `burn-autodiff`
3. CubeCL complex type or interleaved-f64 lowering
4. At least one backend implementation

## 6. GPU Kernel Strategy

### 6.1 Operation Classes

```
tenet-core
├── Contraction (GEMM)
│   ├── CPU: faer / Accelerate
│   ├── CUDA: cuBLAS Zgemm via dlopen
│   └── ROCm: hipBLAS Zgemm via dlopen
│
├── Linear Algebra (SVD, QR, eigen)
│   ├── CPU: faer / Accelerate LAPACK
│   ├── CUDA: cuSOLVER via dlopen
│   └── ROCm: hipSOLVER via dlopen
│
├── Element-wise (exp, log, conj, sqrt, ...)
│   ├── CPU: strided-kernel
│   └── GPU: CubeCL JIT kernels (interleaved f64 pairs)
│       └── Fallback: NVRTC/HIPRTC string kernels
│
└── Fallback
    └── CPU execution when GPU path unavailable
```

### 6.2 CubeCL for Element-wise Kernels

Write kernels in Rust with `#[cube]` macro. CubeCL compiles to PTX/SPIR-V at runtime. No nvcc needed.

```rust
#[cube(launch)]
fn complex_exp(input: &Tensor<f64>, output: &mut Tensor<f64>) {
    let idx = ABSOLUTE_POS * 2;
    let re = input[idx];
    let im = input[idx + 1];
    let e = f64::exp(re);
    output[idx]     = e * f64::cos(im);
    output[idx + 1] = e * f64::sin(im);
}
```

**Complex numbers on GPU** use interleaved real pairs (no native complex type):

```
Complex128: [re₀, im₀, re₁, im₁, ...]  (f64 × 2N)
Complex64:  [re₀, im₀, re₁, im₁, ...]  (f32 × 2N)
```

ABI-compatible with cuBLAS `cuDoubleComplex` / hipBLAS `hipDoubleComplex`.

### 6.3 CubeCL Status & Required Upstream Work

| Issue | Status | Action |
|-------|--------|--------|
| f64 on CUDA/ROCm | Broken (shared memory overflow in matmul, disabled since Jan 2025) | Contribute fix PR (~few lines) |
| Complex type in IR | Not supported | Long-term: contribute complex type |
| Element-wise f64 | Would work if f64 re-enabled (no shared memory) | Blocked on f64 fix |

**CubeCL f64 fix is small:** The matmul autotuner doesn't account for element size when selecting tile dimensions. Fix: halve tile dims for 8-byte types, or exclude f64 from matmul while enabling element-wise.

### 6.4 NVRTC/HIPRTC Fallback

If CubeCL f64 remains broken, embed CUDA/HIP kernel source as Rust string literals and compile at runtime:

```rust
const COMPLEX_EXP_KERNEL: &str = r#"
extern "C" __global__ void complex_exp(
    const double* re_im_in, double* re_im_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double re = re_im_in[2*i], im = re_im_in[2*i+1];
        double e = exp(re);
        re_im_out[2*i]   = e * cos(im);
        re_im_out[2*i+1] = e * sin(im);
    }
}
"#;
```

## 7. Type Support & Roadmap

### 7.1 Current State (strided-rs)

| Type | Status | Notes |
|------|--------|-------|
| f64 | Full support | Primary type |
| Complex128 (f64-based) | Full support | Primary type |
| f32 | Internal SIMD paths | Not in public API |

### 7.2 Target Matrix

| Type | CPU | GPU GEMM | GPU element-wise | Priority |
|------|:---:|:--------:|:----------------:|----------|
| f64 | faer/Accelerate | cuBLAS `Dgemm` / hipBLAS | CubeCL (after f64 fix) | P0 |
| Complex128 | faer/Accelerate | cuBLAS `Zgemm` / hipBLAS | CubeCL interleaved f64 | P0 |
| f32 | faer/Accelerate | cuBLAS `Sgemm` / hipBLAS | CubeCL (works today) | P1 |
| Complex64 | faer/Accelerate | cuBLAS `Cgemm` / hipBLAS | CubeCL interleaved f32 | P1 |
| f16 / bf16 | Software | cuBLAS `Hgemm` / hipBLAS | CubeCL (works today) | P2 (ML) |

## 8. Burn Integration Strategy

### 8.1 Phase 1: Real Tensor Bridge (Short-term)

Exchange real-valued tensors between Burn and tenet via `TensorData` (CPU memory copy). Burn handles NN forward/backward; tenet-ad handles complex TN operations.

### 8.2 Phase 2: Custom Backward Ops (Medium-term)

Register tenet operations (einsum, SVD) as Burn Custom Backward Ops for real-valued inputs. This enables gradient flow through mixed NN+TN graphs for real tensors without modifying Burn.

### 8.3 Phase 3: Burn Complex Kind (Long-term)

Contribute `Complex` as a new `TensorKind` to Burn upstream. This requires:
- `Complex` kind with `ComplexTensorOps` trait
- Wirtinger derivative support in `burn-autodiff`
- CubeCL complex type or interleaved lowering
- Previous Burn PRs (#3330, #3608) went stale — needs a champion

## 9. Comparison with Existing Ecosystems

| | libtorch | Candle | Burn | tenet |
|---|---------|--------|------|-------|
| **Language** | C++ (Python API) | Rust | Rust | Rust |
| **Complex** | Yes (v1.9+) | No | No | Yes (day one) |
| **Autodiff** | Yes | No | Yes | Yes (Wirtinger) |
| **GPU** | CUDA, ROCm, MPS | CUDA, Metal | CUDA, Metal, WGPU | CUDA, ROCm |
| **NN modules** | Full | Full | Full | Via Burn bridge |
| **Einsum** | Yes (basic) | No | No | Yes (optimized contraction tree) |
| **Block tensor** | No | No | No | Yes |
| **SVD backward** | Yes | No | Yes | Yes |
| **Build deps** | CMake, nvcc, 2GB+ | Rust only | Rust only | Rust only |
| **License** | BSD-3-Clause | MIT/Apache | MIT/Apache | MIT/Apache |
| **Target** | General ML+Science | ML inference | ML training+inference | Physics + TN + NN |

**tenet's niche:** Complex-valued tensor computation with AD, optimized einsum, block tensors, and GPU acceleration — the specific combination needed for tensor network and first-principles physics calculations. NN capabilities come from Burn rather than reimplementation.

## 10. Open Questions & Future Work

### Design Questions
- Device placement: type-level vs. runtime enum?
- Cross-device transfer: copy semantics vs. move semantics?
- `TensorPromise` as Rust `Future` for `async`/`await` interop?
- Accelerate auto-detection on macOS vs. explicit feature flag?
- CubeCL f64 fix: upstream PR vs. maintained fork?
- NVRTC/HIPRTC kernel caching strategy (compile once, reuse)?
- ROCm device validation to avoid crashes on unsupported architectures?

### Implementation Roadmap
1. **strided-linalg** — CPU SVD/QR via faer (enables TN algorithms)
2. **tenet-ad** — Wirtinger AD based on ndtensors (enables optimization)
3. **tenet-core** — cuBLAS/hipBLAS dlopen for GPU GEMM
4. **tenet-burn** — Real tensor bridge to Burn NN modules
5. **CubeCL f64 fix** — Upstream PR to unblock GPU element-wise
6. **tenet-block** — Generic block-sparse tensor
7. **Burn Complex Kind** — Long-term upstream contribution

## 11. Dependencies

| Crate | Role | License | Status |
|-------|------|---------|--------|
| `strided-rs` | CPU tensor ops foundation | MIT/Apache | Existing |
| `faer` | CPU GEMM + LAPACK | MIT | Stable |
| `accelerate-src` | Apple Accelerate BLAS | MIT/Apache | Stable |
| `cudarc` | CUDA dlopen | MIT/Apache | Stable |
| `libloading` | Generic dlopen (ROCm) | ISC | Stable |
| `cubecl` | GPU kernel JIT | MIT/Apache | f64 broken |
| `burn` | NN framework + autodiff | MIT/Apache | No complex |
| `num-complex` | Complex number types | MIT/Apache | Stable |
| `opt-einsum` (omeco) | Contraction tree optimizer | MIT/Apache | Existing |

---
*v2.0: Complete rewrite. Reframed around tenet as physics-oriented framework built on strided-rs + CubeCL + Burn.*
