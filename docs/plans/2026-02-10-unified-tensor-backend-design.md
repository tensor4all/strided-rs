# strided-rs: Unified Tensor Backend Design Specification
**Version:** 1.5 (Feb 2026)
**Scope:** Tensor4all Layer 1 Integration (CPU/NVIDIA/AMD)

---

## 1. Executive Summary

This document specifies the design for a multi-device (CPU, CUDA, ROCm) tensor computation library built on top of `strided-rs`. The goal is to combine the ergonomic "View" abstraction for mathematical operations with asynchronous execution and advanced memory management.

**Primary types:** Float64 and Complex128 (tensor network computation).
**Future types:** Float32, Complex64, Float16/BF16 (ML integration, lower-precision workloads).

## 2. Device & Backend Matrix

### 2.1 Supported Backends

| Platform | Backend | GEMM Provider | Float64 | Complex128 |
|----------|---------|---------------|:-------:|:----------:|
| CPU (generic) | faer / OpenBLAS | faer | Yes | Yes |
| CPU (Apple Silicon) | Apple Accelerate | AMX coprocessor | Yes | Yes |
| NVIDIA GPU | CUDA | cuBLAS / cuTENSOR | Yes | Yes |
| AMD GPU | ROCm | hipBLAS / hiptensor | Yes | Yes |

### 2.2 Excluded: Apple Metal GPU

Metal Shading Language (MSL) **does not support `double` (float64)** — this is a hardware-level constraint with no FP64 ALUs on Apple GPU cores. FP64 emulation via integer ALUs yields 1/32–1/64 of FP32 throughput, making it impractical.

| Type | Metal GPU Support |
|------|:-:|
| float32 | Yes |
| float64 | No (emulation only, ~1/50 speed) |
| complex64 | MLX only (via real GEMM pairs) |
| complex128 | No (not in any framework) |

**Conclusion:** On Apple Silicon, the CPU path via Accelerate/AMX is the correct and only viable backend for scientific computing. The Apple M-series AMX coprocessor provides native FP64 GEMM at competitive performance (~1 TFLOPS on M4).

### 2.3 Device Enum

```rust
enum Device {
    Cpu(usize),          // CPU socket index (NUMA node)
    Cuda(usize),         // NVIDIA GPU index
    Rocm(usize),         // AMD GPU index
}
```

No `Metal` variant — Apple Silicon users use `Device::Cpu(0)` with Accelerate auto-detected at compile time.

## 3. Core Components

### 3.1 StridedArray & StridedView

- **StridedArray (Owner)**: The entity that owns the physical buffer. Wraps an `Arc<Storage>` for reference-counted lifetime management.
- **StridedView (Accessor)**: A non-owning reference holding only metadata (`shape`, `strides`, `offset`).
- **Disjoint Slicing**:
    - Safely partition a large buffer into multiple non-overlapping `StridedView`s.
    - This enables concurrent writes to a single `StridedArray` from multiple GPU/CPU streams while maintaining Rust's safety guarantees (`Send`/`Sync`).

### 3.2 Unified Stream

- **Concept**: Abstract all execution contexts (CPU and GPU) as "FIFO task queues".
- **CPU Stream**: Implemented as task submission to a thread pool (e.g., Rayon).
- **GPU Stream**: Maps to CUDA Stream or ROCm HIP Stream.
- Apple Silicon: CPU stream only (no GPU stream).

## 4. Asynchronous Execution and the Promise Pattern

### 4.1 TensorPromise

- **Role**: A "reservation" object returned immediately by operations such as `einsum` or `SVD`.
- **States**: `Pending` (computation in progress) or `Ready` (completed).
- **Automatic synchronization logic**:
    - When an operation function receives a `TensorPromise` as an argument, if it was produced on a different device/stream than the current one, the runtime automatically inserts a `WaitEvent` internally to resolve the dependency.

### 4.2 Synchronization Method (`.wait()`)

- Called explicitly only when the user needs to read actual data on the CPU side.
- Blocks the host CPU thread and waits for device-side computation to complete.

## 5. Memory Management Strategy

### 5.1 Smart Memory Pool

- **Algorithm**: "Larger-serves-smaller" reuse strategy (Slab Allocation).
- **Recycling**: When a `StridedArray` is dropped, its buffer is returned to the pool rather than being immediately deallocated.
- **Async safety**: Buffers remain locked against reallocation until their associated `Event` has completed.

### 5.2 NUMA Affinity (CPU-Specific)

- In multi-CPU environments, maintain an independent memory pool per `Device::Cpu(n)` to avoid cross-socket memory access latency.

## 6. Runtime Loading Strategy (Zero Build-Time GPU Dependencies)

### 6.1 Design Principle

The strided-rs binary must compile with **only a Rust toolchain**. All GPU libraries are loaded at runtime via `dlopen`. No nvcc, no CUDA Toolkit, no ROCm SDK required at build time.

### 6.2 CUDA Runtime Loading

`cudarc` already implements dlopen-based loading via `libloading`:

| Library | Shared Object | Purpose |
|---------|---------------|---------|
| CUDA Driver | `libcuda.so` | Device management, memory, kernel launch |
| cuBLAS | `libcublas.so` | GEMM (Zgemm for complex128) |
| NVRTC | `libnvrtc.so` | Runtime kernel compilation |

### 6.3 ROCm Runtime Loading

No existing Rust crate provides dlopen for HIP. A "hiparc" crate (analogous to cudarc) is needed.

| Library | Shared Object | Purpose |
|---------|---------------|---------|
| HIP Runtime | `libamdhip64.so` | Device management, memory, streams |
| hipBLAS | `libhipblas.so` | GEMM (Zgemm) |
| HIPRTC | `libhiprtc.so` | Runtime kernel compilation |

**Feasibility:** Proven by AMD Orochi (C++, 500+ functions via dlopen), Blender Cycles, TensorFlow-ROCm. HIP API is structurally parallel to CUDA (`cu*` → `hip*`), making the cudarc pattern directly replicable.

Default search path: `/opt/rocm/lib/` with soname version fallback (`.so.7` → `.so.6` → `.so.5` → `.so`).

**Note:** ROCm is Linux-only. Device validation before use is critical (ROCm can crash on unsupported GPU architectures).

### 6.4 Feature Flags (Compile-Time Selection)

```rust
// Cargo.toml — all GPU features are optional, pure Rust by default
[features]
default = ["cpu-faer"]
cpu-faer = ["faer"]
cpu-accelerate = ["accelerate-src"]   # Apple Silicon optimized
cuda = ["cudarc"]                     # dlopen, no link-time dependency
rocm = ["libloading"]                 # custom dlopen via libloading
```

## 7. GPU Kernel Strategy for Element-wise Operations

### 7.1 The Problem: Complex Element-wise on GPU

GEMM is served by cuBLAS/hipBLAS (`Zgemm`), but tensor network computation also requires element-wise complex operations (`exp`, `log`, `conj`, `sqrt`, etc.) on GPU buffers.

**No GPU framework provides a batched complex `exp()` library call** (unlike Intel MKL's `vzExp`). A kernel is always needed.

### 7.2 Available Complex Math Functions per Layer

| Library | Basic arithmetic | Transcendentals (exp, log, sin...) | Batched apply |
|---------|:---:|:---:|:---:|
| cuComplex.h / hipComplex | add, mul, div, conj, abs | **No** | No |
| thrust::complex / rocThrust | Full | **Full** (exp, log, sin, cos, sqrt, pow, ...) | via `thrust::transform` |
| cuda::std::complex (libcu++) | Full | **Full** | Manual kernel |
| cuBLAS / hipBLAS | Zscal, Zaxpy only | **No** | No |
| CubeCL | f32 only (no complex type) | f32 only | JIT kernel |

### 7.3 Option A: NVRTC / HIPRTC (Runtime Compilation of C/C++ Kernels)

Embed CUDA/HIP kernel source as Rust string literals. Compile at runtime via NVRTC/HIPRTC. No nvcc required.

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

- **Pros:** Works today, f64 fully supported, no external dependencies beyond CUDA/ROCm runtime.
- **Cons:** Kernel source in CUDA C strings (not Rust), CUDA and HIP kernels maintained separately (though nearly identical).
- **Note:** Cannot use `thrust::complex` headers in NVRTC (limited header access). Must implement complex math from real primitives (`exp`, `cos`, `sin` are CUDA built-ins).

### 7.4 Option B: CubeCL (Rust JIT to GPU)

Write kernels in Rust with `#[cube]` macro. CubeCL compiles to PTX/SPIR-V/MSL/WGSL at runtime. No nvcc, no external compiler.

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

- **Pros:** Single Rust source for all backends, type-safe, no string-based kernel management.
- **Cons:** f64 currently broken on CUDA/ROCm (see Section 7.5).

### 7.5 CubeCL f64 Status (Critical Blocker)

**f64 is disabled on both CUDA and ROCm backends in CubeCL.**

Timeline:
- 2024-10: f64 added in PR #207 by `@wingertge`.
- 2025-01: f64 commented out in PR #406 by the same author. Reason: `CUDA_ERROR_INVALID_VALUE` for matmul.
- Current: No tracking issue, no roadmap item. The commented-out line has stale type names (`gpu::Elem` instead of `gpu::ElemType`) — uncommenting alone would not compile.

**Root cause analysis:** Almost certainly shared memory overflow in matmul. CubeCL matmul uses tile-based shared memory sized for f32 (4 bytes). f64 (8 bytes) doubles the requirement, exceeding the GPU's per-SM limit (~48KB). The autotuner does not account for element size when selecting tile dimensions.

**Key insight: element-wise kernels (no shared memory) would likely work fine with f64.** The problem is matmul-specific, but f64 is disabled at the type registration level, blocking all operations.

**Feasibility of fix:**
1. Fix type name: `gpu::Elem::Float(...)` → `gpu::ElemType::Float(...)` (1 line)
2. Re-enable f64 in `register_supported_types` (1 line)
3. Adjust matmul tile config to halve dimensions for 8-byte types (few lines)
4. Or: exclude f64 from matmul while enabling it for element-wise

CI has no GPU hardware — f64 tests have never actually been executed.

### 7.6 CubeCL Complex Type Status

**CubeCL has no complex number type.** The IR `ElemType` enum has only `Float`, `Int`, `UInt`, `Bool`. No complex variant exists, no issues track it.

Burn (higher-level framework) had two community PRs for complex numbers (#3330, #3608), both closed as stale. The core team views it as requiring fundamental `Kind` redesign.

**Why PyTorch has complex but CubeCL/Burn don't:**
- PyTorch targets scientific computing + ML with a large team (Meta). Complex numbers added in v1.9 (2021).
- CubeCL/Burn target ML inference deployment. Complex numbers are low priority.
- PyTorch only needs CUDA C++ backend. CubeCL must generate code for 4 backends (CUDA, ROCm, Metal, WebGPU) — cost of adding a type is 4x.
- PyTorch's complex GEMM delegates to cuBLAS `Zgemm` anyway; element-wise uses `thrust::complex` internally.

### 7.7 Recommended Strategy

**Hybrid approach — use the right tool for each operation class:**

```
strided-rs (pure Rust, zero build-time GPU dependencies)
│
├── Contraction (GEMM)
│   ├── CPU: faer / Accelerate (existing)
│   ├── CUDA: cuBLAS Zgemm via dlopen (cudarc)
│   └── ROCm: hipBLAS Zgemm via dlopen (hiparc)
│
├── Element-wise (exp, log, conj, sqrt, ...)
│   ├── CPU: strided-kernel (existing)
│   └── GPU: CubeCL JIT kernels (interleaved f64 pairs)
│       └── Requires: CubeCL f64 fix (PR contribution)
│       └── Fallback: NVRTC/HIPRTC string kernels
│
└── Fallback
    └── CPU execution when GPU path unavailable
```

**Action plan:**
1. **Short term:** CPU for all operations (works today). cuBLAS/hipBLAS Zgemm via dlopen for GPU GEMM.
2. **Medium term:** Contribute f64 fix to CubeCL. Use CubeCL for GPU element-wise with interleaved f64 complex representation.
3. **Long term:** Explore contributing complex type to CubeCL/Burn if community interest grows.

## 8. Type Support Roadmap

### 8.1 Current State (strided-rs)

| Type | einsum API | strided-kernel | Notes |
|------|:---:|:---:|---|
| f64 | `EinsumOperand::F64` | Full support | Primary type |
| Complex64 (f64-based) | `EinsumOperand::C64` | Full support | Primary type |
| f32 | Not exposed | SIMD paths exist internally | Not in public API |
| f16 / bf16 | No | No | — |
| Complex32 (f32-based) | No | No | — |

### 8.2 Target Type Matrix

| Type | CPU | CUDA GEMM | ROCm GEMM | GPU element-wise | Priority |
|------|:---:|:---------:|:---------:|:----------------:|----------|
| **f64** | faer/Accelerate | cuBLAS `Dgemm` | hipBLAS `Dgemm` | CubeCL (after fix) | P0 (current) |
| **Complex128** | faer/Accelerate | cuBLAS `Zgemm` | hipBLAS `Zgemm` | CubeCL interleaved f64 | P0 (current) |
| **f32** | faer/Accelerate | cuBLAS `Sgemm` | hipBLAS `Sgemm` | CubeCL (works today) | P1 |
| **Complex64** | faer/Accelerate | cuBLAS `Cgemm` | hipBLAS `Cgemm` | CubeCL interleaved f32 | P1 |
| **f16 / bf16** | Software or Accelerate | cuBLAS `Hgemm` | hipBLAS `Hgemm` | CubeCL (works today) | P2 (ML integration) |

### 8.3 Complex Numbers on GPU Without Native Complex Type

All GPU backends (CubeCL, NVRTC, HIPRTC) lack native complex types. The universal approach is **interleaved real pairs**:

```
Complex128 buffer: [re₀, im₀, re₁, im₁, ...]  (f64 × 2N elements)
Complex64  buffer: [re₀, im₀, re₁, im₁, ...]  (f32 × 2N elements)
```

- **GEMM:** Delegated to BLAS libraries (cuBLAS/hipBLAS) which natively handle `cuDoubleComplex` / `hipDoubleComplex` (interleaved layout, ABI-compatible with `[f64; 2]`).
- **Element-wise:** Kernels index `buffer[2*i]` (real) and `buffer[2*i+1]` (imag), compute using real-valued built-in math functions.

## 9. Workspace Crate Structure

### 9.1 Overview

```
strided-rs/
├── strided-view/          # StridedArrayView types (zero-copy view abstraction)
├── strided-kernel/        # CPU cache-optimized map/reduce/broadcast kernels
├── strided-device/        # [NEW] Device abstraction, GPU ops, memory, BLAS dispatch
├── strided-einsum2/       # Binary einsum (GEMM backend)
├── strided-opteinsum/     # N-ary einsum with contraction tree optimization
├── strided-linalg/        # [NEW] SVD, QR, LU, Cholesky, eigendecomposition
├── ndarray-opteinsum/     # ndarray frontend for opteinsum
└── mdarray-opteinsum/     # mdarray frontend for opteinsum
```

### 9.2 Dependency Graph

```
strided-view
    ↑
    ├── strided-kernel          (CPU ops: map, reduce, broadcast)
    │
    └── strided-device          (device mgmt, GPU ops, BLAS/LAPACK dispatch)
         │  ├── device.rs       Device enum, runtime detection
         │  ├── stream.rs       Unified stream (CPU thread pool / GPU stream)
         │  ├── memory.rs       Memory pool, recycling, async safety
         │  ├── blas.rs         cuBLAS/hipBLAS Zgemm via dlopen
         │  ├── solver.rs       cuSOLVER/hipSOLVER via dlopen
         │  └── elementwise.rs  GPU element-wise kernels (CubeCL JIT)
         │
         ├── strided-einsum2    (binary einsum → blas.rs for GEMM)
         ├── strided-linalg     (SVD/QR → solver.rs for LAPACK)
         └── strided-opteinsum  (N-ary einsum → einsum2 + linalg)
```

### 9.3 strided-device (New Crate)

Central device abstraction layer. All GPU-related functionality lives here.

**Responsibilities:**
- `Device` enum and runtime GPU detection
- Unified `Stream` abstraction (Rayon thread pool / CUDA stream / HIP stream)
- GPU memory pool with slab allocation and async-safe recycling
- BLAS dispatch: cuBLAS/hipBLAS via dlopen (`cudarc` for CUDA, `libloading` for ROCm)
- LAPACK dispatch: cuSOLVER/hipSOLVER via dlopen
- GPU element-wise kernels: CubeCL JIT (interleaved f64 for complex) or NVRTC/HIPRTC fallback
- `TensorPromise` for async execution

**Why one crate:** GPU element-wise, BLAS, solver, memory, and streams are all tied to a `Device`. Splitting them would create circular dependencies or excessive coupling between crates.

```rust
// Cargo.toml for strided-device
[features]
default = []
cpu-faer = ["faer"]
cpu-accelerate = ["accelerate-src"]
cuda = ["cudarc"]
rocm = ["libloading"]
cubecl = ["cubecl-core", "cubecl-cuda"]  # optional, for GPU element-wise
```

### 9.4 strided-linalg (New Crate)

Dense linear algebra decompositions.

| Operation | CPU Backend | GPU Backend |
|-----------|-------------|-------------|
| SVD | faer / Accelerate LAPACK | cuSOLVER `Zgesvd` / hipSOLVER |
| QR | faer / Accelerate LAPACK | cuSOLVER `Zgeqrf` / hipSOLVER |
| LU | faer / Accelerate LAPACK | cuSOLVER `Zgetrf` / hipSOLVER |
| Cholesky | faer / Accelerate LAPACK | cuSOLVER `Zpotrf` / hipSOLVER |
| Eigendecomposition | faer / Accelerate LAPACK | cuSOLVER `Zheevd` / hipSOLVER |

Depends on `strided-device` for backend dispatch and `strided-view` for input/output types.

### 9.5 strided-kernel (Unchanged)

Remains CPU-only. Contains the cache-optimized map/reduce/broadcast implementation ported from Strided.jl. No GPU code, no renaming for now.

Future consideration: rename to `strided-cpu-ops` if the "kernel" naming causes confusion.

## 10. Full Stack Architecture (Layer 0–3)

### 10.1 Layer Overview

```
Layer 3: Applications (domain-specific)
  ├── Symmetry (U(1), SU(2) → block selection rules)
  ├── Named index / ITensor-like API
  ├── TN algorithms (DMRG, TEBD, TCI, ...)
  └── Physics-informed NN

Layer 2: tenet (tensor network computation framework)
  ├── tenet-ad             Automatic differentiation (Wirtinger-aware)
  ├── tenet-block          Block-sparse tensor (generic, symmetry-agnostic)
  └── tenet-burn           Burn interop (real tensor bridge)

Layer 1: strided-rs (device-aware operations)
  ├── strided-device       Device abstraction, GPU, memory, BLAS/LAPACK dispatch
  ├── strided-einsum2      Binary einsum (GEMM backend)
  ├── strided-opteinsum    N-ary einsum with contraction tree
  └── strided-linalg       SVD, QR, LU, Cholesky, eigen

Layer 0: strided-rs foundation
  ├── strided-view         StridedArrayView types (zero-copy)
  └── strided-kernel       CPU cache-optimized map/reduce/broadcast
```

Layer 2 uses the **tenet** namespace (separate from strided-rs). tenet provides the computational building blocks for tensor network applications without embedding domain-specific physics concepts (symmetry, named indices) — those belong in Layer 3.

### 10.2 Automatic Differentiation Strategy

#### The Problem: Burn's Autodiff Cannot Handle Complex Numbers

Burn's `TensorKind` has only `Float`, `Int`, `Bool` — no `Complex`. Complex autodiff requires **Wirtinger derivatives** (`∂f/∂z` and `∂f/∂z̄`), which Burn's real-valued chain rule does not implement. Operations like `conj`, `abs`, `real`, `imag` would produce incorrect gradients.

#### Short Term: tenet-ad (Self-contained AD)

Build a focused AD system for tensor network operations. Prior art: **ndtensors-rs** (proven approach).

**Scope:** Only differentiate operations that tensor networks actually use:

| Operation | Forward | Backward (known formula) |
|-----------|---------|--------------------------|
| einsum | strided-opteinsum | Transpose subscripts + einsum |
| SVD | strided-linalg | Seeger et al. 2017 |
| QR | strided-linalg | Known formula |
| exp(z) | element-wise | exp(z) * grad |
| Element-wise (+, *, conj) | strided-kernel | Standard + Wirtinger rules |

**Design:**

```rust
// tenet-ad (Layer 2)
pub struct TrackedTensor<T> {
    data: StridedArray<T>,      // actual data (f64 or Complex64)
    node: Option<NodeRef>,       // computation graph node
}

// Wirtinger-aware backward for complex operations
// For f: C → C, the chain rule uses both ∂f/∂z and ∂f/∂z̄
pub trait WirtingerBackward {
    fn backward_holomorphic(&self, grad: &Tensor) -> Tensor;   // ∂f/∂z
    fn backward_conjugate(&self, grad: &Tensor) -> Tensor;     // ∂f/∂z̄
}
```

**Advantages:**
- Complex128 support from day one
- Wirtinger derivatives built-in
- Small, focused scope (einsum + linalg + element-wise)
- Proven by ndtensors-rs

#### Long Term: Extend Burn with Complex Kind

Contribute `Complex` as a new `TensorKind` to Burn, enabling Burn's autodiff for complex tensors. This would allow mixed NN + TN computation in a single autodiff graph.

**Prerequisites (upstream Burn contributions):**
1. Add `Complex` to `TensorKind` enum (with `ComplexTensorOps` trait)
2. Add Wirtinger derivative support to `burn-autodiff`
3. CubeCL complex type or interleaved-f64 lowering
4. At least one backend implementation (e.g., `burn-ndarray`)

**Previous attempts:** Burn PRs #3330 and #3608 both went stale. Core team considers it a fundamental `Kind` redesign. Community interest exists (FFT, quantum computing) but no champion.

**When Burn gets Complex support:**

```rust
// Long-term vision: unified autodiff graph
let nn_output: Tensor<Autodiff<B>, 2, Float> = model.forward(input);

// Convert real → complex
let tn_input: Tensor<Autodiff<B>, 2, Complex> = nn_output.to_complex();

// TN operations within Burn's autodiff graph
let tn_result = einsum_burn("ij,jk->ik", &tn_input, &mps_tensor);

// Gradient flows through both NN and TN
let grads = tn_result.backward();
```

#### Burn Interop During Short-Term Phase

Even before Burn gets Complex support, **real-valued tensors** can be exchanged:

```
Burn Autodiff (real only)     tenet-ad (complex + real)
     │                              │
     │  real Tensor ←→ real Tensor  │
     │  via TensorData (CPU copy)   │
     │                              │
     │  Custom Backward Op:         │
     │  register einsum/SVD with    │
     │  Burn's graph for real parts │
```

- NN layers: Use Burn's autodiff (mature, GPU-accelerated)
- TN operations on complex tensors: Use tenet-ad
- Interface: Exchange real-valued tensors via `TensorData`
- For real-valued TN ops: Register as Burn Custom Backward Ops for seamless gradient flow

### 10.3 Block Tensor (Generic, Symmetry-Agnostic)

`tenet-block` provides a generic block-sparse tensor data structure. It has no knowledge of symmetry groups — block selection logic is the caller's responsibility (Layer 3).

```rust
// tenet-block (Layer 2): generic block-sparse tensor
pub struct BlockTensor<T> {
    blocks: HashMap<BlockIndex, StridedArray<T>>,  // each block is dense
    structure: BlockStructure,                       // which blocks exist
}

// Block-level einsum — caller provides the contraction pairs
fn block_einsum<T>(
    subscripts: &str,
    a: &BlockTensor<T>,
    b: &BlockTensor<T>,
    pairs: &[(BlockIndex, BlockIndex, BlockIndex)],  // (a_block, b_block, out_block)
) -> BlockTensor<T> {
    for &(ai, bi, ci) in pairs {
        einsum2_into(subscripts, &a.blocks[&ai], &b.blocks[&bi], &mut output[ci]);
    }
    output
}
```

Symmetry-aware block selection is implemented at Layer 3:

```rust
// Layer 3: physics-specific symmetry logic
fn u1_contraction_pairs(
    a: &U1Structure,
    b: &U1Structure,
) -> Vec<(BlockIndex, BlockIndex, BlockIndex)> {
    // Enumerate non-zero block pairs from quantum number conservation
}

// Usage: Layer 3 drives Layer 2
let pairs = u1_contraction_pairs(&a_structure, &b_structure);
let result = block_einsum("ij,jk->ik", &a, &b, &pairs);
```

## 11. Execution Pipeline

1. **Definition**: User calls `einsum(..., &device, &stream)`.
2. **Planning**: Select the appropriate kernel backend based on device:
   - `Device::Cpu(_)` → faer or Accelerate
   - `Device::Cuda(_)` → cuBLAS Zgemm (dlopen via cudarc)
   - `Device::Rocm(_)` → hipBLAS Zgemm (dlopen via hiparc)
3. **Allocation**: Acquire an output buffer from the memory pool via recycling (`strided-device` memory pool).
4. **Dispatch**: Submit the operation to the stream and immediately return a `TensorPromise`.
5. **Completion**: Once computation finishes, the buffer becomes "available" for subsequent operations or data retrieval back to the CPU.

---

## 12. Open Design Questions

- How to represent device placement in the type system (type-level vs. runtime enum)?
- Ownership model for cross-device transfers (copy semantics vs. move semantics)?
- Error propagation strategy for async failures surfaced at `.wait()` time.
- Whether `TensorPromise` should implement `Future` for `async`/`await` interop.
- Should Accelerate be auto-selected on macOS, or require explicit feature flag?
- CubeCL f64 fix: contribute upstream or maintain a fork?
- Kernel caching strategy for NVRTC/HIPRTC compiled kernels (compile once, reuse across calls).
- ROCm device validation strategy to avoid crashes on unsupported architectures.
- Should `strided-kernel` be renamed to `strided-cpu-ops` to avoid ambiguity with GPU "kernels"?
- How should `strided-linalg` expose truncated SVD (rank-k approximation)?

## 13. External Dependencies & Upstream Contributions

| Dependency | Role | Status | Action Needed |
|---|---|---|---|
| `cudarc` | CUDA dlopen | Stable, maintained | Use as-is |
| `libloading` | Generic dlopen | Stable | Use for ROCm (hiparc) |
| `CubeCL` | GPU JIT kernels | f64 broken (CUDA/ROCm) | **Contribute f64 fix PR** |
| `Burn` | NN framework + autodiff | No complex type support | Long-term: contribute Complex Kind |
| `faer` | CPU GEMM | Stable | Use as-is |
| `accelerate-src` | Apple BLAS | Stable | Use for macOS |

---
*Origin: Gemini brainstorm - Feb 2026*
*v1.1: Excluded Metal GPU; added Accelerate; added backend matrix*
*v1.2: Added runtime loading strategy; GPU element-wise kernel analysis; CubeCL f64/complex status; type support roadmap; hybrid GEMM+CubeCL architecture*
*v1.3: Added workspace crate structure (strided-device, strided-linalg); dependency graph; crate responsibilities*
*v1.4: Added full stack architecture (Layer 0-3); AD strategy (short-term: self-contained Wirtinger AD, long-term: Burn Complex Kind); Burn interop analysis; block tensor design*
*v1.5: Renamed Layer 2 to "tenet" namespace; separated symmetry (Layer 3) from generic block tensor (Layer 2); removed named index from Layer 2*
