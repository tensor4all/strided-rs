# strided-rs: Unified Tensor Backend Design Specification
**Version:** 1.0 (Feb 2026)
**Scope:** Tensor4all Layer 1 Integration (CPU/NVIDIA/AMD)

---

## 1. Executive Summary

This document specifies the design for a multi-device (CPU, CUDA, ROCm) tensor computation library built on top of `strided-rs`. The goal is to combine the ergonomic "View" abstraction for mathematical operations with asynchronous execution and advanced memory management.

## 2. Core Components

### 2.1 StridedArray & StridedView

- **StridedArray (Owner)**: The entity that owns the physical buffer. Wraps an `Arc<Storage>` for reference-counted lifetime management.
- **StridedView (Accessor)**: A non-owning reference holding only metadata (`shape`, `strides`, `offset`).
- **Disjoint Slicing**:
    - Safely partition a large buffer into multiple non-overlapping `StridedView`s.
    - This enables concurrent writes to a single `StridedArray` from multiple GPU/CPU streams while maintaining Rust's safety guarantees (`Send`/`Sync`).

### 2.2 Unified Stream

- **Concept**: Abstract all execution contexts (CPU and GPU) as "FIFO task queues".
- **CPU Stream**: Implemented as task submission to a thread pool (e.g., Rayon).
- **GPU Stream**: Maps to CUDA Stream or ROCm HIP Stream.

## 3. Asynchronous Execution and the Promise Pattern

### 3.1 TensorPromise

- **Role**: A "reservation" object returned immediately by operations such as `einsum` or `SVD`.
- **States**: `Pending` (computation in progress) or `Ready` (completed).
- **Automatic synchronization logic**:
    - When an operation function receives a `TensorPromise` as an argument, if it was produced on a different device/stream than the current one, the runtime automatically inserts a `WaitEvent` internally to resolve the dependency.

### 3.2 Synchronization Method (`.wait()`)

- Called explicitly only when the user needs to read actual data on the CPU side.
- Blocks the host CPU thread and waits for device-side computation to complete.

## 4. Memory Management Strategy

### 4.1 Smart Memory Pool

- **Algorithm**: "Larger-serves-smaller" reuse strategy (Slab Allocation).
- **Recycling**: When a `StridedArray` is dropped, its buffer is returned to the pool rather than being immediately deallocated.
- **Async safety**: Buffers remain locked against reallocation until their associated `Event` has completed.

### 4.2 NUMA Affinity (CPU-Specific)

- In multi-CPU environments, maintain an independent memory pool per `Device::Cpu(n)` to avoid cross-socket memory access latency.

## 5. Execution Pipeline (Execution Flow)

1. **Definition**: User calls `einsum(..., &device, &stream)`.
2. **Planning**: Select the appropriate kernel backend (cuTENSOR / hiptensor / faer) based on the target device.
3. **Allocation**: Acquire an output buffer from the memory pool via recycling.
4. **Dispatch**: Submit the operation to the stream and immediately return a `TensorPromise`.
5. **Completion**: Once computation finishes, the buffer becomes "available" for subsequent operations or data retrieval back to the CPU.

---

## Open Design Questions

- How to represent device placement in the type system (type-level vs. runtime enum)?
- Ownership model for cross-device transfers (copy semantics vs. move semantics)?
- Error propagation strategy for async failures surfaced at `.wait()` time.
- Integration boundary between `strided-kernel` (existing CPU kernels) and the new stream/device abstraction.
- Whether `TensorPromise` should implement `Future` for `async`/`await` interop.

---
*Origin: Gemini brainstorm - Feb 2026*
*Translated and formatted for strided-rs planning docs*
