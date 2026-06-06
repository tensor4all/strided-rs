# faer-Inspired Kernel Writing Guide

This note summarizes high-performance Rust kernel patterns observed in a
local clone of `faer-rs` and translates them into practical rules for
`strided-rs`.

Reference clone:

- Repository: `https://github.com/sarah-ek/faer-rs.git`
- Commit: `618b567f508604ae08f99460a9404a1a8b2b75ea`
- Local path used for this analysis: `../faer-rs`

The goal is not to copy faer's matrix multiplication machinery. The goal is to
write `strided-kernel` elementwise, reduction, broadcast, and binary-einsum
backend code in a style that keeps bounds checks, shape checks, allocation, and
dispatch overhead out of the inner loop.

## Relevant faer Patterns

### `zip!` as a TensorIterator-like abstraction

faer's public coefficient-wise API is centered on `zip!`, which is close in
spirit to PyTorch's TensorIterator:

- `faer/src/lib.rs:288` documents `zip!` for same-shape coefficient-wise
  operations.
- `faer/src/lib.rs:291` explicitly leaves traversal order unspecified.
- `faer/src/linalg/zip.rs:441` checks shape once when creating a zip.
- `faer/src/linalg/zip.rs:1396` picks a preferred layout before iteration.
- `faer/src/linalg/zip.rs:1413` chooses the contiguous slice path once, or a
  raw `get_unchecked` path for non-contiguous traversal.

Rule for `strided-rs`: the public API should specify shape and semantics, not
physical traversal order. The kernel planner should be free to reorder, fuse,
or reverse dimensions before the hot loop, as long as the logical result is the
same.

### Runtime SIMD dispatch, compile-time kernel body

faer uses `pulp::WithSimd` so runtime ISA selection happens outside the hot
loop, while the loop body is monomorphized over `S: pulp::Simd`.

Key references:

- `faer-traits/src/lib.rs:1146` defines `SimdArch::dispatch`.
- `faer/src/lib.rs:201` defines the `dispatch!` macro.
- `faer/src/linalg/reductions/sum.rs:10` implements `pulp::WithSimd` for a
  concrete reduction body.
- `faer-traits/src/lib.rs:1247` exposes `SIMD_CAPABILITIES` and SIMD associated
  vector/mask types through `ComplexField`.

Rule for `strided-rs`: do feature and type dispatch once before entering the
element loop. Do not branch per element on scalar type, SIMD availability,
conjugation, broadcasting, or layout class.

### Head/body/tail instead of scalar remainder loops

faer's `SimdCtx` partitions a contiguous vector into masked head, full SIMD
body, and masked tail:

- `faer/src/utils/simd.rs:6` stores `head_end`, `body_end`, `tail_end`, and
  masks.
- `faer/src/utils/simd.rs:59` uses normal SIMD load/store for body lanes.
- `faer/src/utils/simd.rs:90` and `faer/src/utils/simd.rs:123` use masked
  load/store for head and tail lanes.
- `faer/src/utils/simd.rs:388` returns typed head/body/tail indices.
- `faer-traits/src/lib.rs:806` wraps masked load/store.

Rule for `strided-rs`: for fixed operations on contiguous inner loops, prefer a
single vectorized path with masked head/tail over a vector body followed by a
scalar cleanup loop. This matters for short and medium inner blocks because
remainder handling can dominate.

### Alignment is a loop-plan property

`SimdCtx::new_align` computes the head/body/tail split from pointer alignment:

- `faer/src/utils/simd.rs:201` computes `align_offset`.
- `faer/src/utils/simd.rs:216` shifts the head so the body is aligned.
- `faer/src/utils/simd.rs:249` and `faer/src/utils/simd.rs:254` construct
  memory masks for partial head/tail loads.
- `faer/src/lib.rs:1036` defines `simd_align(i) = i.wrapping_neg()`.

Rule for `strided-rs`: if a specialized contiguous kernel uses explicit SIMD,
calculate alignment once from the actual inner-block pointer. The hot loop
should only consume the resulting plan.

### Multiple accumulators for reductions

faer avoids loop-carried dependency chains in reductions by using several SIMD
accumulators:

- `faer/src/linalg/reductions/sum.rs:17` allocates four SIMD accumulators.
- `faer/src/linalg/reductions/sum.rs:18` iterates with `batch_indices(); 4`.
- `faer/src/linalg/reductions/sum.rs:22` combines accumulators as a tree.
- `faer/src/linalg/reductions/norm_l2_sqr.rs:20` applies the same pattern to
  `abs2` accumulation.

Rule for `strided-rs`: fixed reductions such as `sum`, `dot`, norm-like
reductions, and inner-product kernels should use independent accumulators.
Do not expect LLVM to vectorize a generic closure-based reduction with a single
scalar accumulator.

### Compile-time unrolling

faer's `simd_iter!` macro expands a batched SIMD loop with a compile-time batch
index:

- `faer/src/utils/simd.rs:433` builds arrays of `SimdBody` indices.
- `faer/src/lib.rs:1284` defines `simd_iter!`.
- `faer/src/lib.rs:1309` makes each batch index a `const`.

Rule for `strided-rs`: use small fixed unroll factors for reductions and
regular pointwise kernels. Avoid dynamic indexing into accumulator arrays in
the hot loop unless the optimizer can see the index as a compile-time constant.

### FMA is explicit

faer exposes fused multiply-add through the scalar trait layer:

- `faer-traits/src/lib.rs:550` wraps `simd_mul_add`.
- `faer-traits/src/lib.rs:565` wraps `simd_conj_mul_add`.
- `faer-traits/src/lib.rs:582` uses a const-generic conjugation branch for
  `maybe_conj_mul_add`.
- `faer-traits/src/lib.rs:604` wraps `abs2_add`.

Rule for `strided-rs`: fixed operations such as `axpy`, `fma`, `dot`, and GEMM
fallbacks should call an explicit FMA-capable path where available. Do not
write these kernels only as generic closures if the operation is known.

### Complex SIMD uses native complex lanes

faer has two complex SIMD layers:

- The generic `num_complex::Complex<T>` implementation composes real SIMD lanes
  as `Complex<T::SimdVec<S>>`.
- Native `c32` and `c64` kernels dispatch through `ComplexImpl<f32>` or
  `ComplexImpl<f64>` and use `pulp::Simd` native complex lanes (`S::c32s`,
  `S::c64s`).

The native path calls operations such as `mul_e_c32s`, `mul_e_c64s`,
`conj_mul_e_c32s`, and `conj_mul_e_c64s` directly. For `strided-kernel`
elementwise complex multiplication, prefer the same direct `pulp::Simd`
operations at the fixed-operation TypeId dispatch boundary. Do not add a broad
trait abstraction unless multiple complex kernels need the same dispatch
surface.

## Bounds-Check Avoidance

faer's pattern is not "use unsafe everywhere." It is:

1. Validate shape, layout, and aliasing before the loop.
2. Convert the validated region to a slice or pointer-based representation.
3. Run the hot loop without per-element validation.

Examples:

- `faer/src/linalg/zip.rs:377` defines `get_slice_unchecked`.
- `faer/src/linalg/zip.rs:384` defines `get_unchecked`.
- `faer/src/linalg/zip.rs:646` builds a mutable slice with
  `core::slice::from_raw_parts_mut`.
- `faer/src/linalg/zip.rs:672` advances through a contiguous slice with
  `split_first_mut().unwrap_unchecked()`.
- `faer/src/linalg/zip.rs:1413` chooses the contiguous slice path once and then
  loops over unchecked slice elements.
- `faer/src/mat/matref.rs:391` exposes an in-bounds pointer helper with explicit
  safety requirements.
- `faer/src/mat/matmut.rs:120` documents mutable view safety invariants,
  including no aliasing and no two elements sharing the same address.

Rule for `strided-rs`: expose safe public APIs, but inside validated kernels,
move error handling and range checks outside the inner loop. The inner loop
should be pointer increments, slice indexing that LLVM can eliminate, or
unchecked slice progression.

For mutable outputs, the planner must prove that every logical output element
maps to a unique address. A stride-0 mutable destination is not a valid target
for a pointwise write kernel.

## Current strided-rs Baseline

`strided-rs` already uses part of this style:

- `strided-kernel/src/map_view.rs:57` specializes binary inner loops by stride
  pattern.
- `strided-kernel/src/map_view.rs:78` handles `dst stride = 1`, one input
  contiguous, one input broadcast (`stride = 0`).
- `strided-kernel/src/map_view.rs:87` handles the symmetric broadcast case.
- `strided-kernel/src/simd.rs:63` implements f32 SIMD `sum`/`dot`.
- `strided-kernel/src/simd.rs:148` implements f64 SIMD `sum`/`dot`.
- `strided-kernel/src/simd.rs:73` and `strided-kernel/src/simd.rs:118` already
  use four accumulators.

The remaining gap versus faer is mostly:

- `strided-kernel/src/simd.rs:71` and `strided-kernel/src/simd.rs:113` use
  `as_simd_*` plus scalar tail loops, not masked head/body/tail.
- Alignment is not treated as an inner-block property.
- Fixed operations such as `add`, `mul`, `axpy`, and `fma` still mostly rely on
  generic closure loops.
- Generic reductions still have loop-carried scalar accumulators.

## Direct Application to strided-kernel

### Broadcasted binary pointwise operations

PyTorch's `aten::mul` performance comes from a TensorIterator-like path:
broadcasted inputs have stride `0`, output can be non-contiguous but dense, and
the inner loop is specialized by stride pattern.

`strided-rs` already has the necessary metadata operation:

- `strided-view/src/view.rs:300` implements `StridedView::broadcast` by setting
  expanded axes to stride `0`.

faer uses the same representation for scalar repetition:

- `faer/src/mat/matref.rs:124` creates a `1 x 1` view over a reference.
- `faer/src/mat/matref.rs:134` creates repeated scalar views with row and
  column stride `0`.

The missing abstraction is a public API that performs TensorIterator-like shape
promotion and then calls the existing stride-specialized machinery:

```rust
pub fn zip_map2_broadcast_into<D, A, B, OpA, OpB>(
    dest: &mut StridedViewMut<D>,
    a: &StridedView<A, OpA>,
    b: &StridedView<B, OpB>,
    f: impl Fn(A, B) -> D + MaybeSync,
) -> Result<()>
```

For `mul`, add a fixed operation wrapper:

```rust
pub fn broadcast_mul_into<T, OpA, OpB>(
    dest: &mut StridedViewMut<T>,
    a: &StridedView<T, OpA>,
    b: &StridedView<T, OpB>,
) -> Result<()>
```

Rule: keep shape promotion and `broadcast()` calls outside the timed inner
loop. Once views are aligned, dispatch to a stride-specialized kernel.

### `zip_map2_into`

Current `zip_map2_into` already has good first-order structure:

- `strided-kernel/src/map_view.rs:357` has a fully sequential contiguous fast
  path.
- `strided-kernel/src/map_view.rs:376` skips full planning for small tensors.
- `strided-kernel/src/map_view.rs:428` enters block iteration only after the
  plan is built.

The next improvements should be:

- Add fixed-operation kernels for `copy`, `mul`, `add`, `axpy`, `fma`, and
  `dot` instead of relying only on closure-based `zip_map2_into`.
- Preserve the current generic `zip_map2_into` as the fallback for arbitrary
  closures.
- For known f32/f64 contiguous or stride-0 broadcast cases, use explicit SIMD
  loops instead of hoping the closure path auto-vectorizes.

### Batched outer product

`batched_outer_product_into` is now a semantic wrapper around
`broadcast_mul_into`. It validates the outer-product dimension grouping, builds
the corresponding axis maps, and lets the generic broadcasted `mul` planner
select the actual kernel.

This avoids two competing implementations for the same operation. Add a
specialized batched outer-product kernel only if the generic
TensorIterator-like path leaves a measured gap that cannot be closed in the
shared pointwise planner.

## Checklist for New Hot Kernels

Before adding a new optimized kernel:

- Validate shapes, ranks, strides, and output size once.
- Decide layout class once: contiguous, dst-contiguous with broadcast input,
  compact grouped, or fully strided.
- Allocate any offset vectors, workspace, or output storage outside the timed
  loop.
- Make broadcast explicit as stride `0`, not as repeated indexing logic.
- Use a fixed-operation path for known operations; use closures only for the
  generic fallback.
- Keep scalar type dispatch outside the hot loop.
- Use pointer increments or slices in the inner loop; avoid `Result`, `Option`,
  `Vec`, `HashMap`, `SmallVec` mutation, or shape lookup per element.
- For reductions, use multiple accumulators and a tree combine.
- For FMA-like operations, use explicit FMA when possible.
- Add a benchmark that isolates the kernel from tensor-wrapper overhead.

## What Not To Do

- Do not add one-off fast paths in tenferro for each benchmark shape.
- Do not put layout detection inside the per-element loop.
- Do not benchmark a kernel through the full tenferro eager/traced stack first.
  Benchmark it directly in `strided-kernel` or `strided-einsum2`, then measure
  tenferro overhead separately.
- Do not treat `unsafe` as the optimization. The optimization is proving the
  invariants once, then expressing the loop in a form LLVM can compile well.

## Implication for Binary Einsum Backend

The intended dependency direction is:

```text
tenferro CPU eager/traced
  -> strided-einsum2 dot_general/tensordot/binary-einsum API
    -> strided-kernel broadcasted pointwise, outer product, reduction, copy
       kernels
```

`strided-einsum2` should expose axis-based `tensordot_into` and
`dot_general_into` APIs so tenferro can call them without label parsing or
contraction-plan allocation. For `sum_dims.is_empty()`, `strided-einsum2`
should lower to a broadcasted pointwise kernel. For true contractions, it
should keep using the existing GEMM backend path.
