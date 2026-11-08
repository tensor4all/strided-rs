# faer SIMD Design Analysis & strided-kernel Optimization Plan

> **Historical design record.** Written 2026-01-30 (the commit date of this
file); the measurements below are the ones it recorded then, on one Apple
Silicon M2 core as stated under "Performance Gap Analysis". Nothing here has
been re-measured since. The code has moved on: the typed kernels this document
targets now live in `strided-basic`, and most of the plan below has shipped.
Read "Current Status" before acting on any gap or priority.

Analysis of SIMD optimization techniques used in `extern/faer`, and a concrete plan for applying them to `strided-kernel`.

## Current Status

**Shipped since this document was written:**

- **faer GEMM backend** (PR #19), **zero-copy conjugation** (PR #21) and the
  crate renames `stridedview` → `strided-view`, `strided` → `strided-kernel`
  (PR #22). The `strided-einsum2` crate that used the faer backend here was
  removed from the workspace after 0.4.4; its release remains on crates.io.
- The typed kernels and plans moved to `strided-basic`; `strided-kernel` owns
  the dtype-erased entries. Paths below that name `strided-kernel/src/` now
  resolve to `strided-basic/src/`.
- **`pulp` runtime dispatch on the map, reduce and update inner loops**
  (`simd::dispatch_if_large`, `strided-basic/src/map_view.rs`,
  `reduce_view.rs`, `ops_view.rs`, `update_view.rs`), which is most of
  Priority 3. `reduce_axis` / `reduce_axis_into` and the plan loops still run
  undispatchsed loops; the exceptions are listed under Priority 3.
- **Hand-written `pulp` SIMD leaves** (`strided-basic/src/simd.rs`): sum,
  product, dot, sum-of-squares and the add/mul block folds for `f32`/`f64`.
  This covers Priority 1 and the block-fold part of Priority 4; `add_acc`, `axpy_acc`
  and `fma_acc` still run dispatched scalar loop bodies.
- **Contiguous identity `memcpy` fast path** in `copy_into`
  (`strided-basic/src/ops_view.rs`), closing Gap 4.
- **The per-element `Option` accumulator this document criticizes is gone**
  from `strided-basic/src/reduce_view.rs` (Priority 5).
- **Inline metadata** (`InlineVec`, `OperandVec`) replaces the per-call `Vec`
  plan allocations of Priority 6.

**Not adopted:** pairwise summation (Priority 7). The shipped reductions use a
fixed multi-accumulator lane count with a documented merge order and
association instead; see the reduction-order contract in the `strided-kernel`
README.

**Remaining:** explicit SIMD specialization of the map/update loop bodies
(`map_into`, `add_acc`, `axpy_acc`, `fma_acc`). They already reach `pulp` runtime dispatch,
so what is left is the vectorized body itself, which is still LLVM's
auto-vectorization inside the dispatched ISA.

## SIMD Abstraction Architecture

faer builds a 3-layer SIMD abstraction on top of the `pulp` crate.

### Layer 1: `pulp::Simd` trait + runtime dispatch

`pulp::Arch::default().dispatch(impl WithSimd)` detects the best SIMD ISA at runtime (SSE, AVX2, AVX-512, NEON, etc.). A struct implementing the `WithSimd` trait is passed in, and the concrete SIMD instruction set is monomorphized through the type parameter `S: Simd`. Separate binaries are generated per ISA, and the runtime jumps to the optimal one.

### Layer 2: `ComplexField` trait's SIMD associated types

Defined in `faer-traits/src/lib.rs:1246-1260`:

```rust
type SimdCtx<S: Simd>: Copy;     // SIMD context
type SimdVec<S: Simd>: Pod;       // SIMD vector type (e.g. f32x8)
type SimdMask<S: Simd>: Copy;     // logical mask
type SimdMemMask<S: Simd>: Copy;  // memory mask for masked load/store
```

For f32 the implementations delegate to `ctx.add_f32s()`, for f64 to `ctx.add_f64s()`, etc. Complex types decompose into real/imaginary parts (SoA-style).

### Layer 3: `SimdCtx<'N, T, S>` (faer side)

Defined in `faer/src/utils/simd.rs:6-17`. Partitions a length-`N` array into head/body/tail regions. Compile-time `SimdCapabilities` enum distinguishes three tiers: None, Copy, Simd.

## Key Optimization Patterns

### 1. Head / Body / Tail masked iteration

`SimdCtx` splits every array into three regions:

```
|-- head (masked) --|-- body (full SIMD width) --|-- tail (masked) --|
```

- **Head**: alignment adjustment. `new_align()` accounts for the pointer's alignment offset, processing the leading partial vector via masked load.
- **Body**: full-width SIMD vectors with regular load/store.
- **Tail**: trailing partial vector handled with masked load/store.

Type-safe index types enforce correct access:
- `SimdBody` → regular `load` / `store`
- `SimdHead` → `mask_load` / `mask_store`
- `SimdTail` → `mask_load` / `mask_store`

**Benefit**: eliminates scalar remainder loops entirely. The masked operations keep everything in SIMD registers.

### 2. Multiple accumulators for ILP

All reduction kernels use **4 independent accumulators** (see `faer/src/linalg/reductions/sum.rs:17-25`):

```rust
let mut acc = [simd.zero(); 4];
simd_iter!(for (IDX, i) in [simd.batch_indices(); 4] {
    let x = simd.read(data, i);
    acc[IDX] = simd.add(acc[IDX], x);
});
let acc0 = simd.add(acc[0], acc[1]);
let acc2 = simd.add(acc[2], acc[3]);
let acc0 = simd.add(acc0, acc2);
simd.reduce_sum(acc0)
```

- `batch_indices()` unrolls the body loop by BATCH=4.
- Each accumulator is independent, saturating the CPU's execution pipeline (Instruction-Level Parallelism).
- After the loop, a tree-reduce combines results.

### 3. Pairwise summation for numerical precision

Large reductions use **pairwise summation** (see `faer/src/linalg/reductions/sum.rs:30-43`):

```rust
fn sum_simd_pairwise_rows<T>(data: ColRef<'_, T, usize, ContiguousFwd>) -> T {
    if data.nrows() <= LINEAR_IMPL_THRESHOLD {  // 128
        sum_simd(data)  // SIMD kernel
    } else {
        let split_point = ((data.nrows() + 1) / 2).next_power_of_two();
        let (head, tail) = data.split_at_row(split_point);
        sum_simd_pairwise_rows(head) + sum_simd_pairwise_rows(tail)
    }
}
```

- Arrays of 128 elements or fewer are processed directly with SIMD; larger arrays are recursively split in half.
- Floating-point rounding error improves from O(n) to O(log n).

### 4. Complex-to-real memory reinterpretation

For operations like L2 norm on `Complex<f64>`, faer reinterprets the memory as `f64` (see `faer/src/linalg/reductions/norm_l2_sqr.rs:74-117`):

```rust
if const { T::IS_NATIVE_C64 } {
    let mat = unsafe {
        MatRef::<f64>::from_raw_parts(
            mat.as_ptr() as *const f64,
            2 * mat.nrows(),  // flatten re,im
            mat.ncols(),
            ContiguousFwd,
            mat.col_stride().wrapping_mul(2),
        )
    };
    return norm_l2_sqr_simd_pairwise_cols::<f64>(mat);
}
```

Since `|z|^2 = re^2 + im^2`, a complex array can be treated as a flat real array for sum-of-squares. This avoids writing a separate complex kernel and reuses the optimized real-valued path.

### 5. `dispatch!` macro: compile-time complex branching

Defined in `faer/src/lib.rs:202-222`:

```rust
macro_rules! dispatch {
    ($imp:expr, $ty:ident, $T:ty) => {
        if const { <$T>::IS_NATIVE_C32 } {
            <ComplexImpl<f32>>::Arch::default().dispatch(transmute($imp))
        } else if const { <$T>::IS_NATIVE_C64 } {
            <ComplexImpl<f64>>::Arch::default().dispatch(transmute($imp))
        } else {
            <$T>::Arch::default().dispatch($imp)
        }
    };
}
```

- `if const { ... }` eliminates dead branches at compile time.
- Complex types use an internal `ComplexImpl` representation for better optimization.

### 6. `simd_iter!` macro: compile-time unrolling

Defined in `faer/src/lib.rs:1288-1328`. The invocation `for (IDX, i) in [simd.batch_indices(); 4]` generates a 4-way unrolled loop that processes elements in order: head → batched body → remainder body → tail.

- `const $batch_id: usize = N` makes the index a compile-time constant, so `acc[IDX]` compiles to direct register access.
- Supports BATCH sizes 1 through 8.

### 7. Matmul microkernel with register tiling

Defined in `faer/src/linalg/matmul/mod.rs:58-206`:

- **MR_DIV_N × NR microkernel**: processes MR (a multiple of SIMD lane count) rows by NR columns at a time.
- **Register tiling**: `local_acc = [[simd.zero(); MR_DIV_N]; NR]` keeps all accumulators in registers.
- **Cache blocking**: NC=2048, KC=128 constants tile the computation to fit cache hierarchy.
- **`new_force_mask`**: forces the last tile into mask mode so it uses body+tail only (no head), simplifying the microkernel.
- **FMA**: `simd.mul_add(b, a[i], local_acc)` fuses multiply-add into a single instruction.

### 8. Alignment optimization

`SimdCtx::new_align()` (defined in `faer/src/utils/simd.rs:200-305`) computes head/body/tail boundaries from the pointer's alignment offset. This ensures body-region accesses are naturally aligned, avoiding cache-line split penalties. Particularly important for AVX-512 (64-byte alignment).

### 9. `SimdCapabilities` three-tier fallback

```rust
pub enum SimdCapabilities {
    None,  // scalar only
    Copy,  // memcpy possible (Pod types)
    Simd,  // full SIMD support
}
```

`if const { T::SIMD_CAPABILITIES.is_simd() }` branches at compile time. Custom numeric types (e.g. Quad) fall back to `None` and use scalar loops.

## Applicability to strided-rs

| faer pattern | strided-rs application | Priority |
|---|---|---|
| Head/body/tail + masked load/store | Contiguous innermost loop remainder handling | High |
| Multiple accumulators (4×) | ILP in `reduce` / `sum` kernels | High |
| `pulp` runtime dispatch | Use AVX2/AVX-512 without `target-cpu=native` | High |
| FMA via `simd.mul_add` | `axpy_acc`, `fma_acc`, `dot` kernels | High |
| Pairwise summation | Precision for large reductions | Medium |
| Complex→real reinterpretation | When adding `Complex` support | Low |
| Register tiling (MR×NR) | Not directly applicable (strided-rs is not matmul-focused) | Low |

---

## Performance Gap Analysis: strided-kernel vs Strided.jl

Benchmark environment: Apple Silicon M2, single-threaded. These numbers were
recorded when this document was written, on the pre-split `strided-kernel`
crate; no raw log was kept and nothing has been re-measured since, so treat them
as reference values rather than a reproducible measurement. The committed
regression benchmarks live in `<crate>/benches/`; cross-project results live in
[`strided-rs-benchmark-suite`](https://github.com/tensor4all/strided-rs-benchmark-suite).
The numbers below are evidence for the decisions in this record, not a current
measurement.

### Gap 1: 1D contiguous sum — Rust ~2x slower than Julia Strided

| size | Rust strided-kernel (μs) | Julia Strided (μs) | Julia Base.sum (μs) |
|---:|---:|---:|---:|
| 1,048,576 | 940 | 474 | 106 |

Julia `Base.sum` uses hand-tuned SIMD reduction. Julia Strided is ~2x faster than Rust.

**Root cause**: the contiguous fast path in `strided-basic/src/reduce_view.rs`:

```rust
for &val in src.iter() {
    acc = reduce_fn(acc, map_fn(Op::apply(val)));
}
```

This has a **loop-carried dependency** — each iteration depends on the previous `acc` value. LLVM cannot auto-vectorize this even with `#[inline(always)]` and `impl Fn`, because it cannot prove floating-point addition is associative (which is required to split into independent accumulators).

The non-contiguous path had the same problem plus a per-element `acc.take().ok_or(...)?`; that `Option` accumulator has since been removed.

### Gap 2: 4D permute — up to 2.0x slower on specific patterns

| Permutation | Rust 1T (ms) | Julia 1T (ms) | Ratio |
|---|---:|---:|---:|
| (4,3,2,1) s=64 | 53 | 50 | 1.1x |
| (2,3,4,1) s=64 | 29 | 23 | 1.3x |
| (3,4,1,2) s=64 | 52 | 26 | **2.0x** |

(4,3,2,1) is near parity. (3,4,1,2) is 2x slower.

**Root cause**: Julia's `@simd` pragma guarantees vectorization of the stride=1 innermost loop. Rust relies on LLVM auto-vectorization through the closure in `strided-basic/src/map_view.rs::inner_loop_map1` (it was `kernel.rs` when this document was written). When the closure is trivial (identity copy), LLVM usually succeeds. But the presence of `Op::apply()` and generic `f` can inhibit vectorization in some code paths.

### Gap 3: complex_elementwise — 1.6x slower

Julia 7.8ms vs Rust 12.7ms. Julia's `@simd` enables aggressive vectorization of transcendental functions (`exp`, `sin`). Rust's LLVM generates more conservative code for the same operations.

### Gap 4: contiguous copy — Julia 10-25x faster

Julia uses optimized `memcpy` for contiguous array copy, while Rust's `copy_into` routed through `map_into` with an identity closure, which never collapsed to `memcpy`. **Closed:** `copy_into` now takes the contiguous-identity `copy_nonoverlapping` path in `strided-basic/src/ops_view.rs`.

### Gap 5: small array overhead — Rust 3-5x slower

4D permute s=4 (256 elements): Julia 0.33μs vs Rust 1.58μs. The internal fused-plan builder allocates multiple `Vec`s for dimension/stride reordering, dominating the cost for small arrays.

## Why `#[inline(always)]` + `impl Fn` is necessary but not sufficient

The current inner loop helpers are correctly designed:

```rust
#[inline(always)]
unsafe fn inner_loop_map1<T: Copy, Op: ElementOp>(
    ...,
    f: &impl Fn(T) -> T,  // monomorphized per closure type
) {
    if ds == 1 && ss == 1 {
        let dst = std::slice::from_raw_parts_mut(dp, len);
        let src = std::slice::from_raw_parts(sp, len);
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = f(Op::apply(*s));
        }
    }
    ...
}
```

- `impl Fn(T) -> T` ensures monomorphization: each closure type generates a dedicated function.
- `#[inline(always)]` ensures the dedicated function is inlined into the caller, giving LLVM full visibility of the loop body.
- Slice-based iteration when stride=1 gives LLVM contiguous memory access patterns.

This is the **correct approach for generic `map_into`** — arbitrary closures (`|x| x.exp() + x.sin()`) cannot be manually SIMD-ized without a SIMD math library. Auto-vectorization via LLVM is the only option.

**However**, this is insufficient for two cases:

1. **Reductions** (`sum`, `dot`, `reduce`): `acc = reduce_fn(acc, val)` has a loop-carried dependency. Even with perfect inlining, LLVM will not break the dependency chain because it cannot assume floating-point associativity. Manual multi-accumulator unrolling is required.

2. **Fixed operations** (`copy_into`, `add_acc`, `axpy_acc`): these are known at compile time and can bypass the closure entirely, using hand-written SIMD kernels or `memcpy`.

## Improvement Plan for strided-kernel

The plan below is the 2026-01-30 proposal, kept as the rationale for the
decisions listed under "Current Status". All items target what is now
`strided-basic/src/`; matmul performance was handled by the faer backend in the
retired `strided-einsum2`.

### Priority 1 (High Impact): SIMD reduce via `pulp` — shipped

Add a hand-written SIMD kernel for contiguous reductions using the faer pattern:

```rust
struct SumKernel<'a> { data: &'a [f64] }

impl pulp::WithSimd for SumKernel<'_> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> f64 {
        let (head, body, tail) = pulp::as_simd::<f64, S>(self.data);

        // Scalar head
        let mut scalar_acc = 0.0;
        for &x in head { scalar_acc += x; }

        // 4-way SIMD body (breaks loop-carried dependency)
        let mut acc = [simd.splat_f64s(0.0); 4];
        let chunks = body.chunks_exact(4);
        let remainder = chunks.remainder();
        for chunk in chunks {
            acc[0] = simd.add_f64s(acc[0], chunk[0]);
            acc[1] = simd.add_f64s(acc[1], chunk[1]);
            acc[2] = simd.add_f64s(acc[2], chunk[2]);
            acc[3] = simd.add_f64s(acc[3], chunk[3]);
        }
        for &v in remainder { acc[0] = simd.add_f64s(acc[0], v); }

        // Tree reduce
        let sum01 = simd.add_f64s(acc[0], acc[1]);
        let sum23 = simd.add_f64s(acc[2], acc[3]);
        scalar_acc + simd.reduce_sum_f64s(simd.add_f64s(sum01, sum23))

        // Scalar tail
        + tail.iter().sum::<f64>()
    }
}

pub fn sum_contiguous(data: &[f64]) -> f64 {
    pulp::Arch::default().dispatch(SumKernel { data })
}
```

**Expected improvement**: 2-4x for 1D sum, closing the gap with Julia Strided.

Applies to: `sum`, `dot`, `norm_l2_sqr`, and any contiguous reduction.

### Priority 2 (High Impact): `memcpy` for contiguous identity copy — shipped

```rust
pub fn copy_into<T: Copy>(dst: &mut StridedViewMut<T>, src: &StridedView<T>) -> Result<()> {
    if is_contiguous(dst.dims(), dst.strides())
        && is_contiguous(src.dims(), src.strides())
    {
        let len = total_len(dst.dims());
        unsafe {
            std::ptr::copy_nonoverlapping(src.ptr(), dst.as_mut_ptr(), len);
        }
        return Ok(());
    }
    // fall through to general kernel
    ...
}
```

**Expected improvement**: 10x+ for contiguous copy, matching Julia's `copy!`.

### Priority 3 (High Impact): `pulp` runtime dispatch for contiguous inner loops — shipped for the map, reduce and update kernels

Wrap the stride=1 fast paths in `pulp::Arch::default().dispatch(...)` so that AVX2/AVX-512/NEON is used without requiring `RUSTFLAGS="-C target-cpu=native"`.

Dispatch exists as `simd::dispatch_if_large` and the map, reduce and update inner
loops go through it: `map_into` / `zip_map*_into` (`strided-basic/src/map_view.rs`),
the fixed operations in `ops_view.rs`, `map_update_into` / `zip_update*_into`
(`update_view.rs`), `reduce` and `zip_reduce` (`reduce_view.rs:196`, `:213`,
`:521`, `:542`), and the erased reduction kernels (`erased/reduce_kernel.rs`).

It is not universal. These contiguous paths still run an undispatchsed loop:

- `reduce_axis` on a rank-1 source (`reduce_view.rs:826-834` -> `fold_run`
  `:104-107` -> `fold_contiguous` `:55-81`). Reproducer: a 1024-element
  `StridedArray::<f64>::col_major(&[1024])` folded with `reduce_axis(...)` under
  `Exec::serial()` never reaches `dispatch_if_large`.
- `reduce_axis_into`'s sweep and cursor loops (`reduce_view.rs:1047`, `:1051`,
  `:1148`, `:1153`).
- The copy, gather and structural plans (`copy_plan.rs`, `gather_plan.rs`,
  `static_indexing_plan.rs`), which contain no `simd::` call at all.

The dispatched bodies are still LLVM's auto-vectorization inside the selected
ISA; dispatch adds the ISA-level codegen path, not a hand-written kernel.

This is orthogonal to the other improvements and benefits every dispatched operation.

### Priority 4 (Medium Impact): dedicated SIMD kernels for fixed ops — block folds shipped

For `add_acc`, `axpy_acc`, `fma_acc`, `dot` — bypass the generic closure path with hand-written SIMD kernels using `pulp`. These operations are simple enough (add, mul-add) that explicit SIMD is straightforward.

The add/mul block folds are implemented in `strided-basic/src/simd.rs`; `add_acc`,
`axpy_acc` and `fma_acc` still run dispatched scalar loop bodies.

```rust
// Instead of: map_into(dst, src, |x| x)        → copy_into with memcpy
// Instead of: zip_map2_into(dst, a, b, |x,y| x+y) → add_simd kernel
// Instead of: reduce(src, |x| x, |a,b| a+b, 0.0)  → sum_simd kernel
```

### Priority 5 (Medium Impact): eliminate `Option` in reduce hot loop — shipped

`strided-basic/src/reduce_view.rs` no longer carries the `Option` accumulator
shown here (the path was `strided-kernel/src/reduce_view.rs` at the time).
Originally the loop was:

```rust
let mut acc = Some(init);
for _ in 0..len {
    let current = acc.take().ok_or(StridedError::OffsetOverflow)?;
    acc = Some(reduce_fn(current, mapped));
}
```
**Expected improvement**: 10-30% for non-contiguous reduce.

### Priority 6 (Low Impact): small-array overhead reduction — shipped as inline metadata

Replace `Vec` allocations in the internal fused-plan builder with `SmallVec<[_; 8]>` or stack arrays for rank ≤ 8 (covers the vast majority of use cases).

**Expected improvement**: 2-5x for arrays with < 1K elements.

### Priority 7 (Low Impact): pairwise summation for precision — not adopted

For large contiguous reductions (> 128 elements), recursively split in half before applying the SIMD kernel. This improves rounding error from O(n) to O(log n) at negligible performance cost.

## Summary: expected impact on strided-kernel

| # | Improvement | Target benchmark | Expected speedup | Crate |
|---|---|---|---|---|
| 1 | SIMD reduce (pulp, 4 accumulators) | 1D sum, dot, norm | 2-4x | strided-kernel |
| 2 | `memcpy` for contiguous copy | copy_into | 10x+ | strided-kernel |
| 3 | `pulp` runtime dispatch | all contiguous ops | 1.5-3x (default build) | strided-kernel |
| 4 | Dedicated SIMD for add/axpy/fma | add, axpy, fma | 1.5-2x | strided-kernel |
| 5 | Remove Option in reduce loop | non-contiguous reduce | 1.1-1.3x | strided-kernel |
| 6 | SmallVec for plan allocation | small arrays (s≤8) | 2-5x | strided-kernel |
| 7 | Pairwise summation | precision (speed-neutral) | — | strided-kernel |

Priorities 1+3 combined have the largest impact: they address the fundamental SIMD gap that accounts for most of the difference between Rust strided-kernel and Julia Strided.

Note: matmul/GEMM performance was addressed by the faer backend in
`strided-einsum2` (PR #19), which left the workspace after 0.4.4.
