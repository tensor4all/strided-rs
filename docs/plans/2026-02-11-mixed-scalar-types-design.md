# Design: Mixed Scalar Types in map/broadcast/ops (Issue #5)

## Goal

Generalize `strided-kernel` element-wise operations to accept different scalar types
per operand (e.g., `f64 + Complex64 → Complex64`), without bulk pre-promotion.
Each array is read in its native type; the closure or trait bound handles per-element
conversion.

## Approach

Replace in-place: change single-type signatures to multi-type signatures.
Backward compatible via Rust type inference (`D=A=B=T` inferred for existing callers).
SIMD fast paths remain same-type only; mixed-type falls back to scalar.

## Changes by module

### map_view.rs — Separate type parameters per operand

```rust
// map_into: A → D
pub fn map_into<D, A, Op>(
    dest: &mut StridedViewMut<D>,
    src: &StridedView<A, Op>,
    f: impl Fn(A) -> D + MaybeSync,
) -> Result<()>
where
    D: Copy + MaybeSendSync,
    A: Copy + ElementOpApply + MaybeSendSync,
    Op: ElementOp,

// zip_map2_into: (A, B) → D
pub fn zip_map2_into<D, A, B, OpA, OpB>(
    dest: &mut StridedViewMut<D>,
    a: &StridedView<A, OpA>,
    b: &StridedView<B, OpB>,
    f: impl Fn(A, B) -> D + MaybeSync,
) -> Result<()>
where
    D: Copy + MaybeSendSync,
    A: Copy + ElementOpApply + MaybeSendSync,
    B: Copy + ElementOpApply + MaybeSendSync,

// zip_map3_into: (A, B, C) → D
pub fn zip_map3_into<D, A, B, C, OpA, OpB, OpC>(...)

// zip_map4_into: (A, B, C, E) → D
pub fn zip_map4_into<D, A, B, C, E, OpA, OpB, OpC, OpE>(...)
```

Inner loop functions gain matching type parameters. elem_size changes to:
```rust
let elem_size = size_of::<D>().max(size_of::<A>()).max(size_of::<B>());
```

### ops_view.rs — Trait bounds express type promotion

```rust
// add: dest[i] += src[i]
pub fn add<D, S, Op>(dest: &mut StridedViewMut<D>, src: &StridedView<S, Op>)
where D: Copy + Add<S, Output = D> + MaybeSendSync,
      S: Copy + ElementOpApply + MaybeSendSync,

// mul: dest[i] *= src[i]
pub fn mul<D, S, Op>(dest: &mut StridedViewMut<D>, src: &StridedView<S, Op>)
where D: Copy + Mul<S, Output = D> + MaybeSendSync,
      S: Copy + ElementOpApply + MaybeSendSync,

// axpy: dest[i] += alpha * src[i]
pub fn axpy<D, S, A, Op>(dest: &mut StridedViewMut<D>, src: &StridedView<S, Op>, alpha: A)
where A: Copy + Mul<S, Output = D>,
      D: Copy + Add<D, Output = D> + MaybeSendSync,
      S: Copy + ElementOpApply + MaybeSendSync,

// fma: dest[i] += a[i] * b[i]
pub fn fma<D, A, B, OpA, OpB>(dest: &mut StridedViewMut<D>, a: &StridedView<A, OpA>, b: &StridedView<B, OpB>)
where A: Copy + ElementOpApply + Mul<B, Output = D> + MaybeSendSync,
      B: Copy + ElementOpApply + MaybeSendSync,
      D: Copy + Add<D, Output = D> + MaybeSendSync,

// dot: sum(a[i] * b[i])
pub fn dot<A, B, R, OpA, OpB>(a: &StridedView<A, OpA>, b: &StridedView<B, OpB>) -> Result<R>
where A: Copy + ElementOpApply + Mul<B, Output = R> + MaybeSendSync,
      B: Copy + ElementOpApply + MaybeSendSync,
      R: Copy + Zero + Add<Output = R> + MaybeSendSync,

// copy_scale: dest[i] = scale * src[i]
pub fn copy_scale<D, S, A, Op>(dest: &mut StridedViewMut<D>, src: &StridedView<S, Op>, scale: A)
where A: Copy + Mul<S, Output = D>,
      D: Copy + MaybeSendSync,
      S: Copy + ElementOpApply + MaybeSendSync,
```

### block.rs / kernel.rs — No changes

`build_plan_fused` and `compute_block_sizes` already accept `elem_size: usize`.
Only call sites change: use `.max()` across all operand sizes.

### Unchanged modules

- `reduce_view.rs`: Already supports T→U with `.max()` elem_size
- `broadcast.rs`: CaptureArgs works with any closure type
- `kernel.rs`: Iteration engine is type-agnostic (operates on `isize` offsets)
- `copy_into`: Same-type by definition
- `symmetrize_into`, `copy_transpose_scale_into`: Keep single-type

### SIMD

`sum()` and `dot()` SIMD fast paths (`MaybeSimdOps`) remain same-type + Identity only.
Mixed-type calls bypass SIMD, use scalar closure path via generalized `reduce`/`zip_map2_into`.

## Implementation order

1. **map_view.rs**: Generalize `map_into`, `zip_map2_into`, `zip_map3_into`, `zip_map4_into`
2. **ops_view.rs**: Update `add`, `mul`, `axpy`, `fma`, `dot`, `copy_scale`
3. **Tests**: Add mixed f64/Complex64 test cases for each changed function
4. **Verify**: All existing tests pass (backward compat)

## Files to modify

| File | Change |
|------|--------|
| `strided-kernel/src/map_view.rs` | Generalize type params, inner loops, elem_size |
| `strided-kernel/src/ops_view.rs` | Generalize type params, delegate to map functions |
| `strided-kernel/tests/correctness_view.rs` | Add mixed-type tests |

## Out of scope

- `strided-opteinsum` / `strided-einsum2`: Downstream consumers, adopt later
- Mixed-type SIMD kernels: Scalar fallback is sufficient
- `broadcast.rs`: Already works via closure genericity
