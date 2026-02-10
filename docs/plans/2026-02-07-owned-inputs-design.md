# Design: Accept Owned Inputs in einsum2_into (Issue #35)

## Problem

`einsum2_into` currently accepts inputs only as borrowed views. Before GEMM,
non-contiguous operands are copied into fresh col-major buffers. For C (output),
this means two full copies: copy-in (if beta != 0) and copy-back after GEMM.

When the caller owns an input and won't use it again, the library could avoid
the copy-back by transferring ownership of the buffer.

## Design Decisions

- **Scope**: A, B, and C — all three operands
- **Permutation strategy**: Out-of-place copy + ownership transfer (same as
  Strided.jl). Trait design allows future in-place permutation drop-in.
- **Trait design**: Separate traits for inputs (A/B) and output (C)
- **API compatibility**: Replace concrete types with trait bounds. Existing
  `&StridedView` / `StridedViewMut` calls continue to work via trait impls.

## Trait Definitions

### IntoContiguousView (A/B inputs)

```rust
/// Trait for input operands that can provide a contiguous view for GEMM.
/// Abstracts over owned vs. borrowed; future impls can use in-place permutation.
pub trait IntoContiguousView<T: Scalar> {
    /// Permute to canonical axis order and ensure contiguity.
    /// Returns a GEMM-ready operand.
    fn into_contiguous(self, perm: &[usize]) -> Result<ContiguousOperand<T>>;

    /// Whether the element operation implies conjugation.
    fn is_conj(&self) -> bool;
}
```

### IntoContiguousViewMut (C output)

```rust
/// Trait for output operands. Handles pre-GEMM setup and post-GEMM writeback.
pub trait IntoContiguousViewMut<T: Scalar> {
    /// Prepare C for GEMM: permute to canonical order, ensure contiguity.
    /// If beta != 0, existing data must be preserved.
    fn prepare_for_gemm(&mut self, perm: &[usize], beta: T)
        -> Result<ContiguousOperandMut<T>>;
}
```

## GEMM-Ready Intermediate Types

```rust
/// GEMM-ready input operand.
pub struct ContiguousOperand<T> {
    pub ptr: *const T,
    pub row_stride: isize,
    pub col_stride: isize,
    pub batch_stride: isize,
    pub conj: bool,
    /// Owns the buffer if a copy was made (or input was consumed).
    _buf: Option<StridedArray<T>>,
}

/// GEMM-ready output operand.
/// After GEMM, call finalize() to write back if needed.
pub struct ContiguousOperandMut<'a, T> {
    pub ptr: *mut T,
    pub row_stride: isize,
    pub col_stride: isize,
    pub batch_stride: isize,
    /// Write-back destination (only for borrowed non-contiguous C).
    writeback: Option<WritebackInfo<'a, T>>,
    _buf: Option<StridedArray<T>>,
}

struct WritebackInfo<'a, T> {
    dest: StridedViewMut<'a, T>,
}

impl<'a, T: Scalar> ContiguousOperandMut<'a, T> {
    /// After GEMM. Borrowed non-contiguous C: copy-back. Owned C: no-op.
    pub fn finalize(self) -> Result<()> {
        if let Some(info) = self.writeback {
            let buf = self._buf.unwrap();
            copy_into(&mut info.dest, &buf.view())?;
        }
        Ok(())
    }
}
```

## Trait Implementations

### Borrowed inputs (`&StridedView`) — existing behavior

```rust
impl<'a, T: Scalar, Op: ElementOp> IntoContiguousView<T> for &'a StridedView<'a, T, Op> {
    fn into_contiguous(self, perm: &[usize]) -> Result<ContiguousOperand<T>> {
        let permuted = self.permute(perm)?;
        if is_fusable(&permuted) {
            // Already contiguous -> zero-copy view
            Ok(ContiguousOperand { ptr, strides, conj, _buf: None })
        } else {
            // Non-contiguous -> alloc + copy (current behavior)
            let mut buf = alloc_batched_col_major(permuted.dims(), n_batch);
            copy_into(&mut buf.view_mut(), &strip_op(permuted))?;
            Ok(ContiguousOperand { ptr: buf.ptr(), strides, conj, _buf: Some(buf) })
        }
    }
    fn is_conj(&self) -> bool { op_is_conj::<Op>() }
}
```

### Owned inputs (`StridedArray`) — new path

```rust
impl<T: Scalar> IntoContiguousView<T> for StridedArray<T> {
    fn into_contiguous(self, perm: &[usize]) -> Result<ContiguousOperand<T>> {
        let permuted_view = self.view().permute(perm)?;
        if is_fusable(&permuted_view) {
            // Already contiguous -> pass ownership through (no copy)
            Ok(ContiguousOperand { ptr, strides, conj: false, _buf: Some(self) })
        } else {
            // Non-contiguous -> alloc new col-major buffer, copy
            // Same alloc cost as borrowed, but caller doesn't need original back
            let mut buf = alloc_batched_col_major(permuted_view.dims(), n_batch);
            copy_into(&mut buf.view_mut(), &permuted_view)?;
            Ok(ContiguousOperand { ptr: buf.ptr(), strides, conj: false, _buf: Some(buf) })
        }
    }
    fn is_conj(&self) -> bool { false }
}
```

### Borrowed output (`StridedViewMut`) — existing behavior

```rust
impl<'a, T: Scalar> IntoContiguousViewMut<T> for StridedViewMut<'a, T> {
    fn prepare_for_gemm(&mut self, perm: &[usize], beta: T)
        -> Result<ContiguousOperandMut<T>>
    {
        let permuted = self.permute(perm)?;  // metadata-only
        if is_fusable(&permuted) {
            // Contiguous -> use directly
            Ok(ContiguousOperandMut { ptr, strides, writeback: None, _buf: None })
        } else {
            // Non-contiguous -> alloc buffer, copy-in if beta != 0
            let mut buf = alloc_batched_col_major(permuted.dims(), n_batch);
            if beta != T::zero() {
                copy_into(&mut buf.view_mut(), &permuted.as_view())?;
            }
            Ok(ContiguousOperandMut {
                ptr: buf.ptr_mut(),
                strides,
                writeback: Some(WritebackInfo { dest: *self }),
                _buf: Some(buf),
            })
        }
    }
}
```

### Owned output (`StridedArray`) — new path, biggest win

```rust
impl<T: Scalar> IntoContiguousViewMut<T> for StridedArray<T> {
    fn prepare_for_gemm(&mut self, perm: &[usize], beta: T)
        -> Result<ContiguousOperandMut<T>>
    {
        let permuted = self.view().permute(perm)?;
        if is_fusable(&permuted) {
            Ok(ContiguousOperandMut { ptr, strides, writeback: None, _buf: None })
        } else {
            // Alloc col-major buffer, copy-in if beta != 0
            let mut buf = alloc_batched_col_major(permuted.dims(), n_batch);
            if beta != T::zero() {
                copy_into(&mut buf.view_mut(), &permuted.as_view())?;
            }
            // No writeback needed — GEMM result stays in buf
            // Caller receives the col-major buffer as the result
            Ok(ContiguousOperandMut {
                ptr: buf.ptr_mut(),
                strides,
                writeback: None,  // <-- key difference: no copy-back
                _buf: Some(buf),
            })
        }
    }
}
```

## Public API Change

```rust
// Before:
pub fn einsum2_into<T: Scalar, OpA, OpB, ID: AxisId>(
    c: StridedViewMut<T>,
    a: &StridedView<T, OpA>,
    b: &StridedView<T, OpB>,
    ic: &[ID], ia: &[ID], ib: &[ID],
    alpha: T, beta: T,
) -> Result<()>

// After:
pub fn einsum2_into<T: Scalar, A, B, C, ID: AxisId>(
    c: C,
    a: A,
    b: B,
    ic: &[ID], ia: &[ID], ib: &[ID],
    alpha: T, beta: T,
) -> Result<()>
where
    A: IntoContiguousView<T>,
    B: IntoContiguousView<T>,
    C: IntoContiguousViewMut<T>,
```

Existing calls with `&StridedView` / `StridedViewMut` continue to work.

## Internal Flow Changes

```
1. Plan construction              (unchanged)
2. Trace reduction                (unchanged — returns StridedArray)
   -> trace result can now be passed as owned to into_contiguous()
3. Permutation + contiguity
   - Before: bgemm_faer.rs does permute -> try_fuse -> alloc+copy
   - After:  a.into_contiguous(perm) / b.into_contiguous(perm) / c.prepare_for_gemm(perm, beta)
4. Element-wise fast path         (unchanged — zip_map2_into)
5. GEMM dispatch
   - bgemm_strided_into receives ContiguousOperand / ContiguousOperandMut
   - Just passes pointers + strides to faer
6. Finalize C                     (c_operand.finalize())
```

Copy decision logic moves from `bgemm_faer.rs` into trait impls.
`bgemm_strided_into` simplifies to "call faer with pre-contiguous data".

## Integration with strided-opteinsum

`eval_pair_ref` in `strided-opteinsum/src/expr.rs` is the main caller.
`StridedData` already distinguishes `Owned(StridedArray)` / `View(&StridedView)`.

Currently both are converted to views before calling `einsum2_into`.
After this change, owned operands can be passed by value:

```rust
match (left, right) {
    (Owned(a), Owned(b)) => einsum2_into(c, a, b, ...),
    (Owned(a), View(b))  => einsum2_into(c, a, &b, ...),
    (View(a),  Owned(b)) => einsum2_into(c, &a, b, ...),
    (View(a),  View(b))  => einsum2_into(c, &a, &b, ...),
}
```

In contraction tree evaluation, intermediate results are almost always `Owned`,
so the owned path is used for the majority of pairwise contractions.

C is always freshly allocated with beta == 0 in `eval_pair_ref`, so:
**zero copies for C** (alloc + GEMM writes directly).

## Savings Summary

| Operand | Borrowed (current)       | Owned (new)                     |
|---------|--------------------------|---------------------------------|
| A       | alloc + copy             | alloc + copy (same)             |
| B       | alloc + copy             | alloc + copy (same)             |
| C       | alloc + copy-in + copy-back | alloc only (no copies if beta==0) |
| C (beta!=0) | alloc + copy-in + copy-back | alloc + copy-in (no copy-back) |

Primary win: **C's copy-back elimination** in the owned path.
Secondary win: **trait design enables future in-place permutation** to also
eliminate alloc+copy for A/B.

## Test Strategy

1. **Correctness**: All 8 combinations (A/B/C each owned/borrowed), with:
   - Contiguous and non-contiguous layouts
   - beta == 0 and beta != 0
   - Conjugated operands
2. **Regression**: Existing tests pass unchanged after signature change
3. **Benchmarks**: `strided-opteinsum/benches/` with `RAYON_NUM_THREADS=1`

## Non-Goals

- True in-place cycle-following permutation (future optimization)
- Mixed scalar types (#5) — trait design is forward-compatible
- SIMD optimization for permutation/copy
- Changing behavior for users who only have views
