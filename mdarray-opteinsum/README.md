# mdarray-opteinsum

N-ary einsum frontend for [`mdarray`](https://crates.io/crates/mdarray) arrays, powered by `strided-opteinsum`.

## Scope

- Thin wrapper over `strided-opteinsum` that accepts `mdarray` `Array<T, DynRank>` and `View<T, DynRank>` types directly
- Transparent row-major (mdarray) / column-major (strided) layout conversion via dim and index-label reversal
- `einsum()` returns an owned `Array<T, DynRank>`; `einsum_into()` writes into a pre-allocated `ViewMut`
- Supports `f64` and `Complex64` operands

## Quick Example

```rust
use mdarray::{Array, DynRank};
use mdarray_opteinsum::einsum;

// Row-major 2x3 and 3x2 matrices
let a: Array<f64, DynRank> = /* ... */;
let b: Array<f64, DynRank> = /* ... */;

// Matrix multiplication
let c: Array<f64, DynRank> = einsum("ij,jk->ik", vec![(&a).into(), (&b).into()]).unwrap();
```

## How It Works

`mdarray` uses row-major (C) layout, while `strided-view` uses column-major (Fortran) layout.
A row-major array with shape `[d0, d1, ..., dn]` has the same memory layout as a column-major array with shape `[dn, ..., d1, d0]`.

The crate exploits this by:
1. Reversing each operand's dimensions and strides when creating `StridedView` wrappers (zero-copy)
2. Reversing the index labels in the einsum notation (e.g. `"ij,jk->ik"` becomes `"ji,kj->ki"`)
3. Reversing the result dimensions back to row-major when returning `Array<T, DynRank>`

`einsum_into()` is fully copy-free (the output `ViewMut` is wrapped as a `StridedViewMut` with reversed dims).
`einsum()` performs one copy via `copy_into` to materialize the result into dense row-major order.

## API

```rust
// Allocating: returns owned Array
pub fn einsum<T: EinsumScalar>(
    notation: &str,
    operands: Vec<MdOperand<'_>>,
) -> Result<Array<T, DynRank>>

// In-place: output = alpha * einsum(operands) + beta * output
pub fn einsum_into<'a, T: EinsumScalar>(
    notation: &str,
    operands: Vec<MdOperand<'_>>,
    output: &'a mut ViewMut<'a, T, DynRank>,
    alpha: T,
    beta: T,
) -> Result<()>
```

`MdOperand<'a>` is constructed via `From` impls on `&Array<T, DynRank>` and `View<'a, T, DynRank>`.
