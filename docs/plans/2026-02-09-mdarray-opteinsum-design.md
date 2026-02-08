# mdarray-opteinsum Design

## Summary

A thin wrapper crate that lets `mdarray` users call N-ary einsum directly on `Array<T, DynRank>` and `View<T, DynRank>` types. Delegates all computation to `strided-opteinsum`.

## Scope

- Accept `mdarray` DynRank Dense arrays and views as einsum operands (f64, Complex64)
- Return results as owned `Array<T, DynRank>`
- Provide `einsum_into` for writing into pre-allocated `ViewMut`
- Transparently handle row-major (mdarray) to column-major (strided) conversion via dim reversal
- Zero-copy input wrapping, zero-copy output construction

## Public API

```rust
use mdarray::{Array, DynRank, View, ViewMut};
use num_complex::Complex64;

/// Type-erased operand wrapping mdarray types.
pub enum MdOperand<'a> { ... }

impl From<&Array<f64, DynRank>> for MdOperand<'_> { ... }
impl From<&Array<Complex64, DynRank>> for MdOperand<'_> { ... }
impl<'a> From<View<'a, f64, DynRank>> for MdOperand<'a> { ... }
impl<'a> From<View<'a, Complex64, DynRank>> for MdOperand<'a> { ... }

/// Compute einsum, return owned mdarray Array.
pub fn einsum<T: EinsumScalar>(
    notation: &str,
    operands: Vec<MdOperand<'_>>,
) -> Result<Array<T, DynRank>>

/// Compute einsum into pre-allocated mdarray output.
pub fn einsum_into<T: EinsumScalar>(
    notation: &str,
    operands: Vec<MdOperand<'_>>,
    output: &mut ViewMut<'_, T, DynRank>,
    alpha: T,
    beta: T,
) -> Result<()>
```

### Usage

```rust
let a: Array<f64, DynRank> = /* [3,4] matrix */;
let b: Array<f64, DynRank> = /* [4,5] matrix */;

// Allocating version
let c: Array<f64, DynRank> = einsum("ij,jk->ik", vec![(&a).into(), (&b).into()])?;

// Into pre-allocated output
let mut c: Array<f64, DynRank> = /* [3,5] matrix */;
einsum_into("ij,jk->ik", vec![(&a).into(), (&b).into()], &mut c.expr_mut(), 1.0, 0.0)?;
```

## Row-Major to Column-Major Conversion

mdarray uses row-major (C order), strided-view uses column-major (Fortran order). A row-major array with shape `[d0, d1, ..., dn]` has the same memory layout as a column-major array with shape `[dn, ..., d1, d0]`.

The crate handles this transparently by reversing dims and index labels:

```
User writes:   einsum("ij,jk->ik", [A[3,4], B[4,5]])

Step 1: Reverse each operand's index labels
         "ij" → "ji",  "jk" → "kj",  "ik" → "ki"
         Converted notation: "ji,kj->ki"

Step 2: Wrap each mdarray as StridedView with reversed dims
         A[3,4] row-major → StridedView [4,3] (same memory, zero-copy)
         B[4,5] row-major → StridedView [5,4] (same memory, zero-copy)

Step 3: Call strided_opteinsum::einsum("ji,kj->ki", operands)
         Returns StridedArray with col-major layout

Step 4: Convert result back to mdarray
         Result StridedArray [5,3] col-major
         → mdarray Array [3,5] row-major (reverse dims, transfer Vec)
```

Nested notation (e.g. `"(ij,jk),kl->il"`) works the same way — each leaf's labels are reversed independently, tree structure preserved.

## File Structure

```
mdarray-opteinsum/
  Cargo.toml
  src/
    lib.rs          # Public API (einsum, einsum_into) + MdOperand type
    convert.rs      # mdarray ↔ StridedView conversion + notation reversal
```

### convert.rs (~100-150 lines)

- `reverse_notation(notation: &str) -> String` — reverse index labels per operand/output group, handles nested `()` syntax
- `mdarray_to_strided_view<T>(&Array<T, DynRank>) -> StridedView<T>` — wrap with reversed dims, zero-copy
- `mdarray_view_to_strided_view<T>(View<T, DynRank>) -> StridedView<T>` — same for views
- `strided_array_to_mdarray<T>(StridedArray<T>) -> Array<T, DynRank>` — reverse dims back, transfer Vec ownership

### lib.rs (~100-150 lines)

- `MdOperand` enum + `From` impls
- `einsum()` — convert inputs, reverse notation, call `strided_opteinsum::einsum`, convert output
- `einsum_into()` — convert inputs + output view, reverse notation, call `strided_opteinsum::einsum_into`

## Dependencies

```toml
[dependencies]
mdarray = "0.7"
strided-opteinsum = { path = "../strided-opteinsum" }
strided-view = { path = "../strided-view" }
num-complex = "0.4"
thiserror = "1.0"
```

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Einsum(#[from] strided_opteinsum::EinsumError),

    #[error("unsupported element type: expected f64 or Complex64")]
    UnsupportedType,
}
```

Most validation (dimension mismatch, bad notation, shape errors) is handled by `strided-opteinsum`. No duplication.

## Edge Cases

- **Scalar output** (`"ij,ji->"`): 0-rank result. `DynRank` supports empty dims.
- **Single operand** (`"ij->ji"`): notation reversal applies normally.
- **Trace** (`"ii->"`): reversed is `"ii->"` (palindrome). Correct.
- **Nested notation** (`"(ij,jk),kl->il"`): each leaf reversed independently → `"(ji,kj),lk->li"`.

## Estimated Size

~200-300 lines total. Pure conversion glue, no algorithmic complexity.
