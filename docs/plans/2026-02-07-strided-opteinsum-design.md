# strided-opteinsum Design

N-ary optimized einsum built on StridedView, porting mdarray-einsum from tensor4all-rs.

## Overview

`strided-opteinsum` is a new crate providing N-ary Einstein summation with:

- Nested string notation for contraction order: `"(ij,jk),kl->il"`
- Mixed scalar types (f64/c64) with automatic promotion
- Owned/borrowed operands with in-place permutation for owned intermediates
- Fallback to omeco optimizer when contraction order is underspecified
- Single-tensor operations via stride-trick diagonal views (zero-copy)

Pairwise contraction is delegated to `strided-einsum2`.

## Crate Location

`strided-opteinsum/` at repository root, added to workspace `Cargo.toml`.

## Dependencies

| Crate | Purpose |
|-------|---------|
| strided-view | StridedView, StridedArray, element ops |
| strided-kernel | reduce_trace_axes (new), copy_into, reduce_axis |
| strided-einsum2 | einsum2_into for pairwise contraction |
| omeco | Contraction path optimizer (fallback for 3+ tensor nodes) |
| num-complex | Complex64 |
| num-traits | Zero, One |
| thiserror | Error types |

## Module Layout

```
strided-opteinsum/src/
  lib.rs           -- Public API
  error.rs         -- EinsumError
  expr.rs          -- EinsumExpr tree type + recursive evaluation
  operand.rs       -- EinsumOperand enum (F64/C64 x Owned/View)
  parse.rs         -- Nested string parser "(ij,jk),kl->il"
  single_tensor.rs -- Single-tensor: permute + diagonal view + reduce
  typed_tensor.rs  -- TypedTensor enum + type promotion/demotion
```

## Core Types

### EinsumOperand

Heterogeneous inputs: each operand carries its own scalar type and ownership.

```rust
pub enum EinsumOperand<'a> {
    F64(StridedData<'a, f64>),
    C64(StridedData<'a, Complex64>),
}

pub enum StridedData<'a, T> {
    Owned(StridedArray<T>),
    View(StridedView<T>),  // Op materialized to Identity at construction
}
```

Element operations (Conj, Adjoint, etc.) are materialized into the data at
`EinsumOperand` construction via `From` impls. Lazy Op optimization still
happens inside `einsum2_into` for the pairwise GEMM step.

### EinsumExpr

Binary contraction tree, matching OMEinsum.jl's `NestedEinsum`.

```rust
pub enum EinsumExpr<'a, ID> {
    // Leaf: input tensor with axis labels
    Leaf {
        ids: Vec<ID>,
        operand: EinsumOperand<'a>,
    },
    // Contraction node: contract children, produce output_ids
    Contract {
        args: Vec<EinsumExpr<'a, ID>>,
        output_ids: Vec<ID>,
    },
}
```

- `args.len() == 1` -> single-tensor operation (permute/trace)
- `args.len() == 2` -> delegate to `einsum2_into`
- `args.len() >= 3` -> fallback to omeco to split into binary tree

### TypedTensor

For mixed-type results:

```rust
pub enum TypedTensor {
    F64(StridedArray<f64>),
    C64(StridedArray<Complex64>),
}
```

Type rules:
- All f64 inputs -> f64 result
- Any c64 input -> promote f64 operands to c64, result is c64
- If result is real-valued -> demote back to f64

## Nested String Notation

Parser for OMEinsum-compatible notation:

```
"(ij,jk),kl->il"
```

Parentheses define contraction order as a binary tree:
1. First contract tensors 1,2: `(ij,jk) -> ik`
2. Then contract result with tensor 3: `(ik,kl) -> il`

Flat notation (no parentheses) triggers omeco fallback:
```
"ij,jk,kl->il"   -- omeco determines optimal contraction order
```

Partial nesting also supported:
```
"(ij,jk,kl),lm->im"  -- 3-tensor group optimized by omeco, then contracted with tensor 4
```

## Execution Flow

### Tree Evaluation (recursive)

```
evaluate(expr) =
  match expr:
    Leaf -> return operand
    Contract { args, output_ids } ->
      if args.len() == 1:
        single_tensor(args[0], output_ids)
      else if args.len() == 2:
        a = evaluate(args[0])
        b = evaluate(args[1])
        pairwise_contract(a, b, output_ids)  // -> einsum2_into
      else:
        // 3+ tensors: omeco fallback
        path = omeco::optimize(args, output_ids)
        // restructure into binary tree, then evaluate
```

### Pairwise Contraction

For each pair:
1. Check scalar types of both operands
2. If mixed (f64 x c64): promote f64 operand to c64
3. Call `strided-einsum2::einsum2_into` with alpha=1, beta=0
4. Result becomes `Owned` operand for subsequent steps

### Single-Tensor Operations

| Pattern | Example | Implementation |
|---------|---------|----------------|
| Permutation only | `ijk->kji` | Stride reorder (zero-copy). In-place if Owned |
| Full trace | `ii->` | diagonal_view + reduce_axis |
| Partial trace | `iij->j` | diagonal_view + reduce_axis + permute |
| Diagonal extraction | `ijj->ij` | diagonal_view only (no reduction) |

### In-Place Permutation for Owned Operands

Intermediate results are always `Owned(StridedArray)`. When the next step only
requires permutation, update shape/strides metadata without copying data.

## Changes to Existing Crates

### strided-view

Add `diagonal_view` method:

```rust
impl StridedArrayView {
    /// Create a diagonal view by fusing repeated axis pairs.
    /// Strides are summed: s_new = s[axis_a] + s[axis_b]
    /// Shape: min(shape[axis_a], shape[axis_b])
    /// One dimension removed per pair. Zero-copy.
    fn diagonal_view(&self, axis_pairs: &[(usize, usize)]) -> StridedArrayView;
}
```

Example: `A[i,i,j]` shape=[n,n,m] strides=[s0,s1,s2]
-> shape=[n,m] strides=[s0+s1, s2]

### strided-kernel

Add `reduce_trace_axes`:

```rust
/// Diagonal view + reduce_axis composition.
/// Fuses repeated axis pairs via stride trick, then reduces.
pub fn reduce_trace_axes<T, Op>(
    src: &StridedView<T, Op>,
    trace_axis_pairs: &[(usize, usize)],
) -> Result<StridedArray<T>>;
```

### strided-einsum2

Remove `trace.rs`, replace with calls to `strided-kernel::reduce_trace_axes`.

## Dependency Graph

```
strided-view
    |
strided-kernel  (+ diagonal_view, reduce_trace_axes)
    |              \
strided-einsum2    strided-opteinsum (new)
                       |
                   (uses einsum2 for pairwise contraction)
                   (uses omeco for fallback optimization)
```

## Error Handling

```rust
pub enum EinsumError {
    DuplicateAxis(String),
    OrphanOutputAxis(String),
    DimensionMismatch { axis: String, dim_a: usize, dim_b: usize },
    ParseError(String),
    Strided(strided_view::StridedError),
    Einsum2(strided_einsum2::EinsumError),
}
```

## Future Work (out of scope)

- Compile-time einsum macro: `einsum!("(ij,jk),kl->il", a, b, c)` (issue #42)
- Avoid f64->c64 promotion via mixed-type einsum2 (issue #41)
