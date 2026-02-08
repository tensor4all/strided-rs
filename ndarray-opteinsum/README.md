# ndarray-opteinsum

N-ary einsum frontend for [`ndarray`](https://crates.io/crates/ndarray) arrays, powered by `strided-opteinsum`.

## Scope

- Thin wrapper over `strided-opteinsum` that accepts `ndarray` `ArrayD<T>` and `ArrayViewD<T>` types directly
- Direct dims/strides passthrough (no notation reversal needed, unlike mdarray-opteinsum)
- Supports negative strides (reversed views)
- `einsum()` returns an owned `ArrayD<T>`; `einsum_into()` writes into a pre-allocated `ArrayViewMutD`
- Supports `f64` and `Complex64` operands

## Quick Example

```rust
use ndarray::ArrayD;
use ndarray_opteinsum::einsum;

// Row-major 2x3 and 3x2 matrices
let a = ArrayD::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
let b = ArrayD::from_shape_vec(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

// Matrix multiplication
let c: ArrayD<f64> = einsum("ij,jk->ik", vec![(&a).into(), (&b).into()]).unwrap();
```

## How It Works

ndarray has explicit strides (`&[isize]`, element-based) matching StridedView's convention,
so dims and strides are passed through directly without any reversal.

- **Input conversion**: Zero-copy wrapping of ndarray pointers, shapes, and strides as `StridedView`
- **Negative strides**: Supported via offset calculation (`compute_offset_range`)
- **`einsum_into()`**: Fully copy-free
- **`einsum()`**: One copy via `copy_into` to materialize the result into contiguous row-major order

### Comparison with mdarray-opteinsum

| | mdarray-opteinsum | ndarray-opteinsum |
|---|---|---|
| notation | label reversal required | passthrough |
| input conversion | reverse dims/strides | direct passthrough |
| negative strides | N/A | supported |
| output construction | unsafe `from_raw_parts` | safe `from_shape_vec` |

## API

```rust
// Allocating: returns owned ArrayD
pub fn einsum<T: EinsumScalar>(
    notation: &str,
    operands: Vec<NdOperand<'_>>,
) -> Result<ArrayD<T>>

// In-place: output = alpha * einsum(operands) + beta * output
pub fn einsum_into<T: EinsumScalar>(
    notation: &str,
    operands: Vec<NdOperand<'_>>,
    output: &mut ArrayViewMutD<'_, T>,
    alpha: T,
    beta: T,
) -> Result<()>
```

`NdOperand<'a>` is constructed via `From` impls on `&ArrayD<T>` and `ArrayViewD<'a, T>`.
