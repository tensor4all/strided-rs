# ndarray-opteinsum Design

## Summary

A thin wrapper crate that lets `ndarray` users call N-ary einsum directly on `ArrayD<T>` and `ArrayViewD<T>` types. Delegates all computation to `strided-opteinsum`.

## Scope

- Accept `ndarray` dynamic-rank arrays and views as einsum operands (f64, Complex64)
- Return results as owned `ArrayD<T>`
- Provide `einsum_into` for writing into pre-allocated `ArrayViewMutD`
- Direct dims/strides passthrough (no notation reversal needed, unlike mdarray-opteinsum)
- Support negative strides (reversed views)

## Key Difference from mdarray-opteinsum

mdarray has implicit strides (always row-major dense), so mdarray-opteinsum reverses dims and notation labels to bridge the row-major/column-major gap. ndarray has **explicit strides** (`&[isize]`, element-based — same unit as StridedView), so dims and strides can be passed directly to StridedView without any reversal.

| | mdarray-opteinsum | ndarray-opteinsum |
|---|---|---|
| notation | label reversal required | passthrough |
| input conversion | reverse dims/strides | direct passthrough |
| negative strides | N/A (always positive) | supported via offset calculation |
| output construction | `ManuallyDrop` + `from_raw_parts` (unsafe) | `from_shape_vec` (safe) |

## Public API

```rust
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use num_complex::Complex64;

/// Type-erased operand wrapping ndarray types.
pub enum NdOperand<'a> {
    F64Array(&'a ArrayD<f64>),
    C64Array(&'a ArrayD<Complex64>),
    F64View(ArrayViewD<'a, f64>),
    C64View(ArrayViewD<'a, Complex64>),
}

impl From<&ArrayD<f64>> for NdOperand<'_> { ... }
impl From<&ArrayD<Complex64>> for NdOperand<'_> { ... }
impl<'a> From<ArrayViewD<'a, f64>> for NdOperand<'a> { ... }
impl<'a> From<ArrayViewD<'a, Complex64>> for NdOperand<'a> { ... }

/// Compute einsum, return owned ArrayD.
pub fn einsum<T: EinsumScalar>(
    notation: &str,
    operands: Vec<NdOperand<'_>>,
) -> Result<ArrayD<T>>

/// Compute einsum into pre-allocated output.
/// output = alpha * einsum(operands) + beta * output
pub fn einsum_into<T: EinsumScalar>(
    notation: &str,
    operands: Vec<NdOperand<'_>>,
    output: &mut ArrayViewMutD<'_, T>,
    alpha: T,
    beta: T,
) -> Result<()>
```

## Input Conversion (zero-copy)

ndarray's `as_ptr()` points to element `[0,0,...,0]`. With negative strides, accessible memory extends before this pointer. We compute the offset range:

```rust
fn compute_offset_range(shape: &[usize], strides: &[isize]) -> (isize, isize) {
    let mut min_off: isize = 0;
    let mut max_off: isize = 0;
    for (&d, &s) in shape.iter().zip(strides.iter()) {
        if d == 0 { continue; }
        let end = s * (d as isize - 1);
        if end < 0 { min_off += end; } else { max_off += end; }
    }
    (min_off, max_off)
}

fn ndarray_to_strided_view<'a, T>(arr: &'a ArrayD<T>) -> StridedView<'a, T> {
    let shape = arr.shape();
    let strides = arr.strides();
    let (min_off, max_off) = compute_offset_range(shape, strides);
    let base_ptr = unsafe { arr.as_ptr().offset(min_off) };
    let data_len = (max_off - min_off + 1) as usize;
    let data = unsafe { std::slice::from_raw_parts(base_ptr, data_len) };
    let offset = (-min_off) as usize;
    StridedView::new(data, shape, strides, offset).expect("valid bounds")
}
```

Same logic applies to `ArrayViewD` and `ArrayViewMutD` (with `as_mut_ptr`).

## Output Conversion

### `einsum()` — one copy to row-major

```rust
fn strided_array_to_ndarray<T>(arr: StridedArray<T>) -> ArrayD<T>
where
    T: Copy + ElementOpApply + Send + Sync + Zero + Default,
{
    let dims = arr.dims().to_vec();
    let total: usize = dims.iter().product();
    let mut buf = vec![T::default(); total];
    let rm_strides = row_major_strides(&dims);
    {
        let mut dest = StridedViewMut::new(&mut buf, &dims, &rm_strides, 0).expect("valid");
        copy_into(&mut dest, &arr.view()).expect("copy failed");
    }
    ArrayD::from_shape_vec(dims, buf).expect("shape matches")
}
```

No dims reversal needed. `from_shape_vec` is safe (no `ManuallyDrop`/`from_raw_parts`).

### `einsum_into()` — copy-free

The output `ArrayViewMutD` is wrapped as `StridedViewMut` using the same offset calculation as input views.

## File Structure

```
ndarray-opteinsum/
  Cargo.toml
  src/
    lib.rs        # NdOperand, einsum(), einsum_into()
    convert.rs    # ndarray ↔ StridedView, compute_offset_range
```

## Dependencies

```toml
[package]
name = "ndarray-opteinsum"
version = "0.1.0"
edition = "2021"
rust-version = "1.64"
license = "MIT OR Apache-2.0"
authors = ["Satoshi Terasaki", "Hiroshi Shinaoka"]
repository = "https://github.com/tensor4all/strided-rs"
description = "N-ary einsum frontend for ndarray arrays, powered by strided-opteinsum."
publish = false

[dependencies]
ndarray = "0.17"
strided-opteinsum = { path = "../strided-opteinsum", default-features = false }
strided-view = { path = "../strided-view" }
strided-kernel = { path = "../strided-kernel" }
num-complex = "0.4"
num-traits = "0.2"
thiserror = "1.0"

[features]
default = ["faer"]
faer = ["strided-opteinsum/faer"]
blas = ["strided-opteinsum/blas"]

[dev-dependencies]
approx = "0.5"
num-complex = "0.4"
```

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Einsum(#[from] strided_opteinsum::EinsumError),
}
```

All validation (dimension mismatch, bad notation, shape errors) delegated to `strided-opteinsum`.

## Edge Cases

- **Scalar output** (`"ij,ji->"`): 0-rank result. `IxDyn` supports empty dims.
- **Negative strides**: Handled by offset calculation in `compute_offset_range`.
- **Empty arrays** (dim = 0): `compute_offset_range` skips zero dims; `from_shape_vec` handles empty vecs.
- **Nested notation** (`"(ij,jk),kl->il"`): Passed through unchanged to `strided-opteinsum`.

## Estimated Size

~200-250 lines total. Simpler than mdarray-opteinsum (no notation reversal logic).
