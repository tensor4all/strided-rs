# Batch-Last Canonical Order for einsum2

## Problem

The current `strided-einsum2` uses batch-first canonical order:
- A[batch, lo, sum], B[batch, sum, ro], C[batch, lo, ro]

This is a row-major convention. In column-major (which this library uses),
the first dimension has stride 1. When col-major tensors have batch as the
last dimension (Julia convention), permuting to batch-first puts the
large-stride batch dims at the front, leaving inner dims with unit strides.
But when batch is already at the front in the physical layout, the inner
matrix gets non-unit strides (e.g., row_stride=b, col_stride=b*i), causing:

1. CBLAS: forced copy (REQUIRES_UNIT_STRIDE)
2. faer: slow non-vectorized path

## Solution

Change canonical order to batch-last:
- A[lo, sum, batch], B[sum, ro, batch], C[lo, ro, batch]

With column-major allocation, batch dims are last and naturally get the
largest strides. Each batch slice is a contiguous col-major matrix with
unit row stride.

## Changes

### 1. plan.rs - Permutation construction (3 lines)

Change permutation chain order:

```
left_perm:  batch.chain(lo).chain(sum)  -->  lo.chain(sum).chain(batch)
right_perm: batch.chain(sum).chain(ro)  -->  sum.chain(ro).chain(batch)
c_to_internal_perm: batch.chain(lo).chain(ro)  -->  lo.chain(ro).chain(batch)
```

### 2. contiguous.rs - Allocation and group extraction

Replace `alloc_batched_col_major` (row-major batch + col-major inner hack)
with pure col-major allocation. Batch dims are last, so standard col-major
gives them the largest strides automatically.

Flip all group extraction indexing in `prepare_input_view`,
`prepare_input_owned`, `prepare_output_view`, `prepare_output_owned`,
`prepare_input_view_for_backend`, `prepare_output_view_for_backend`:

```
Before: batch=[..n_batch], group1=[n_batch..n_batch+n_g1], group2=[n_batch+n_g1..]
After:  group1=[..n_g1], group2=[n_g1..n_g1+n_g2], batch=[n_g1+n_g2..]
```

### 3. bgemm_faer.rs - Dimension/stride extraction

Same index flipping in `bgemm_strided_into` and `bgemm_contiguous_into`.

### 4. bgemm_naive.rs - Dimension/stride extraction

Same index flipping in `bgemm_strided_into` and
`bgemm_strided_into_with_map`.

### 5. bgemm_blas.rs - Dimension/stride extraction

Same index flipping pattern.

### 6. lib.rs - Entry point functions

Same index flipping in `einsum2_into`, `einsum2_naive_into`,
`einsum2_with_backend_into`, `einsum2_gemm_dispatch`, `einsum2_into_owned`.

## Out of scope

- `strided-opteinsum`: no changes needed (calls einsum2_into internally)
- Public API: no changes (einsum2_into signature unchanged)

## Testing

- All existing tests should pass (they test einsum correctness, not internal
  ordering)
- Update plan.rs tests that assert specific permutation values
- Add regression test: col-major batch-last input verifying zero-copy path
  with unit inner strides
