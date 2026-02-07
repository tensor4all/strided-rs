# Owned Inputs for einsum2_into — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Accept owned `StridedArray` inputs in `einsum2_into` to eliminate unnecessary buffer copies, especially the C copy-back.

**Architecture:** Introduce `IntoContiguousView` (inputs) and `IntoContiguousViewMut` (output) traits that abstract over owned vs. borrowed operands. Move the copy-decision logic from `bgemm_faer.rs` into trait impls. The `bgemm_strided_into` function receives pre-contiguous operands.

**Tech Stack:** Rust, strided-view, strided-kernel, strided-einsum2, strided-opteinsum, faer

**Working directory:** `/Users/hiroshi/projects/tensor4all/strided-rs/.worktrees/owned-inputs`

---

### Task 1: Define ContiguousOperand and ContiguousOperandMut types

**Files:**
- Create: `strided-einsum2/src/contiguous.rs`
- Modify: `strided-einsum2/src/lib.rs:24-34` (add `mod contiguous;`)

**Step 1: Create the contiguous module with types**

Create `strided-einsum2/src/contiguous.rs` with the GEMM-ready intermediate types:

```rust
//! GEMM-ready operand types and traits for contiguous data preparation.

use crate::util::try_fuse_group;
use crate::Scalar;
use strided_view::{StridedArray, StridedView, StridedViewMut};

/// GEMM-ready input operand with contiguous data.
pub struct ContiguousOperand<T> {
    /// Pointer to the start of the data (first batch slice).
    ptr: *const T,
    /// Row stride (lo dimension stride for A, sum dim stride for B).
    row_stride: isize,
    /// Column stride (sum dim stride for A, ro dim stride for B).
    col_stride: isize,
    /// Batch strides (one per batch dimension).
    batch_strides: Vec<isize>,
    /// Whether the element operation implies conjugation.
    conj: bool,
    /// Owns the buffer if a copy was made or input was consumed.
    _buf: Option<StridedArray<T>>,
}

/// GEMM-ready output operand with contiguous data.
pub struct ContiguousOperandMut<'a, T> {
    /// Mutable pointer to the start of the data.
    ptr: *mut T,
    /// Row stride (lo dimension stride).
    row_stride: isize,
    /// Column stride (ro dimension stride).
    col_stride: isize,
    /// Batch strides (one per batch dimension).
    batch_strides: Vec<isize>,
    /// Write-back destination for borrowed non-contiguous C.
    writeback: Option<StridedViewMut<'a, T>>,
    /// Owns the buffer if a copy was made.
    _buf: Option<StridedArray<T>>,
}

impl<T> ContiguousOperand<T> {
    pub fn ptr(&self) -> *const T { self.ptr }
    pub fn row_stride(&self) -> isize { self.row_stride }
    pub fn col_stride(&self) -> isize { self.col_stride }
    pub fn batch_strides(&self) -> &[isize] { &self.batch_strides }
    pub fn conj(&self) -> bool { self.conj }
}

impl<'a, T> ContiguousOperandMut<'a, T> {
    pub fn ptr(&self) -> *mut T { self.ptr }
    pub fn row_stride(&self) -> isize { self.row_stride }
    pub fn col_stride(&self) -> isize { self.col_stride }
    pub fn batch_strides(&self) -> &[isize] { &self.batch_strides }
}
```

**Step 2: Add `mod contiguous;` to lib.rs**

In `strided-einsum2/src/lib.rs`, add after line 34 (`pub mod util;`):

```rust
/// GEMM-ready operand types and traits for accepting owned inputs.
pub mod contiguous;
```

**Step 3: Verify it compiles**

Run: `cargo build -p strided-einsum2`
Expected: compiles with warnings about dead code (types unused yet)

**Step 4: Commit**

```bash
git add strided-einsum2/src/contiguous.rs strided-einsum2/src/lib.rs
git commit -m "feat: add ContiguousOperand/ContiguousOperandMut types"
```

---

### Task 2: Add IntoContiguousView trait with borrowed impl

**Files:**
- Modify: `strided-einsum2/src/contiguous.rs`

This task adds the `IntoContiguousView` trait and the impl for borrowed `&StridedView`.
The logic mirrors `bgemm_faer.rs:77-99` (A copy path) and `106-127` (B copy path).

**Step 1: Write a test for borrowed input contiguous preparation**

Add at the bottom of `strided-einsum2/src/contiguous.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use strided_view::StridedArray;

    #[test]
    fn test_borrowed_contiguous_no_copy() {
        // Col-major 2x3: lo strides [1], sum strides [2] -> fusable
        let a = StridedArray::<f64>::col_major(&[2, 3]);
        let view = a.view();
        let op = prepare_input_view(
            &view, 0, 1, 1, false,
        ).unwrap();
        // Should not allocate a buffer (already contiguous)
        assert!(op._buf.is_none());
        assert_eq!(op.row_stride(), 1);
        assert_eq!(op.col_stride(), 2);
    }

    #[test]
    fn test_borrowed_non_contiguous_copies() {
        // Row-major 3x4x5 with n_batch=1, n_lo=1, n_sum=1
        // After batch split: lo_dims=[4], lo_strides=[5], sum_dims=[5], sum_strides=[1]
        // Fusion: lo=[4] stride [5] -> fuse=(4,5), sum=[5] stride [1] -> fuse=(5,1)
        // BUT the fused strides are 5 and 1, which means lo_stride=5, sum_stride=1.
        // For col-major GEMM we want lo_stride=1 (rows), sum_stride=m (cols),
        // so if strides don't match col-major, copy is needed.
        //
        // Actually try_fuse_group just checks internal contiguity.
        // A row-major [4,5] has strides [5,1], fused_lo=(4,5), fused_sum=(5,1).
        // Both fuse, so no copy is needed. Let's test a truly non-contiguous case.
        //
        // Create a view with non-contiguous inner dims via permute.
        let a = StridedArray::<f64>::from_fn_row_major(&[3, 4], |idx| {
            (idx[0] * 4 + idx[1] + 1) as f64
        });
        // Permute to [4, 3] -> strides become [1, 4] (transposed)
        let view = a.view().permute(&[1, 0]).unwrap();
        // dims=[4,3], strides=[1,4]. n_batch=0, n_lo=1, n_sum=1.
        // lo_dims=[4], lo_strides=[1], sum_dims=[3], sum_strides=[4].
        // try_fuse lo -> (4, 1) ok. try_fuse sum -> (3, 4) ok.
        // Both fuse! Still no copy. The strides just tell faer the layout.
        let op = prepare_input_view(
            &view, 0, 1, 1, false,
        ).unwrap();
        assert!(op._buf.is_none());

        // For a truly non-fusable case: 2D view where dims are split
        // into multiple lo/sum dims that aren't contiguous.
        // E.g. dims [2, 3, 4] with n_batch=0, n_lo=2 (dims [2,3]), n_sum=1 (dim [4])
        // Row-major strides: [12, 4, 1].
        // lo_dims=[2,3], lo_strides=[12,4]. try_fuse: sort by |s| -> [(4,3),(12,2)]
        // check: 4*3=12 == 12 ✓ -> fuse ok.
        // This is still fusable. We need strides that don't satisfy the product rule.
        //
        // Create via from_parts with custom strides:
        let data = vec![0.0f64; 100];
        let arr = StridedArray::<f64>::from_parts(data, &[2, 3, 4], &[20, 4, 1], 0).unwrap();
        // lo_dims=[2,3], lo_strides=[20,4]. sort -> [(4,3),(20,2)]. 4*3=12 != 20 -> NOT fusable!
        let view = arr.view();
        let op = prepare_input_view(
            &view, 0, 2, 1, false,
        ).unwrap();
        assert!(op._buf.is_some()); // Should have allocated a copy
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p strided-einsum2 test_borrowed_contiguous_no_copy`
Expected: FAIL (function `prepare_input_view` not found)

**Step 3: Implement `prepare_input_view` function**

Add to `strided-einsum2/src/contiguous.rs` (above tests):

```rust
use crate::bgemm_faer::alloc_batched_col_major;

/// Prepare a borrowed input view for GEMM.
///
/// Checks if the lo and sum (or sum and ro) dimension groups are fusable.
/// If not, copies to a contiguous col-major buffer.
///
/// - `n_batch`: number of leading batch dimensions
/// - `n_group1`: number of dims in first inner group (lo for A, sum for B)
/// - `n_group2`: number of dims in second inner group (sum for A, ro for B)
/// - `conj`: whether the element operation implies conjugation
pub fn prepare_input_view<T: Scalar>(
    view: &StridedView<T>,
    n_batch: usize,
    n_group1: usize,
    n_group2: usize,
    conj: bool,
) -> crate::Result<ContiguousOperand<T>> {
    let dims = view.dims();
    let strides = view.strides();

    let group1_dims = &dims[n_batch..n_batch + n_group1];
    let group1_strides = &strides[n_batch..n_batch + n_group1];
    let group2_dims = &dims[n_batch + n_group1..n_batch + n_group1 + n_group2];
    let group2_strides = &strides[n_batch + n_group1..n_batch + n_group1 + n_group2];

    let fused1 = try_fuse_group(group1_dims, group1_strides);
    let fused2 = try_fuse_group(group2_dims, group2_strides);

    let m: usize = group1_dims.iter().product::<usize>().max(1);

    if fused1.is_some() && fused2.is_some() {
        // Already contiguous — use view directly
        let (_, rs) = fused1.unwrap();
        let (_, cs) = fused2.unwrap();
        Ok(ContiguousOperand {
            ptr: view.ptr(),
            row_stride: rs,
            col_stride: cs,
            batch_strides: strides[..n_batch].to_vec(),
            conj,
            _buf: None,
        })
    } else {
        // Non-contiguous — copy to col-major buffer
        let mut buf = alloc_batched_col_major(dims, n_batch);
        strided_kernel::copy_into(&mut buf.view_mut(), view)?;
        let ptr = buf.view().ptr();
        let batch_strides = buf.strides()[..n_batch].to_vec();
        Ok(ContiguousOperand {
            ptr,
            row_stride: if m == 0 { 0 } else { 1 },
            col_stride: m as isize,
            batch_strides,
            conj,
            _buf: Some(buf),
        })
    }
}
```

Note: `alloc_batched_col_major` is currently private in `bgemm_faer.rs`. We'll need to make it `pub(crate)`.

**Step 4: Make `alloc_batched_col_major` pub(crate)**

In `strided-einsum2/src/bgemm_faer.rs:218`, change:

```rust
fn alloc_batched_col_major<T: Copy>(dims: &[usize], n_batch: usize) -> StridedArray<T> {
```

to:

```rust
pub(crate) fn alloc_batched_col_major<T: Copy>(dims: &[usize], n_batch: usize) -> StridedArray<T> {
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p strided-einsum2 test_borrowed_contiguous`
Expected: PASS

**Step 6: Commit**

```bash
git add strided-einsum2/src/contiguous.rs strided-einsum2/src/bgemm_faer.rs
git commit -m "feat: add prepare_input_view for borrowed inputs"
```

---

### Task 3: Add prepare_input_owned for owned StridedArray inputs

**Files:**
- Modify: `strided-einsum2/src/contiguous.rs`

**Step 1: Write a test for owned input**

Add to the test module in `contiguous.rs`:

```rust
#[test]
fn test_owned_contiguous_no_copy() {
    let a = StridedArray::<f64>::col_major(&[2, 3]);
    let op = prepare_input_owned(a, 0, 1, 1, false).unwrap();
    // Contiguous — buffer is the original owned array (transferred, not copied)
    assert!(op._buf.is_some()); // owned always stored in _buf
    assert_eq!(op.row_stride(), 1);
    assert_eq!(op.col_stride(), 2);
}

#[test]
fn test_owned_non_contiguous_copies() {
    let data = vec![0.0f64; 100];
    let arr = StridedArray::<f64>::from_parts(data, &[2, 3, 4], &[20, 4, 1], 0).unwrap();
    let op = prepare_input_owned(arr, 0, 2, 1, false).unwrap();
    // Non-contiguous — should allocate a new col-major buffer
    assert!(op._buf.is_some());
    assert_eq!(op.row_stride(), 1);
    assert_eq!(op.col_stride(), 6); // m = 2*3 = 6
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p strided-einsum2 test_owned_contiguous_no_copy`
Expected: FAIL (function not found)

**Step 3: Implement `prepare_input_owned`**

Add to `contiguous.rs`:

```rust
/// Prepare an owned input array for GEMM.
///
/// If already contiguous after dimension grouping, transfers ownership
/// without copying. Otherwise, copies to a new col-major buffer.
pub fn prepare_input_owned<T: Scalar>(
    arr: StridedArray<T>,
    n_batch: usize,
    n_group1: usize,
    n_group2: usize,
    conj: bool,
) -> crate::Result<ContiguousOperand<T>> {
    let dims = arr.dims().to_vec();
    let strides = arr.strides().to_vec();

    let group1_dims = &dims[n_batch..n_batch + n_group1];
    let group1_strides = &strides[n_batch..n_batch + n_group1];
    let group2_dims = &dims[n_batch + n_group1..n_batch + n_group1 + n_group2];
    let group2_strides = &strides[n_batch + n_group1..n_batch + n_group1 + n_group2];

    let fused1 = try_fuse_group(group1_dims, group1_strides);
    let fused2 = try_fuse_group(group2_dims, group2_strides);

    let m: usize = group1_dims.iter().product::<usize>().max(1);

    if fused1.is_some() && fused2.is_some() {
        // Already contiguous — transfer ownership, no copy
        let (_, rs) = fused1.unwrap();
        let (_, cs) = fused2.unwrap();
        let batch_strides = strides[..n_batch].to_vec();
        let ptr = arr.view().ptr();
        Ok(ContiguousOperand {
            ptr,
            row_stride: rs,
            col_stride: cs,
            batch_strides,
            conj,
            _buf: Some(arr),
        })
    } else {
        // Non-contiguous — copy to col-major buffer (same as borrowed path)
        let mut buf = alloc_batched_col_major(&dims, n_batch);
        strided_kernel::copy_into(&mut buf.view_mut(), &arr.view())?;
        let ptr = buf.view().ptr();
        let batch_strides = buf.strides()[..n_batch].to_vec();
        Ok(ContiguousOperand {
            ptr,
            row_stride: if m == 0 { 0 } else { 1 },
            col_stride: m as isize,
            batch_strides,
            conj,
            _buf: Some(buf),
        })
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p strided-einsum2 test_owned_`
Expected: PASS

**Step 5: Commit**

```bash
git add strided-einsum2/src/contiguous.rs
git commit -m "feat: add prepare_input_owned for owned inputs"
```

---

### Task 4: Add prepare_output_view and prepare_output_owned + finalize

**Files:**
- Modify: `strided-einsum2/src/contiguous.rs`

**Step 1: Write tests for output preparation**

Add to tests in `contiguous.rs`:

```rust
#[test]
fn test_output_view_contiguous() {
    let mut c = StridedArray::<f64>::col_major(&[2, 3]);
    let op = prepare_output_view(&mut c.view_mut(), 0, 1, 1, 0.0).unwrap();
    assert!(op.writeback.is_none());
    assert!(op._buf.is_none());
}

#[test]
fn test_output_view_non_contiguous_beta_zero() {
    // Create non-fusable output
    let data = vec![0.0f64; 100];
    let mut arr = StridedArray::<f64>::from_parts(data, &[2, 3, 4], &[20, 4, 1], 0).unwrap();
    let mut vm = arr.view_mut();
    let op = prepare_output_view(&mut vm, 0, 2, 1, 0.0).unwrap();
    assert!(op.writeback.is_some()); // needs copy-back
    assert!(op._buf.is_some());
}

#[test]
fn test_output_owned_non_contiguous_no_writeback() {
    let data = vec![0.0f64; 100];
    let arr = StridedArray::<f64>::from_parts(data, &[2, 3, 4], &[20, 4, 1], 0).unwrap();
    let op = prepare_output_owned(arr, 0, 2, 1, 0.0).unwrap();
    // Owned path: no writeback needed even when non-contiguous
    assert!(op.writeback.is_none());
    assert!(op._buf.is_some());
}
```

**Step 2: Run test to verify failure**

Run: `cargo test -p strided-einsum2 test_output_view_contiguous`
Expected: FAIL

**Step 3: Implement output preparation functions**

Add to `contiguous.rs`:

```rust
/// Prepare a borrowed mutable output view for GEMM.
///
/// If non-contiguous, copies C into a col-major buffer (if beta != 0)
/// and sets up writeback for after GEMM.
pub fn prepare_output_view<'a, T: Scalar>(
    view: &mut StridedViewMut<'a, T>,
    n_batch: usize,
    n_group1: usize,
    n_group2: usize,
    beta: T,
) -> crate::Result<ContiguousOperandMut<'a, T>> {
    let dims = view.dims().to_vec();
    let strides = view.strides().to_vec();

    let group1_dims = &dims[n_batch..n_batch + n_group1];
    let group1_strides = &strides[n_batch..n_batch + n_group1];
    let group2_dims = &dims[n_batch + n_group1..n_batch + n_group1 + n_group2];
    let group2_strides = &strides[n_batch + n_group1..n_batch + n_group1 + n_group2];

    let fused1 = try_fuse_group(group1_dims, group1_strides);
    let fused2 = try_fuse_group(group2_dims, group2_strides);

    let m: usize = group1_dims.iter().product::<usize>().max(1);

    if fused1.is_some() && fused2.is_some() {
        let (_, rs) = fused1.unwrap();
        let (_, cs) = fused2.unwrap();
        Ok(ContiguousOperandMut {
            ptr: view.as_mut_ptr(),
            row_stride: rs,
            col_stride: cs,
            batch_strides: strides[..n_batch].to_vec(),
            writeback: None,
            _buf: None,
        })
    } else {
        let mut buf = alloc_batched_col_major(&dims, n_batch);
        if beta != T::zero() {
            strided_kernel::copy_into(&mut buf.view_mut(), &view.as_view())?;
        }
        let ptr = buf.view_mut().as_mut_ptr();
        let batch_strides = buf.strides()[..n_batch].to_vec();
        // SAFETY: We need to store the StridedViewMut for writeback.
        // This requires splitting the borrow. We'll use a raw pointer approach
        // and reconstruct the view for writeback in finalize().
        // For now, store the writeback info as dims/strides/ptr.
        Ok(ContiguousOperandMut {
            ptr,
            row_stride: if m == 0 { 0 } else { 1 },
            col_stride: m as isize,
            batch_strides,
            writeback: Some(unsafe { steal_view_mut(view) }),
            _buf: Some(buf),
        })
    }
}

/// Prepare an owned output array for GEMM.
///
/// If non-contiguous, copies to a col-major buffer (if beta != 0).
/// No writeback is needed — the GEMM result stays in the buffer.
pub fn prepare_output_owned<T: Scalar>(
    arr: StridedArray<T>,
    n_batch: usize,
    n_group1: usize,
    n_group2: usize,
    beta: T,
) -> crate::Result<ContiguousOperandMut<'static, T>> {
    let dims = arr.dims().to_vec();
    let strides = arr.strides().to_vec();

    let group1_dims = &dims[n_batch..n_batch + n_group1];
    let group1_strides = &strides[n_batch..n_batch + n_group1];
    let group2_dims = &dims[n_batch + n_group1..n_batch + n_group1 + n_group2];
    let group2_strides = &strides[n_batch + n_group1..n_batch + n_group1 + n_group2];

    let fused1 = try_fuse_group(group1_dims, group1_strides);
    let fused2 = try_fuse_group(group2_dims, group2_strides);

    let m: usize = group1_dims.iter().product::<usize>().max(1);

    if fused1.is_some() && fused2.is_some() {
        let (_, rs) = fused1.unwrap();
        let (_, cs) = fused2.unwrap();
        let batch_strides = strides[..n_batch].to_vec();
        let mut arr = arr;
        let ptr = arr.view_mut().as_mut_ptr();
        Ok(ContiguousOperandMut {
            ptr,
            row_stride: rs,
            col_stride: cs,
            batch_strides,
            writeback: None,
            _buf: Some(arr),
        })
    } else {
        let mut buf = alloc_batched_col_major(&dims, n_batch);
        if beta != T::zero() {
            strided_kernel::copy_into(&mut buf.view_mut(), &arr.view())?;
        }
        let ptr = buf.view_mut().as_mut_ptr();
        let batch_strides = buf.strides()[..n_batch].to_vec();
        Ok(ContiguousOperandMut {
            ptr,
            row_stride: if m == 0 { 0 } else { 1 },
            col_stride: m as isize,
            batch_strides,
            writeback: None, // No writeback for owned C!
            _buf: Some(buf),
        })
    }
}

/// After GEMM: write results back to original C if needed.
///
/// For borrowed non-contiguous C: copies buffer → original strided layout.
/// For owned C or contiguous C: no-op.
impl<'a, T: Scalar> ContiguousOperandMut<'a, T> {
    pub fn finalize(self) -> crate::Result<()> {
        if let (Some(mut dest), Some(ref buf)) = (self.writeback, &self._buf) {
            strided_kernel::copy_into(&mut dest, &buf.view())?;
        }
        Ok(())
    }
}
```

Note: The `steal_view_mut` helper needs careful unsafe code to reborrow the `StridedViewMut` for deferred writeback. An alternative is to restructure so that `finalize` takes the original `&mut StridedViewMut` as a parameter instead of storing it. **During implementation, choose the approach that avoids unsafe if possible** — e.g. have `finalize` take a `&mut StridedViewMut` parameter, or return the buffer for the caller to do the copy-back.

**Step 4: Run tests**

Run: `cargo test -p strided-einsum2 test_output_`
Expected: PASS

**Step 5: Commit**

```bash
git add strided-einsum2/src/contiguous.rs
git commit -m "feat: add output preparation with finalize for C operand"
```

---

### Task 5: Refactor bgemm_faer to accept ContiguousOperand types

**Files:**
- Modify: `strided-einsum2/src/bgemm_faer.rs:19-210`
- Modify: `strided-einsum2/src/contiguous.rs` (expose needed fields or add methods)

**Step 1: Verify existing tests pass before refactoring**

Run: `cargo test -p strided-einsum2`
Expected: all PASS

**Step 2: Add a new `bgemm_contiguous_into` function**

Add a new function in `bgemm_faer.rs` that accepts pre-contiguous operands. This runs alongside the old `bgemm_strided_into` (which we'll deprecate later).

```rust
/// Batched GEMM on pre-contiguous operands.
///
/// Operands must already have contiguous inner dimensions (prepared via
/// `prepare_input_view`/`prepare_input_owned` and `prepare_output_*`).
pub fn bgemm_contiguous_into<T>(
    c: &mut ContiguousOperandMut<'_, T>,
    a: &ContiguousOperand<T>,
    b: &ContiguousOperand<T>,
    batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    beta: T,
) -> strided_view::Result<()>
where
    T: ComplexField + Copy + strided_view::ElementOpApply + Send + Sync
        + std::ops::Mul<Output = T> + std::ops::Add<Output = T>
        + num_traits::Zero + num_traits::One + PartialEq,
{
    // This function contains the batch loop + faer matmul call,
    // moved from the bottom half of bgemm_strided_into (lines 155-209).
    // The copy logic is gone — operands are already contiguous.
    // ...
}
```

The body is the batch iteration loop from `bgemm_faer.rs:155-209`, using the strides from `ContiguousOperand` fields instead of local variables.

**Step 3: Run all existing bgemm_faer tests**

Run: `cargo test -p strided-einsum2 test_faer_bgemm`
Expected: PASS (old function untouched)

**Step 4: Add tests for the new function**

Write a test that creates `ContiguousOperand` via `prepare_input_view` and `ContiguousOperandMut` via `prepare_output_view`, then calls `bgemm_contiguous_into`.

```rust
#[test]
fn test_bgemm_contiguous_basic() {
    // 2x2 matmul through contiguous operand path
    let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
        [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
    });
    let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
        [[5.0, 6.0], [7.0, 8.0]][idx[0]][idx[1]]
    });
    let mut c = StridedArray::<f64>::row_major(&[2, 2]);

    let a_op = prepare_input_view(&a.view(), 0, 1, 1, false).unwrap();
    let b_op = prepare_input_view(&b.view(), 0, 1, 1, false).unwrap();
    let mut c_op = prepare_output_view(&mut c.view_mut(), 0, 1, 1, 0.0).unwrap();

    bgemm_contiguous_into(&mut c_op, &a_op, &b_op, &[], 2, 2, 2, 1.0, 0.0).unwrap();
    c_op.finalize().unwrap();

    assert_eq!(c.get(&[0, 0]), 19.0);
    assert_eq!(c.get(&[1, 1]), 50.0);
}
```

**Step 5: Run tests**

Run: `cargo test -p strided-einsum2 test_bgemm_contiguous`
Expected: PASS

**Step 6: Commit**

```bash
git add strided-einsum2/src/bgemm_faer.rs strided-einsum2/src/contiguous.rs
git commit -m "feat: add bgemm_contiguous_into accepting pre-contiguous operands"
```

---

### Task 6: Add IntoContiguousView trait and implementations

**Files:**
- Modify: `strided-einsum2/src/contiguous.rs`

**Step 1: Define the trait**

```rust
/// Trait for input operands that can provide a contiguous view for GEMM.
///
/// Implemented for `&StridedView` (copies if non-contiguous) and
/// `StridedArray` (transfers ownership, copies only if non-contiguous).
pub trait IntoContiguousInput<T: Scalar> {
    fn prepare_input(
        self,
        n_batch: usize,
        n_group1: usize,
        n_group2: usize,
        conj: bool,
    ) -> crate::Result<ContiguousOperand<T>>;
}

impl<'a, T: Scalar> IntoContiguousInput<T> for &'a StridedView<'a, T> {
    fn prepare_input(
        self, n_batch: usize, n_group1: usize, n_group2: usize, conj: bool,
    ) -> crate::Result<ContiguousOperand<T>> {
        prepare_input_view(self, n_batch, n_group1, n_group2, conj)
    }
}

impl<T: Scalar> IntoContiguousInput<T> for StridedArray<T> {
    fn prepare_input(
        self, n_batch: usize, n_group1: usize, n_group2: usize, conj: bool,
    ) -> crate::Result<ContiguousOperand<T>> {
        prepare_input_owned(self, n_batch, n_group1, n_group2, conj)
    }
}
```

**Step 2: Define the output trait**

```rust
/// Trait for output operands.
pub trait IntoContiguousOutput<T: Scalar> {
    fn prepare_output(
        self,
        n_batch: usize,
        n_group1: usize,
        n_group2: usize,
        beta: T,
    ) -> crate::Result<ContiguousOperandMut<'static, T>>;

    /// After GEMM, finalize (copy-back if needed).
    fn finalize_output(operand: ContiguousOperandMut<'_, T>, dest: &mut Self) -> crate::Result<()>;
}
```

Note: The exact trait design for the output may need iteration during implementation. The key constraint is that `StridedViewMut` (borrowed C) needs copy-back while `StridedArray` (owned C) does not. The implementor should choose whatever approach compiles cleanly — this might mean keeping the output as standalone functions rather than a trait, or using a different trait shape.

**Step 3: Run all tests**

Run: `cargo test -p strided-einsum2`
Expected: PASS

**Step 4: Commit**

```bash
git add strided-einsum2/src/contiguous.rs
git commit -m "feat: add IntoContiguousInput trait with borrowed/owned impls"
```

---

### Task 7: Refactor einsum2_into to use generic inputs

**Files:**
- Modify: `strided-einsum2/src/lib.rs:154-255`

This is the core API change. Replace concrete input types with trait bounds.

**Step 1: Verify all existing tests pass**

Run: `cargo test -p strided-einsum2`
Expected: PASS

**Step 2: Change the signature**

Change `einsum2_into` signature from:

```rust
pub fn einsum2_into<T: Scalar, OpA, OpB, ID: AxisId>(
    c: StridedViewMut<T>,
    a: &StridedView<T, OpA>,
    b: &StridedView<T, OpB>,
    ...
```

To accept generic inputs. The exact approach depends on how the `ElementOp` (conj) information is conveyed. Two options:

**Option A: Separate the Op-stripping into the caller**
Keep the current approach where `einsum2_into` accepts pre-stripped views but add an overload:

```rust
pub fn einsum2_into<T: Scalar, A, B, ID: AxisId>(
    c: StridedViewMut<T>,
    a: A,
    b: B,
    ic: &[ID], ia: &[ID], ib: &[ID],
    alpha: T, beta: T,
) -> Result<()>
where
    A: IntoContiguousInput<T>,
    B: IntoContiguousInput<T>,
```

**Option B: Keep the existing function signature, add a new `einsum2_into_generic`**
This is less disruptive to the API surface.

The implementor should start with **Option A** (replacing the signature). Since `&StridedView<'_, T, Op>` where `Op != Identity` won't impl `IntoContiguousInput<T>`, the trait impl needs to handle Op stripping.

Actually, the cleanest approach is: keep `einsum2_into` doing the trace reduction and Op stripping as it does today, then pass the stripped views (which are `StridedView<T>` with `Op=Identity`) to an internal function that uses `IntoContiguousInput`. This way the public API change is minimal.

**The implementation should:**
1. Keep the existing `einsum2_into` signature for backward compatibility
2. Extract the GEMM-dispatch portion (lines 208-252) into a new internal function `einsum2_gemm_dispatch` that accepts generic inputs
3. Have `einsum2_into` call the new function after trace reduction and Op stripping

**Step 3: Run all tests after refactoring**

Run: `cargo test -p strided-einsum2`
Expected: all 31+ tests PASS

**Step 4: Commit**

```bash
git add strided-einsum2/src/lib.rs
git commit -m "refactor: extract GEMM dispatch to use ContiguousOperand types"
```

---

### Task 8: Add public API for owned inputs

**Files:**
- Modify: `strided-einsum2/src/lib.rs`

**Step 1: Add `einsum2_into_owned` or modify `einsum2_into` to accept owned arrays**

Add a new public function (or modify the existing one with trait bounds) that accepts `StridedArray<T>` inputs:

```rust
/// Binary einsum accepting owned inputs for zero-copy optimization.
///
/// Same as `einsum2_into` but accepts owned `StridedArray` inputs.
/// When inputs are non-contiguous, ownership avoids the copy-back for C.
pub fn einsum2_into_owned<T: Scalar, ID: AxisId>(
    c: StridedViewMut<T>,
    a: StridedArray<T>,
    b: StridedArray<T>,
    ic: &[ID], ia: &[ID], ib: &[ID],
    alpha: T, beta: T,
    conj_a: bool, conj_b: bool,
) -> Result<()>
```

Or, if the refactoring in Task 7 produced a clean generic internal function, simply expose it.

**Step 2: Add tests for the new API**

```rust
#[test]
fn test_einsum2_owned_inputs() {
    let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
        [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
    });
    let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
        [[5.0, 6.0], [7.0, 8.0]][idx[0]][idx[1]]
    });
    let mut c = StridedArray::<f64>::row_major(&[2, 2]);

    // Call with owned inputs
    einsum2_into_owned(
        c.view_mut(), a, b,
        &['i', 'k'], &['i', 'j'], &['j', 'k'],
        1.0, 0.0, false, false,
    ).unwrap();

    assert_eq!(c.get(&[0, 0]), 19.0);
    assert_eq!(c.get(&[1, 1]), 50.0);
}
```

**Step 3: Run all tests**

Run: `cargo test -p strided-einsum2`
Expected: PASS

**Step 4: Commit**

```bash
git add strided-einsum2/src/lib.rs
git commit -m "feat: add einsum2_into_owned accepting owned StridedArray inputs"
```

---

### Task 9: Integrate with strided-opteinsum's eval_pair

**Files:**
- Modify: `strided-opteinsum/src/expr.rs:119-183`

**Step 1: Verify existing opteinsum tests pass**

Run: `cargo test -p strided-opteinsum`
Expected: PASS

**Step 2: Modify eval_pair to pass owned operands when available**

Change `eval_pair` to consume operands by value (not by reference). Since `eval_pair` is called from `eval_node` where operands are already moved out of `Option` slots, this is natural.

The key change: when an `EinsumOperand` contains `StridedData::Owned`, extract the `StridedArray` and pass it to `einsum2_into_owned`. When it contains `StridedData::View`, pass the view as before via `einsum2_into`.

```rust
fn eval_pair(
    left: EinsumOperand<'_>,      // take by value instead of &
    left_ids: &[char],
    right: EinsumOperand<'_>,     // take by value instead of &
    right_ids: &[char],
    output_ids: &[char],
) -> crate::Result<EinsumOperand<'static>> {
    match (left, right) {
        (EinsumOperand::F64(ld), EinsumOperand::F64(rd)) => {
            // ... compute dim_map and out_dims ...
            let mut c_arr = StridedArray::<f64>::col_major(&out_dims);

            match (ld, rd) {
                (StridedData::Owned(a), StridedData::Owned(b)) => {
                    einsum2_into_owned(c_arr.view_mut(), a, b, ...)?;
                }
                (StridedData::Owned(a), StridedData::View(b)) => {
                    // ... mixed path ...
                }
                // etc.
            }
            Ok(EinsumOperand::F64(StridedData::Owned(c_arr)))
        }
        // ... C64 branch similarly ...
    }
}
```

**Step 3: Update all call sites of eval_pair**

In `eval_node` and `execute_nested`, change from `eval_pair(&left, ..., &right, ...)` to `eval_pair(left, ..., right, ...)`.

Call sites:
- `expr.rs:280` in `execute_nested`
- `expr.rs:353` in `eval_node`

**Step 4: Run all tests**

Run: `cargo test -p strided-opteinsum`
Expected: PASS

Run: `cargo test` (all crates)
Expected: PASS

**Step 5: Commit**

```bash
git add strided-opteinsum/src/expr.rs
git commit -m "feat: pass owned operands through eval_pair to einsum2_into"
```

---

### Task 10: Final verification and cleanup

**Files:**
- All modified files

**Step 1: Run the full test suite**

Run: `cargo test`
Expected: all tests PASS

**Step 2: Check formatting**

Run: `cargo fmt --check`
If fails: `cargo fmt`

**Step 3: Remove dead code warnings**

Clean up any `#[allow(dead_code)]` annotations that are no longer needed, and remove the old `bgemm_strided_into` copy logic if it's fully replaced.

**Step 4: Run tests once more**

Run: `cargo test`
Expected: PASS

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: cleanup dead code and formatting"
```

---

## Execution Notes

- Tasks 1-6 build the infrastructure (types, traits, functions)
- Task 7 is the critical refactoring of `bgemm_faer.rs`
- Task 8 exposes the public API
- Task 9 integrates with opteinsum
- Task 10 is cleanup

**Lifetimes are the hardest part.** The `ContiguousOperandMut` needs to store either a borrowed `StridedViewMut` (for writeback) or an owned `StridedArray`. The `finalize` method's lifetime must work with both. If lifetime issues arise, the fallback is to have `finalize` take the original `&mut StridedViewMut` as a separate parameter rather than storing it in the struct.

**The old `bgemm_strided_into` should be kept** alongside the new `bgemm_contiguous_into` until all call sites are migrated. Then it can be deprecated/removed.
