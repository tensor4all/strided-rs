# BLAS Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a BLAS backend to strided-einsum2 using `cblas-sys` + `blas-src`, with `cblas-inject` support for Julia/Python interop.

**Architecture:** Feature-gated BLAS backend (`bgemm_blas.rs`) that plugs into the existing `ContiguousOperand` / `bgemm_contiguous_into` pattern. Crate alias pattern (`extern crate cblas_inject as cblas_sys`) unifies code for both `blas` and `blas-inject` features. Conjugation is pre-resolved during copy-in so CBLAS always receives `CblasNoTrans`.

**Tech Stack:** `cblas-sys` 0.2, `blas-src` 0.14, `cblas-inject` 0.1, `num-complex` 0.4

**Design doc:** `docs/plans/2026-02-07-blas-backend-design.md`

---

### Task 1: Add dependencies and feature flags

**Files:**
- Modify: `strided-einsum2/Cargo.toml`

**Step 1: Add optional dependencies and feature flags**

In `strided-einsum2/Cargo.toml`, add:

```toml
[features]
default = ["faer", "faer-traits"]
blas = ["dep:cblas-sys", "dep:blas-src"]
blas-inject = ["dep:cblas-inject"]

[dependencies]
cblas-sys = { version = "0.2", optional = true }
blas-src = { version = "0.14", optional = true }
cblas-inject = { version = "0.1", optional = true }
```

**Step 2: Verify it compiles with each feature combination**

```bash
# Default (faer) — should pass
cargo check -p strided-einsum2

# blas — should pass (may warn about unused)
cargo check -p strided-einsum2 --no-default-features --features blas

# naive only — should pass
cargo check -p strided-einsum2 --no-default-features
```

**Step 3: Commit**

```bash
git add strided-einsum2/Cargo.toml
git commit -m "feat: add cblas-sys, blas-src, cblas-inject optional dependencies"
```

---

### Task 2: Add mutual exclusion and crate alias in lib.rs

**Files:**
- Modify: `strided-einsum2/src/lib.rs`

**Step 1: Add compile_error guards and crate alias at the top of lib.rs**

After the existing `#[cfg(feature = "faer")]` module declaration, add:

```rust
#[cfg(all(feature = "faer", feature = "blas"))]
compile_error!("Features `faer` and `blas` are mutually exclusive. Use one or the other.");

#[cfg(all(feature = "faer", feature = "blas-inject"))]
compile_error!("Features `faer` and `blas-inject` are mutually exclusive.");

#[cfg(all(feature = "blas", feature = "blas-inject"))]
compile_error!("Features `blas` and `blas-inject` are mutually exclusive.");

#[cfg(feature = "blas")]
extern crate cblas_sys;
#[cfg(feature = "blas-inject")]
extern crate cblas_inject as cblas_sys;

#[cfg(any(feature = "blas", feature = "blas-inject"))]
pub mod bgemm_blas;
```

**Step 2: Verify mutual exclusion works**

```bash
# This should fail with compile_error
cargo check -p strided-einsum2 --features faer,blas 2>&1 | grep "mutually exclusive"
# Expected: error about mutual exclusion
```

**Step 3: Commit**

```bash
git add strided-einsum2/src/lib.rs
git commit -m "feat: add mutual exclusion guards and crate alias for BLAS"
```

---

### Task 3: Add Scalar trait variant for BLAS features

**Files:**
- Modify: `strided-einsum2/src/lib.rs`

**Step 1: Update the Scalar trait cfg blocks**

Currently there are two blocks: `#[cfg(feature = "faer")]` and `#[cfg(not(feature = "faer"))]`.

Change the second block's cfg from `not(feature = "faer")` to `not(any(feature = "faer", feature = "blas", feature = "blas-inject"))` for the naive fallback, and add a new block for BLAS:

```rust
/// Scalar trait for BLAS backends.
/// No additional trait bounds beyond the basics — type dispatch uses BlasGemm.
#[cfg(any(feature = "blas", feature = "blas-inject"))]
pub trait Scalar:
    Copy
    + ElementOpApply
    + Send
    + Sync
    + std::ops::Mul<Output = Self>
    + std::ops::Add<Output = Self>
    + num_traits::Zero
    + num_traits::One
    + PartialEq
{
}

#[cfg(any(feature = "blas", feature = "blas-inject"))]
impl<T> Scalar for T where
    T: Copy
        + ElementOpApply
        + Send
        + Sync
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + num_traits::Zero
        + num_traits::One
        + PartialEq
{
}

/// Scalar trait fallback (no GEMM backend).
#[cfg(not(any(feature = "faer", feature = "blas", feature = "blas-inject")))]
pub trait Scalar:
    // ... same bounds as above ...
```

**Step 2: Verify all feature combos compile**

```bash
cargo check -p strided-einsum2
cargo check -p strided-einsum2 --no-default-features --features blas
cargo check -p strided-einsum2 --no-default-features
```

**Step 3: Commit**

```bash
git add strided-einsum2/src/lib.rs
git commit -m "feat: add Scalar trait variant for BLAS features"
```

---

### Task 4: Create bgemm_blas.rs with BlasGemm trait and bgemm_contiguous_into

**Files:**
- Create: `strided-einsum2/src/bgemm_blas.rs`

**Step 1: Write the BlasGemm trait and impls for f64 and Complex64**

```rust
//! CBLAS-backed batched GEMM kernel on contiguous operands.
//!
//! Uses `cblas_dgemm` / `cblas_zgemm` for matrix multiplication.
//! Operands must be pre-contiguous (prepared via `contiguous` module).
//! Conjugation must be pre-resolved (conj flags are always false).

use crate::contiguous::{ContiguousOperand, ContiguousOperandMut};
use crate::util::MultiIndex;
use crate::Scalar;
use num_complex::Complex64;

/// Type-level dispatch for CBLAS GEMM.
///
/// Each scalar type maps to the appropriate CBLAS function
/// (e.g., f64 → cblas_dgemm, Complex64 → cblas_zgemm).
pub(crate) trait BlasGemm: Sized {
    /// Call the appropriate cblas_?gemm function.
    ///
    /// # Safety
    /// Pointers must be valid for the specified dimensions and strides.
    unsafe fn gemm(
        m: i32, n: i32, k: i32,
        alpha: Self,
        a: *const Self, lda: i32,
        b: *const Self, ldb: i32,
        beta: Self,
        c: *mut Self, ldc: i32,
    );
}

impl BlasGemm for f64 {
    unsafe fn gemm(
        m: i32, n: i32, k: i32,
        alpha: f64,
        a: *const f64, lda: i32,
        b: *const f64, ldb: i32,
        beta: f64,
        c: *mut f64, ldc: i32,
    ) {
        unsafe {
            cblas_sys::cblas_dgemm(
                cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                m, n, k,
                alpha,
                a, lda,
                b, ldb,
                beta,
                c, ldc,
            );
        }
    }
}

impl BlasGemm for Complex64 {
    unsafe fn gemm(
        m: i32, n: i32, k: i32,
        alpha: Complex64,
        a: *const Complex64, lda: i32,
        b: *const Complex64, ldb: i32,
        beta: Complex64,
        c: *mut Complex64, ldc: i32,
    ) {
        unsafe {
            cblas_sys::cblas_zgemm(
                cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                m, n, k,
                &alpha as *const _ as *const _,
                a as *const _,  lda,
                b as *const _,  ldb,
                &beta as *const _ as *const _,
                c as *mut _,    ldc,
            );
        }
    }
}
```

Note on `cblas-inject` compatibility: `cblas-inject` uses `CBLAS_ORDER` instead of `CBLAS_LAYOUT`. If this causes a compilation error with `--features blas-inject`, add a thin shim:
```rust
#[cfg(feature = "blas-inject")]
use cblas_sys::CBLAS_ORDER as CBLAS_LAYOUT;
#[cfg(feature = "blas")]
use cblas_sys::CBLAS_LAYOUT;
```
Test this during implementation and adjust accordingly.

**Step 2: Write bgemm_contiguous_into**

```rust
/// Batched GEMM on pre-contiguous operands using CBLAS.
///
/// Same interface as `bgemm_faer::bgemm_contiguous_into`.
/// Conjugation must already be resolved (conj=false) in the operands.
///
/// - `batch_dims`: sizes of the batch dimensions
/// - `m`: fused lo dimension size (rows of A/C)
/// - `n`: fused ro dimension size (cols of B/C)
/// - `k`: fused sum dimension size (inner dimension)
pub fn bgemm_contiguous_into<T: Scalar + BlasGemm>(
    c: &mut ContiguousOperandMut<T>,
    a: &ContiguousOperand<T>,
    b: &ContiguousOperand<T>,
    batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    beta: T,
) -> strided_view::Result<()> {
    debug_assert!(!a.conj(), "BLAS backend: conj must be pre-resolved for A");
    debug_assert!(!b.conj(), "BLAS backend: conj must be pre-resolved for B");

    let a_batch_strides = a.batch_strides();
    let b_batch_strides = b.batch_strides();
    let c_batch_strides = c.batch_strides();

    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let c_ptr = c.ptr();

    // CBLAS col-major: A is m×k with lda = row_stride or col_stride
    // Our ContiguousOperand stores col-major: row_stride=1, col_stride=m (after copy-in)
    // For non-copied fusable cases, strides may differ.
    // lda = the stride between columns for col-major = a.col_stride()
    // But CBLAS lda is "leading dimension" = stride between consecutive columns
    // for CblasColMajor.
    let lda = a.col_stride() as i32;
    let ldb = b.col_stride() as i32;
    let ldc = c.col_stride() as i32;

    let m_i32 = m as i32;
    let n_i32 = n as i32;
    let k_i32 = k as i32;

    let mut batch_iter = MultiIndex::new(batch_dims);
    while batch_iter.next().is_some() {
        let a_off = batch_iter.offset(a_batch_strides);
        let b_off = batch_iter.offset(b_batch_strides);
        let c_off = batch_iter.offset(c_batch_strides);

        unsafe {
            T::gemm(
                m_i32, n_i32, k_i32,
                alpha,
                a_ptr.offset(a_off), lda,
                b_ptr.offset(b_off), ldb,
                beta,
                c_ptr.offset(c_off), ldc,
            );
        }
    }

    Ok(())
}
```

**Step 3: Verify it compiles**

```bash
cargo check -p strided-einsum2 --no-default-features --features blas
```

**Step 4: Commit**

```bash
git add strided-einsum2/src/bgemm_blas.rs
git commit -m "feat: add bgemm_blas.rs with BlasGemm trait and bgemm_contiguous_into"
```

---

### Task 5: Wire up contiguous.rs for BLAS features

**Files:**
- Modify: `strided-einsum2/src/contiguous.rs`

**Step 1: Extend cfg gates on prepare/alloc functions**

Currently `prepare_input_view`, `prepare_input_owned`, `prepare_output_view`, `prepare_output_owned`, and the `alloc_batched_col_major` import are gated on `#[cfg(feature = "faer")]`. Widen to include BLAS:

```rust
// Change all occurrences of:
#[cfg(feature = "faer")]
// To:
#[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
```

The `alloc_batched_col_major` import should also be updated:
```rust
#[cfg(feature = "faer")]
use crate::bgemm_faer::alloc_batched_col_major;

#[cfg(any(feature = "blas", feature = "blas-inject"))]
use crate::bgemm_blas::alloc_batched_col_major;
```

This means `alloc_batched_col_major` must also be defined in `bgemm_blas.rs` (or extracted to a shared location). The simplest approach: copy the function to `bgemm_blas.rs` with `pub(crate)` visibility (it's 30 lines, no faer dependency).

**Step 2: Add conj pre-resolution for BLAS**

In `prepare_input_view`, after the `needs_copy` branch, add conj handling for BLAS:

```rust
// In the needs_copy=true branch, after copy_into:
#[cfg(any(feature = "blas", feature = "blas-inject"))]
let conj = if conj {
    // Conjugate in-place after copy
    strided_kernel::map_into(&mut buf.view_mut(), &buf.view(), |x| {
        num_traits::Float::conj(&x)  // or ElementOpApply
    })?;
    false
} else {
    false
};

// In the needs_copy=false branch, if conj=true for BLAS:
#[cfg(any(feature = "blas", feature = "blas-inject"))]
if conj {
    // Force copy with conjugation
    let m: usize = group1_dims.iter().product::<usize>().max(1);
    let mut buf = alloc_batched_col_major(view.dims(), n_batch);
    // Copy with conjugation applied
    strided_kernel::map_into(&mut buf.view_mut(), view, |x| Conj::apply(x))?;
    let ptr = buf.view().ptr();
    let batch_strides = buf.strides()[..n_batch].to_vec();
    let row_stride = if m == 0 { 0 } else { 1isize };
    let col_stride = m as isize;
    return Ok(ContiguousOperand {
        ptr, row_stride, col_stride, batch_strides, conj: false, _buf: Some(buf),
    });
}
```

The same pattern applies to `prepare_input_owned`.

Note: The exact implementation may differ — the key requirement is that when `any(blas, blas-inject)` is active, the returned `ContiguousOperand` always has `conj=false`. Use `strided_view::Conj::apply` for the element-level conjugation.

**Step 3: Update tests cfg gate**

```rust
// Change:
#[cfg(test)]
#[cfg(feature = "faer")]
mod tests {
// To:
#[cfg(test)]
#[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
mod tests {
```

**Step 4: Verify compilation and tests**

```bash
cargo check -p strided-einsum2 --no-default-features --features blas
cargo test -p strided-einsum2 --no-default-features --features blas
```

**Step 5: Commit**

```bash
git add strided-einsum2/src/contiguous.rs strided-einsum2/src/bgemm_blas.rs
git commit -m "feat: wire up contiguous.rs for BLAS features with conj pre-resolution"
```

---

### Task 6: Wire up einsum2_gemm_dispatch for BLAS

**Files:**
- Modify: `strided-einsum2/src/lib.rs`

**Step 1: Widen cfg gate on einsum2_gemm_dispatch**

Change `#[cfg(feature = "faer")]` on `einsum2_gemm_dispatch` to:
```rust
#[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
```

**Step 2: Add BLAS dispatch branch inside the function**

After the existing faer GEMM call, add:
```rust
    // 4. GEMM
    #[cfg(feature = "faer")]
    bgemm_faer::bgemm_contiguous_into(&mut c_op, &a_op, &b_op, &batch_dims, m, n, k, alpha, beta)?;

    #[cfg(any(feature = "blas", feature = "blas-inject"))]
    bgemm_blas::bgemm_contiguous_into(&mut c_op, &a_op, &b_op, &batch_dims, m, n, k, alpha, beta)?;
```

**Step 3: Widen cfg gate on einsum2_into's GEMM dispatch call**

In `einsum2_into`, the existing code has:
```rust
    #[cfg(feature = "faer")]
    einsum2_gemm_dispatch(c, &a_view, &b_view, &plan, alpha, beta, conj_a, conj_b)?;
```
Change to:
```rust
    #[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
    einsum2_gemm_dispatch(c, &a_view, &b_view, &plan, alpha, beta, conj_a, conj_b)?;
```

Also update the `#[cfg(not(feature = "faer"))]` fallback block to:
```rust
    #[cfg(not(any(feature = "faer", feature = "blas", feature = "blas-inject")))]
```

**Step 4: Do the same for einsum2_into_owned**

Widen the `#[cfg(feature = "faer")]` gate on `einsum2_into_owned` to include blas. This function calls `bgemm_faer::bgemm_contiguous_into` — add a parallel BLAS branch.

**Step 5: Verify full test suite passes with BLAS**

```bash
cargo test -p strided-einsum2 --no-default-features --features blas
```

All existing tests (`test_matmul_ij_jk_ik`, `test_batched_matmul`, `test_complex_matmul`, `test_alpha_beta`, etc.) should pass.

**Step 6: Verify faer still works unchanged**

```bash
cargo test -p strided-einsum2
```

**Step 7: Commit**

```bash
git add strided-einsum2/src/lib.rs
git commit -m "feat: wire up einsum2_gemm_dispatch for BLAS backend"
```

---

### Task 7: Add blas-inject compatibility shim (if needed)

**Files:**
- Modify: `strided-einsum2/src/bgemm_blas.rs`

**Step 1: Test blas-inject compilation**

```bash
cargo check -p strided-einsum2 --no-default-features --features blas-inject
```

**Step 2: If CBLAS_LAYOUT vs CBLAS_ORDER mismatch, add shim**

`cblas-sys` exposes `CBLAS_LAYOUT`, `cblas-inject` exposes `CBLAS_ORDER`. If compilation fails, add at the top of `bgemm_blas.rs`:

```rust
// Compatibility shim: cblas-sys uses CBLAS_LAYOUT, cblas-inject uses CBLAS_ORDER
#[cfg(feature = "blas")]
use cblas_sys::CBLAS_LAYOUT;
#[cfg(feature = "blas-inject")]
use cblas_sys::CBLAS_ORDER as CBLAS_LAYOUT;

#[cfg(feature = "blas")]
use cblas_sys::CBLAS_TRANSPOSE;
#[cfg(feature = "blas-inject")]
use cblas_sys::CBLAS_TRANSPOSE;  // same name in both crates
```

Then update `BlasGemm` impls to use `CBLAS_LAYOUT::CblasColMajor` etc.

**Step 3: Verify**

```bash
cargo check -p strided-einsum2 --no-default-features --features blas-inject
```

**Step 4: Commit (if changes needed)**

```bash
git add strided-einsum2/src/bgemm_blas.rs
git commit -m "feat: add cblas-inject compatibility shim for CBLAS_LAYOUT/CBLAS_ORDER"
```

---

### Task 8: Run full test suite and verify all feature combos

**Files:** None (verification only)

**Step 1: Run tests for each feature combination**

```bash
# faer (default)
cargo test -p strided-einsum2

# blas
cargo test -p strided-einsum2 --no-default-features --features blas

# blas-inject (if available/installable)
cargo test -p strided-einsum2 --no-default-features --features blas-inject

# naive only
cargo test -p strided-einsum2 --no-default-features

# full workspace test with default features
cargo test
```

**Step 2: Verify mutual exclusion**

```bash
# These should all fail with compile_error
cargo check -p strided-einsum2 --features faer,blas 2>&1 | grep "mutually exclusive"
cargo check -p strided-einsum2 --features faer,blas-inject 2>&1 | grep "mutually exclusive"
cargo check -p strided-einsum2 --no-default-features --features blas,blas-inject 2>&1 | grep "mutually exclusive"
```

**Step 3: Check formatting**

```bash
cargo fmt --check
```

**Step 4: Commit any fixes**

---

### Task 9: Update strided-opteinsum for BLAS feature passthrough

**Files:**
- Modify: `strided-opteinsum/Cargo.toml`

**Step 1: Add blas feature passthrough**

`strided-opteinsum` depends on `strided-einsum2`. Add passthrough features:

```toml
[features]
default = ["strided-einsum2/faer", "strided-einsum2/faer-traits"]
blas = ["strided-einsum2/blas"]
blas-inject = ["strided-einsum2/blas-inject"]
```

**Step 2: Verify opteinsum tests pass with BLAS**

```bash
cargo test -p strided-opteinsum --no-default-features --features blas
```

**Step 3: Commit**

```bash
git add strided-opteinsum/Cargo.toml
git commit -m "feat: add BLAS feature passthrough to strided-opteinsum"
```
