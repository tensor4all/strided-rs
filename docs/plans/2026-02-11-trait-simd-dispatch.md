# Trait-Based SIMD Dispatch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace runtime `TypeId`-based SIMD dispatch with compile-time trait dispatch, eliminating `unsafe transmute_copy` and making the SIMD layer extensible.

**Architecture:** Three changes: (1) Add `IS_IDENTITY` const to `ElementOp` to replace `TypeId::of::<Op>()` checks. (2) Introduce a `MaybeSimdOps` trait with default no-op `try_simd_sum`/`try_simd_dot` methods, implemented with SIMD kernels for f32/f64 when `simd` feature is enabled. (3) Rewrite `sum()`/`dot()` to use trait dispatch via `MaybeSimdOps` — no TypeId, no transmute_copy. The existing SIMD kernel code (4-way unrolled pulp loops) stays unchanged.

**Tech Stack:** Rust, pulp (SIMD), strided-kernel crate, strided-view crate

**Scope note:** `sum`/`dot` are not imported by downstream crates (strided-opteinsum, strided-einsum2), so adding the `MaybeSimdOps` bound is safe.

---

## Task 1: Add `IS_IDENTITY` const to `ElementOp` trait

**Files:**
- Modify: `strided-view/src/element_op.rs`

**Step 1: Add the const to the trait and Identity impl**

In `strided-view/src/element_op.rs`, add `IS_IDENTITY` to the `ElementOp` trait definition (line 25):

```rust
pub trait ElementOp: Copy + Default + 'static {
    /// Whether this operation is the identity (no-op).
    const IS_IDENTITY: bool = false;

    /// Apply the operation to a value.
    fn apply<T: ElementOpApply>(value: T) -> T;

    // ... rest unchanged
}
```

And in the `impl ElementOp for Identity` block (line 109), add the override:

```rust
impl ElementOp for Identity {
    const IS_IDENTITY: bool = true;

    #[inline(always)]
    fn apply<T: ElementOpApply>(value: T) -> T {
        value
    }
    // ... rest unchanged
}
```

No changes needed for `Conj`, `Transpose`, `Adjoint` — they inherit the default `false`.

**Step 2: Run tests**

Run: `cargo test -p strided-view`
Expected: All pass.

**Step 3: Commit**

```bash
git add strided-view/src/element_op.rs
git commit -m "refactor: add IS_IDENTITY const to ElementOp trait"
```

---

## Task 2: Replace `TypeId::of::<Op>()` in `copy_into` with `Op::IS_IDENTITY`

**Files:**
- Modify: `strided-kernel/src/ops_view.rs` (line 203)

**Step 1: Replace the TypeId check**

Change line 203 from:
```rust
if TypeId::of::<Op>() == TypeId::of::<crate::Identity>() {
```
to:
```rust
if Op::IS_IDENTITY {
```

Do NOT remove the `use std::any::TypeId` import yet — `sum`/`dot` still need it until Task 4/5.

**Step 2: Run tests**

Run: `cargo test -p strided-kernel`
Expected: All pass. Tests `test_copy_into*` and `test_copy_into_conj*` exercise both paths.

**Step 3: Commit**

```bash
git add strided-kernel/src/ops_view.rs
git commit -m "refactor: use Op::IS_IDENTITY instead of TypeId in copy_into"
```

---

## Task 3: Create `MaybeSimdOps` trait in simd.rs

**Files:**
- Modify: `strided-kernel/src/simd.rs`

**Step 1: Restructure simd.rs**

Replace the entire contents of `strided-kernel/src/simd.rs` with:

```rust
#[inline(always)]
pub(crate) fn dispatch<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(feature = "simd")]
    {
        pulp::Arch::new().dispatch(f)
    }
    #[cfg(not(feature = "simd"))]
    {
        f()
    }
}

#[inline(always)]
pub(crate) fn dispatch_if_large<R>(len: usize, f: impl FnOnce() -> R) -> R {
    if len >= 64 {
        dispatch(f)
    } else {
        f()
    }
}

/// Trait for types that may have SIMD-accelerated sum/dot operations.
///
/// Default implementations return `None` (no SIMD available).
/// f32/f64 override these with SIMD kernels when the `simd` feature is enabled.
pub trait MaybeSimdOps: Copy + Sized {
    fn try_simd_sum(_src: &[Self]) -> Option<Self> {
        None
    }
    fn try_simd_dot(_a: &[Self], _b: &[Self]) -> Option<Self> {
        None
    }
}

// Default (no-op) impls for integer types and Complex
macro_rules! impl_no_simd {
    ($($t:ty),*) => {
        $(impl MaybeSimdOps for $t {})*
    };
}

impl_no_simd!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

impl<T: num_traits::Num + Copy + Clone + std::ops::Neg<Output = T>> MaybeSimdOps
    for num_complex::Complex<T>
{
}

// f32/f64: SIMD-accelerated when feature enabled, no-op otherwise
#[cfg(not(feature = "simd"))]
impl MaybeSimdOps for f32 {}

#[cfg(not(feature = "simd"))]
impl MaybeSimdOps for f64 {}

#[cfg(feature = "simd")]
mod simd_impls {
    use super::MaybeSimdOps;
    use pulp::{Simd, WithSimd};

    impl MaybeSimdOps for f32 {
        fn try_simd_sum(src: &[f32]) -> Option<f32> {
            struct Sum<'a>(&'a [f32]);
            impl WithSimd for Sum<'_> {
                type Output = f32;
                #[inline(always)]
                fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                    let (head, tail) = S::as_simd_f32s(self.0);
                    let mut acc0 = simd.splat_f32s(0.0);
                    let mut acc1 = simd.splat_f32s(0.0);
                    let mut acc2 = simd.splat_f32s(0.0);
                    let mut acc3 = simd.splat_f32s(0.0);
                    let mut i = 0usize;
                    while i + 4 <= head.len() {
                        acc0 = simd.add_f32s(acc0, head[i]);
                        acc1 = simd.add_f32s(acc1, head[i + 1]);
                        acc2 = simd.add_f32s(acc2, head[i + 2]);
                        acc3 = simd.add_f32s(acc3, head[i + 3]);
                        i += 4;
                    }
                    for &v in &head[i..] {
                        acc0 = simd.add_f32s(acc0, v);
                    }
                    let acc =
                        simd.add_f32s(simd.add_f32s(acc0, acc1), simd.add_f32s(acc2, acc3));
                    let mut sum = simd.reduce_sum_f32s(acc);
                    for &x in tail {
                        sum += x;
                    }
                    sum
                }
            }
            Some(pulp::Arch::new().dispatch(Sum(src)))
        }

        fn try_simd_dot(a: &[f32], b: &[f32]) -> Option<f32> {
            struct Dot<'a> {
                a: &'a [f32],
                b: &'a [f32],
            }
            impl WithSimd for Dot<'_> {
                type Output = f32;
                #[inline(always)]
                fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                    debug_assert_eq!(self.a.len(), self.b.len());
                    let (a_head, a_tail) = S::as_simd_f32s(self.a);
                    let (b_head, b_tail) = S::as_simd_f32s(self.b);
                    debug_assert_eq!(a_head.len(), b_head.len());
                    debug_assert_eq!(a_tail.len(), b_tail.len());
                    let mut acc0 = simd.splat_f32s(0.0);
                    let mut acc1 = simd.splat_f32s(0.0);
                    let mut acc2 = simd.splat_f32s(0.0);
                    let mut acc3 = simd.splat_f32s(0.0);
                    let mut i = 0usize;
                    while i + 4 <= a_head.len() {
                        acc0 = simd.mul_add_f32s(a_head[i], b_head[i], acc0);
                        acc1 = simd.mul_add_f32s(a_head[i + 1], b_head[i + 1], acc1);
                        acc2 = simd.mul_add_f32s(a_head[i + 2], b_head[i + 2], acc2);
                        acc3 = simd.mul_add_f32s(a_head[i + 3], b_head[i + 3], acc3);
                        i += 4;
                    }
                    for j in i..a_head.len() {
                        acc0 = simd.mul_add_f32s(a_head[j], b_head[j], acc0);
                    }
                    let acc =
                        simd.add_f32s(simd.add_f32s(acc0, acc1), simd.add_f32s(acc2, acc3));
                    let mut sum = simd.reduce_sum_f32s(acc);
                    for (&x, &y) in a_tail.iter().zip(b_tail.iter()) {
                        sum += x * y;
                    }
                    sum
                }
            }
            Some(pulp::Arch::new().dispatch(Dot { a, b }))
        }
    }

    impl MaybeSimdOps for f64 {
        fn try_simd_sum(src: &[f64]) -> Option<f64> {
            struct Sum<'a>(&'a [f64]);
            impl WithSimd for Sum<'_> {
                type Output = f64;
                #[inline(always)]
                fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                    let (head, tail) = S::as_simd_f64s(self.0);
                    let mut acc0 = simd.splat_f64s(0.0);
                    let mut acc1 = simd.splat_f64s(0.0);
                    let mut acc2 = simd.splat_f64s(0.0);
                    let mut acc3 = simd.splat_f64s(0.0);
                    let mut i = 0usize;
                    while i + 4 <= head.len() {
                        acc0 = simd.add_f64s(acc0, head[i]);
                        acc1 = simd.add_f64s(acc1, head[i + 1]);
                        acc2 = simd.add_f64s(acc2, head[i + 2]);
                        acc3 = simd.add_f64s(acc3, head[i + 3]);
                        i += 4;
                    }
                    for &v in &head[i..] {
                        acc0 = simd.add_f64s(acc0, v);
                    }
                    let acc =
                        simd.add_f64s(simd.add_f64s(acc0, acc1), simd.add_f64s(acc2, acc3));
                    let mut sum = simd.reduce_sum_f64s(acc);
                    for &x in tail {
                        sum += x;
                    }
                    sum
                }
            }
            Some(pulp::Arch::new().dispatch(Sum(src)))
        }

        fn try_simd_dot(a: &[f64], b: &[f64]) -> Option<f64> {
            struct Dot<'a> {
                a: &'a [f64],
                b: &'a [f64],
            }
            impl WithSimd for Dot<'_> {
                type Output = f64;
                #[inline(always)]
                fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                    debug_assert_eq!(self.a.len(), self.b.len());
                    let (a_head, a_tail) = S::as_simd_f64s(self.a);
                    let (b_head, b_tail) = S::as_simd_f64s(self.b);
                    debug_assert_eq!(a_head.len(), b_head.len());
                    debug_assert_eq!(a_tail.len(), b_tail.len());
                    let mut acc0 = simd.splat_f64s(0.0);
                    let mut acc1 = simd.splat_f64s(0.0);
                    let mut acc2 = simd.splat_f64s(0.0);
                    let mut acc3 = simd.splat_f64s(0.0);
                    let mut i = 0usize;
                    while i + 4 <= a_head.len() {
                        acc0 = simd.mul_add_f64s(a_head[i], b_head[i], acc0);
                        acc1 = simd.mul_add_f64s(a_head[i + 1], b_head[i + 1], acc1);
                        acc2 = simd.mul_add_f64s(a_head[i + 2], b_head[i + 2], acc2);
                        acc3 = simd.mul_add_f64s(a_head[i + 3], b_head[i + 3], acc3);
                        i += 4;
                    }
                    for j in i..a_head.len() {
                        acc0 = simd.mul_add_f64s(a_head[j], b_head[j], acc0);
                    }
                    let acc =
                        simd.add_f64s(simd.add_f64s(acc0, acc1), simd.add_f64s(acc2, acc3));
                    let mut sum = simd.reduce_sum_f64s(acc);
                    for (&x, &y) in a_tail.iter().zip(b_tail.iter()) {
                        sum += x * y;
                    }
                    sum
                }
            }
            Some(pulp::Arch::new().dispatch(Dot { a, b }))
        }
    }
}
```

**Key design points:**
- `MaybeSimdOps` is a `pub` trait (needed for `sum`/`dot` public function bounds)
- Default methods return `None` — types without SIMD get the generic `reduce` path
- f32/f64 impls wrap existing kernel code with `Some(...)`
- No `transmute_copy` anywhere

**Step 2: Run tests**

Run: `cargo test -p strided-kernel`
Expected: All pass (ops_view.rs still uses old code; the trait exists but isn't called yet).

**Step 3: Commit**

```bash
git add strided-kernel/src/simd.rs
git commit -m "refactor: introduce MaybeSimdOps trait with f32/f64 SIMD impls"
```

---

## Task 4: Rewrite `sum()` to use `MaybeSimdOps` trait dispatch

**Files:**
- Modify: `strided-kernel/src/ops_view.rs` (lines 629-675)
- Modify: `strided-kernel/src/lib.rs` (re-export `MaybeSimdOps`)

**Step 1: Re-export `MaybeSimdOps` from lib.rs**

In `strided-kernel/src/lib.rs`, add to the existing re-exports:

```rust
pub use simd::MaybeSimdOps;
```

(Add after line 61, near the `mod simd;` declaration, or in a dedicated section.)

**Step 2: Rewrite the `sum` function**

Replace the `sum` function (lines 629-675) with:

```rust
/// Sum all elements: `sum(src)`.
pub fn sum<
    T: Copy + ElementOpApply + Zero + Add<Output = T> + MaybeSendSync + simd::MaybeSimdOps,
    Op: ElementOp,
>(
    src: &StridedView<T, Op>,
) -> Result<T> {
    // SIMD fast path: contiguous Identity view with SIMD support
    if Op::IS_IDENTITY {
        if let Some(layout) = same_contiguous_layout(src.dims(), &[src.strides()]) {
            let len = total_len(src.dims());
            let src_slice = unsafe { std::slice::from_raw_parts(src.ptr(), len) };

            #[cfg(feature = "parallel")]
            if len > MINTHREADLENGTH {
                if let Some(result) = parallel_simd_sum(src_slice) {
                    return Ok(result);
                }
            }

            if let Some(result) = T::try_simd_sum(src_slice) {
                return Ok(result);
            }
        }
    }
    reduce(src, |x| x, |a, b| a + b, T::zero())
}
```

Also add a parallel SIMD sum helper (only when `parallel` feature is enabled):

```rust
#[cfg(feature = "parallel")]
fn parallel_simd_sum<T: Copy + Zero + Add<Output = T> + simd::MaybeSimdOps + Send + Sync>(
    src: &[T],
) -> Option<T> {
    use rayon::prelude::*;
    // Check that T has SIMD support by trying a small slice
    if T::try_simd_sum(&[]).is_none() {
        return None;
    }
    let nthreads = rayon::current_num_threads();
    let chunk_size = (src.len() + nthreads - 1) / nthreads;
    let result = src
        .par_chunks(chunk_size)
        .map(|chunk| T::try_simd_sum(chunk).unwrap())
        .reduce(|| T::zero(), |a, b| a + b);
    Some(result)
}
```

Remove the `'static` bound from `sum` since it was only needed for `TypeId`.

**Step 3: Run tests**

Run: `cargo test -p strided-kernel`
Expected: All pass. Tests `test_sum_f32_contiguous`, `test_sum_permuted`, `test_sum_col_major_small` exercise the affected paths.

**Step 4: Commit**

```bash
git add strided-kernel/src/ops_view.rs strided-kernel/src/lib.rs
git commit -m "refactor: use MaybeSimdOps trait dispatch in sum()"
```

---

## Task 5: Rewrite `dot()` to use `MaybeSimdOps` trait dispatch

**Files:**
- Modify: `strided-kernel/src/ops_view.rs` (lines 677-750)

**Step 1: Rewrite the `dot` function**

Replace the `dot` function with:

```rust
/// Dot product: `sum(a[i] * b[i])`.
pub fn dot<
    T: Copy + ElementOpApply + Zero + Mul<Output = T> + Add<Output = T> + MaybeSendSync + simd::MaybeSimdOps,
    OpA: ElementOp,
    OpB: ElementOp,
>(
    a: &StridedView<T, OpA>,
    b: &StridedView<T, OpB>,
) -> Result<T> {
    ensure_same_shape(a.dims(), b.dims())?;

    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let a_strides = a.strides();
    let b_strides = b.strides();
    let a_dims = a.dims();

    if same_contiguous_layout(a_dims, &[a_strides, b_strides]).is_some() {
        let len = total_len(a_dims);

        // SIMD fast path: both contiguous, both Identity ops
        if OpA::IS_IDENTITY && OpB::IS_IDENTITY {
            let sa = unsafe { std::slice::from_raw_parts(a_ptr, len) };
            let sb = unsafe { std::slice::from_raw_parts(b_ptr, len) };
            if let Some(result) = T::try_simd_dot(sa, sb) {
                return Ok(result);
            }
        }

        // Generic contiguous fast path
        let sa = unsafe { std::slice::from_raw_parts(a_ptr, len) };
        let sb = unsafe { std::slice::from_raw_parts(b_ptr, len) };
        let mut acc = T::zero();
        simd::dispatch_if_large(len, || {
            for i in 0..len {
                acc = acc + OpA::apply(sa[i]) * OpB::apply(sb[i]);
            }
        });
        return Ok(acc);
    }

    let strides_list: [&[isize]; 2] = [a_strides, b_strides];

    let (fused_dims, ordered_strides, plan) =
        build_plan_fused(a_dims, &strides_list, None, std::mem::size_of::<T>());

    let mut acc = T::zero();
    let initial_offsets = vec![0isize; ordered_strides.len()];
    for_each_inner_block_preordered(
        &fused_dims,
        &plan.block,
        &ordered_strides,
        &initial_offsets,
        |offsets, len, strides| {
            acc = unsafe {
                inner_loop_dot::<T, OpA, OpB>(
                    a_ptr.offset(offsets[0]),
                    strides[0],
                    b_ptr.offset(offsets[1]),
                    strides[1],
                    len,
                    acc,
                )
            };
            Ok(())
        },
    )?;

    Ok(acc)
}
```

**Key changes from original:**
- Replace `#[cfg(feature = "simd")] { if TypeId... transmute_copy }` block with `if OpA::IS_IDENTITY && OpB::IS_IDENTITY { T::try_simd_dot(...) }`
- No transmute_copy, no TypeId
- Remove `'static` bound

Note: The original `dot` didn't check `OpA::IS_IDENTITY` / `OpB::IS_IDENTITY` for the SIMD path. This was arguably a bug — applying SIMD dot on data that needs `Conj`/`Adjoint` would skip the element op. The new code gates SIMD on both ops being Identity.

**Step 2: Run tests**

Run: `cargo test -p strided-kernel`
Expected: All pass. Tests `test_dot`, `test_dot_f32_contiguous`, `test_dot_permuted`, `test_dot_mixed_layout` exercise all paths.

**Step 3: Commit**

```bash
git add strided-kernel/src/ops_view.rs
git commit -m "refactor: use MaybeSimdOps trait dispatch in dot()"
```

---

## Task 6: Clean up — remove TypeId import and dead code

**Files:**
- Modify: `strided-kernel/src/ops_view.rs` (line 14)

**Step 1: Remove the TypeId import**

Delete line 14:
```rust
use std::any::TypeId;
```

**Step 2: Verify no remaining TypeId uses**

Run: `grep -n TypeId strided-kernel/src/ops_view.rs`
Expected: No matches.

**Step 3: Run full test suite**

Run: `cargo test`
Expected: All tests across all crates pass.

Also run: `cargo test -p strided-kernel --no-default-features`
Expected: All pass (verifies the non-SIMD path works).

**Step 4: Run fmt check**

Run: `cargo fmt --check`
Expected: No formatting issues.

**Step 5: Commit**

```bash
git add strided-kernel/src/ops_view.rs
git commit -m "refactor: remove TypeId import, complete trait-based SIMD dispatch (#61)"
```

---

## Summary of Changes

| File | Change |
|------|--------|
| `strided-view/src/element_op.rs` | Add `IS_IDENTITY` const to `ElementOp` trait |
| `strided-kernel/src/simd.rs` | Replace type-suffixed functions with `MaybeSimdOps` trait |
| `strided-kernel/src/ops_view.rs` | Replace all `TypeId`/`transmute_copy` with trait dispatch |
| `strided-kernel/src/lib.rs` | Re-export `MaybeSimdOps` |

**Eliminated:**
- 5 `TypeId::of::<>()` runtime checks
- 4 `unsafe { std::mem::transmute_copy }` calls
- `use std::any::TypeId` import
- Type-suffixed `sum_f32`/`sum_f64`/`dot_f32`/`dot_f64` functions

**Added:**
- `ElementOp::IS_IDENTITY` const (compile-time)
- `MaybeSimdOps` trait with `try_simd_sum`/`try_simd_dot` (extensible)
