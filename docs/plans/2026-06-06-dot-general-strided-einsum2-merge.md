# DotGeneral strided-einsum2 Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the reusable general dot product implementation into `strided-einsum2`, then replace tenferro's duplicated CPU GEMM/dot-general kernel with a thin adapter.

**Architecture:** `strided-einsum2` becomes the kernel crate for binary tensor contraction and `DotGeneralConfig`-style axis APIs. tenferro keeps runtime concerns only: dtype dispatch, backend selection, buffer-pool allocation, runtime cache slots, and graph/compiler passes. The graph-level `DotGeneral` decomposition stays in `tenferro-runtime`.

**Tech Stack:** Rust 2021, `strided-view`, `strided-kernel`, `strided-perm`, `strided-einsum2`, `tenferro-cpu`, `faer`, CBLAS provider features.

---

## File Structure

- Modify: `strided-rs/strided-einsum2/Cargo.toml`
  - Add `smallvec` for low-allocation dot-general planning.
- Create: `strided-rs/strided-einsum2/src/dot_general.rs`
  - Define borrowed-slice `DotGeneralConfig`.
  - Validate axis roles and shape compatibility.
  - Build a reusable axis-label plan for the existing `einsum2_dispatch`.
  - Expose `dot_general_into`, `dot_general_with_backend_into`, and internal helpers.
- Modify: `strided-rs/strided-einsum2/src/lib.rs`
  - Export the new module/types/functions.
  - Make the existing internal dispatch reusable by dot-general entry points.
  - Adjust active backend selection so BLAS can coexist with faer and be preferred when explicitly enabled.
- Modify: `strided-rs/strided-einsum2/src/backend.rs`
  - Allow compiling `FaerBackend` and `BlasBackend` in the same crate build.
  - Keep one `ActiveBackend` for legacy `einsum2_into`.
- Create: `strided-rs/strided-einsum2/tests/dot_general.rs`
  - Integration tests for matmul, batched matmul, rank-0 dot, transposed output, invalid configs, zero-sized outputs.
- Modify: `tenferro-rs/Cargo.toml`
  - Add `strided-einsum2` as a dependency pinned to the updated `strided-rs` revision after the strided change lands.
- Modify: `tenferro-rs/crates/tenferro-cpu/Cargo.toml`
  - Wire tenferro CPU features to `strided-einsum2` features without breaking `provider-src` and `provider-inject`.
- Create: `tenferro-rs/crates/tenferro-cpu/src/gemm/strided_dot.rs`
  - Convert `TypedTensorRead` inputs into `strided_view::StridedView`.
  - Allocate tenferro outputs through `BufferPool`.
  - Call `strided_einsum2::dot_general_with_backend_into`.
- Modify: `tenferro-rs/crates/tenferro-cpu/src/gemm/mod.rs`
  - Route `dot_general_faer*_cached` and `dot_general_blas*_cached` through the adapter.
  - Keep `GemmAnalysisCache` as the tenferro runtime cache wrapper during the first replacement.
  - Remove duplicated direct GEMM loops only after all tests pass.
- Modify: `tenferro-rs/crates/tenferro-cpu/src/tests/cpu_tests/dot_structural_analytic.rs`
  - Keep existing public behavior tests; add one test that confirms transposed `TensorRead` inputs stay accepted.
- Modify: `tenferro-rs/crates/tenferro-cpu/benches/dot_general_overhead.rs`
  - Compare old cache-slot behavior against the strided adapter path for direct matmul and view input cases.

---

### Task 1: Add dot-general correctness tests in strided-einsum2

**Files:**
- Create: `strided-rs/strided-einsum2/tests/dot_general.rs`

- [ ] **Step 1: Write integration tests before implementation**

Create `strided-rs/strided-einsum2/tests/dot_general.rs` with:

```rust
use strided_einsum2::{dot_general_into, DotGeneralConfig};
use strided_view::StridedArray;

fn get_col_major(data: &[f64], shape: &[usize], idx: &[usize]) -> f64 {
    let mut stride = 1usize;
    let mut offset = 0usize;
    for (&i, &dim) in idx.iter().zip(shape) {
        offset += i * stride;
        stride *= dim;
    }
    data[offset]
}

fn expected_matmul_col_major(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for j in 0..n {
        for i in 0..m {
            let mut acc = 0.0;
            for p in 0..k {
                acc += get_col_major(a, a_shape, &[i, p])
                    * get_col_major(b, b_shape, &[p, j]);
            }
            out[i + m * j] = acc;
        }
    }
    out
}

#[test]
fn dot_general_matmul_matches_col_major_reference() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let a = StridedArray::<f64>::from_vec_col_major(&[2, 3], a_data.clone()).unwrap();
    let b = StridedArray::<f64>::from_vec_col_major(&[3, 2], b_data.clone()).unwrap();
    let mut c = StridedArray::<f64>::col_major(&[2, 2]);

    dot_general_into(
        c.view_mut(),
        &a.view(),
        &b.view(),
        &DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(
        c.data(),
        expected_matmul_col_major(&a_data, &[2, 3], &b_data, &[3, 2], 2, 2, 3).as_slice()
    );
}

#[test]
fn dot_general_batched_matmul_uses_batch_trailing_output_shape() {
    let a = StridedArray::<f64>::from_fn_col_major(&[2, 3, 2], |idx| {
        (100 * idx[2] + 10 * idx[1] + idx[0] + 1) as f64
    });
    let b = StridedArray::<f64>::from_fn_col_major(&[3, 4, 2], |idx| {
        (100 * idx[2] + 10 * idx[1] + idx[0] + 1) as f64
    });
    let mut c = StridedArray::<f64>::col_major(&[2, 4, 2]);

    dot_general_into(
        c.view_mut(),
        &a.view(),
        &b.view(),
        &DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![2],
            rhs_batch_dims: vec![2],
        },
        1.0,
        0.0,
    )
    .unwrap();

    for batch in 0..2 {
        for j in 0..4 {
            for i in 0..2 {
                let mut expected = 0.0;
                for p in 0..3 {
                    expected += a[[i, p, batch]] * b[[p, j, batch]];
                }
                assert_eq!(c[[i, j, batch]], expected);
            }
        }
    }
}

#[test]
fn dot_general_inner_product_returns_rank0_scalar() {
    let a = StridedArray::<f64>::from_vec_col_major(&[3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = StridedArray::<f64>::from_vec_col_major(&[3], vec![4.0, 5.0, 6.0]).unwrap();
    let mut c = StridedArray::<f64>::col_major(&[]);

    dot_general_into(
        c.view_mut(),
        &a.view(),
        &b.view(),
        &DotGeneralConfig {
            lhs_contracting_dims: vec![0],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(c.data(), &[32.0]);
}

#[test]
fn dot_general_rejects_wrong_output_shape() {
    let a = StridedArray::<f64>::col_major(&[2, 3]);
    let b = StridedArray::<f64>::col_major(&[3, 4]);
    let mut c = StridedArray::<f64>::col_major(&[4, 2]);

    let err = dot_general_into(
        c.view_mut(),
        &a.view(),
        &b.view(),
        &DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
        1.0,
        0.0,
    )
    .unwrap_err();

    assert!(err.to_string().contains("output shape mismatch"));
}
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/strided-rs
cargo test -p strided-einsum2 --test dot_general
```

Expected: compile failure because `DotGeneralConfig` and `dot_general_into` are not defined.

- [ ] **Step 3: Commit the failing tests**

```bash
cd /Users/hiroshi/projects/tensor4all/strided-rs
git add strided-einsum2/tests/dot_general.rs
git commit -m "test: add dot-general coverage for strided-einsum2"
```

---

### Task 2: Add DotGeneralConfig and axis-label planning in strided-einsum2

**Files:**
- Modify: `strided-rs/strided-einsum2/Cargo.toml`
- Create: `strided-rs/strided-einsum2/src/dot_general.rs`
- Modify: `strided-rs/strided-einsum2/src/lib.rs`

- [ ] **Step 1: Add `smallvec` to strided-einsum2**

In `strided-rs/strided-einsum2/Cargo.toml`, add:

```toml
smallvec.workspace = true
```

- [ ] **Step 2: Add error variants**

In `strided-rs/strided-einsum2/src/lib.rs`, extend `EinsumError`:

```rust
#[error("invalid dot-general config: {0}")]
InvalidDotGeneralConfig(String),
#[error("output shape mismatch: expected {expected:?}, got {got:?}")]
OutputShapeMismatch {
    expected: Vec<usize>,
    got: Vec<usize>,
},
```

- [ ] **Step 3: Add dot-general planning module**

Create `strided-rs/strided-einsum2/src/dot_general.rs`:

```rust
use smallvec::SmallVec;

use crate::{EinsumError, Result};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct DotGeneralConfig {
    pub lhs_contracting_dims: Vec<usize>,
    pub rhs_contracting_dims: Vec<usize>,
    pub lhs_batch_dims: Vec<usize>,
    pub rhs_batch_dims: Vec<usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct DotGeneralLabels {
    pub lhs_labels: SmallVec<[usize; 8]>,
    pub rhs_labels: SmallVec<[usize; 8]>,
    pub out_labels: SmallVec<[usize; 8]>,
    pub out_shape: SmallVec<[usize; 8]>,
}

fn check_no_duplicates(dims: &[usize], label: &str) -> Result<()> {
    for (i, &dim) in dims.iter().enumerate() {
        if dims[..i].contains(&dim) {
            return Err(EinsumError::InvalidDotGeneralConfig(format!(
                "{label} contains duplicate dim {dim}"
            )));
        }
    }
    Ok(())
}

fn check_bounds(dims: &[usize], rank: usize, label: &str) -> Result<()> {
    for &dim in dims {
        if dim >= rank {
            return Err(EinsumError::InvalidDotGeneralConfig(format!(
                "{label} dim {dim} out of bounds for rank {rank}"
            )));
        }
    }
    Ok(())
}

fn free_dims(rank: usize, contracting: &[usize], batch: &[usize]) -> SmallVec<[usize; 8]> {
    (0..rank)
        .filter(|dim| !contracting.contains(dim) && !batch.contains(dim))
        .collect()
}

impl DotGeneralConfig {
    pub fn validate_dims_with_ranks(&self, lhs_rank: usize, rhs_rank: usize) -> Result<()> {
        check_bounds(&self.lhs_contracting_dims, lhs_rank, "lhs_contracting")?;
        check_bounds(&self.rhs_contracting_dims, rhs_rank, "rhs_contracting")?;
        check_bounds(&self.lhs_batch_dims, lhs_rank, "lhs_batch")?;
        check_bounds(&self.rhs_batch_dims, rhs_rank, "rhs_batch")?;
        check_no_duplicates(&self.lhs_contracting_dims, "lhs_contracting_dims")?;
        check_no_duplicates(&self.rhs_contracting_dims, "rhs_contracting_dims")?;
        check_no_duplicates(&self.lhs_batch_dims, "lhs_batch_dims")?;
        check_no_duplicates(&self.rhs_batch_dims, "rhs_batch_dims")?;
        for &dim in &self.lhs_contracting_dims {
            if self.lhs_batch_dims.contains(&dim) {
                return Err(EinsumError::InvalidDotGeneralConfig(format!(
                    "lhs dim {dim} appears in both contracting and batch dims"
                )));
            }
        }
        for &dim in &self.rhs_contracting_dims {
            if self.rhs_batch_dims.contains(&dim) {
                return Err(EinsumError::InvalidDotGeneralConfig(format!(
                    "rhs dim {dim} appears in both contracting and batch dims"
                )));
            }
        }
        if self.lhs_contracting_dims.len() != self.rhs_contracting_dims.len() {
            return Err(EinsumError::InvalidDotGeneralConfig(format!(
                "lhs/rhs contracting dim counts differ ({} vs {})",
                self.lhs_contracting_dims.len(),
                self.rhs_contracting_dims.len()
            )));
        }
        if self.lhs_batch_dims.len() != self.rhs_batch_dims.len() {
            return Err(EinsumError::InvalidDotGeneralConfig(format!(
                "lhs/rhs batch dim counts differ ({} vs {})",
                self.lhs_batch_dims.len(),
                self.rhs_batch_dims.len()
            )));
        }
        Ok(())
    }

    pub(crate) fn labels_for_shapes(
        &self,
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        out_shape: &[usize],
    ) -> Result<DotGeneralLabels> {
        self.validate_dims_with_ranks(lhs_shape.len(), rhs_shape.len())?;

        let lhs_free = free_dims(lhs_shape.len(), &self.lhs_contracting_dims, &self.lhs_batch_dims);
        let rhs_free = free_dims(rhs_shape.len(), &self.rhs_contracting_dims, &self.rhs_batch_dims);

        let mut lhs_labels = SmallVec::<[usize; 8]>::from_vec(vec![usize::MAX; lhs_shape.len()]);
        let mut rhs_labels = SmallVec::<[usize; 8]>::from_vec(vec![usize::MAX; rhs_shape.len()]);
        let mut next_label = 0usize;

        for (&lhs_dim, &rhs_dim) in self.lhs_batch_dims.iter().zip(&self.rhs_batch_dims) {
            if lhs_shape[lhs_dim] != rhs_shape[rhs_dim] {
                return Err(EinsumError::DimensionMismatch {
                    axis: format!("batch lhs {lhs_dim} rhs {rhs_dim}"),
                    dim_a: lhs_shape[lhs_dim],
                    dim_b: rhs_shape[rhs_dim],
                });
            }
            lhs_labels[lhs_dim] = next_label;
            rhs_labels[rhs_dim] = next_label;
            next_label += 1;
        }

        for &lhs_dim in &lhs_free {
            lhs_labels[lhs_dim] = next_label;
            next_label += 1;
        }
        for &rhs_dim in &rhs_free {
            rhs_labels[rhs_dim] = next_label;
            next_label += 1;
        }

        for (&lhs_dim, &rhs_dim) in self
            .lhs_contracting_dims
            .iter()
            .zip(&self.rhs_contracting_dims)
        {
            if lhs_shape[lhs_dim] != rhs_shape[rhs_dim] {
                return Err(EinsumError::DimensionMismatch {
                    axis: format!("contract lhs {lhs_dim} rhs {rhs_dim}"),
                    dim_a: lhs_shape[lhs_dim],
                    dim_b: rhs_shape[rhs_dim],
                });
            }
            lhs_labels[lhs_dim] = next_label;
            rhs_labels[rhs_dim] = next_label;
            next_label += 1;
        }

        let mut expected_out_shape = SmallVec::<[usize; 8]>::new();
        expected_out_shape.extend(lhs_free.iter().map(|&dim| lhs_shape[dim]));
        expected_out_shape.extend(rhs_free.iter().map(|&dim| rhs_shape[dim]));
        expected_out_shape.extend(self.lhs_batch_dims.iter().map(|&dim| lhs_shape[dim]));
        if expected_out_shape.as_slice() != out_shape {
            return Err(EinsumError::OutputShapeMismatch {
                expected: expected_out_shape.iter().copied().collect(),
                got: out_shape.to_vec(),
            });
        }

        let mut out_labels = SmallVec::<[usize; 8]>::new();
        out_labels.extend(lhs_free.iter().map(|&dim| lhs_labels[dim]));
        out_labels.extend(rhs_free.iter().map(|&dim| rhs_labels[dim]));
        out_labels.extend(self.lhs_batch_dims.iter().map(|&dim| lhs_labels[dim]));

        Ok(DotGeneralLabels {
            lhs_labels,
            rhs_labels,
            out_labels,
            out_shape: expected_out_shape,
        })
    }
}
```

- [ ] **Step 4: Export the module**

In `strided-rs/strided-einsum2/src/lib.rs`, add:

```rust
/// Axis-based general dot product API.
pub mod dot_general;
pub use dot_general::DotGeneralConfig;
```

- [ ] **Step 5: Run focused tests and verify failure has moved to missing functions**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/strided-rs
cargo test -p strided-einsum2 --test dot_general
```

Expected: compile failure only for missing `dot_general_into`.

- [ ] **Step 6: Commit config and planning**

```bash
cd /Users/hiroshi/projects/tensor4all/strided-rs
git add strided-einsum2/Cargo.toml strided-einsum2/src/lib.rs strided-einsum2/src/dot_general.rs
git commit -m "feat: add dot-general axis planning"
```

---

### Task 3: Implement strided-einsum2 dot_general entry points

**Files:**
- Modify: `strided-rs/strided-einsum2/src/lib.rs`
- Modify: `strided-rs/strided-einsum2/src/dot_general.rs`

- [ ] **Step 1: Make the internal dispatch reusable**

In `strided-rs/strided-einsum2/src/lib.rs`, change:

```rust
fn einsum2_dispatch<T, B, ID>(
```

to:

```rust
pub(crate) fn einsum2_dispatch<T, B, ID>(
```

- [ ] **Step 2: Add generic backend dot-general function**

In `strided-rs/strided-einsum2/src/dot_general.rs`, add:

```rust
use strided_view::{StridedView, StridedViewMut};

use crate::backend::Backend;
use crate::{einsum2_dispatch, Einsum2Plan, ScalarBase};

pub fn dot_general_with_backend_into<T, B>(
    c: StridedViewMut<T>,
    a: &StridedView<T>,
    b: &StridedView<T>,
    config: &DotGeneralConfig,
    alpha: T,
    beta: T,
) -> Result<()>
where
    T: ScalarBase,
    B: Backend<T>,
{
    let labels = config.labels_for_shapes(a.dims(), b.dims(), c.dims())?;
    let plan = Einsum2Plan::new(
        labels.lhs_labels.as_slice(),
        labels.rhs_labels.as_slice(),
        labels.out_labels.as_slice(),
    )?;
    einsum2_dispatch::<T, B, _>(c, a, b, &plan, alpha, beta, false, false, None)
}
```

- [ ] **Step 3: Add active-backend dot-general function**

In `strided-rs/strided-einsum2/src/dot_general.rs`, add:

```rust
#[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
pub fn dot_general_into<T: crate::Scalar>(
    c: StridedViewMut<T>,
    a: &StridedView<T>,
    b: &StridedView<T>,
    config: &DotGeneralConfig,
    alpha: T,
    beta: T,
) -> Result<()>
where
    crate::backend::ActiveBackend: Backend<T>,
{
    dot_general_with_backend_into::<T, crate::backend::ActiveBackend>(c, a, b, config, alpha, beta)
}

#[cfg(not(any(feature = "faer", feature = "blas", feature = "blas-inject")))]
pub fn dot_general_into<T: crate::Scalar>(
    c: StridedViewMut<T>,
    a: &StridedView<T>,
    b: &StridedView<T>,
    config: &DotGeneralConfig,
    alpha: T,
    beta: T,
) -> Result<()> {
    let labels = config.labels_for_shapes(a.dims(), b.dims(), c.dims())?;
    crate::einsum2_naive_into(
        c,
        a,
        b,
        labels.out_labels.as_slice(),
        labels.lhs_labels.as_slice(),
        labels.rhs_labels.as_slice(),
        alpha,
        beta,
        |x| x,
        |x| x,
    )
}
```

- [ ] **Step 4: Re-export public functions**

In `strided-rs/strided-einsum2/src/lib.rs`, replace the earlier re-export with:

```rust
pub use dot_general::{dot_general_into, dot_general_with_backend_into, DotGeneralConfig};
```

- [ ] **Step 5: Run strided dot-general tests**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/strided-rs
cargo test -p strided-einsum2 --test dot_general
```

Expected: all tests in `tests/dot_general.rs` pass.

- [ ] **Step 6: Run existing strided-einsum2 tests**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/strided-rs
cargo test -p strided-einsum2
cargo test -p strided-einsum2 --no-default-features
```

Expected: all tests pass.

- [ ] **Step 7: Commit entry points**

```bash
cd /Users/hiroshi/projects/tensor4all/strided-rs
git add strided-einsum2/src/lib.rs strided-einsum2/src/dot_general.rs
git commit -m "feat: add strided dot-general entry points"
```

---

### Task 4: Allow faer and BLAS implementations to coexist in strided-einsum2

**Files:**
- Modify: `strided-rs/strided-einsum2/src/lib.rs`
- Modify: `strided-rs/strided-einsum2/src/backend.rs`

- [ ] **Step 1: Relax feature exclusivity**

In `strided-rs/strided-einsum2/src/lib.rs`, remove the compile errors for `faer + blas` and `faer + blas-inject`. Keep the error for `blas + blas-inject`:

```rust
#[cfg(all(feature = "blas", feature = "blas-inject"))]
compile_error!("Features `blas` and `blas-inject` are mutually exclusive.");
```

- [ ] **Step 2: Make `Scalar` bounds work when both faer and BLAS are enabled**

Replace the cfg blocks for `Scalar` with precedence-based cfgs:

```rust
#[cfg(any(feature = "blas", feature = "blas-inject"))]
pub trait Scalar: ScalarBase + ElementOpApply + bgemm_blas::BlasGemm {}

#[cfg(any(feature = "blas", feature = "blas-inject"))]
impl<T> Scalar for T where T: ScalarBase + ElementOpApply + bgemm_blas::BlasGemm {}

#[cfg(all(feature = "faer", not(any(feature = "blas", feature = "blas-inject"))))]
pub trait Scalar: ScalarBase + ElementOpApply + faer_traits::ComplexField {}

#[cfg(all(feature = "faer", not(any(feature = "blas", feature = "blas-inject"))))]
impl<T> Scalar for T where T: ScalarBase + ElementOpApply + faer_traits::ComplexField {}

#[cfg(not(any(feature = "faer", feature = "blas", feature = "blas-inject")))]
pub trait Scalar: ScalarBase + ElementOpApply {}

#[cfg(not(any(feature = "faer", feature = "blas", feature = "blas-inject")))]
impl<T> Scalar for T where T: ScalarBase + ElementOpApply {}
```

- [ ] **Step 3: Prefer BLAS for ActiveBackend when BLAS exists**

In `strided-rs/strided-einsum2/src/backend.rs`, set `ActiveBackend` in this order:

```rust
#[cfg(any(feature = "blas", feature = "blas-inject"))]
pub type ActiveBackend = BlasBackend;

#[cfg(all(feature = "faer", not(any(feature = "blas", feature = "blas-inject"))))]
pub type ActiveBackend = FaerBackend;

#[cfg(not(any(feature = "faer", feature = "blas", feature = "blas-inject")))]
pub type ActiveBackend = NaiveBackend;
```

- [ ] **Step 4: Compile the feature matrix**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/strided-rs
cargo check -p strided-einsum2
cargo check -p strided-einsum2 --no-default-features
cargo check -p strided-einsum2 --no-default-features --features faer
cargo check -p strided-einsum2 --no-default-features --features blas
cargo check -p strided-einsum2 --no-default-features --features faer,blas
cargo check -p strided-einsum2 --no-default-features --features blas,blas-inject
```

Expected:
- The first five commands pass.
- The final `blas,blas-inject` command fails with the intended compile error.

- [ ] **Step 5: Commit backend coexistence**

```bash
cd /Users/hiroshi/projects/tensor4all/strided-rs
git add strided-einsum2/src/lib.rs strided-einsum2/src/backend.rs
git commit -m "feat: allow explicit faer and blas dot backends"
```

---

### Task 5: Add tenferro adapter using strided-einsum2

**Files:**
- Modify: `tenferro-rs/Cargo.toml`
- Modify: `tenferro-rs/crates/tenferro-cpu/Cargo.toml`
- Create: `tenferro-rs/crates/tenferro-cpu/src/gemm/strided_dot.rs`
- Modify: `tenferro-rs/crates/tenferro-cpu/src/gemm/mod.rs`

- [ ] **Step 1: Add the dependency for local development**

In `tenferro-rs/Cargo.toml`, add a temporary local dependency while developing:

```toml
strided-einsum2 = { path = "../strided-rs/strided-einsum2", default-features = false }
```

Before committing to `origin/main`, replace the path with a git revision from the pushed `strided-rs` commit:

```toml
strided-einsum2 = { git = "https://github.com/tensor4all/strided-rs", rev = "the output of git rev-parse HEAD from the pushed strided-rs commit", default-features = false }
```

- [ ] **Step 2: Wire tenferro-cpu features**

In `tenferro-rs/crates/tenferro-cpu/Cargo.toml`, add:

```toml
[dependencies]
strided-einsum2 = { workspace = true, optional = true }
```

Then update feature wiring so `cpu-faer` enables `strided-einsum2/faer`, and normal BLAS provider builds enable `strided-einsum2/blas`. Keep `provider-inject` on `strided-einsum2/blas-inject` only if the crate-level feature matrix confirms it does not also enable `strided-einsum2/blas`.

- [ ] **Step 3: Add adapter skeleton**

Create `tenferro-rs/crates/tenferro-cpu/src/gemm/strided_dot.rs`:

```rust
use num_traits::{One, Zero};
use strided_view::{col_major_strides, StridedView, StridedViewMut};
use tenferro_tensor::{Buffer, DotGeneralConfig, TypedTensor};

use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::{default_placement, Error};

use super::TypedTensorRead;

fn map_strided_error(err: impl std::fmt::Display) -> Error {
    Error::backend_failure("dot_general", err.to_string())
}

fn as_strided_view<'a, R, T>(read: &'a R) -> crate::Result<Option<StridedView<'a, T>>>
where
    R: TypedTensorRead<T>,
    T: 'static,
{
    let Some(data) = read.host_data_opt() else {
        return Ok(None);
    };
    StridedView::new(data, read.shape(), read.strides().as_slice(), read.offset())
        .map(Some)
        .map_err(map_strided_error)
}

pub(crate) fn dot_general_strided_with_backend<L, R, T, B>(
    buffers: &mut BufferPool,
    lhs: &L,
    rhs: &R,
    config: &DotGeneralConfig,
) -> crate::Result<Option<TypedTensor<T>>>
where
    L: TypedTensorRead<T>,
    R: TypedTensorRead<T>,
    T: PoolScalar
        + Copy
        + Clone
        + Zero
        + One
        + PartialEq
        + strided_einsum2::ScalarBase
        + 'static,
    B: strided_einsum2::Backend<T>,
{
    let Some(lhs_view) = as_strided_view(lhs)? else {
        return Ok(None);
    };
    let Some(rhs_view) = as_strided_view(rhs)? else {
        return Ok(None);
    };

    let local_config = strided_einsum2::DotGeneralConfig {
        lhs_contracting_dims: config.lhs_contracting_dims.clone(),
        rhs_contracting_dims: config.rhs_contracting_dims.clone(),
        lhs_batch_dims: config.lhs_batch_dims.clone(),
        rhs_batch_dims: config.rhs_batch_dims.clone(),
    };
    let out_shape = local_config
        .expected_output_shape(lhs.shape(), rhs.shape())
        .map_err(map_strided_error)?;
    let out_n = out_shape.iter().product::<usize>().max(1);
    let mut out_data = unsafe { T::pool_acquire(buffers, out_n) };
    if out_n > 0 {
        out_data.fill(T::zero());
    }
    let out_strides = col_major_strides(&out_shape);
    let out_view = StridedViewMut::new(&mut out_data, &out_shape, &out_strides, 0)
        .map_err(map_strided_error)?;

    strided_einsum2::dot_general_with_backend_into::<T, B>(
        out_view,
        &lhs_view,
        &rhs_view,
        &local_config,
        T::one(),
        T::zero(),
    )
    .map_err(map_strided_error)?;

    Ok(Some(TypedTensor::from_buffer_col_major(
        out_shape,
        Buffer::Host(out_data),
        default_placement(),
    )))
}
```

This step requires `expected_output_shape` to be exposed from `strided-einsum2::DotGeneralConfig`. If Task 2 kept it private inside `labels_for_shapes`, add:

```rust
pub fn expected_output_shape(
    &self,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> Result<Vec<usize>> {
    let lhs_free = free_dims(lhs_shape.len(), &self.lhs_contracting_dims, &self.lhs_batch_dims);
    let rhs_free = free_dims(rhs_shape.len(), &self.rhs_contracting_dims, &self.rhs_batch_dims);
    let mut out = Vec::with_capacity(lhs_free.len() + rhs_free.len() + self.lhs_batch_dims.len());
    out.extend(lhs_free.iter().map(|&dim| lhs_shape[dim]));
    out.extend(rhs_free.iter().map(|&dim| rhs_shape[dim]));
    out.extend(self.lhs_batch_dims.iter().map(|&dim| lhs_shape[dim]));
    Ok(out)
}
```

- [ ] **Step 4: Expose the adapter module**

In `tenferro-rs/crates/tenferro-cpu/src/gemm/mod.rs`, add:

```rust
mod strided_dot;
```

- [ ] **Step 5: Compile tenferro-cpu**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/tenferro-rs
cargo check -p tenferro-cpu
```

Expected: the adapter compiles before it is used.

- [ ] **Step 6: Commit adapter skeleton**

```bash
cd /Users/hiroshi/projects/tensor4all/tenferro-rs
git add Cargo.toml crates/tenferro-cpu/Cargo.toml crates/tenferro-cpu/src/gemm/mod.rs crates/tenferro-cpu/src/gemm/strided_dot.rs
git commit -m "feat: add strided dot-general adapter"
```

---

### Task 6: Route tenferro dot_general through the strided adapter

**Files:**
- Modify: `tenferro-rs/crates/tenferro-cpu/src/gemm/mod.rs`
- Modify: `tenferro-rs/crates/tenferro-cpu/src/tests/cpu_tests/dot_structural_analytic.rs`

- [ ] **Step 1: Replace faer direct GEMM call**

In the faer dispatch path inside `dot_general_faer_read_cached`, replace the call to `typed_faer_gemm` with:

```rust
return strided_dot::dot_general_strided_with_backend::<_, _, _, strided_einsum2::backend::FaerBackend>(
    buffers,
    a,
    b,
    config,
)
.map(|result| result.map(crate::Tensor::$wrap));
```

Keep `typed_faer_gemm` in the file until all tests pass; this makes rollback and comparison easy during the task.

- [ ] **Step 2: Replace BLAS direct GEMM call**

In the BLAS dispatch path inside `dot_general_blas_read_cached`, replace the call to `typed_blas_gemm` or `typed_blas_gemm_with_conj` for non-conjugated inputs with:

```rust
return strided_dot::dot_general_strided_with_backend::<_, _, _, strided_einsum2::backend::BlasBackend>(
    buffers,
    a,
    b,
    config,
)
.map(|result| result.map(crate::Tensor::$wrap));
```

Keep existing conjugation fallback paths until `strided-einsum2` has a public conjugating dot-general entry point.

- [ ] **Step 3: Preserve TensorRead transposed view behavior**

Add this assertion to the existing transposed-view test in `tenferro-rs/crates/tenferro-cpu/src/tests/cpu_tests/dot_structural_analytic.rs`:

```rust
assert_eq!(out.shape(), &[2, 2]);
assert_eq!(out.as_slice::<f64>().unwrap(), &[50.0, 122.0, 68.0, 167.0]);
```

This is already the expected behavior; the purpose is to keep the test tied to the adapter path after routing changes.

- [ ] **Step 4: Run tenferro dot-general tests**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/tenferro-rs
cargo test -p tenferro-cpu dot_general
```

Expected: all dot-general tests pass with the strided adapter route.

- [ ] **Step 5: Run tenferro-cpu default tests**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/tenferro-rs
cargo test -p tenferro-cpu
```

Expected: all default-feature tenferro-cpu tests pass.

- [ ] **Step 6: Commit routing**

```bash
cd /Users/hiroshi/projects/tensor4all/tenferro-rs
git add crates/tenferro-cpu/src/gemm/mod.rs crates/tenferro-cpu/src/tests/cpu_tests/dot_structural_analytic.rs
git commit -m "refactor: route cpu dot-general through strided-einsum2"
```

---

### Task 7: Remove duplicated tenferro GEMM analysis only after parity is proven

**Files:**
- Modify: `tenferro-rs/crates/tenferro-cpu/src/gemm/mod.rs`
- Modify: `tenferro-rs/crates/tenferro-cpu/src/gemm/tests.rs`

- [ ] **Step 1: Identify dead functions**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/tenferro-rs
rg -n "typed_faer_gemm|typed_blas_gemm|analyse_gemm|canonical_gemm_layout|GemmDims|GemmAnalysisPlan" crates/tenferro-cpu/src/gemm
```

Expected: after Task 6, only tests and unused helper definitions reference the old direct-analysis path.

- [ ] **Step 2: Remove old direct-analysis tests that duplicate strided coverage**

Keep cache-stat tests if tenferro still stores cache slot metadata. Remove tests that only check `try_fuse_dims`, `canonical_gemm_layout`, or low-level BLAS/Faer pointer loops after equivalent tests exist in `strided-einsum2`.

- [ ] **Step 3: Move retained cache tests to adapter-level behavior**

Change the cache test to assert tenferro still records a runtime cache entry for the operation, not that it owns the GEMM stride-analysis logic.

- [ ] **Step 4: Compile and test**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/tenferro-rs
cargo test -p tenferro-cpu dot_general
cargo test -p tenferro-cpu
```

Expected: tests pass and `rg` no longer finds unused direct GEMM analysis functions.

- [ ] **Step 5: Commit cleanup**

```bash
cd /Users/hiroshi/projects/tensor4all/tenferro-rs
git add crates/tenferro-cpu/src/gemm/mod.rs crates/tenferro-cpu/src/gemm/tests.rs
git commit -m "refactor: remove duplicated cpu gemm analysis"
```

---

### Task 8: Add benchmark comparison and performance gate

**Files:**
- Modify: `tenferro-rs/crates/tenferro-cpu/benches/dot_general_overhead.rs`

- [ ] **Step 1: Add cases that expose adapter overhead**

Add benchmark cases for:

```rust
// Direct col-major matmul: [256, 256] x [256, 256]
// Transposed lhs TensorRead view: [256, 256]^T x [256, 256]
// Batched matmul: [64, 64, 32] x [64, 64, 32]
// Rank-0 dot: [4096] dot [4096]
```

Use one warmed `CpuBackend`, pre-created inputs, and no input clone inside the timed loop.

- [ ] **Step 2: Run default faer benchmark**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/tenferro-rs
cargo bench -p tenferro-cpu --bench dot_general_overhead
```

Expected: adapter overhead is not larger than the old direct path by more than 5% for direct col-major matmul. If the old path has already been removed, compare against the last committed benchmark YAML in `tenferro-benchmark`.

- [ ] **Step 3: Run BLAS provider benchmark when available**

Run on macOS Accelerate:

```bash
cd /Users/hiroshi/projects/tensor4all/tenferro-rs
RUSTFLAGS="-C target-cpu=native" cargo bench -p tenferro-cpu --no-default-features --features cpu-faer,src-accelerate --bench dot_general_overhead
```

Expected: BLAS path still routes to the selected CPU backend and does not fall back to faer when `CpuBackendKind::Blas` is requested.

- [ ] **Step 4: Commit benchmark coverage**

```bash
cd /Users/hiroshi/projects/tensor4all/tenferro-rs
git add crates/tenferro-cpu/benches/dot_general_overhead.rs
git commit -m "bench: cover strided dot-general adapter overhead"
```

---

### Task 9: Final verification and dependency pin update

**Files:**
- Modify: `tenferro-rs/Cargo.toml`
- Modify: `tenferro-rs/Cargo.lock`

- [ ] **Step 1: Verify strided-rs**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/strided-rs
cargo test -p strided-einsum2
cargo test -p strided-einsum2 --no-default-features
cargo check -p strided-einsum2 --no-default-features --features faer,blas
git diff --check
```

Expected: tests pass, feature check passes, and `git diff --check` reports no whitespace errors.

- [ ] **Step 2: Push strided-rs and capture commit hash**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/strided-rs
git rev-parse HEAD
git push origin HEAD
```

Expected: output contains the commit hash used in tenferro dependency pinning.

- [ ] **Step 3: Replace tenferro local path dependency with git rev**

In `tenferro-rs/Cargo.toml`, replace:

```toml
strided-einsum2 = { path = "../strided-rs/strided-einsum2", default-features = false }
```

with:

```toml
strided-einsum2 = { git = "https://github.com/tensor4all/strided-rs", rev = "the exact commit printed by git rev-parse HEAD in Step 2", default-features = false }
```

- [ ] **Step 4: Verify tenferro-rs**

Run:

```bash
cd /Users/hiroshi/projects/tensor4all/tenferro-rs
cargo update -p strided-einsum2
cargo test -p tenferro-cpu
cargo test -p tenferro-runtime
git diff --check
```

Expected: tests pass and `git diff --check` reports no whitespace errors.

- [ ] **Step 5: Commit tenferro dependency pin**

```bash
cd /Users/hiroshi/projects/tensor4all/tenferro-rs
git add Cargo.toml Cargo.lock
git commit -m "chore: pin strided-einsum2 dot-general dependency"
```

---

## Nontrivial Decisions

1. **Merge base:** use `strided-einsum2` as the base. tenferro's implementation is the behavior reference and integration adapter source, not the kernel-library base.
2. **Backend choice:** `strided-einsum2` should compile faer and BLAS together. `ActiveBackend` should prefer BLAS when BLAS is present, matching tenferro's expected default priority.
3. **provider-inject:** verify this separately. If Cargo feature unification enables both `strided-einsum2/blas` and `strided-einsum2/blas-inject`, split tenferro CPU BLAS feature wiring before routing through the adapter.
4. **Cache location:** tenferro keeps runtime cache slots. `strided-einsum2` should not depend on tenferro cache types. Add a reusable `PreparedDotGeneralPlan` in `strided-einsum2` only after adapter overhead is measured.
5. **Graph optimizer:** keep `tenferro-runtime/src/compiler/dot_decomposer.rs` in tenferro. It is graph/symbolic lowering, not a strided kernel concern.
6. **Conjugation:** first replacement can preserve tenferro's existing conjugation fallback. Add public conjugating dot-general APIs to `strided-einsum2` after non-conjugated parity is established.

## Self-Review

- Spec coverage: the plan compares the two implementations by assigning kernel ownership to `strided-einsum2` and tenferro ownership to runtime integration, then defines tasks for both repos.
- Placeholder scan: the plan contains concrete files, code snippets, commands, expected outcomes, and commit boundaries.
- Type consistency: public `DotGeneralConfig` borrows `&[usize]` axis slices so tenferro can pass its existing config metadata without cloning. Internal planning uses `SmallVec` to reduce allocation after validation.
