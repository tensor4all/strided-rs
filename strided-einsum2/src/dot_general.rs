//! Axis-based general dot product API.
//!
//! The dimension-numbering config borrows axis slices so callers can reuse
//! their existing metadata without allocating a second owned config.
//!
//! ```
//! use strided_einsum2::{dot_general_into, DotGeneralConfig};
//! use strided_view::StridedArray;
//!
//! let a = StridedArray::<f64>::from_fn_col_major(&[2, 3], |idx| {
//!     (idx[0] + 2 * idx[1] + 1) as f64
//! });
//! let b = StridedArray::<f64>::from_fn_col_major(&[3, 2], |idx| {
//!     (idx[0] + 3 * idx[1] + 1) as f64
//! });
//! let mut c = StridedArray::<f64>::col_major(&[2, 2]);
//!
//! let config = DotGeneralConfig {
//!     lhs_contracting_dims: &[1],
//!     rhs_contracting_dims: &[0],
//!     lhs_batch_dims: &[],
//!     rhs_batch_dims: &[],
//! };
//! dot_general_into(c.view_mut(), &a.view(), &b.view(), &config, 1.0, 0.0).unwrap();
//! ```

use smallvec::SmallVec;
use std::mem::MaybeUninit;
use strided_kernel::ExecContext;
use strided_view::{RawStridedMut, StridedView, StridedViewMut};

use crate::backend::Backend;
use crate::{einsum2_dispatch, Einsum2Plan, EinsumError, Result, ScalarBase};

/// DotGeneral dimension configuration.
///
/// The output shape is `[lhs_free..., rhs_free..., batch...]`, matching
/// tenferro's batch-trailing col-major convention.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct DotGeneralConfig<'a> {
    pub lhs_contracting_dims: &'a [usize],
    pub rhs_contracting_dims: &'a [usize],
    pub lhs_batch_dims: &'a [usize],
    pub rhs_batch_dims: &'a [usize],
}

#[derive(Clone, Debug)]
pub(crate) struct DotGeneralLabels {
    pub lhs_labels: SmallVec<[usize; 8]>,
    pub rhs_labels: SmallVec<[usize; 8]>,
    pub out_labels: SmallVec<[usize; 8]>,
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

impl DotGeneralConfig<'_> {
    /// Validate dimension indices for explicit operand ranks.
    pub fn validate_dims_with_ranks(&self, lhs_rank: usize, rhs_rank: usize) -> Result<()> {
        check_bounds(&self.lhs_contracting_dims, lhs_rank, "lhs_contracting")?;
        check_bounds(&self.rhs_contracting_dims, rhs_rank, "rhs_contracting")?;
        check_bounds(&self.lhs_batch_dims, lhs_rank, "lhs_batch")?;
        check_bounds(&self.rhs_batch_dims, rhs_rank, "rhs_batch")?;
        check_no_duplicates(&self.lhs_contracting_dims, "lhs_contracting_dims")?;
        check_no_duplicates(&self.rhs_contracting_dims, "rhs_contracting_dims")?;
        check_no_duplicates(&self.lhs_batch_dims, "lhs_batch_dims")?;
        check_no_duplicates(&self.rhs_batch_dims, "rhs_batch_dims")?;

        for &dim in self.lhs_contracting_dims {
            if self.lhs_batch_dims.contains(&dim) {
                return Err(EinsumError::InvalidDotGeneralConfig(format!(
                    "lhs dim {dim} appears in both contracting and batch dims"
                )));
            }
        }
        for &dim in self.rhs_contracting_dims {
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

    /// Compute the output shape `[lhs_free..., rhs_free..., batch...]`.
    pub fn expected_output_shape(
        &self,
        lhs_shape: &[usize],
        rhs_shape: &[usize],
    ) -> Result<Vec<usize>> {
        self.validate_dims_with_ranks(lhs_shape.len(), rhs_shape.len())?;
        self.validate_pair_shapes(lhs_shape, rhs_shape)?;

        let lhs_free = free_dims(
            lhs_shape.len(),
            &self.lhs_contracting_dims,
            &self.lhs_batch_dims,
        );
        let rhs_free = free_dims(
            rhs_shape.len(),
            &self.rhs_contracting_dims,
            &self.rhs_batch_dims,
        );

        let mut out =
            Vec::with_capacity(lhs_free.len() + rhs_free.len() + self.lhs_batch_dims.len());
        out.extend(lhs_free.iter().map(|&dim| lhs_shape[dim]));
        out.extend(rhs_free.iter().map(|&dim| rhs_shape[dim]));
        out.extend(self.lhs_batch_dims.iter().map(|&dim| lhs_shape[dim]));
        Ok(out)
    }

    fn validate_pair_shapes(&self, lhs_shape: &[usize], rhs_shape: &[usize]) -> Result<()> {
        for (&lhs_dim, &rhs_dim) in self.lhs_batch_dims.iter().zip(self.rhs_batch_dims) {
            if lhs_shape[lhs_dim] != rhs_shape[rhs_dim] {
                return Err(EinsumError::DimensionMismatch {
                    axis: format!("batch lhs {lhs_dim} rhs {rhs_dim}"),
                    dim_a: lhs_shape[lhs_dim],
                    dim_b: rhs_shape[rhs_dim],
                });
            }
        }
        for (&lhs_dim, &rhs_dim) in self
            .lhs_contracting_dims
            .iter()
            .zip(self.rhs_contracting_dims)
        {
            if lhs_shape[lhs_dim] != rhs_shape[rhs_dim] {
                return Err(EinsumError::DimensionMismatch {
                    axis: format!("contract lhs {lhs_dim} rhs {rhs_dim}"),
                    dim_a: lhs_shape[lhs_dim],
                    dim_b: rhs_shape[rhs_dim],
                });
            }
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
        self.validate_pair_shapes(lhs_shape, rhs_shape)?;

        let lhs_free = free_dims(
            lhs_shape.len(),
            &self.lhs_contracting_dims,
            &self.lhs_batch_dims,
        );
        let rhs_free = free_dims(
            rhs_shape.len(),
            &self.rhs_contracting_dims,
            &self.rhs_batch_dims,
        );

        let mut lhs_labels = smallvec::smallvec![usize::MAX; lhs_shape.len()];
        let mut rhs_labels = smallvec::smallvec![usize::MAX; rhs_shape.len()];
        let mut next_label = 0usize;

        for (&lhs_dim, &rhs_dim) in self.lhs_batch_dims.iter().zip(self.rhs_batch_dims) {
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
            .zip(self.rhs_contracting_dims)
        {
            lhs_labels[lhs_dim] = next_label;
            rhs_labels[rhs_dim] = next_label;
            next_label += 1;
        }

        let expected_out_shape = self.expected_output_shape(lhs_shape, rhs_shape)?;
        if expected_out_shape.as_slice() != out_shape {
            return Err(EinsumError::OutputShapeMismatch {
                expected: expected_out_shape,
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
        })
    }
}

/// Compute `C = alpha * dot_general(A, B) + beta * C` with an explicit backend.
pub fn dot_general_with_backend_into<T, B>(
    c: StridedViewMut<T>,
    a: &StridedView<T>,
    b: &StridedView<T>,
    config: &DotGeneralConfig<'_>,
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

/// Compute `C = alpha * dot_general(A, B) + beta * C` with the active backend.
#[cfg(any(feature = "faer", feature = "blas", feature = "blas-inject"))]
pub fn dot_general_into<T: crate::Scalar>(
    c: StridedViewMut<T>,
    a: &StridedView<T>,
    b: &StridedView<T>,
    config: &DotGeneralConfig<'_>,
    alpha: T,
    beta: T,
) -> Result<()>
where
    crate::backend::ActiveBackend: Backend<T>,
{
    dot_general_with_backend_into::<T, crate::backend::ActiveBackend>(c, a, b, config, alpha, beta)
}

/// Compute dot-general into a genuinely uninitialized destination.
///
/// The destination is only exposed as initialized after this function returns
/// `Ok(())`; holes in a strided backing allocation are never touched.
pub fn dot_general_into_uninit<T: crate::Scalar>(
    c: &mut RawStridedMut<'_, MaybeUninit<T>>,
    a: &StridedView<'_, T>,
    b: &StridedView<'_, T>,
    config: &DotGeneralConfig<'_>,
    alpha: T,
    ctx: &ExecContext,
) -> Result<()> {
    let labels = config.labels_for_shapes(a.dims(), b.dims(), c.dims())?;
    crate::einsum2_into_uninit(
        c,
        a,
        b,
        labels.out_labels.as_slice(),
        labels.lhs_labels.as_slice(),
        labels.rhs_labels.as_slice(),
        alpha,
        ctx,
    )
}

/// Compute `C = alpha * dot_general(A, B) + beta * C` with the naive fallback.
#[cfg(not(any(feature = "faer", feature = "blas", feature = "blas-inject")))]
pub fn dot_general_into<T: crate::Scalar>(
    c: StridedViewMut<T>,
    a: &StridedView<T>,
    b: &StridedView<T>,
    config: &DotGeneralConfig<'_>,
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
