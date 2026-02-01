//! Binary Einstein summation on strided views.
//!
//! Provides `einsum2_into` for computing binary tensor contractions with
//! accumulation semantics: `C = alpha * A * B + beta * C`.
//!
//! # Example
//!
//! ```
//! use stridedview::StridedArray;
//! use strided_einsum2::einsum2_into;
//!
//! // Matrix multiply: C_ik = A_ij * B_jk
//! let a = StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
//! let b = StridedArray::<f64>::from_fn_row_major(&[3, 2], |idx| (idx[0] * 2 + idx[1] + 1) as f64);
//! let mut c = StridedArray::<f64>::row_major(&[2, 2]);
//!
//! einsum2_into(
//!     c.view_mut(), &a.view(), &b.view(),
//!     &['i', 'k'], &['i', 'j'], &['j', 'k'],
//!     1.0, 0.0,
//! ).unwrap();
//! ```

pub mod bgemm;
pub mod plan;
pub mod trace;
pub mod util;

use std::fmt::Debug;
use std::hash::Hash;

use stridedview::{ElementOp, ElementOpApply, StridedView, StridedViewMut};

pub use plan::Einsum2Plan;

/// Trait alias for axis label types.
pub trait AxisId: Clone + Eq + Hash + Debug {}
impl<T: Clone + Eq + Hash + Debug> AxisId for T {}

/// Errors specific to einsum operations.
#[derive(Debug, thiserror::Error)]
pub enum EinsumError {
    #[error("duplicate axis label: {0}")]
    DuplicateAxis(String),
    #[error("output axis {0} not found in any input")]
    OrphanOutputAxis(String),
    #[error("dimension mismatch for axis {axis:?}: {dim_a} vs {dim_b}")]
    DimensionMismatch {
        axis: String,
        dim_a: usize,
        dim_b: usize,
    },
    #[error(transparent)]
    Strided(#[from] stridedview::StridedError),
}

pub type Result<T> = std::result::Result<T, EinsumError>;

/// Binary einsum contraction: `C = alpha * contract(A, B) + beta * C`.
///
/// `ic`, `ia`, `ib` are axis labels for C, A, B respectively.
/// Axes are classified as:
/// - **batch**: in A, B, and C
/// - **lo** (left-output): in A and C, not B
/// - **ro** (right-output): in B and C, not A
/// - **sum** (contraction): in A and B, not C
/// - **left_trace**: only in A (summed out before contraction)
/// - **right_trace**: only in B (summed out before contraction)
pub fn einsum2_into<T, OpA, OpB, ID: AxisId>(
    c: StridedViewMut<T>,
    a: &StridedView<T, OpA>,
    b: &StridedView<T, OpB>,
    ic: &[ID],
    ia: &[ID],
    ib: &[ID],
    alpha: T,
    beta: T,
) -> Result<()>
where
    T: Copy
        + ElementOpApply
        + Send
        + Sync
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + num_traits::Zero
        + num_traits::One
        + PartialEq,
    OpA: ElementOp,
    OpB: ElementOp,
{
    // 1. Build plan
    let plan = Einsum2Plan::new(ia, ib, ic)?;

    // 2. Validate dimension consistency across operands
    validate_dimensions::<ID>(&plan, a.dims(), b.dims(), c.dims(), ia, ib, ic)?;

    // 3. Reduce trace axes if present
    let a_reduced;
    let a_view: StridedView<T>;
    let has_left_trace = !plan.left_trace.is_empty();
    if has_left_trace {
        let trace_indices = plan.left_trace_indices(ia);
        a_reduced = trace::reduce_trace_axes(a, &trace_indices)?;
        a_view = a_reduced.view();
    } else {
        // No trace: just apply the element op by creating an Identity view
        // We need to strip the Op to get StridedView<T, Identity>
        a_view = strip_op_view(a);
    }

    let b_reduced;
    let b_view: StridedView<T>;
    let has_right_trace = !plan.right_trace.is_empty();
    if has_right_trace {
        let trace_indices = plan.right_trace_indices(ib);
        b_reduced = trace::reduce_trace_axes(b, &trace_indices)?;
        b_view = b_reduced.view();
    } else {
        b_view = strip_op_view(b);
    }

    // 4. Permute to canonical order
    //    A -> [batch, lo, sum]
    //    B -> [batch, sum, ro]
    //    C -> [batch, lo, ro]
    let a_perm = a_view.permute(&plan.left_perm)?;
    let b_perm = b_view.permute(&plan.right_perm)?;
    let mut c_perm = c.permute(&plan.c_to_internal_perm)?;

    // 5. Call batched GEMM
    bgemm::bgemm_strided_into(
        &mut c_perm,
        &a_perm,
        &b_perm,
        plan.batch.len(),
        plan.lo.len(),
        plan.ro.len(),
        plan.sum.len(),
        alpha,
        beta,
    )?;

    Ok(())
}

/// Strip the element operation, returning a plain `StridedView<T, Identity>`.
///
/// This reuses the same underlying data slice with identical dims/strides/offset.
/// It is correct only when the element op does not change values (i.e., `Identity`
/// on real types). For complex types with `Conj`/`Adjoint`, the caller must
/// materialize first (the trace path already does this).
fn strip_op_view<'a, T, Op>(src: &StridedView<'a, T, Op>) -> StridedView<'a, T>
where
    T: Copy + ElementOpApply,
    Op: ElementOp,
{
    StridedView::new(src.data(), src.dims(), src.strides(), src.offset())
        .expect("strip_op_view: metadata already validated")
}

/// Validate that dimensions match across operands for each axis group.
fn validate_dimensions<ID: AxisId>(
    plan: &Einsum2Plan<ID>,
    a_dims: &[usize],
    b_dims: &[usize],
    c_dims: &[usize],
    ia: &[ID],
    ib: &[ID],
    ic: &[ID],
) -> Result<()> {
    let find_dim = |labels: &[ID], dims: &[usize], id: &ID| -> usize {
        labels
            .iter()
            .position(|x| x == id)
            .map(|i| dims[i])
            .unwrap()
    };

    // Batch: must match in A, B, and C
    for id in &plan.batch {
        let da = find_dim(ia, a_dims, id);
        let db = find_dim(ib, b_dims, id);
        let dc = find_dim(ic, c_dims, id);
        if da != db || da != dc {
            return Err(EinsumError::DimensionMismatch {
                axis: format!("{:?}", id),
                dim_a: da,
                dim_b: db,
            });
        }
    }

    // Sum: must match in A and B
    for id in &plan.sum {
        let da = find_dim(ia, a_dims, id);
        let db = find_dim(ib, b_dims, id);
        if da != db {
            return Err(EinsumError::DimensionMismatch {
                axis: format!("{:?}", id),
                dim_a: da,
                dim_b: db,
            });
        }
    }

    // LO: must match in A and C
    for id in &plan.lo {
        let da = find_dim(ia, a_dims, id);
        let dc = find_dim(ic, c_dims, id);
        if da != dc {
            return Err(EinsumError::DimensionMismatch {
                axis: format!("{:?}", id),
                dim_a: da,
                dim_b: dc,
            });
        }
    }

    // RO: must match in B and C
    for id in &plan.ro {
        let db = find_dim(ib, b_dims, id);
        let dc = find_dim(ic, c_dims, id);
        if db != dc {
            return Err(EinsumError::DimensionMismatch {
                axis: format!("{:?}", id),
                dim_a: db,
                dim_b: dc,
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use stridedview::StridedArray;

    #[test]
    fn test_matmul_ij_jk_ik() {
        // C_ik = A_ij * B_jk
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[5.0, 6.0], [7.0, 8.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f64>::row_major(&[2, 2]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['i', 'k'],
            &['i', 'j'],
            &['j', 'k'],
            1.0,
            0.0,
        )
        .unwrap();

        assert_eq!(c.get(&[0, 0]), 19.0);
        assert_eq!(c.get(&[0, 1]), 22.0);
        assert_eq!(c.get(&[1, 0]), 43.0);
        assert_eq!(c.get(&[1, 1]), 50.0);
    }

    #[test]
    fn test_matmul_rect() {
        // A: 2x3, B: 3x4, C: 2x4
        let a =
            StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        let b =
            StridedArray::<f64>::from_fn_row_major(&[3, 4], |idx| (idx[0] * 4 + idx[1] + 1) as f64);
        let mut c = StridedArray::<f64>::row_major(&[2, 4]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['i', 'k'],
            &['i', 'j'],
            &['j', 'k'],
            1.0,
            0.0,
        )
        .unwrap();

        // A = [[1,2,3],[4,5,6]], B = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
        assert_eq!(c.get(&[0, 0]), 38.0);
        assert_eq!(c.get(&[1, 3]), 128.0);
    }

    #[test]
    fn test_batched_matmul() {
        // C_bik = A_bij * B_bjk
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2, 3], |idx| {
            (idx[0] * 6 + idx[1] * 3 + idx[2] + 1) as f64
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 3, 2], |idx| {
            (idx[0] * 6 + idx[1] * 2 + idx[2] + 1) as f64
        });
        let mut c = StridedArray::<f64>::row_major(&[2, 2, 2]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['b', 'i', 'k'],
            &['b', 'i', 'j'],
            &['b', 'j', 'k'],
            1.0,
            0.0,
        )
        .unwrap();

        // Batch 0: A0=[[1,2,3],[4,5,6]], B0=[[1,2],[3,4],[5,6]]
        // C0[0,0] = 1*1+2*3+3*5 = 22
        assert_eq!(c.get(&[0, 0, 0]), 22.0);
    }

    #[test]
    fn test_outer_product() {
        // C_ij = A_i * B_j
        let a = StridedArray::<f64>::from_fn_row_major(&[3], |idx| (idx[0] + 1) as f64);
        let b = StridedArray::<f64>::from_fn_row_major(&[4], |idx| (idx[0] + 1) as f64);
        let mut c = StridedArray::<f64>::row_major(&[3, 4]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['i', 'j'],
            &['i'],
            &['j'],
            1.0,
            0.0,
        )
        .unwrap();

        assert_eq!(c.get(&[0, 0]), 1.0);
        assert_eq!(c.get(&[2, 3]), 12.0);
    }

    #[test]
    fn test_dot_product() {
        // C = A_i * B_i (scalar output)
        let a = StridedArray::<f64>::from_fn_row_major(&[3], |idx| (idx[0] + 1) as f64);
        let b = StridedArray::<f64>::from_fn_row_major(&[3], |idx| (idx[0] + 1) as f64);
        let mut c = StridedArray::<f64>::row_major(&[]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &[] as &[char],
            &['i'],
            &['i'],
            1.0,
            0.0,
        )
        .unwrap();

        // 1*1 + 2*2 + 3*3 = 14
        assert_eq!(c.get(&[]), 14.0);
    }

    #[test]
    fn test_alpha_beta() {
        // C = 2*A*B + 3*C_old
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 0.0], [0.0, 1.0]][idx[0]][idx[1]] // identity
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[10.0, 20.0], [30.0, 40.0]][idx[0]][idx[1]]
        });

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['i', 'k'],
            &['i', 'j'],
            &['j', 'k'],
            2.0,
            3.0,
        )
        .unwrap();

        // C = 2*I*B + 3*C_old
        assert_eq!(c.get(&[0, 0]), 32.0); // 2*1 + 3*10
        assert_eq!(c.get(&[1, 1]), 128.0); // 2*4 + 3*40
    }

    #[test]
    fn test_transposed_output() {
        // C_ki = A_ij * B_jk (output transposed)
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[5.0, 6.0], [7.0, 8.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f64>::row_major(&[2, 2]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['k', 'i'], // C indexed as (k, i) instead of (i, k)
            &['i', 'j'],
            &['j', 'k'],
            1.0,
            0.0,
        )
        .unwrap();

        // Normal matmul result: C_ik = [[19,22],[43,50]]
        // But C is indexed as (k, i), so C[k, i] = (A*B)[i, k]
        assert_eq!(c.get(&[0, 0]), 19.0); // C[k=0, i=0]
        assert_eq!(c.get(&[0, 1]), 43.0); // C[k=0, i=1]
        assert_eq!(c.get(&[1, 0]), 22.0); // C[k=1, i=0]
        assert_eq!(c.get(&[1, 1]), 50.0); // C[k=1, i=1]
    }

    #[test]
    fn test_left_trace() {
        // C_k = sum_j (sum_i A_ij) * B_jk
        // left_trace=[i], sum=[j], ro=[k]
        let a =
            StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        // A = [[1,2,3],[4,5,6]]
        // sum over i: [5, 7, 9]
        let b =
            StridedArray::<f64>::from_fn_row_major(&[3, 2], |idx| (idx[0] * 2 + idx[1] + 1) as f64);
        // B = [[1,2],[3,4],[5,6]]
        let mut c = StridedArray::<f64>::row_major(&[2]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &['k'],
            &['i', 'j'],
            &['j', 'k'],
            1.0,
            0.0,
        )
        .unwrap();

        // C_k = sum_j [5,7,9][j] * B[j,k]
        // C[0] = 5*1 + 7*3 + 9*5 = 5 + 21 + 45 = 71
        // C[1] = 5*2 + 7*4 + 9*6 = 10 + 28 + 54 = 92
        assert_eq!(c.get(&[0]), 71.0);
        assert_eq!(c.get(&[1]), 92.0);
    }

    #[test]
    fn test_u32_labels() {
        // Same as matmul but with u32 labels
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[5.0, 6.0], [7.0, 8.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f64>::row_major(&[2, 2]);

        einsum2_into(
            c.view_mut(),
            &a.view(),
            &b.view(),
            &[0u32, 2],
            &[0u32, 1],
            &[1u32, 2],
            1.0,
            0.0,
        )
        .unwrap();

        assert_eq!(c.get(&[0, 0]), 19.0);
        assert_eq!(c.get(&[1, 1]), 50.0);
    }
}
