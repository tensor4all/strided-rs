//! faer-backed batched GEMM kernel on strided views.
//!
//! Uses `faer::linalg::matmul::matmul` for SIMD-optimized matrix multiplication.
//! When dimension groups cannot be fused into 2D matrices (non-contiguous strides),
//! copies operands to contiguous column-major buffers before calling faer.

use crate::util::{try_fuse_group, MultiIndex};
use faer::linalg::matmul::matmul;
use faer::mat::{MatMut, MatRef};
use faer::{Accum, Par};
use faer_traits::ComplexField;
use stridedview::{StridedArray, StridedView, StridedViewMut};

/// Batched strided GEMM using faer: C = alpha * A * B + beta * C
///
/// Same interface as `bgemm_naive::bgemm_strided_into`. Uses faer's optimized
/// matmul for all cases. When dimension groups have non-contiguous strides,
/// copies operands to contiguous column-major buffers first using `strided::copy_into`.
pub fn bgemm_strided_into<T>(
    c: &mut StridedViewMut<T>,
    a: &StridedView<T>,
    b: &StridedView<T>,
    n_batch: usize,
    n_lo: usize,
    n_ro: usize,
    n_sum: usize,
    alpha: T,
    beta: T,
) -> stridedview::Result<()>
where
    T: ComplexField
        + Copy
        + stridedview::ElementOpApply
        + Send
        + Sync
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + num_traits::Zero
        + num_traits::One
        + PartialEq,
{
    let a_dims = a.dims();
    let b_dims = b.dims();
    let a_strides = a.strides();
    let b_strides = b.strides();
    let c_strides = c.strides();

    // Extract dimension groups
    let batch_dims = &a_dims[..n_batch];
    let lo_dims = &a_dims[n_batch..n_batch + n_lo];
    let sum_dims = &a_dims[n_batch + n_lo..n_batch + n_lo + n_sum];
    let ro_dims = &b_dims[n_batch + n_sum..n_batch + n_sum + n_ro];

    // Fused sizes for the matrix multiply
    let m: usize = lo_dims.iter().product::<usize>().max(1);
    let k: usize = sum_dims.iter().product::<usize>().max(1);
    let n: usize = ro_dims.iter().product::<usize>().max(1);

    // Extract stride groups
    let a_lo_strides = &a_strides[n_batch..n_batch + n_lo];
    let a_sum_strides = &a_strides[n_batch + n_lo..n_batch + n_lo + n_sum];
    let b_sum_strides = &b_strides[n_batch..n_batch + n_sum];
    let b_ro_strides = &b_strides[n_batch + n_sum..n_batch + n_sum + n_ro];
    let c_lo_strides = &c_strides[n_batch..n_batch + n_lo];
    let c_ro_strides = &c_strides[n_batch + n_lo..n_batch + n_lo + n_ro];

    // Try to fuse each dimension group
    let fused_a_lo = try_fuse_group(lo_dims, a_lo_strides);
    let fused_a_sum = try_fuse_group(sum_dims, a_sum_strides);
    let fused_b_sum = try_fuse_group(sum_dims, b_sum_strides);
    let fused_b_ro = try_fuse_group(ro_dims, b_ro_strides);
    let fused_c_lo = try_fuse_group(lo_dims, c_lo_strides);
    let fused_c_ro = try_fuse_group(ro_dims, c_ro_strides);

    let a_needs_copy = fused_a_lo.is_none() || fused_a_sum.is_none();
    let b_needs_copy = fused_b_sum.is_none() || fused_b_ro.is_none();
    let c_needs_copy = fused_c_lo.is_none() || fused_c_ro.is_none();

    // Copy A to contiguous column-major if inner dims aren't fusable
    let a_contig_buf: Option<StridedArray<T>>;
    let (a_ptr, a_row_stride, a_col_stride, a_batch_strides_vec);
    if a_needs_copy {
        let (mut buf, batch_strides) = alloc_batched_col_major(a.dims(), n_batch);
        strided::copy_into(&mut buf.view_mut(), a)?;
        a_ptr = buf.view().ptr();
        // Col-major inner A [lo..., sum...]: lo stride = 1, sum stride = m
        a_row_stride = if m == 0 { 0 } else { 1isize };
        a_col_stride = m as isize;
        a_batch_strides_vec = batch_strides;
        a_contig_buf = Some(buf);
    } else {
        let (_, rs) = fused_a_lo.unwrap();
        let (_, cs) = fused_a_sum.unwrap();
        a_ptr = a.ptr();
        a_row_stride = rs;
        a_col_stride = cs;
        a_batch_strides_vec = a_strides[..n_batch].to_vec();
        a_contig_buf = None;
    }
    let _ = &a_contig_buf; // keep alive

    // Copy B to contiguous column-major if inner dims aren't fusable
    let b_contig_buf: Option<StridedArray<T>>;
    let (b_ptr, b_row_stride, b_col_stride, b_batch_strides_vec);
    if b_needs_copy {
        let (mut buf, batch_strides) = alloc_batched_col_major(b.dims(), n_batch);
        strided::copy_into(&mut buf.view_mut(), b)?;
        b_ptr = buf.view().ptr();
        // Col-major inner B [sum..., ro...]: sum stride = 1, ro stride = k
        b_row_stride = if k == 0 { 0 } else { 1isize };
        b_col_stride = k as isize;
        b_batch_strides_vec = batch_strides;
        b_contig_buf = Some(buf);
    } else {
        let (_, rs) = fused_b_sum.unwrap();
        let (_, cs) = fused_b_ro.unwrap();
        b_ptr = b.ptr();
        b_row_stride = rs;
        b_col_stride = cs;
        b_batch_strides_vec = b_strides[..n_batch].to_vec();
        b_contig_buf = None;
    }
    let _ = &b_contig_buf; // keep alive

    // Copy C to contiguous column-major if inner dims aren't fusable
    let c_contig_buf: Option<StridedArray<T>>;
    let (c_ptr, c_row_stride, c_col_stride, c_batch_strides_vec);
    if c_needs_copy {
        let (mut buf, batch_strides) = alloc_batched_col_major(c.dims(), n_batch);
        if beta != T::zero() {
            strided::copy_into(&mut buf.view_mut(), &c.as_view())?;
        }
        c_ptr = buf.view_mut().as_mut_ptr();
        // Col-major inner C [lo..., ro...]: lo stride = 1, ro stride = m
        c_row_stride = if m == 0 { 0 } else { 1isize };
        c_col_stride = m as isize;
        c_batch_strides_vec = batch_strides;
        c_contig_buf = Some(buf);
    } else {
        let (_, rs) = fused_c_lo.unwrap();
        let (_, cs) = fused_c_ro.unwrap();
        c_ptr = c.as_mut_ptr();
        c_row_stride = rs;
        c_col_stride = cs;
        c_batch_strides_vec = c_strides[..n_batch].to_vec();
        c_contig_buf = None;
    }

    let is_beta_zero = beta == T::zero();
    let is_beta_one = beta == T::one();

    // Determine accumulation mode
    let accum = if is_beta_zero {
        Accum::Replace
    } else {
        Accum::Add
    };

    let mut batch_iter = MultiIndex::new(batch_dims);
    while batch_iter.next().is_some() {
        let a_batch_off = batch_iter.offset(&a_batch_strides_vec);
        let b_batch_off = batch_iter.offset(&b_batch_strides_vec);
        let c_batch_off = batch_iter.offset(&c_batch_strides_vec);

        // Pre-scale C by beta if beta is not 0 or 1
        if !is_beta_zero && !is_beta_one {
            let c_base = unsafe { c_ptr.offset(c_batch_off) };
            for i in 0..m {
                for j in 0..n {
                    let offset = i as isize * c_row_stride + j as isize * c_col_stride;
                    unsafe {
                        let elem = c_base.offset(offset);
                        *elem = beta * *elem;
                    }
                }
            }
        }

        unsafe {
            let a_mat: MatRef<'_, T> =
                MatRef::from_raw_parts(a_ptr.offset(a_batch_off), m, k, a_row_stride, a_col_stride);
            let b_mat: MatRef<'_, T> =
                MatRef::from_raw_parts(b_ptr.offset(b_batch_off), k, n, b_row_stride, b_col_stride);
            let c_mat: MatMut<'_, T> = MatMut::from_raw_parts_mut(
                c_ptr.offset(c_batch_off),
                m,
                n,
                c_row_stride,
                c_col_stride,
            );

            matmul(c_mat, accum, a_mat, b_mat, alpha, Par::Seq);
        }
    }

    // If C was copied to a temp buffer, copy the result back
    if let Some(ref c_buf) = c_contig_buf {
        strided::copy_into(c, &c_buf.view())?;
    }

    Ok(())
}

/// Allocate a StridedArray with column-major inner dims and row-major batch dims.
///
/// For dims `[batch..., inner...]`, the inner dimensions are stored column-major
/// (first inner dim has stride 1), while batch dimensions are stored row-major
/// (outermost batch dim has the largest stride). This ensures each batch slice
/// is a contiguous column-major matrix, which faer prefers.
fn alloc_batched_col_major<T: num_traits::Zero + Clone>(
    dims: &[usize],
    n_batch: usize,
) -> (StridedArray<T>, Vec<isize>) {
    let total: usize = dims.iter().product::<usize>().max(1);
    let data = vec![T::zero(); total];

    // Inner dims: column-major (stride 1 for first inner dim)
    let inner_dims = &dims[n_batch..];
    let mut strides = vec![0isize; dims.len()];
    if !inner_dims.is_empty() {
        strides[n_batch] = 1;
        for i in 1..inner_dims.len() {
            strides[n_batch + i] = strides[n_batch + i - 1] * inner_dims[i - 1] as isize;
        }
    }

    // Batch dims: row-major (outermost has largest stride)
    let inner_size: usize = inner_dims.iter().product::<usize>().max(1);
    if n_batch > 0 {
        strides[n_batch - 1] = inner_size as isize;
        for i in (0..n_batch - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1] as isize;
        }
    }

    let batch_strides = strides[..n_batch].to_vec();
    let arr =
        StridedArray::from_parts(data, dims, &strides, 0).expect("batched col-major allocation");
    (arr, batch_strides)
}

#[cfg(test)]
mod tests {
    use super::*;
    use stridedview::StridedArray;

    #[test]
    fn test_faer_bgemm_2x2() {
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[5.0, 6.0], [7.0, 8.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f64>::row_major(&[2, 2]);

        bgemm_strided_into(
            &mut c.view_mut(),
            &a.view(),
            &b.view(),
            0,
            1,
            1,
            1,
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
    fn test_faer_bgemm_rect() {
        let a =
            StridedArray::<f64>::from_fn_row_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
        let b =
            StridedArray::<f64>::from_fn_row_major(&[3, 4], |idx| (idx[0] * 4 + idx[1] + 1) as f64);
        let mut c = StridedArray::<f64>::row_major(&[2, 4]);

        bgemm_strided_into(
            &mut c.view_mut(),
            &a.view(),
            &b.view(),
            0,
            1,
            1,
            1,
            1.0,
            0.0,
        )
        .unwrap();

        assert_eq!(c.get(&[0, 0]), 38.0);
        assert_eq!(c.get(&[1, 3]), 128.0);
    }

    #[test]
    fn test_faer_bgemm_batched() {
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2, 3], |idx| {
            (idx[0] * 6 + idx[1] * 3 + idx[2] + 1) as f64
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 3, 2], |idx| {
            (idx[0] * 6 + idx[1] * 2 + idx[2] + 1) as f64
        });
        let mut c = StridedArray::<f64>::row_major(&[2, 2, 2]);

        bgemm_strided_into(
            &mut c.view_mut(),
            &a.view(),
            &b.view(),
            1,
            1,
            1,
            1,
            1.0,
            0.0,
        )
        .unwrap();

        // Batch 0: A0=[[1,2,3],[4,5,6]], B0=[[1,2],[3,4],[5,6]]
        // C0[0,0] = 1*1+2*3+3*5 = 22
        assert_eq!(c.get(&[0, 0, 0]), 22.0);
    }

    #[test]
    fn test_faer_bgemm_beta_zero() {
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[5.0, 6.0], [7.0, 8.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[100.0, 200.0], [300.0, 400.0]][idx[0]][idx[1]]
        });

        bgemm_strided_into(
            &mut c.view_mut(),
            &a.view(),
            &b.view(),
            0,
            1,
            1,
            1,
            1.0,
            0.0, // beta=0: C_old should be ignored
        )
        .unwrap();

        assert_eq!(c.get(&[0, 0]), 19.0);
        assert_eq!(c.get(&[1, 1]), 50.0);
    }

    #[test]
    fn test_faer_bgemm_beta_one() {
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 0.0], [0.0, 1.0]][idx[0]][idx[1]] // identity
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[10.0, 20.0], [30.0, 40.0]][idx[0]][idx[1]]
        });

        bgemm_strided_into(
            &mut c.view_mut(),
            &a.view(),
            &b.view(),
            0,
            1,
            1,
            1,
            1.0,
            1.0, // beta=1: C = A*B + C_old
        )
        .unwrap();

        // C[0,0] = 1*1+0*3 + 10 = 11
        assert_eq!(c.get(&[0, 0]), 11.0);
        // C[1,1] = 0*2+1*4 + 40 = 44
        assert_eq!(c.get(&[1, 1]), 44.0);
    }

    #[test]
    fn test_faer_bgemm_alpha_beta() {
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 0.0], [0.0, 1.0]][idx[0]][idx[1]] // identity
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[10.0, 20.0], [30.0, 40.0]][idx[0]][idx[1]]
        });

        bgemm_strided_into(
            &mut c.view_mut(),
            &a.view(),
            &b.view(),
            0,
            1,
            1,
            1,
            2.0,
            3.0, // C = 2*I*B + 3*C_old
        )
        .unwrap();

        // C[0,0] = 2*1 + 3*10 = 32
        assert_eq!(c.get(&[0, 0]), 32.0);
        // C[1,1] = 2*4 + 3*40 = 128
        assert_eq!(c.get(&[1, 1]), 128.0);
    }

    #[test]
    fn test_faer_bgemm_outer_product() {
        let a = StridedArray::<f64>::from_fn_row_major(&[3], |idx| (idx[0] + 1) as f64);
        let b = StridedArray::<f64>::from_fn_row_major(&[4], |idx| (idx[0] + 1) as f64);
        let mut c = StridedArray::<f64>::row_major(&[3, 4]);

        bgemm_strided_into(
            &mut c.view_mut(),
            &a.view(),
            &b.view(),
            0,
            1,
            1,
            0, // no sum
            1.0,
            0.0,
        )
        .unwrap();

        assert_eq!(c.get(&[0, 0]), 1.0);
        assert_eq!(c.get(&[2, 3]), 12.0);
    }

    #[test]
    fn test_faer_bgemm_f32() {
        let a = StridedArray::<f32>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0f32, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        let b = StridedArray::<f32>::from_fn_row_major(&[2, 2], |idx| {
            [[5.0f32, 6.0], [7.0, 8.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f32>::row_major(&[2, 2]);

        bgemm_strided_into(
            &mut c.view_mut(),
            &a.view(),
            &b.view(),
            0,
            1,
            1,
            1,
            1.0f32,
            0.0f32,
        )
        .unwrap();

        assert_eq!(c.get(&[0, 0]), 19.0f32);
        assert_eq!(c.get(&[1, 1]), 50.0f32);
    }

    #[test]
    fn test_faer_bgemm_col_major_input() {
        // A is col-major (non-contiguous for row-major fusion) → triggers copy path
        let a_data = vec![1.0, 3.0, 2.0, 4.0]; // col-major [[1,2],[3,4]]
        let a = StridedArray::<f64>::from_parts(a_data, &[2, 2], &[1, 2], 0).unwrap();

        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[5.0, 6.0], [7.0, 8.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f64>::row_major(&[2, 2]);

        bgemm_strided_into(
            &mut c.view_mut(),
            &a.view(),
            &b.view(),
            0,
            1,
            1,
            1,
            1.0,
            0.0,
        )
        .unwrap();

        // Same A=[[1,2],[3,4]], B=[[5,6],[7,8]]
        assert_eq!(c.get(&[0, 0]), 19.0);
        assert_eq!(c.get(&[0, 1]), 22.0);
        assert_eq!(c.get(&[1, 0]), 43.0);
        assert_eq!(c.get(&[1, 1]), 50.0);
    }

    #[test]
    fn test_faer_bgemm_col_major_output() {
        // C is col-major → triggers C copy path
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[5.0, 6.0], [7.0, 8.0]][idx[0]][idx[1]]
        });
        let mut c = StridedArray::<f64>::col_major(&[2, 2]);

        bgemm_strided_into(
            &mut c.view_mut(),
            &a.view(),
            &b.view(),
            0,
            1,
            1,
            1,
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
    fn test_faer_bgemm_col_major_with_beta() {
        // C is col-major with beta != 0 → copy C in, matmul, copy back
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 0.0], [0.0, 1.0]][idx[0]][idx[1]] // identity
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[2, 2], |idx| {
            [[1.0, 2.0], [3.0, 4.0]][idx[0]][idx[1]]
        });
        // C col-major with initial values
        let c_data = vec![10.0, 30.0, 20.0, 40.0]; // col-major [[10,20],[30,40]]
        let mut c = StridedArray::<f64>::from_parts(c_data, &[2, 2], &[1, 2], 0).unwrap();

        bgemm_strided_into(
            &mut c.view_mut(),
            &a.view(),
            &b.view(),
            0,
            1,
            1,
            1,
            2.0,
            3.0, // C = 2*I*B + 3*C_old
        )
        .unwrap();

        // C[0,0] = 2*1 + 3*10 = 32
        assert_eq!(c.get(&[0, 0]), 32.0);
        // C[1,1] = 2*4 + 3*40 = 128
        assert_eq!(c.get(&[1, 1]), 128.0);
    }
}
