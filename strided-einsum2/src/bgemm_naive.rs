//! Naive batched GEMM kernel on strided views.
//!
//! Operates on N-dimensional permuted views where dimensions are grouped as:
//! - A: [lo..., sum..., batch...]
//! - B: [sum..., ro..., batch...]
//! - C: [lo..., ro..., batch...]

use crate::util::MultiIndex;
use strided_view::{ElementOp, ElementOpApply, StridedView, StridedViewMut};

/// Batched strided GEMM: C = alpha * A * B + beta * C
///
/// The views must be pre-permuted so that their dimensions are grouped as
/// (batch-last canonical order):
/// - A: `n_lo` dims, then `n_sum` dims, then `n_batch` batch dims
/// - B: `n_sum` dims, then `n_ro` dims, then `n_batch` batch dims
/// - C: `n_lo` dims, then `n_ro` dims, then `n_batch` batch dims
///
/// Dimension sizes must match across operands within each group.
pub fn bgemm_strided_into<T>(
    c: &mut StridedViewMut<T>,
    a: &StridedView<T>,
    b: &StridedView<T>,
    _n_batch: usize,
    n_lo: usize,
    n_ro: usize,
    n_sum: usize,
    alpha: T,
    beta: T,
    conj_a: bool,
    conj_b: bool,
) -> strided_view::Result<()>
where
    T: Copy
        + ElementOpApply
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + num_traits::Zero
        + num_traits::One
        + PartialEq,
{
    let a_dims = a.dims();
    let b_dims = b.dims();
    let c_dims = c.dims();
    let a_strides = a.strides();
    let b_strides = b.strides();
    let c_strides = c.strides();

    // Extract dimension groups (batch-last canonical order)
    let lo_dims = &a_dims[..n_lo];
    let sum_dims = &a_dims[n_lo..n_lo + n_sum];
    let batch_dims = &a_dims[n_lo + n_sum..];
    let ro_dims = &b_dims[n_sum..n_sum + n_ro];

    // Extract stride groups (batch-last)
    let a_lo_strides = &a_strides[..n_lo];
    let a_sum_strides = &a_strides[n_lo..n_lo + n_sum];
    let a_batch_strides = &a_strides[n_lo + n_sum..];

    let b_sum_strides = &b_strides[..n_sum];
    let b_ro_strides = &b_strides[n_sum..n_sum + n_ro];
    let b_batch_strides = &b_strides[n_sum + n_ro..];

    let c_lo_strides = &c_strides[..n_lo];
    let c_ro_strides = &c_strides[n_lo..n_lo + n_ro];
    let c_batch_strides = &c_strides[n_lo + n_ro..];

    // Validate dimension consistency
    debug_assert_eq!(&c_dims[..n_lo], lo_dims);
    debug_assert_eq!(&c_dims[n_lo..n_lo + n_ro], ro_dims);
    debug_assert_eq!(&c_dims[n_lo + n_ro..], batch_dims);
    debug_assert_eq!(&b_dims[..n_sum], sum_dims);
    debug_assert_eq!(&b_dims[n_sum + n_ro..], batch_dims);

    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let c_ptr = c.as_mut_ptr();

    let is_beta_zero = beta == T::zero();
    let is_alpha_one = alpha == T::one();

    let mut batch_iter = MultiIndex::new(batch_dims);
    let mut lo_iter = MultiIndex::new(lo_dims);
    let mut ro_iter = MultiIndex::new(ro_dims);
    let mut sum_iter = MultiIndex::new(sum_dims);
    while batch_iter.next().is_some() {
        let a_batch_off = batch_iter.offset(a_batch_strides);
        let b_batch_off = batch_iter.offset(b_batch_strides);
        let c_batch_off = batch_iter.offset(c_batch_strides);

        lo_iter.reset();
        while lo_iter.next().is_some() {
            let a_lo_off = lo_iter.offset(a_lo_strides);
            let c_lo_off = lo_iter.offset(c_lo_strides);

            ro_iter.reset();
            while ro_iter.next().is_some() {
                let b_ro_off = ro_iter.offset(b_ro_strides);
                let c_ro_off = ro_iter.offset(c_ro_strides);

                // Accumulate sum over contraction indices
                let mut acc = T::zero();
                sum_iter.reset();
                while sum_iter.next().is_some() {
                    let a_sum_off = sum_iter.offset(a_sum_strides);
                    let b_sum_off = sum_iter.offset(b_sum_strides);

                    let a_raw = unsafe { *a_ptr.offset(a_batch_off + a_lo_off + a_sum_off) };
                    let b_raw = unsafe { *b_ptr.offset(b_batch_off + b_sum_off + b_ro_off) };
                    let a_val = if conj_a {
                        strided_view::Conj::apply(a_raw)
                    } else {
                        a_raw
                    };
                    let b_val = if conj_b {
                        strided_view::Conj::apply(b_raw)
                    } else {
                        b_raw
                    };
                    acc = acc + a_val * b_val;
                }

                // Write: c = alpha * acc + beta * c_old
                let c_off = c_batch_off + c_lo_off + c_ro_off;
                unsafe {
                    let c_elem = c_ptr.offset(c_off);
                    if is_beta_zero {
                        if is_alpha_one {
                            *c_elem = acc;
                        } else {
                            *c_elem = alpha * acc;
                        }
                    } else {
                        let old = *c_elem;
                        if is_alpha_one {
                            *c_elem = acc + beta * old;
                        } else {
                            *c_elem = alpha * acc + beta * old;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Batched strided GEMM with closure-based element mapping: C = alpha * map_a(A) * map_b(B) + beta * C
///
/// Like [`bgemm_strided_into`] but uses closures instead of conjugation flags,
/// allowing custom scalar types that don't implement `ElementOpApply`.
pub fn bgemm_strided_into_with_map<T, MapA, MapB>(
    c: &mut StridedViewMut<T>,
    a: &StridedView<T>,
    b: &StridedView<T>,
    _n_batch: usize,
    n_lo: usize,
    n_ro: usize,
    n_sum: usize,
    alpha: T,
    beta: T,
    map_a: MapA,
    map_b: MapB,
) -> strided_view::Result<()>
where
    T: Copy
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + num_traits::Zero
        + num_traits::One
        + PartialEq,
    MapA: Fn(T) -> T,
    MapB: Fn(T) -> T,
{
    let a_dims = a.dims();
    let b_dims = b.dims();
    let c_dims = c.dims();
    let a_strides = a.strides();
    let b_strides = b.strides();
    let c_strides = c.strides();

    let lo_dims = &a_dims[..n_lo];
    let sum_dims = &a_dims[n_lo..n_lo + n_sum];
    let batch_dims = &a_dims[n_lo + n_sum..];
    let ro_dims = &b_dims[n_sum..n_sum + n_ro];

    let a_lo_strides = &a_strides[..n_lo];
    let a_sum_strides = &a_strides[n_lo..n_lo + n_sum];
    let a_batch_strides = &a_strides[n_lo + n_sum..];

    let b_sum_strides = &b_strides[..n_sum];
    let b_ro_strides = &b_strides[n_sum..n_sum + n_ro];
    let b_batch_strides = &b_strides[n_sum + n_ro..];

    let c_lo_strides = &c_strides[..n_lo];
    let c_ro_strides = &c_strides[n_lo..n_lo + n_ro];
    let c_batch_strides = &c_strides[n_lo + n_ro..];

    debug_assert_eq!(&c_dims[..n_lo], lo_dims);
    debug_assert_eq!(&c_dims[n_lo..n_lo + n_ro], ro_dims);
    debug_assert_eq!(&c_dims[n_lo + n_ro..], batch_dims);
    debug_assert_eq!(&b_dims[..n_sum], sum_dims);
    debug_assert_eq!(&b_dims[n_sum + n_ro..], batch_dims);

    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let c_ptr = c.as_mut_ptr();

    let is_beta_zero = beta == T::zero();
    let is_alpha_one = alpha == T::one();

    let mut batch_iter = MultiIndex::new(batch_dims);
    let mut lo_iter = MultiIndex::new(lo_dims);
    let mut ro_iter = MultiIndex::new(ro_dims);
    let mut sum_iter = MultiIndex::new(sum_dims);
    while batch_iter.next().is_some() {
        let a_batch_off = batch_iter.offset(a_batch_strides);
        let b_batch_off = batch_iter.offset(b_batch_strides);
        let c_batch_off = batch_iter.offset(c_batch_strides);

        lo_iter.reset();
        while lo_iter.next().is_some() {
            let a_lo_off = lo_iter.offset(a_lo_strides);
            let c_lo_off = lo_iter.offset(c_lo_strides);

            ro_iter.reset();
            while ro_iter.next().is_some() {
                let b_ro_off = ro_iter.offset(b_ro_strides);
                let c_ro_off = ro_iter.offset(c_ro_strides);

                let mut acc = T::zero();
                sum_iter.reset();
                while sum_iter.next().is_some() {
                    let a_sum_off = sum_iter.offset(a_sum_strides);
                    let b_sum_off = sum_iter.offset(b_sum_strides);

                    let a_raw = unsafe { *a_ptr.offset(a_batch_off + a_lo_off + a_sum_off) };
                    let b_raw = unsafe { *b_ptr.offset(b_batch_off + b_sum_off + b_ro_off) };
                    acc = acc + map_a(a_raw) * map_b(b_raw);
                }

                let c_off = c_batch_off + c_lo_off + c_ro_off;
                unsafe {
                    let c_elem = c_ptr.offset(c_off);
                    if is_beta_zero {
                        if is_alpha_one {
                            *c_elem = acc;
                        } else {
                            *c_elem = alpha * acc;
                        }
                    } else {
                        let old = *c_elem;
                        if is_alpha_one {
                            *c_elem = acc + beta * old;
                        } else {
                            *c_elem = alpha * acc + beta * old;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use strided_view::StridedArray;

    #[test]
    fn test_bgemm_2x2() {
        // Simple 2x2 matmul: C = A * B
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // C = [[19, 22], [43, 50]]
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
            1, // n_batch=0, n_lo=1(i), n_ro=1(k), n_sum=1(j)
            1.0,
            0.0,
            false,
            false,
        )
        .unwrap();

        assert_eq!(c.get(&[0, 0]), 19.0);
        assert_eq!(c.get(&[0, 1]), 22.0);
        assert_eq!(c.get(&[1, 0]), 43.0);
        assert_eq!(c.get(&[1, 1]), 50.0);
    }

    #[test]
    fn test_bgemm_rect() {
        // A: 2x3, B: 3x4, C: 2x4
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
            false,
            false,
        )
        .unwrap();

        // A = [[1,2,3],[4,5,6]]
        // B = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
        // C[0,0] = 1*1+2*5+3*9 = 38
        assert_eq!(c.get(&[0, 0]), 38.0);
        // C[1,3] = 4*4+5*8+6*12 = 16+40+72 = 128
        assert_eq!(c.get(&[1, 3]), 128.0);
    }

    #[test]
    fn test_bgemm_batched() {
        // Batch=2, lo=2, sum=3, ro=2
        // Batch-last: A: [lo, sum, batch]=[2,3,2], B: [sum, ro, batch]=[3,2,2], C: [lo, ro, batch]=[2,2,2]
        let a = StridedArray::<f64>::from_fn_row_major(&[2, 3, 2], |idx| {
            // idx=[lo, sum, batch] → same values as batch*6 + lo*3 + sum + 1
            (idx[2] * 6 + idx[0] * 3 + idx[1] + 1) as f64
        });
        let b = StridedArray::<f64>::from_fn_row_major(&[3, 2, 2], |idx| {
            // idx=[sum, ro, batch] → same values as batch*6 + sum*2 + ro + 1
            (idx[2] * 6 + idx[0] * 2 + idx[1] + 1) as f64
        });
        let mut c = StridedArray::<f64>::row_major(&[2, 2, 2]);

        bgemm_strided_into(
            &mut c.view_mut(),
            &a.view(),
            &b.view(),
            1,
            1,
            1,
            1, // n_batch=1, n_lo=1, n_ro=1, n_sum=1
            1.0,
            0.0,
            false,
            false,
        )
        .unwrap();

        // C: [lo, ro, batch]
        // Batch 0: A0=[[1,2,3],[4,5,6]], B0=[[1,2],[3,4],[5,6]]
        // C0[0,0] = 1*1+2*3+3*5 = 22
        assert_eq!(c.get(&[0, 0, 0]), 22.0);
    }

    #[test]
    fn test_bgemm_alpha_beta() {
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

        bgemm_strided_into(
            &mut c.view_mut(),
            &a.view(),
            &b.view(),
            0,
            1,
            1,
            1,
            2.0,
            3.0, // alpha=2, beta=3
            false,
            false,
        )
        .unwrap();

        // C = 2 * I * B + 3 * C_old = 2*B + 3*C_old
        // C[0,0] = 2*1 + 3*10 = 32
        assert_eq!(c.get(&[0, 0]), 32.0);
        // C[1,1] = 2*4 + 3*40 = 128
        assert_eq!(c.get(&[1, 1]), 128.0);
    }

    #[test]
    fn test_bgemm_outer_product() {
        // Outer product: no sum dims
        // a: [3], b: [4], c: [3, 4]
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
            0, // no batch, no sum
            1.0,
            0.0,
            false,
            false,
        )
        .unwrap();

        assert_eq!(c.get(&[0, 0]), 1.0);
        assert_eq!(c.get(&[2, 3]), 12.0);
    }
}
