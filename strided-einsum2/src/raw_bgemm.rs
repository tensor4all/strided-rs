//! Raw borrowed-layout batched GEMM entry points.
//!
//! This module is the prepared-replay boundary for callers that already own
//! validated layout metadata. It keeps the public API independent of a concrete
//! GEMM backend while still allowing backend modules to provide specialized raw
//! implementations.

use crate::backend::Backend;
use crate::{contiguous, Scalar, ScalarBase};
use strided_view::{Conj, ElementOp, ElementOpApply, RawStridedMut, RawStridedRef};

#[derive(Clone, Copy)]
pub(crate) struct BgemmGroupLayout {
    pub(crate) a_sum_end: usize,
    pub(crate) a_rank: usize,
    pub(crate) b_ro_end: usize,
    pub(crate) b_rank: usize,
    pub(crate) c_ro_end: usize,
    pub(crate) c_rank: usize,
    pub(crate) label_len: usize,
}

pub(crate) fn checked_bgemm_group_layout(
    n_batch: usize,
    n_lo: usize,
    n_ro: usize,
    n_sum: usize,
) -> crate::Result<BgemmGroupLayout> {
    let a_sum_end = n_lo
        .checked_add(n_sum)
        .ok_or(strided_view::StridedError::OffsetOverflow)?;
    let a_rank = a_sum_end
        .checked_add(n_batch)
        .ok_or(strided_view::StridedError::OffsetOverflow)?;
    let b_ro_end = n_sum
        .checked_add(n_ro)
        .ok_or(strided_view::StridedError::OffsetOverflow)?;
    let b_rank = b_ro_end
        .checked_add(n_batch)
        .ok_or(strided_view::StridedError::OffsetOverflow)?;
    let c_ro_end = n_lo
        .checked_add(n_ro)
        .ok_or(strided_view::StridedError::OffsetOverflow)?;
    let c_rank = c_ro_end
        .checked_add(n_batch)
        .ok_or(strided_view::StridedError::OffsetOverflow)?;
    let label_len = c_rank
        .checked_add(n_sum)
        .ok_or(strided_view::StridedError::OffsetOverflow)?;
    Ok(BgemmGroupLayout {
        a_sum_end,
        a_rank,
        b_ro_end,
        b_rank,
        c_ro_end,
        c_rank,
        label_len,
    })
}

/// Batched strided GEMM on raw borrowed layout metadata using the active backend.
///
/// This is the raw-layout counterpart to backend-specific `bgemm_strided_into`
/// functions. It avoids constructing owned-metadata `StridedView` wrappers when
/// a caller already has borrowed `dims`/`strides`/`offset` descriptors.
#[allow(clippy::too_many_arguments)]
pub fn bgemm_raw_strided_into<T>(
    c: RawStridedMut<'_, T>,
    a: RawStridedRef<'_, T>,
    b: RawStridedRef<'_, T>,
    n_batch: usize,
    n_lo: usize,
    n_ro: usize,
    n_sum: usize,
    alpha: T,
    beta: T,
    conj_a: bool,
    conj_b: bool,
) -> crate::Result<()>
where
    T: Scalar,
    crate::backend::ActiveBackend: Backend<T>,
{
    validate_bgemm_shapes(&c, &a, &b, n_batch, n_lo, n_ro, n_sum)?;
    unsafe {
        bgemm_raw_strided_into_unchecked(
            c, a, b, n_batch, n_lo, n_ro, n_sum, alpha, beta, conj_a, conj_b,
        )
    }
}

/// Batched strided GEMM on raw borrowed layout metadata without validation.
///
/// # Safety
/// The caller must ensure:
/// - all raw strided operands are in bounds,
/// - `n_lo`, `n_ro`, `n_sum`, and `n_batch` partition operand ranks as
///   `[lo, sum, batch]`, `[sum, ro, batch]`, and `[lo, ro, batch]`,
/// - matching dimension groups have identical extents,
/// - `c` does not alias `a` or `b` in a way that violates mutable access.
#[allow(clippy::too_many_arguments)]
pub unsafe fn bgemm_raw_strided_into_unchecked<T>(
    c: RawStridedMut<'_, T>,
    a: RawStridedRef<'_, T>,
    b: RawStridedRef<'_, T>,
    n_batch: usize,
    n_lo: usize,
    n_ro: usize,
    n_sum: usize,
    alpha: T,
    beta: T,
    conj_a: bool,
    conj_b: bool,
) -> crate::Result<()>
where
    T: Scalar,
    crate::backend::ActiveBackend: Backend<T>,
{
    bgemm_raw_with_backend_into_unchecked::<T, crate::backend::ActiveBackend>(
        c, a, b, n_batch, n_lo, n_ro, n_sum, alpha, beta, conj_a, conj_b,
    )
}

/// Batched strided GEMM on raw borrowed metadata using an explicit backend.
///
/// Backend implementations that do not provide a specialized raw path use the
/// same preparation pipeline as `einsum2_dispatch`: materialize/copy only when
/// the backend requires it, call `Backend::bgemm_contiguous_into`, then finalize
/// the destination.
#[allow(clippy::too_many_arguments)]
pub fn bgemm_raw_with_backend_into<T, B>(
    c: RawStridedMut<'_, T>,
    a: RawStridedRef<'_, T>,
    b: RawStridedRef<'_, T>,
    n_batch: usize,
    n_lo: usize,
    n_ro: usize,
    n_sum: usize,
    alpha: T,
    beta: T,
    conj_a: bool,
    conj_b: bool,
) -> crate::Result<()>
where
    T: ScalarBase + ElementOpApply,
    B: Backend<T>,
{
    validate_bgemm_shapes(&c, &a, &b, n_batch, n_lo, n_ro, n_sum)?;
    unsafe {
        bgemm_raw_with_backend_into_unchecked::<T, B>(
            c, a, b, n_batch, n_lo, n_ro, n_sum, alpha, beta, conj_a, conj_b,
        )
    }
}

/// Unchecked variant of [`bgemm_raw_with_backend_into`].
///
/// # Safety
/// The caller must uphold the same layout and aliasing invariants as
/// [`bgemm_raw_strided_into_unchecked`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn bgemm_raw_with_backend_into_unchecked<T, B>(
    mut c: RawStridedMut<'_, T>,
    a: RawStridedRef<'_, T>,
    b: RawStridedRef<'_, T>,
    _n_batch: usize,
    n_lo: usize,
    n_ro: usize,
    n_sum: usize,
    alpha: T,
    beta: T,
    conj_a: bool,
    conj_b: bool,
) -> crate::Result<()>
where
    T: ScalarBase + ElementOpApply,
    B: Backend<T>,
{
    let a_dims = a.dims();
    let b_dims = b.dims();
    let lo_dims = &a_dims[..n_lo];
    let sum_dims = &a_dims[n_lo..n_lo + n_sum];
    let batch_dims = &a_dims[n_lo + n_sum..];
    let ro_dims = &b_dims[n_sum..n_sum + n_ro];

    if c.dims().iter().any(|&dim| dim == 0) {
        return Ok(());
    }
    if sum_dims.iter().any(|&dim| dim == 0) {
        scale_or_zero_raw_mut(&mut c, beta);
        return Ok(());
    }

    let use_pool = true;
    let materialize = if B::MATERIALIZES_CONJ {
        Some(Conj::apply as fn(T) -> T)
    } else {
        None
    };

    let a_op = contiguous::prepare_input_raw(
        &a,
        n_lo,
        n_sum,
        conj_a,
        B::REQUIRES_UNIT_STRIDE,
        use_pool,
        materialize,
    )?;
    let b_op = contiguous::prepare_input_raw(
        &b,
        n_sum,
        n_ro,
        conj_b,
        B::REQUIRES_UNIT_STRIDE,
        use_pool,
        materialize,
    )?;
    let mut c_op = contiguous::prepare_output_raw(
        &mut c,
        n_lo,
        n_ro,
        beta,
        B::REQUIRES_UNIT_STRIDE,
        use_pool,
    )?;

    let m: usize = lo_dims.iter().product::<usize>().max(1);
    let k: usize = sum_dims.iter().product::<usize>().max(1);
    let n: usize = ro_dims.iter().product::<usize>().max(1);

    B::bgemm_contiguous_into(&mut c_op, &a_op, &b_op, batch_dims, m, n, k, alpha, beta)?;
    c_op.finalize_raw_into(&mut c)?;

    Ok(())
}

pub(crate) fn validate_bgemm_shapes<T, U>(
    c: &RawStridedMut<'_, U>,
    a: &RawStridedRef<'_, T>,
    b: &RawStridedRef<'_, T>,
    n_batch: usize,
    n_lo: usize,
    n_ro: usize,
    n_sum: usize,
) -> crate::Result<()> {
    let groups = checked_bgemm_group_layout(n_batch, n_lo, n_ro, n_sum)?;
    if a.dims().len() != groups.a_rank {
        return Err(strided_view::StridedError::RankMismatch(groups.a_rank, a.dims().len()).into());
    }
    if b.dims().len() != groups.b_rank {
        return Err(strided_view::StridedError::RankMismatch(groups.b_rank, b.dims().len()).into());
    }
    if c.dims().len() != groups.c_rank {
        return Err(strided_view::StridedError::RankMismatch(groups.c_rank, c.dims().len()).into());
    }

    let lo_dims = &a.dims()[..n_lo];
    let sum_dims = &a.dims()[n_lo..groups.a_sum_end];
    let batch_dims = &a.dims()[groups.a_sum_end..];
    let ro_dims = &b.dims()[n_sum..groups.b_ro_end];

    if &b.dims()[..n_sum] != sum_dims {
        return Err(strided_view::StridedError::ShapeMismatch(
            sum_dims.to_vec(),
            b.dims()[..n_sum].to_vec(),
        )
        .into());
    }
    if &b.dims()[groups.b_ro_end..] != batch_dims {
        return Err(strided_view::StridedError::ShapeMismatch(
            batch_dims.to_vec(),
            b.dims()[groups.b_ro_end..].to_vec(),
        )
        .into());
    }
    if &c.dims()[..n_lo] != lo_dims {
        return Err(strided_view::StridedError::ShapeMismatch(
            lo_dims.to_vec(),
            c.dims()[..n_lo].to_vec(),
        )
        .into());
    }
    if &c.dims()[n_lo..groups.c_ro_end] != ro_dims {
        return Err(strided_view::StridedError::ShapeMismatch(
            ro_dims.to_vec(),
            c.dims()[n_lo..groups.c_ro_end].to_vec(),
        )
        .into());
    }
    if &c.dims()[groups.c_ro_end..] != batch_dims {
        return Err(strided_view::StridedError::ShapeMismatch(
            batch_dims.to_vec(),
            c.dims()[groups.c_ro_end..].to_vec(),
        )
        .into());
    }
    Ok(())
}

pub(crate) fn scale_or_zero_raw_mut<T: ScalarBase>(c: &mut RawStridedMut<'_, T>, beta: T) {
    if c.dims().iter().any(|&dim| dim == 0) {
        return;
    }

    fn visit<T: ScalarBase>(
        ptr: *mut T,
        dims: &[usize],
        strides: &[isize],
        axis: usize,
        offset: isize,
        beta: T,
        zero: T,
    ) {
        if axis == dims.len() {
            unsafe {
                let dst = ptr.offset(offset);
                if beta == zero {
                    *dst = zero;
                } else {
                    *dst = beta * *dst;
                }
            }
            return;
        }

        for i in 0..dims[axis] {
            visit(
                ptr,
                dims,
                strides,
                axis + 1,
                offset + i as isize * strides[axis],
                beta,
                zero,
            );
        }
    }

    visit(c.as_mut_ptr(), c.dims(), c.strides(), 0, 0, beta, T::zero());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_bgemm_2x2<T>(one: T, zero: T) -> Vec<T>
    where
        T: Scalar,
        crate::backend::ActiveBackend: Backend<T>,
        T: From<f32>,
    {
        let dims = [2, 2];
        let strides = [2, 1];
        let a_data = [T::from(1.0), T::from(2.0), T::from(3.0), T::from(4.0)];
        let b_data = [T::from(5.0), T::from(6.0), T::from(7.0), T::from(8.0)];
        let mut c_data = vec![zero; 4];
        let a = RawStridedRef::new(&a_data, &dims, &strides, 0).unwrap();
        let b = RawStridedRef::new(&b_data, &dims, &strides, 0).unwrap();
        let c = RawStridedMut::new(&mut c_data, &dims, &strides, 0).unwrap();
        bgemm_raw_strided_into(c, a, b, 0, 1, 1, 1, one, zero, false, false).unwrap();
        c_data
    }

    #[test]
    fn raw_bgemm_active_backend_f64() {
        assert_eq!(raw_bgemm_2x2(1.0f64, 0.0), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn raw_bgemm_active_backend_f32() {
        assert_eq!(raw_bgemm_2x2(1.0f32, 0.0), vec![19.0f32, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn raw_bgemm_active_backend_complex_conj() {
        use num_complex::Complex64;

        let i = Complex64::i();
        let dims = [2, 2];
        let strides = [2, 1];
        let a_data = [
            Complex64::new(1.0, 0.0) + i,
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0) - i,
        ];
        let b_data = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let mut c_data = vec![Complex64::new(0.0, 0.0); 4];
        let a = RawStridedRef::new(&a_data, &dims, &strides, 0).unwrap();
        let b = RawStridedRef::new(&b_data, &dims, &strides, 0).unwrap();
        let c = RawStridedMut::new(&mut c_data, &dims, &strides, 0).unwrap();
        bgemm_raw_strided_into(
            c,
            a,
            b,
            0,
            1,
            1,
            1,
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            true,
            false,
        )
        .unwrap();
        assert_eq!(
            c_data,
            vec![
                Complex64::new(1.0, -1.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 1.0),
            ]
        );
    }

    #[test]
    fn raw_bgemm_active_backend_checked_shape_mismatch() {
        let a_dims = [2, 2];
        let b_dims = [3, 2];
        let c_dims = [2, 2];
        let a_strides = [2, 1];
        let b_strides = [2, 1];
        let c_strides = [2, 1];
        let a_data = [1.0, 2.0, 3.0, 4.0];
        let b_data = [0.0; 6];
        let mut c_data = [0.0; 4];
        let a = RawStridedRef::new(&a_data, &a_dims, &a_strides, 0).unwrap();
        let b = RawStridedRef::new(&b_data, &b_dims, &b_strides, 0).unwrap();
        let c = RawStridedMut::new(&mut c_data, &c_dims, &c_strides, 0).unwrap();
        let err = bgemm_raw_strided_into(c, a, b, 0, 1, 1, 1, 1.0, 0.0, false, false).unwrap_err();
        assert!(matches!(
            err,
            crate::EinsumError::Strided(strided_view::StridedError::ShapeMismatch(_, _))
        ));
    }

    #[test]
    fn raw_bgemm_explicit_backend_checked_rank_mismatch() {
        let a_dims = [2, 2];
        let b_dims = [2, 2];
        let c_dims = [2];
        let strides = [2, 1];
        let c_strides = [1];
        let a_data = [1.0, 2.0, 3.0, 4.0];
        let b_data = [5.0, 6.0, 7.0, 8.0];
        let mut c_data = [0.0; 2];
        let a = RawStridedRef::new(&a_data, &a_dims, &strides, 0).unwrap();
        let b = RawStridedRef::new(&b_data, &b_dims, &strides, 0).unwrap();
        let c = RawStridedMut::new(&mut c_data, &c_dims, &c_strides, 0).unwrap();
        let err = bgemm_raw_with_backend_into::<f64, crate::backend::ActiveBackend>(
            c, a, b, 0, 1, 1, 1, 1.0, 0.0, false, false,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            crate::EinsumError::Strided(strided_view::StridedError::RankMismatch(2, 1))
        ));
    }

    #[test]
    fn raw_bgemm_zero_sum_scales_destination() {
        let a_dims = [2, 0];
        let b_dims = [0, 2];
        let c_dims = [2, 2];
        let a_strides = [0, 0];
        let b_strides = [0, 0];
        let c_strides = [2, 1];
        let a_data = [0.0; 1];
        let b_data = [0.0; 1];
        let mut c_data = [1.0, 2.0, 3.0, 4.0];
        let a = RawStridedRef::new(&a_data, &a_dims, &a_strides, 0).unwrap();
        let b = RawStridedRef::new(&b_data, &b_dims, &b_strides, 0).unwrap();
        let c = RawStridedMut::new(&mut c_data, &c_dims, &c_strides, 0).unwrap();

        bgemm_raw_strided_into(c, a, b, 0, 1, 1, 1, 1.0, 2.0, false, false).unwrap();

        assert_eq!(c_data, [2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn raw_bgemm_zero_sum_beta_zero_clears_destination() {
        let a_dims = [2, 0];
        let b_dims = [0, 2];
        let c_dims = [2, 2];
        let a_strides = [0, 0];
        let b_strides = [0, 0];
        let c_strides = [2, 1];
        let a_data = [0.0; 1];
        let b_data = [0.0; 1];
        let mut c_data = [1.0, 2.0, 3.0, 4.0];
        let a = RawStridedRef::new(&a_data, &a_dims, &a_strides, 0).unwrap();
        let b = RawStridedRef::new(&b_data, &b_dims, &b_strides, 0).unwrap();
        let c = RawStridedMut::new(&mut c_data, &c_dims, &c_strides, 0).unwrap();

        bgemm_raw_strided_into(c, a, b, 0, 1, 1, 1, 1.0, 0.0, false, false).unwrap();

        assert_eq!(c_data, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn raw_bgemm_empty_output_is_noop() {
        let a_dims = [0, 2];
        let b_dims = [2, 2];
        let c_dims = [0, 2];
        let a_strides = [2, 1];
        let b_strides = [2, 1];
        let c_strides = [2, 1];
        let a_data = [1.0, 2.0];
        let b_data = [3.0, 4.0, 5.0, 6.0];
        let mut c_data = [7.0, 8.0, 9.0, 10.0];
        let expected = c_data;
        let a = RawStridedRef::new(&a_data, &a_dims, &a_strides, 0).unwrap();
        let b = RawStridedRef::new(&b_data, &b_dims, &b_strides, 0).unwrap();
        let c = RawStridedMut::new(&mut c_data, &c_dims, &c_strides, 0).unwrap();

        bgemm_raw_strided_into(c, a, b, 0, 1, 1, 1, 1.0, 1.0, false, false).unwrap();

        assert_eq!(c_data, expected);
    }

    #[test]
    fn raw_bgemm_noncontiguous_output_writes_back() {
        let a_dims = [2, 2];
        let b_dims = [2, 2];
        let c_dims = [2, 2];
        let a_strides = [2, 1];
        let b_strides = [2, 1];
        let c_strides = [1, 3];
        let a_data = [1.0, 2.0, 3.0, 4.0];
        let b_data = [5.0, 6.0, 7.0, 8.0];
        let mut c_data = [0.0; 8];
        let a = RawStridedRef::new(&a_data, &a_dims, &a_strides, 0).unwrap();
        let b = RawStridedRef::new(&b_data, &b_dims, &b_strides, 0).unwrap();
        let c = RawStridedMut::new(&mut c_data, &c_dims, &c_strides, 1).unwrap();

        bgemm_raw_strided_into(c, a, b, 0, 1, 1, 1, 1.0, 0.0, false, false).unwrap();

        assert_eq!(c_data[1], 19.0);
        assert_eq!(c_data[4], 22.0);
        assert_eq!(c_data[2], 43.0);
        assert_eq!(c_data[5], 50.0);
        assert_eq!(c_data[0], 0.0);
        assert_eq!(c_data[3], 0.0);
    }
}
