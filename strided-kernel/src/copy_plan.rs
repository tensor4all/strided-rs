//! Prepared (compile-once, execute-many) copy plans over raw strided layouts.
//!
//! [`copy_scale_raw`](crate::copy_scale_raw) and friends rebuild the fused
//! loop nest on every call; for prepared-replay consumers that issue many
//! small copies with a fixed layout, that per-call planning dominates
//! (see issue #139). [`CopyPlan`] splits the work: [`CopyPlan::compile`]
//! validates the layout pair and builds the fused traversal once,
//! [`CopyPlan::execute`]/[`CopyPlan::execute_scale`]/[`CopyPlan::execute_conj`]
//! replay it with no planning and no heap allocation for ranks at most
//! [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT).

use core::mem::MaybeUninit;
use core::ops::Mul;

use crate::map_view::map_raw_into;
use crate::ops_view::{copy_conj, copy_into, copy_scale};
use crate::raw_ops::{apply_fused_pair, fuse_pair_layout, FusedPairLayout};
use crate::{
    ElementOpApply, Identity, MaybeSendSync, RawStridedMut, RawStridedRef, Result, StridedError,
};

// Same pattern as map_view.rs / outer_product.rs: stack storage when the
// parallel feature pulls in smallvec, plain Vec otherwise. Only `compile`
// touches these; `execute*` never allocates either way.
#[cfg(feature = "parallel")]
type AxisVec<T> = smallvec::SmallVec<[T; crate::RAW_FUSED_RANK_LIMIT]>;
#[cfg(not(feature = "parallel"))]
type AxisVec<T> = Vec<T>;

/// A compiled copy traversal for one `(dims, dst_strides, src_strides)`
/// layout pair.
///
/// `compile` proves the layout facts once (rank agreement, extent overflow,
/// destination injectivity) and fuses/orders the loop nest; each `execute*`
/// call then only re-checks the per-call facts (that the supplied views carry
/// exactly the compiled layout) before replaying the prepared loops.
///
/// Overlapping `src`/`dest` memory is not supported, matching the rest of the
/// crate.
///
/// Ranks above [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT) are supported through the view-based
/// kernels; only that fallback path may allocate.
///
/// # Example
///
/// ```rust
/// use strided_kernel::{CopyPlan, RawStridedMut, RawStridedRef};
///
/// let dims = [2usize, 3];
/// let src_strides = [3isize, 1];
/// let dst_strides = [1isize, 2]; // transposed destination
/// let plan = CopyPlan::compile(&dims, &dst_strides, &src_strides).unwrap();
///
/// let src = [0.0f64, 1.0, 2.0, 10.0, 11.0, 12.0];
/// let mut dst = [0.0f64; 6];
/// let src_ref = RawStridedRef::new(&src, &dims, &src_strides, 0).unwrap();
/// let mut dst_mut = RawStridedMut::new(&mut dst, &dims, &dst_strides, 0).unwrap();
/// plan.execute(&mut dst_mut, &src_ref).unwrap();
/// assert_eq!(dst, [0.0, 10.0, 1.0, 11.0, 2.0, 12.0]);
/// ```
#[derive(Clone, Debug)]
pub struct CopyPlan {
    dims: AxisVec<usize>,
    dst_strides: AxisVec<isize>,
    src_strides: AxisVec<isize>,
    /// `None` when rank exceeds [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT); `execute*` then
    /// falls back to the view-based kernels.
    fused: Option<FusedPairLayout>,
}

impl CopyPlan {
    /// Compile a copy plan for the given layout pair.
    ///
    /// Performs the layout validation and traversal construction
    /// (fuse + order) once:
    ///
    /// - `dims`, `dst_strides`, and `src_strides` must have equal length
    ///   ([`StridedError::StrideLengthMismatch`]);
    /// - the total element count must not overflow `usize`
    ///   ([`StridedError::OffsetOverflow`]);
    /// - the destination layout must be injective, i.e. map distinct logical
    ///   indices to distinct offsets
    ///   ([`StridedError::NonInjectiveOutputLayout`]).
    pub fn compile(dims: &[usize], dst_strides: &[isize], src_strides: &[isize]) -> Result<Self> {
        if dims.len() != dst_strides.len() || dims.len() != src_strides.len() {
            return Err(StridedError::StrideLengthMismatch);
        }
        if dims
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .is_none()
        {
            return Err(StridedError::OffsetOverflow);
        }
        if !crate::fused::is_injective_layout(dims, dst_strides) {
            return Err(StridedError::NonInjectiveOutputLayout);
        }
        Ok(Self {
            dims: dims.into(),
            dst_strides: dst_strides.into(),
            src_strides: src_strides.into(),
            fused: fuse_pair_layout(dims, dst_strides, src_strides),
        })
    }

    /// Check the per-call facts: the supplied views must carry exactly the
    /// compiled layout. Buffer bounds against that layout were already proven
    /// by [`RawStridedRef::new`]/[`RawStridedMut::new`] (or asserted by the
    /// caller of the `new_unchecked` constructors), so layout equality is the
    /// complete precondition for the unsafe-free fused replay below.
    fn check_call<D, S>(
        &self,
        dest: &RawStridedMut<'_, D>,
        src: &RawStridedRef<'_, S>,
    ) -> Result<()> {
        if dest.dims() != &self.dims[..]
            || src.dims() != &self.dims[..]
            || dest.strides() != &self.dst_strides[..]
            || src.strides() != &self.src_strides[..]
        {
            return Err(StridedError::PlanLayoutMismatch);
        }
        Ok(())
    }

    /// `dest = src` into a potentially uninitialized destination.
    ///
    /// On success every reachable logical destination element is initialized.
    /// Non-reachable holes in the backing allocation are not written.
    pub fn execute_uninit<T>(
        &self,
        dest: &mut RawStridedMut<'_, MaybeUninit<T>>,
        src: &RawStridedRef<'_, T>,
    ) -> Result<()>
    where
        T: Copy + MaybeSendSync,
    {
        self.check_call(dest, src)?;
        match &self.fused {
            Some(layout) => {
                apply_fused_pair(
                    dest,
                    src,
                    layout,
                    |dst, value| {
                        dst.write(value);
                    },
                    |value| value,
                );
                Ok(())
            }
            None => map_raw_into::<MaybeUninit<T>, T, Identity>(dest, src, MaybeUninit::new),
        }
    }

    /// `dest = src`. Allocation-free for ranks at most [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT).
    pub fn execute<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        src: &RawStridedRef<'_, T>,
    ) -> Result<()>
    where
        T: Copy + MaybeSendSync,
    {
        self.check_call(dest, src)?;
        match &self.fused {
            Some(layout) => {
                apply_fused_pair(
                    dest,
                    src,
                    layout,
                    |dst, value| *dst = value,
                    |value: T| value,
                );
                Ok(())
            }
            None => copy_into(&mut dest.as_view_mut(), &src.as_view()),
        }
    }

    /// `dest = scale * src`. Allocation-free for ranks at most
    /// [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT).
    pub fn execute_scale<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        src: &RawStridedRef<'_, T>,
        scale: T,
    ) -> Result<()>
    where
        T: Copy + Mul<T, Output = T> + MaybeSendSync,
    {
        self.check_call(dest, src)?;
        match &self.fused {
            Some(layout) => {
                apply_fused_pair(
                    dest,
                    src,
                    layout,
                    |dst, value| *dst = value,
                    |value: T| scale * value,
                );
                Ok(())
            }
            None => copy_scale(&mut dest.as_view_mut(), &src.as_view(), scale),
        }
    }

    /// `dest = conj(src)`. Allocation-free for ranks at most
    /// [`RAW_FUSED_RANK_LIMIT`](crate::RAW_FUSED_RANK_LIMIT).
    pub fn execute_conj<T>(
        &self,
        dest: &mut RawStridedMut<'_, T>,
        src: &RawStridedRef<'_, T>,
    ) -> Result<()>
    where
        T: Copy + ElementOpApply + MaybeSendSync,
    {
        self.check_call(dest, src)?;
        match &self.fused {
            Some(layout) => {
                apply_fused_pair(
                    dest,
                    src,
                    layout,
                    |dst, value| *dst = value,
                    |value: T| value.conj(),
                );
                Ok(())
            }
            None => copy_conj(&mut dest.as_view_mut(), &src.as_view()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};

    /// Reference: the per-call raw kernel (which itself is differential-tested
    /// against the view kernels in raw_ops.rs).
    fn plan_matches_direct<T>(
        dims: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        src: &[T],
    ) where
        T: Copy
            + PartialEq
            + core::fmt::Debug
            + Default
            + Mul<T, Output = T>
            + ElementOpApply
            + MaybeSendSync
            + num_traits::One,
    {
        let len = src.len();
        let plan = CopyPlan::compile(dims, dst_strides, src_strides).unwrap();

        let mut expected = vec![T::default(); len];
        {
            let mut dest = RawStridedMut::new(&mut expected, dims, dst_strides, 0).unwrap();
            let source = RawStridedRef::new(src, dims, src_strides, 0).unwrap();
            crate::copy_scale_raw(&mut dest, &source, T::one()).unwrap();
        }

        let mut actual = vec![T::default(); len];
        {
            let mut dest = RawStridedMut::new(&mut actual, dims, dst_strides, 0).unwrap();
            let source = RawStridedRef::new(src, dims, src_strides, 0).unwrap();
            plan.execute(&mut dest, &source).unwrap();
        }
        assert_eq!(actual, expected);
    }

    fn fill_f64(len: usize) -> Vec<f64> {
        (0..len).map(|value| value as f64 - 2.5).collect()
    }

    #[test]
    fn plan_copy_matches_direct_rank0() {
        plan_matches_direct::<f64>(&[], &[], &[], &[7.0]);
    }

    #[test]
    fn plan_copy_matches_direct_rank1() {
        plan_matches_direct::<f64>(&[5], &[1], &[1], &fill_f64(5));
    }

    #[test]
    fn plan_copy_matches_direct_rank2_transposed() {
        plan_matches_direct::<f64>(&[3, 4], &[1, 3], &[4, 1], &fill_f64(12));
    }

    #[test]
    fn plan_copy_matches_direct_rank4() {
        plan_matches_direct::<f64>(&[2, 3, 2, 2], &[12, 4, 2, 1], &[1, 2, 6, 12], &fill_f64(24));
    }

    #[test]
    fn plan_copy_matches_direct_rank8() {
        let dims = [2usize; 8];
        let dst: Vec<isize> = (0..8).map(|axis| 1isize << axis).collect();
        let src: Vec<isize> = (0..8).rev().map(|axis| 1isize << axis).collect();
        plan_matches_direct::<f64>(&dims, &dst, &src, &fill_f64(256));
    }

    #[test]
    fn plan_copy_matches_direct_zero_size() {
        plan_matches_direct::<f64>(&[2, 0, 3], &[3, 3, 1], &[1, 6, 2], &fill_f64(6));
    }

    #[test]
    fn plan_copy_matches_direct_f32_and_complex() {
        let dims = [2usize, 3];
        let dst = [1isize, 2];
        let src = [3isize, 1];
        plan_matches_direct::<f32>(&dims, &dst, &src, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let complex: Vec<Complex32> = (0..6)
            .map(|value| Complex32::new(value as f32, -(value as f32)))
            .collect();
        plan_matches_direct::<Complex32>(&dims, &dst, &src, &complex);
        let complex: Vec<Complex64> = (0..6)
            .map(|value| Complex64::new(value as f64, 1.0 - value as f64))
            .collect();
        plan_matches_direct::<Complex64>(&dims, &dst, &src, &complex);
    }

    #[test]
    fn plan_copy_negative_stride_matches_view_kernel() {
        // Negative source stride: src viewed reversed, offset at the end.
        let dims = [4usize];
        let src_strides = [-1isize];
        let dst_strides = [1isize];
        let src = [1.0f64, 2.0, 3.0, 4.0];
        let plan = CopyPlan::compile(&dims, &dst_strides, &src_strides).unwrap();

        let mut actual = [0.0f64; 4];
        let mut dest = RawStridedMut::new(&mut actual, &dims, &dst_strides, 0).unwrap();
        let source = RawStridedRef::new(&src, &dims, &src_strides, 3).unwrap();
        plan.execute(&mut dest, &source).unwrap();
        assert_eq!(actual, [4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn plan_execute_scale_and_conj() {
        let dims = [2usize, 2];
        let strides = [2isize, 1];
        let src = [
            Complex64::new(1.0, 2.0),
            Complex64::new(-3.0, 4.0),
            Complex64::new(0.5, -1.0),
            Complex64::new(2.0, 0.0),
        ];
        let plan = CopyPlan::compile(&dims, &strides, &strides).unwrap();

        let mut scaled = [Complex64::default(); 4];
        let mut dest = RawStridedMut::new(&mut scaled, &dims, &strides, 0).unwrap();
        let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
        plan.execute_scale(&mut dest, &source, Complex64::new(2.0, 0.0))
            .unwrap();
        assert_eq!(scaled[1], Complex64::new(-6.0, 8.0));

        let mut conjugated = [Complex64::default(); 4];
        let mut dest = RawStridedMut::new(&mut conjugated, &dims, &strides, 0).unwrap();
        let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
        plan.execute_conj(&mut dest, &source).unwrap();
        assert_eq!(conjugated[0], Complex64::new(1.0, -2.0));
        assert_eq!(conjugated[3], Complex64::new(2.0, 0.0));
    }

    #[test]
    fn plan_rank_above_limit_falls_back_to_view_kernels() {
        let dims = [2usize; 9];
        let dst: Vec<isize> = (0..9).map(|axis| 1isize << axis).collect();
        let src: Vec<isize> = (0..9).rev().map(|axis| 1isize << axis).collect();
        let source_data = fill_f64(512);
        let plan = CopyPlan::compile(&dims, &dst, &src).unwrap();
        assert!(plan.fused.is_none());

        let mut expected = vec![0.0f64; 512];
        {
            let mut dest = RawStridedMut::new(&mut expected, &dims, &dst, 0).unwrap();
            let source = RawStridedRef::new(&source_data, &dims, &src, 0).unwrap();
            crate::copy_scale_raw(&mut dest, &source, 1.0).unwrap();
        }
        let mut actual = vec![0.0f64; 512];
        let mut dest = RawStridedMut::new(&mut actual, &dims, &dst, 0).unwrap();
        let source = RawStridedRef::new(&source_data, &dims, &src, 0).unwrap();
        plan.execute(&mut dest, &source).unwrap();
        assert_eq!(actual, expected);

        // Fallback also serves scale and conj.
        let mut scaled = vec![0.0f64; 512];
        let mut dest = RawStridedMut::new(&mut scaled, &dims, &dst, 0).unwrap();
        plan.execute_scale(&mut dest, &source, 2.0).unwrap();
        assert_eq!(scaled[0], 2.0 * actual[0]);
        let mut conjugated = vec![0.0f64; 512];
        let mut dest = RawStridedMut::new(&mut conjugated, &dims, &dst, 0).unwrap();
        plan.execute_conj(&mut dest, &source).unwrap();
        assert_eq!(conjugated, actual);
    }

    #[test]
    fn compile_rejects_length_mismatch() {
        let err = CopyPlan::compile(&[2, 3], &[3, 1], &[1]).unwrap_err();
        assert!(matches!(err, StridedError::StrideLengthMismatch));
        let err = CopyPlan::compile(&[2, 3], &[3], &[1, 2]).unwrap_err();
        assert!(matches!(err, StridedError::StrideLengthMismatch));
    }

    #[test]
    fn compile_rejects_extent_overflow() {
        let err = CopyPlan::compile(&[usize::MAX, 2], &[1, 1], &[1, 1]).unwrap_err();
        assert!(matches!(err, StridedError::OffsetOverflow));
    }

    #[test]
    fn compile_rejects_unrepresentable_positive_and_negative_offset_spans() {
        for strides in [
            [isize::MAX / 2 + 1, isize::MAX],
            [isize::MIN / 2 - 1, isize::MIN],
        ] {
            let err = CopyPlan::compile(&[2, 2], &strides, &strides).unwrap_err();
            assert!(matches!(err, StridedError::NonInjectiveOutputLayout));
        }
    }

    #[test]
    fn compile_accepts_representable_mixed_sign_span_without_fusion_overflow() {
        let positive = isize::MAX / 4;
        let negative = -(isize::MAX - positive);
        let strides = [positive, negative];
        CopyPlan::compile(&[2, 2], &strides, &strides).unwrap();
    }

    #[test]
    fn compile_rejects_non_injective_destination() {
        // Two logical columns land on the same offsets: forbidden overlap in
        // the mutable destination.
        let err = CopyPlan::compile(&[2, 2], &[1, 0], &[2, 1]).unwrap_err();
        assert!(matches!(err, StridedError::NonInjectiveOutputLayout));
        // Broadcast-like (stride 0) source layouts remain allowed.
        CopyPlan::compile(&[2, 2], &[2, 1], &[0, 1]).unwrap();
    }

    #[test]
    fn execute_rejects_layout_drift() {
        let dims = [2usize, 3];
        let strides = [3isize, 1];
        let plan = CopyPlan::compile(&dims, &strides, &strides).unwrap();
        let src = fill_f64(6);
        let mut dst = vec![0.0f64; 6];

        // Different dims than compiled.
        let other_dims = [3usize, 2];
        let other_strides = [2isize, 1];
        let mut dest = RawStridedMut::new(&mut dst, &other_dims, &other_strides, 0).unwrap();
        let source = RawStridedRef::new(&src, &other_dims, &other_strides, 0).unwrap();
        let err = plan.execute(&mut dest, &source).unwrap_err();
        assert!(matches!(err, StridedError::PlanLayoutMismatch));

        // Same dims, different source strides.
        let column_major = [1isize, 2];
        let mut dest = RawStridedMut::new(&mut dst, &dims, &strides, 0).unwrap();
        let source = RawStridedRef::new(&src, &dims, &column_major, 0).unwrap();
        let err = plan.execute_scale(&mut dest, &source, 1.0).unwrap_err();
        assert!(matches!(err, StridedError::PlanLayoutMismatch));

        // Same dims, different destination strides.
        let mut dest = RawStridedMut::new(&mut dst, &dims, &column_major, 0).unwrap();
        let source = RawStridedRef::new(&src, &dims, &strides, 0).unwrap();
        let err = plan.execute_conj(&mut dest, &source).unwrap_err();
        assert!(matches!(err, StridedError::PlanLayoutMismatch));
    }

    #[test]
    fn identity_layout_uses_single_fused_axis() {
        let plan = CopyPlan::compile(&[2, 3, 4], &[12, 4, 1], &[12, 4, 1]).unwrap();
        let fused = plan.fused.expect("rank 3 stays on the fused path");
        assert_eq!(fused.rank, 1);
        assert_eq!(fused.dims[0], 24);
    }
}
