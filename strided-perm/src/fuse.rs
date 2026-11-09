//! Dimension fusion logic ported from Strided.jl/src/mapreduce.jl
//!
//! This module implements the core dimension fusion algorithm that merges
//! contiguous dimensions to reduce iteration complexity.

use strided_view::INLINE_RANK;
use tinyvec::TinyVec;

/// Fused axis extents, inline up to [`INLINE_RANK`].
///
/// Part of the public [`BilateralFusionPlan`]; a rank above the inline
/// capacity spills to the heap instead of erroring.
pub type FusionDimVec = TinyVec<[usize; INLINE_RANK]>;

/// Fused source or destination strides, inline up to [`INLINE_RANK`].
pub type FusionStrideVec = TinyVec<[isize; INLINE_RANK]>;

/// Validated result of fusing adjacent dimensions that are contiguous in
/// both source and destination layouts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BilateralFusionPlan {
    /// Fused non-trivial dimensions in iteration order.
    pub dims: FusionDimVec,
    /// Source strides corresponding to [`Self::dims`].
    pub src_strides: FusionStrideVec,
    /// Destination strides corresponding to [`Self::dims`].
    pub dst_strides: FusionStrideVec,
}

/// Failure while validating or constructing a bilateral fusion plan.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum FusionPlanError {
    /// Shape and stride ranks do not agree.
    #[error(
        "metadata lengths differ: dims={dims}, src_strides={src_strides}, \
         dst_strides={dst_strides}"
    )]
    LengthMismatch {
        /// Number of dimensions.
        dims: usize,
        /// Number of source strides.
        src_strides: usize,
        /// Number of destination strides.
        dst_strides: usize,
    },
    /// A dimension cannot be represented for stride arithmetic, or a fused
    /// dimension product exceeds `usize`.
    #[error("fused dimension product overflows usize")]
    DimensionOverflow,
}

/// Plan bilateral dimension fusion for source and destination layouts.
///
/// Size-one axes are removed. Two remaining adjacent axes are fused only when
/// each layout is affine-contiguous across the boundary. Negative strides are
/// supported; stride multiplication overflow simply prevents fusion.
///
/// # Errors
///
/// Returns [`FusionPlanError::LengthMismatch`] when the metadata ranks differ,
/// and [`FusionPlanError::DimensionOverflow`] when a dimension cannot
/// participate safely in `isize` stride arithmetic or a fused extent
/// overflows `usize`.
pub fn plan_bilateral_fusion(
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
) -> Result<BilateralFusionPlan, FusionPlanError> {
    if dims.len() != src_strides.len() || dims.len() != dst_strides.len() {
        return Err(FusionPlanError::LengthMismatch {
            dims: dims.len(),
            src_strides: src_strides.len(),
            dst_strides: dst_strides.len(),
        });
    }

    let mut plan = BilateralFusionPlan {
        dims: FusionDimVec::with_capacity(dims.len()),
        src_strides: FusionStrideVec::with_capacity(dims.len()),
        dst_strides: FusionStrideVec::with_capacity(dims.len()),
    };

    for ((&dim, &src_stride), &dst_stride) in dims.iter().zip(src_strides).zip(dst_strides) {
        if dim == 1 {
            continue;
        }
        isize::try_from(dim).map_err(|_| FusionPlanError::DimensionOverflow)?;

        if let Some(last) = plan.dims.len().checked_sub(1) {
            let current_dim =
                isize::try_from(plan.dims[last]).map_err(|_| FusionPlanError::DimensionOverflow)?;
            let src_contiguous = plan.src_strides[last]
                .checked_mul(current_dim)
                .is_some_and(|expected| src_stride == expected);
            let dst_contiguous = plan.dst_strides[last]
                .checked_mul(current_dim)
                .is_some_and(|expected| dst_stride == expected);

            if src_contiguous && dst_contiguous {
                plan.dims[last] = plan.dims[last]
                    .checked_mul(dim)
                    .ok_or(FusionPlanError::DimensionOverflow)?;
                continue;
            }
        }

        plan.dims.push(dim);
        plan.src_strides.push(src_stride);
        plan.dst_strides.push(dst_stride);
    }

    Ok(plan)
}

#[cfg(test)]
mod tests;
